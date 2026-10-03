"""
build_features.py
=================
Builds the model-ready feature table from GX-validated Silver data — the
reproducible, scripted replacement for the original March 2025 EDA notebook.

PIPELINE POSITION:
  Bronze (raw) -> Silver (clean) -> Validated -> FEATURES (this) -> Feast -> Model

WHAT THIS SCRIPT DOES (ported 1:1 from the March 2025 EDA notebook so the output
byte-matches the previously notebook-baked parquet):
  1. Load validated DAM, RTM, Weather, Calendar (valid rows only)
  2. Clean each source (drop pipeline metadata, dedup, clip MCP outliers)
  3. Build a 15-min DAM spine; expand RTM (30-min) and Weather (hourly) to 15-min
  4. Join all sources; engineer time / cyclical / lag / rolling / spread features
  5. Drop lag warm-up rows and write the prepared feature parquet

OUTPUT:
  data/features/march_2025_eda/march_2025_prepared.parquet
  One row = one 15-min block, 39 features + Datetime + dam_mcp target.
  Direct input to prepare_feast_data.py.

Usage
-----
  python src/3_feature_engineering/1_build_features.py
  python src/3_feature_engineering/1_build_features.py --year 2025 --month 3

Dependencies
------------
  pip install pandas pyarrow
"""

import sys
import glob
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VALIDATED_DIR = PROJECT_ROOT / "data" / "validated"
OUTPUT_DIR = PROJECT_ROOT / "data" / "features" / "march_2025_eda"
OUTPUT_PATH = OUTPUT_DIR / "march_2025_prepared.parquet"

# Pipeline metadata columns dropped from every source (added during ingestion)
META_COLS = ["ingestion_date", "source_file", "pipeline_run_id"]

# Final model features (order matters — must match feature_definitions.py / app)
FEATURE_COLS = [
    # Time
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "block_sin", "block_cos", "day_of_month",
    "is_weekend", "is_peak_hour", "is_morning_ramp", "is_off_peak",
    # Market — DAM
    "dam_purchase_bid", "dam_sell_bid", "dam_mcv", "dam_volume",
    "dam_bid_imbalance",
    # Market — RTM
    "rtm_purchase_bid", "rtm_sell_bid", "rtm_mcv", "rtm_volume",
    "mcp_spread",
    # Lag / rolling MCP
    "dam_mcp_lag_1d", "dam_mcp_lag_2d", "dam_mcp_lag_7d",
    "dam_mcp_roll_4h", "dam_mcp_roll_24h", "dam_mcp_roll_std_24h",
    # Weather
    "avg_temp", "avg_humidity", "avg_windspeed", "avg_cloud_cover", "total_rainfall",
    # Calendar (derived from text columns in clean_calendar)
    "is_ipl_match", "is_event", "is_festival", "is_wedding_season", "impact",
]
TARGET_COL = "dam_mcp"

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BUILD_FEATURES] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("build_features")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def build_args():
    p = argparse.ArgumentParser(prog="build_features.py")
    p.add_argument("--year", type=int, default=2025, help="Year (default 2025)")
    p.add_argument("--month", type=int, default=3, help="Month (default 3 = March)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD VALIDATED DATA
# ─────────────────────────────────────────────────────────────
def load_parquet(pattern: str) -> pd.DataFrame:
    """Concat all valid Parquet files matching a glob pattern (sorted for determinism)."""
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    df = pd.concat([pd.read_parquet(f) for f in sorted(files)], ignore_index=True)
    log.info(f"  Loaded {len(files)} files -> {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def _valid_glob(dataset: str, year: int, month: str) -> str:
    if dataset == "calendar":
        return str(VALIDATED_DIR / "calendar" / "valid" / "calendar.parquet")
    return str(VALIDATED_DIR / dataset / "valid" / f"year={year}" / f"month={month}" / "**" / "*.parquet")


# ─────────────────────────────────────────────────────────────
# STEP 2 — CLEAN EACH SOURCE
# ─────────────────────────────────────────────────────────────
def clean_market(df_raw: pd.DataFrame, market: str) -> pd.DataFrame:
    """
    Clean DAM or RTM: drop metadata, parse/sort Datetime, rename to {market}_* names,
    dedup, and clip MCP to ±3·IQR. Matches notebook cells 3.1 (RTM) and 3.2 (DAM).
    """
    prefix = market.lower()  # "dam" or "rtm"
    df = df_raw.copy()
    df.drop(columns=[c for c in META_COLS if c in df.columns], inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.sort_values("Datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.rename(columns={
        "MCP (Rs/MWh) *": f"{prefix}_mcp",
        "Purchase Bid (MW)": f"{prefix}_purchase_bid",
        "Sell Bid (MW)": f"{prefix}_sell_bid",
        "MCV (MW)": f"{prefix}_mcv",
        "Final Scheduled Volume (MW)": f"{prefix}_volume",
        "Session ID": f"{prefix}_session_id",
    }, inplace=True)

    # Dedup: RTM keeps Datetime+session, DAM keeps Datetime
    if market == "RTM":
        df.drop_duplicates(subset=["Datetime", "rtm_session_id"], inplace=True)
    else:
        df.drop_duplicates(subset=["Datetime"], inplace=True)

    # Clamp extreme MCP outliers (±3·IQR)
    mcp_col = f"{prefix}_mcp"
    q1, q3 = df[mcp_col].quantile(0.25), df[mcp_col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
    n_out = int(((df[mcp_col] < lower) | (df[mcp_col] > upper)).sum())
    df[mcp_col] = df[mcp_col].clip(lower, upper)
    log.info(f"  {market} cleaned: {df.shape[0]} rows, {n_out} MCP outliers capped "
             f"to [{lower:.1f}, {upper:.1f}]")
    return df


def clean_weather(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Drop metadata, sort by city/time, forward/back fill small gaps per city. Cell 3.3."""
    df = df_raw.copy()
    df.drop(columns=[c for c in META_COLS if c in df.columns], inplace=True)
    df["time"] = pd.to_datetime(df["time"])
    df.sort_values(["city_name", "time"], inplace=True)
    df = df.groupby("city_name", group_keys=False).apply(
        lambda g: g.sort_values("time").ffill().bfill()
    )
    log.info(f"  Weather cleaned: {df.shape[0]} rows, {df['city_name'].nunique()} cities")
    return df


def clean_calendar(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Drop metadata, parse date, fill text columns, derive boolean features."""
    df = df_raw.copy()
    df.drop(columns=[c for c in META_COLS + ["validated_at", "validation_run_id"]
                      if c in df.columns], inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    for col, fill in [("festival_name", "None"), ("event", "None"),
                      ("election_type", "None"), ("wedding_season", "None"),
                      ("TIME", ""), ("STADIUM", ""), ("location", ""),
                      ("region", ""), ("impact", 1.0)]:
        if col in df.columns:
            df[col] = df[col].fillna(fill)

    # Derive boolean columns from text columns
    if "TIME" in df.columns:
        df["is_ipl_match"] = (df["TIME"].str.strip() != "").astype(int)
    if "event" in df.columns:
        df["is_event"] = (df["event"] != "None").astype(int)
    if "festival_name" in df.columns:
        df["is_festival"] = (df["festival_name"] != "None").astype(int)
    if "wedding_season" in df.columns:
        df["is_wedding_season"] = (df["wedding_season"] != "None").astype(int)

    # Aggregate to one row per date (IPL double-headers etc.)
    agg_dict = {"DAY": "first", "is_ipl_match": "max", "is_event": "max",
                "is_festival": "max", "is_wedding_season": "max", "impact": "max"}
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    df = df.groupby("date").agg(agg_dict).reset_index()

    log.info(f"  Calendar cleaned: {df.shape[0]} rows")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 3 — AGGREGATE / EXPAND / JOIN
# ─────────────────────────────────────────────────────────────
def aggregate_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all cities to India-level hourly stats. Cell 4.4."""
    agg = weather.groupby("time").agg(
        avg_temp=("temperature", "mean"),
        max_temp=("temperature", "max"),
        min_temp=("temperature", "min"),
        avg_humidity=("humidity", "mean"),
        avg_windspeed=("windspeed_100m", "mean"),
        avg_cloud_cover=("cloud_cover", "mean"),
        total_rainfall=("rainfall", "sum"),
    ).reset_index()
    agg.rename(columns={"time": "hour_ts"}, inplace=True)
    return agg


def build_spine(dam: pd.DataFrame, rtm: pd.DataFrame, weather_agg: pd.DataFrame,
                calendar: pd.DataFrame) -> pd.DataFrame:
    """Build 15-min DAM spine and join RTM, weather, calendar. Cells 5.1–5.4."""
    # 5.1 — DAM spine
    spine = dam[["Datetime", "dam_mcp", "dam_purchase_bid", "dam_sell_bid",
                 "dam_mcv", "dam_volume"]].copy()
    spine["date"] = spine["Datetime"].dt.normalize()

    # 5.2 — Expand RTM (30-min sessions) to 2 x 15-min blocks, then join
    rtm_expanded = rtm.copy()
    second_block = rtm_expanded.copy()
    second_block["Datetime"] = second_block["Datetime"] + pd.Timedelta(minutes=15)
    rtm_expanded = pd.concat([rtm_expanded, second_block], ignore_index=True)
    rtm_expanded.sort_values("Datetime", inplace=True)
    rtm_expanded.reset_index(drop=True, inplace=True)
    rtm_for_join = rtm_expanded[["Datetime", "rtm_mcp", "rtm_purchase_bid",
                                 "rtm_sell_bid", "rtm_mcv", "rtm_volume"]]
    spine = spine.merge(rtm_for_join, on="Datetime", how="left")
    log.info(f"  After RTM join: {spine.shape}")

    # 5.3 — Expand weather (hourly) to 4 x 15-min blocks, then join
    w = weather_agg.copy()
    w.rename(columns={"hour_ts": "hour_ts_orig"}, inplace=True)
    w_blocks = []
    for offset in [0, 15, 30, 45]:
        block = w.copy()
        block["Datetime"] = block["hour_ts_orig"] + pd.Timedelta(minutes=offset)
        w_blocks.append(block)
    w_expanded = pd.concat(w_blocks, ignore_index=True).drop(columns="hour_ts_orig")
    w_expanded.sort_values("Datetime", inplace=True)
    spine = spine.merge(w_expanded, on="Datetime", how="left")
    log.info(f"  After Weather join: {spine.shape}")

    # 5.4 — Join calendar on normalized date (broadcasts 1 row to all daily blocks)
    cal_join = calendar.copy()
    cal_join["date"] = pd.to_datetime(cal_join["date"]).dt.normalize()
    spine = spine.merge(cal_join, on="date", how="left")
    log.info(f"  After Calendar join: {spine.shape}")
    return spine


# ─────────────────────────────────────────────────────────────
# STEP 4 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def add_features(spine: pd.DataFrame) -> pd.DataFrame:
    """Time, cyclical, lag, rolling, spread features. Cells 5.5–5.6."""
    df = spine.copy()

    # 5.5 — time features
    df["hour"] = df["Datetime"].dt.hour
    df["minute"] = df["Datetime"].dt.minute
    df["block_number"] = df["hour"] * 4 + df["minute"] // 15  # 0–95
    df["day_of_week"] = df["Datetime"].dt.dayofweek            # 0=Mon
    df["day_of_month"] = df["Datetime"].dt.day
    df["week_of_month"] = (df["Datetime"].dt.day - 1) // 7 + 1
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].between(18, 20).astype(int)
    df["is_morning_ramp"] = df["hour"].between(6, 7).astype(int)
    df["is_off_peak"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)

    # cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["block_sin"] = np.sin(2 * np.pi * df["block_number"] / 96)
    df["block_cos"] = np.cos(2 * np.pi * df["block_number"] / 96)

    # 5.6 — lag / rolling / spread (sort by Datetime first, like the notebook)
    df.sort_values("Datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    bpd = 96  # blocks per day
    df["dam_mcp_lag_1d"] = df["dam_mcp"].shift(1 * bpd)
    df["dam_mcp_lag_2d"] = df["dam_mcp"].shift(2 * bpd)
    df["dam_mcp_lag_7d"] = df["dam_mcp"].shift(7 * bpd)
    df["dam_mcp_roll_4h"] = df["dam_mcp"].shift(1).rolling(window=16, min_periods=4).mean()
    df["dam_mcp_roll_24h"] = df["dam_mcp"].shift(1).rolling(window=96, min_periods=24).mean()
    df["dam_mcp_roll_std_24h"] = df["dam_mcp"].shift(1).rolling(window=96, min_periods=24).std()
    df["mcp_spread"] = df["rtm_mcp"] - df["dam_mcp"]
    df["dam_bid_imbalance"] = (df["dam_purchase_bid"] - df["dam_sell_bid"]) / \
                              (df["dam_purchase_bid"] + df["dam_sell_bid"] + 1e-6)
    log.info(f"  Features engineered: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 5 — FINALIZE + SAVE
# ─────────────────────────────────────────────────────────────
def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Drop lag warm-up rows and select the final columns. Cell 5.7 + 5.10."""
    df_model = df.dropna(subset=["dam_mcp_lag_7d", "avg_temp"]).copy()
    cols = [c for c in FEATURE_COLS if c in df_model.columns]
    out = df_model[["Datetime"] + cols + [TARGET_COL]].copy()
    log.info(f"  Final feature table: {out.shape[0]} rows x {out.shape[1]} cols")
    return out


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    from src.lineage import lineage_run, ds

    args = build_args()
    month_str = f"{args.month:02d}"

    log.info("=" * 55)
    log.info("BUILD FEATURES  —  Validated -> Prepared feature table")
    log.info(f"Input:  data/validated/  ({args.year}-{month_str})")
    log.info(f"Output: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    log.info("=" * 55)

    # If the full validated month isn't on disk, keep the existing committed snapshot
    # instead of failing. build_features regenerates features only when fresh validated
    # data is available; otherwise the demo runs from the pinned snapshot.
    required = ["dam", "rtm", "weather", "calendar"]
    missing = [d for d in required
               if not (Path(_valid_glob(d, args.year, month_str)).exists()
                       if d == "calendar"
                       else glob.glob(_valid_glob(d, args.year, month_str)))]
    if missing:
        if OUTPUT_PATH.exists():
            log.warning(
                f"Validated data missing for {missing} ({args.year}-{month_str}). "
                f"Keeping existing snapshot: {OUTPUT_PATH.relative_to(PROJECT_ROOT)} "
                "(run ingestion + validation to regenerate)."
            )
            return
        raise FileNotFoundError(
            f"No validated data for {missing} and no existing snapshot at "
            f"{OUTPUT_PATH.relative_to(PROJECT_ROOT)}. Run ingestion + validation first."
        )

    with lineage_run(
        "build_features",
        inputs=[ds("data/validated")],
        outputs=[ds("data/features/march_2025_eda/march_2025_prepared.parquet")],
    ):
        # 1. Load
        log.info("\nStep 1: Loading validated data")
        rtm_raw = load_parquet(_valid_glob("rtm", args.year, month_str))
        dam_raw = load_parquet(_valid_glob("dam", args.year, month_str))
        cal_path = _valid_glob("calendar", args.year, month_str)
        cal_raw = pd.read_parquet(cal_path)
        cal_raw = cal_raw[cal_raw["date"].between(
            f"{args.year}-{month_str}-01", f"{args.year}-{month_str}-31")]
        log.info(f"  Loaded calendar: {cal_raw.shape[0]} rows (filtered to {args.year}-{month_str})")
        weather_raw = load_parquet(_valid_glob("weather", args.year, month_str))

        # 2. Clean
        log.info("\nStep 2: Cleaning sources")
        rtm = clean_market(rtm_raw, "RTM")
        dam = clean_market(dam_raw, "DAM")
        weather = clean_weather(weather_raw)
        calendar = clean_calendar(cal_raw)

        # 3. Aggregate / expand / join
        log.info("\nStep 3: Aggregating, expanding, joining")
        weather_agg = aggregate_weather(weather)
        spine = build_spine(dam, rtm, weather_agg, calendar)

        # 4. Feature engineering
        log.info("\nStep 4: Engineering features")
        df = add_features(spine)

        # 5. Finalize + save
        log.info("\nStep 5: Finalizing and saving")
        out = finalize(df)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUTPUT_PATH, index=False)
        log.info(f"  Saved: {OUTPUT_PATH.relative_to(PROJECT_ROOT)} "
                 f"({out.shape[0]} rows x {out.shape[1]} cols)")

    log.info("=" * 55)
    log.info("=== BUILD FEATURES COMPLETE ===")


if __name__ == "__main__":
    main()
