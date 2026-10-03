"""
populate_raw_data.py
====================
Populate data/raw/ (Bronze) and data/processed/ (Silver) from the bundled
March 2025 dataset (inputdata/march_2025_data.zip), so students can start at
validation without running live ingestion.

It writes BOTH medallion layers in exactly the layout the ingestion scripts
produce, because the validation stage reads Silver, not Bronze:

  Bronze — data/raw/ (CSV, as supplied)
    dam / rtm  ->  data/raw/<ds>/year=YYYY/month=MM/date=YYYY-MM-DD/<ds>.csv
    weather    ->  data/raw/weather/<City_State>.csv
    calendar   ->  data/raw/calendar/calendar.csv

  Silver — data/processed/ (Parquet, snappy-compressed)
    dam / rtm  ->  data/processed/<ds>/year=YYYY/month=MM/date=YYYY-MM-DD/part-0001.parquet
    weather    ->  data/processed/weather/city=<City_State>/part-0001.parquet
    calendar   ->  data/processed/calendar/calendar.parquet

Every Silver row carries the same lineage columns the real ingestion adds
(ingestion_date, source_file, pipeline_run_id) so rows stay traceable to the
Bronze file they came from. Weather rows also carry city_name.

The bundled dam/rtm/weather CSVs are already cleaned — they arrive with the
Silver schema — so no parsing is re-done here. Calendar is the exception: its
date column ships as DD-MM-YYYY and the validators filter on YYYY-MM-DD, so it
is normalised on the way into Silver.

Existing partitions for the same dates are overwritten.

Usage
-----
  python scripts/populate_raw_data.py
  python scripts/populate_raw_data.py --zip inputdata/march_2025_data.zip
  python scripts/populate_raw_data.py --layer processed   # Silver only
"""

import argparse
import io
import uuid
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DEFAULT_ZIP = ROOT / "inputdata" / "march_2025_data.zip"

# One id per run, stamped on every Silver row — mirrors the ingestion scripts.
RUN_ID = str(uuid.uuid4())

# datasets that get split into year=/month=/date= partitions, keyed by datetime column
PARTITIONED = {"dam": "Datetime", "rtm": "Datetime"}

# Silver weather columns, in the order weather_suite.json expects.
WEATHER_COLS = [
    "time",
    "temperature",
    "humidity",
    "windspeed_100m",
    "cloud_cover",
    "rainfall",
]


# ─────────────────────────────────────────────────────────────
# PATHS  (identical to src/1_ingestion/*, so cloud migration is a no-op)
# ─────────────────────────────────────────────────────────────
def _partition(name: str, date_str: str) -> str:
    """e.g. dam/year=2025/month=03/date=2025-03-06"""
    return f"{name}/year={date_str[:4]}/month={date_str[5:7]}/date={date_str}"


def _bronze_path(name: str, date_str: str) -> Path:
    return RAW / _partition(name, date_str) / f"{name}.csv"


def _silver_path(name: str, date_str: str) -> Path:
    return PROCESSED / _partition(name, date_str) / "part-0001.parquet"


def _add_lineage(df: pd.DataFrame, ingestion_date: str, source_file: Path) -> pd.DataFrame:
    """The three columns every Silver row carries, in the ingestion scripts' order."""
    out = df.copy()
    out["ingestion_date"] = ingestion_date
    out["source_file"] = str(source_file)
    out["pipeline_run_id"] = RUN_ID
    return out


# ─────────────────────────────────────────────────────────────
# DAM / RTM  — one date partition per calendar day
# ─────────────────────────────────────────────────────────────
def _write_partitions(name: str, df: pd.DataFrame, dt_col: str, layer: str) -> int:
    """Split df by calendar date; write Bronze CSV and/or Silver Parquet per date."""
    dt = pd.to_datetime(df[dt_col], errors="coerce")
    df = df[dt.notna()]
    dt = dt[dt.notna()]

    days = 0
    for day, chunk in df.groupby(dt.dt.strftime("%Y-%m-%d")):
        chunk = chunk.reset_index(drop=True)
        bronze = _bronze_path(name, day)

        if layer in ("raw", "both"):
            bronze.parent.mkdir(parents=True, exist_ok=True)
            chunk.to_csv(bronze, index=False)

        if layer in ("processed", "both"):
            silver = _add_lineage(chunk, day, bronze)
            out = _silver_path(name, day)
            out.parent.mkdir(parents=True, exist_ok=True)
            silver.to_parquet(out, index=False, compression="snappy")

        days += 1
    return days


# ─────────────────────────────────────────────────────────────
# WEATHER  — one file per city, covering every date
# ─────────────────────────────────────────────────────────────
def _write_weather(z: zipfile.ZipFile, members: list, layer: str) -> int:
    (RAW / "weather").mkdir(parents=True, exist_ok=True)

    for member in members:
        filename = Path(member).name
        city = filename[:-4] if filename.endswith(".csv") else filename
        payload = z.read(member)
        bronze = RAW / "weather" / filename

        if layer in ("raw", "both"):
            bronze.write_bytes(payload)

        if layer in ("processed", "both"):
            df = pd.read_csv(io.BytesIO(payload))
            silver = df[[c for c in WEATHER_COLS if c in df.columns]].copy()

            # Weather is not date-partitioned, so stamp the latest day it covers.
            times = pd.to_datetime(silver["time"], errors="coerce")
            ingestion_date = (
                times.max().strftime("%Y-%m-%d") if times.notna().any() else ""
            )

            silver = _add_lineage(silver, ingestion_date, bronze)
            # city_name sits with the other lineage columns, as the ingester writes it.
            silver.insert(len(WEATHER_COLS), "city_name", city)

            out = PROCESSED / "weather" / f"city={city}" / "part-0001.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            silver.to_parquet(out, index=False, compression="snappy")

    return len(members)


# ─────────────────────────────────────────────────────────────
# CALENDAR  — single reference file, date normalised to YYYY-MM-DD
# ─────────────────────────────────────────────────────────────
def _write_calendar(payload: bytes, layer: str) -> int:
    bronze = RAW / "calendar" / "calendar.csv"

    if layer in ("raw", "both"):
        bronze.parent.mkdir(parents=True, exist_ok=True)
        bronze.write_bytes(payload)

    if layer not in ("processed", "both"):
        return 0

    df = pd.read_csv(io.BytesIO(payload))

    # The bundle ships DD-MM-YYYY; every validator filters on YYYY-MM-DD
    # (df[df["date"] == date_str]), so an unconverted column matches nothing.
    parsed = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    dropped = int(parsed.isna().sum())
    if dropped:
        print(f"  calendar: {dropped} row(s) with unparseable dates dropped")
    df = df[parsed.notna()].copy()
    df["date"] = parsed[parsed.notna()].dt.strftime("%Y-%m-%d")

    silver = _add_lineage(df, "", bronze)
    # Each calendar row is its own day, so ingestion_date is that row's date.
    silver["ingestion_date"] = silver["date"]

    out = PROCESSED / "calendar" / "calendar.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(out, index=False, compression="snappy")
    return len(silver)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Populate data/raw (Bronze) and data/processed (Silver) from the March 2025 bundle"
    )
    ap.add_argument("--zip", default=str(DEFAULT_ZIP), help="path to march_2025_data.zip")
    ap.add_argument(
        "--layer",
        choices=["raw", "processed", "both"],
        default="both",
        help="which medallion layer to write (default: both)",
    )
    args = ap.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise SystemExit(f"Bundle not found: {zip_path}")

    layer = args.layer
    print(f"Reading {zip_path}")
    print(f"Layer: {layer}   Run id: {RUN_ID[:8]}…")

    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()

        # dam / rtm -> year/month/date partitions
        for name, dt_col in PARTITIONED.items():
            member = f"{name}.csv"
            if member not in names:
                print(f"  {name}: not in bundle, skipped")
                continue
            df = pd.read_csv(io.BytesIO(z.read(member)))
            days = _write_partitions(name, df, dt_col, layer)
            print(f"  {name}: {len(df)} rows -> {days} date partitions")

        # weather -> per-city files
        weather_files = [n for n in names if n.startswith("weather/") and n.endswith(".csv")]
        if weather_files:
            count = _write_weather(z, weather_files, layer)
            print(f"  weather: {count} city files")

        # calendar -> single reference file
        if "calendar.csv" in names:
            rows = _write_calendar(z.read("calendar.csv"), layer)
            print(f"  calendar: {rows} rows" if rows else "  calendar: written")

    if layer in ("raw", "both"):
        print("Bronze ready: data/raw/")
    if layer in ("processed", "both"):
        print("Silver ready: data/processed/  (validation can now run)")


if __name__ == "__main__":
    main()
