"""
ingest_dam_rtm.py  —  LOCAL VERSION (no AWS required)
======================================================
Scrapes DAM and RTM market clearing data from IEX India.
Saves raw CSV  → data/raw/   (Bronze layer, one file per day).
Saves cleaned  → data/processed/  (Silver Parquet, one file per day).
Writes audit   → run_summary.json next to each Bronze CSV.

NO database. NO psycopg2. NO sqlalchemy. NO boto3.
Database logging belongs in gx_validate.py — not here.

Usage
-----
  python src/1_ingestion/ingest_dam_rtm.py
  python src/1_ingestion/ingest_dam_rtm.py --days 30

Local Bronze output  (raw CSV — one file per day, never modified)
-----------------------------------------------------------------
  data/raw/dam/year=YYYY/month=MM/date=YYYY-MM-DD/dam.csv
  data/raw/rtm/year=YYYY/month=MM/date=YYYY-MM-DD/rtm.csv

Local Silver output  (cleaned Parquet — one file per day)
---------------------------------------------------------
  data/processed/dam/year=YYYY/month=MM/date=YYYY-MM-DD/part-0001.parquet
  data/processed/rtm/year=YYYY/month=MM/date=YYYY-MM-DD/part-0001.parquet

Local Audit  (JSON summary — next to each Bronze CSV)
-----------------------------------------------------
  data/raw/dam/year=YYYY/month=MM/date=YYYY-MM-DD/run_summary.json
  data/raw/rtm/year=YYYY/month=MM/date=YYYY-MM-DD/run_summary.json

Columns — DAM
  Datetime, Purchase Bid (MW), Sell Bid (MW),
  MCV (MW), Final Scheduled Volume (MW), MCP (Rs/MWh) *,
  ingestion_date, source_file, pipeline_run_id

Columns — RTM
  Datetime, Session ID, Purchase Bid (MW), Sell Bid (MW),
  MCV (MW), Final Scheduled Volume (MW), MCP (Rs/MWh) *,
  ingestion_date, source_file, pipeline_run_id

.env keys needed
-----------------
  None — this version has no external service dependencies.

Dependencies
------------
  pip install selenium beautifulsoup4 requests pandas pyarrow python-dotenv
  NO boto3. NO psycopg2. NO sqlalchemy.
"""

import sys
import json
import time
import uuid
import logging
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

load_dotenv()

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SCRAPE_DAYS = 1

# Path resolves: src/1_ingestion/ → up 2 levels → project root
# Matches your MY-MLOPS-PROJECT folder structure:
#   data/raw/        ← Bronze
#   data/processed/  ← Silver
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LOCAL_BRONZE = PROJECT_ROOT / "data" / "raw"
LOCAL_SILVER = PROJECT_ROOT / "data" / "processed"

RUN_ID = str(uuid.uuid4())

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DAM/RTM] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("dam_rtm")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def build_args():
    p = argparse.ArgumentParser(prog="ingest_dam_rtm.py")
    p.add_argument(
        "--days",
        type=int,
        default=SCRAPE_DAYS,
        help=f"Days to scrape (default: {SCRAPE_DAYS}, max: 30)",
    )
    p.add_argument(
        "--date",
        default=None,
        help="Scrape a single specific date YYYY-MM-DD. Overrides --days.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-scrape even if Bronze file already exists locally.",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# LOCAL PATH HELPERS
# ─────────────────────────────────────────────────────────────
def _partition(market: str, date_str: str) -> str:
    """
    Partition sub-path — same structure as S3 cloud version.
    e.g. dam/year=2026/month=03/date=2026-03-20
    Keeping local and cloud structures identical makes migration easy.
    When you move to cloud, only the storage calls change — not the paths.
    """
    year = date_str[:4]
    month = date_str[5:7]
    return f"{market.lower()}/year={year}/month={month}/date={date_str}"


def _bronze_path(market: str, date_str: str) -> Path:
    """data/raw/dam/year=2026/month=03/date=2026-03-20/dam.csv"""
    return LOCAL_BRONZE / _partition(market, date_str) / f"{market.lower()}.csv"


def _silver_path(market: str, date_str: str) -> Path:
    """data/processed/dam/year=2026/month=03/date=2026-03-20/part-0001.parquet"""
    return LOCAL_SILVER / _partition(market, date_str) / "part-0001.parquet"


def _summary_path(market: str, date_str: str) -> Path:
    """data/raw/dam/year=2026/month=03/date=2026-03-20/run_summary.json"""
    return LOCAL_BRONZE / _partition(market, date_str) / "run_summary.json"


# ─────────────────────────────────────────────────────────────
# IDEMPOTENCY
# ─────────────────────────────────────────────────────────────
def _already_done(market: str, date_str: str, force: bool = False) -> bool:
    """Skip if Bronze CSV already exists locally — safe to rerun script.
    Pass force=True to re-scrape even if the file exists."""
    if force:
        return False
    return _bronze_path(market, date_str).exists()


# ─────────────────────────────────────────────────────────────
# SAVE BRONZE  (raw CSV — exactly as scraped)
# ─────────────────────────────────────────────────────────────
def _save_bronze(df: pd.DataFrame, market: str, date_str: str):
    """
    Bronze rule: save exactly what IEX returned — no transformations.
    Creates parent folders automatically (parents=True).
    One file per day. Never overwritten after first write.
    """
    path = _bronze_path(market, date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"  Bronze saved: {path.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────
# SAVE SILVER  (cleaned Parquet)
# ─────────────────────────────────────────────────────────────
def _save_silver(df: pd.DataFrame, market: str, date_str: str):
    """
    Parquet: 6x smaller than CSV, 10x faster for training.
    Snappy compression: best balance of speed and size.
    """
    path = _silver_path(market, date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")
    log.info(f"  Silver saved: {path.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────
# SAVE RUN SUMMARY  (plain JSON — no database needed)
# ─────────────────────────────────────────────────────────────
def _save_summary(
    market: str,
    date_str: str,
    rows: int,
    status: str,
    started: datetime,
    error: str = None,
):
    """
    Plain JSON file saved next to the Bronze CSV.
    gx_validate.py reads this for audit information.
    No Postgres. No SQLite. The info travels with the data itself.
    """
    completed = datetime.utcnow()
    summary = {
        "run_id": RUN_ID,
        "market": market,
        "date": date_str,
        "rows": rows,
        "status": status,
        "scraped_at": started.isoformat(),
        "duration_s": int((completed - started).total_seconds()),
        "bronze": str(_bronze_path(market, date_str)),
        "silver": str(_silver_path(market, date_str)),
        "error": error,
    }
    path = _summary_path(market, date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    log.info(f"  Summary saved: {path.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────
# TIMER DECORATOR
# ─────────────────────────────────────────────────────────────
def timed(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        secs = time.time() - t0
        log.info(f"  {func.__name__} — {secs/60:.1f} min ({secs:.0f}s)")
        return result

    return wrapper


# ─────────────────────────────────────────────────────────────
# IEX URL
# ─────────────────────────────────────────────────────────────
def _iex_url(market: str, days: int) -> str:
    endpoints = {"DAM": "day-ahead-market", "RTM": "real-time-market"}
    return (
        f"https://www.iexindia.com/market-data/{endpoints[market]}/market-snapshot"
        f"?interval=ONE_FOURTH_HOUR&dp=LAST_{days}_DAYS"
        f"&showGraph=false&toDate=1&fromDate=1"
    )


# ─────────────────────────────────────────────────────────────
# SELENIUM SCRAPER
# ─────────────────────────────────────────────────────────────
@timed
def _scrape(url: str, market: str, days: int) -> pd.DataFrame:
    """Open Chrome headless, navigate IEX table, read rows day by day."""
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=opts)
    wait = WebDriverWait(driver, 30)

    TABLE_XPATH = "/html/body/div[1]/div[4]/section/div[1]/div[3]/div[1]/table"
    ROW_XPATH = TABLE_XPATH + "/tbody/tr"
    NEXT_BTN_XPATH = (
        "/html/body/div[1]/div[4]/section/div[1]/div[3]" "/div[2]/div/div[2]/button[3]"
    )

    driver.get(url)
    wait.until(EC.presence_of_element_located((By.XPATH, TABLE_XPATH)))

    all_days = pd.DataFrame()

    for day in range(days):
        log.info(f"  Scraping {market} day {day + 1}/{days} …")
        wait.until(EC.presence_of_element_located((By.XPATH, ROW_XPATH)))

        table_html = driver.find_element(By.XPATH, TABLE_XPATH).get_attribute(
            "outerHTML"
        )
        soup = BeautifulSoup(table_html, "html.parser")
        table = soup.find("table")

        rows = []
        if table:
            for tr in table.find_all("tr"):
                cols = [td.text.strip() for td in tr.find_all("td")]
                if cols:
                    rows.append(cols)

        if not rows:
            log.warning(f"  No rows found on day {day + 1}")
        else:
            max_len = max(len(r) for r in rows)
            for r in rows:
                while len(r) < max_len:
                    r.insert(0, None)

            if max_len == 8:
                cols = [
                    "Date",
                    "Hour",
                    "Time Block",
                    "Purchase Bid (MW)",
                    "Sell Bid (MW)",
                    "MCV (MW)",
                    "Final Scheduled Volume (MW)",
                    "MCP (Rs/MWh) *",
                ]
            elif max_len == 9:
                cols = [
                    "Date",
                    "Hour",
                    "Session ID",
                    "Time Block",
                    "Purchase Bid (MW)",
                    "Sell Bid (MW)",
                    "MCV (MW)",
                    "Final Scheduled Volume (MW)",
                    "MCP (Rs/MWh) *",
                ]
            else:
                cols = [f"Col_{i}" for i in range(max_len)]

            df_day = pd.DataFrame(rows, columns=cols)
            ffill_cols = (
                ["Date", "Hour", "Session ID"]
                if "Session ID" in df_day.columns
                else ["Date", "Hour"]
            )
            df_day[ffill_cols] = df_day[ffill_cols].ffill()
            all_days = pd.concat([all_days, df_day], ignore_index=True)

        if day < days - 1:
            try:
                btn = wait.until(EC.element_to_be_clickable((By.XPATH, NEXT_BTN_XPATH)))
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(3)
            except Exception as e:
                log.warning(f"  Could not click next: {e}")
                break

    driver.quit()
    return all_days


# ─────────────────────────────────────────────────────────────
# SPLIT BY DATE
# ─────────────────────────────────────────────────────────────
def _split_by_date(raw: pd.DataFrame) -> dict:
    """Group multi-day DataFrame into { 'YYYY-MM-DD': df } dict."""
    if raw is None or raw.empty:
        return {}

    raw = raw[raw["Date"].astype(str).str.match(r"\d{2}-\d{2}-\d{4}", na=False)].copy()
    if raw.empty:
        return {}

    raw["_date"] = pd.to_datetime(raw["Date"], format="%d-%m-%Y", errors="coerce")
    raw = raw.dropna(subset=["_date"])

    result = {}
    for date_val, group in raw.groupby("_date"):
        date_str = date_val.strftime("%Y-%m-%d")
        result[date_str] = group.drop(columns=["_date"]).reset_index(drop=True)

    log.info(f"  Split into {len(result)} daily chunks: {sorted(result.keys())}")
    return result


# ─────────────────────────────────────────────────────────────
# CLEAN ONE DAY  (Bronze → Silver)
# ─────────────────────────────────────────────────────────────
def _clean(raw_day: pd.DataFrame, market: str, date_str: str) -> pd.DataFrame | None:
    """
    Parse Datetime, drop intermediate columns, cast numerics to float64.
    Add 3 lineage columns so every row is traceable back to its source.
    Row values themselves are never altered.
    """
    df = raw_day.copy()

    # Remove repeated header rows IEX sometimes inserts mid-table
    header_mask = df["Date"] == "Date"
    if header_mask.any():
        df = df.iloc[: header_mask.idxmax()]

    df = df[df["Date"].astype(str).str.match(r"\d{2}-\d{2}-\d{4}", na=False)]
    if df.empty:
        log.warning(f"  No valid date rows — {market} {date_str}")
        return None

    # ── Parse Time Block ────────────────────────────────────────
    # IEX changed DAM format in early 2026:
    #   Old DAM format: "00:00 - 00:15"  (spaces around dash)
    #   New DAM format: "00:00-00:15"    (no spaces) — same as RTM now
    #   RTM format:     "00:00-00:15"    (no spaces, unchanged)
    #
    # Both DAM and RTM now return Session ID column.
    # We handle both old and new formats for backwards compatibility.

    tb = df["Time Block"].astype(str)

    if market == "DAM":
        # Try new format first: "00:00-00:15"
        new_fmt = tb.str.match(r"^\d{2}:\d{2}-\d{2}:\d{2}$", na=False)
        # Try old format: "00:00 - 00:15"
        old_fmt = tb.str.contains(" - ", na=False)

        if new_fmt.any():
            df = df[new_fmt]
            time_start = df["Time Block"].str.split("-").str[0].str.strip()
        elif old_fmt.any():
            df = df[old_fmt]
            time_start = df["Time Block"].str.split(" - ").str[0].str.strip()
        else:
            log.warning(f"  Unrecognised Time Block format for DAM {date_str}")
            return None
    else:
        # RTM: "00:00-00:15" — no spaces around dash
        mask = tb.str.contains("-", na=False) & ~tb.str.contains(" - ", na=False)
        df = df[mask]
        if df.empty:
            return None
        time_start = df["Time Block"].str.split("-").str[0].str.strip().str.lstrip("-")

    if df.empty:
        return None

    df["Datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + time_start,
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )
    df = df.dropna(subset=["Datetime"])
    df = df.drop(columns=["Date", "Time Block", "Hour"], errors="ignore")

    for col in [
        "Purchase Bid (MW)",
        "Sell Bid (MW)",
        "MCV (MW)",
        "Final Scheduled Volume (MW)",
        "MCP (Rs/MWh) *",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ""), errors="coerce"
            )

    # Session ID — now present on both DAM and RTM (IEX changed format)
    if "Session ID" in df.columns:
        df["Session ID"] = pd.to_numeric(df["Session ID"], errors="coerce")

    # Lineage columns — every row traceable to its source file
    df["ingestion_date"] = date_str
    df["source_file"] = str(_bronze_path(market, date_str))
    df["pipeline_run_id"] = RUN_ID

    log.info(f"  Cleaned {len(df)} rows — {market} {date_str}")
    return df


# ─────────────────────────────────────────────────────────────
# SHOW OUTPUT TREE
# ─────────────────────────────────────────────────────────────
def _show_output():
    """Print what was saved — for easy verification after run."""
    log.info("\n  Files saved:")
    for base, label in [
        (LOCAL_BRONZE, "Bronze  data/raw/"),
        (LOCAL_SILVER, "Silver  data/processed/"),
    ]:
        if base.exists():
            files = sorted(
                f
                for f in base.rglob("*.*")
                if f.suffix in (".csv", ".parquet", ".json")
            )
            log.info(f"\n  {label} — {len(files)} files:")
            for f in files[:14]:
                log.info(f"    {f.relative_to(PROJECT_ROOT)}")
            if len(files) > 14:
                log.info(f"    … and {len(files) - 14} more")


# ─────────────────────────────────────────────────────────────
# PROCESS ONE MARKET
# ─────────────────────────────────────────────────────────────
def _process_market(
    market: str, days: int, target_date: str = None, force: bool = False
):
    log.info(f"\n{'─' * 60}")
    log.info(f"  Market  : {market}")
    if target_date:
        log.info(f"  Date    : {target_date}  (single-date mode)")
    else:
        log.info(f"  Days    : {days}")
    log.info(f"  Force   : {force}")
    log.info(f"  Run ID  : {RUN_ID[:8]}…")
    log.info(f"{'─' * 60}")

    # In single-date mode always scrape days=1 so IEX returns only that day
    scrape_days = 1 if target_date else days
    raw_all = _scrape(_iex_url(market, scrape_days), market, scrape_days)
    if raw_all is None or raw_all.empty:
        log.warning(f"  Scraper returned no data for {market}")
        return

    daily_chunks = _split_by_date(raw_all)
    if not daily_chunks:
        log.warning(f"  Could not split by date for {market}")
        return

    # In single-date mode filter to only the requested date
    if target_date:
        if target_date not in daily_chunks:
            log.warning(
                f"  {target_date} not found in scraped data — "
                f"IEX returned: {sorted(daily_chunks.keys())}. "
                f"The date may be too old or not yet published."
            )
            return
        daily_chunks = {target_date: daily_chunks[target_date]}
        log.info(f"  Confirmed: {target_date} found in scraped data")

    total_rows = 0
    total_skipped = 0

    for date_str, raw_day in sorted(daily_chunks.items()):
        started = datetime.utcnow()

        if _already_done(market, date_str, force=force):
            log.info(f"  {date_str} already exists locally — skipping")
            total_skipped += 1
            continue

        log.info(f"  Processing {date_str} …")

        _save_bronze(raw_day, market, date_str)

        cleaned = _clean(raw_day, market, date_str)
        if cleaned is None or cleaned.empty:
            log.warning(f"  Cleaning failed for {date_str}")
            _save_summary(
                market,
                date_str,
                0,
                "FAILED",
                started,
                error="Cleaning returned empty DataFrame",
            )
            continue

        _save_silver(cleaned, market, date_str)
        _save_summary(market, date_str, len(cleaned), "SUCCESS", started)

        total_rows += len(cleaned)
        log.info(f"  {date_str} — {len(cleaned)} rows")

    log.info(
        f"\n  {market} done — " f"{total_rows} rows saved, {total_skipped} days skipped"
    )


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    from src.lineage import lineage_run, ds

    args = build_args()
    force = args.force
    target_date = args.date
    days = args.days

    # Validate --date format
    if target_date:
        try:
            datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            log.error(f"Invalid --date format '{target_date}'. Use YYYY-MM-DD.")
            return

    log.info("=" * 60)
    log.info("DAM / RTM INGEST PIPELINE  —  LOCAL MODE")
    if target_date:
        log.info(f"Mode      : single-date  ({target_date})")
    else:
        log.info(f"Mode      : last {days} days")
    log.info(f"Force     : {force}")
    log.info(f"Run ID    : {RUN_ID[:8]}…")
    log.info("Bronze    : data/raw/dam/  &  data/raw/rtm/")
    log.info("Silver    : data/processed/dam/  &  data/processed/rtm/")
    log.info("Audit     : run_summary.json  (next to each Bronze CSV)")
    log.info("=" * 60)

    with lineage_run("ingest_dam_rtm",
        inputs=[ds("IEX India", namespace="https://www.iexindia.com")],
        outputs=[
            ds("data/raw/dam"), ds("data/raw/rtm"),
            ds("data/processed/dam"), ds("data/processed/rtm"),
        ],
    ):
        for market in ["DAM", "RTM"]:
            _process_market(market, days, target_date=target_date, force=force)

    _show_output()

    log.info("\n" + "=" * 60)
    log.info("=== DAM/RTM INGEST COMPLETE ===")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
