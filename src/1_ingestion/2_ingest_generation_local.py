"""
ingest_generation_local.py
==========================
Downloads CEA Daily Generation Reports (DGR Sub-Report-1) from NPP.
Saves raw PDF    → data/raw/generation/raw/{year}/{YYYY-MM-DD}.pdf   (Bronze)
Saves parsed CSV → data/raw/generation/csv/dgr_{FY}.csv              (Bronze)
Saves master CSV → data/raw/generation/dgr_master.csv                (Bronze)
Saves Silver     → data/processed/generation/year=.../date=.../part-0001.parquet
Writes audit     → data/raw/generation/run_summary.json

NO AWS. NO S3. NO boto3.

Usage
-----
  python src/1_ingestion/2_ingest_generation_local.py
  python src/1_ingestion/2_ingest_generation_local.py --days 30
  python src/1_ingestion/2_ingest_generation_local.py --start 2024-09-01 --end 2025-03-20

.env keys needed
-----------------
  None

Dependencies
------------
  pip install pandas pyarrow pdfplumber requests playwright python-dotenv
  python -m playwright install chromium
"""

import io
import json
import uuid
import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import pdfplumber
import requests
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

load_dotenv()

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DEMO_DAYS = 7
NPP_BASE_URL = "https://npp.gov.in/public-reports/cea/daily/dgr"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LOCAL_BRONZE = PROJECT_ROOT / "data" / "raw" / "generation"
LOCAL_SILVER = PROJECT_ROOT / "data" / "processed" / "generation"

RUN_ID = str(uuid.uuid4())

COLUMNS = [
    "date",
    "financial_year",
    "region",
    "fuel_type",
    "installed_capacity_mw",
    "monitored_capacity_mw",
    "annual_target_mu",
    "today_program_mu",
    "today_actual_mu",
    "apr1_till_date_program_mu",
    "apr1_till_date_actual_mu",
    "deviation_mu",
    "deviation_pct",
    "source_pdf",
    "parse_ok",
    "error",
]

REGION_LABELS = {
    "NORTHERN",
    "WESTERN",
    "SOUTHERN",
    "EASTERN",
    "NORTH EASTERN",
    "N.EASTERN",
    "ALL INDIA",
}
FUEL_KEYS = ["THERMAL", "NUCLEAR", "HYDRO", "BHUTAN IMP", "R.E.S", "RES"]

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DGR] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("dgr_local")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def _parse_date(v: str) -> datetime:
    for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"]:
        try:
            return datetime.strptime(v.strip(), fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"Bad date '{v}'. Use YYYY-MM-DD.")


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=DEMO_DAYS)
    p.add_argument("--start", type=_parse_date, default=None)
    p.add_argument("--end", type=_parse_date, default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# PATH HELPERS
# ─────────────────────────────────────────────────────────────
def financial_year(d: datetime) -> str:
    return (
        f"{d.year}-{str(d.year+1)[2:]}"
        if d.month >= 4
        else f"{d.year-1}-{str(d.year)[2:]}"
    )


def fy_folder(d: datetime) -> str:
    return str(d.year) if d.month >= 4 else str(d.year - 1)


def bronze_pdf_path(d: datetime) -> Path:
    return LOCAL_BRONZE / "raw" / fy_folder(d) / f"{d.strftime('%Y-%m-%d')}.pdf"


def bronze_csv_path(fy: str) -> Path:
    return LOCAL_BRONZE / "csv" / f"dgr_{fy}.csv"


def bronze_master_path() -> Path:
    return LOCAL_BRONZE / "dgr_master.csv"


def missing_log_path() -> Path:
    return LOCAL_BRONZE / "missing_dates.txt"


def silver_path(date_str: str) -> Path:
    year = date_str[:4]
    month = date_str[5:7]
    return (
        LOCAL_SILVER
        / f"year={year}"
        / f"month={month}"
        / f"date={date_str}"
        / "part-0001.parquet"
    )


def pdf_url(d: datetime) -> str:
    return f"{NPP_BASE_URL}/{d.strftime('%d-%m-%Y')}/dgr1-{d.strftime('%Y-%m-%d')}.pdf"


# ─────────────────────────────────────────────────────────────
# MISSING DATE LOG
# ─────────────────────────────────────────────────────────────
def load_missing() -> set:
    p = missing_log_path()
    if not p.exists():
        return set()
    return {line.strip() for line in p.read_text().splitlines() if line.strip()}


def save_missing(s: set):
    p = missing_log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(sorted(s)))


# ─────────────────────────────────────────────────────────────
# PLAYWRIGHT SESSION
# ─────────────────────────────────────────────────────────────
def get_session() -> requests.Session:
    log.info("Launching Playwright to get NPP cookies")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0")
        )
        page = ctx.new_page()
        page.goto("https://npp.gov.in", timeout=30000, wait_until="domcontentloaded")
        page.wait_for_timeout(3000)
        cookies = ctx.cookies()
        browser.close()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
            "Referer": "https://npp.gov.in/",
            "Accept": "application/pdf,*/*",
        }
    )
    for c in cookies:
        session.cookies.set(c["name"], c["value"])
    log.info(f"Got {len(cookies)} cookies")
    return session


# ─────────────────────────────────────────────────────────────
# DOWNLOAD PDF
# ─────────────────────────────────────────────────────────────
def download_pdf(session, d: datetime) -> bytes | None:
    """Download DGR PDF for date d. Returns bytes or None."""
    date_str = d.strftime("%Y-%m-%d")

    # Already in Bronze
    path = bronze_pdf_path(d)
    if path.exists():
        log.info(f"  {date_str} already in Bronze")
        return path.read_bytes()

    try:
        r = session.get(pdf_url(d), timeout=30)
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
            log.info(f"  {date_str} downloaded")
            return r.content
        else:
            log.info(f"  {date_str} not published")
            return None
    except Exception as e:
        log.warning(f"  {date_str} download failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# PARSE PDF
# ─────────────────────────────────────────────────────────────
def _to_float(v):
    try:
        return float(str(v).replace(",", "").strip())
    except Exception:
        return None


def _clean(v) -> str:
    return str(v).replace("\n", " ").strip() if v else ""


def parse_pdf(pdf_bytes: bytes, d: datetime) -> list:
    date_str = d.strftime("%Y-%m-%d")
    meta = {
        "date": date_str,
        "financial_year": financial_year(d),
        "source_pdf": str(bronze_pdf_path(d)),
    }
    rows, current_region = [], None

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables() or []:
                    for trow in table:
                        if not trow:
                            continue
                        cells = [_clean(c) for c in trow]
                        label = cells[0].upper()
                        if not label or label in ("1", "ALL INDIA / REGIONS"):
                            continue
                        is_region = label in REGION_LABELS or any(
                            r in label for r in REGION_LABELS
                        )
                        if (
                            is_region
                            and _to_float(cells[1] if len(cells) > 1 else "") is None
                        ):
                            current_region = cells[0].strip()
                            continue
                        fuel = next(
                            (fk.title() for fk in FUEL_KEYS if fk in label), None
                        )
                        if fuel is None and "TOTAL" in label:
                            fuel = "Total"
                        if fuel and current_region:
                            nums = [_to_float(v) for v in cells[1:]]
                            def g(i):
                                return nums[i] if i < len(nums) else None
                            rows.append(
                                {
                                    **meta,
                                    "region": current_region,
                                    "fuel_type": fuel,
                                    "installed_capacity_mw": g(0),
                                    "monitored_capacity_mw": g(1),
                                    "annual_target_mu": g(2),
                                    "today_program_mu": g(3),
                                    "today_actual_mu": g(4),
                                    "apr1_till_date_program_mu": g(5),
                                    "apr1_till_date_actual_mu": g(6),
                                    "deviation_mu": g(7),
                                    "deviation_pct": g(8),
                                    "parse_ok": True,
                                    "error": "",
                                }
                            )
    except Exception as exc:
        rows.append(
            {
                **meta,
                **{
                    c: None
                    for c in COLUMNS
                    if c
                    not in ("date", "financial_year", "source_pdf", "parse_ok", "error")
                },
                "parse_ok": False,
                "error": str(exc)[:200],
            }
        )

    if not rows:
        rows.append(
            {
                **meta,
                **{
                    c: None
                    for c in COLUMNS
                    if c
                    not in ("date", "financial_year", "source_pdf", "parse_ok", "error")
                },
                "parse_ok": False,
                "error": "No rows extracted",
            }
        )
    return rows


# ─────────────────────────────────────────────────────────────
# SAVE SILVER
# ─────────────────────────────────────────────────────────────
def save_silver(rows: list, date_str: str):
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["ingestion_date"] = date_str
    df["source_file"] = str(bronze_pdf_path(datetime.strptime(date_str, "%Y-%m-%d")))
    df["pipeline_run_id"] = RUN_ID

    path = silver_path(date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")
    log.info(f"  Silver saved: {path.relative_to(PROJECT_ROOT)} ({len(df)} rows)")


# ─────────────────────────────────────────────────────────────
# UPSERT FY CSV
# ─────────────────────────────────────────────────────────────
def upsert_fy_csv(fy: str, df_new: pd.DataFrame):
    path = bronze_csv_path(fy)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path, dtype=str)
        combined = pd.concat([existing, df_new.astype(str)], ignore_index=True)
        combined = (
            combined.drop_duplicates(
                subset=["date", "region", "fuel_type"], keep="last"
            )
            .sort_values("date")
            .reset_index(drop=True)
        )
    else:
        combined = df_new
    combined.to_csv(path, index=False)
    log.info(
        f"  FY CSV updated: {path.relative_to(PROJECT_ROOT)} ({len(combined)} rows)"
    )


def upsert_master(df_new: pd.DataFrame):
    path = bronze_master_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path, dtype=str)
        combined = pd.concat([existing, df_new.astype(str)], ignore_index=True)
        combined = (
            combined.drop_duplicates(
                subset=["date", "region", "fuel_type"], keep="last"
            )
            .sort_values("date")
            .reset_index(drop=True)
        )
    else:
        combined = df_new
    combined.to_csv(path, index=False)
    log.info(f"  Master CSV updated: {len(combined)} total rows")


# ─────────────────────────────────────────────────────────────
# CHECK ALREADY PARSED
# ─────────────────────────────────────────────────────────────
def already_parsed(date_str: str, fy: str) -> bool:
    path = bronze_csv_path(fy)
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, usecols=["date"], dtype=str)
        return date_str in df["date"].values
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    from src.lineage import lineage_run, ds

    args = build_args()
    today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = args.end or today
    start_date = args.start or (today - timedelta(days=args.days))

    if start_date < datetime(2017, 4, 1):
        log.error("Start date before earliest DGR (2017-04-01)")
        sys.exit(1)
    if end_date > today:
        end_date = today

    log.info("=" * 55)
    log.info("GENERATION INGEST  —  LOCAL MODE")
    log.info(f"Range   : {start_date.date()} to {end_date.date()}")
    log.info(f"Run ID  : {RUN_ID[:8]}")
    log.info("Bronze  : data/raw/generation/")
    log.info("Silver  : data/processed/generation/")
    log.info("=" * 55)

    with lineage_run("ingest_generation",
        inputs=[ds("NPP DGR Reports", namespace="https://npp.gov.in")],
        outputs=[ds("data/raw/generation"), ds("data/processed/generation")],
    ):
        session = get_session()
        missing = load_missing()
        results = {}
        by_fy = {}
        cur = start_date

        while cur <= end_date:
            date_str = cur.strftime("%Y-%m-%d")
            fy = financial_year(cur)

            if date_str in missing:
                log.info(f"  {date_str} skipped (known missing)")
                cur += timedelta(days=1)
                continue

            # Skip if already parsed
            if already_parsed(date_str, fy):
                log.info(f"  {date_str} already parsed")
                cur += timedelta(days=1)
                continue

            # Download
            pdf_bytes = download_pdf(session, cur)
            if pdf_bytes is None:
                missing.add(date_str)
                results[date_str] = "NOT_PUBLISHED"
                cur += timedelta(days=1)
                continue

            # Parse
            log.info(f"  Parsing {date_str}")
            rows = parse_pdf(pdf_bytes, cur)

            # Save Silver
            save_silver(rows, date_str)

            # Accumulate by FY
            by_fy.setdefault(fy, []).extend(rows)
            results[date_str] = "SUCCESS"
            cur += timedelta(days=1)

        # Save FY CSVs and master
        all_rows = []
        for fy, rows in sorted(by_fy.items()):
            df_new = pd.DataFrame(rows, columns=COLUMNS)
            upsert_fy_csv(fy, df_new)
            all_rows.extend(rows)

        if all_rows:
            upsert_master(pd.DataFrame(all_rows, columns=COLUMNS))

        save_missing(missing)

        # Save run summary
        summary_path = LOCAL_BRONZE / "run_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "run_id": RUN_ID,
                    "run_at": datetime.utcnow().isoformat(),
                    "mode": "local",
                    "results": results,
                },
                indent=2,
            )
        )

    success = sum(1 for v in results.values() if v == "SUCCESS")
    failed = sum(1 for v in results.values() if v != "SUCCESS")
    log.info("=" * 55)
    log.info(f"Done: {success} success, {failed} failed/missing")
    log.info("=== GENERATION INGEST COMPLETE ===")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
