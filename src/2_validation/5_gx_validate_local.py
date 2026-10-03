"""
gx_validate_local.py
====================
Validates Silver Parquet files from data/processed/.
Saves valid rows   to data/validated/{dataset}/valid/   + local Postgres.
Saves invalid rows to data/validated/{dataset}/invalid/ + local Postgres.
Logs every run to pipeline_runs table.
Saves validation results to GE store so Data Docs shows pass/fail history.

Uses RuntimeBatchRequest — does NOT modify great_expectations.yml.

Usage
-----
  python src/2_validation/5_gx_validate_local.py
  python src/2_validation/5_gx_validate_local.py --date 2026-03-23
  python src/2_validation/5_gx_validate_local.py --dataset dam
  python src/2_validation/5_gx_validate_local.py --dataset rtm --date 2026-03-23
  python src/2_validation/5_gx_validate_local.py --start 2025-03-01 --end 2025-03-31
  python src/2_validation/5_gx_validate_local.py --start 01-03-2025 --end 31-03-2025

  Dates accept YYYY-MM-DD or DD-MM-YYYY.

.env keys needed
-----------------
  POSTGRES_URL   postgresql+psycopg2://postgres:CHANGE_ME@localhost:5432/gmr_project

Dependencies
------------
  pip install great-expectations==0.17.22 pandas pyarrow sqlalchemy psycopg2-binary python-dotenv
"""

import os
import sys
import json
import uuid
import logging
import argparse
from pathlib import Path
from datetime import datetime, date

import pandas as pd
from dotenv import load_dotenv
import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.core.run_identifier import RunIdentifier
from great_expectations.data_context.types.resource_identifiers import (
    ValidationResultIdentifier,
    ExpectationSuiteIdentifier,
)
from sqlalchemy import create_engine, text

load_dotenv()

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql+psycopg2://postgres:CHANGE_ME@localhost:5432/gmr_project",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
GE_ROOT = PROJECT_ROOT / "gx"
LOCAL_SILVER = PROJECT_ROOT / "data" / "processed"
LOCAL_VALID = PROJECT_ROOT / "data" / "validated"

RUN_ID = str(uuid.uuid4())

DATASETS = {
    "dam": "dam",
    "rtm": "rtm",
    "weather": "weather",
    "generation": "generation",
    "calendar": "calendar",
}

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [VALIDATE] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("gx_local")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def build_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--date", default=None, help="Single date YYYY-MM-DD (default: today)"
    )
    p.add_argument("--start", default=None, help="Start of date range YYYY-MM-DD")
    p.add_argument(
        "--end", default=None, help="End of date range YYYY-MM-DD (inclusive)"
    )
    p.add_argument(
        "--dataset",
        default="all",
        choices=["all", "dam", "rtm", "weather", "generation", "calendar"],
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# POSTGRES
# ─────────────────────────────────────────────────────────────
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(POSTGRES_URL)
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log.info("Postgres connected")
    return _engine


# ─────────────────────────────────────────────────────────────
# COLUMN RENAME MAP
# Silver Parquet uses original scraped column names.
# Postgres tables use snake_case lowercase.
# ─────────────────────────────────────────────────────────────
RENAME = {
    "Datetime": "datetime",
    "Purchase Bid (MW)": "purchase_bid_mw",
    "Sell Bid (MW)": "sell_bid_mw",
    "MCV (MW)": "mcv_mw",
    "Final Scheduled Volume (MW)": "final_scheduled_volume_mw",
    "MCP (Rs/MWh) *": "mcp_rs_per_mwh",
    "Session ID": "session_id",
}


# ─────────────────────────────────────────────────────────────
# LOAD SILVER
# ─────────────────────────────────────────────────────────────
def load_silver(dataset: str, date_str: str) -> pd.DataFrame:
    """
    DAM/RTM/Generation: partitioned by date
      data/processed/{dataset}/year=YYYY/month=MM/date=YYYY-MM-DD/part-0001.parquet
    Weather: partitioned by city (one file per city, all dates)
      data/processed/weather/city={City}/part-0001.parquet
    """
    year = date_str[:4]
    month = date_str[5:7]
    name = DATASETS.get(dataset, dataset)

    if dataset == "weather":
        weather_root = LOCAL_SILVER / "weather"
        if not weather_root.exists():
            log.warning(f"  Weather Silver folder not found: {weather_root}")
            return pd.DataFrame()

        frames = []
        for city_dir in weather_root.iterdir():
            parquet = city_dir / "part-0001.parquet"
            if parquet.exists():
                df = pd.read_parquet(parquet)
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"], errors="coerce")
                    df = df[df["time"].dt.date.astype(str) == date_str]
                if not df.empty:
                    frames.append(df)

        if not frames:
            log.warning(f"  No weather Silver data found for {date_str}")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        log.info(f"  Loaded {len(combined)} weather rows for {date_str}")
        return combined

    elif dataset == "calendar":
        cal_path = LOCAL_SILVER / "calendar" / "calendar.parquet"
        if not cal_path.exists():
            log.warning(f"  Calendar Silver not found: {cal_path.relative_to(PROJECT_ROOT)}")
            return pd.DataFrame()
        df = pd.read_parquet(cal_path)
        df = df[df["date"] == date_str]
        if df.empty:
            log.warning(f"  No calendar data for {date_str}")
            return pd.DataFrame()
        log.info(f"  Loaded {len(df)} rows from {cal_path.relative_to(PROJECT_ROOT)} for {date_str}")
        return df

    else:
        path = (
            LOCAL_SILVER
            / name
            / f"year={year}"
            / f"month={month}"
            / f"date={date_str}"
            / "part-0001.parquet"
        )
        if not path.exists():
            log.warning(f"  Silver not found: {path.relative_to(PROJECT_ROOT)}")
            return pd.DataFrame()
        df = pd.read_parquet(path)
        log.info(f"  Loaded {len(df)} rows from {path.relative_to(PROJECT_ROOT)}")
        return df


# ─────────────────────────────────────────────────────────────
# RUNTIME DATASOURCE
# great_expectations.yml has `datasources: {}`, so the pandas runtime
# datasource is registered in memory here (not saved to the yml).
# ─────────────────────────────────────────────────────────────
DATASOURCE_NAME = "pandas_default"


def ensure_datasource(context):
    existing = [d["name"] for d in context.list_datasources()]
    if DATASOURCE_NAME in existing:
        return
    context.add_datasource(
        name=DATASOURCE_NAME,
        class_name="Datasource",
        execution_engine={"class_name": "PandasExecutionEngine"},
        data_connectors={
            "default_runtime_data_connector_name": {
                "class_name": "RuntimeDataConnector",
                "batch_identifiers": ["default_identifier_name"],
            }
        },
        save_changes=False,
    )
    log.info(f"Registered runtime datasource: {DATASOURCE_NAME}")


# ─────────────────────────────────────────────────────────────
# RUN GE VALIDATION
# ─────────────────────────────────────────────────────────────
def run_validation(context, df: pd.DataFrame, dataset: str, date_str: str) -> dict:
    """
    Validate DataFrame against the dataset suite.
    Saves result to GE validations store so Data Docs shows pass/fail history.
    """
    suite_name = f"{dataset}_suite"
    log.info(f"  Running suite: {suite_name}")

    try:
        suite = context.get_expectation_suite(suite_name)

        batch_request = RuntimeBatchRequest(
            datasource_name=DATASOURCE_NAME,
            data_connector_name="default_runtime_data_connector_name",
            data_asset_name=dataset,
            runtime_parameters={"batch_data": df},
            batch_identifiers={"default_identifier_name": "default_identifier"},
        )
        validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite=suite,
        )

        # Run validation. result_format="COMPLETE" is REQUIRED so each failed
        # expectation returns `unexpected_index_list` — the per-row indices used
        # below to split valid/invalid. The default ("BASIC") omits those indices,
        # which silently marks every row valid even when expectations fail.
        result = validator.validate(result_format="COMPLETE")

        # Save result to GE validations store so Data Docs shows it
        try:
            run_name = f"{dataset}__{date_str}__{datetime.utcnow().strftime('%H%M%S')}"
            vid = ValidationResultIdentifier(
                expectation_suite_identifier=ExpectationSuiteIdentifier(
                    expectation_suite_name=suite_name
                ),
                run_id=RunIdentifier(run_name=run_name),
                batch_identifier="default_identifier",
            )
            context.validations_store.set(vid, result)
            context.build_data_docs()
            log.info(f"  Data Docs updated — run: {run_name}")
        except Exception as e:
            log.warning(f"  Data Docs update skipped: {e}")

        # Collect failing rows
        failing_by_row = {}
        for exp_result in result.results:
            if not exp_result.success:
                exp_type = exp_result.expectation_config.expectation_type
                column = exp_result.expectation_config.kwargs.get("column")
                if column:
                    exp_type = f"{exp_type}({column})"
                for idx in exp_result.result.get("unexpected_index_list", []):
                    failing_by_row.setdefault(idx, []).append(exp_type)

        failing = set(failing_by_row.keys())
        passing = set(df.index) - failing
        log.info(f"  Result: {len(passing)} pass, {len(failing)} fail")

        return {
            "success": len(failing) == 0,
            "passing_indices": list(passing),
            "failing_indices": list(failing),
            "failure_reasons": failing_by_row,
            "total": len(df),
            "valid_count": len(passing),
            "invalid_count": len(failing),
        }

    except Exception as e:
        # A GX setup/runtime error says nothing about the data — re-raise so
        # the dataset is reported CRASHED instead of dumping every row to invalid/.
        log.error(f"  Validation error: {e}")
        raise


# ─────────────────────────────────────────────────────────────
# SPLIT VALID / INVALID
# ─────────────────────────────────────────────────────────────
def split_rows(df: pd.DataFrame, result: dict):
    passing = result["passing_indices"]
    failing = result["failing_indices"]
    reasons = result["failure_reasons"]

    valid_df = df.loc[df.index.isin(passing)].copy()
    invalid_df = df.loc[df.index.isin(failing)].copy()

    valid_df["validated_at"] = datetime.utcnow().isoformat()
    valid_df["validation_run_id"] = RUN_ID

    if not invalid_df.empty:
        invalid_df["failure_reason"] = invalid_df.index.map(
            lambda i: " | ".join(reasons.get(i, ["unknown"]))
        )
        invalid_df["failed_expectations"] = invalid_df.index.map(
            lambda i: json.dumps(reasons.get(i, []))
        )
        invalid_df["flagged_at"] = datetime.utcnow().isoformat()
        invalid_df["validation_run_id"] = RUN_ID
        invalid_df["reviewed"] = False

    return valid_df, invalid_df


# ─────────────────────────────────────────────────────────────
# SAVE TO LOCAL PARQUET
# ─────────────────────────────────────────────────────────────
def save_parquet(df: pd.DataFrame, dataset: str, date_str: str, kind: str):
    # On an empty result, still clear this date's output from earlier runs
    # so stale rows don't linger in valid/ or invalid/.
    if dataset == "calendar":
        path = LOCAL_VALID / dataset / kind / "calendar.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pd.read_parquet(path)
            existing = existing[existing["date"] != date_str]
            df = pd.concat([existing, df], ignore_index=True)
            df = df.sort_values("date").reset_index(drop=True)
        if df.empty:
            path.unlink(missing_ok=True)
            return
        df.to_parquet(path, index=False, compression="snappy")
        log.info(
            f"  Saved {kind} parquet: {path.relative_to(PROJECT_ROOT)} ({len(df)} rows total)"
        )
    else:
        year = date_str[:4]
        month = date_str[5:7]
        path = (
            LOCAL_VALID
            / dataset
            / kind
            / f"year={year}"
            / f"month={month}"
            / f"date={date_str}"
            / "part-0001.parquet"
        )
        if df.empty:
            if path.exists():
                path.unlink()
                log.info(f"  Removed stale {kind} parquet: {path.relative_to(PROJECT_ROOT)}")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, compression="snappy")
        log.info(
            f"  Saved {kind} parquet: " f"{path.relative_to(PROJECT_ROOT)} ({len(df)} rows)"
        )


# ─────────────────────────────────────────────────────────────
# SAVE TO POSTGRES  (idempotent — delete then insert)
# ─────────────────────────────────────────────────────────────
def save_postgres(df: pd.DataFrame, table: str, date_str: str):
    """Delete existing rows for this date, then insert fresh. No duplicates on re-run."""
    if df.empty:
        return
    try:
        pg_df = df.rename(columns=RENAME)
        engine = get_engine()
        with engine.begin() as conn:
            if "datetime" in pg_df.columns:
                conn.execute(
                    text(f"DELETE FROM {table} WHERE DATE(datetime) = :d"),
                    {"d": date_str},
                )
            elif "date" in pg_df.columns:
                conn.execute(
                    text(f"DELETE FROM {table} WHERE date = :d"), {"d": date_str}
                )
            elif "time" in pg_df.columns:
                conn.execute(
                    text(f"DELETE FROM {table} WHERE DATE(time) = :d"), {"d": date_str}
                )
        pg_df.to_sql(
            table,
            engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=500,
        )
        log.info(f"  Saved to Postgres {table} ({len(pg_df)} rows)")
    except Exception as e:
        log.error(f"  Postgres write failed [{table}]: {e}")


# ─────────────────────────────────────────────────────────────
# LOG TO pipeline_runs
# ─────────────────────────────────────────────────────────────
def log_run(dataset: str, result: dict):
    try:
        with get_engine().begin() as conn:
            conn.execute(
                text(
                    "DELETE FROM pipeline_runs WHERE dataset_name = :ds AND DATE(started_at) = :d"
                ),
                {"ds": dataset, "d": result.get("date")},
            )
            conn.execute(
                text(
                    """
                INSERT INTO pipeline_runs (
                    run_id, pipeline_name, dataset_name, status,
                    total_rows, valid_rows, invalid_rows,
                    started_at, completed_at, run_duration_seconds, triggered_by
                ) VALUES (
                    :run_id, :pipeline, :dataset, :status,
                    :total, :valid, :invalid,
                    :started, :completed, :duration, :triggered_by
                )
            """
                ),
                {
                    "run_id": RUN_ID,
                    "pipeline": "gx_validate_local",
                    "dataset": dataset,
                    "status": result.get("status", "UNKNOWN"),
                    "total": result.get("total", 0),
                    "valid": result.get("valid_count", 0),
                    "invalid": result.get("invalid_count", 0),
                    "started": result.get("started_at"),
                    "completed": result.get("completed_at"),
                    "duration": result.get("duration_s", 0),
                    "triggered_by": "manual",
                },
            )
    except Exception as e:
        log.warning(f"  pipeline_runs log failed: {e}")


# ─────────────────────────────────────────────────────────────
# VALIDATE ONE DATASET
# ─────────────────────────────────────────────────────────────
def validate_dataset(context, dataset: str, date_str: str) -> dict:
    from src.lineage import lineage_run, ds

    started = datetime.utcnow()
    log.info(f"\n--- {dataset.upper()} ---")

    df = load_silver(dataset, date_str)
    if df.empty:
        return {
            "dataset": dataset,
            "date": date_str,
            "success": False,
            "status": "NO_DATA",
            "total": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "started_at": started.isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "duration_s": 0,
        }

    with lineage_run(f"validate_{dataset}",
        inputs=[ds(f"data/processed/{dataset}")],
        outputs=[
            ds(f"data/validated/{dataset}/valid"),
            ds(f"data/validated/{dataset}/invalid"),
        ],
    ):
        ge_result = run_validation(context, df, dataset, date_str)
        valid_df, invalid_df = split_rows(df, ge_result)

        save_parquet(valid_df, dataset, date_str, "valid")
        save_parquet(invalid_df, dataset, date_str, "invalid")
        save_postgres(valid_df, f"{dataset}_valid", date_str)
        save_postgres(invalid_df, f"{dataset}_invalid", date_str)

    completed = datetime.utcnow()
    result = {
        "dataset": dataset,
        "date": date_str,
        "success": ge_result["success"],
        "status": "SUCCESS" if ge_result["success"] else "PARTIAL_FAIL",
        "total": ge_result["total"],
        "valid_count": ge_result["valid_count"],
        "invalid_count": ge_result["invalid_count"],
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "duration_s": int((completed - started).total_seconds()),
    }
    log_run(dataset, result)
    log.info(
        f"  {dataset.upper()} done: "
        f"{result['valid_count']} valid, "
        f"{result['invalid_count']} invalid"
    )
    return result


# ─────────────────────────────────────────────────────────────
# PRINT REPORT
# ─────────────────────────────────────────────────────────────
def print_report(all_results: dict, date_str: str):
    log.info(f"\n{'=' * 50}")
    log.info(f"  REPORT  {date_str}  LOCAL")
    log.info(f"  {'Dataset':12s} {'Total':>7s} {'Valid':>7s} {'Invalid':>9s}  Status")
    log.info(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*9}  {'-'*14}")

    grand_valid = grand_invalid = 0
    for ds, r in all_results.items():
        log.info(
            f"  {ds:12s} {r['total']:>7d} "
            f"{r['valid_count']:>7d} "
            f"{r['invalid_count']:>9d}  {r['status']}"
        )
        grand_valid += r["valid_count"]
        grand_invalid += r["invalid_count"]

    log.info(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*9}")
    log.info(
        f"  {'TOTAL':12s} "
        f"{grand_valid + grand_invalid:>7d} "
        f"{grand_valid:>7d} "
        f"{grand_invalid:>9d}"
    )
    log.info(f"{'=' * 50}")

    LOCAL_VALID.mkdir(parents=True, exist_ok=True)
    path = LOCAL_VALID / f"run_summary_{date_str}.json"
    path.write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "date": date_str,
                "mode": "local",
                "run_at": datetime.utcnow().isoformat(),
                "results": all_results,
            },
            indent=2,
        )
    )
    log.info(f"  Summary: {path.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def _parse_date(value: str) -> date:
    """Parse YYYY-MM-DD or DD-MM-YYYY."""
    for fmt in ("%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Invalid date '{value}' — use YYYY-MM-DD or DD-MM-YYYY")


def _build_date_list(args) -> list:
    """Build date list from --date, --start/--end, or default to today."""
    from datetime import timedelta

    try:
        if args.start and args.end:
            start = _parse_date(args.start)
            end = _parse_date(args.end)
            if end < start:
                log.error("--end must be >= --start")
                return []
            dates, cur = [], start
            while cur <= end:
                dates.append(cur.strftime("%Y-%m-%d"))
                cur += timedelta(days=1)
            return dates
        elif args.start or args.end:
            log.error("--start and --end must be used together")
            return []
        elif args.date:
            return [_parse_date(args.date).strftime("%Y-%m-%d")]
    except ValueError as e:
        log.error(str(e))
        return []
    else:
        return [date.today().strftime("%Y-%m-%d")]


def main():
    args = build_args()
    dataset = args.dataset
    dates = _build_date_list(args)

    if not dates:
        return

    log.info("=" * 50)
    log.info("GX VALIDATION  —  LOCAL")
    log.info(f"Dates   : {dates[0]} to {dates[-1]}  ({len(dates)} day(s))")
    log.info(f"Dataset : {dataset}")
    log.info(f"Run ID  : {RUN_ID[:8]}")
    log.info(f"GE Root : {GE_ROOT}")
    log.info("=" * 50)

    try:
        context = gx.DataContext(context_root_dir=str(GE_ROOT))
        ensure_datasource(context)
        log.info("GE context loaded")
        log.info(f"Suites: {context.list_expectation_suite_names()}")
    except Exception as e:
        log.error(f"GE context failed: {e}")
        return

    active = list(DATASETS.keys())
    to_run = active if dataset == "all" else ([dataset] if dataset in active else [])

    if not to_run:
        log.warning(f"Dataset '{dataset}' is not active. Active: {active}")
        return

    for date_str in dates:
        log.info("\n" + "=" * 50)
        log.info(f"  DATE: {date_str}")
        log.info("=" * 50)
        all_results = {}
        for ds in to_run:
            try:
                all_results[ds] = validate_dataset(context, ds, date_str)
            except Exception as e:
                log.error(f"{ds} crashed: {e}")
                all_results[ds] = {
                    "dataset": ds,
                    "date": date_str,
                    "success": False,
                    "status": "CRASHED",
                    "total": 0,
                    "valid_count": 0,
                    "invalid_count": 0,
                    "started_at": datetime.utcnow().isoformat(),
                    "completed_at": datetime.utcnow().isoformat(),
                    "duration_s": 0,
                }
        print_report(all_results, date_str)

    log.info("=== VALIDATION COMPLETE ===")


if __name__ == "__main__":
    main()
