"""
validate_dam.py
===============
Validates DAM (Day Ahead Market) Silver Parquet data.
Applies dam_suite.json rules via Great Expectations.
Saves valid rows to *_valid, invalid rows to *_invalid.

Called by gx_validate.py — do not run directly.
Can also be run standalone for testing:
  python src/2_validation/1_validate_dam.py --date 2026-03-23 --mode local
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

log = logging.getLogger("validate_dam")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SILVER = PROJECT_ROOT / "data" / "processed"
LOCAL_VALID = PROJECT_ROOT / "data" / "validated"
S3_BUCKET = "aispry-projects-demo-bucket"

DATASET = "dam"
SUITE_NAME = "dam_suite"
SILVER_FOLDER = "dam"


# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────
def load(date_str: str, mode: str, s3=None) -> pd.DataFrame:
    """Load DAM Silver Parquet for the given date."""
    year = date_str[:4]
    month = date_str[5:7]

    if mode == "local":
        path = (
            LOCAL_SILVER
            / SILVER_FOLDER
            / f"year={year}"
            / f"month={month}"
            / f"date={date_str}"
            / "part-0001.parquet"
        )
        if not path.exists():
            log.warning(f"DAM silver not found: {path.relative_to(PROJECT_ROOT)}")
            return pd.DataFrame()
        df = pd.read_parquet(path)
        log.info(f"DAM loaded {len(df)} rows from {path.relative_to(PROJECT_ROOT)}")
        return df

    else:
        import io

        key = (
            f"silver/{SILVER_FOLDER}/year={year}/month={month}"
            f"/date={date_str}/part-0001.parquet"
        )
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
            log.info(f"DAM loaded {len(df)} rows from s3://{S3_BUCKET}/{key}")
            return df
        except Exception as e:
            log.warning(f"DAM silver not found in S3: {key} ({e})")
            return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# VALIDATE
# ─────────────────────────────────────────────────────────────
def validate(context, df: pd.DataFrame) -> dict:
    """Run dam_suite expectations against the DataFrame."""
    log.info(f"DAM running suite: {SUITE_NAME}")
    try:
        validator = context.sources.pandas_default.read_dataframe(
            df, asset_name=DATASET
        )
        suite = context.get_expectation_suite(SUITE_NAME)
        result = validator.validate(expectation_suite=suite)

        failing_by_row = {}
        for exp_result in result.results:
            if not exp_result.success:
                exp_type = exp_result.expectation_config.expectation_type
                for idx in exp_result.result.get("unexpected_index_list", []):
                    failing_by_row.setdefault(idx, []).append(exp_type)

        failing = set(failing_by_row.keys())
        passing = set(df.index) - failing

        log.info(f"DAM result: {len(passing)} pass, {len(failing)} fail")
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
        log.error(f"DAM validation error: {e}")
        return {
            "success": False,
            "passing_indices": [],
            "failing_indices": list(df.index),
            "failure_reasons": {i: [str(e)] for i in df.index},
            "total": len(df),
            "valid_count": 0,
            "invalid_count": len(df),
        }


# ─────────────────────────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────────────────────────
def split(df: pd.DataFrame, ge_result: dict, run_id: str):
    """Split into valid and invalid DataFrames."""
    passing = ge_result["passing_indices"]
    failing = ge_result["failing_indices"]
    reasons = ge_result["failure_reasons"]

    valid_df = df.loc[df.index.isin(passing)].copy()
    invalid_df = df.loc[df.index.isin(failing)].copy()

    valid_df["validated_at"] = datetime.utcnow().isoformat()
    valid_df["validation_run_id"] = run_id

    if not invalid_df.empty:
        invalid_df["failure_reason"] = invalid_df.index.map(
            lambda i: " | ".join(reasons.get(i, ["unknown"]))
        )
        invalid_df["failed_expectations"] = invalid_df.index.map(
            lambda i: json.dumps(reasons.get(i, []))
        )
        invalid_df["flagged_at"] = datetime.utcnow().isoformat()
        invalid_df["validation_run_id"] = run_id
        invalid_df["reviewed"] = False

    return valid_df, invalid_df


# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────
def save(df: pd.DataFrame, date_str: str, kind: str, mode: str, s3=None):
    """Save valid or invalid rows to local Parquet or S3."""
    if df.empty:
        return
    year = date_str[:4]
    month = date_str[5:7]

    if mode == "local":
        path = (
            LOCAL_VALID
            / DATASET
            / kind
            / f"year={year}"
            / f"month={month}"
            / f"date={date_str}"
            / "part-0001.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, compression="snappy")
        log.info(f"DAM {kind} saved: {path.relative_to(PROJECT_ROOT)} ({len(df)} rows)")

    else:
        import io as _io

        key = (
            f"validated/{DATASET}/{kind}/year={year}/month={month}"
            f"/date={date_str}/part-0001.parquet"
        )
        buf = _io.BytesIO()
        df.to_parquet(buf, index=False, compression="snappy")
        buf.seek(0)
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
        log.info(f"DAM {kind} saved: s3://{S3_BUCKET}/{key} ({len(df)} rows)")


# ─────────────────────────────────────────────────────────────
# SAVE TO POSTGRES
# ─────────────────────────────────────────────────────────────
def save_postgres(valid_df, invalid_df, engine):
    """Save valid and invalid rows to Postgres."""
    for df, table in [(valid_df, "dam_valid"), (invalid_df, "dam_invalid")]:
        if df.empty:
            continue
        try:
            df.to_sql(
                table,
                engine,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=500,
            )
            log.info(f"DAM saved to Postgres {table} ({len(df)} rows)")
        except Exception as e:
            log.error(f"DAM Postgres write failed [{table}]: {e}")


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT (called by gx_validate.py)
# ─────────────────────────────────────────────────────────────
def run(context, date_str: str, mode: str, run_id: str, engine=None, s3=None) -> dict:
    """
    Full DAM validation pipeline.
    Called by gx_validate.py with shared context, engine, s3 client.
    """
    started = datetime.utcnow()

    df = load(date_str, mode, s3)
    if df.empty:
        return {
            "dataset": DATASET,
            "status": "NO_DATA",
            "total": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "started_at": started.isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "duration_s": 0,
        }

    ge_result = validate(context, df)
    valid_df, invalid_df = split(df, ge_result, run_id)

    save(valid_df, date_str, "valid", mode, s3)
    save(invalid_df, date_str, "invalid", mode, s3)

    if engine:
        save_postgres(valid_df, invalid_df, engine)

    completed = datetime.utcnow()
    return {
        "dataset": DATASET,
        "status": "SUCCESS" if ge_result["success"] else "PARTIAL_FAIL",
        "total": ge_result["total"],
        "valid_count": ge_result["valid_count"],
        "invalid_count": ge_result["invalid_count"],
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "duration_s": int((completed - started).total_seconds()),
    }


# ─────────────────────────────────────────────────────────────
# STANDALONE (for testing)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    import great_expectations as gx

    load_dotenv()

    p = argparse.ArgumentParser()
    p.add_argument("--date", default=datetime.today().strftime("%Y-%m-%d"))
    p.add_argument("--mode", default="local", choices=["local", "cloud"])
    args = build_args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [DAM] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    ge_root = PROJECT_ROOT / "gx"
    context = gx.DataContext(context_root_dir=str(ge_root))

    engine = s3 = None
    if args.mode == "cloud":
        import boto3
        import os
        from sqlalchemy import create_engine

        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
        engine = create_engine(os.getenv("POSTGRES_URL"))
    else:
        from sqlalchemy import create_engine

        engine = create_engine(
            os.getenv(
                "POSTGRES_URL",
                "postgresql+psycopg2://postgres:CHANGE_ME@localhost:5432/gmr_project",
            )
        )

    result = run(
        context, args.date, args.mode, str(__import__("uuid").uuid4()), engine, s3
    )
    print(result)
