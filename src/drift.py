"""
drift.py
========
Data drift detection using Evidently AI.

Compares incoming (current) prediction data against the training reference data.
Produces a drift summary JSON that the Flask app reads and exposes to Prometheus.

Can be used:
  1. Automatically — called by app.py after batch predictions
  2. Manually — python src/drift.py --current data.csv

Pipeline position:
  train.py -> create_reference_data.py -> app.py (batch-predict) -> drift.py -> Grafana alert

Usage:
    # Auto: called by app.py after /batch-predict
    from src.drift import run_drift_check
    summary = run_drift_check(current_df)

    # Manual: run against a CSV file
    python src/drift.py --current path/to/data.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = PROJECT_ROOT / "drift_baselines" / "reference_data.parquet"
DRIFT_REPORT_PATH = PROJECT_ROOT / "monitoring" / "drift_summary.json"

FEATURES = [
    "dam_purchase_bid", "dam_sell_bid", "dam_mcv", "dam_volume",
    "dam_bid_imbalance", "rtm_purchase_bid", "rtm_sell_bid",
    "rtm_mcv", "rtm_volume", "mcp_spread",
    "dam_mcp_lag_1d", "dam_mcp_lag_2d", "dam_mcp_lag_7d",
    "dam_mcp_roll_4h", "dam_mcp_roll_24h", "dam_mcp_roll_std_24h",
    "avg_temp", "avg_humidity", "avg_windspeed",
    "avg_cloud_cover", "total_rainfall",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "block_sin", "block_cos", "day_of_month",
    "is_weekend", "is_peak_hour", "is_morning_ramp", "is_off_peak",
    "is_ipl_match", "is_event", "is_festival", "is_wedding_season", "impact",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [DRIFT] %(message)s")
log = logging.getLogger("drift")


def load_reference() -> pd.DataFrame:
    """Load the reference (training) dataset."""
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Reference data not found at {REFERENCE_PATH}. "
            "Run: python src/8_monitoring/1_create_reference_data.py"
        )
    return pd.read_parquet(REFERENCE_PATH)


def run_drift_check(current_df: pd.DataFrame, reference_df: pd.DataFrame = None) -> dict:
    """
    Run drift detection comparing current data against reference data.

    Args:
        current_df: DataFrame with incoming prediction features
        reference_df: Reference DataFrame (loaded from disk if not provided)

    Returns:
        Drift summary dict with per-feature results
    """
    if reference_df is None:
        reference_df = load_reference()

    # Use only features present in both datasets
    common_features = [f for f in FEATURES if f in current_df.columns and f in reference_df.columns]

    if len(common_features) < 5:
        log.warning("Too few common features (%d). Skipping drift check.", len(common_features))
        return {"dataset_drift": False, "error": "Too few common features"}

    # Filter out constant columns (zero variance breaks K-S test)
    varying_features = [f for f in common_features
                        if reference_df[f].nunique() > 1 and current_df[f].nunique() > 1]
    skipped = set(common_features) - set(varying_features)
    if skipped:
        log.info("Skipping %d constant features: %s", len(skipped), ", ".join(sorted(skipped)))

    ref_data = reference_df[varying_features].copy()
    cur_data = current_df[varying_features].copy()

    log.info("Running drift check: %d reference rows vs %d current rows, %d features",
             len(ref_data), len(cur_data), len(varying_features))

    # Build Evidently datasets
    data_def = DataDefinition(numerical_columns=varying_features)
    ref_ds = Dataset.from_pandas(ref_data, data_definition=data_def)
    cur_ds = Dataset.from_pandas(cur_data, data_definition=data_def)

    # Run drift report
    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)
    result = snapshot.dict()

    # Parse results
    drifted_features = []
    feature_details = {}
    drifted_share = 0.0

    for metric in result.get("metrics", []):
        metric_name = metric.get("metric_name", "")
        config = metric.get("config", {})

        # DriftedColumnsCount gives the overall summary
        if "DriftedColumnsCount" in metric_name:
            drifted_share = float(metric["value"].get("share", 0))

        # ValueDrift gives per-column drift score
        # The interpretation depends on the method Evidently chooses:
        #   - K-S p_value: low value = drift (value < threshold)
        #   - Wasserstein distance: high value = drift (value >= threshold)
        # In both cases, Evidently's DriftedColumnsCount handles the logic,
        # so we compare value >= threshold for distance-based methods
        # and value < threshold for p_value methods.
        if "ValueDrift" in metric_name:
            column = config.get("column", "")
            method = config.get("method", "")
            threshold = config.get("threshold", 0.1)
            score = float(metric.get("value", 0.0))

            # p_value methods: low = drift. Distance methods: high = drift.
            if "p_value" in method.lower():
                is_drifted = score < threshold
            else:
                is_drifted = score >= threshold

            feature_details[column] = {
                "score": round(score, 6),
                "method": method,
                "threshold": threshold,
                "drifted": bool(is_drifted),
            }
            if is_drifted:
                drifted_features.append(column)

    # Dataset-level drift: use Evidently's own threshold (>50% columns drifted)
    dataset_drift = drifted_share > 0.5

    summary = {
        "dataset_drift": dataset_drift,
        "drifted_features_count": len(drifted_features),
        "total_features": len(common_features),
        "drift_share": round(len(drifted_features) / len(common_features), 4) if common_features else 0,
        "drifted_features": drifted_features,
        "feature_details": feature_details,
        "reference_rows": len(ref_data),
        "current_rows": len(cur_data),
        "report_time": datetime.now(timezone.utc).isoformat(),
    }

    # Save to disk (convert numpy types for JSON serialization)
    def _convert(obj):
        import numpy as np
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=_convert)

    log.info("Drift result: %s | %d/%d features drifted (%.1f%%)",
             "DRIFT DETECTED" if dataset_drift else "No drift",
             len(drifted_features), len(common_features),
             len(drifted_features) / len(common_features) * 100 if common_features else 0)

    if drifted_features:
        log.info("Drifted features: %s", ", ".join(drifted_features))

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run drift detection")
    parser.add_argument("--current", required=True, help="Path to current data CSV or Parquet")
    args = parser.parse_args()

    path = Path(args.current)
    if path.suffix == ".parquet":
        current_df = pd.read_parquet(path)
    else:
        current_df = pd.read_csv(path)

    summary = run_drift_check(current_df)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
