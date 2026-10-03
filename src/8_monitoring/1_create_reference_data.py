"""
create_reference_data.py
========================
Creates reference data for drift detection by saving the training set features.

Evidently compares incoming (current) data against this reference to detect drift.
Run this once after training, or as part of the training pipeline.

Usage:
    python src/8_monitoring/1_create_reference_data.py
"""

import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEAST_DATA = PROJECT_ROOT / "my_feature_store" / "feature_repo" / "data" / "march_2025_features.parquet"
OUTPUT_PATH = PROJECT_ROOT / "drift_baselines" / "reference_data.parquet"

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
TARGET = "dam_mcp"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [REF-DATA] %(message)s")
log = logging.getLogger("ref_data")


def main():
    log.info("Loading training data from %s", FEAST_DATA)
    df = pd.read_parquet(FEAST_DATA)

    # Keep only features + target that exist in the data
    cols = [c for c in FEATURES + [TARGET] if c in df.columns]
    ref = df[cols].copy()

    # Use the training portion (first 80%) as reference
    n = int(len(ref) * 0.8)
    ref = ref.iloc[:n]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved reference data: %s (%d rows, %d columns)", OUTPUT_PATH, len(ref), len(cols))


if __name__ == "__main__":
    main()
