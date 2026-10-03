"""
prepare_feast_data.py
=====================
Reads the March 2025 EDA feature Parquet, deduplicates, adds Feast-required
columns (event_timestamp, block_id entity key), and writes a clean Parquet
that Feast can ingest.

Usage:
    python src/3_feature_engineering/2_prepare_feast_data.py
"""

import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Ensure project root is importable (so `from src.lineage import ...` works when this
# script is run directly, e.g. by the DVC `prepare_feast_data` stage from the repo root).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
INPUT_PATH = PROJECT_ROOT / "data" / "features" / "march_2025_eda" / "march_2025_prepared.parquet"
OUTPUT_PATH = PROJECT_ROOT / "my_feature_store" / "feature_repo" / "data" / "march_2025_features.parquet"


def main():
    from src.lineage import lineage_run, ds

    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Feast requires an event_timestamp (datetime64[ns, UTC] or datetime64[ns])
    df["event_timestamp"] = pd.to_datetime(df["Datetime"])
    df = df.drop(columns=["Datetime"])

    # Entity key: block_id = unique 15-min slot identifier
    # Format: "2025-03-15T14:30" — one per 15-min block
    df["block_id"] = df["event_timestamp"].dt.strftime("%Y-%m-%dT%H:%M")

    # Deduplicate to one row per block_id — average numeric cols, first for others
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols and c != "block_id"]
    agg_dict = {c: "mean" for c in numeric_cols}
    agg_dict.update({c: "first" for c in non_numeric_cols})
    df = df.groupby("block_id", as_index=False).agg(agg_dict)
    print(f"After dedup: {df.shape[0]} rows (one per 15-min block)")

    # Sort by time
    df = df.sort_values("event_timestamp").reset_index(drop=True)

    with lineage_run("prepare_feast_data",
        inputs=[ds("data/features/march_2025_eda/march_2025_prepared.parquet")],
        outputs=[ds("feast/march_2025_features.parquet")],
    ):
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUTPUT_PATH, index=False)
        print(f"Saved: {OUTPUT_PATH}")
        print(f"Final shape: {df.shape}")
        print(f"Date range: {df['event_timestamp'].min()} to {df['event_timestamp'].max()}")


if __name__ == "__main__":
    main()
