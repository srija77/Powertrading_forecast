"""
predict.py
==========
Loads the best model from MLflow Model Registry, fetches features from Feast
for a given time range, and generates DAM MCP predictions.

Can be used as:
  - Batch inference: python src/7_prediction/1_predict.py
  - Imported by app.py or other services for on-demand prediction

Pipeline position:
  train.py -> evaluate.py -> predict.py -> predictions/output.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from feast import FeatureStore

# Fix Windows cp1252 encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# -- Config ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEAST_REPO = PROJECT_ROOT / "my_feature_store" / "feature_repo"
OUTPUT_DIR = PROJECT_ROOT / "predictions"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = "dam_mcp_forecast"
TARGET_COL = "dam_mcp"
TIMESTAMP_COL = "event_timestamp"

# Same Feast FeatureService as train.py — the single source of truth for the
# model's feature bundle (defined in feature_definitions.py).
FEATURE_SERVICE = "dam_mcp_forecast_v1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PREDICT] %(message)s")
log = logging.getLogger("predict")
logging.getLogger("feast").setLevel(logging.WARNING)


# -- CLI ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch predictions using best MLflow model")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Specific MLflow run ID to load. If omitted, loads best run by RMSE.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: predictions/batch_predictions.csv)")
    parser.add_argument("--online", action="store_true",
                        help="Fetch features from the Feast ONLINE store instead of the "
                             "offline snapshot. Requires --block-id.")
    parser.add_argument("--block-id", action="append", default=None,
                        help="block_id to score in online mode (e.g. 2025-03-20T18:00). "
                             "Repeatable. Only used with --online.")
    return parser.parse_args()


# -- Load model from MLflow --------------------------------------------------
def load_best_model(run_id: str | None = None):
    """Load model from MLflow — specific run or best run by RMSE."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    if run_id:
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        run = client.get_run(run_id)
        rmse = run.data.metrics.get("rmse", "N/A")
        log.info(f"Loaded model from run {run_id} (RMSE={rmse})")
        return model, run_id

    # Find best run by RMSE
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if not experiment:
        raise RuntimeError(f"Experiment '{MLFLOW_EXPERIMENT}' not found. Run train.py first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.rmse ASC"],
        max_results=5,
    )

    # Only consider runs that have a logged sklearn model artifact
    for run in runs:
        try:
            model_uri = f"runs:/{run.info.run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            rmse = run.data.metrics.get("rmse", "N/A")
            log.info(f"Loaded best model: {run.info.run_name} (run={run.info.run_id}, RMSE={rmse})")
            return model, run.info.run_id
        except Exception:
            continue

    raise RuntimeError("No sklearn model found in MLflow. Run train.py first.")


# -- Fetch features from Feast -----------------------------------------------
def fetch_inference_data() -> pd.DataFrame:
    """Fetch features from Feast offline store for prediction."""
    store = FeatureStore(repo_path=str(FEAST_REPO))
    source_path = FEAST_REPO / "data" / "march_2025_features.parquet"
    source_df = pd.read_parquet(source_path)
    entity_df = source_df[["block_id", "event_timestamp"]].copy()
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])

    feature_service = store.get_feature_service(FEATURE_SERVICE)
    log.info(f"Fetching feature service '{FEATURE_SERVICE}' for {len(entity_df)} blocks...")
    df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_service,
    ).to_df()

    log.info(f"Fetched: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def fetch_online_data(block_ids: list[str]) -> pd.DataFrame:
    """Fetch features from the Feast ONLINE store for the given block_ids."""
    store = FeatureStore(repo_path=str(FEAST_REPO))
    feature_service = store.get_feature_service(FEATURE_SERVICE)
    log.info(f"Fetching online feature service '{FEATURE_SERVICE}' for {len(block_ids)} block(s)...")
    df = store.get_online_features(
        features=feature_service,
        entity_rows=[{"block_id": b} for b in block_ids],
    ).to_df()
    # Online store returns no timestamp; derive it from the block_id key.
    df["event_timestamp"] = pd.to_datetime(df["block_id"])
    missing = df[TARGET_COL].isna().sum() if TARGET_COL in df.columns else 0
    if missing:
        log.warning(f"{missing} block_id(s) had no materialised features (null row).")
    log.info(f"Fetched online: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# -- Run predictions ----------------------------------------------------------
def predict(model, df: pd.DataFrame) -> pd.DataFrame:
    """Run model predictions and return DataFrame with results."""
    df = df.copy()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    timestamps = df[TIMESTAMP_COL]
    block_ids = df["block_id"]
    actuals = df[TARGET_COL] if TARGET_COL in df.columns else None

    exclude_cols = {TARGET_COL, TIMESTAMP_COL, "block_id"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]

    # Match dtype handling from train.py
    num_cols = X.select_dtypes(exclude=["object", "string"]).columns
    for col in num_cols:
        if pd.api.types.is_integer_dtype(X[col]):
            X[col] = X[col].astype(float)

    log.info(f"Running predictions on {len(X)} rows...")
    preds = model.predict(X)

    result = pd.DataFrame({
        "event_timestamp": timestamps,
        "block_id": block_ids,
        "predicted_dam_mcp": np.ravel(preds),
    })

    if actuals is not None:
        result["actual_dam_mcp"] = actuals.values
        result["residual"] = result["actual_dam_mcp"] - result["predicted_dam_mcp"]

    return result


# -- Main --------------------------------------------------------------------
def main():
    args = parse_args()

    mode = "ONLINE" if args.online else "BATCH (offline)"
    log.info("=" * 55)
    log.info(f"{mode} PREDICTION - DAM MCP")
    log.info("=" * 55)

    if args.online and not args.block_id:
        raise SystemExit("--online requires at least one --block-id (e.g. --block-id 2025-03-20T18:00)")

    # 1. Load model
    model, run_id = load_best_model(args.run_id)

    # 2. Fetch features (online store by block_id, or offline snapshot)
    if args.online:
        df = fetch_online_data(args.block_id)
    else:
        df = fetch_inference_data()

    # 3. Predict
    results = predict(model, df)

    # 4. Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "batch_predictions.csv"
    results.to_csv(output_path, index=False)
    log.info(f"Predictions saved: {output_path} ({len(results)} rows)")

    # 5. Summary stats
    log.info("")
    log.info("Prediction statistics:")
    log.info(f"  Mean:   {results['predicted_dam_mcp'].mean():.2f}")
    log.info(f"  Std:    {results['predicted_dam_mcp'].std():.2f}")
    log.info(f"  Min:    {results['predicted_dam_mcp'].min():.2f}")
    log.info(f"  Max:    {results['predicted_dam_mcp'].max():.2f}")

    if "actual_dam_mcp" in results.columns:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import math
        rmse = math.sqrt(mean_squared_error(results["actual_dam_mcp"], results["predicted_dam_mcp"]))
        mae = mean_absolute_error(results["actual_dam_mcp"], results["predicted_dam_mcp"])
        log.info(f"  RMSE:   {rmse:.4f}")
        log.info(f"  MAE:    {mae:.4f}")

    log.info("=" * 55)


if __name__ == "__main__":
    main()
