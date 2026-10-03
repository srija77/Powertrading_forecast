"""
evaluate.py
===========
Evaluates the best trained model against a holdout test set, logs evaluation
metrics and plots to MLflow, and optionally gates model promotion.

If the model passes the quality threshold, it is tagged as "passed_eval=true"
in MLflow so downstream systems (app.py, CI/CD) can trust it.

The test partition scored here is the tail that train.py withheld (see
src/splits.py) — no model was fitted on it and no champion was selected using
it, so these metrics are an unbiased generalisation estimate and the gate below
means something. The split fractions are read back from metrics.json, so this
script always scores exactly the rows the training run set aside.

Pipeline position:
  train.py -> evaluate.py -> predict.py
  (DVC stage: evaluate, runs after train)

Usage:
    python src/5_evaluation/1_evaluate.py
    python src/5_evaluation/1_evaluate.py --rmse-threshold 500 --mape-threshold 15
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from feast import FeatureStore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Fix Windows cp1252 encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# -- Config ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.splits import resolve_split, split_bounds  # noqa: E402

FEAST_REPO = PROJECT_ROOT / "my_feature_store" / "feature_repo"
METRICS_PATH = PROJECT_ROOT / "metrics.json"
EVAL_METRICS_PATH = PROJECT_ROOT / "eval_metrics.json"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = "dam_mcp_forecast"
TARGET_COL = "dam_mcp"
TIMESTAMP_COL = "event_timestamp"

# Same Feast FeatureService as train.py — the single source of truth for the
# model's feature bundle (defined in feature_definitions.py).
FEATURE_SERVICE = "dam_mcp_forecast_v1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [EVALUATE] %(message)s")
log = logging.getLogger("evaluate")
logging.getLogger("feast").setLevel(logging.WARNING)


# -- CLI ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate best model and gate promotion")
    parser.add_argument("--val-size", type=float, default=None,
                        help="Override the validation fraction. Defaults to the split "
                             "recorded in metrics.json by the training run.")
    parser.add_argument("--test-size", type=float, default=None,
                        help="Override the test fraction. Defaults to the split "
                             "recorded in metrics.json by the training run.")
    parser.add_argument("--rmse-threshold", type=float, default=None,
                        help="Max acceptable RMSE. If exceeded, model fails evaluation.")
    parser.add_argument("--mape-threshold", type=float, default=None,
                        help="Max acceptable MAPE (percent). If exceeded, model fails evaluation.")
    return parser.parse_args()


# -- Load best model from MLflow ---------------------------------------------
REGISTERED_MODEL_NAME = "dam_mcp_forecast"

def _get_proxy_host():
    """Extract host:port from tracking URI for artifact proxy."""
    from urllib.parse import urlparse
    parsed = urlparse(MLFLOW_TRACKING_URI)
    return f"{parsed.hostname}:{parsed.port}"

def load_best_model():
    """Load the latest version of the registered model from MLflow.

    Returns (model, run_id, run_name, version) so callers can tag
    both the run and the model version in the registry.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    # Get latest model version from registry
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        raise RuntimeError(f"No model versions for '{REGISTERED_MODEL_NAME}'. Run train.py first.")
    latest = max(versions, key=lambda v: int(v.version))
    run_id = latest.run_id

    # Load model using standard MLflow URIs
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{latest.version}"
    log.info(f"Loading model: {model_uri}")
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        # Fallback: load from the run's artifact path
        run_uri = f"runs:/{run_id}/model"
        log.info(f"Falling back to run artifact: {run_uri}")
        model = mlflow.sklearn.load_model(run_uri)

    run = client.get_run(run_id)
    run_name = run.info.run_name
    rmse = run.data.metrics.get("rmse", "N/A")
    log.info(f"Loaded model: {REGISTERED_MODEL_NAME} v{latest.version} "
             f"(run={run_name}, id={run_id}, train_RMSE={rmse})")
    return model, run_id, run_name, latest.version


# -- Fetch and prepare data --------------------------------------------------
def fetch_and_split(val_size_override: float | None, test_size_override: float | None):
    """Fetch from Feast and return only the test partition train.py withheld."""
    val_size, test_size, source = resolve_split(
        val_size_override, test_size_override, METRICS_PATH
    )
    log.info(f"Split: val={val_size}, test={test_size} (from {source})")

    store = FeatureStore(repo_path=str(FEAST_REPO))
    source_path = FEAST_REPO / "data" / "march_2025_features.parquet"
    source_df = pd.read_parquet(source_path)
    entity_df = source_df[["block_id", "event_timestamp"]].copy()
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])

    feature_service = store.get_feature_service(FEATURE_SERVICE)
    log.info(f"Fetching feature service '{FEATURE_SERVICE}' for {len(entity_df)} blocks from Feast...")
    df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_service,
    ).to_df()

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    y = df[TARGET_COL]
    exclude_cols = {TARGET_COL, TIMESTAMP_COL, "block_id"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]

    num_cols = X.select_dtypes(exclude=["object", "string"]).columns
    for col in num_cols:
        if pd.api.types.is_integer_dtype(X[col]):
            X[col] = X[col].astype(float)

    # Same boundaries as train.py, via the same helper. Only [test_start:] is
    # read — the rows the model was fitted on and selected with stay excluded.
    _, test_start = split_bounds(len(X), val_size, test_size)
    X_test = X.iloc[test_start:]
    y_test = y.iloc[test_start:]
    ts_test = df[TIMESTAMP_COL].iloc[test_start:]

    log.info(f"Holdout test set: {len(X_test)} rows (last {test_size*100:.0f}% by time), "
             f"unseen by training and model selection")
    return X_test, y_test, ts_test


# -- Compute metrics ---------------------------------------------------------
def compute_metrics(y_true, y_pred) -> dict:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    r2 = r2_score(y_true, y_pred)
    return {"eval_rmse": rmse, "eval_mae": mae, "eval_mape": mape, "eval_r2": r2}


# -- Generate evaluation plots -----------------------------------------------
def log_eval_plots(y_true, y_pred, timestamps):
    """Log evaluation charts to the active MLflow run as artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Predictions vs Actuals over time
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(timestamps.values, y_true, label="Actual", linewidth=0.8, alpha=0.8)
        ax.plot(timestamps.values, y_pred, label="Predicted", linewidth=0.8, alpha=0.8)
        ax.set_title("Evaluation: Predictions vs Actuals")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("DAM MCP")
        ax.legend()
        fig.tight_layout()
        p = Path(tmpdir) / "eval_pred_vs_actual.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        mlflow.log_artifact(str(p), artifact_path="eval_plots")

        # 2. Residuals over time
        residuals = np.array(y_true) - np.array(y_pred)
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(timestamps.values, residuals, linewidth=0.6, alpha=0.7)
        ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
        ax.set_title("Evaluation: Residuals")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Residual")
        fig.tight_layout()
        p = Path(tmpdir) / "eval_residuals.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        mlflow.log_artifact(str(p), artifact_path="eval_plots")

        # 3. Scatter plot
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_true, y_pred, alpha=0.3, s=5)
        mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=0.8, label="Perfect")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Evaluation: Actual vs Predicted")
        ax.legend()
        fig.tight_layout()
        p = Path(tmpdir) / "eval_scatter.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        mlflow.log_artifact(str(p), artifact_path="eval_plots")

        # 4. Residual distribution
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--")
        ax.set_title("Evaluation: Residual Distribution")
        ax.set_xlabel("Residual")
        fig.tight_layout()
        p = Path(tmpdir) / "eval_residual_dist.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        mlflow.log_artifact(str(p), artifact_path="eval_plots")


# -- Quality gate ------------------------------------------------------------
def check_thresholds(metrics: dict, rmse_threshold: float | None,
                     mape_threshold: float | None) -> bool:
    """Return True if model passes all quality gates."""
    passed = True

    if rmse_threshold is not None:
        if metrics["eval_rmse"] > rmse_threshold:
            log.warning(f"FAILED: RMSE {metrics['eval_rmse']:.4f} > threshold {rmse_threshold}")
            passed = False
        else:
            log.info(f"PASSED: RMSE {metrics['eval_rmse']:.4f} <= threshold {rmse_threshold}")

    if mape_threshold is not None:
        if metrics["eval_mape"] > mape_threshold:
            log.warning(f"FAILED: MAPE {metrics['eval_mape']:.2f}% > threshold {mape_threshold}%")
            passed = False
        else:
            log.info(f"PASSED: MAPE {metrics['eval_mape']:.2f}% <= threshold {mape_threshold}%")

    if rmse_threshold is None and mape_threshold is None:
        log.info("No thresholds set — model passes by default.")

    return passed


# -- Main --------------------------------------------------------------------
def main():
    from src.lineage import lineage_run, ds

    args = parse_args()

    log.info("=" * 55)
    log.info("EVALUATION - DAM MCP Best Model")
    log.info("=" * 55)

    with lineage_run("evaluate",
        inputs=[ds("models/model.pkl"), ds("feast/march_2025_features.parquet")],
        outputs=[ds("eval_metrics.json")],
    ):
        # 1. Load best model from MLflow
        model, run_id, run_name, model_version = load_best_model()

        # 2. Fetch the holdout partition training never saw
        X_test, y_test, ts_test = fetch_and_split(args.val_size, args.test_size)

        # 3. Run predictions
        log.info("Running predictions on evaluation set...")
        y_pred = model.predict(X_test)

        # 4. Compute metrics
        metrics = compute_metrics(y_test.values, y_pred)

        # 5. Quality gate
        passed = check_thresholds(metrics, args.rmse_threshold, args.mape_threshold)
        quality = "PASSED" if passed else "FAILED"

        if passed:
            log.info("Model PASSED evaluation.")
        else:
            log.warning("Model FAILED evaluation. Not recommended for production.")

        # 6. Reopen the training run and log everything directly onto it
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        client = mlflow.MlflowClient()

        with mlflow.start_run(run_id=run_id):
            # Log eval metrics
            mlflow.log_metrics(metrics)

            # Set quality tags on the run (for backward compatibility)
            mlflow.set_tag("passed_eval", str(passed))
            mlflow.set_tag("eval_status", quality)
            mlflow.set_tag("mlflow.note.content",
                           f"Evaluation: {quality} | RMSE={metrics['eval_rmse']:.4f} "
                           f"| MAPE={metrics['eval_mape']:.2f}% | R2={metrics['eval_r2']:.4f}")

            # Log eval plots as artifacts
            log_eval_plots(y_test, y_pred, ts_test)

        # Tag the model version directly in the registry
        client.set_model_version_tag(REGISTERED_MODEL_NAME, str(model_version),
                                     "passed_eval", str(passed))
        client.set_model_version_tag(REGISTERED_MODEL_NAME, str(model_version),
                                     "eval_status", quality)
        client.set_model_version_tag(REGISTERED_MODEL_NAME, str(model_version),
                                     "eval_rmse", f"{metrics['eval_rmse']:.4f}")
        client.set_model_version_tag(REGISTERED_MODEL_NAME, str(model_version),
                                     "eval_mape", f"{metrics['eval_mape']:.2f}")

        log.info(f"Logged eval metrics and plots to training run {run_id}")
        log.info(f"Tagged model version v{model_version} in registry: {quality}")

        # 8. Save eval metrics to disk (for DVC metrics tracking)
        eval_output = {
            "evaluated_model": run_name,
            "evaluated_run_id": run_id,
            "eval_rows": len(X_test),
            "passed": passed,
            **metrics,
        }
        with open(EVAL_METRICS_PATH, "w") as f:
            json.dump(eval_output, f, indent=2)
        log.info(f"Eval metrics saved: {EVAL_METRICS_PATH}")

    # 9. Summary
    log.info("")
    log.info("=" * 55)
    log.info("EVALUATION RESULTS (holdout test set — unseen by training)")
    log.info("-" * 55)
    log.info(f"  Model:     {run_name} ({run_id})")
    log.info(f"  Test rows: {len(X_test)}")
    log.info(f"  RMSE:      {metrics['eval_rmse']:.4f}")
    log.info(f"  MAE:       {metrics['eval_mae']:.4f}")
    log.info(f"  MAPE:      {metrics['eval_mape']:.2f}%")
    log.info(f"  R2:        {metrics['eval_r2']:.4f}")
    log.info(f"  Status:    {'PASSED' if passed else 'FAILED'}")
    log.info("=" * 55)

    # Exit with non-zero if failed (useful for CI/CD gating)
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
