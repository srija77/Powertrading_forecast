import io
import json
import logging
import os
import pathlib
import pickle
import threading
import time
from collections import deque
from datetime import datetime, timezone

import mlflow
import numpy as np
import pandas as pd
import yaml
from flask import Flask, Response, jsonify, render_template, request
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = pathlib.Path("app_config/config.yaml")
FEATURE_STATS_PATH = pathlib.Path("drift_baselines/feature_stats.json")
DRIFT_REPORT_PATH = pathlib.Path("monitoring/drift_summary.json")
LOCAL_MODEL_PATH = pathlib.Path("models/model.pkl")

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_settings():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_local_model():
    """Load the bind-mounted training artifact when MLflow is unavailable."""
    if not LOCAL_MODEL_PATH.exists():
        return None, None

    with open(LOCAL_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    info = {
        "name": "local_pickle",
        "run_id": "-",
        "full_run_id": None,
        "version": "N/A",
        "stage": "local fallback",
        "rmse": 0,
        "mae": 0,
        "mape": 0,
        "loaded_at": datetime.now(timezone.utc).isoformat(),
    }
    return model, info


# ---------------------------------------------------------------------------
# Feature definitions (grouped for the UI)
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "Market Features": [
        "dam_purchase_bid",
        "dam_sell_bid",
        "dam_mcv",
        "dam_volume",
        "dam_bid_imbalance",
        "rtm_purchase_bid",
        "rtm_sell_bid",
        "rtm_mcv",
        "rtm_volume",
        "mcp_spread",
    ],
    "Lag & Rolling Features": [
        "dam_mcp_lag_1d",
        "dam_mcp_lag_2d",
        "dam_mcp_lag_7d",
        "dam_mcp_roll_4h",
        "dam_mcp_roll_24h",
        "dam_mcp_roll_std_24h",
    ],
    "Weather Features": [
        "avg_temp",
        "avg_humidity",
        "avg_windspeed",
        "avg_cloud_cover",
        "total_rainfall",
    ],
    "Calendar Features": [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "block_sin",
        "block_cos",
        "day_of_month",
        "is_weekend",
        "is_peak_hour",
        "is_morning_ramp",
        "is_off_peak",
        "is_ipl_match",
        "is_event",
        "is_festival",
        "is_wedding_season",
        "impact",
    ],
}

ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]

# Realistic defaults (median values from March 2025 training data) for demo
FEATURE_DEFAULTS = {
    "dam_purchase_bid": 13496.25,
    "dam_sell_bid": 11900.2,
    "dam_mcv": 7266.75,
    "dam_volume": 7266.75,
    "dam_bid_imbalance": 0.05,
    "rtm_purchase_bid": 6585.18,
    "rtm_sell_bid": 7379.28,
    "rtm_mcv": 4836.21,
    "rtm_volume": 4836.21,
    "mcp_spread": -197.95,
    "dam_mcp_lag_1d": 3768.54,
    "dam_mcp_lag_2d": 3783.5,
    "dam_mcp_lag_7d": 3768.42,
    "dam_mcp_roll_4h": 3855.35,
    "dam_mcp_roll_24h": 4179.01,
    "dam_mcp_roll_std_24h": 927.78,
    "avg_temp": 25.02,
    "avg_humidity": 45.37,
    "avg_windspeed": 14.66,
    "avg_cloud_cover": 18.85,
    "total_rainfall": 1.4,
    "hour_sin": 0.0,
    "hour_cos": 0.0,
    "dow_sin": 0.0,
    "dow_cos": -0.22,
    "block_sin": 0.0,
    "block_cos": 0.0,
    "day_of_month": 17.0,
    "is_weekend": 0,
    "is_peak_hour": 0,
    "is_morning_ramp": 0,
    "is_off_peak": 0,
    "is_ipl_match": 0,
    "is_event": 0,
    "is_festival": 0,
    "is_wedding_season": 0,
    "impact": 1.0,
}


# ---------------------------------------------------------------------------
# Feast online feature store (in-process, lazy, optional)
# ---------------------------------------------------------------------------

FEAST_REPO_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "my_feature_store" / "feature_repo"
)

# Feast FeatureService bundling the model's features (defined once in
# feature_definitions.py). The same service powers training, evaluation, and
# batch/online serving, so the online features can never drift from training.
FEATURE_SERVICE = "dam_mcp_forecast_v1"

_feature_store = None
_feature_store_lock = threading.Lock()


def get_feature_store():
    """Lazily initialise an in-process Feast FeatureStore.

    Returns None on any failure so the raw-feature /predict path keeps working
    even when Feast or the online store is unavailable.
    """
    global _feature_store
    if _feature_store is not None:
        return _feature_store
    with _feature_store_lock:
        if _feature_store is None:
            try:
                from feast import FeatureStore

                _feature_store = FeatureStore(repo_path=str(FEAST_REPO_PATH))
                logger.info("Feast online store initialised at %s", FEAST_REPO_PATH)
            except Exception as exc:
                logger.warning("Feast online store unavailable: %s", exc)
                return None
    return _feature_store


def fetch_online_features(block_id: str):
    """Fetch the 39 model features for a block_id from the Feast online store.

    Returns a single-row DataFrame in ALL_FEATURES order, or None if Feast is
    unavailable or the block_id has no materialised features.
    """
    store = get_feature_store()
    if store is None:
        return None
    rows = store.get_online_features(
        features=store.get_feature_service(FEATURE_SERVICE),
        entity_rows=[{"block_id": block_id}],
    ).to_df()
    if rows.empty:
        return None
    feat = rows[[c for c in ALL_FEATURES if c in rows.columns]]
    # All-null row => block_id not present / not materialised in the online store.
    if feat.isnull().all(axis=1).iloc[0]:
        return None
    # Reorder to the model's expected column order; cast ints to float (matches predict.py).
    feat = feat.reindex(columns=ALL_FEATURES)
    for col in feat.columns:
        if pd.api.types.is_integer_dtype(feat[col]):
            feat[col] = feat[col].astype(float)
    return feat


# ---------------------------------------------------------------------------
# Application state — lazy loaded, thread-safe, hot-reloadable
# ---------------------------------------------------------------------------


class AppState:
    """Holds model, metadata, and deployment info with thread-safe reload."""

    def __init__(self):
        self._lock = threading.Lock()
        self.model = None
        self.model_info = {
            "name": "loading",
            "run_id": "-",
            "rmse": 0,
            "mae": 0,
            "mape": 0,
        }
        self.ready = False
        self.start_time = datetime.now(timezone.utc)
        self.predictions = deque(maxlen=50)
        self.total_predictions = 0
        self.load_error = None

    @property
    def uptime_seconds(self):
        return int((datetime.now(timezone.utc) - self.start_time).total_seconds())

    def load_model(self):
        cfg = load_settings()
        mlflow_cfg = cfg["mlflow"]
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", mlflow_cfg["tracking_uri"])
        mlflow.set_tracking_uri(tracking_uri)
        registered_model = mlflow_cfg.get("registered_model_name", "dam_mcp_forecast")

        max_retries = 30
        for attempt in range(1, max_retries + 1):
            try:
                client = mlflow.MlflowClient()

                # Strategy 1: Load the champion (production) model via alias
                model, info = None, None
                try:
                    mv = client.get_model_version_by_alias(registered_model, "champion")
                    model = mlflow.sklearn.load_model(
                        f"models:/{registered_model}@champion"
                    )
                    run = client.get_run(mv.run_id)
                    info = {
                        "name": run.info.run_name or "unknown",
                        "run_id": mv.run_id[:8],
                        "full_run_id": mv.run_id,
                        "version": mv.version,
                        "stage": "champion",
                        "rmse": round(run.data.metrics.get("rmse", 0), 4),
                        "mae": round(run.data.metrics.get("mae", 0), 4),
                        "mape": round(run.data.metrics.get("mape", 0), 2),
                        "loaded_at": datetime.now(timezone.utc).isoformat(),
                    }
                    logger.info(
                        "Loaded champion model: %s v%s", registered_model, mv.version
                    )
                except Exception as exc:
                    logger.info(
                        "No champion model in registry, falling back to best run: %s",
                        exc,
                    )

                # Strategy 2: Use the newest registry version when training
                # has registered a model but evaluation/promotion has not yet
                # assigned the champion alias. This keeps the app in sync
                # with the registry rather than selecting an unrelated run.
                if model is None:
                    try:
                        versions = client.search_model_versions(
                            f"name='{registered_model}'"
                        )
                        if versions:
                            latest = max(versions, key=lambda version: int(version.version))
                            model = mlflow.sklearn.load_model(
                                f"models:/{registered_model}/{latest.version}"
                            )
                            run = client.get_run(latest.run_id)
                            info = {
                                "name": run.info.run_name or "unknown",
                                "run_id": latest.run_id[:8],
                                "full_run_id": latest.run_id,
                                "version": latest.version,
                                "stage": "latest registered",
                                "rmse": round(run.data.metrics.get("rmse", 0), 4),
                                "mae": round(run.data.metrics.get("mae", 0), 4),
                                "mape": round(run.data.metrics.get("mape", 0), 2),
                                "loaded_at": datetime.now(timezone.utc).isoformat(),
                            }
                            logger.info(
                                "Loaded latest registered model: %s v%s",
                                registered_model,
                                latest.version,
                            )
                    except Exception as exc:
                        logger.warning("Latest registered model could not be loaded: %s", exc)

                # Strategy 3: Fall back to the best run by RMSE.
                if model is None:
                    experiment = client.get_experiment_by_name(mlflow_cfg["experiment"])
                    if experiment:
                        runs = client.search_runs(
                            experiment_ids=[experiment.experiment_id],
                            order_by=["metrics.rmse ASC"],
                            max_results=1,
                        )
                        if runs:
                            best = runs[0]
                            try:
                                model = mlflow.sklearn.load_model(
                                    f"runs:/{best.info.run_id}/model"
                                )
                            except Exception:
                                logger.warning(
                                    "Best MLflow run exists but its model artifact could not be loaded"
                                )
                            info = {
                                "name": best.info.run_name or "unknown",
                                "run_id": best.info.run_id[:8],
                                "full_run_id": best.info.run_id,
                                "version": "N/A",
                                "stage": "None (best by RMSE)",
                                "rmse": round(best.data.metrics.get("rmse", 0), 4),
                                "mae": round(best.data.metrics.get("mae", 0), 4),
                                "mape": round(best.data.metrics.get("mape", 0), 2),
                                "loaded_at": datetime.now(timezone.utc).isoformat(),
                            }

                # Strategy 4: A locally trained model must remain serveable
                # when MLflow is empty, unavailable, or its artifacts are not
                # shared with this container.
                if model is None:
                    model, info = load_local_model()
                if model is not None and info["name"] == "local_pickle":
                    logger.info("Loaded model from local pickle: %s", LOCAL_MODEL_PATH)

                if model is not None:
                    with self._lock:
                        self.model = model
                        self.model_info = info
                        self.ready = True
                        self.load_error = None
                    logger.info(
                        "Loaded model: %s (RMSE=%.4f)", info["name"], info["rmse"]
                    )
                    return True
            except Exception as exc:
                logger.warning(
                    "Model load attempt %d/%d failed: %s", attempt, max_retries, exc
                )
                try:
                    model, info = load_local_model()
                    if model is not None:
                        with self._lock:
                            self.model = model
                            self.model_info = info
                            self.ready = True
                            self.load_error = None
                        logger.info(
                            "Loaded model from local pickle after MLflow failure: %s",
                            LOCAL_MODEL_PATH,
                        )
                        return True
                except Exception as local_exc:
                    logger.warning("Local model fallback failed: %s", local_exc)

            logger.info("Retrying in 10s... (%d/%d)", attempt, max_retries)
            time.sleep(10)

        self.load_error = "No model found after retries"
        logger.error(self.load_error)
        return False

    def load_all_runs(self):
        try:
            cfg = load_settings()
            mlflow_cfg = cfg["mlflow"]
            client = mlflow.MlflowClient()
            experiment = client.get_experiment_by_name(mlflow_cfg["experiment"])
            if not experiment:
                return []
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.rmse ASC"],
            )
            return [
                {
                    "name": r.info.run_name or "unknown",
                    "run_id": r.info.run_id[:8],
                    "rmse": round(r.data.metrics.get("rmse", 0), 4),
                    "mae": round(r.data.metrics.get("mae", 0), 4),
                    "mape": round(r.data.metrics.get("mape", 0), 2),
                    "status": r.info.status,
                }
                for r in runs
            ]
        except Exception as exc:
            logger.warning("Failed to load runs: %s", exc)
            return []

    def get_drift_summary(self):
        try:
            if DRIFT_REPORT_PATH.exists():
                with open(DRIFT_REPORT_PATH, "r") as f:
                    return json.load(f)
        except Exception as exc:
            logger.warning("Failed to read drift summary: %s", exc)
        return None


state = AppState()


# ---------------------------------------------------------------------------
# Background model loader — does not block app startup
# ---------------------------------------------------------------------------


def _background_load():
    state.load_model()


_loader_thread = threading.Thread(target=_background_load, daemon=True)
_loader_thread.start()

db.init_db()


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

REQUEST_COUNTER = Counter("inference_requests_total", "Total inference requests")
PREDICTION_VALUE = Histogram(
    "prediction_value",
    "Distribution of predicted dam_mcp values",
    buckets=[0, 500, 1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000],
)
LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
ERROR_COUNTER = Counter("inference_errors_total", "Total inference errors")
# multiprocess_mode="mostrecent" so the aggregated /metrics reports the latest
# value across workers (not one series per PID). Ignored in single-process mode.
MODEL_LOADED = Gauge(
    "model_loaded", "Whether a model is loaded (1=yes, 0=no)",
    multiprocess_mode="mostrecent",
)
DRIFT_DETECTED = Gauge(
    "drift_detected", "Whether data drift is detected (1=yes, 0=no)",
    multiprocess_mode="mostrecent",
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/", methods=["GET"])
def index():
    runs = state.load_all_runs()
    drift = state.get_drift_summary()
    return render_template(
        "index.html",
        feature_groups=FEATURE_GROUPS,
        feature_defaults=FEATURE_DEFAULTS,
        model_info=state.model_info,
        runs=runs,
        predictions=list(state.predictions),
        drift_summary=drift,
        ready=state.ready,
        uptime=state.uptime_seconds,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if not state.ready:
        return (
            jsonify({"error": "Model not loaded yet. Check /health for status."}),
            503,
        )

    REQUEST_COUNTER.inc()
    start = time.time()
    with LATENCY.time():
        payload = request.get_json(force=False, silent=True) or request.form.to_dict(
            flat=True
        )

        # Online path: fetch features from the Feast online store by block_id.
        block_id = payload.get("block_id")
        if block_id and not any(f in payload for f in ALL_FEATURES):
            df = fetch_online_features(block_id)
            if df is None:
                ERROR_COUNTER.inc()
                return (
                    jsonify(
                        {
                            "error": (
                                f"No online features for block_id '{block_id}'. "
                                "Materialise the online store "
                                "(python src/3_feature_engineering/3_refresh_online_store.py) or check the id."
                            )
                        }
                    ),
                    404,
                )
            values = df.iloc[0].tolist()
        else:
            # Raw path: caller supplies all feature values (back-compat).
            try:
                values = [float(payload.get(f, 0)) for f in ALL_FEATURES]
            except Exception as exc:
                ERROR_COUNTER.inc()
                return jsonify({"error": f"Invalid input: {exc}"}), 400
            df = pd.DataFrame([values], columns=ALL_FEATURES)

        preds = state.model.predict(df)
        prediction = round(float(np.ravel(preds)[0]), 2)
        latency_ms = round((time.time() - start) * 1000, 1)
        PREDICTION_VALUE.observe(prediction)

        state.total_predictions += 1
        state.predictions.appendleft(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "prediction": prediction,
                "latency_ms": latency_ms,
            }
        )

        db.log_prediction(
            model_name=state.model_info.get("name", "unknown"),
            model_version=str(state.model_info.get("version", "")),
            features=dict(zip(ALL_FEATURES, values)),
            prediction=prediction,
            latency_ms=latency_ms,
        )

        response = {
            "prediction": prediction,
            "model": state.model_info["name"],
            "latency_ms": latency_ms,
        }
        if block_id:
            response["block_id"] = block_id
            response["source"] = "feast_online_store"
        return jsonify(response)


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    """Upload a CSV file, predict all rows, return results with metrics."""
    if not state.ready:
        return (
            jsonify({"error": "Model not loaded yet. Check /health for status."}),
            503,
        )

    f = request.files.get("file")
    if not f or f.filename == "":
        return jsonify({"error": "No CSV file uploaded"}), 400

    start = time.time()
    try:
        df = pd.read_csv(io.StringIO(f.read().decode("utf-8")))
    except Exception as exc:
        return jsonify({"error": f"Failed to parse CSV: {exc}"}), 400

    # Check for target column (optional — used for accuracy metrics)
    cfg = load_settings()
    target_col = cfg["data"].get("target_col", "dam_mcp")
    has_actuals = target_col in df.columns

    # Identify which feature columns are present
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}"}), 400

    # Run predictions
    X = df[ALL_FEATURES].fillna(0)
    preds = state.model.predict(X)
    predictions = np.ravel(preds).tolist()
    df["predicted_dam_mcp"] = [round(p, 2) for p in predictions]

    latency_ms = round((time.time() - start) * 1000, 1)
    REQUEST_COUNTER.inc(len(predictions))
    state.total_predictions += len(predictions)

    model_name = state.model_info.get("name", "unknown")
    model_version = str(state.model_info.get("version", ""))
    per_row_latency = latency_ms / max(len(predictions), 1)
    log_rows = []
    for i, row in X.iterrows():
        log_rows.append(
            {
                "model_name": model_name,
                "model_version": model_version,
                "features": json.loads(row.to_json()),
                "prediction": (
                    float(predictions[i])
                    if i < len(predictions)
                    else float(predictions[-1])
                ),
                "actual": float(df[target_col].iloc[i]) if has_actuals else None,
                "latency_ms": per_row_latency,
                "request_id": None,
            }
        )
    db.log_predictions_batch(log_rows)

    # Compute metrics if actuals are available
    metrics_result = None
    if has_actuals:
        actuals = df[target_col].values
        rmse = round(float(np.sqrt(mean_squared_error(actuals, predictions))), 4)
        mae = round(float(mean_absolute_error(actuals, predictions)), 4)
        non_zero = actuals != 0
        mape = (
            round(
                float(
                    np.mean(
                        np.abs(
                            (actuals[non_zero] - np.array(predictions)[non_zero])
                            / actuals[non_zero]
                        )
                    )
                    * 100
                ),
                2,
            )
            if non_zero.any()
            else 0
        )
        metrics_result = {"rmse": rmse, "mae": mae, "mape": mape}

    # Build response rows (first 500 for UI, full set in download)
    rows = []
    for i, row in df.head(500).iterrows():
        entry = {"row": i}
        if has_actuals:
            entry["actual"] = round(float(row[target_col]), 2)
        entry["predicted"] = round(float(row["predicted_dam_mcp"]), 2)
        if has_actuals:
            entry["error"] = round(float(row["predicted_dam_mcp"] - row[target_col]), 2)
        rows.append(entry)

    # Run drift detection in background
    drift_summary = None
    try:
        from src.drift import run_drift_check

        drift_summary = run_drift_check(df)
        DRIFT_DETECTED.set(1 if drift_summary.get("dataset_drift", False) else 0)
        logger.info(
            "Drift check: %s (%d/%d features drifted)",
            "DETECTED" if drift_summary["dataset_drift"] else "none",
            drift_summary["drifted_features_count"],
            drift_summary["total_features"],
        )
    except Exception as exc:
        logger.warning("Drift check failed: %s", exc)

    # Generate downloadable CSV
    output_csv = df.to_csv(index=False)

    return jsonify(
        {
            "total_rows": len(df),
            "model": state.model_info["name"],
            "latency_ms": latency_ms,
            "metrics": metrics_result,
            "has_actuals": has_actuals,
            "rows": rows,
            "csv_download": output_csv,
            "drift": drift_summary,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    """Liveness probe — always returns 200 so the container is not killed."""
    return jsonify(
        {
            "status": "healthy",
            "model_ready": state.ready,
            "model": state.model_info["name"],
            "rmse": state.model_info["rmse"],
            "uptime_seconds": state.uptime_seconds,
            "total_predictions": state.total_predictions,
            "error": state.load_error,
            "demo_version": "v3",
            "pod": os.environ.get("HOSTNAME", "unknown"),
        }
    )


@app.route("/ready", methods=["GET"])
def ready():
    """Readiness probe — returns 503 until model is loaded."""
    if state.ready:
        return jsonify({"ready": True}), 200
    return jsonify({"ready": False, "error": state.load_error or "loading"}), 503


@app.route("/reload", methods=["POST"])
def reload_model():
    """Hot-reload the model from MLflow without restarting the container."""
    old_name = state.model_info.get("name", "none")
    thread = threading.Thread(target=state.load_model, daemon=True)
    thread.start()
    thread.join(timeout=60)
    if state.ready:
        return jsonify(
            {
                "status": "reloaded",
                "previous_model": old_name,
                "current_model": state.model_info["name"],
                "loaded_at": state.model_info.get("loaded_at"),
            }
        )
    return jsonify({"status": "failed", "error": state.load_error}), 500


@app.route("/drift", methods=["GET"])
def drift():
    """Return the latest drift detection summary as JSON."""
    summary = state.get_drift_summary()
    if summary:
        is_drifted = summary.get("dataset_drift", False)
        DRIFT_DETECTED.set(1 if is_drifted else 0)
        return jsonify(summary)
    return (
        jsonify(
            {"error": "No drift report found. Run: python -m monitoring.drift_report"}
        ),
        404,
    )


@app.route("/deployment", methods=["GET"])
def deployment():
    """Show deployment metadata for debugging and monitoring."""
    drift = state.get_drift_summary()
    return jsonify(
        {
            "model": state.model_info,
            "ready": state.ready,
            "uptime_seconds": state.uptime_seconds,
            "total_predictions": state.total_predictions,
            "recent_predictions": len(state.predictions),
            "drift": {
                "detected": drift.get("dataset_drift", False) if drift else None,
                "features_drifted": (
                    drift.get("drifted_features_count", 0) if drift else 0
                ),
                "report_time": drift.get("report_time") if drift else None,
            },
            "config": {
                "mlflow_uri": os.environ.get(
                    "MLFLOW_TRACKING_URI", "http://localhost:5000"
                ),
                "features_count": len(ALL_FEATURES),
            },
        }
    )


@app.route("/metrics", methods=["GET"])
def metrics():
    MODEL_LOADED.set(1 if state.ready else 0)
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        # Running under gunicorn with multiple workers: aggregate every worker's
        # metrics from the shared multiprocess dir so counters/histograms are exact.
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        data = generate_latest(registry)
    else:
        # Single-process (Flask dev server): the default registry has everything.
        data = generate_latest()
    return Response(data, mimetype="text/plain; version=0.0.4; charset=utf-8")


if __name__ == "__main__":
    cfg = load_settings()
    app.run(host=cfg["serving"]["host"], port=cfg["serving"]["port"], debug=False)
