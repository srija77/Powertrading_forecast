"""
promote_model.py
================
Compares the latest registered model version against the current production
model in MLflow. If the latest version has better eval scores AND its
``passed_eval`` tag is "True", it is promoted to the "champion" alias
(production). Otherwise, the current production model stays.

Pipeline position:
  train.py -> evaluate.py -> promote_model.py
  (DVC stage: promote, runs after evaluate)

Usage:
    python src/6_promotion/1_promote_model.py
    python src/6_promotion/1_promote_model.py --metric eval_rmse --direction lower
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

# Fix Windows cp1252 encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# -- Config ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EVAL_METRICS_PATH = PROJECT_ROOT / "eval_metrics.json"
PROMOTE_REPORT_PATH = PROJECT_ROOT / "promote_report.json"
MODEL_VERSION_PATH = PROJECT_ROOT / "app_config" / "model_version.json"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
REGISTERED_MODEL_NAME = "dam_mcp_forecast"
PRODUCTION_ALIAS = "champion"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PROMOTE] %(message)s")
log = logging.getLogger("promote")


# -- CLI ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote best model to production")
    parser.add_argument(
        "--metric", type=str, default="eval_rmse",
        help="Primary metric to compare (default: eval_rmse)",
    )
    parser.add_argument(
        "--direction", type=str, default="lower", choices=["lower", "higher"],
        help="Whether lower or higher metric is better (default: lower)",
    )
    return parser.parse_args()


# -- Helpers -----------------------------------------------------------------
def get_latest_version(client: MlflowClient) -> mlflow.entities.model_registry.ModelVersion | None:
    """Return the latest model version from the registry."""
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version))


def get_production_version(client: MlflowClient) -> mlflow.entities.model_registry.ModelVersion | None:
    """Return the current production model version (by alias)."""
    try:
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
        return mv
    except mlflow.exceptions.MlflowException:
        return None


def get_run_eval_metrics(client: MlflowClient, run_id: str) -> dict:
    """Fetch eval metrics from a run."""
    run = client.get_run(run_id)
    return {
        "eval_rmse": run.data.metrics.get("eval_rmse"),
        "eval_mae": run.data.metrics.get("eval_mae"),
        "eval_mape": run.data.metrics.get("eval_mape"),
        "eval_r2": run.data.metrics.get("eval_r2"),
    }


def version_has_passed_eval(mv) -> bool:
    """Check if the model version's passed_eval tag is True."""
    tag_val = mv.tags.get("passed_eval", "").strip()
    return tag_val == "True"


def is_better(candidate_val: float, production_val: float, direction: str) -> bool:
    """Return True if candidate metric is better than production."""
    if direction == "lower":
        return candidate_val < production_val
    return candidate_val > production_val


def promote_version(client: MlflowClient, version: str, old_production=None):
    """Set champion alias AND transition the legacy stage to Production.

    MLflow 2.x has two parallel systems — aliases (new) and stages (old).
    The UI shows the stage field, so we update both to keep them in sync.
    Also writes app_config/model_version.json so Git tracks the active model —
    a commit to this file triggers the CD pipeline (build → deploy → ArgoCD).
    """
    # Set the new alias
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, version,
    )
    # Sync the legacy stage so the UI doesn't show "Archived" / "None"
    client.transition_model_version_stage(
        REGISTERED_MODEL_NAME, version, stage="Production",
        archive_existing_versions=True,
    )
    log.info(f"Set alias '{PRODUCTION_ALIAS}' + legacy stage 'Production' on v{version}")

    # Write version file — triggers CD when committed
    run = client.get_run(
        client.get_model_version(REGISTERED_MODEL_NAME, version).run_id
    )
    version_info = {
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": version,
        "alias": PRODUCTION_ALIAS,
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "eval_rmse": run.data.metrics.get("eval_rmse"),
        "eval_mae": run.data.metrics.get("eval_mae"),
        "eval_mape": run.data.metrics.get("eval_mape"),
        "promoted_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    MODEL_VERSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_VERSION_PATH, "w") as f:
        json.dump(version_info, f, indent=2)
    log.info(f"Model version file written: {MODEL_VERSION_PATH}")


# -- Main --------------------------------------------------------------------
def main():
    from src.lineage import lineage_run, ds

    args = parse_args()

    log.info("=" * 55)
    log.info("PROMOTE - Compare & push best model to production")
    log.info("=" * 55)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    with lineage_run("promote",
        inputs=[ds("eval_metrics.json")],
        outputs=[ds("promote_report.json"), ds("mlflow/model-registry/champion")],
    ):
        # 1. Get latest version
        latest = get_latest_version(client)
        if latest is None:
            log.error(f"No model versions found for '{REGISTERED_MODEL_NAME}'. Run train.py first.")
            sys.exit(1)

        log.info(f"Latest model version: v{latest.version} (run_id={latest.run_id})")

        # 2. Check passed_eval tag
        if not version_has_passed_eval(latest):
            log.warning(
                f"Latest version v{latest.version} does NOT have passed_eval=True. "
                "Skipping promotion."
            )
            _save_report(
                action="skipped",
                reason="latest version failed evaluation (passed_eval != True)",
                latest_version=latest.version,
                latest_run_id=latest.run_id,
            )
            sys.exit(1)

        log.info(f"Latest version v{latest.version} has passed_eval=True")

        # 3. Get current production version
        production = get_production_version(client)

        if production is None:
            # No production model yet — promote directly
            log.info("No current production model. Promoting latest version directly.")
            promote_version(client, latest.version)
            latest_metrics = get_run_eval_metrics(client, latest.run_id)
            log.info(f"Promoted v{latest.version} to '{PRODUCTION_ALIAS}' alias (first production model)")
            _save_report(
                action="promoted",
                reason="no existing production model",
                latest_version=latest.version,
                latest_run_id=latest.run_id,
                latest_metrics=latest_metrics,
            )
            _log_summary(latest.version, latest_metrics, None, None)
            return

        if str(production.version) == str(latest.version):
            log.info(f"Latest version v{latest.version} is already in production. Nothing to do.")
            latest_metrics = get_run_eval_metrics(client, latest.run_id)
            _save_report(
                action="no_change",
                reason="latest version is already production",
                latest_version=latest.version,
                latest_run_id=latest.run_id,
                latest_metrics=latest_metrics,
                production_version=production.version,
            )
            return

        log.info(f"Current production: v{production.version} (run_id={production.run_id})")

        # 4. Compare metrics
        latest_metrics = get_run_eval_metrics(client, latest.run_id)
        prod_metrics = get_run_eval_metrics(client, production.run_id)

        metric_key = args.metric
        latest_val = latest_metrics.get(metric_key)
        prod_val = prod_metrics.get(metric_key)

        if latest_val is None or prod_val is None:
            log.error(
                f"Cannot compare: {metric_key} is missing. "
                f"Latest={latest_val}, Production={prod_val}"
            )
            _save_report(
                action="skipped",
                reason=f"metric '{metric_key}' missing from one or both versions",
                latest_version=latest.version,
                latest_run_id=latest.run_id,
                latest_metrics=latest_metrics,
                production_version=production.version,
                production_run_id=production.run_id,
                production_metrics=prod_metrics,
            )
            sys.exit(1)

        log.info(f"Comparing {metric_key} ({args.direction} is better):")
        log.info(f"  Latest     v{latest.version}: {latest_val:.4f}")
        log.info(f"  Production v{production.version}: {prod_val:.4f}")

        if is_better(latest_val, prod_val, args.direction):
            # Promote latest
            promote_version(client, latest.version, old_production=production)
            log.info(
                f"PROMOTED: v{latest.version} -> '{PRODUCTION_ALIAS}' "
                f"({metric_key}: {latest_val:.4f} beats {prod_val:.4f})"
            )
            _save_report(
                action="promoted",
                reason=f"{metric_key} improved: {latest_val:.4f} vs {prod_val:.4f}",
                latest_version=latest.version,
                latest_run_id=latest.run_id,
                latest_metrics=latest_metrics,
                production_version=production.version,
                production_run_id=production.run_id,
                production_metrics=prod_metrics,
            )
        else:
            log.info(
                f"NOT PROMOTED: v{latest.version} ({metric_key}={latest_val:.4f}) is not better "
                f"than production v{production.version} ({metric_key}={prod_val:.4f})"
            )
            _save_report(
                action="not_promoted",
                reason=f"{metric_key} not better: {latest_val:.4f} vs {prod_val:.4f}",
                latest_version=latest.version,
                latest_run_id=latest.run_id,
                latest_metrics=latest_metrics,
                production_version=production.version,
                production_run_id=production.run_id,
                production_metrics=prod_metrics,
            )

        _log_summary(latest.version, latest_metrics, production.version, prod_metrics)


def _save_report(**fields):
    """Save promotion decision to disk for DVC tracking."""
    with open(PROMOTE_REPORT_PATH, "w") as f:
        json.dump(fields, f, indent=2)
    log.info(f"Promotion report saved: {PROMOTE_REPORT_PATH}")


def _log_summary(latest_ver, latest_m, prod_ver, prod_m):
    log.info("")
    log.info("=" * 55)
    log.info("PROMOTION SUMMARY")
    log.info("-" * 55)
    log.info(f"  Latest v{latest_ver}:")
    if latest_m:
        for k, v in latest_m.items():
            log.info(f"    {k}: {v:.4f}" if v is not None else f"    {k}: N/A")
    if prod_ver and prod_m:
        log.info(f"  Production v{prod_ver}:")
        for k, v in prod_m.items():
            log.info(f"    {k}: {v:.4f}" if v is not None else f"    {k}: N/A")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
