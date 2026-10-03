"""
refresh_online_store.py
=======================
On-demand command that rebuilds features from GX-validated data and refreshes the
Feast ONLINE store, so app.py / predict.py can serve fresh features by block_id.

Runs, in order:
  1. python src/3_feature_engineering/1_build_features.py   (validated -> prepared parquet)
  2. python src/3_feature_engineering/2_prepare_feast_data.py  (dedup + event_timestamp -> Feast source)
  3. feast apply                                  (register feature views + sync registry)
  4. feast materialize-incremental <now>          (offline -> online store)

Ingestion and validation are intentionally NOT run here — this refreshes features
from already-validated data. Run ingestion/validation separately for new dates.

Usage
-----
  python src/3_feature_engineering/3_refresh_online_store.py
  python src/3_feature_engineering/3_refresh_online_store.py --year 2025 --month 3
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEAST_REPO = PROJECT_ROOT / "my_feature_store" / "feature_repo"
PYTHON = sys.executable  # same interpreter that launched this script


def run(cmd, cwd=None, step=""):
    """Run a command, streaming output; abort the whole refresh on failure."""
    banner = f" STEP: {step} " if step else ""
    print("\n" + "=" * 70)
    print(banner.center(70, "="))
    print("=" * 70)
    print(f"$ {' '.join(str(c) for c in cmd)}   (cwd={cwd or PROJECT_ROOT})\n", flush=True)
    result = subprocess.run(cmd, cwd=str(cwd or PROJECT_ROOT))
    if result.returncode != 0:
        print(f"\n[FAILED] step '{step}' exited {result.returncode}. Aborting refresh.")
        sys.exit(result.returncode)


def main():
    p = argparse.ArgumentParser(prog="refresh_online_store.py")
    p.add_argument("--year", type=int, default=2025, help="Year for build_features (default 2025)")
    p.add_argument("--month", type=int, default=3, help="Month for build_features (default 3)")
    args = p.parse_args()

    print("Refreshing Feast online store from validated data")
    print(f"Project root: {PROJECT_ROOT}")

    # 1. Build the prepared feature table from validated data
    run([PYTHON, "src/3_feature_engineering/1_build_features.py", "--year", str(args.year), "--month", str(args.month)],
        step="build_features (validated -> prepared parquet)")

    # 2. Prepare the Feast source parquet (dedup by block_id, add event_timestamp)
    run([PYTHON, "src/3_feature_engineering/2_prepare_feast_data.py"],
        step="prepare_feast_data (-> Feast source parquet)")

    # 3. Register feature views + sync the registry
    run(["feast", "apply"], cwd=FEAST_REPO, step="feast apply")

    # 4. Materialize offline -> online store (up to now)
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    run(["feast", "materialize-incremental", now], cwd=FEAST_REPO,
        step=f"feast materialize-incremental {now}")

    print("\n" + "=" * 70)
    print(" ONLINE STORE REFRESH COMPLETE ".center(70, "="))
    print("=" * 70)
    print("Serve features by block_id, e.g.:")
    print("  python src/7_prediction/1_predict.py --online --block-id 2025-03-20T18:00")
    print('  curl -X POST localhost:8000/predict -H "Content-Type: application/json" '
          '-d \'{"block_id":"2025-03-20T18:00"}\'')


if __name__ == "__main__":
    main()
