#!/bin/bash
set -e

# Use the MLflow server from docker-compose.yml (mlflow service)
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://mlflow:5000}"

echo "=== Waiting for MLflow server at $MLFLOW_TRACKING_URI ==="
for i in $(seq 1 60); do
    if curl -sf "$MLFLOW_TRACKING_URI/api/2.0/mlflow/experiments/search?max_results=1" > /dev/null 2>&1; then
        echo "MLflow server is ready."
        break
    fi
    echo "  waiting... ($i/60)"
    sleep 2
done

echo "=== Starting model training ==="
python src/4_training/1_train.py "$@"

echo "=== Training complete ==="
