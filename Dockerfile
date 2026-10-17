FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY app/ ./app/
COPY src/ ./src/
COPY app_config/ ./app_config/
COPY models/ ./models/
COPY monitoring/ ./monitoring/

EXPOSE 8000

# Prometheus multiprocess mode so metrics aggregate across all gunicorn workers.
ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc

CMD ["gunicorn", "-c", "app/gunicorn.conf.py", "app.app:app"]
