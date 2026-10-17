"""
Gunicorn configuration — enables Prometheus multiprocess mode.

With multiple workers each process keeps its own in-memory metrics, so a scrape
sees only whichever worker answered (counts appear to jump around). In
multiprocess mode every worker writes to shared mmap files under
PROMETHEUS_MULTIPROC_DIR, and the /metrics endpoint aggregates them — so
counters/histograms are exact across all workers.

Launch with:  gunicorn -c app/gunicorn.conf.py app.app:app
"""
import os
import shutil

# Must be set BEFORE the app (and its metric objects) are imported. Gunicorn loads
# this config in the master before forking, and forked workers inherit the env, so
# setting it here guarantees prometheus_client starts in multiprocess mode.
MULTIPROC_DIR = os.environ.setdefault(
    "PROMETHEUS_MULTIPROC_DIR", "/tmp/prometheus_multiproc"
)

# ---- server settings (mirror the previous inline flags) ----
bind = "0.0.0.0:8000"
workers = int(os.environ.get("GUNICORN_WORKERS", "4"))
timeout = 120
accesslog = "-"


def on_starting(server):
    """Master startup: wipe stale metric files from any previous run."""
    shutil.rmtree(MULTIPROC_DIR, ignore_errors=True)
    os.makedirs(MULTIPROC_DIR, exist_ok=True)


def child_exit(server, worker):
    """When a worker dies, mark its files dead so its metrics are merged/retired
    correctly (otherwise a crashed worker's counters would double-count)."""
    from prometheus_client import multiprocess

    multiprocess.mark_process_dead(worker.pid)
