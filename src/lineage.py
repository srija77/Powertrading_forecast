"""
Lightweight OpenLineage client for emitting data lineage events to Marquez.

If Marquez is unavailable or openlineage-python is not installed, all
calls silently no-op so the pipeline continues without interruption.

Usage:
    from src.lineage import lineage_run, ds

    with lineage_run("ingest_weather",
        inputs=[ds("iex-india", "DAM market data")],
        outputs=[ds("data/raw/dam"), ds("data/raw/rtm")],
    ):
        # pipeline code here
        pass
"""

import os
import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

log = logging.getLogger("lineage")

NAMESPACE = "gmr-energy-pipeline"
# Keep datasets in the SAME namespace as jobs so the Marquez UI shows jobs + datasets
# together (a split namespace makes one of them look empty per selected namespace).
DATASET_NAMESPACE = NAMESPACE

_HAS_OL = False
_client = None

try:
    from openlineage.client import OpenLineageClient
    from openlineage.client.run import (
        RunEvent,
        RunState,
        Run,
        Job,
        InputDataset,
        OutputDataset,
    )

    _HAS_OL = True
except ImportError:
    pass


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not _HAS_OL:
        return None
    # Allow fully turning lineage off without a running Marquez.
    if os.getenv("OPENLINEAGE_DISABLED", "").lower().strip() == "true":
        return None
    url = os.getenv("OPENLINEAGE_URL", "http://localhost:5002")
    try:
        from openlineage.client.transport.http import HttpConfig, HttpTransport

        # Fast-fail transport: short timeout and NO retries. The default client
        # retries 5x with backoff (~30s) when Marquez is down, which blocks the
        # whole pipeline. With these settings a missing Marquez fails instantly
        # and the emit is silently skipped, keeping lineage truly non-blocking.
        config = HttpConfig(
            url=url,
            endpoint="api/v1/lineage",
            timeout=2.0,
            retry={"total": 0, "connect": 0, "read": 0, "backoff_factor": 0},
        )
        _client = OpenLineageClient(transport=HttpTransport(config))
        return _client
    except Exception as e:
        log.debug(f"OpenLineage client init failed: {e}")
        return None


def ds(name, namespace=None):
    return {"namespace": namespace or DATASET_NAMESPACE, "name": name}


def _make_inputs(datasets):
    if not datasets or not _HAS_OL:
        return []
    return [
        InputDataset(namespace=d["namespace"], name=d["name"]) for d in datasets
    ]


def _make_outputs(datasets):
    if not datasets or not _HAS_OL:
        return []
    return [
        OutputDataset(namespace=d["namespace"], name=d["name"]) for d in datasets
    ]


def _emit(state_name, job_name, run_id, inputs=None, outputs=None):
    # state_name is a plain string, not a RunState member: RunState only exists when
    # openlineage-python imported successfully, so resolving it at the call sites
    # raised NameError before this no-op guard could be reached.
    client = _get_client()
    if not client:
        return
    try:
        event = RunEvent(
            eventType=getattr(RunState, state_name),
            eventTime=datetime.now(timezone.utc).isoformat(),
            run=Run(runId=str(run_id)),
            job=Job(namespace=NAMESPACE, name=job_name),
            inputs=_make_inputs(inputs),
            outputs=_make_outputs(outputs),
            producer="https://github.com/srija77/my-mlops-project",
        )
        client.emit(event)
        log.info(f"Lineage {state_name}: {job_name}")
    except Exception as e:
        log.debug(f"Lineage emit skipped: {e}")


def emit_start(job_name, inputs=None, outputs=None):
    run_id = uuid.uuid4()
    _emit("START", job_name, run_id, inputs, outputs)
    return run_id


def emit_complete(job_name, run_id, inputs=None, outputs=None):
    _emit("COMPLETE", job_name, run_id, inputs, outputs)


def emit_fail(job_name, run_id, inputs=None, outputs=None):
    _emit("FAIL", job_name, run_id, inputs, outputs)


@contextmanager
def lineage_run(job_name, inputs=None, outputs=None):
    run_id = emit_start(job_name, inputs, outputs)
    try:
        yield run_id
        emit_complete(job_name, run_id, inputs, outputs)
    except Exception:
        emit_fail(job_name, run_id, inputs, outputs)
        raise
