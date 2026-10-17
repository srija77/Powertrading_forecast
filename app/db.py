"""Prediction logging to Postgres.

The app calls log_prediction() after every inference. Failures are
swallowed with a warning — the database must never break inference.
Inserts run in a background thread pool so they don't add latency to
the request path.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping

from sqlalchemy import (
    TIMESTAMP,
    BigInteger,
    Column,
    Float,
    Index,
    MetaData,
    Table,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pred-log")
_metadata = MetaData()

predictions_table = Table(
    "predictions",
    _metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column(
        "created_at",
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    ),
    Column("model_name", Text, nullable=False),
    Column("model_version", Text),
    Column("features", JSONB, nullable=False),
    Column("prediction", Float, nullable=False),
    Column("actual", Float),
    Column("latency_ms", Float),
    Column("request_id", Text),
    Index("idx_predictions_created_at", "created_at"),
    Index("idx_predictions_model", "model_name", "model_version"),
)


def init_db() -> bool:
    """Initialize engine and create the predictions table. Idempotent.

    Returns True if the DB is reachable and ready, False otherwise.
    Callers should treat a False return as "logging disabled" and keep
    serving requests.
    """
    global _engine
    url = os.environ.get("DATABASE_URL")
    if not url:
        logger.info("DATABASE_URL not set — prediction logging disabled")
        return False

    try:
        _engine = create_engine(
            url,
            pool_size=5,
            max_overflow=5,
            pool_pre_ping=True,
            future=True,
        )
        _metadata.create_all(_engine)
        logger.info("Prediction logging enabled (Postgres)")
        return True
    except Exception as exc:
        logger.warning("Failed to init prediction DB: %s", exc)
        _engine = None
        return False


def _insert(row: dict[str, Any]) -> None:
    if _engine is None:
        return
    try:
        with _engine.begin() as conn:
            conn.execute(predictions_table.insert().values(**row))
    except Exception as exc:
        logger.warning("Failed to log prediction: %s", exc)


def log_prediction(
    *,
    model_name: str,
    model_version: str | None,
    features: Mapping[str, Any],
    prediction: float,
    latency_ms: float | None = None,
    request_id: str | None = None,
) -> None:
    """Fire-and-forget: submits insert to a background thread."""
    if _engine is None:
        return
    row = {
        "model_name": model_name,
        "model_version": model_version,
        "features": json.loads(json.dumps(dict(features), default=float)),
        "prediction": float(prediction),
        "latency_ms": float(latency_ms) if latency_ms is not None else None,
        "request_id": request_id or str(uuid.uuid4()),
    }
    _executor.submit(_insert, row)


def log_predictions_batch(rows: list[dict[str, Any]]) -> None:
    """Bulk insert for /batch-predict. Fire-and-forget."""
    if _engine is None or not rows:
        return

    def _bulk():
        try:
            with _engine.begin() as conn:
                conn.execute(predictions_table.insert(), rows)
        except Exception as exc:
            logger.warning("Failed to log prediction batch (%d rows): %s", len(rows), exc)

    _executor.submit(_bulk)
