"""
splits.py
=========
Single source of truth for the chronological train / validation / test split.

The pipeline needs three disjoint, time-ordered partitions:

    |<------------- train ------------->|<-- val -->|<-- test -->|
    oldest                                                   newest

  * train  - what the models are fit on.
  * val    - what train.py scores the 5 candidate models on to pick a champion.
             Because model *selection* looks at these rows, they are no longer
             an unbiased estimate of generalisation.
  * test   - never touched by train.py. evaluate.py is the only reader, so its
             metrics are an honest holdout measurement and the quality gate
             built on them is meaningful.

Both train.py and evaluate.py import this module rather than each computing
their own `iloc` boundary, so the partitions can never silently disagree.

Pipeline position:
  train.py -> (val metrics, champion) -> evaluate.py -> (test metrics, gate)
"""

from __future__ import annotations

import argparse
from typing import Tuple

# Defaults: 70% train / 15% validation / 15% test.
# On the March 2025 feature set (96 blocks/day x 31 days = 2976 rows) that is
# roughly 21 days train, 4.5 days validation, 4.5 days test — enough of each to
# cover several full daily seasonal cycles.
DEFAULT_VAL_SIZE = 0.15
DEFAULT_TEST_SIZE = 0.15


def add_split_args(parser: argparse.ArgumentParser, *, default_none: bool = False) -> None:
    """Register --val-size / --test-size on a parser.

    default_none=True leaves both as None so the caller can fall back to the
    split actually recorded by the last training run (see resolve_test_size).
    """
    val_default = None if default_none else DEFAULT_VAL_SIZE
    test_default = None if default_none else DEFAULT_TEST_SIZE
    parser.add_argument(
        "--val-size", type=float, default=val_default,
        help=f"Fraction held out for model selection (default: {DEFAULT_VAL_SIZE})",
    )
    parser.add_argument(
        "--test-size", type=float, default=test_default,
        help=f"Fraction held out as the final unseen test set (default: {DEFAULT_TEST_SIZE})",
    )


def split_bounds(n_rows: int, val_size: float, test_size: float) -> Tuple[int, int]:
    """Return (val_start, test_start) row indices for a time-ordered frame.

    train = [:val_start]   val = [val_start:test_start]   test = [test_start:]
    """
    for name, frac in (("val_size", val_size), ("test_size", test_size)):
        if not 0.0 < frac < 1.0:
            raise ValueError(f"{name} must be between 0 and 1, got {frac}")
    if val_size + test_size >= 1.0:
        raise ValueError(
            f"val_size + test_size must leave room for training data, "
            f"got {val_size} + {test_size} = {val_size + test_size}"
        )
    if n_rows < 3:
        raise ValueError(f"need at least 3 rows to make a 3-way split, got {n_rows}")

    val_start = int(n_rows * (1.0 - val_size - test_size))
    test_start = int(n_rows * (1.0 - test_size))

    # Guard the degenerate ends so no partition comes back empty on small frames.
    val_start = max(1, val_start)
    test_start = max(val_start + 1, test_start)
    test_start = min(test_start, n_rows - 1)
    if val_start >= test_start:
        raise ValueError(f"cannot form a non-empty 3-way split over {n_rows} rows")

    return val_start, test_start


def split_summary(n_rows: int, val_size: float, test_size: float) -> dict:
    """Serialisable description of the split, written into metrics.json."""
    val_start, test_start = split_bounds(n_rows, val_size, test_size)
    return {
        "val_size": val_size,
        "test_size": test_size,
        "n_rows": n_rows,
        "train_rows": val_start,
        "val_rows": test_start - val_start,
        "test_rows": n_rows - test_start,
    }


def resolve_split(cli_val_size: float | None, cli_test_size: float | None,
                  metrics_path) -> Tuple[float, float, str]:
    """Pick the fractions evaluate.py should use, and say where they came from.

    Precedence: explicit CLI flags > the split recorded by the training run that
    produced the model > module defaults. Reading them back from metrics.json is
    what keeps evaluate.py's holdout identical to the rows train.py withheld,
    even if the defaults here change after a model was trained.

    Both fractions are needed, not just test_size: split_bounds clamps the test
    boundary against the validation boundary, so reproducing train.py's exact
    cut requires reproducing both.
    """
    recorded = {}
    try:
        import json
        with open(metrics_path) as f:
            recorded = json.load(f).get("split", {}) or {}
    except (OSError, ValueError):
        pass

    if cli_val_size is not None or cli_test_size is not None:
        source = "CLI flags"
    elif recorded:
        source = f"split recorded in {getattr(metrics_path, 'name', metrics_path)}"
    else:
        source = "module defaults"

    val_size = cli_val_size if cli_val_size is not None else recorded.get("val_size", DEFAULT_VAL_SIZE)
    test_size = cli_test_size if cli_test_size is not None else recorded.get("test_size", DEFAULT_TEST_SIZE)
    return float(val_size), float(test_size), source
