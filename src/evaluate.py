"""src/evaluate.py — walk-forward evaluation, canary test, risk-coverage, bootstrap.

W2.3: the canary section only. The harness itself lands in W2.4.

Specification: docs/harness_design.md §6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import pandas as pd

SENTINEL = 1e6  # loud enough to move a tree split; see docs/harness_design.md §6

# The prediction columns. y_true and any residual column are deliberately
# absent: poisoning y at t* legitimately changes the truth for origin t* - 6,
# whose origin is earlier than t* and therefore survives the row filter.
# Restricting the comparison to these columns is the only thing preventing a
# false failure there.
PRED_COLS: tuple[str, ...] = ("yhat_F0", "yhat_F1", "yhat_F2", "yhat_F3")


def poison_raw(
    raw: pd.DataFrame,
    t_star: pd.Timestamp,
    column: str = "pm2_5",
    sentinel: float = SENTINEL,
) -> pd.DataFrame:
    """Return a copy of `raw` with one value overwritten at t*.

    Copy, never mutate — the caller still needs the clean frame.
    """
    out = raw.copy(deep=True)
    if t_star not in out.index:
        raise KeyError(f"t* {t_star} is not in the raw index")
    if column not in out.columns:
        raise KeyError(f"column {column!r} is not in the raw frame")
    out.loc[t_star, column] = sentinel
    return out


@dataclass
class CanaryResult:
    placement: str
    t_star: pd.Timestamp
    rows_compared: int
    passed: bool
    first_divergence: pd.Timestamp | None


def canary_check(
    pipeline: Callable[[pd.DataFrame], pd.DataFrame],
    raw: pd.DataFrame,
    t_star: pd.Timestamp,
    placement: str,
    pred_cols: Sequence[str] = PRED_COLS,
) -> CanaryResult:
    """Assert that poisoning the raw series at t* cannot change any prediction
    made at an earlier origin.

    `pipeline` runs the FULL chain — raw series -> features -> walk-forward
    harness -> predictions — so this covers both src/features.py and the
    harness in one test. Poisoning the feature table instead would leave any
    forward-reaching bug inside features.py invisible.

    Returns a CanaryResult; never raises on a divergence. The caller decides
    whether a divergence is fatal, so a suite can report all three placements
    rather than stopping at the first.
    """
    pred_cols = list(pred_cols)

    # --- run both pipelines -------------------------------------------------
    clean = pipeline(raw)
    poisoned = pipeline(poison_raw(raw, t_star))

    for name, frame in (("clean", clean), ("poisoned", poisoned)):
        missing = {"origin", *pred_cols} - set(frame.columns)
        if missing:
            raise KeyError(f"{name} predictions missing columns: {sorted(missing)}")

    # --- keep only rows the canary is entitled to assert on ------------------
    # Strictly earlier than t*. Rows at or after t* may legitimately change:
    # the poisoned value is a legal input to their features.
    clean = clean.loc[clean["origin"] < t_star]
    poisoned = poisoned.loc[poisoned["origin"] < t_star]

    # --- align on origin, never on row order --------------------------------
    clean = clean.set_index("origin").sort_index()
    poisoned = poisoned.set_index("origin").sort_index()

    for name, frame in (("clean", clean), ("poisoned", poisoned)):
        if frame.index.has_duplicates:
            raise ValueError(
                f"{name} predictions contain duplicate origins — every origin "
                "must be predicted exactly once (see docs/harness_design.md §2)"
            )

    # Union, not intersection. An origin present in one frame and absent from
    # the other is itself a divergence; intersecting would silently hide it.
    index = clean.index.union(poisoned.index)
    a = clean.reindex(index)[pred_cols].astype("float64")
    b = poisoned.reindex(index)[pred_cols].astype("float64")

    # --- exact comparison, NaN equal to NaN ---------------------------------
    # No tolerance. A leak of any size is a leak, and np.isclose would hide the
    # small ones — which are exactly the ones a missing purge produces.
    differs = (a != b) & ~(a.isna() & b.isna())
    row_differs = differs.any(axis=1)

    first_divergence = row_differs.idxmax() if bool(row_differs.any()) else None

    return CanaryResult(
        placement=placement,
        t_star=t_star,
        rows_compared=len(index),
        passed=first_divergence is None,
        first_divergence=first_divergence,
    )