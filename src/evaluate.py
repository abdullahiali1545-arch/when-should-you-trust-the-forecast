"""src/evaluate.py — walk-forward evaluation, canary test, risk-coverage, bootstrap.

W2.3: canary test. W2.4: walk-forward harness.

Specification: docs/harness_design.md — §6 for the canary, §§2-4 for the
fold geometry (expanding window, purge at t + h <= a_k, holdout guard at
t + h < H).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from src.forecast import LAG0_COL, TARGET_COL

# ===========================================================================
# W2.3 — canary test
# ===========================================================================

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


# ===========================================================================
# W2.4 — walk-forward harness
# ===========================================================================

HORIZON = 6
FIRST_TEST = pd.Timestamp("2020-01-01 00:00", tz="UTC")
HOLDOUT_START = pd.Timestamp("2025-01-01 00:00", tz="UTC")


def build_target(pm25: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """y(t + horizon), aligned to origin t.

    Deliberately NOT in src/features.py: the target is information from after
    the origin, and keeping it out of the feature table is what makes the W1
    canary a real test rather than a test of whatever was excused from it.

    Reindexes to a complete hourly grid first. A positional shift on a frame
    with missing hours would pull a value from the wrong distance in time —
    silently, with no error.
    """
    if pm25.index.tz is None:
        raise ValueError("pm2_5 index must be tz-aware UTC")
    full = pd.date_range(pm25.index.min(), pm25.index.max(), freq="h", tz="UTC")
    s = pm25.reindex(full)
    return s.shift(-horizon).rename(TARGET_COL)


def make_folds(
    index: pd.DatetimeIndex,
    first_test: pd.Timestamp = FIRST_TEST,
    holdout_start: pd.Timestamp = HOLDOUT_START,
    horizon: int = HORIZON,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Expanding-window quarterly folds with purge and holdout guard.

    Boundaries are timestamps, never row positions: the index has gaps, so row
    counts and elapsed time do not correspond.
    """
    if index.tz is None:
        raise ValueError("index must be tz-aware UTC")
    index = index.sort_values()
    h = pd.Timedelta(hours=horizon)
    t0 = index.min()

    bounds = pd.date_range(first_test, holdout_start, freq="QS", tz="UTC")
    folds: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []

    for a_k, a_next in zip(bounds[:-1], bounds[1:]):
        # Purge: a training row is admissible only if its ANSWER had arrived by
        # the refit moment. Convention '<=' admits a target landing exactly on
        # a_k; this removes h - 1 = 5 rows per fold.
        train = index[(index >= t0) & (index + h <= a_k)]

        # Holdout guard, stated on TARGET time. An origin-based guard would
        # score the last 6 origins of 2024 against 2025 truth.
        test = index[
            (index >= a_k) & (index < a_next) & (index + h < holdout_start)
        ]

        if len(train) == 0 or len(test) == 0:
            continue

        assert train.max() + h <= a_k, "purge violated"
        assert test.min() >= a_k, "test row before fold start"
        assert test.max() + h < holdout_start, "holdout guard violated"
        assert train.max() < test.min(), "train/test overlap"

        folds.append((train, test))

    return folds


def scorable_mask(frame: pd.DataFrame, f2_cols: Sequence[str]) -> pd.Series:
    """Rows every model can be scored on.

    F2 cannot accept NaN and LightGBM can. Letting each model drop its own rows
    would score F2 on an easier subset and make the comparison meaningless, so
    one mask is computed per fold and applied to all four.
    """
    ok = frame[TARGET_COL].notna() & frame[LAG0_COL].notna()
    for col in f2_cols:
        ok &= frame[col].notna()
    return ok


def run_walk_forward(
    frame: pd.DataFrame,
    models: Sequence,
    f2_cols: Sequence[str],
    station: str,
    horizon: int = HORIZON,
    first_test: pd.Timestamp = FIRST_TEST,
    holdout_start: pd.Timestamp = HOLDOUT_START,
) -> pd.DataFrame:
    """Out-of-fold predictions for one station.

    `frame` is indexed by origin (UTC) and already contains the feature columns,
    LAG0_COL, the calendar columns and TARGET_COL.

    Returns: station | origin | target_time | fold | y_true | yhat_<name>...
    Every origin appears at most once — the folds do not overlap.
    """
    if frame.index.has_duplicates:
        raise ValueError("frame has duplicate origins")
    frame = frame.sort_index()
    h = pd.Timedelta(hours=horizon)

    folds = make_folds(frame.index, first_test, holdout_start, horizon)
    rows: list[pd.DataFrame] = []

    for k, (train_idx, test_idx) in enumerate(folds, start=1):
        train = frame.loc[train_idx]
        test = frame.loc[test_idx]

        # The mask is computed on both sides from the same rule. Training on
        # rows a model could not be scored on would train the four models on
        # different samples.
        train = train.loc[scorable_mask(train, f2_cols)]
        test_ok = scorable_mask(test, f2_cols)
        n_dropped = int((~test_ok).sum())
        test = test.loc[test_ok]

        if len(train) == 0 or len(test) == 0:
            continue

        out = pd.DataFrame(
            {
                "station": station,
                "origin": test.index,
                "target_time": test.index + h,
                "fold": k,
                "n_test_dropped": n_dropped,
                "y_true": test[TARGET_COL].to_numpy(dtype="float64"),
            }
        )

        for model in models:
            model.fit(train)                      # refit every fold, on T_k only
            out[f"yhat_{model.name}"] = model.predict(test)

        rows.append(out)

    result = pd.concat(rows, ignore_index=True)
    assert not result["origin"].duplicated().any(), "an origin was predicted twice"
    return result