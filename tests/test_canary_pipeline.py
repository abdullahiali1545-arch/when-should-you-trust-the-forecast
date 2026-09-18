"""tests/test_canary_pipeline.py — the canary on the REAL pipeline.

tests/test_canary.py proves canary_check works, using toy pipelines. features.py's
own canary proves build_features doesn't reach forward. Neither touches the
walk-forward harness — the fold boundaries, the purge, the per-fold refits.

This does. It poisons ONE raw observation at t* and asserts that no prediction
made at an earlier origin changes by a single bit.

    raw parquet -> build_features -> run_walk_forward -> OOF table
                         ^ poison enters here          ^ compared here

Three placements, each exercising a different part of the harness:
    mid-train     deep inside a training window   -> feature-level forward reach
    pre-boundary  hours before a fold's test start -> a missing or wrong-sided purge
    mid-test      inside a test fold               -> target/scoring contamination

Run from the repo root:   python -m tests.test_canary_pipeline

Expect ~25-30 minutes: the pipeline runs twice per placement, plus twice more
for the determinism pre-check. Start it and go and do something else.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import (  # noqa: E402
    FIRST_TEST,
    HOLDOUT_START,
    HORIZON,
    canary_check,
    make_folds,
    run_walk_forward,
)
from src.features import build_features  # noqa: E402
from src.forecast import (  # noqa: E402
    F0Persistence,
    F1Climatology,
    F2Linear,
    F3LightGBM,
    F2_COLS,
    build_feature_cols,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STATION = "MY1"
RAW_PATH = REPO_ROOT / "data/processed" / f"{STATION}.parquet"


# ---------------------------------------------------------------------------
# The adapter: wrap the real pipeline into the shape canary_check wants,
# i.e. Callable[[pd.DataFrame], pd.DataFrame] mapping raw -> predictions.
# ---------------------------------------------------------------------------

def pipeline(raw: pd.DataFrame) -> pd.DataFrame:
    """Full chain, exactly as src/run_forecast.py runs it — except that features
    are rebuilt from raw rather than read from the features parquet. Reading the
    parquet would place the poison downstream of build_features and leave any
    forward-reaching bug inside that module invisible.
    """
    frame = build_features(raw)

    # build_feature_cols strips y_t6 and imputed, and asserts the target is
    # absent. Computed from THIS frame, so clean and poisoned runs cannot
    # silently disagree on their column set.
    feature_cols = build_feature_cols(frame)

    # Constructed fresh on every call. F2Linear and F3LightGBM carry fitted
    # state; reusing instances across the clean and poisoned runs would make
    # them differ for reasons unrelated to the poison.
    models = [
        F0Persistence(),
        F1Climatology(),
        F2Linear(),
        F3LightGBM(feature_cols),
    ]

    return run_walk_forward(
        frame, models, f2_cols=F2_COLS, station=STATION,
        horizon=HORIZON, first_test=FIRST_TEST, holdout_start=HOLDOUT_START,
    )


# ---------------------------------------------------------------------------
# Placement selection
# ---------------------------------------------------------------------------

def _snap(ts: pd.Timestamp, raw: pd.DataFrame) -> pd.Timestamp:
    """Nearest raw timestamp at or before `ts` carrying an observed pm2_5.

    build_features grids the index, so a timestamp that exists in the feature
    table may not exist in raw — and poison_raw raises on a missing index.
    Poisoning a NaN is also useless: nothing downstream would change and the
    canary would pass while testing nothing.
    """
    observed = raw.index[raw["pm2_5"].notna() & (raw.index <= ts)]
    if len(observed) == 0:
        raise ValueError(f"no observed pm2_5 at or before {ts}")
    return observed[-1]


def choose_placements(raw: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Three t* values derived from the real fold geometry, not hardcoded."""
    frame = build_features(raw)
    folds = make_folds(frame.index, FIRST_TEST, HOLDOUT_START, HORIZON)
    if len(folds) < 3:
        raise RuntimeError(f"expected many folds, got {len(folds)}")

    # A fold in the middle of the sequence: it has a long training window behind
    # it and further folds after it, so all three placements have earlier origins
    # to compare against.
    train_idx, test_idx = folds[len(folds) // 2]

    return {
        "mid-train": _snap(train_idx[len(train_idx) // 2], raw),
        "pre-boundary": _snap(test_idx.min() - pd.Timedelta(hours=3), raw),
        "mid-test": _snap(test_idx[len(test_idx) // 2], raw),
    }


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Canary on the real pipeline.")
    ap.add_argument("--raw", type=Path, default=RAW_PATH)
    ap.add_argument("--skip-determinism", action="store_true",
                    help="skip the two-run reproducibility pre-check")
    args = ap.parse_args()

    if not args.raw.exists():
        raise SystemExit(f"raw parquet not found: {args.raw}\nPass --raw <path>.")

    raw = pd.read_parquet(args.raw).sort_index()
    print(f"{STATION}: {len(raw):,} raw rows, {raw.index.min()} -> {raw.index.max()}")

    # --- determinism pre-check ---------------------------------------------
    # Without this the whole test is void: if two identical runs disagree by
    # 1e-9, every "divergence" below is floating-point noise, not leakage.
    if not args.skip_determinism:
        print("\ndeterminism pre-check: running the pipeline twice, unpoisoned...")
        t0 = time.time()
        a = pipeline(raw).set_index("origin").sort_index()
        b = pipeline(raw).set_index("origin").sort_index()
        if not a.equals(b):
            raise SystemExit(
                "FAIL: two identical runs differ. Seed the models and set n_jobs=1 "
                "before trusting any canary result."
            )
        print(f"  identical ({time.time() - t0:.0f}s for two runs)")

    # --- three placements ---------------------------------------------------
    placements = choose_placements(raw)
    print("\nplacements:")
    for name, ts in placements.items():
        print(f"  {name:<13} {ts}")

    results = []
    for name, t_star in placements.items():
        print(f"\nrunning {name} at {t_star} ...")
        t0 = time.time()
        res = canary_check(pipeline, raw, t_star, placement=name)
        print(f"  {res}  ({time.time() - t0:.0f}s)")
        results.append(res)

    # --- verdict ------------------------------------------------------------
    print("\n" + "=" * 70)
    failed = [r for r in results if not r.passed]
    for r in results:
        mark = "PASS" if r.passed else f"FAIL at {r.first_divergence}"
        print(f"{r.placement:<13} rows={r.rows_compared:>7,}  {mark}")

    if failed:
        raise SystemExit(
            "\nLEAK: a prediction made before t* changed when t* was poisoned. "
            "Information moved backwards in time. Stop and fix before W3."
        )
    print("\nAll placements clean. No origin earlier than t* was affected.")


if __name__ == "__main__":
    main()