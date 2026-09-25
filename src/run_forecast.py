"""src/run_forecast.py — score the forecasters through the walk-forward harness.

W2.5 added F0 and F1. W2.6 added F2, which tightened the common scoring mask:
F2 cannot accept NaN, so rows missing any F2 feature drop for every model.
W2.7 adds F3 (LightGBM), the model under scrutiny.

Run from the repo root:
    python -m src.run_forecast              # walk-forward, folds 1-20 (unchanged)
    python -m src.run_forecast --holdout    # adds 2025 as folds 21-24

F3 refits 20 times with 500 trees and deterministic=True, so expect minutes
rather than seconds. With --holdout it refits 24 times.

--holdout (PROJECT_SPEC changelog 2026-09-24):
  * writes to data/oof/<STATION>_oof_holdout.parquet, never over the committed
    walk-forward file;
  * first checks that folds 1-20 are bit-identical to the committed
    walk-forward file, and stops if they are not;
  * prints NO 2025 error numbers. Holdout results are read once, at the end,
    after routing and the bootstrap have run.

[W4a] Any of these also takes --station KC1 | BEX | HRL (default MY1).
    e.g. python -m src.run_forecast --station KC1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import run_walk_forward
from src.forecast import (
    F0Persistence,
    F1Climatology,
    F2Linear,
    F3LightGBM,
    F2_COLS,
    HOUR_COL,
    LAG0_COL,
    MONTH_COL,
    TARGET_COL,
    build_feature_cols,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
# [W4a] --station picks the station; default MY1 keeps every earlier command unchanged.
STATIONS = ("MY1", "KC1", "BEX", "HRL")


def _station_from_argv(default: str = "MY1") -> str:
    if "--station" not in sys.argv:
        return default
    i = sys.argv.index("--station")
    if i + 1 >= len(sys.argv):
        raise SystemExit("--station needs a value, e.g. --station KC1")
    s = sys.argv[i + 1].upper()
    if s not in STATIONS:
        raise SystemExit(f"unknown station {s!r}; choose from {STATIONS}")
    return s


STATION = _station_from_argv()
if "--holdout" in sys.argv and STATION != "MY1":
    raise SystemExit("the 2025 holdout is pre-registered for MY1 only (PROJECT_SPEC 2026-09-24)")
FEATURES_PATH = REPO_ROOT / "data/features" / f"{STATION}.parquet"
OUT_PATH = REPO_ROOT / "data/oof" / f"{STATION}_oof.parquet"
HOLDOUT_OUT_PATH = REPO_ROOT / "data/oof" / f"{STATION}_oof_holdout.parquet"   # [HOLDOUT]

N_WALK_FORWARD_FOLDS = 20                                                        # [HOLDOUT]
HOLDOUT_YEAR = 2025                                                              # [HOLDOUT]


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{path} is not indexed by timestamp — got {type(frame.index)}")
    if frame.index.tz is None:
        raise ValueError(f"{path} index is tz-naive; expected UTC")

    for col in (LAG0_COL, HOUR_COL, MONTH_COL, TARGET_COL, *F2_COLS):
        if col not in frame.columns:
            raise KeyError(
                f"{col!r} not in {path.name}. Columns present: {list(frame.columns)}"
            )

    return frame.sort_index()


def summarise(oof: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        err = oof["y_true"] - oof[f"yhat_{name}"]
        rows.append(
            {
                "model": name,
                "MAE": float(np.abs(err).mean()),
                "RMSE": float(np.sqrt((err ** 2).mean())),
                "n": int(err.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def per_fold_mae(oof: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    ae = pd.DataFrame({"fold": oof["fold"]})
    for name in names:
        ae[f"MAE_{name}"] = (oof["y_true"] - oof[f"yhat_{name}"]).abs()
    return ae.groupby("fold").mean()


# ---------------------------------------------------------------------------
# [HOLDOUT] Regression check and holdout sanity checks
# ---------------------------------------------------------------------------

def regression_check(oof: pd.DataFrame, committed_path: Path) -> None:
    """Folds 1-20 of the holdout run must equal the committed walk-forward file
    exactly — same rows, same order, same values, no tolerance.

    If they differ, something other than the fold count changed, and no 2025
    number can be trusted. Raises before anything about 2025 is written.
    """
    if not committed_path.exists():
        raise FileNotFoundError(
            f"{committed_path} not found. Run `python -m src.run_forecast` "
            "(without --holdout) first, so there is something to compare against."
        )
    committed = pd.read_parquet(committed_path).reset_index(drop=True)
    new = oof.loc[oof["fold"] <= N_WALK_FORWARD_FOLDS].reset_index(drop=True)

    if committed["fold"].max() != N_WALK_FORWARD_FOLDS:
        raise AssertionError(
            f"committed file has {committed['fold'].max()} folds, "
            f"expected {N_WALK_FORWARD_FOLDS}"
        )

    # check_exact: a leak or a changed setting shows up as a tiny difference,
    # which a tolerance would hide.
    pd.testing.assert_frame_equal(new, committed, check_exact=True)
    print(f"regression check PASSED: folds 1-{N_WALK_FORWARD_FOLDS} identical "
          f"to {committed_path.name} ({len(committed):,} rows)")


def holdout_sanity(oof: pd.DataFrame) -> None:
    """Structure only — counts and dates, never errors."""
    hold = oof.loc[oof["fold"] > N_WALK_FORWARD_FOLDS]
    folds = sorted(hold["fold"].unique())
    expected = list(range(N_WALK_FORWARD_FOLDS + 1, N_WALK_FORWARD_FOLDS + 5))
    if folds != expected:
        raise AssertionError(f"holdout folds are {folds}, expected {expected}")

    # Every holdout origin and target must be inside the holdout year.
    if not (hold["origin"].dt.year == HOLDOUT_YEAR).all():
        raise AssertionError("a holdout-fold origin falls outside 2025")
    if not (hold["target_time"].dt.year == HOLDOUT_YEAR).all():
        raise AssertionError("a holdout-fold target falls outside 2025")

    # And no walk-forward fold may have a target in 2025.
    wf = oof.loc[oof["fold"] <= N_WALK_FORWARD_FOLDS]
    if (wf["target_time"].dt.year >= HOLDOUT_YEAR).any():
        raise AssertionError("a walk-forward fold scores a 2025 target")

    counts = hold.groupby("fold").size()
    print("holdout folds (row counts only, no errors shown):")
    for k, n in counts.items():
        print(f"  fold {k}: {n:,} rows")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="add 2025 as folds 21-24 (run once; see PROJECT_SPEC 2026-09-24)",
    )
    parser.add_argument(
        "--station",
        default="MY1",
        help="station code: MY1 (default), KC1, BEX or HRL",       # [W4a]
    )
    args = parser.parse_args()
    print(f"station: {STATION}")

    frame = load_frame(FEATURES_PATH)

    # Never list(frame.columns): build_feature_cols strips y_t6 and imputed and
    # asserts the target is absent. That assert is the only thing standing
    # between this run and a near-zero MAE that would look like success.
    feature_cols = build_feature_cols(frame)
    print(f"F3 features: {len(feature_cols)}")

    models = [
        F0Persistence(),
        F1Climatology(),
        F2Linear(),
        F3LightGBM(feature_cols),
    ]
    names = [m.name for m in models]

    oof = run_walk_forward(
        frame, models, f2_cols=F2_COLS, station=STATION,
        include_holdout=args.holdout,                       # [HOLDOUT]
    )

    # [HOLDOUT] Separate branch: check first, save to a separate file, print no errors.
    if args.holdout:
        regression_check(oof, OUT_PATH)
        holdout_sanity(oof)
        HOLDOUT_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        oof.to_parquet(HOLDOUT_OUT_PATH)
        print(f"\nwritten: {HOLDOUT_OUT_PATH}")
        print("No 2025 error numbers printed. They are read once, after routing "
              "and the bootstrap.")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(OUT_PATH)

    dropped = int(oof.groupby("fold")["n_test_dropped"].first().sum())
    print(f"rows: {len(oof):,}   folds: {oof['fold'].nunique()}   dropped: {dropped:,}")
    print()
    print(summarise(oof, names).to_string(index=False))
    print()
    print(per_fold_mae(oof, names).round(3).to_string())
    print(f"\nwritten: {OUT_PATH}")


if __name__ == "__main__":
    main()