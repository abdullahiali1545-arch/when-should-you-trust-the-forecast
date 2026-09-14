"""src/run_w2_5.py — score F0 (persistence) and F1 (climatology).

The first numbers in the project. F0 is the bar every later model must clear:
if F3 does not beat persistence in Mode B, that is the result and it gets
reported, not hidden.

Run from the repo root:   python src/run_w2_5.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import run_walk_forward
from src.forecast import (
    F0Persistence,
    F1Climatology,
    HOUR_COL,
    LAG0_COL,
    MONTH_COL,
    TARGET_COL,
)

FEATURES_PATH = Path("data/features/MY1.parquet")
STATION = "MY1"
OUT_PATH = Path("data/oof/MY1_w2_5.parquet")


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)

    # Index must be the origin, tz-aware UTC. A tz-naive index would make every
    # fold boundary comparison wrong by up to an hour for half the year.
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{path} is not indexed by timestamp — got {type(frame.index)}")
    if frame.index.tz is None:
        raise ValueError(f"{path} index is tz-naive; expected UTC")

    for col in (LAG0_COL, HOUR_COL, MONTH_COL, TARGET_COL):
        if col not in frame.columns:
            raise KeyError(
                f"{col!r} not in {path.name}. Columns present: {list(frame.columns)}. "
                "Fix the constants at the top of src/forecast.py."
            )

    return frame.sort_index()


def summarise(oof: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """MAE and RMSE per model, over all out-of-fold rows."""
    rows = []
    for name in models:
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


def main() -> None:
    frame = load_frame(FEATURES_PATH)

    models = [F0Persistence(), F1Climatology()]
    # F2 is not in this run, so no F2 columns constrain the mask yet. When F2
    # lands in W2.6 the mask tightens and these numbers will change — that is
    # expected, and it is why the common scoring set exists. Re-run F0/F1 then
    # and report the tightened numbers, not these.
    oof = run_walk_forward(frame, models, f2_cols=[], station=STATION)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(OUT_PATH)

    print(f"rows: {len(oof):,}   folds: {oof['fold'].nunique()}")
    print(
        "dropped from test folds: "
        f"{oof.groupby('fold')['n_test_dropped'].first().sum():,}"
    )
    print()
    print(summarise(oof, ["F0", "F1"]).to_string(index=False))
    print()
    per_fold = (
        oof.assign(
            ae0=(oof["y_true"] - oof["yhat_F0"]).abs(),
            ae1=(oof["y_true"] - oof["yhat_F1"]).abs(),
        )
        .groupby("fold")[["ae0", "ae1"]]
        .mean()
        .rename(columns={"ae0": "MAE_F0", "ae1": "MAE_F1"})
    )
    print(per_fold.round(3).to_string())
    print(f"\nwritten: {OUT_PATH}")


if __name__ == "__main__":
    main()