"""src/run_forecast.py — score the forecasters through the walk-forward harness.

W2.5 added F0 and F1. W2.6 added F2, which tightened the common scoring mask:
F2 cannot accept NaN, so rows missing any F2 feature drop for every model.
W2.7 adds F3 (LightGBM), the model under scrutiny.

Run from the repo root:   python -m src.run_forecast

F3 refits 20 times with 500 trees and deterministic=True, so expect minutes
rather than seconds.
"""

from __future__ import annotations

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
STATION = "MY1"
FEATURES_PATH = REPO_ROOT / "data/features" / f"{STATION}.parquet"
OUT_PATH = REPO_ROOT / "data/oof" / f"{STATION}_oof.parquet"


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


def main() -> None:
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

    oof = run_walk_forward(frame, models, f2_cols=F2_COLS, station=STATION)

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