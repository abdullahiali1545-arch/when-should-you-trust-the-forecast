"""src/forecast.py — F0 persistence, F1 climatology, F2 linear, F3 LightGBM.

All four share one interface so the harness can treat them identically:

    model.fit(train: pd.DataFrame) -> None
    model.predict(test: pd.DataFrame) -> np.ndarray

Frames are indexed by ORIGIN (UTC) and carry the feature columns plus:
    LAG0_COL   current PM2.5 at the origin   (F0 reads this)
    HOUR_COL   hour of day, Europe/London    (F1 groups on this)
    MONTH_COL  calendar month                (F1 groups on this)
    TARGET_COL PM2.5 at origin + h           (built by the harness, never features.py)

Fitting happens on T_k only, every fold. Nothing here is fitted once globally —
F1's climatology means in particular must not see months from after the fold
boundary.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --- column names -----------------------------------------------------------
# CHECK THESE AGAINST src/features.py. If a name is wrong the harness raises
# immediately rather than silently scoring a different model.
LAG0_COL = "pm2_5"
HOUR_COL = "hour"
MONTH_COL = "month"
TARGET_COL = "target"

# --- F3 hyperparameters, pre-registered and not tuned -----------------------
# deterministic=True gives stable results across thread counts; it must be
# paired with force_row_wise (or force_col_wise) to avoid numerical
# instability, and it may slow training. Determinism holds within a LightGBM
# version, not across versions — record the version in the README.
F3_PARAMS = dict(
    objective="regression_l1",   # MAE is the headline metric; train the loss reported
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    subsample=1.0,               # no stochastic row sampling — determinism
    colsample_bytree=0.8,
    random_state=42,
    deterministic=True,
    force_row_wise=True,
    n_jobs=4,
    verbosity=-1,
)


def _require(frame: pd.DataFrame, cols: Sequence[str], who: str) -> None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"{who}: missing columns {missing}")


class F0Persistence:
    """yhat(t+6) = y(t). The bar, and the routing fallback. Nothing to fit."""

    name = "F0"

    def fit(self, train: pd.DataFrame) -> None:
        _require(train, [LAG0_COL], "F0.fit")

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        _require(test, [LAG0_COL], "F0.predict")
        return test[LAG0_COL].to_numpy(dtype="float64")


class F1Climatology:
    """Mean target by (hour of day x month), refitted on T_k every fold.

    Refitting matters: means computed once over all years would include months
    from after the fold boundary, which is a fold-dependency violation.
    """

    name = "F1"

    def __init__(self) -> None:
        self._table: pd.Series | None = None
        self._global: float = np.nan

    def fit(self, train: pd.DataFrame) -> None:
        _require(train, [HOUR_COL, MONTH_COL, TARGET_COL], "F1.fit")
        self._table = train.groupby([HOUR_COL, MONTH_COL])[TARGET_COL].mean()
        self._global = float(train[TARGET_COL].mean())

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        if self._table is None:
            raise RuntimeError("F1.predict called before fit")
        _require(test, [HOUR_COL, MONTH_COL], "F1.predict")
        keys = pd.MultiIndex.from_arrays([test[HOUR_COL], test[MONTH_COL]])
        # An (hour, month) pair unseen in training falls back to the training
        # global mean — never to a value computed from the test fold.
        return self._table.reindex(keys).fillna(self._global).to_numpy(dtype="float64")


class F2Linear:
    """Weather-conditioned ridge regression. The disagreement partner for the
    watcher, and the reference that answers 'did the watcher just detect
    unusual weather?'.

    Cannot accept NaN — this is why the harness computes one common scoring
    mask and applies it to all four models.
    """

    name = "F2"

    def __init__(self, feature_cols: Sequence[str], alpha: float = 1.0) -> None:
        self.feature_cols = list(feature_cols)
        self._pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(self, train: pd.DataFrame) -> None:
        _require(train, self.feature_cols + [TARGET_COL], "F2.fit")
        X = train[self.feature_cols].to_numpy(dtype="float64")
        if not np.isfinite(X).all():
            raise ValueError("F2.fit received NaN — the scoring mask is wrong")
        self._pipe.fit(X, train[TARGET_COL].to_numpy(dtype="float64"))

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        _require(test, self.feature_cols, "F2.predict")
        X = test[self.feature_cols].to_numpy(dtype="float64")
        return self._pipe.predict(X).astype("float64")


class F3LightGBM:
    """The ML model under scrutiny. Hyperparameters fixed, not tuned."""

    name = "F3"

    def __init__(self, feature_cols: Sequence[str], params: dict | None = None) -> None:
        self.feature_cols = list(feature_cols)
        self._model = LGBMRegressor(**(params or F3_PARAMS))

    def fit(self, train: pd.DataFrame) -> None:
        _require(train, self.feature_cols + [TARGET_COL], "F3.fit")
        self._model.fit(
            train[self.feature_cols],
            train[TARGET_COL].to_numpy(dtype="float64"),
        )

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        _require(test, self.feature_cols, "F3.predict")
        return self._model.predict(test[self.feature_cols]).astype("float64")