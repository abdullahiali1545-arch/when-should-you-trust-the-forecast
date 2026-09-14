"""src/forecast.py — F0 persistence, F1 climatology, F2 linear, F3 LightGBM.

All four share one interface so the harness can treat them identically:

    model.fit(train: pd.DataFrame) -> None
    model.predict(test: pd.DataFrame) -> np.ndarray

Frames are indexed by ORIGIN (UTC) and carry the feature columns plus:
    LAG0_COL    PM2.5 at the origin                  (F0 reads this)
    HOUR_COL    hour of day at the TARGET time       (F1 groups on this)
    MONTH_COL   calendar month at the TARGET time    (F1 groups on this)
    TARGET_COL  PM2.5 at origin + h

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

# --- column names, matched to data/features/<STATION>.parquet ---------------
LAG0_COL = "pm2_5_lag_0"

# F1 predicts PM2.5 at t+6, so it groups on the TARGET's hour and month, not
# the origin's. Grouping on the origin's clock would mis-specify F1 by six
# hours. target_hour / target_month are legitimate features: the calendar at
# t+6 is deterministic and fully known at time t.
HOUR_COL = "target_hour"
MONTH_COL = "target_month"

# The target lives in the feature table as y_t6 (built by src/features.py,
# contrary to docs/harness_design.md §1 — recorded in the spec changelog).
# It must therefore NEVER appear in any model's feature_cols. See EXCLUDE
# below and the assert in build_feature_cols().
TARGET_COL = "y_t6"

# Columns that are not features, for any model, ever.
#   y_t6     the target — including it means predicting the target from the
#            target: MAE near zero, no error, no warning. The highest-severity
#            leak available in this project.
#   imputed  a data-quality flag, not a physical predictor.
EXCLUDE: frozenset[str] = frozenset({TARGET_COL, "imputed"})

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

# F2's feature set: weather-conditioned plus the short autoregressive terms.
# Deliberately explicit and small. F2 exists to answer "did the watcher merely
# detect unusual weather?", so it has to be a weather-conditioned reference a
# reviewer can read in one glance.
F2_COLS: tuple[str, ...] = (
    "pm2_5_lag_0",
    "pm2_5_lag_1",
    "pm2_5_lag_3",
    "pm2_5_lag_6",
    "pm2_5_lag_24",
    "no2_lag_0",
    "pm2_5_mean_24h",
    "pm2_5_delta_6h",
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_u",
    "wind_v",
)


def build_feature_cols(frame: pd.DataFrame, exclude: frozenset[str] = EXCLUDE) -> list[str]:
    """Every column except the target and non-feature flags.

    Never write `feature_cols = list(df.columns)` anywhere in this project.
    The assert below is the last line of defence against the target leak.
    """
    cols = [c for c in frame.columns if c not in exclude]
    assert TARGET_COL not in cols, (
        f"{TARGET_COL} is in the feature list — this is the catastrophic leak"
    )
    return cols


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
    """Mean target by (target hour x target month), refitted on T_k every fold.

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

    def __init__(self, feature_cols: Sequence[str] = F2_COLS, alpha: float = 1.0) -> None:
        self.feature_cols = [c for c in feature_cols if c not in EXCLUDE]
        assert TARGET_COL not in self.feature_cols, "target in F2 feature list"
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
        self.feature_cols = [c for c in feature_cols if c not in EXCLUDE]
        assert TARGET_COL not in self.feature_cols, "target in F3 feature list"
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