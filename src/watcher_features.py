"""src/watcher_features.py — the watcher's inputs (W3.2).

W3.1 built the answer key: was F3 unreliable at this hour? This module builds
the exam paper — what the watcher may look at when it guesses, at time t,
before the truth arrives.

Four families, per Part 9:

    distribution distance   are current conditions unlike the training window?
    volatility              is the signal jumpy right now?
    model disagreement      do F2 and F3 split?
    recent residuals        has F3 been wrong lately?

Three of the four are fold-dependent and therefore live here rather than in
src/features.py: the distances are measured against THIS fold's training
window, and the disagreement and residual families come from models refitted
each fold. Volatility is the exception — plain arithmetic on observed PM2.5,
identical wherever the fold boundaries fall — so it is read from the feature
table rather than recomputed.

THE SIX-HOUR RULE. A forecast made at origin s targets s+6h, so its residual
does not exist until s+6h. At time t the only residuals with realised outcomes
come from origins at or before t minus the horizon. Every residual feature here
is built by shifting residual timestamps forward by the horizon FIRST, so that
each value sits at the moment it became knowable; after that shift an ordinary
backward-looking rolling window cannot reach into the future.

Residual history deliberately crosses the fold boundary. At 09:00 in the middle
of a test quarter, forecasts made earlier that same quarter have realised
outcomes and a real service would use them. Those are out-of-fold predictions,
so using them leaks nothing. History is therefore cut by TIME, never by fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

HORIZON = 6

# Pre-registered 2026-09-18. Not tuned against watcher performance.
DIST_VARS = [
    "pm2_5_lag_0",
    "wind_u",
    "wind_v",
    "temperature_2m",
    "relative_humidity_2m",
    "yhat_F3",
]
DIST_WINDOWS = ["24h", "168h"]
RESID_WINDOWS = ["24h", "168h"]

# Volatility columns read straight from the feature table.
VOL_STD_COLS = ["pm2_5_std_3h", "pm2_5_std_6h", "pm2_5_std_12h", "pm2_5_std_24h"]
VOL_DELTA_COLS = ["pm2_5_delta_1h", "pm2_5_delta_3h", "pm2_5_delta_6h"]

# A distance estimated from a handful of observations is noise.
MIN_RECENT_OBS = 12
# The reference distribution is subsampled for speed. Evenly spaced, so the
# sample spans the whole training period rather than one corner of it.
MAX_REF_SAMPLE = 3000


# ---------------------------------------------------------------------------
# Family 4 — recent residuals
# ---------------------------------------------------------------------------

def residual_features(
    oof_history: pd.DataFrame,
    origins: pd.DatetimeIndex,
    horizon: int = HORIZON,
    windows: list[str] = RESID_WINDOWS,
) -> pd.DataFrame:
    """Rolling MAE, signed bias and residual spread of F3, legally lagged.

    `oof_history` needs columns origin, y_true, yhat_F3. It may contain origins
    from any fold: the horizon shift below, not a fold filter, is what keeps
    this honest.

    Bias keeps its sign on purpose. A persistently positive bias says F3 is
    running low, which is a different condition from large but unbiased error —
    that is why it is a separate feature from MAE.
    """
    resid = pd.Series(
        (oof_history["y_true"] - oof_history["yhat_F3"]).to_numpy(),
        index=pd.DatetimeIndex(oof_history["origin"]),
    ).sort_index()
    resid = resid[resid.notna()]

    # THE KEY LINE. Move each residual to the moment it became knowable. After
    # this the index no longer means "when the forecast was made" but "when its
    # error could first be observed", so a backward rolling window is safe.
    known = resid.copy()
    known.index = known.index + pd.Timedelta(hours=horizon)
    known = known[~known.index.duplicated(keep="last")].sort_index()

    out = pd.DataFrame(index=origins)
    for w in windows:
        # Time-based windows, not row counts. MY1 has multi-week outages; a
        # rolling(24) would happily span one and call it a day.
        roll = known.rolling(w)
        mae = known.abs().rolling(w).mean()
        bias = roll.mean()
        sd = roll.std()

        # reindex onto the origins we need, taking the last value known at or
        # before each origin. No interpolation: a gap means no information.
        out[f"resid_mae_{w}"] = mae.reindex(origins, method="ffill")
        out[f"resid_bias_{w}"] = bias.reindex(origins, method="ffill")
        out[f"resid_std_{w}"] = sd.reindex(origins, method="ffill")

    return out


# ---------------------------------------------------------------------------
# Family 3 — model disagreement
# ---------------------------------------------------------------------------

def disagreement_features(
    oof_fold: pd.DataFrame,
    oof_history: pd.DataFrame,
    origins: pd.DatetimeIndex,
) -> pd.DataFrame:
    """abs(yhat_F3 - yhat_F2) at the origin, plus its 24h rolling mean.

    Needs no ground truth and no waiting: both predictions exist the moment the
    forecast is made, so unlike the residual family there is no horizon shift.
    """
    gap_hist = pd.Series(
        (oof_history["yhat_F3"] - oof_history["yhat_F2"]).abs().to_numpy(),
        index=pd.DatetimeIndex(oof_history["origin"]),
    ).sort_index()
    gap_hist = gap_hist[~gap_hist.index.duplicated(keep="last")]

    gap_now = pd.Series(
        (oof_fold["yhat_F3"] - oof_fold["yhat_F2"]).abs().to_numpy(),
        index=pd.DatetimeIndex(oof_fold["origin"]),
    ).sort_index()

    out = pd.DataFrame(index=origins)
    out["disagree"] = gap_now.reindex(origins)
    out["disagree_mean_24h"] = gap_hist.rolling("24h").mean().reindex(
        origins, method="ffill"
    )
    return out


# ---------------------------------------------------------------------------
# Family 2 — volatility
# ---------------------------------------------------------------------------

def volatility_features(feats: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
    """Read the fold-independent volatility columns straight from the feature
    table. Recomputing them here would risk two definitions drifting apart.
    """
    wanted = [c for c in VOL_STD_COLS + VOL_DELTA_COLS if c in feats.columns]
    missing = set(VOL_STD_COLS + VOL_DELTA_COLS) - set(wanted)
    if missing:
        raise KeyError(f"volatility columns absent from the feature table: {sorted(missing)}")

    out = feats.loc[feats.index.intersection(origins), wanted].reindex(origins)
    # Rate of change matters by size, not direction; direction is already
    # carried by the residual bias feature.
    for c in VOL_DELTA_COLS:
        out[c] = out[c].abs()
    return out.add_prefix("vol_")


# ---------------------------------------------------------------------------
# Family 1 — distribution distance
# ---------------------------------------------------------------------------

def _reference(values: np.ndarray) -> np.ndarray:
    """Evenly spaced subsample of the training values, for speed."""
    values = values[~np.isnan(values)]
    if values.size > MAX_REF_SAMPLE:
        idx = np.linspace(0, values.size - 1, MAX_REF_SAMPLE).astype(int)
        values = values[idx]
    return values


def distribution_features(
    series: pd.DataFrame,
    origins: pd.DatetimeIndex,
    train_index: pd.DatetimeIndex,
    variables: list[str] = DIST_VARS,
    windows: list[str] = DIST_WINDOWS,
) -> pd.DataFrame:
    """Wasserstein distance, recent window against this fold's training window.

    Each variable is standardised using TRAINING mean and standard deviation
    before the distance is taken. Without that, pressure near 1000 and wind
    speed near 5 would not be comparable and the larger-scale variable would
    dominate every column.

    Computed once per calendar day at midnight and held for that day. A per-hour
    computation is far too slow to run inside a 19-fold harness. The value an
    origin sees is therefore built from data ending at the previous midnight:
    slightly stale, and unable to reach forward, which is the direction that
    matters.
    """
    anchors = pd.DatetimeIndex(sorted(set(origins.normalize())))
    out = pd.DataFrame(index=anchors)

    for var in variables:
        if var not in series.columns:
            raise KeyError(f"{var!r} not in the frame passed to distribution_features")
        col = series[var]
        train_vals = col.reindex(train_index).to_numpy(dtype=float)
        ref = _reference(train_vals)
        if ref.size == 0:
            raise ValueError(f"{var!r} has no usable training values")

        mu = float(np.nanmean(ref))
        sd = float(np.nanstd(ref))
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        ref_std = (ref - mu) / sd

        for w in windows:
            span = pd.Timedelta(w)
            vals = []
            for a in anchors:
                # Half-open window ending AT the anchor: strictly past data.
                recent = col.loc[(col.index > a - span) & (col.index <= a)]
                recent = recent.to_numpy(dtype=float)
                recent = recent[~np.isnan(recent)]
                if recent.size >= MIN_RECENT_OBS:
                    vals.append(wasserstein_distance(ref_std, (recent - mu) / sd))
                else:
                    vals.append(np.nan)
            out[f"wass_{var}_{w}"] = vals

    # Map each origin to its own day's value.
    return out.reindex(origins.normalize()).set_axis(origins)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_watcher_features(
    oof_fold: pd.DataFrame,
    oof_all: pd.DataFrame,
    feats: pd.DataFrame,
    train_index: pd.DatetimeIndex,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """All four families for one fold, one row per origin in `oof_fold`.

    oof_fold    rows of the fold being labelled: origin, yhat_F2, yhat_F3, y_true
    oof_all     the full OOF table. Cut by TIME below, never by fold.
    feats       the fold-independent feature table, indexed by origin
    train_index origins of this fold's training window, defining the reference
                distribution for family 1
    """
    origins = pd.DatetimeIndex(oof_fold["origin"]).sort_values()

    # Residual history: every origin whose outcome had arrived by the LAST
    # origin we need a feature for. The per-origin horizon shift inside
    # residual_features does the real work; this is just a cheap pre-filter.
    cutoff = origins.max() - pd.Timedelta(hours=horizon)
    hist = oof_all[pd.DatetimeIndex(oof_all["origin"]) <= cutoff]

    # yhat_F3 joins the distribution monitor, so the frame family 1 sees is the
    # feature table plus that one column.
    yhat = pd.Series(
        oof_all["yhat_F3"].to_numpy(),
        index=pd.DatetimeIndex(oof_all["origin"]),
    ).sort_index()
    yhat = yhat[~yhat.index.duplicated(keep="last")]
    series = feats.copy()
    series["yhat_F3"] = yhat.reindex(feats.index)

    parts = [
        distribution_features(series, origins, train_index),
        volatility_features(feats, origins),
        disagreement_features(oof_fold, hist, origins),
        residual_features(hist, origins, horizon=horizon),
    ]
    out = pd.concat(parts, axis=1)
    out.index.name = "origin"
    return out


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    oof = pd.read_parquet(root / "data/oof" / "MY1_oof.parquet")
    feats = pd.read_parquet(root / "data/features" / "MY1.parquet")

    K = 10
    fold = oof[oof["fold"].eq(K)]
    origins = pd.DatetimeIndex(fold["origin"]).sort_values()
    train_index = feats.index[feats.index < origins.min()]

    print(f"fold {K}: {len(fold):,} origins, "
          f"{origins.min()} to {origins.max()}")
    print(f"training window: {len(train_index):,} rows\n")

    X = build_watcher_features(fold, oof, feats, train_index)
    print(f"{X.shape[0]:,} rows x {X.shape[1]} features\n")
    print(X.isna().mean().round(3).to_string())

    # --- window self-check ---------------------------------------------------
    # Recompute resid_mae_24h by hand at three origins. The eligible window is
    # origins strictly after t minus 30h and at or before t minus 6h.
    resid = pd.Series(
        (oof["y_true"] - oof["yhat_F3"]).to_numpy(),
        index=pd.DatetimeIndex(oof["origin"]),
    ).sort_index()

    print("\nmanual check of resid_mae_24h:")
    for t in origins[[200, 800, 1500]]:
        lo = t - pd.Timedelta(hours=30)
        hi = t - pd.Timedelta(hours=6)
        w = resid[(resid.index > lo) & (resid.index <= hi)].abs()
        manual = w.mean()
        got = X.loc[t, "resid_mae_24h"]
        ok = "ok" if (np.isnan(manual) and np.isnan(got)) or np.isclose(manual, got) else "MISMATCH"
        print(f"  {t}  manual {manual:.4f}  feature {got:.4f}  n={len(w):3d}  {ok}")