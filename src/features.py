"""
src/features.py — fold-independent feature engineering.

W1.7. Consumes the cleaned per-station Parquet written by src/ingest.py and
writes a feature table per station.

THE ONE RULE THIS FILE OBEYS
----------------------------
Every column here passes the fold test (docs/information_contract.md §5):

    Would this column's value at a fixed timestamp change if I moved a
    fold boundary?  No -> here.  Yes -> the walk-forward harness.

Nothing in this file is fitted. No scaler, no climatology, no residual, no
model output, no distribution distance against a training window. Those are
deferred to src/evaluate.py by design, not by omission — see §5's
"Consequence for W1.7".

Practical consequence: three of the watcher's four feature families are NOT
built here. Only volatility (rolling sigma) is.

Usage
-----
    python src/features.py                     # all stations found
    python src/features.py --station MY1       # one station
    python src/features.py --canary            # run the leakage self-test
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration. Every number here is a design decision; see the module notes.
# ----------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")

HORIZON_H = 6                      # forecast horizon: PM2.5 at t+6h
LOCAL_TZ = "Europe/London"         # calendar features are a local-clock question

POLLUTANTS = ["pm2_5", "no2"]      # lagged / rolled. PM2.5 is the target.
LAGS_H = [0, 1, 3, 6, 12, 24]      # spec Part 7. lag 0 = the reading at t.
ROLL_WINDOWS_H = [3, 6, 12, 24]
DELTA_WINDOWS_H = [1, 3, 6]

# Weather observed at t. ERA5 reanalysis - see contract §6 for the caveat.
WEATHER_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "boundary_layer_height",
]
WIND_SPEED_COL = "wind_speed_10m"
WIND_DIR_COL = "wind_direction_10m"

# A rolling statistic over a mostly-empty window is not the same statistic.
# Require this fraction of the window to be real observations, else NaN.
MIN_PERIODS_FRAC = 0.75


def _min_periods(window: int) -> int:
    return max(1, math.ceil(MIN_PERIODS_FRAC * window))


# ----------------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------------

def ensure_hourly_grid(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Reindex onto a gapless hourly UTC grid.

    Why: every rolling window below is specified in ROWS, not in time. If an
    hour is missing as a row rather than present-as-NaN, `.rolling(24)` quietly
    reaches back 25 or 30 real hours and the feature stops meaning what its
    name says. Inserting the missing rows as NaN makes the window honest, and
    MIN_PERIODS_FRAC then decides whether enough real data survives.

    Fold-independent: the grid comes from the station's own first and last
    timestamp, not from any fold.
    """
    if df.index.tz is None:
        raise ValueError("index must be tz-aware UTC; got naive")

    full = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    inserted = len(full) - len(df)
    out = df.reindex(full)
    out.index.name = df.index.name or "date"
    return out, inserted


# ----------------------------------------------------------------------------
# Feature blocks. Each returns a dict of {column_name: Series}.
# ----------------------------------------------------------------------------

def lag_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Past readings. Fold-independent: an observation at a fixed timestamp is
    a fixed fact — once known it never changes."""
    out = {}
    for col in POLLUTANTS:
        if col not in df:
            continue
        for lag in LAGS_H:
            out[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return out


def rolling_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Rolling mean and standard deviation over the row's OWN past.

    The window ends at t and includes t (legal — contract §1: the reading at t
    has arrived). It never reaches forward. Fold-independent: arithmetic on
    observed readings, no fitted parameter.

    Rolling sigma is the volatility family — the only watcher feature family
    that survives the fold test and can be built here.
    """
    out = {}
    for col in POLLUTANTS:
        if col not in df:
            continue
        s = df[col]
        for w in ROLL_WINDOWS_H:
            mp = _min_periods(w)
            out[f"{col}_mean_{w}h"] = s.rolling(w, min_periods=mp).mean()
            out[f"{col}_std_{w}h"] = s.rolling(w, min_periods=mp).std()
    return out


def delta_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Rate of change: value now minus value w hours ago. Fold-independent."""
    out = {}
    for col in POLLUTANTS:
        if col not in df:
            continue
        for w in DELTA_WINDOWS_H:
            out[f"{col}_delta_{w}h"] = df[col] - df[col].shift(w)
    return out


def wind_components(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Decompose wind into u/v components.

    Why not raw degrees: direction is circular. 359 deg and 1 deg are one
    degree apart in reality and 358 apart numerically. A tree splitting on
    "direction < 180" cuts the compass in an arbitrary place and cannot
    represent "northerly". Projecting onto two axes removes the discontinuity.

        u = speed * cos(theta),  v = speed * sin(theta)   [spec Part 7]

    Note on convention: wind_direction_10m is the direction the wind blows
    FROM, so u/v as defined here are a fixed rotation/reflection of the true
    meteorological vector. That is a constant transform applied to every row,
    so it cannot bias a tree model — but do not describe these as "eastward
    and northward wind" in the write-up without checking the sign.

    Fold-independent: pure per-row trigonometry on that row's own values.
    """
    if WIND_SPEED_COL not in df or WIND_DIR_COL not in df:
        return {}
    theta = np.deg2rad(df[WIND_DIR_COL].astype("float64"))
    speed = df[WIND_SPEED_COL].astype("float64")
    return {"wind_u": speed * np.cos(theta), "wind_v": speed * np.sin(theta)}


def weather_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """ERA5 weather observed at t, passed through unchanged.

    Mode B forbids weather after t (contract §2). Nothing here is shifted
    forward. See §6 on the mild reanalysis look-ahead — accepted, not
    engineered around, but it must be stated in the README.
    """
    return {c: df[c].astype("float64") for c in WEATHER_COLS if c in df}


def calendar_features(idx: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Calendar features for t and for the target hour t+6h.

    Legal despite naming a future hour (contract §4): calendars are
    deterministic. Standing at 09:00 you can compute 15:00 and check the date
    without waiting for anything.

    These are the pointer, not the pattern. Remove them and F3 cannot tell a
    February afternoon from 3am in July.

    Derived in Europe/London, NOT UTC. "Is it rush hour" is a local-clock
    question; the index is UTC, so in summer UTC hour 8 is local hour 9. Get
    this wrong and every diurnal feature shifts by an hour for half the year.
    """
    local = idx.tz_convert(LOCAL_TZ)
    target_local = (idx + pd.Timedelta(hours=HORIZON_H)).tz_convert(LOCAL_TZ)

    def s(values):
        return pd.Series(values, index=idx)

    return {
        "hour": s(local.hour),
        "dow": s(local.dayofweek),
        "month": s(local.month),
        "target_hour": s(target_local.hour),
        "target_dow": s(target_local.dayofweek),
        "target_month": s(target_local.month),
        "target_is_weekend": s((target_local.dayofweek >= 5).astype("int8")),
    }


def build_target(df: pd.DataFrame) -> pd.Series:
    """y_t6: PM2.5 at t+6h. The TARGET, never an input.

    shift(-HORIZON) is the one backwards-in-time operation in this file and it
    is correct precisely because this column is not a feature. It is the thing
    being predicted. Any model that reads it has not leaked — it has cheated
    outright.
    """
    return df["pm2_5"].shift(-HORIZON_H)


# ----------------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw station frame -> feature table. Pure function: same input, same
    output, no reference to any fold, file or global statistic."""
    df, _ = ensure_hourly_grid(df)

    blocks: dict[str, pd.Series] = {}
    blocks.update(lag_features(df))
    blocks.update(rolling_features(df))
    blocks.update(delta_features(df))
    blocks.update(weather_features(df))
    blocks.update(wind_components(df))
    blocks.update(calendar_features(df.index))

    feats = pd.DataFrame(blocks, index=df.index)
    feats["y_t6"] = build_target(df)

    # Carried through, not a model input: flags rows touched by ingest-time
    # interpolation (gaps < 2h). Fold-independent — deterministic arithmetic on
    # two observed neighbours, no fitted parameter (contract §5).
    if "imputed" in df:
        feats["imputed"] = df["imputed"].fillna(False).astype(bool)

    return feats


# ----------------------------------------------------------------------------
# Self-checks
# ----------------------------------------------------------------------------

def assert_no_forward_reach(raw: pd.DataFrame, feats: pd.DataFrame) -> None:
    """Cheap structural check that no FEATURE column reads the future.

    Recomputes two features by hand and compares. Not a substitute for the
    Week 2 canary test on the harness — it checks this file only.
    """
    raw, _ = ensure_hourly_grid(raw)
    expected = raw["pm2_5"].shift(6)
    pd.testing.assert_series_equal(
        feats["pm2_5_lag_6"], expected, check_names=False
    )
    expected_delta = raw["pm2_5"] - raw["pm2_5"].shift(3)
    pd.testing.assert_series_equal(
        feats["pm2_5_delta_3h"], expected_delta, check_names=False
    )


def canary(raw: pd.DataFrame) -> None:
    """Leakage canary for this file (spec Part 10, scaled down).

    Corrupt one input value at a single timestamp t*, rebuild, and assert every
    FEATURE row strictly before t* is bit-identical. If an earlier row moved,
    information travelled backwards in time.

    y_t6 is excluded from the comparison: it is the target and is *supposed* to
    look forward. Corrupting pm2_5 at t* legitimately changes y_t6 at t*-6h.
    """
    base = build_features(raw)

    t_star = raw.index[len(raw) // 2]
    poisoned = raw.copy()
    poisoned.loc[t_star, "pm2_5"] = 99999.0
    after = build_features(poisoned)

    cols = [c for c in base.columns if c != "y_t6"]
    b = base.loc[base.index < t_star, cols]
    a = after.loc[after.index < t_star, cols]

    pd.testing.assert_frame_equal(a, b)
    print(f"  canary PASS  (t* = {t_star}, {len(b):,} earlier rows unchanged)")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def process_station(path: Path, run_canary: bool) -> None:
    station = path.stem
    raw = pd.read_parquet(path)
    print(f"\n{station}: {len(raw):,} rows, {raw.index.min()} -> {raw.index.max()}")

    gridded, inserted = ensure_hourly_grid(raw)
    if inserted:
        print(f"  grid: inserted {inserted:,} missing hourly rows as NaN")

    # Audit boundary_layer_height. It was entirely null for H1 2024 at MY1's
    # coordinates. A silently-null physical driver would make the watcher
    # misfire; surface it here rather than in Week 2.
    if "boundary_layer_height" in gridded:
        blh = gridded["boundary_layer_height"]
        print(f"  boundary_layer_height null: {blh.isna().mean():.1%} overall")
        by_year = blh.isna().groupby(gridded.index.year).mean()
        bad = by_year[by_year > 0.5]
        if len(bad):
            print("  WARNING - >50% null in: "
                  + ", ".join(f"{y} ({v:.0%})" for y, v in bad.items()))

    feats = build_features(raw)
    assert_no_forward_reach(raw, feats)
    if run_canary:
        canary(raw)

    usable = feats.dropna(subset=["y_t6"])
    complete = usable.dropna()
    print(f"  features: {feats.shape[1]} columns, {len(feats):,} rows")
    print(f"  rows with a target:        {len(usable):,}")
    print(f"  rows with no NaN anywhere: {len(complete):,} "
          f"({len(complete) / len(feats):.1%})")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FEATURES_DIR / f"{station}.parquet"
    feats.to_parquet(out)
    print(f"  wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build fold-independent features.")
    p.add_argument("--station", help="station code, e.g. MY1. Default: all.")
    p.add_argument("--canary", action="store_true", help="run the leakage canary")
    args = p.parse_args()

    if args.station:
        paths = [PROCESSED_DIR / f"{args.station}.parquet"]
    else:
        paths = sorted(PROCESSED_DIR.glob("*.parquet"))

    if not paths:
        raise SystemExit(f"no parquet files found in {PROCESSED_DIR}")

    for path in paths:
        if not path.exists():
            raise SystemExit(f"missing: {path}")
        process_station(path, run_canary=args.canary)

    print("\ndone.")


if __name__ == "__main__":
    main()