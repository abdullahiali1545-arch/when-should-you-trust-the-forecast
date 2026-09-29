"""src/live.py - W4b step 5b: the live prospective logger.

One run does this, per station:

  1. Fetch recent AURN (current-year file FORCE re-downloaded - the ingest
     cache would otherwise serve a stale copy forever) and Open-Meteo weather,
     through src.ingest's own functions. Weather after the latest published
     reading is dropped.
  2. Clean with ingest.impute and build features with features.build_features:
     the exact code the training data went through.
  3. Load the frozen F3 and check it against its manifest: feature list and
     order, and the LightGBM version. Any mismatch stops the run.
  4. SHADOW forecasts: F3 at every recent hourly origin whose inputs pass the
     training mask. Used only to compute R2, exactly as the backtest's
     resid_mae_24h used F3 forecasts from every hour. Never logged, never
     scored.
  5. PROSPECTIVE forecasts: only origins whose target has not been published
     yet (latest reading L, origins in (L-6h, L]). Each (station, origin) is
     forecast once and appended to forecasts.csv. Never edited.
  6. R2 flag per prospective forecast: "warn" if resid_mae_24h >= the frozen
     threshold, "ok" if below, "grey" if fewer than 12 scored shadow forecasts
     fall in the window.
  7. Score: for logged forecasts whose truth has now been published, append
     the truth to scores.csv. Never edited.
  8. Write summary.json for the map.

Files (all under --live-dir, default data/live):
    forecasts.csv   append-only, one row per prospective forecast
    scores.csv      append-only, one row per truth when it arrives
    skips.csv       append-only, origins that could not be forecast and why
    summary.json    overwritten each run: the scoreboard, descriptive only

Run from the repo root:
    python -m src.live --live-dir data/live_test      # testing: never the real log
    python -m src.live                                # the real log, from launch
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import warnings

# rdata cannot map R's POSIXct class and says so once per file read. Harmless:
# ingest converts the raw epoch seconds itself. Silenced so real errors stand out.
warnings.filterwarnings("ignore", message="Missing constructor for R class")

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import build_features
from src.forecast import F2_COLS, build_feature_cols
from src.ingest import DEFRA, RAW, _download, impute, load_aurn_year, load_weather, site_coords
from src.watcher_features import residual_features

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
STATIONS = ["MY1", "KC1", "BEX", "HRL"]
HORIZON = pd.Timedelta(hours=6)
HIST = pd.Timedelta(days=10)       # enough for lag-24 features plus the 30h R2 window
MIN_RESID = 12                     # matches MIN_RECENT_OBS in watcher_features

FORECAST_COLS = ["run_utc", "station", "origin_utc", "target_utc", "latest_obs_utc",
                 "data_delay_h", "yhat_F3", "yhat_F0", "resid_mae_24h", "n_resid_24h",
                 "threshold", "flag", "origin_imputed", "model_commit"]
SCORE_COLS = ["scored_run_utc", "station", "origin_utc", "target_utc", "y_true", "truth_imputed"]
SKIP_COLS = ["run_utc", "station", "origin_utc", "reason"]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def fetch_station_frame(st: str, now: pd.Timestamp) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Recent AURN + weather, cleaned exactly as ingest.build() cleans history."""
    lat, lon, _ = site_coords(st)
    years = sorted({(now - HIST).year, now.year})
    frames = []
    for y in years:
        # Force: these files change daily. The cache is right for history, fatal here.
        _download(f"{DEFRA}/{st}_{y}.RData", RAW / f"{st}_{y}.RData", force=True)
        frames.append(load_aurn_year(st, y))
    aurn = (pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset="date", keep="first").sort_values("date").set_index("date"))

    L = aurn["pm2_5"].last_valid_index()
    if L is None:
        raise RuntimeError("no PM2.5 readings at all")
    aurn = aurn[(aurn.index >= L - HIST) & (aurn.index <= L)]

    weather = load_weather(lat, lon, (L - HIST).strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"))
    weather = weather[weather.index <= L]          # nothing after the latest reading

    df = impute(aurn.join(weather, how="left"))
    return df, L


def load_frozen(st: str, feats: pd.DataFrame) -> tuple[lgb.Booster, dict, list[str]]:
    manifest = json.loads((MODELS / f"manifest_{st}.json").read_text())
    cols = manifest["feature_cols"]
    if build_feature_cols(feats) != cols:
        raise RuntimeError(f"{st}: live feature columns differ from the frozen model's")
    if lgb.__version__ != manifest["versions"]["lightgbm"]:
        raise RuntimeError(f"{st}: LightGBM {lgb.__version__} != frozen "
                           f"{manifest['versions']['lightgbm']}; predictions would not be identical")
    booster = lgb.Booster(model_file=str(MODELS / manifest["file"]))
    if booster.feature_name() != cols:
        raise RuntimeError(f"{st}: model file's feature names differ from its manifest")
    return booster, manifest, cols


# ---------------------------------------------------------------------------
# append-only files
# ---------------------------------------------------------------------------

def read_csv(path: Path, cols: list[str]) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=cols)


def append(path: Path, rows: list[dict], cols: list[str]) -> None:
    if not rows:
        return
    pd.DataFrame(rows, columns=cols).to_csv(path, mode="a", header=not path.exists(), index=False)


def iso(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# one station
# ---------------------------------------------------------------------------

def run_station(st: str, now: pd.Timestamp, logged: set, unscored: pd.DataFrame
                ) -> tuple[list[dict], list[dict], list[dict], dict]:
    df, L = fetch_station_frame(st, now)
    feats = build_features(df)
    booster, manifest, cols = load_frozen(st, feats)
    thr = float(manifest["r2_threshold_ug_m3"])

    # training mask: every F2 input present at the origin
    ok = feats[list(F2_COLS)].notna().all(axis=1)

    # shadow forecasts at every recent hour - R2's inputs only
    shadow = pd.DataFrame({
        "origin": feats.index[ok],
        "y_true": feats.loc[ok, "y_t6"].to_numpy(),
        "yhat_F3": booster.predict(feats.loc[ok, cols]),
    })
    known = shadow.dropna(subset=["y_true"])
    known_at = pd.DatetimeIndex(known["origin"]) + HORIZON     # when each error became knowable

    forecasts, skips = [], []
    candidates = feats.index[(feats.index > L - HORIZON) & (feats.index <= L)]
    for t in candidates:
        key = (st, iso(t))
        if key in logged:
            continue
        if not ok.loc[t]:
            missing = [c for c in F2_COLS if pd.isna(feats.at[t, c])]
            skips.append({"run_utc": iso(now), "station": st, "origin_utc": iso(t),
                          "reason": "missing inputs: " + ",".join(missing)})
            continue
        r24 = residual_features(known, pd.DatetimeIndex([t]))["resid_mae_24h"].iloc[0]
        n24 = int(((known_at > t - pd.Timedelta(hours=24)) & (known_at <= t)).sum())
        if n24 < MIN_RESID or pd.isna(r24):
            flag = "grey"
        else:
            flag = "warn" if r24 >= thr else "ok"
        forecasts.append({
            "run_utc": iso(now), "station": st, "origin_utc": iso(t), "target_utc": iso(t + HORIZON),
            "latest_obs_utc": iso(L), "data_delay_h": round((now - L).total_seconds() / 3600, 1),
            "yhat_F3": float(booster.predict(feats.loc[[t], cols])[0]),
            "yhat_F0": float(feats.at[t, "pm2_5_lag_0"]),
            "resid_mae_24h": None if pd.isna(r24) else float(r24), "n_resid_24h": n24,
            "threshold": thr, "flag": flag, "origin_imputed": bool(feats.at[t, "imputed"]),
            "model_commit": manifest["git"]["commit"][:7],
        })

    # scoring: truths for earlier forecasts that have now been published
    scores = []
    for _, r in unscored[unscored["station"] == st].iterrows():
        t = pd.Timestamp(r["origin_utc"])
        if t in feats.index and pd.notna(feats.at[t, "y_t6"]):
            target = t + HORIZON
            scores.append({"scored_run_utc": iso(now), "station": st, "origin_utc": r["origin_utc"],
                           "target_utc": iso(target), "y_true": float(feats.at[t, "y_t6"]),
                           "truth_imputed": bool(df.at[target, "imputed"]) if target in df.index else None})

    status = {"latest_obs_utc": iso(L), "data_delay_h": round((now - L).total_seconds() / 3600, 1),
              "shadow_forecasts": int(len(shadow)), "new_forecasts": len(forecasts),
              "new_skips": len(skips), "new_scores": len(scores)}
    return forecasts, scores, skips, status


# ---------------------------------------------------------------------------
# summary for the map - descriptive only
# ---------------------------------------------------------------------------

def build_summary(live: Path, now: pd.Timestamp, status: dict) -> dict:
    fc = read_csv(live / "forecasts.csv", FORECAST_COLS)
    sc = read_csv(live / "scores.csv", SCORE_COLS)
    out = {"generated_utc": iso(now),
           "note": "Descriptive only until 26 weeks incl. a full Jan-Mar (PROJECT_SPEC 2026-09-29).",
           "stations": {}}
    if len(fc):
        out["first_forecast_utc"] = fc["run_utc"].min()
    for st in STATIONS:
        s = {"run_status": status.get(st)}
        f = fc[fc["station"] == st]
        if len(f):
            last = f.sort_values("origin_utc").iloc[-1]
            s["latest"] = {k: (None if pd.isna(last[k]) else last[k].item() if hasattr(last[k], "item") else last[k])
                           for k in ["origin_utc", "target_utc", "yhat_F3", "flag", "data_delay_h"]}
            m = f.merge(sc[sc["station"] == st][["origin_utc", "y_true"]], on="origin_utc")
            if len(m):
                e3 = (m["y_true"] - m["yhat_F3"]).abs()
                e0 = (m["y_true"] - m["yhat_F0"]).abs()
                warn, okay = m["flag"] == "warn", m["flag"] == "ok"
                s["scoreboard"] = {
                    "n_scored": int(len(m)),
                    "mae_F3": round(float(e3.mean()), 3),
                    "mae_persistence": round(float(e0.mean()), 3),
                    "n_warn": int(warn.sum()), "n_ok": int(okay.sum()), "n_grey": int((m["flag"] == "grey").sum()),
                    "mae_F3_warn": round(float(e3[warn].mean()), 3) if warn.any() else None,
                    "mae_F3_ok": round(float(e3[okay].mean()), 3) if okay.any() else None,
                }
        out["stations"][st] = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-dir", default="data/live")
    ap.add_argument("--stations", nargs="+", default=STATIONS)
    args = ap.parse_args()

    live = ROOT / args.live_dir
    live.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC").floor("s")

    fc = read_csv(live / "forecasts.csv", FORECAST_COLS)
    sc = read_csv(live / "scores.csv", SCORE_COLS)
    logged = set(zip(fc["station"], fc["origin_utc"]))
    done = set(zip(sc["station"], sc["origin_utc"]))
    is_unscored = pd.Series([k not in done for k in zip(fc["station"], fc["origin_utc"])],
                            index=fc.index, dtype=bool)
    unscored = fc[is_unscored]                     # a boolean Series, never a bare list:
                                                   # fc[[]] would select no COLUMNS

    status = {}
    for st in args.stations:
        try:
            f, s, k, stat = run_station(st, now, logged, unscored)
            append(live / "forecasts.csv", f, FORECAST_COLS)
            append(live / "scores.csv", s, SCORE_COLS)
            append(live / "skips.csv", k, SKIP_COLS)
            status[st] = stat
        except Exception as e:                        # one station failing must not stop the others
            status[st] = {"error": f"{type(e).__name__}: {e}"[:200]}
        print(f"{st}: {status[st]}")

    summary = build_summary(live, now, status)
    (live / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {live.relative_to(ROOT)}/summary.json")


if __name__ == "__main__":
    main()