"""src/freeze.py - W4b step 5a: freeze one F3 per station for the live log.

Pre-registered 2026-09-29 ("W4b live prospective log"): one F3 per station,
same features and settings as W2, trained once on all data to 2025-12-31.

Two stages, both per station:

1. VERIFY the training recipe. The frozen F3 must be trained exactly the way
   the evaluated F3 was, or the live model is not the model the results
   describe. This refits F3 on fold 20's training window under candidate row
   rules and checks which one reproduces the committed OOF yhat_F3 for fold 20
   BIT-FOR-BIT. Candidates, most likely first:
     mask   "f2"   target and all F2 columns non-null (the common mask)
            "all"  target and all F3 features non-null
     purge  "lt"   training target_time <  fold start
            "le"   training target_time <= fold start
   Stops at the first exact match. If none match, it refuses to freeze.

2. FREEZE with the verified recipe on every origin whose target_time is on or
   before 2025-12-31 23:00 UTC. Saves the booster as text plus a manifest,
   then reloads the saved file and checks it predicts identically.

Outputs (commit these):
    models/F3_<STATION>.txt
    models/manifest_<STATION>.json

Run from the repo root, one station at a time (each takes a few minutes):
    python -m src.freeze --station MY1
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from src.forecast import F2_COLS, F3_PARAMS, F3LightGBM, TARGET_COL, build_feature_cols

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
HORIZON = pd.Timedelta(hours=6)
TRAIN_TARGET_END = pd.Timestamp("2025-12-31 23:00", tz="UTC")   # pre-registered
VERIFY_FOLD = 20
CANDIDATES = [("f2", "lt"), ("f2", "le"), ("all", "lt"), ("all", "le")]


def load_features(station: str) -> pd.DataFrame:
    f = pd.read_parquet(ROOT / "data" / "features" / f"{station}.parquet")
    if not isinstance(f.index, pd.DatetimeIndex) or f.index.tz is None:
        raise TypeError("feature table must be indexed by tz-aware UTC origin")
    return f.sort_index()


def train_rows(frame: pd.DataFrame, feature_cols: list[str], mask: str,
               purge: str, boundary: pd.Timestamp) -> pd.DataFrame:
    """Rows F3 may train on: target known, mask satisfied, target before boundary."""
    target_time = frame.index + HORIZON
    ok_time = target_time < boundary if purge == "lt" else target_time <= boundary
    need = [TARGET_COL] + (list(F2_COLS) if mask == "f2" else feature_cols)
    ok_data = frame[need].notna().all(axis=1).to_numpy()
    return frame[ok_time & ok_data]


def verify_recipe(station: str, frame: pd.DataFrame, feature_cols: list[str]) -> tuple[str, str]:
    oof = pd.read_parquet(ROOT / "data" / "oof" / f"{station}_oof.parquet")
    assert oof["fold"].max() <= 20, "holdout folds in the walk-forward OOF file - stop"
    test = oof[oof["fold"] == VERIFY_FOLD]
    start = pd.DatetimeIndex(test["origin"]).min()
    X_test = frame.loc[pd.DatetimeIndex(test["origin"])]
    want = test["yhat_F3"].to_numpy(dtype="float64")

    print(f"\nverify: refit F3 on fold {VERIFY_FOLD}'s training window "
          f"(fold starts {start}), compare with committed OOF")
    for mask, purge in CANDIDATES:
        tr = train_rows(frame, feature_cols, mask, purge, start)
        m = F3LightGBM(feature_cols)
        m.fit(tr)
        got = m.predict(X_test)
        exact = np.array_equal(got, want)
        print(f"  mask={mask:<3} purge={purge}  train rows {len(tr):>6,}  "
              f"max |diff| {np.max(np.abs(got - want)):.3e}  {'EXACT' if exact else '-'}")
        if exact:
            return mask, purge
    raise SystemExit(
        "\nNo candidate reproduced the committed F3 exactly. NOT freezing.\n"
        "Paste this output to Claude - the training recipe in src/evaluate.py "
        "differs from every candidate, so it needs reading directly."
    )


def git_state() -> dict:
    def run(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"),
            "uncommitted_tracked_changes": bool(run("status", "--porcelain", "--untracked-files=no"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", required=True, choices=["MY1", "KC1", "BEX", "HRL"])
    args = ap.parse_args()
    st = args.station

    frame = load_features(st)
    feature_cols = build_feature_cols(frame)          # strips y_t6 and imputed, asserts
    print(f"station {st}: {len(feature_cols)} F3 features")

    mask, purge = verify_recipe(st, frame, feature_cols)
    print(f"\nrecipe verified: mask={mask}, purge={purge}")

    # --- freeze ------------------------------------------------------------
    boundary = TRAIN_TARGET_END
    tr = train_rows(frame, feature_cols, mask, "le", boundary)   # "le": include the 23:00 target
    assert (tr.index + HORIZON).max() <= TRAIN_TARGET_END
    model = F3LightGBM(feature_cols)
    model.fit(tr)

    MODELS.mkdir(exist_ok=True)
    path = MODELS / f"F3_{st}.txt"
    booster = model._model.booster_
    booster.save_model(str(path))

    # reload check: the saved file must predict exactly what the fitted model does
    reloaded = lgb.Booster(model_file=str(path))
    sample = frame[feature_cols].iloc[-5000:]
    same = np.array_equal(reloaded.predict(sample), model.predict(sample))
    if not same:
        raise SystemExit("reloaded model predicts differently from the fitted one - do not use")

    thr_file = ROOT / "results" / f"{st}_live_threshold.csv"
    thr = float(pd.read_csv(thr_file)["threshold"].iloc[0]) if thr_file.exists() else None

    manifest = {
        "station": st,
        "model": "F3 LightGBM",
        "file": path.name,
        "frozen_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "train_origin_first": str(tr.index.min()),
        "train_origin_last": str(tr.index.max()),
        "train_target_end": str(TRAIN_TARGET_END),
        "train_rows": int(len(tr)),
        "row_rule": {"mask": mask, "verified_purge_in_walk_forward": purge},
        "feature_cols": feature_cols,
        "f3_params": F3_PARAMS,
        "r2_threshold_ug_m3": thr,
        "r2_rule": "flag when resid_mae_24h >= threshold; grey if < 12 scored forecasts in window",
        "versions": {"lightgbm": lgb.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__, "scikit-learn": sklearn.__version__,
                     "python": platform.python_version()},
        "git": git_state(),
        "reload_check": "passed",
    }
    (MODELS / f"manifest_{st}.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nfrozen: {path.relative_to(ROOT)}  ({len(tr):,} training rows, "
          f"origins {tr.index.min()} .. {tr.index.max()})")
    print(f"manifest: models/manifest_{st}.json")
    print(f"R2 threshold recorded: {thr}")
    if manifest["git"]["uncommitted_tracked_changes"]:
        print("WARNING: tracked files had uncommitted changes when this ran - "
              "commit them and re-run so the manifest's git commit is honest")


if __name__ == "__main__":
    main()