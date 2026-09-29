"""src/live_threshold.py - W4b step 2: R2 as it will run live.

Two questions, both on walk-forward folds only (2025 not read):

2a. FROZEN THRESHOLD. W3's R2 flagged the top 20% of resid_mae_24h within
    each test quarter. A live system cannot do that: it would need the whole
    coming quarter. Live, R2 flags an hour when resid_mae_24h is at or above
    a threshold fixed in advance. Here the threshold for fold k is the 80th
    percentile of resid_mae_24h over all origins in folds before k (expanding,
    like the harness), frozen, then applied to fold k. This mirrors live use.

2b. EVENING ORIGINS. AURN publishes a daily batch. If a batch ends at hour H,
    only origins H-5 .. H are genuine forecasts (their targets are not yet
    published). So the live log only ever contains those origin hours. This
    reruns the warning check on that subset, giving the live scoreboard an
    honest backtest expectation.

resid_mae_24h is IMPORTED from src.watcher_features, not re-implemented, so
the live definition cannot drift from the one W3 used. A consistency check
confirms it reproduces W3's R2 flags.

Deployment threshold: the 80th percentile over folds 1-20, saved to results/.

Caveat to carry into the spec: these residuals come from walk-forward F3,
refitted each quarter. The live F3 is trained on more data, so its residuals
may run smaller and the live flag rate may fall below 20%. That drift is
reported on the scoreboard, not corrected.

Run from the repo root:
    python -m src.live_threshold --station MY1
    python -m src.live_threshold --station MY1 --batch-end-hour 20
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.check_warning import level_adjust
from src.watcher_features import residual_features

ROOT = Path(__file__).resolve().parents[1]
Q = 0.80                 # flag the top 20%, matching W3
N_BOOT = 2000
SEED = 42
ALPHA = 0.05
N_EVENING = 6            # origins H-5 .. H


def ratio_ci(err: np.ndarray, flagged: np.ndarray, origin: pd.Series) -> tuple[float, float, float]:
    """Ratio of mean error, flagged / unflagged, with week-block bootstrap CI."""
    t = pd.DatetimeIndex(origin)
    block = np.asarray((t - t.min()) // pd.Timedelta(days=7))
    _, b = np.unique(block, return_inverse=True)
    nb = b.max() + 1
    fl = flagged.astype(float)
    s_f = np.bincount(b, weights=err * fl, minlength=nb)
    n_f = np.bincount(b, weights=fl, minlength=nb)
    s_u = np.bincount(b, weights=err * (1 - fl), minlength=nb)
    n_u = np.bincount(b, weights=1 - fl, minlength=nb)
    point = (s_f.sum() / n_f.sum()) / (s_u.sum() / n_u.sum())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, nb, size=(N_BOOT, nb))
    with np.errstate(invalid="ignore", divide="ignore"):
        br = (s_f[idx].sum(1) / n_f[idx].sum(1)) / (s_u[idx].sum(1) / n_u[idx].sum(1))
    lo, hi = np.nanquantile(br, [ALPHA / 2, 1 - ALPHA / 2])
    return point, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="MY1")
    ap.add_argument("--batch-end-hour", type=int, default=23,
                    help="UTC hour of the last reading in a daily AURN batch")
    args = ap.parse_args()
    st, H = args.station, args.batch_end_hour

    oof = pd.read_parquet(ROOT / "data" / "oof" / f"{st}_oof.parquet")
    routed = pd.read_parquet(ROOT / "data" / "oof" / f"{st}_routed.parquet")

    # --- guards -----------------------------------------------------------
    assert oof["fold"].max() <= 20, "holdout folds in the OOF file - stop, do not read"
    assert routed["fold"].max() <= 20, "holdout folds in the routed file - stop, do not read"

    # --- resid_mae_24h at every OOF origin, W3's own definition -----------
    hist = oof[["origin", "y_true", "yhat_F3"]]
    all_origins = pd.DatetimeIndex(oof["origin"]).sort_values()
    rf = residual_features(hist, all_origins)
    r24 = rf["resid_mae_24h"]
    fold_of = pd.Series(oof["fold"].to_numpy(), index=pd.DatetimeIndex(oof["origin"]))
    fold_of = fold_of[~fold_of.index.duplicated(keep="last")].reindex(all_origins)

    df = routed[["origin", "fold", "y_true", "always_ML", "R2"]].copy()
    df["r24"] = r24.reindex(pd.DatetimeIndex(df["origin"])).to_numpy()

    # --- consistency: does the imported definition reproduce W3's R2? -----
    w3_flag = (df["R2"] != df["always_ML"]).to_numpy()
    rank_flag = (df.groupby("fold")["r24"]
                 .transform(lambda s: s >= s.quantile(Q)).fillna(False).to_numpy(bool))
    agree = (w3_flag == rank_flag).mean()
    print(f"\n=== {st}: W4b step 2 (walk-forward folds {df['fold'].min()}-{df['fold'].max()}, "
          f"{len(df):,} hours) ===")
    print(f"\nConsistency: imported resid_mae_24h reproduces W3's R2 flags on "
          f"{agree:.1%} of hours" + ("" if agree > 0.98 else "   <-- LOW, tell Claude before going on"))

    # --- 2a. frozen, expanding threshold ----------------------------------
    thr_rows = []
    frozen = np.zeros(len(df), dtype=bool)
    for k in sorted(df["fold"].unique()):
        past = r24[(fold_of < k).to_numpy()].dropna()
        thr = float(past.quantile(Q))
        m = (df["fold"] == k).to_numpy()
        frozen[m] = (df.loc[m, "r24"] >= thr).to_numpy()   # NaN -> not flagged
        thr_rows.append({"fold": k, "threshold": thr, "n_past": len(past),
                         "flag_rate": frozen[m].mean(),
                         "r24_missing": df.loc[m, "r24"].isna().mean()})
    thr_tab = pd.DataFrame(thr_rows).set_index("fold")

    deploy_thr = float(r24.dropna().quantile(Q))

    # --- errors -----------------------------------------------------------
    err_raw = (df["y_true"] - df["always_ML"]).abs().to_numpy()
    hour = pd.DatetimeIndex(df["origin"]).hour
    evening_hours = [(H - i) % 24 for i in range(N_EVENING)]
    evening = np.isin(hour, evening_hours)

    results = []
    for subset_name, mask in [("all hours", np.ones(len(df), bool)),
                              (f"evening origins {sorted(evening_hours)} UTC", evening)]:
        sub = df[mask]
        e_raw = err_raw[mask]
        e_lvl = level_adjust(e_raw, sub["always_ML"], sub["fold"])
        for flag_name, fl in [("W3 within-fold top 20%", w3_flag[mask]),
                              ("frozen threshold (live)", frozen[mask])]:
            for scale, e in [("raw", e_raw), ("level", e_lvl)]:
                r, lo, hi = ratio_ci(e, fl, sub["origin"])
                results.append({"subset": subset_name, "flag": flag_name, "scale": scale,
                                "n_hours": int(mask.sum()), "flag_rate": fl.mean(),
                                "ratio": r, "lo": lo, "hi": hi,
                                "detectable": "yes" if lo > 1 else "no"})
    res = pd.DataFrame(results)

    # --- report -----------------------------------------------------------
    pd.set_option("display.width", 160)
    print("\n2a. Frozen threshold per fold (fitted on earlier folds only):")
    print(thr_tab.round(3).to_string())
    print(f"\nDeployment threshold (80th pct of resid_mae_24h, folds 1-20): {deploy_thr:.3f} ug/m3")
    print("\n2b. Warning check, R2, frozen vs W3 flags, all hours vs evening origins:")
    print(res.round(3).to_string(index=False))

    out = ROOT / "results"
    thr_tab.assign(station=st).to_csv(out / f"{st}_live_threshold_by_fold.csv")
    res.assign(station=st, batch_end_hour=H).to_csv(out / f"{st}_live_check.csv", index=False)
    pd.DataFrame([{"station": st, "rule": "R2", "feature": "resid_mae_24h", "quantile": Q,
                   "threshold": deploy_thr, "fitted_on": "OOF folds 1-20 (2018-2024)"}]) \
        .to_csv(out / f"{st}_live_threshold.csv", index=False)
    print(f"\nSaved results/{st}_live_threshold.csv, {st}_live_threshold_by_fold.csv, {st}_live_check.csv\n")


if __name__ == "__main__":
    main()