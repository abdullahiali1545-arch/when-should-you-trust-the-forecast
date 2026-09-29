"""
check_warning.py - W4b step 1: is the watcher's warning worth showing,
and does it beat trivial warnings?

For each rule R0-R3: in walk-forward hours, is F3's error larger in hours the
rule flagged than in hours it did not? Then: is R3's ratio larger than R0's,
R1's and R2's? (Paired week-block bootstrap: every rule is scored on the same
resampled weeks.)

Two error scales:
  raw (default)      F3 absolute error in ug/m3.
  --level-adjusted   F3 absolute error divided by the mean F3 absolute error
                     of its predicted-concentration decile, within the same
                     fold. 1.0 = normal for this level. Removes the Part 8
                     level effect that favours R1 on the raw scale.
                     Pre-registered 2026-09-29 before this mode was run.

Using the test fold's own deciles is not leakage: this is the scoring metric,
and nothing here feeds back into any flag.

Flags are recovered from the routed table. A rule hands a flagged hour to the
fallback, so in a flagged hour rule == always_fallback and rule != always_ML.
R0 (random) is the sanity check: its ratio must be close to 1.

CAVEAT: these are W3's within-fold top-20% flags (coverage-matched ranking
inside each test quarter). A live system cannot reproduce that ranking.

Walk-forward folds only. The 2025 holdout is not read.

Usage:  python src/check_warning.py --station MY1
        python src/check_warning.py --station MY1 --level-adjusted
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RULES = ["R0", "R1", "R2", "R3"]
N_DECILES = 10
N_BOOT = 2000
SEED = 42
ALPHA = 0.05


def level_adjust(err: np.ndarray, pred: pd.Series, fold: pd.Series) -> np.ndarray:
    """Divide each error by the mean error of its predicted-level decile, per fold."""
    out = np.empty_like(err)
    for f in fold.unique():
        m = (fold == f).to_numpy()
        # rank first so tied predictions cannot collapse decile edges
        dec = pd.qcut(pred[m].rank(method="first"), N_DECILES, labels=False).to_numpy()
        e = err[m]
        band_mean = pd.Series(e).groupby(dec).transform("mean").to_numpy()
        assert (band_mean > 0).all(), f"fold {f}: a decile has zero mean error"
        out[m] = e / band_mean
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="MY1")
    ap.add_argument("--level-adjusted", action="store_true")
    args = ap.parse_args()
    scale = "level" if args.level_adjusted else "raw"

    path = ROOT / "data" / "oof" / f"{args.station}_routed.parquet"
    df = pd.read_parquet(path)

    # --- guards -----------------------------------------------------------
    assert df["fold"].max() <= 20, "holdout folds present - stop, do not read"
    assert df[["y_true", "always_ML", "always_fallback"] + RULES].notna().all().all()

    err_f3 = (df["y_true"] - df["always_ML"]).abs().to_numpy()
    err_f0 = (df["y_true"] - df["always_fallback"]).abs().to_numpy()
    if args.level_adjusted:
        err_f3 = level_adjust(err_f3, df["always_ML"], df["fold"])

    # --- week-long blocks -------------------------------------------------
    t = df["origin"]
    block = ((t - t.min()) // pd.Timedelta(days=7)).to_numpy()
    _, b = np.unique(block, return_inverse=True)
    nb = b.max() + 1

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, nb, size=(N_BOOT, nb))   # shared by all rules = paired

    q = [ALPHA / 2, 1 - ALPHA / 2]
    rows, boot_ratio, per_fold = [], {}, {}

    for rule in RULES:
        flagged = (df[rule] != df["always_ML"]).to_numpy()
        assert np.allclose(df.loc[flagged, rule], df.loc[flagged, "always_fallback"]), \
            f"{rule}: a 'flagged' hour does not carry the fallback value"

        fl = flagged.astype(float)
        s_f = np.bincount(b, weights=err_f3 * fl, minlength=nb)
        n_f = np.bincount(b, weights=fl, minlength=nb)
        s_u = np.bincount(b, weights=err_f3 * (1 - fl), minlength=nb)
        n_u = np.bincount(b, weights=1 - fl, minlength=nb)

        mean_f = s_f.sum() / n_f.sum()
        mean_u = s_u.sum() / n_u.sum()
        br = (s_f[idx].sum(1) / n_f[idx].sum(1)) / (s_u[idx].sum(1) / n_u[idx].sum(1))
        boot_ratio[rule] = br
        lo, hi = np.quantile(br, q)

        row = {"rule": rule, "share_flagged": flagged.mean(),
               "f3_err_flagged": mean_f, "f3_err_unflagged": mean_u,
               "ratio": mean_f / mean_u, "ratio_lo": lo, "ratio_hi": hi}
        if scale == "raw":
            row["f0_mae_flagged"] = err_f0[flagged].mean()
        rows.append(row)

        g = pd.DataFrame({"fold": df["fold"], "flag": flagged, "e": err_f3})
        per_fold[rule] = (g[g["flag"]].groupby("fold")["e"].mean()
                          / g[~g["flag"]].groupby("fold")["e"].mean())

    summary = pd.DataFrame(rows).set_index("rule")

    # --- R3 vs each rival: difference in ratio, paired --------------------
    comp = []
    for rival in ["R0", "R1", "R2"]:
        d = boot_ratio["R3"] - boot_ratio[rival]
        lo, hi = np.quantile(d, q)
        point = summary.loc["R3", "ratio"] - summary.loc[rival, "ratio"]
        if lo > 0:
            v = "R3 detectably better"
        elif hi < 0:
            v = "R3 detectably worse"
        else:
            v = "no detectable difference"
        comp.append({"comparison": f"R3 - {rival}", "diff": point,
                     "lo": lo, "hi": hi, "verdict": v})
    comp = pd.DataFrame(comp).set_index("comparison")

    # --- report -----------------------------------------------------------
    unit = "level-adjusted (1.0 = normal for this level)" if scale == "level" else "raw, ug/m3"
    pd.set_option("display.width", 140)
    print(f"\n=== {args.station}: W4b warning check, {unit} (walk-forward folds "
          f"{df['fold'].min()}-{df['fold'].max()}, {len(df):,} hours, {nb} week-blocks) ===")
    print("\nPer-fold ratio (F3 error flagged / unflagged):")
    print(pd.DataFrame(per_fold).round(3).to_string())
    print("\nOverall, per rule (95% CI on the ratio):")
    print(summary.round(3).to_string())
    print("\nSanity: R0 (random) ratio should be close to 1 and its CI should include 1.")
    print("\nDoes the watcher's warning beat the trivial warnings? (paired, 95% CI)")
    print(comp.round(3).to_string())

    suffix = "_level" if scale == "level" else ""
    out_dir = ROOT / "results"
    summary.assign(station=args.station, scale=scale, n_hours=len(df), n_blocks=nb,
                   flag_source="W3 within-fold top-20% (not live-reproducible)") \
        .to_csv(out_dir / f"{args.station}_warning_check{suffix}.csv")
    comp.assign(station=args.station, scale=scale) \
        .to_csv(out_dir / f"{args.station}_warning_check_comparisons{suffix}.csv")
    print(f"\nSaved results/{args.station}_warning_check{suffix}.csv and "
          f"results/{args.station}_warning_check_comparisons{suffix}.csv\n")


if __name__ == "__main__":
    main()