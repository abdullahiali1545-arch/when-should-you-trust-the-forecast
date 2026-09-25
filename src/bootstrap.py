"""
W3.6 - Block bootstrap confidence intervals on every headline difference.

Pre-registered in PROJECT_SPEC.md Part 10: week-long blocks, >= 1000 resamples,
a difference counts only if its interval excludes zero.

Settings used here:
  block length   168 hours (one week)
  resamples      2000
  interval       percentile, 2.5% and 97.5%
  seed           42
  pairing        both systems scored on the SAME resampled weeks

Why blocks: hourly forecast errors come in runs, so resampling single hours
would pretend there are 31,572 independent observations when there are really
about 190 independent weeks. That mistake makes intervals far too narrow.

Run:  python -m src.bootstrap              (primary labels)
      python -m src.bootstrap --relative   (Part 8 robustness labels)
      python -m src.bootstrap --holdout    (2025 holdout: THE ONE READING)

Reads the published forecasts written by src/routing.py, so it never refits a
model and never touches y beyond scoring.

--holdout (PROJECT_SPEC changelog 2026-09-24): this is where the holdout is read,
once. It prints the bootstrap table, the routing table and the watcher PR-AUC
for folds 21-24 together.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HOLDOUT = "--holdout" in sys.argv                                              # [HOLDOUT]
if HOLDOUT and "--relative" in sys.argv:                                       # [HOLDOUT]
    raise SystemExit("--holdout is pre-registered for primary labels only; drop --relative")
LABEL_MODE = "relative" if "--relative" in sys.argv else "stratified"
if HOLDOUT:                                                                    # [HOLDOUT]
    SUFFIX = "_holdout"
else:
    SUFFIX = "" if LABEL_MODE == "stratified" else "_rel"

REPO_ROOT = Path(__file__).resolve().parents[1]
STATION = "MY1"
ROUTED_PATH = REPO_ROOT / "data/oof" / f"{STATION}_routed{SUFFIX}.parquet"
OUT_PATH = REPO_ROOT / "results" / f"{STATION}_bootstrap{SUFFIX}.csv"
ROUTING_TABLE_HOLDOUT = REPO_ROOT / "results" / f"{STATION}_routing_headline_holdout.csv"  # [HOLDOUT]
WATCHER_DIAG_HOLDOUT = REPO_ROOT / "results" / f"{STATION}_watcher_diag_holdout.csv"      # [HOLDOUT]
HOLDOUT_FOLDS = [21, 22, 23, 24]                                               # [HOLDOUT]

TRUTH = "y_true"
BLOCK_HOURS = 168
N_RESAMPLES = 2000
SEED = 42
ALPHA = 0.05

# Every difference the write-up discusses. (A, B) reports mean|err_A| - mean|err_B|,
# so a negative value means A is better than B.
COMPARISONS = [
    ("R3", "always_ML"),
    ("R2", "always_ML"),
    ("R1", "always_ML"),
    ("R0", "always_ML"),
    ("always_fallback", "always_ML"),
    ("R3", "R1"),
    ("R3", "R2"),
    ("R3", "R0"),
]


# =====================================================================
#  Core function
# =====================================================================
def block_bootstrap_diff(
    err_a: np.ndarray,
    err_b: np.ndarray,
    block_id: np.ndarray,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap the difference in MAE between two systems, by whole weeks.

    Inputs
      err_a, err_b : absolute errors of the two systems. Same hours, same order.
      block_id     : week number for each hour (equal length, ints).
      n_resamples  : how many resampled datasets to build.
      rng          : numpy Generator, already seeded.

    Returns
      array of length n_resamples. Entry i = mean(err_a) - mean(err_b) on
      resample i.

    Draws WHOLE blocks with replacement, as many blocks as there are in the
    data, and scores A and B on the SAME drawn blocks each time. Resampling
    individual hours would break the runs of correlated errors and make the
    interval far too narrow.
    """
    # Row positions belonging to each block, worked out once rather than inside
    # the loop. blocks[i] is an array of the row positions in week i.
    labels, inverse = np.unique(block_id, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(len(labels)))
    blocks = np.split(order, starts[1:])
    n_blocks = len(blocks)

    out = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        # Draw n_blocks whole weeks with replacement, so the resample is about
        # the size of the original data. Some weeks appear twice, some not at all.
        picked = rng.integers(0, n_blocks, size=n_blocks)
        rows = np.concatenate([blocks[j] for j in picked])

        # The SAME rows score both systems. That pairing is what removes the
        # shared difficulty of a hard week from the difference.
        out[i] = err_a[rows].mean() - err_b[rows].mean()

    return out


def _toy_test() -> None:
    """Fast check before the real run."""
    rng = np.random.default_rng(0)
    n = 1000
    block = np.repeat(np.arange(10), 100)
    a = np.ones(n)
    b = np.ones(n) * 1.5
    d = block_bootstrap_diff(a, b, block, 200, rng)
    assert d.shape == (200,), f"expected 200 values, got {d.shape}"
    assert np.allclose(d, -0.5), "constant errors must give a constant difference of -0.5"

    # with real variation the spread must be non-zero and the mean near the truth
    a2 = rng.normal(3.5, 1.0, n)
    b2 = rng.normal(3.0, 1.0, n)
    d2 = block_bootstrap_diff(a2, b2, block, 500, rng)
    assert d2.std() > 0, "every resample identical - blocks are not being redrawn"
    assert abs(d2.mean() - (a2.mean() - b2.mean())) < 0.3, "resamples are biased"
    print("toy test: PASS")


# =====================================================================
#  Boilerplate
# =====================================================================
def load_published() -> tuple[pd.DataFrame, np.ndarray]:
    """Published forecasts per rule, plus a week number for every hour."""
    df = pd.read_parquet(ROUTED_PATH).sort_values("origin").reset_index(drop=True)

    # [HOLDOUT] The holdout file must contain folds 21-24 and nothing else.
    if HOLDOUT:
        folds = sorted(df["fold"].unique())
        if folds != HOLDOUT_FOLDS:
            raise AssertionError(f"holdout file has folds {folds}, expected {HOLDOUT_FOLDS}")

    origin = pd.DatetimeIndex(df["origin"])
    hours_since_start = (origin - origin.min()) // pd.Timedelta("1h")
    block_id = (hours_since_start // BLOCK_HOURS).to_numpy()
    print(f"{len(df):,} hours, {len(np.unique(block_id)):,} week-long blocks, "
          f"labels = {LABEL_MODE}{' (HOLDOUT 2025)' if HOLDOUT else ''}")
    # 'fold' is not a forecast; keep it out of the systems list below.
    return df, block_id


def main() -> None:
    _toy_test()
    df, block_id = load_published()
    rng = np.random.default_rng(SEED)

    systems = [c for c in df.columns
               if c not in ("station", "origin", "fold", TRUTH)]
    err = {s: (df[TRUTH] - df[s]).abs().to_numpy() for s in systems}

    print("\nMAE per system:")
    for s in systems:
        print(f"  {s:16s} {err[s].mean():.4f}")

    rows = []
    for a, b in COMPARISONS:
        d = block_bootstrap_diff(err[a], err[b], block_id, N_RESAMPLES, rng)
        lo, hi = np.percentile(d, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
        rows.append({
            "A": a, "B": b,
            "diff_MAE": float(err[a].mean() - err[b].mean()),
            "ci_low": float(lo), "ci_high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "verdict": ("A worse" if lo > 0 else
                        "A better" if hi < 0 else
                        "no detectable difference"),
        })

    table = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print(f"W3.6 block bootstrap at {STATION}: {N_RESAMPLES} resamples, "
          f"{BLOCK_HOURS}h blocks, seed {SEED}"
          f"{'  —  HOLDOUT 2025 (folds 21-24)' if HOLDOUT else ''}")
    print("=" * 78)
    print(table.round(4).to_string(index=False))
    print("\ndiff_MAE = MAE(A) - MAE(B). Negative means A is better.")
    print("A difference counts only if the interval excludes zero.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_PATH, index=False)
    print(f"\nwritten: {OUT_PATH}")

    # [HOLDOUT] The rest of the one reading: routing table and watcher PR-AUC.
    if HOLDOUT:
        print("\n" + "=" * 78)
        print("HOLDOUT routing table (folds 21-24)")
        print("=" * 78)
        print(pd.read_csv(ROUTING_TABLE_HOLDOUT, index_col=0).round(4).to_string())

        diag = pd.read_csv(WATCHER_DIAG_HOLDOUT, index_col=0)
        print("\n" + "=" * 78)
        print("HOLDOUT watcher PR-AUC (folds 21-24)")
        print("=" * 78)
        print(diag.round(4).to_string())
        print(f"\nmean PR-AUC {diag['pr_auc'].mean():.4f}   "
              f"mean baseline {diag['baseline'].mean():.4f}   "
              f"mean lift {diag['lift'].mean():.3f}x   "
              f"folds beating baseline {(diag['pr_auc'] > diag['baseline']).sum()} of {len(diag)}")
        print(f"\nHoldout read {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
              "Record this date in PROJECT_SPEC.md. These numbers now stand.")


if __name__ == "__main__":
    main()