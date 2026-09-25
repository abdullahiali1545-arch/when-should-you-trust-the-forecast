"""
W3.4 - Routing rules and the headline comparison (H3).

Pre-registered 2026-09-22 in the PROJECT_SPEC.md changelog:
  fallback      F0 (persistence)
  share routed  q = 0.20 in every test fold, fixed in advance, not tuned
  coverage      every rule routes the top 20% of hours by its own score,
                ranked within each test fold
  scores        R0 random (seed 42) | R1 yhat_F3 | R2 resid_mae_24h | R3 p_unreliable

Division of labour
  Boilerplate (loading, joining, scoring, saving): written for you.
  route(): YOURS. It is the routing policy and it is viva-critical.

Run:  python -m src.routing              (primary, stratified labels)
      python -m src.routing --relative   (Part 8 robustness check)
      python -m src.routing --holdout    (2025 holdout, folds 21-24; primary labels only)

--holdout (PROJECT_SPEC changelog 2026-09-24): routes all folds exactly as the
walk-forward run did, checks folds 3-20 are identical to the committed routed
file, then scores folds 21-24 on their own. Prints NO 2025 numbers; they are
read once, by src.bootstrap --holdout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.watcher import build_all_fold_features

# ---------------------------------------------------------------- paths
# --relative switches to the Part 8 robustness labels (watcher_rel file)
# --holdout switches to the 2025 holdout files (primary labels only)
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
OOF_PATH = REPO_ROOT / "data/oof" / f"{STATION}_oof{'_holdout' if HOLDOUT else ''}.parquet"  # [HOLDOUT]
WATCHER_PATH = REPO_ROOT / "data/oof" / f"{STATION}_watcher{SUFFIX}.parquet"
FEATURES_PATH = REPO_ROOT / "data/features" / f"{STATION}.parquet"
OUT_PATH = REPO_ROOT / "data/oof" / f"{STATION}_routed{SUFFIX}.parquet"
TABLE_PATH = REPO_ROOT / "results" / f"{STATION}_routing_headline{SUFFIX}.csv"
COMMITTED_ROUTED_PATH = REPO_ROOT / "data/oof" / f"{STATION}_routed.parquet"   # [HOLDOUT]
N_WALK_FORWARD_FOLDS = 20                                                      # [HOLDOUT]

# ---------------------------------------------------------- pre-registered
Q = 0.20
SEED = 42
TRUTH = "y_true"
ML = "yhat_F3"
FALLBACK = "yhat_F0"


# =====================================================================
#  YOUR FUNCTION
# =====================================================================
def route(score: pd.Series, fold: pd.Series, q: float) -> pd.Series:
    """Decide which hours are sent to the fallback.

    Inputs
      score : float Series. Higher = more distrust. No NaNs.
      fold  : int Series of fold ids, same index as `score`.
      q     : share of hours to route in each fold (0.20).

    Returns
      bool Series, same index as `score`. True = publish the fallback.

    Must
      - Work on each fold separately.
      - In a fold with n_k rows, mark exactly k = int(round(q * n_k)) rows
        as True: the k rows with the highest score.
      - Break ties the same way every run, so a rerun gives the same answer.

    Must not
      - Read y_true or any forecast. The decision uses the score only.
      - Use one cutoff across all folds.
      - Let one fold's scores affect another fold's decisions.
    """
    # Group by fold, so each quarter is handled on its own and no fold's
    # scores can influence another fold's decisions.
    by_fold = score.groupby(fold)

    # n_k: every row gets the size of its own fold (same order as the input).
    n_k = by_fold.transform("size")

    # k: how many hours to route in that fold. round() here must match the
    # rule _check_counts uses, or the count check will fail.
    k = (q * n_k).round().astype(int)

    # Rank inside each fold, highest score = rank 1 (most distrusted).
    # method="first" breaks ties by row order, so reruns give the same answer.
    rank = by_fold.rank(ascending=False, method="first")

    # Route an hour if it is among the k most distrusted in its fold.
    # Nothing here reads y_true or any forecast - only the score.
    return (rank <= k).astype(bool)


# =====================================================================
#  Checks on route()
# =====================================================================
def _toy_test() -> None:
    """Fast check on route() before the slow data load."""
    score = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                       10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
    fold = pd.Series([1] * 10 + [2] * 10)
    got = route(score, fold, 0.20)

    assert got.dtype == bool, "route() must return a bool Series"
    assert got.index.equals(score.index), "route() must keep the input index"
    # fold 1: highest scores are rows 8, 9; fold 2: rows 10, 11
    assert set(got[got].index) == {8, 9, 10, 11}, (
        f"expected rows [8, 9, 10, 11] routed, got {sorted(got[got].index)}"
    )
    print("toy test: PASS")


def _check_counts(routed: pd.Series, fold: pd.Series, q: float, name: str) -> None:
    """Every fold must route exactly round(q * n_k) rows."""
    for k, grp in routed.groupby(fold):
        expected = int(round(q * len(grp)))
        assert int(grp.sum()) == expected, (
            f"{name}, fold {k}: routed {int(grp.sum())}, expected {expected}"
        )


# =====================================================================
#  Boilerplate
# =====================================================================
def load_inputs() -> pd.DataFrame:
    """One row per scored hour: truth, both forecasts, and all four scores."""
    oof = pd.read_parquet(OOF_PATH)
    watcher = pd.read_parquet(WATCHER_PATH)
    feats = pd.read_parquet(FEATURES_PATH)

    df = oof.merge(
        watcher[["origin", "fold", "label", "p_unreliable"]],
        on=["origin", "fold"], how="left", validate="one_to_one",
    )
    assert len(df) == len(oof), "join changed the row count"

    # R2's score, rebuilt exactly as the watcher saw it (six-hour rule enforced)
    print("building watcher features for R2's score (slow)...")
    cache = build_all_fold_features(oof, feats)
    resid = pd.concat([cache[k]["resid_mae_24h"] for k in sorted(cache)])
    assert resid.index.is_unique, "resid_mae_24h has duplicate origins"
    df["resid_mae_24h"] = df["origin"].map(resid)

    # One common set of hours for every rule, so the comparison is like for like
    needed = [TRUTH, ML, FALLBACK, "p_unreliable", "resid_mae_24h"]
    before = len(df)
    df = df.dropna(subset=needed).reset_index(drop=True)
    print(f"scored rows: {len(df):,} of {before:,} "
          f"({before - len(df):,} dropped: burn-in folds or missing inputs)")
    print(f"folds scored: {sorted(df['fold'].unique())}")
    return df


def make_scores(df: pd.DataFrame) -> dict[str, pd.Series]:
    rng = np.random.default_rng(SEED)
    return {
        "R0": pd.Series(rng.random(len(df)), index=df.index),
        "R1": df[ML],
        "R2": df["resid_mae_24h"],
        "R3": df["p_unreliable"],
    }


def score_rules(df: pd.DataFrame, scores: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[TRUTH]
    err_ml = (y - df[ML]).abs()
    err_fb = (y - df[FALLBACK]).abs()
    g = err_ml - err_fb          # > 0 means persistence beat F3 on that hour
    n = len(df)

    published = pd.DataFrame({
        "station": df["station"], "origin": df["origin"], "fold": df["fold"],
        TRUTH: y, "always_ML": df[ML], "always_fallback": df[FALLBACK],
    })
    routed_masks = {"always_ML": pd.Series(False, index=df.index),
                    "always_fallback": pd.Series(True, index=df.index)}

    for name, s in scores.items():
        mask = route(s, df["fold"], Q)
        _check_counts(mask, df["fold"], Q, name)
        routed_masks[name] = mask
        published[name] = df[FALLBACK].where(mask, df[ML])

    rows = []
    for name, mask in routed_masks.items():
        err = (y - published[name]).abs()
        mae = float(err.mean())
        # the g-identity: routed MAE = MAE_F3 - (1/n) * sum of g over routed hours
        assert np.isclose(mae, err_ml.mean() - g[mask].sum() / n), f"{name}: identity failed"
        rows.append({
            "rule": name,
            "MAE": mae,
            "RMSE": float(np.sqrt((err ** 2).mean())),
            "share_routed": float(mask.mean()),
            "mean_g_routed": float(g[mask].mean()) if mask.any() else np.nan,
            "share_routed_g_pos": float((g[mask] > 0).mean()) if mask.any() else np.nan,
        })
    table = pd.DataFrame(rows).set_index("rule")
    table["MAE_vs_always_ML"] = table["MAE"] - table.loc["always_ML", "MAE"]
    return table, published


# =====================================================================
#  [HOLDOUT] Holdout branch
# =====================================================================
def main_holdout(df: pd.DataFrame) -> None:
    # Scores for ALL folds, in the same row order as the walk-forward run, so
    # R0's seed-42 draws for folds 3-20 are the same numbers as before.
    scores = make_scores(df)
    _, published_all = score_rules(df, scores)

    # Regression check: folds 3-20 routed exactly as in the committed run.
    if not COMMITTED_ROUTED_PATH.exists():
        raise FileNotFoundError(f"{COMMITTED_ROUTED_PATH} not found; nothing to compare against")
    committed = pd.read_parquet(COMMITTED_ROUTED_PATH).reset_index(drop=True)
    new = published_all.loc[published_all["fold"].le(N_WALK_FORWARD_FOLDS)].reset_index(drop=True)
    pd.testing.assert_frame_equal(new, committed, check_exact=True)
    print(f"\nregression check PASSED: folds up to {N_WALK_FORWARD_FOLDS} identical "
          f"to {COMMITTED_ROUTED_PATH.name} ({len(committed):,} rows)")

    # The holdout on its own: folds 21-24 only. Routing is per fold, so these
    # decisions are the same as in published_all; scoring them alone keeps
    # 2025 from being averaged in with the walk-forward years.
    df_h = df.loc[df["fold"].gt(N_WALK_FORWARD_FOLDS)]
    scores_h = {name: s.loc[df_h.index] for name, s in scores.items()}
    table_h, published_h = score_rules(df_h, scores_h)

    folds = sorted(published_h["fold"].unique())
    expected = list(range(N_WALK_FORWARD_FOLDS + 1, N_WALK_FORWARD_FOLDS + 5))
    if folds != expected:
        raise AssertionError(f"holdout routed folds are {folds}, expected {expected}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    published_h.to_parquet(OUT_PATH)
    table_h.to_csv(TABLE_PATH)
    print(f"holdout rows routed and saved: {len(published_h):,} (folds {folds})")
    print(f"written: {OUT_PATH}")
    print(f"written: {TABLE_PATH}")
    print("No 2025 numbers printed. They are read once, by src.bootstrap --holdout.")


def main() -> None:
    _toy_test()
    df = load_inputs()

    if HOLDOUT:                                                                # [HOLDOUT]
        main_holdout(df)
        return

    table, published = score_rules(df, make_scores(df))

    print("\n" + "=" * 72)
    print(f"W3.4 headline at {STATION}, q = {Q}, fallback = F0, labels = {LABEL_MODE}")
    print("=" * 72)
    print(table.round(4).to_string())
    print("\nmean_g_routed > 0 means the rule picked hours where persistence beat F3.")
    print("No difference counts as a win until the W3.6 bootstrap interval excludes zero.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    published.to_parquet(OUT_PATH)
    table.to_csv(TABLE_PATH)
    print(f"\nwritten: {OUT_PATH}")
    print(f"written: {TABLE_PATH}")


if __name__ == "__main__":
    main()