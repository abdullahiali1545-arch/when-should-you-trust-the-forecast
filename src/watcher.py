"""src/watcher.py — the watcher classifier (W3.3).

W3.1 built the answer key: was F3 unreliable at this hour?
W3.2 built the exam paper: what may be looked at, at time t, before the truth.
This trains a model to sit the exam.

THREE LAYERS OF LAG, each for the same reason one level up:

    F3                 fitted on its fold's training window       (W2)
    label thresholds   fitted on folds strictly before k          (W3.1)
    watcher            trained on folds 2..k-1                    (here)

Fold 1 has no labels, because thresholds need an earlier fold and none exists.
So the watcher's training set starts at fold 2, and its first scoreable fold is
3. Eighteen folds are scored, not twenty. That burn-in is reported rather than
engineered around: shortening it would mean letting a fold inform the standard
it is judged against.

The output is a PROBABILITY, not a decision. Turning it into a routing call
needs a threshold, and that threshold is a policy choice fitted on training
folds in W3.4. Keeping them apart means PR-AUC measures the ranking and the
threshold cannot be quietly tuned to flatter it.

PR-AUC, not accuracy. The positive rate averages 18.6%, so a model that always
says "trust" scores 81% while detecting nothing. The no-skill baseline for
PR-AUC is the fold's own positive rate, and that is reported alongside every
score.

Run from the repo root:
    python -m src.watcher              # walk-forward, folds 1-20 (unchanged)
    python -m src.watcher --holdout    # reads the 24-fold OOF file (2025 = folds 21-24)

--holdout (PROJECT_SPEC changelog 2026-09-24): checks folds 1-20 are identical
to the committed watcher file, saves to separate files, and prints NO 2025
numbers. Holdout results are read once, by src.bootstrap --holdout.

[W4a] Any of these also takes --station KC1 | BEX | HRL (default MY1).
    e.g. python -m src.watcher --station KC1
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

from src.labels import label_walk_forward
from src.watcher_features import build_watcher_features

# [HOLDOUT] Only affects the paths used when this file is run directly.
# Importing build_all_fold_features from elsewhere is unaffected.
HOLDOUT = "--holdout" in sys.argv
N_WALK_FORWARD_FOLDS = 20
_SUFFIX = "_holdout" if HOLDOUT else ""

REPO_ROOT = Path(__file__).resolve().parents[1]
# [W4a] --station picks the station; default MY1 keeps every earlier command unchanged.
STATIONS = ("MY1", "KC1", "BEX", "HRL")


def _station_from_argv(default: str = "MY1") -> str:
    if "--station" not in sys.argv:
        return default
    i = sys.argv.index("--station")
    if i + 1 >= len(sys.argv):
        raise SystemExit("--station needs a value, e.g. --station KC1")
    s = sys.argv[i + 1].upper()
    if s not in STATIONS:
        raise SystemExit(f"unknown station {s!r}; choose from {STATIONS}")
    return s


STATION = _station_from_argv()
if "--holdout" in sys.argv and STATION != "MY1":
    raise SystemExit("the 2025 holdout is pre-registered for MY1 only (PROJECT_SPEC 2026-09-24)")
OOF_PATH = REPO_ROOT / "data/oof" / f"{STATION}_oof{_SUFFIX}.parquet"           # [HOLDOUT]
FEATURES_PATH = REPO_ROOT / "data/features" / f"{STATION}.parquet"
OUT_PATH = REPO_ROOT / "data/oof" / f"{STATION}_watcher{_SUFFIX}.parquet"       # [HOLDOUT]
COMMITTED_PATH = REPO_ROOT / "data/oof" / f"{STATION}_watcher.parquet"          # [HOLDOUT]
DIAG_HOLDOUT_PATH = REPO_ROOT / "results" / f"{STATION}_watcher_diag_holdout.csv"  # [HOLDOUT]

# Pre-registered 2026-09-18. NOT tuned against PR-AUC. Deliberately smaller
# than F3: the training set is a fraction of F3's and carries ~20% positives,
# so a 500-tree model would memorise rather than generalise.
WATCHER_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=15,
    max_depth=4,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    deterministic=True,
    n_jobs=1,
    verbose=-1,
)

FIRST_LABELLED_FOLD = 2      # fold 1 has no labels
FIRST_SCORED_FOLD = 3        # needs at least one labelled fold to train on


def fold_training_index(feats: pd.DataFrame, fold_start: pd.Timestamp) -> pd.DatetimeIndex:
    """Origins available for training before this fold begins.

    Defines the reference distribution for the Wasserstein family. Expanding,
    matching the harness and the label thresholds.
    """
    return feats.index[feats.index < fold_start]


def build_all_fold_features(
    oof: pd.DataFrame,
    feats: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """Watcher features for every fold, built once and cached.

    Rebuilding earlier folds' features inside each fold's training loop would be
    quadratic. Caching is safe because a fold's features depend only on that
    fold's own training window, which never changes.
    """
    cache: dict[int, pd.DataFrame] = {}
    for k in sorted(oof["fold"].unique()):
        if k < FIRST_LABELLED_FOLD:
            continue
        fold = oof[oof["fold"].eq(k)]
        origins = pd.DatetimeIndex(fold["origin"]).sort_values()
        train_index = fold_training_index(feats, origins.min())
        cache[k] = build_watcher_features(fold, oof, feats, train_index)
        print(f"  fold {k:>2}: {cache[k].shape[0]:>5,} rows x {cache[k].shape[1]} features")
    return cache


def run_watcher(
    oof: pd.DataFrame,
    feats: pd.DataFrame,
    labels: pd.Series,
    params: dict = WATCHER_PARAMS,
    quiet_above: int | None = None,             # [HOLDOUT] don't print folds above this
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Walk-forward watcher. Train on folds 2..k-1, predict fold k.

    Returns
    -------
    proba : P(unreliable) per origin, NaN for unscored folds
    diag  : one row per fold — n_train, n_test, positive_rate, pr_auc, baseline
    imp   : mean feature importance across folds, for the W4 ablation
    """
    # labels is aligned to oof's positional index; attach it as a column so it
    # travels with the fold and origin identifiers.
    work = oof[["fold", "origin"]].copy()
    work["label"] = labels.to_numpy()

    print("\nbuilding watcher features per fold...")
    feat_cache = build_all_fold_features(oof, feats)

    proba = pd.Series(np.nan, index=oof.index, dtype=float)
    diag_rows: list[dict] = []
    importances: list[pd.Series] = []

    folds = sorted(oof["fold"].unique())
    print("\ntraining...")
    for k in folds:
        if k < FIRST_SCORED_FOLD:
            continue

        # ---- training set: folds 2..k-1, labelled rows only -----------------
        train_parts = []
        for j in folds:
            if j < FIRST_LABELLED_FOLD or j >= k:
                continue
            Xj = feat_cache[j]
            yj = work[work["fold"].eq(j)].set_index("origin")["label"]
            yj = yj.reindex(Xj.index)
            keep = yj.notna()
            if keep.any():
                train_parts.append((Xj[keep.to_numpy()], yj[keep]))

        if not train_parts:
            continue

        X_train = pd.concat([p[0] for p in train_parts])
        y_train = pd.concat([p[1] for p in train_parts]).astype(int)

        # A fold of one class would train a degenerate model; skip and report.
        if y_train.nunique() < 2:
            diag_rows.append({"fold": k, "n_train": len(y_train), "n_test": 0,
                              "positive_rate": np.nan, "pr_auc": np.nan,
                              "baseline": np.nan, "lift": np.nan})
            continue

        # ---- test fold -------------------------------------------------------
        X_test = feat_cache[k]
        y_test = work[work["fold"].eq(k)].set_index("origin")["label"].reindex(X_test.index)
        keep = y_test.notna()
        X_test, y_test = X_test[keep.to_numpy()], y_test[keep].astype(int)

        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)

        p = model.predict_proba(X_test)[:, 1]

        # Write back by origin, never by position: X_test was filtered.
        origin_to_pos = pd.Series(oof.index.to_numpy(),
                                  index=pd.DatetimeIndex(oof["origin"]))
        proba.loc[origin_to_pos.reindex(X_test.index).to_numpy()] = p

        base = float(y_test.mean())
        pr = float(average_precision_score(y_test, p)) if y_test.nunique() > 1 else np.nan

        diag_rows.append({
            "fold": k,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "positive_rate": base,
            "pr_auc": pr,
            "baseline": base,
            "lift": pr / base if base > 0 else np.nan,
        })
        importances.append(pd.Series(model.feature_importances_, index=X_train.columns))

        # [HOLDOUT] Holdout folds train and score as normal, but their numbers
        # are not printed here.
        if quiet_above is not None and k > quiet_above:
            print(f"  fold {k:>2}: train {len(y_train):>6,}  test {len(y_test):>5,}  "
                  f"(holdout fold: scores saved, not printed)")
        else:
            print(f"  fold {k:>2}: train {len(y_train):>6,}  test {len(y_test):>5,}  "
                  f"PR-AUC {pr:.3f}  baseline {base:.3f}  lift {pr / base:.2f}")

    diag = pd.DataFrame(diag_rows).set_index("fold")
    imp = (pd.concat(importances, axis=1).mean(axis=1)
           .sort_values(ascending=False).to_frame("importance"))
    return proba, diag, imp


# ---------------------------------------------------------------------------
# [HOLDOUT] The holdout branch of the script
# ---------------------------------------------------------------------------
def main_holdout(oof: pd.DataFrame, feats: pd.DataFrame) -> None:
    labels, _ = label_walk_forward(oof)

    # Label counts for folds 1-20 only. The 2025 positive rate is a property of
    # F3's 2025 errors, so it stays unprinted until the one reading.
    wf = oof["fold"].le(N_WALK_FORWARD_FOLDS).to_numpy()
    lab_wf = pd.Series(labels.to_numpy()[wf])
    print(f"labels (folds 1-{N_WALK_FORWARD_FOLDS} only): {lab_wf.notna().sum():,} "
          f"of {len(lab_wf):,} rows, positive rate {lab_wf.mean():.4f}")

    proba, diag, _ = run_watcher(oof, feats, labels, quiet_above=N_WALK_FORWARD_FOLDS)

    out = oof[["station", "origin", "fold"]].copy()
    out["label"] = labels.to_numpy()
    out["p_unreliable"] = proba.to_numpy()

    # Regression check: folds 1-20 must equal the committed watcher file exactly.
    if not COMMITTED_PATH.exists():
        raise FileNotFoundError(f"{COMMITTED_PATH} not found; nothing to compare against")
    committed = pd.read_parquet(COMMITTED_PATH).reset_index(drop=True)
    new = out.loc[out["fold"].le(N_WALK_FORWARD_FOLDS)].reset_index(drop=True)
    pd.testing.assert_frame_equal(new, committed, check_exact=True)
    print(f"\nregression check PASSED: folds 1-{N_WALK_FORWARD_FOLDS} identical "
          f"to {COMMITTED_PATH.name} ({len(committed):,} rows)")

    hold_diag = diag.loc[diag.index > N_WALK_FORWARD_FOLDS]
    expected = list(range(N_WALK_FORWARD_FOLDS + 1, N_WALK_FORWARD_FOLDS + 5))
    if list(hold_diag.index) != expected:
        raise AssertionError(f"holdout watcher folds are {list(hold_diag.index)}, expected {expected}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIAG_HOLDOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH)
    hold_diag.to_csv(DIAG_HOLDOUT_PATH)
    print(f"written: {OUT_PATH}")
    print(f"written: {DIAG_HOLDOUT_PATH}")
    print("No 2025 numbers printed. They are read once, by src.bootstrap --holdout.")


if __name__ == "__main__":
    oof = pd.read_parquet(OOF_PATH)
    feats = pd.read_parquet(FEATURES_PATH)

    if HOLDOUT:                                          # [HOLDOUT]
        main_holdout(oof, feats)
        sys.exit(0)

    labels, label_diag = label_walk_forward(oof)
    print(f"labels: {labels.notna().sum():,} of {len(labels):,} rows, "
          f"positive rate {labels.mean():.4f}")

    proba, diag, imp = run_watcher(oof, feats, labels)

    print("\n" + "=" * 72)
    print(diag.round(4).to_string())

    scored = diag.dropna(subset=["pr_auc"])
    print(f"\nfolds scored: {len(scored)} of {len(oof['fold'].unique())}")
    print(f"mean PR-AUC:  {scored['pr_auc'].mean():.4f}")
    print(f"mean baseline:{scored['baseline'].mean():.4f}")
    print(f"mean lift:    {scored['lift'].mean():.3f}x")
    print(f"folds beating baseline: {(scored['pr_auc'] > scored['baseline']).sum()}"
          f" of {len(scored)}")

    print("\ntop 12 features by mean importance:")
    print(imp.head(12).round(1).to_string())

    # Which families carry the model. This is the question the W4 ablation
    # exists to answer properly; this is a first look, not the answer.
    fam = imp.copy()
    fam["family"] = np.select(
        [fam.index.str.startswith("wass_"),
         fam.index.str.startswith("vol_"),
         fam.index.str.startswith("disagree"),
         fam.index.str.startswith("resid_")],
        ["distribution", "volatility", "disagreement", "residual"],
        default="other",
    )
    print("\nimportance by family (share):")
    share = fam.groupby("family")["importance"].sum()
    print((share / share.sum()).sort_values(ascending=False).round(3).to_string())

    out = oof[["station", "origin", "fold"]].copy()
    out["label"] = labels.to_numpy()
    out["p_unreliable"] = proba.to_numpy()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH)
    print(f"\nwritten: {OUT_PATH}")