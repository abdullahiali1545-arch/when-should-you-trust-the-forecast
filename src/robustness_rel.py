"""
Part 8 robustness check, step 1: relative-error labels -> retrained watcher.

Same watcher features, hyperparameters and seed as W3.3. Only the labels change.
Writes data/oof/MY1_watcher_rel.parquet, which routing.py and risk_coverage.py
read when run with --relative.

Run:  python -m src.robustness_rel
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.labels import label_walk_forward
from src.watcher import run_watcher

REPO_ROOT = Path(__file__).resolve().parents[1]
STATION = "MY1"
OOF_PATH = REPO_ROOT / "data/oof" / f"{STATION}_oof.parquet"
FEATURES_PATH = REPO_ROOT / "data/features" / f"{STATION}.parquet"
OUT_PATH = REPO_ROOT / "data/oof" / f"{STATION}_watcher_rel.parquet"


def main() -> None:
    oof = pd.read_parquet(OOF_PATH)
    feats = pd.read_parquet(FEATURES_PATH)

    labels, label_diag = label_walk_forward(oof, mode="relative")
    rates = label_diag["positive_rate"].dropna()
    print("relative-error labels")
    print(f"  labelled: {labels.notna().sum():,} of {len(labels):,} rows")
    print(f"  per-fold positive rate: min {rates.min():.3f}  "
          f"max {rates.max():.3f}  mean {rates.mean():.3f}")

    # How different are these labels from the primary ones? Low overlap means
    # the robustness check is asking a genuinely different question.
    strat, _ = label_walk_forward(oof, mode="stratified")
    both = labels.notna() & strat.notna()
    agree = float((labels[both] == strat[both]).mean())
    overlap = float((labels[both].eq(1) & strat[both].eq(1)).sum()
                    / max(int(strat[both].eq(1).sum()), 1))
    print(f"  agreement with stratified labels: {agree:.3f}")
    print(f"  share of stratified positives also positive here: {overlap:.3f}")

    print("\ntraining watcher on relative-error labels (same features, params, seed)...")
    proba, diag, imp = run_watcher(oof, feats, labels)

    print("\n" + "=" * 72)
    print(f"watcher on relative-error labels at {STATION}")
    print("=" * 72)
    print(diag.round(4).to_string())
    num = diag.select_dtypes("number")
    print("\nmean across scored folds:")
    print(num.dropna(how="all").mean().round(4).to_string())

    out = oof[["station", "origin", "fold"]].copy()
    out["label"] = labels.to_numpy()
    out["p_unreliable"] = proba.to_numpy()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH)
    print(f"\nwritten: {OUT_PATH}")


if __name__ == "__main__":
    main()