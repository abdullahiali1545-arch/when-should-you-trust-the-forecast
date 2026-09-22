"""
W3.5 - Risk-coverage curve, AURC, and the routing sweep.

Two views of the same within-fold ranking used in W3.4:

  (a) Risk-coverage (selective view). Sort hours from most to least trusted.
      Keep the most-trusted fraction c. Risk(c) = F3's MAE on the kept hours.
      AURC = area under that curve. Lower = the rule ranks F3's errors better.
      Asks: does the rule know which hours F3 gets right?

  (b) Routing sweep (system view). Route the top share s of each fold to F0
      (persistence), publish F3 elsewhere, score the whole system.
      Asks: is there ANY share at which routing beats always-ML?
      This is the sweep the W3.4 pre-registration promised.

References (use the truth, so they are yardsticks, never rules):
  oracle_sel   ranks by F3's actual absolute error -> best possible AURC
  oracle_route ranks by g = |e_F3| - |e_F0|        -> best possible routing

Run:  python -m src.risk_coverage              (primary)
      python -m src.risk_coverage --relative   (Part 8 robustness check)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # write files, no pop-up window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.routing import (FALLBACK, LABEL_MODE, ML, STATION, SUFFIX, TRUTH,
                         load_inputs, make_scores, route)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_PATH = REPO_ROOT / "results/figures" / f"{STATION}_risk_coverage{SUFFIX}.png"
AURC_PATH = REPO_ROOT / "results" / f"{STATION}_aurc{SUFFIX}.csv"
SWEEP_PATH = REPO_ROOT / "results" / f"{STATION}_routing_sweep{SUFFIX}.csv"

SWEEP_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)   # shares routed
Q_HEADLINE = 0.20                                         # W3.4's pre-registered share


# ------------------------------------------------------------ selective view
def within_fold_trust_order(score: pd.Series, fold: pd.Series) -> np.ndarray:
    """Positions of rows sorted from most to least trusted.

    Each score becomes a within-fold percentile (low score = trusted), so
    ranking is per fold exactly as in W3.4, then all folds are pooled.
    Ties broken by row order, so the result is deterministic.
    """
    pct = (score.groupby(fold).rank(ascending=True, method="first")
           / score.groupby(fold).transform("size"))
    return np.argsort(pct.to_numpy(), kind="stable")


def risk_coverage(score: pd.Series, fold: pd.Series, err: pd.Series):
    """Coverage grid, risk at each coverage, and AURC (lower is better)."""
    e = err.to_numpy()[within_fold_trust_order(score, fold)]
    k = np.arange(1, len(e) + 1)
    coverage = k / len(e)
    risk = np.cumsum(e) / k               # running mean error of kept hours
    aurc = float(risk.mean())             # discrete integral over coverage
    return coverage, risk, aurc


def risk_at(coverage: np.ndarray, risk: np.ndarray, c: float) -> float:
    return float(risk[np.searchsorted(coverage, c)])


# -------------------------------------------------------------- system view
def routing_sweep(score: pd.Series, fold: pd.Series, df: pd.DataFrame) -> pd.Series:
    """Whole-system MAE at every share routed, using your route()."""
    out = {}
    for s in SWEEP_GRID:
        mask = route(score, fold, float(s))
        published = df[FALLBACK].where(mask, df[ML])
        out[float(s)] = float((df[TRUTH] - published).abs().mean())
    return pd.Series(out, name="MAE")


# ---------------------------------------------------------------------- main
def main() -> None:
    df = load_inputs()
    fold = df["fold"]
    err_ml = (df[TRUTH] - df[ML]).abs()
    g = err_ml - (df[TRUTH] - df[FALLBACK]).abs()

    rules = make_scores(df)
    sel_scores = {**rules, "oracle_sel": err_ml}
    route_scores = {**rules, "oracle_route": g}

    # (a) selective view
    curves, rows = {}, []
    for name, s in sel_scores.items():
        cov, risk, aurc = risk_coverage(s, fold, err_ml)
        curves[name] = (cov, risk)
        rows.append({"rule": name, "AURC": aurc,
                     "risk_at_80pct": risk_at(cov, risk, 1 - Q_HEADLINE),
                     "risk_at_50pct": risk_at(cov, risk, 0.50)})
    aurc = pd.DataFrame(rows).set_index("rule")
    aurc["AURC_vs_R0"] = aurc["AURC"] - aurc.loc["R0", "AURC"]

    # (b) system view
    sweep = pd.DataFrame({name: routing_sweep(s, fold, df)
                          for name, s in route_scores.items()})
    sweep.index.name = "share_routed"
    always_ml = sweep.loc[0.0, "R0"]
    best = pd.DataFrame({"best_share": sweep.idxmin(),
                         "best_MAE": sweep.min(),
                         "best_vs_always_ML": sweep.min() - always_ml})

    # print
    print("\n" + "=" * 72)
    print(f"W3.5 at {STATION}: selective view (F3 error on kept hours)")
    print("=" * 72)
    print(aurc.round(4).to_string())
    print(f"\nR0 (random) AURC should sit near overall F3 MAE = {err_ml.mean():.4f}")

    print("\n" + "=" * 72)
    print(f"W3.5 at {STATION}: routing sweep (whole-system MAE by share routed)")
    print("=" * 72)
    print(sweep.round(4).to_string())
    print("\nbest share per rule (best_vs_always_ML below 0 = routing helped there):")
    print(best.round(4).to_string())
    print("\noracle rows use the truth: yardsticks, not rules.")
    print("No difference counts as a win until the W3.6 bootstrap interval excludes zero.")

    # figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    styles = {"R0": ("grey", "--"), "R1": ("tab:orange", "-"), "R2": ("tab:green", "-"),
              "R3": ("tab:blue", "-"), "oracle_sel": ("black", ":"),
              "oracle_route": ("black", ":")}
    for name, (cov, risk) in curves.items():
        c, ls = styles[name]
        ax1.plot(cov, risk, color=c, ls=ls, lw=1.6,
                 label=f"{name} (AURC {aurc.loc[name, 'AURC']:.3f})")
    ax1.set_xlim(0.05, 1.0)
    ax1.set_ylim(0, None)
    ax1.set_xlabel("coverage: share of hours F3 answers (most trusted first)")
    ax1.set_ylabel("F3 MAE on answered hours (µg/m³)")
    ax1.set_title("Risk–coverage (lower = better ranking)")
    ax1.legend(fontsize=8)

    for name in sweep.columns:
        c, ls = styles[name]
        ax2.plot(sweep.index, sweep[name], color=c, ls=ls, lw=1.6, marker=".", label=name)
    ax2.axhline(always_ml, color="red", lw=0.8, label=f"always-ML ({always_ml:.3f})")
    ax2.axvline(Q_HEADLINE, color="red", lw=0.6, ls="--")
    ax2.set_xlabel("share of hours routed to persistence")
    ax2.set_ylabel("whole-system MAE (µg/m³)")
    ax2.set_title("Routing sweep (below red line = routing helped)")
    ax2.legend(fontsize=8)

    fig.suptitle(f"{STATION}, Mode B, labels = {LABEL_MODE}, folds {int(fold.min())}–{int(fold.max())}, "
                 f"{len(df):,} hours")
    fig.tight_layout()

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    aurc.to_csv(AURC_PATH)
    sweep.to_csv(SWEEP_PATH)
    print(f"\nwritten: {FIG_PATH}\nwritten: {AURC_PATH}\nwritten: {SWEEP_PATH}")


if __name__ == "__main__":
    main()