"""
Headline figure for the README (PROJECT_SPEC Part 16, item 2).

Reads   results/<STATION>_bootstrap.csv   (written by src/bootstrap.py)
Writes  results/figures/<STATION>_headline.png

Two panels, both showing the paired block-bootstrap interval on an MAE difference:
  Left:  each routing rule vs always-ML        -> answers H3 (does routing help at all?)
  Right: the watcher R3 vs each trivial rival  -> answers H2 (does the watcher add anything?)

Sign convention (inherited from bootstrap.py): diff_MAE = MAE(A) - MAE(B).
Positive = A is worse. The dashed line at zero is "no difference".
Filled marker = interval excludes zero (a detectable difference, per the Part 10 win condition).
Hollow marker = interval includes zero (no detectable difference).

Run from the project root:
    python -m src.plot_headline
    python -m src.plot_headline --relative     # robustness version (relative-error labels)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # write to file; no window needed
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # anchor paths to the repo, not the working directory

# Human-readable names for the axis labels
NAMES = {
    "R0": "R0  random",
    "R1": "R1  high predicted PM2.5",
    "R2": "R2  high recent error",
    "R3": "R3  watcher",
    "always_fallback": "Always persistence",
}

# Rows each panel needs, in display order (top to bottom)
LEFT_ROWS = [("R3", "always_ML"), ("R2", "always_ML"), ("R1", "always_ML"),
             ("R0", "always_ML"), ("always_fallback", "always_ML")]
RIGHT_ROWS = [("R3", "R2"), ("R3", "R1"), ("R3", "R0")]

COLOUR_WORSE = "#c0392b"   # A worse
COLOUR_BETTER = "#1e8449"  # A better
COLOUR_NULL = "#555555"    # no detectable difference


def load(station: str, relative: bool) -> pd.DataFrame:
    suffix = "_rel" if relative else ""
    path = ROOT / "results" / f"{station}_bootstrap{suffix}.csv"
    df = pd.read_csv(path)
    # Fail loudly if bootstrap.py's output format ever changes
    needed = {"A", "B", "diff_MAE", "ci_low", "ci_high", "excludes_zero"}
    missing = needed - set(df.columns)
    assert not missing, f"{path.name} is missing columns: {missing}"
    return df


def pick(df: pd.DataFrame, rows: list[tuple[str, str]]) -> pd.DataFrame:
    out = []
    for a, b in rows:
        hit = df[(df["A"] == a) & (df["B"] == b)]
        assert len(hit) == 1, f"expected exactly one row for {a} vs {b}, found {len(hit)}"
        out.append(hit.iloc[0])
    return pd.DataFrame(out).reset_index(drop=True)


def colour(row) -> str:
    if not bool(row["excludes_zero"]):
        return COLOUR_NULL
    return COLOUR_WORSE if row["diff_MAE"] > 0 else COLOUR_BETTER


def draw_panel(ax, data: pd.DataFrame, labels: list[str], title: str) -> None:
    y = list(range(len(data)))[::-1]      # first row at the top
    for yi, (_, row) in zip(y, data.iterrows()):
        c = colour(row)
        filled = bool(row["excludes_zero"])
        ax.errorbar(row["diff_MAE"], yi,
                    xerr=[[row["diff_MAE"] - row["ci_low"]], [row["ci_high"] - row["diff_MAE"]]],
                    fmt="o", color=c, ecolor=c, elinewidth=2, capsize=5, markersize=8,
                    markerfacecolor=c if filled else "white", markeredgewidth=2)
        ax.annotate(f"{row['diff_MAE']:+.3f}  [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]",
                    xy=(row["ci_high"], yi), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=c)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel("MAE difference (µg/m³)    ← better   |   worse →")
    ax.grid(axis="x", alpha=0.3)
    # leave room on the right for the number labels
    lo, hi = data["ci_low"].min(), data["ci_high"].max()
    span = hi - lo
    ax.set_xlim(min(lo, 0) - 0.1 * span, hi + 0.75 * span)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", default="MY1")
    parser.add_argument("--relative", action="store_true",
                        help="plot the relative-error-label robustness version")
    args = parser.parse_args()

    df = load(args.station, args.relative)
    left = pick(df, LEFT_ROWS)
    right = pick(df, RIGHT_ROWS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    draw_panel(ax1, left, [NAMES[a] for a, _ in LEFT_ROWS],
               "Does routing beat always using the model?\n(each rule minus always-ML)")
    draw_panel(ax2, right, [f"R3 vs {NAMES[b].split('  ')[0]}  ({NAMES[b].split('  ')[1]})"
                            for _, b in RIGHT_ROWS],
               "Does the watcher beat the trivial rules?\n(R3 minus each rival)")

    label_note = " (relative-error labels — robustness check)" if args.relative else ""
    fig.suptitle(f"{args.station}: routed forecasting system vs always-ML{label_note}",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, -0.02,
             "Paired block-bootstrap intervals, week-long blocks. Filled marker = interval excludes zero "
             "(detectable difference). Hollow grey = no detectable difference.",
             fontsize=8.5, color="#444444", ha="left")
    fig.tight_layout()

    suffix = "_rel" if args.relative else ""
    out = ROOT / "results" / "figures" / f"{args.station}_headline{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
