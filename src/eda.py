"""
src/eda.py — W1.8. Exploratory figures for the development period only.

Reads data/processed/*.parquet (raw truth, not engineered features: if a
feature were wrong, plotting the feature would hide the error rather than
reveal it) and writes five figures to results/figures/.

    python src/eda.py

THE HOLDOUT IS SEALED
---------------------
Holdout = 2025, pre-registered 2026-08-29 (PROJECT_SPEC changelog, "no
results seen"). Spec Part 10: do not look at it, plot it, or compute anything
on it while developing.

Every figure below is produced from data filtered to DEV_END or earlier. The
filter is applied once, at load, so no individual figure can forget. Looking
at 2025 cannot be undone — it would contaminate every downstream judgement
call and there is no diagnostic that would reveal it afterwards.

WHAT EACH FIGURE IS FOR
-----------------------
1  missingness   — find outages before Week 2 does
2  diurnal       — independent check that UTC -> Europe/London is correct
3  seasonality   — domain sanity: winter should exceed summer
4  stations      — how different are the four? bears on H4 (leave-one-station-out)
5  persistence   — how hard is F0 to beat at +6h? sets Week 2 expectations
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")           # write files, never try to open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "results" / "figures"

# --------------------------------------------------------------------------
# THE SEAL. Do not change this line without a dated PROJECT_SPEC entry.
# --------------------------------------------------------------------------
DEV_END = pd.Timestamp("2024-12-31 23:00", tz="UTC")

LOCAL_TZ = "Europe/London"
HORIZON_H = 6
SUMMER = [6, 7, 8]
WINTER = [12, 1, 2]


def load_dev() -> dict[str, pd.DataFrame]:
    """Load every station, truncated at DEV_END. The only place data enters."""
    out = {}
    for path in sorted(PROCESSED.glob("*.parquet")):
        df = pd.read_parquet(path)
        n_before = len(df)
        df = df.loc[df.index <= DEV_END]
        out[path.stem] = df
        print(f"  {path.stem}: {len(df):,} rows "
              f"({n_before - len(df):,} sealed in holdout)")
    if not out:
        raise SystemExit(f"no parquet files in {PROCESSED}")
    return out


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------
# 1 — Missingness map
# --------------------------------------------------------------------------
def fig_missingness(data: dict[str, pd.DataFrame]) -> None:
    """Monthly PM2.5 coverage per station.

    Looking for: contiguous dark blocks (an instrument down for months),
    or a station that degrades over time. A scattered speckle is normal
    hourly dropout and harmless.
    """
    rows = {}
    for site, df in data.items():
        monthly = df["pm2_5"].notna().groupby(
            [df.index.year, df.index.month]).mean()
        rows[site] = monthly

    mat = pd.DataFrame(rows).T
    fig, ax = plt.subplots(figsize=(14, 2.6))
    im = ax.imshow(mat.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")

    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index)
    ticks = [i for i, (y, m) in enumerate(mat.columns) if m == 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([mat.columns[i][0] for i in ticks])
    ax.set_title("PM2.5 hourly coverage by month (development period only)")
    fig.colorbar(im, ax=ax, label="fraction present")
    _save(fig, "01_missingness.png")

    worst = mat.min(axis=1)
    for site, v in worst.items():
        month = mat.columns[int(np.argmin(mat.loc[site].values))]
        print(f"    {site}: worst month {month[0]}-{month[1]:02d} at {v:.0%}")


# --------------------------------------------------------------------------
# 2 — Diurnal cycle. The load-bearing figure.
# --------------------------------------------------------------------------
def fig_diurnal(data: dict[str, pd.DataFrame]) -> None:
    """Mean concentration by LOCAL hour, summer vs winter.

    This is the second, independent line of evidence on timezone handling
    (spec Part 6). The raw epoch-step test established what the data IS;
    this establishes whether the handling of it is correct.

    Read NO2, not PM2.5. NO2 is exhaust-driven and tracks the traffic
    rhythm, so it has a sharp morning peak. MY1's kerbside PM2.5 is
    dominated by brake and tyre wear and has almost no diurnal structure —
    a flat line there is the physics, not a bug, and a flat line cannot
    confirm alignment either way.

    PASS: the NO2 morning peak lands at the same local hour in both seasons.
    FAIL: exactly one hour of offset between summer and winter -> the
    UTC -> Europe/London conversion is wrong and every diurnal feature is
    corrupted for half the year.
    """
    sites = list(data)
    fig, axes = plt.subplots(2, len(sites), figsize=(4 * len(sites), 6.5),
                             sharex=True)
    if len(sites) == 1:
        axes = axes.reshape(2, 1)

    for j, site in enumerate(sites):
        df = data[site]
        local_hour = df.index.tz_convert(LOCAL_TZ).hour
        month = df.index.month

        for i, pollutant in enumerate(["no2", "pm2_5"]):
            ax = axes[i, j]
            for season, months, style in [("summer", SUMMER, "-"),
                                          ("winter", WINTER, "--")]:
                mask = np.isin(month, months)
                curve = df.loc[mask, pollutant].groupby(local_hour[mask]).mean()
                ax.plot(curve.index, curve.values, style, label=season)
                if pollutant == "no2" and len(curve):
                    peak = curve.loc[5:11].idxmax()   # morning window
                    print(f"    {site} {season} NO2 morning peak: "
                          f"{peak:02d}:00 local")
            ax.set_title(f"{site} — {pollutant}")
            ax.set_xlabel("hour (Europe/London)")
            if j == 0:
                ax.set_ylabel("mean concentration")
            if i == 0 and j == 0:
                ax.legend()

    fig.suptitle("Diurnal cycle by local hour — NO2 (top) is the alignment test")
    _save(fig, "02_diurnal.png")


# --------------------------------------------------------------------------
# 3 — Seasonality
# --------------------------------------------------------------------------
def fig_seasonality(data: dict[str, pd.DataFrame]) -> None:
    """Mean PM2.5 by calendar month.

    Expect a winter/spring maximum: shallow boundary layer traps emissions,
    plus springtime continental transport episodes. Summer higher than
    winter would need explaining before you trust the series.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for site, df in data.items():
        monthly = df["pm2_5"].groupby(df.index.month).mean()
        ax.plot(monthly.index, monthly.values, marker="o", label=site)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("month")
    ax.set_ylabel("mean PM2.5 (ug/m3)")
    ax.set_title("Seasonality of PM2.5 (development period)")
    ax.legend()
    _save(fig, "03_seasonality.png")


# --------------------------------------------------------------------------
# 4 — Station comparison
# --------------------------------------------------------------------------
def fig_stations(data: dict[str, pd.DataFrame]) -> None:
    """PM2.5 distribution per station.

    Bears on H4. If all four look identical, leave-one-station-out is a
    trivially easy transfer test and a success there means less. Genuine
    spread between site types makes H4 informative.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    series = [df["pm2_5"].dropna().values for df in data.values()]
    ax.boxplot(series, tick_labels=list(data), showfliers=False)
    ax.set_ylabel("PM2.5 (ug/m3)")
    ax.set_title("PM2.5 distribution by station (outliers hidden)")
    _save(fig, "04_stations.png")

    print("    station summary (ug/m3):")
    for site, df in data.items():
        s = df["pm2_5"].dropna()
        print(f"      {site}: median {s.median():5.1f}  "
              f"mean {s.mean():5.1f}  p95 {s.quantile(0.95):5.1f}")


# --------------------------------------------------------------------------
# 5 — How hard is persistence to beat?
# --------------------------------------------------------------------------
def fig_persistence(data: dict[str, pd.DataFrame]) -> None:
    """PM2.5 now vs PM2.5 in six hours, and F0's naive error.

    F0 (persistence) predicts y(t+6) = y(t). It is the bar F3 must clear
    AND the fallback the routing policy switches to, so its difficulty is
    the single most useful number to know before Week 2.

    NOTE: these MAE figures are computed over the whole development period
    at once. They are NOT the walk-forward result and must never be quoted
    as one. Week 2's harness produces the scoreable number. This is a
    difficulty gauge, nothing more.
    """
    sites = list(data)
    fig, axes = plt.subplots(1, len(sites), figsize=(3.6 * len(sites), 3.8),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, site in zip(axes, sites):
        df = data[site]
        now = df["pm2_5"]
        later = now.shift(-HORIZON_H)
        ok = now.notna() & later.notna()
        x, y = now[ok], later[ok]

        ax.scatter(x, y, s=1, alpha=0.05)
        lim = float(np.nanpercentile(x, 99))
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(site)
        ax.set_xlabel("PM2.5 at t")

        r = float(np.corrcoef(x, y)[0, 1])
        mae = float((y - x).abs().mean())
        print(f"    {site}: corr(t, t+6h) = {r:.3f}   "
              f"persistence MAE = {mae:.2f} ug/m3   "
              f"(mean level {x.mean():.1f})")

    axes[0].set_ylabel("PM2.5 at t+6h")
    fig.suptitle("Persistence difficulty: the bar F3 has to clear")
    _save(fig, "05_persistence.png")


# --------------------------------------------------------------------------
def main() -> None:
    print(f"development period: everything up to {DEV_END}")
    print("holdout 2025 is sealed and is not read by this script\n")

    data = load_dev()

    print("\n[1/5] missingness")
    fig_missingness(data)
    print("\n[2/5] diurnal — check the NO2 peaks below line up across seasons")
    fig_diurnal(data)
    print("\n[3/5] seasonality")
    fig_seasonality(data)
    print("\n[4/5] stations")
    fig_stations(data)
    print("\n[5/5] persistence")
    fig_persistence(data)

    print("\ndone. Record the diurnal verdict in docs/ingest_checks.md §1.")


if __name__ == "__main__":
    main()
