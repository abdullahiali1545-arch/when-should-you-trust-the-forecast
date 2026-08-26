"""
W1.4 — AURN publication lag (data freshness) at MY1.

Answers two questions:
  1. How many hours behind "now" is the newest published reading?
  2. Is a long lag a pipeline property, or a dead instrument at this one site?

Question 2 is why both PM2.5 and NO2 are measured. They come off different
instruments down the same publication pipeline, so:
    both stale together  -> genuine publication lag
    one stale, one fresh -> instrument outage, not a pipeline problem

Writes docs/freshness_snippet.md, ready to paste into docs/ingest_checks.md.

Run:  right-click -> Run in PyCharm, with the `aq` interpreter.
"""

from __future__ import annotations

import pathlib
import tempfile
import urllib.error
import urllib.request
import warnings

import pandas as pd
import rdata

SITE = "MY1"
FRESH_H = 6         # a live forecast needs lag < horizon, and the horizon is 6 h
DEAD_H = 168        # over this (7 days) = treat as an outage, not a lag
WINDOW_DAYS = 30

REPO = pathlib.Path(__file__).resolve().parents[1]
SNIPPET = REPO / "docs" / "freshness_snippet.md"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_aurn(site: str, year: int) -> pd.DataFrame:
    """Download one site-year of AURN data. Temp file lives outside the repo."""
    url = f"https://uk-air.defra.gov.uk/openair/R_data/{site}_{year}.RData"
    tmp = pathlib.Path(tempfile.gettempdir()) / f"{site}_{year}.RData"
    tmp.write_bytes(urllib.request.urlopen(url).read())   # plain Path, NOT
                                                          # NamedTemporaryFile:
    with warnings.catch_warnings():                       # Windows holds an
        warnings.simplefilter("ignore")                   # exclusive lock
        parsed = rdata.read_rda(tmp)                      # (POSIXct warning
                                                          #  is benign)
    key = f"{site}_{year}"
    df = parsed[key] if key in parsed else next(iter(parsed.values()))
    df["date"] = pd.to_datetime(df["date"], unit="s", utc=True)
    return df


def find_col(df: pd.DataFrame, name: str) -> str | None:
    """Resolve a pollutant column without assuming DEFRA's exact casing."""
    key = name.lower().replace(".", "").replace("_", "")
    for c in df.columns:
        if str(c).lower().replace(".", "").replace("_", "") == key:
            return str(c)
    return None


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def last_valid(df: pd.DataFrame, col: str) -> pd.Timestamp | None:
    """Newest timestamp carrying an actual value. NOT df['date'].max()."""
    valid = df.loc[df[col].notna(), "date"]
    return valid.max() if len(valid) else None


def recent_coverage(df: pd.DataFrame, col: str, now: pd.Timestamp) -> float:
    """Non-null fraction over the trailing window, against hours expected."""
    start = now - pd.Timedelta(days=WINDOW_DAYS)
    window = df[(df["date"] > start) & (df["date"] <= now)]
    return window[col].notna().sum() / (24 * WINDOW_DAYS)


def hours_since(ts: pd.Timestamp | None, now: pd.Timestamp) -> float | None:
    return None if ts is None else (now - ts).total_seconds() / 3600


def fmt(h: float | None) -> str:
    if h is None:
        return "no data at all"
    if h < 72:
        return f"{h:.1f} h"
    return f"{h:.1f} h ({h / 24:.1f} days)"


# --------------------------------------------------------------------------
# interpretation
# --------------------------------------------------------------------------

def verdict(pm: float | None, no2: float | None) -> tuple[str, str]:
    """Return (headline, consequence)."""
    if pm is None and no2 is None:
        return ("NO DATA for either pollutant this year.",
                "Site may be decommissioned. Escalate to the W1.6 station audit.")
    if pm is not None and no2 is not None and pm <= FRESH_H and no2 <= FRESH_H:
        return ("Genuine publication lag; both pollutants fresh together.",
                "Live scoreboard (spec Part 12) is FEASIBLE from AURN.")
    if (no2 is not None and no2 <= FRESH_H) and (pm is None or pm > DEAD_H):
        return ("PM2.5 INSTRUMENT OUTAGE at MY1 — NO2 is fresh, PM2.5 is not.",
                "Pipeline is fine. This is a station-selection problem, not a "
                "freshness one: PM2.5 is the forecasting target. Carry to W1.6 "
                "alongside the 78.7% 2020 coverage note.")
    if (pm is not None and pm <= FRESH_H) and (no2 is None or no2 > DEAD_H):
        return ("NO2 outage; PM2.5 fresh.",
                "Target pollutant is fine. NO2 lag features degrade. Note in W1.6.")
    if (pm or 0) > DEAD_H and (no2 or 0) > DEAD_H:
        return ("Both pollutants stale — whole site offline, or slow pipeline.",
                "Live scoreboard NOT feasible from AURN. Fall back to Imperial's "
                "LAQN API (spec Part 3, residual concern 3).")
    return ("Moderate lag, between fresh and dead.",
            "Live scoreboard is marginal. Compare against LAQN before promising it.")


# --------------------------------------------------------------------------

def main() -> None:
    now = pd.Timestamp.now(tz="UTC")          # UTC, not local. BST would
    year = now.year                           # inflate every lag by 1 h.

    print(f"\nSnapshot taken {now:%Y-%m-%d %H:%M} UTC\n" + "=" * 64)

    try:
        df = load_aurn(SITE, year)
    except urllib.error.HTTPError as e:
        print(f"!! {SITE}_{year}.RData -> HTTP {e.code}. "
              f"Current-year file may not exist. Falling back to {year - 1}.")
        year -= 1
        df = load_aurn(SITE, year)

    print(f"File: {SITE}_{year}.RData   shape: {df.shape[0]:,} x {df.shape[1]}")

    # Does DEFRA pad the current year to 31 Dec, or truncate at the last
    # published hour? The gap between these two lines answers it.
    last_row = df["date"].max()
    print(f"Last ROW timestamp     : {last_row:%Y-%m-%d %H:%M} UTC  "
          f"<- meaningless if padded")

    results = {}
    for name in ("PM2.5", "NO2"):
        col = find_col(df, name)
        if col is None:
            print(f"!! column for {name} not found. Columns: {list(df.columns)}")
            results[name] = (None, None, None)
            continue
        ts = last_valid(df, col)
        lag = hours_since(ts, now)
        cov = recent_coverage(df, col, now)
        results[name] = (ts, lag, cov)
        stamp = f"{ts:%Y-%m-%d %H:%M} UTC" if ts is not None else "none"
        print(f"Last non-null {name:<6}   : {stamp}   lag {fmt(lag)}   "
              f"last {WINDOW_DAYS}d coverage {cov:.1%}")

    pm_lag, no2_lag = results["PM2.5"][1], results["NO2"][1]
    headline, consequence = verdict(pm_lag, no2_lag)

    print("\n" + "-" * 64)
    print("VERDICT:", headline)
    print(consequence)
    if pm_lag is not None:
        print(f"\nHonesty constraint for the README: a live deployment carries "
              f"~{pm_lag:.0f} h less residual information than the backtest, "
              f"because the truth at s+6h is not published until then.")
    print("-" * 64)

    write_snippet(now, year, df, last_row, results, headline, consequence)
    print(f"\nWrote {SNIPPET.relative_to(REPO)} — paste it into "
          f"docs/ingest_checks.md, then delete it.\n")


def write_snippet(now, year, df, last_row, results, headline, consequence) -> None:
    pm_ts, pm_lag, pm_cov = results["PM2.5"]
    no2_ts, no2_lag, no2_cov = results["NO2"]

    def row(name, ts, lag, cov):
        stamp = f"{ts:%Y-%m-%d %H:%M}" if ts is not None else "—"
        return (f"| {name} | {stamp} | {fmt(lag)} | "
                f"{'—' if cov is None else f'{cov:.1%}'} |")

    body = f"""
## §2 — Data freshness / publication lag (W1.4)

**Run:** {now:%d %B %Y}, {now:%H:%M} UTC. **Station:** {SITE}. **File:** `{SITE}_{year}.RData`
({df.shape[0]:,} rows x {df.shape[1]} cols). **Snapshot date pinned:** {now:%Y-%m-%d}.

### Verdict

**{headline}**

{consequence}

### Measurements

Last row timestamp in file: **{last_row:%Y-%m-%d %H:%M} UTC**. This is reported
separately from the last *reading* because DEFRA emits complete regular hourly
grids (see §1), so a padded current-year file carries rows for hours that have
not happened yet. `date.max()` is therefore not a freshness measurement.

| Pollutant | Last non-null (UTC) | Lag | Coverage, trailing {WINDOW_DAYS} d |
|---|---|---|---|
{row("PM2.5", pm_ts, pm_lag, pm_cov)}
{row("NO₂", no2_ts, no2_lag, no2_cov)}

### Why two pollutants

PM2.5 and NO₂ come off different instruments but travel the same publication
pipeline. A single-pollutant measurement cannot separate a slow pipeline from a
dead sensor: both present as a large lag. Measured together, they can —
stale-together indicates the pipeline, stale-alone indicates the instrument.
MY1's autumn 2020 PM2.5 outage (§1.1) makes this a live possibility, not a
hypothetical one.

### Consequence for the headline result

Residual features assume the truth at `s+6h` is available at `s+6h`. It is not:
it is available at `s+6h+N`, where N is the lag above. A deployed watcher
therefore runs on N hours staler residual information than the backtested one.

This is **not leakage** — the backtest uses only genuinely past values. It is a
deployment gap, and spec Part 6 requires it be stated in the README with a
number rather than a caveat.

### Reproduce

`src/freshness_check.py`. Re-running produces a different lag; the snapshot
date above is the one that stands unless a re-pull is recorded here.
"""
    SNIPPET.parent.mkdir(parents=True, exist_ok=True)
    SNIPPET.write_text(body.lstrip(), encoding="utf-8")


if __name__ == "__main__":
    main()