"""
src/select_stations.py — W1.6 station selection.

Screens Greater London AURN sites against the coverage thresholds
pre-registered in PROJECT_SPEC.md changelog 2026-09-02:

    PM2.5 (target)   pooled >= 80%, per-year floor >= 70%, record starts <= 2019-01-01
    NO2   (feature)  pooled >= 70%, no floor

Writes docs/station_selection.md — survivors AND rejections, each with the
number that decided it.

Coverage is measured on RAW AURN data. This script never reads
data/processed/: ingest-time interpolation fills gaps under 2 h, which would
inflate coverage unevenly — a station with many short gaps gets heavily
patched, one with a single long outage gets none.

Usage:
    python src/select_stations.py
"""

from __future__ import annotations

import logging
import sys
import urllib.error
from pathlib import Path

import pandas as pd

# Put src/ on the import path so `import ingest` works however this script is
# launched — from the repo root, from src/, or via PyCharm right-click Run.
# The tidy long-term fix is a pyproject.toml and `pip install -e .`; that is a
# Week 5 packaging job, not a Week 1 one.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest import DEFRA, RAW, ROOT, _download, _read_rda  # noqa: E402

log = logging.getLogger("select_stations")

DOCS = ROOT / "docs"
YEARS = range(2018, 2026)          # 2018-2025 inclusive

# Thresholds — see PROJECT_SPEC.md changelog 2026-09-02.
PM25_POOLED_MIN = 0.80
PM25_YEAR_FLOOR = 0.70
NO2_POOLED_MIN = 0.70
# End of 2019-01-01, not its first instant: the pre-registered rule is
# "record begins <= 2019-01-01", so a station whose first reading falls
# anywhere on that date passes. Comparing against 00:00 made the code
# stricter than the rule it implements.
PM25_MUST_START_BY = pd.Timestamp("2019-01-01 23:59", tz="UTC")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_metadata() -> pd.DataFrame:
    """Site metadata: 3,076 rows x 13 cols, long format — one row per
    site x parameter, so a site appears once for each pollutant it measures."""
    return _read_rda(_download(f"{DEFRA}/AURN_metadata.RData",
                               RAW / "AURN_metadata.RData"))


def london_candidates(meta: pd.DataFrame) -> pd.DataFrame:
    """Greater London sites that claim a PM2.5 instrument.

    Metadata is a CLAIM, not a measurement. A site listed here may still
    have no usable record; that is what the coverage screen is for.
    Returns one row per site: site_id, site_name, location_type.
    """
    m = meta.copy()
    m["zone"] = m["zone"].astype(str)
    m["parameter"] = m["parameter"].astype(str)

    london = m[m["zone"].str.contains("London", case=False, na=False)]
    has_pm25 = london[london["parameter"].str.upper().isin(
        {"PM2.5", "PM25", "PM<SUB>2.5</SUB>"}
    )]

    out = (has_pm25[["site_id", "site_name", "location_type"]]
           .astype(str)
           .drop_duplicates(subset="site_id")
           .sort_values("site_id")
           .reset_index(drop=True))
    return out


def load_station_year(site: str, year: int) -> pd.DataFrame | None:
    """One site-year, coverage columns only. None if DEFRA has no such file.

    Deliberately NOT ingest.load_aurn_year: that requires all nine allowlist
    columns and raises if any is missing. Correct there, wrong here — a site
    with no ozone monitor must be screened on its PM2.5 record, not rejected
    for lacking a column this screen never reads.

    A missing pollutant column comes back absent, which the coverage
    function reads as zero available hours. That is the honest reading.
    """
    dest = RAW / f"{site}_{year}.RData"
    try:
        _download(f"{DEFRA}/{site}_{year}.RData", dest)
    except urllib.error.HTTPError:
        return None                      # site not operating that year

    df = _read_rda(dest)
    keep = ["date"] + [c for c in ("PM2.5", "NO2") if c in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"], unit="s", utc=True)
    return df.set_index("date")


def load_station(site: str) -> pd.DataFrame:
    """All available years for one site, concatenated. Empty if none."""
    frames = [f for f in (load_station_year(site, y) for y in YEARS)
              if f is not None]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="first")].sort_index()


# --------------------------------------------------------------------------
# The screen itself
# --------------------------------------------------------------------------
def _empty_coverage() -> dict:
    """Uniform zero-coverage result, so callers never branch on None."""
    return {"pooled": 0.0, "per_year": {}, "start": None, "end": None,
            "n_observed": 0, "n_expected": 0}


def coverage_for_pollutant(df: pd.DataFrame, pollutant: str) -> dict:
    """Coverage for one pollutant at one site.

    Args:
        df:         all years for this site, from load_station()
        pollutant:  "PM2.5" or "NO2"

    Returns:
        pooled      float 0-1, over the instrument's operational record
        per_year    {int year: float}
        start/end   first and last non-null reading, or None
        n_observed  hours with a real reading
        n_expected  hours that should have existed in the window

    Denominators follow PROJECT_SPEC.md changelog 2026-09-02: the record of
    THIS pollutant at this site, not the full 2018-2025 span, and not the
    site's overall record.
    """
    # A site with no instrument for this pollutant is not an error. It is
    # zero coverage, and screen() turns that into a documented rejection.
    if pollutant not in df.columns:
        return _empty_coverage()

    mask = df[pollutant].notna()
    if not mask.any():
        return _empty_coverage()

    # The operational window is bounded by real readings, so leading and
    # trailing gaps are excluded rather than counted as failures. A monitor
    # installed in June 2021 is not penalised for January 2021.
    start = mask.idxmax()             # first True
    end = mask[::-1].idxmax()         # last True
    mask = mask.loc[start:end]

    # THE DENOMINATOR. date_range counts hours that SHOULD exist; len(mask)
    # would count only rows present in the file. Where DEFRA published no
    # rows at all for a period, those hours vanish from len(mask) and
    # coverage comes out too high. date_range cannot be fooled that way.
    expected = len(pd.date_range(start, end, freq="h"))
    observed = int(mask.sum())
    pooled = observed / expected if expected else 0.0

    per_year: dict[int, float] = {}
    for year in range(start.year, end.year + 1):
        # Clip each calendar year to the operational window, so a partial
        # year is judged on the months the instrument was running.
        y_start = max(start, pd.Timestamp(f"{year}-01-01", tz="UTC"))
        y_end = min(end, pd.Timestamp(f"{year}-12-31 23:00", tz="UTC"))
        if y_start > y_end:
            continue
        y_expected = len(pd.date_range(y_start, y_end, freq="h"))
        y_observed = int(mask.loc[y_start:y_end].sum())
        per_year[year] = y_observed / y_expected if y_expected else 0.0

    return {"pooled": pooled, "per_year": per_year,
            "start": start, "end": end,
            "n_observed": observed, "n_expected": expected}


def screen(cov_pm25: dict, cov_no2: dict) -> tuple[bool, str]:
    """Apply the pre-registered thresholds to one station.

    Returns (passed, reason). The reason always carries the number that
    decided it, for passes as well as failures: a reader of
    station_selection.md should see why a station survived, not only that
    it did. All failing conditions are reported, not just the first.
    """
    if cov_pm25["start"] is None:
        return False, "no PM2.5 readings 2018-2025"

    failures: list[str] = []

    if cov_pm25["pooled"] < PM25_POOLED_MIN:
        failures.append(
            f"PM2.5 pooled {cov_pm25['pooled']:.1%} < {PM25_POOLED_MIN:.0%}")

    # The floor catches what pooled coverage averages away: one crippled
    # year inside a training window, hidden by seven healthy ones.
    below = {y: c for y, c in cov_pm25["per_year"].items()
             if c < PM25_YEAR_FLOOR}
    if below:
        detail = ", ".join(f"{y} {c:.1%}" for y, c in sorted(below.items()))
        failures.append(f"PM2.5 below {PM25_YEAR_FLOOR:.0%} floor in {detail}")

    # Walk-forward fold 1 trains on 2018-2019. A later record contributes
    # nothing to it and would fail silently in Week 2 rather than here.
    if cov_pm25["start"] > PM25_MUST_START_BY:
        failures.append(
            f"PM2.5 record starts {cov_pm25['start']:%Y-%m-%d %H:%M}, "
            f"after {PM25_MUST_START_BY:%Y-%m-%d %H:%M}")

    if cov_no2["pooled"] < NO2_POOLED_MIN:
        failures.append(
            f"NO2 pooled {cov_no2['pooled']:.1%} < {NO2_POOLED_MIN:.0%}")

    if failures:
        return False, "; ".join(failures)

    worst = min(cov_pm25["per_year"], key=cov_pm25["per_year"].get)
    return True, (
        f"PM2.5 {cov_pm25['pooled']:.1%} pooled, "
        f"worst year {cov_pm25['per_year'][worst]:.1%} ({worst}); "
        f"NO2 {cov_no2['pooled']:.1%} pooled; "
        f"record from {cov_pm25['start']:%Y-%m}")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def write_report(rows: list[dict]) -> None:
    """docs/station_selection.md — survivors, rejections, methodology."""
    DOCS.mkdir(parents=True, exist_ok=True)
    kept = [r for r in rows if r["passed"]]
    dropped = [r for r in rows if not r["passed"]]

    L = ["# Station Selection (W1.6)", "",
         f"Run {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC. "
         f"Thresholds pre-registered in `PROJECT_SPEC.md` changelog "
         f"2026-09-02, before this script was run.", "",
         f"**{len(kept)} of {len(rows)} candidates kept.**", "",
         "## Kept", "",
         "| Site | Name | Type | PM2.5 pooled | PM2.5 worst year | "
         "NO2 pooled | Record |", "|---|---|---|---|---|---|---|"]

    for r in kept:
        L.append(f"| {r['site_id']} | {r['site_name']} | {r['location_type']} "
                 f"| {r['pm25_pooled']:.1%} | {r['pm25_worst']} "
                 f"| {r['no2_pooled']:.1%} | {r['record']} |")

    L += ["", "## Rejected", "",
          "| Site | Name | Reason |", "|---|---|---|"]
    for r in dropped:
        L.append(f"| {r['site_id']} | {r['site_name']} | {r['reason']} |")

    L += ["", "## Method", "",
          "Coverage is measured on **raw AURN data**, before the ingest-time",
          "interpolation of sub-2-hour gaps. Measuring post-imputation would",
          "inflate coverage unevenly: a station with many short gaps is",
          "heavily patched and scores well, a station with one long outage is",
          "not patched at all. That would favour intermittently broken sites",
          "over briefly broken ones — the opposite of what is wanted, since",
          "scattered gaps do more damage to lags and rolling windows than one",
          "contiguous hole.", "",
          "Denominator is the operational record of **that pollutant** at that",
          "site, not the full 2018-2025 span. Scoring absent years as 0% would",
          "reject every station commissioned after 2018 on grounds unrelated",
          "to data quality.", "",
          "A station whose PM2.5 record begins after 2019-01-01 is rejected",
          "regardless of coverage: walk-forward fold 1 trains on 2018-2019, so",
          "such a station contributes nothing to it and would fail silently in",
          "Week 2 rather than here.", ""]

    (DOCS / "station_selection.md").write_text("\n".join(L), encoding="utf-8")
    log.info("wrote %s", DOCS / "station_selection.md")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    candidates = london_candidates(load_metadata())
    log.info("%d London candidates claiming PM2.5", len(candidates))

    rows = []
    for _, c in candidates.iterrows():
        site = c["site_id"]
        log.info("--- %s (%s)", site, c["site_name"])
        df = load_station(site)

        if df.empty:
            rows.append({**c.to_dict(), "passed": False,
                         "reason": "no AURN files 2018-2025"})
            continue

        cov_pm25 = coverage_for_pollutant(df, "PM2.5")
        cov_no2 = coverage_for_pollutant(df, "NO2")
        passed, reason = screen(cov_pm25, cov_no2)

        worst_year = min(cov_pm25["per_year"], key=cov_pm25["per_year"].get) \
            if cov_pm25["per_year"] else None
        rows.append({
            **c.to_dict(),
            "passed": passed,
            "reason": reason,
            "pm25_pooled": cov_pm25["pooled"],
            "no2_pooled": cov_no2["pooled"],
            "pm25_worst": (f"{cov_pm25['per_year'][worst_year]:.1%} "
                           f"({worst_year})") if worst_year else "—",
            "record": f"{cov_pm25['start']:%Y-%m} to {cov_pm25['end']:%Y-%m}"
                      if cov_pm25["start"] is not None else "—",
        })

    write_report(rows)


if __name__ == "__main__":
    main()