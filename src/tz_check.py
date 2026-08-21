"""
W1.3 — Timezone verification for AURN data.

Determines empirically whether AURN `.RData` timestamps are genuine UTC instants
or British local wall-clock time encoded as if it were UTC.

Three lines of evidence:
  1. Raw epoch-seconds step sizes across both 2020 clock changes
  2. Rows per UTC calendar date and null rate on the transition dates
  3. Diurnal alignment of NO2 between a GMT month and a BST month (2019)

Result is written up in docs/ingest_checks.md §1.
Run: python src/tz_check.py
"""

import urllib.request
import pathlib
import warnings

import pandas as pd
import rdata

# rdata has no converter for R's POSIXct class and returns the underlying
# numeric instead. That is why `date` arrives as float64 epoch seconds and is
# converted manually below. Benign — suppressed so the output stays readable.
warnings.filterwarnings("ignore", category=UserWarning)

SITE = "MY1"


def load(site: str, year: int) -> pd.DataFrame:
    """Download (once) and read one AURN year. Adds a `utc` column."""
    tmp = pathlib.Path(f"{site}_{year}.RData")
    if not tmp.exists():
        url = f"https://uk-air.defra.gov.uk/openair/R_data/{site}_{year}.RData"
        tmp.write_bytes(urllib.request.urlopen(url).read())
    d = rdata.read_rda(tmp)[f"{site}_{year}"]
    d["utc"] = pd.to_datetime(d["date"].astype("float64"), unit="s", utc=True)
    return d


def step_check(d: pd.DataFrame, year: int) -> None:
    """Evidence 1: consecutive differences in the raw epoch-seconds column."""
    raw = d["date"].astype("float64")
    step = raw.diff()

    print(f"--- {SITE} {year}: raw step sizes ---")
    print("shape                  :", d.shape)
    print("unique steps (s)       :", sorted(step.dropna().unique()))
    print("repeated raw values    :", int(raw.duplicated().sum()))
    print("negative steps         :", int((step < 0).sum()))
    print("rows where step != 3600:", int((step.notna() & (step != 3600)).sum()))
    print()


def transition_windows(d: pd.DataFrame) -> None:
    """Evidence 1 (detail): inspect both 2020 transitions row by row.

    March skips the 01:00 label, so only March can open a 7200 gap.
    October repeats the 01:00 label, so it can only show a repeat or a
    negative step. March is load-bearing; October corroborates.
    """
    raw = d["date"].astype("float64")
    look = pd.DataFrame({"raw": raw, "utc": d["utc"], "step_s": raw.diff()})

    for label, lo, hi in [
        ("29 Mar 2020  GMT->BST  (label 01:00 is SKIPPED)",
         "2020-03-28 21:00Z", "2020-03-29 07:00Z"),
        ("25 Oct 2020  BST->GMT  (label 01:00 is REPEATED)",
         "2020-10-24 21:00Z", "2020-10-25 07:00Z"),
    ]:
        w = look[(look["utc"] >= lo) & (look["utc"] < hi)]
        print(f"--- {label} ---")
        print(w.to_string(index=False))
        print()


def rows_and_nulls(d: pd.DataFrame) -> None:
    """Evidence 2: rows per UTC date and null rate on the transition dates.

    A constant 3600 step is not on its own proof of UTC — a local-encoded
    series whose missing March hour was padded with an all-null row reads the
    same. A padded row would be null, so a 0% null rate rules that out.
    Local wall-clock labels would give 23 rows on 29 Mar and 25 on 25 Oct.
    """
    day = d["utc"].dt.date
    print("--- rows per UTC calendar date, and PM2.5 null rate ---")
    for date in ["2020-03-28", "2020-03-29", "2020-03-30",
                 "2020-10-24", "2020-10-25", "2020-10-26"]:
        m = day == pd.Timestamp(date).date()
        nulls = d.loc[m, "PM2.5"].isna().mean()
        print(f"{date}  rows={int(m.sum()):3d}  null={nulls:.0%}")
    print()


def coverage(d: pd.DataFrame, year: int) -> None:
    """Incidental: monthly PM2.5 coverage, feeds the station audit in W1.6."""
    cov = d.groupby(d["utc"].dt.month)["PM2.5"].apply(lambda s: s.notna().mean())
    print(f"--- {SITE} {year}: PM2.5 coverage by month ---")
    print(cov.round(3).to_string())
    print()


def diurnal_alignment(d: pd.DataFrame) -> None:
    """Evidence 3: NO2 morning ramp, GMT month vs BST month.

    Grouped in Europe/London, NOT UTC. Human activity follows the local clock,
    so under correct UTC storage the local-hour profiles align and the UTC-hour
    profiles sit one hour apart.

    NO2 rather than PM2.5: kerbside NO2 is close to pure traffic exhaust.
    PM2.5 at MY1 is dominated by non-exhaust wear and regional secondary
    aerosol and has almost no diurnal structure.

    Steepest hour-on-hour RISE rather than peak hour: NO2 at MY1 climbs
    monotonically into the late afternoon, so idxmax over a morning window
    just reports the window edge.
    """
    local = d["utc"].dt.tz_convert("Europe/London")
    prof = d.groupby([local.dt.month, local.dt.hour])["NO2"].mean()
    prof_utc = d.groupby([local.dt.month, d["utc"].dt.hour])["NO2"].mean()

    print("--- NO2 morning ramp, hour-on-hour difference ---")
    for name, month in [("February (GMT)", 2), ("June (BST)", 6)]:
        loc = prof.loc[month].diff().loc[5:9]
        utc = prof_utc.loc[month].diff().loc[5:9]
        print(f"{name}  local hour: steepest at {loc.idxmax()} "
              f"(+{loc.max():.1f})   |   UTC hour: steepest at {utc.idxmax()} "
              f"(+{utc.max():.1f})")
    print()
    print("Expected under correct UTC storage: both steepest at local hour 7")
    print("(the 06->07 step), and one hour apart in UTC.")
    print()


if __name__ == "__main__":
    d20 = load(SITE, 2020)
    step_check(d20, 2020)
    transition_windows(d20)
    rows_and_nulls(d20)
    coverage(d20, 2020)

    d19 = load(SITE, 2019)
    step_check(d19, 2019)
    coverage(d19, 2019)
    diurnal_alignment(d19)

    # Raw column counts exclude the `utc` column added by load().
    c19, c20 = set(d19.columns) - {"utc"}, set(d20.columns) - {"utc"}
    print(f"raw columns: {SITE} 2019 = {len(c19)}, {SITE} 2020 = {len(c20)}")
    print("  in 2019 but not 2020:", sorted(c19 - c20))
    print("  in 2020 but not 2019:", sorted(c20 - c19))
    print("  <- schema drift, see ingest_checks.md")