# Ingest checks

Empirical checks on the AURN and Open-Meteo feeds, run before any modelling.
Each section records what was measured, what it showed, and what action follows.

---

## §1 — Timezone verification (W1.3)

**Run:** 21 August 2026. **Station:** MY1 (Marylebone Road). **Years:** 2020, 2019.
**Source:** DEFRA openair `.RData` via the `rdata` package.

### Verdict

**AURN timestamps are genuine UTC instants.** The `date` column carries seconds
since the Unix epoch as a true UTC clock, not British local wall-clock time
encoded as if it were UTC.

**Action:** convert with `pd.to_datetime(unit="s", utc=True)` and do nothing
further. **No timezone conversion is applied anywhere in the pipeline.** Applying
one would shift every diurnal feature by an hour for roughly half of each year —
silently, with no error raised.

### Why the check looks at two dates

The two 2020 clock changes are not symmetric, and only one of them can produce a
gap.

- **29 March, GMT→BST.** The clock jumps from 01:00 GMT to 02:00 BST, so the
  label `01:00` is **skipped**. A local-encoded series therefore shows a **7200**
  step here.
- **25 October, BST→GMT.** The clock falls from 02:00 BST to 01:00 GMT, so the
  label `01:00` is **repeated**, not skipped. A local-encoded series shows a
  **repeated value** (step of 0), or possibly a negative step. No gap can open,
  because nothing is missing.

March is therefore load-bearing: it fires whatever the provider does with
October's duplicate. October corroborates.

### Line of evidence 1 — raw epoch-seconds step sizes

Measured on the raw `date` column before any conversion.

| | 2020 | 2019 |
|---|---|---|
| Shape (raw, before adding `utc`) | 8,784 × 49 | 8,760 × 51 |
| Unique consecutive differences | `[3600.0]` | `[3600.0]` |
| Repeated raw values | 0 | 0 |
| Negative (out-of-order) steps | 0 | 0 |
| Rows where step ≠ 3600 | 0 | 0 |

8,784 = 366 × 24 and 8,760 = 365 × 24, so both years are complete regular hourly
grids with no dropped rows.

Windows around both 2020 transitions were inspected row by row. Both run
`…3600, 3600, 3600…` with no repeat, no gap and no reversal.

### Line of evidence 2 — rows per UTC calendar date, and null rate

A constant 3600 step is **not on its own** a confirmation of UTC. Two provider
conventions produce exactly that signature from local-encoded data:
de-duplicating October's repeated hour, and padding March's missing hour with an
all-null row to preserve a regular grid. Both were checked for.

| UTC date | Rows | PM2.5 null |
|---|---|---|
| 2020-03-28 | 24 | 0% |
| **2020-03-29** | **24** | **0%** |
| 2020-03-30 | 24 | 0% |
| 2020-10-24 | 24 | 100% |
| **2020-10-25** | **24** | **100%** |
| 2020-10-26 | 24 | 100% |

Twenty-four rows on both transition dates. Local wall-clock labels would give
**23** on 29 March and **25** on 25 October.

The 0% null rate on 29 March is the decisive number. DEFRA emits a complete
regular grid, so a padded row was a live possibility — but a padded row would be
null, and there are none. March's clean 3600 is genuine, not a filled gap.

The 100% null rate in late October is a real PM2.5 outage at MY1, unrelated to
the clock — see §1.1.

### Line of evidence 3 — diurnal alignment (2019, NO₂)

The step test establishes what the data **is**. It does not establish that the
handling of it is **correct**. Independent check: compare the hourly profile of a
traffic-driven pollutant between a GMT month and a BST month.

Grouped by **hour of day in `Europe/London`**, not UTC. This matters: human
activity follows the local clock, so under correct UTC storage the local-hour
profiles align and the UTC-hour profiles sit one hour apart. Grouping by UTC hour
and expecting alignment would make correct data appear to fail.

**Pollutant: NO₂, not PM2.5.** NO₂ at a kerbside site is close to pure traffic
exhaust and carries a sharp, human-scheduled signal. PM2.5 at MY1 is dominated by
non-exhaust wear and regional secondary aerosol and has almost no diurnal
structure — the Feb/Jun profiles varied by under 1 µg/m³ across the whole morning,
so the peak hour was noise. See the amendment note below.

**Statistic: the steepest hour-on-hour rise**, not the peak hour. NO₂ at MY1 has
no morning peak either — it climbs monotonically into the late afternoon
(February tops out at 17:00, June at 16:00), so an `idxmax` over a morning window
merely reports the window edge. The morning *rise* is sharp and unambiguous.

Mean NO₂ (µg/m³), 2019, hour-on-hour differences across the morning ramp:

| | 04→05 | 05→06 | 06→07 | 07→08 | 08→09 |
|---|---|---|---|---|---|
| **February (GMT), local hour** | +6.0 | +9.5 | **+11.4** | +5.9 | +2.7 |
| **June (BST), local hour** | +6.3 | +6.1 | **+11.5** | +6.3 | +6.5 |

Steepest rise at **06→07 local in both months**, magnitudes within 0.1 µg/m³.

In UTC hours the same comparison gives 06→07 for February and **05→06** for June
— one hour apart, which is what BST = UTC+1 requires. February's local and UTC
columns are byte-identical, as expected for a GMT month.

**All three lines of evidence agree.**

### Reproduce

`src/tz_check.py`, run against MY1 2020 and 2019.

```python
raw  = df["date"].astype("float64")
step = raw.diff()
utc  = pd.to_datetime(raw, unit="s", utc=True)
print(sorted(step.dropna().unique()), raw.duplicated().sum(), (step < 0).sum())

local = utc.dt.tz_convert("Europe/London")
prof  = df.groupby([local.dt.month, local.dt.hour])["NO2"].mean()
print(prof.loc[2].diff().loc[5:9], prof.loc[6].diff().loc[5:9])
```

---

## §1.1 — Incidental findings

**MY1 PM2.5 outage, autumn 2020.** Coverage by month in 2020: Jan 93%, Feb–Jun
99–100%, Jul 92%, Aug 97%, Sep 100%, **Oct 41%, Nov 0%, Dec 23%**. Weighted total
**78.7%**, below the ≥80% threshold in spec Part 5. This is one contiguous
instrument outage, not a generally unreliable site — every month from January to
September is 92–100%. Carried forward to the station audit (W1.6); the
distinction between "one autumn outage" and "patchy site" should be recorded
there.

**Column count drift.** MY1 2019 returns **51** columns; MY1 2020 returns **49** —
two columns dropped between consecutive years. Schema drift is therefore not
confined to the 2022-onwards change already noted.
`src/ingest.py` must reconcile columns explicitly across years rather than
concatenating — a naive `pd.concat` fills the mismatch with NaN and raises
nothing.

**`rdata` POSIXct warnings.** Every load emits
`UserWarning: Missing constructor for R class "POSIXct"`. Benign: `rdata` has no
converter for R's datetime class and returns the underlying numeric instead,
which is exactly why `date` arrives as float64 epoch seconds and is converted
manually. A side effect is that R's `tzone` attribute is discarded, so the
provider's stated intent is unavailable — not worth recovering, since three
independent lines of evidence already agree.

---

## Amendments to spec Part 6 arising from this check

Both are **results-seen**, unlike the 20–21 August changelog entries. The
diagnostic was changed *because* the PM2.5 and NO₂ profiles had been inspected.
They alter a sanity check on the instrument, not a hypothesis, a label definition
or a headline metric — but the distinction is recorded rather than glossed.

1. **Hour of day is grouped in `Europe/London`, not UTC.** Part 6 said "group by
   hour of day" without naming the clock. Under correct UTC storage, local-hour
   profiles align and UTC-hour profiles sit one hour apart; applying the stated
   pass condition to UTC hours would fail correct data. Consistent with the rule
   already in `docs/information_contract.md` that calendar features are derived
   in `Europe/London`.

2. **The alignment check uses NO₂ and the steepest morning rise**, not PM2.5 and
   the morning peak. Neither pollutant has a morning peak at MY1. PM2.5 has
   essentially no diurnal structure there at all, so the original statistic was
   reporting noise and could not have distinguished a one-hour bug from a
   one-hour seasonal effect. PM2.5 remains the forecasting target throughout;
   this is a change of diagnostic instrument only.

Both folded into `PROJECT_SPEC.md` Part 6 and its Changelog on 26 August 2026.

## §2 — Data freshness / publication lag (W1.4)

**Run:** 26 August 2026, 14:00 UTC. **Station:** MY1. **File:** `MY1_2026.RData`
(5,688 rows × 43 cols). **Snapshot date pinned:** 2026-08-26.

### Verdict

**Publication lag is 15 h, measured at 14:00 UTC. Both pollutants are fresh
together, so this is a genuine pipeline property, not an instrument outage.**

**The lag exceeds the 6-hour forecast horizon.** A forecast made from the
freshest available observation targets a time already 9 h in the past. AURN
therefore cannot support the forward-looking half of spec Part 12 — today's
forecast and its trust flag. It fully supports the retrospective half: a rolling
scoreboard of how the watcher's past calls turned out, scored after the fact,
which is the component Part 12 argues for. A genuinely live forecast page would
need Imperial's LAQN API (Part 3, residual concern 3). Decision deferred to
Week 6; the condition's outcome is recorded here.

### Measurements

| Pollutant | Last non-null (UTC) | Lag | Coverage, trailing 30 d |
|---|---|---|---|
| PM2.5 | 2026-08-25 23:00 | 15.0 h | 97.1% |
| NO₂ | 2026-08-25 23:00 | 15.0 h | 93.9% |

### The lag is a daily batch, not a rolling feed

The file ends at 23:00 UTC — the end of a UTC day, not an arbitrary recent hour.
DEFRA publishes in daily batches. The measured 15 h is therefore a snapshot of a
value that cycles: at 23:00 UTC the same file would read ~24 h stale.

Confirmed directly by a second run at 14:35 UTC, 37 minutes after the first. The
last non-null timestamp was unchanged at 2026-08-25 23:00, and the lag had grown
by exactly the elapsed time, 15.0 h → 15.6 h. No data arrived in the interval. A
rolling feed would have advanced; a batch feed does not, and this one did not.

The batch containing 25 August had arrived by 13:58 on 26 August, so it lands
somewhere in 00:00–13:58 UTC. Nothing for 26 August was published as of 14:35,
so the batch covers a complete UTC day and is issued only after that day closes.
Worst case at any hour is therefore **≤ 39 h** (15 + 24).

**Remaining unknown:** the batch hour itself is not pinned. A run early one
morning would bracket it. Not required for any decision currently open.

### Why two pollutants

PM2.5 and NO₂ come off different instruments but share a publication pipeline. A
single-pollutant measurement cannot separate a slow pipeline from a dead sensor —
both present as a large lag. Measured together they can: stale-together indicates
the pipeline, stale-alone indicates the instrument. MY1's autumn 2020 PM2.5
outage (§1.1) makes this a live possibility rather than a hypothetical.

Identical 15.0 h lags on both confirm the pipeline explanation.

### Consequence for the headline result

Residual features assume the truth at `s+6h` is available at `s+6h`. It is
available at `s+6h+N`. A deployed watcher therefore runs on **15–39 h staler**
residual information than the backtested one, depending on time of day.

This is **not leakage** — the backtest uses only genuinely past values. It is a
deployment gap, and spec Part 6 requires it be stated in the README with a number
rather than a caveat.

### Incidental

**Current-year file is truncated, not padded.** Last row timestamp equals last
non-null reading (2026-08-25 23:00). 5,688 rows = 237 × 24 exactly, for 1 January
to 25 August inclusive — still the complete regular grid found in §1, cut at the
last published hour rather than extended to 31 December. `date.max()` happens to
be correct here, but only by coincidence of convention; §2 measures the last
non-null regardless.

**MY1 PM2.5 is currently healthy.** 97.1% over the trailing 30 days, above NO₂'s
93.9%. Evidence that the 78.7% figure for 2020 (§1.1) reflects one contiguous
instrument outage rather than a chronically unreliable site. Carry to W1.6.

**Schema drift, continued.** Column counts by year at MY1: 2019 = 51, 2020 = 49,
2024 = 43, 2026 = 43. `src/ingest.py` must reconcile columns explicitly.

### Reproduce

`src/freshness_check.py`. Re-running gives a different lag — the value above is a
snapshot at the pinned date and stands unless a re-pull is recorded here.