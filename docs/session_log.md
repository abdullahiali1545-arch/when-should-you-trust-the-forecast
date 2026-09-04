2026-08-20 — Phase 0 audit + W1.1
Phase 0 was not clean: two live repos (personal + City OneDrive), no remote,
repo inside a sync client, PyCharm on a stale project and an unrelated .venv.
All fixed. Single repo C:\dev\when-should-you-trust-the-forecast, pushed to
GitHub (private), interpreter = envs\aq py3.12.13, env pinned via
environment.yml + requirements.txt.
W1.1 done: spec amended (rdata ingestion, H1 narrowed to COVID, three leakage
rules + canary test into Part 10, wind_speed_unit=ms), changelog added.
docs/information_contract.md written.
Verified live: MY1_2024.RData = 8784x43; AURN_metadata.RData = 3076x13 and
replaces importMeta() for W1.6; boundary_layer_height works on /v1/archive;
Open-Meteo defaults to km/h.
Understood: the "could I write it down at t?" test; the two clocks and .shift(6);
the fold test. Struggled with: which slice "training window" means per fold.
Unresolved: portfolio vs graded dissertation. Related-work section still needed
before Week 3.
Next: W1.2 — the April 2020 one-pager. One page, no notes open, no code.
Done w1.2 in domain notes explained why no2 fell and pm2.5 rose during covid era and the watcher system. forgot
to include it so i will i do it later . key



## 2026-08-21 — W1.3 complete: AURN clock verified as UTC

**Did:** Rewrote the Part 6 timezone test twice — the local-day row count cannot
discriminate, then an over-widened October condition was corrected. Ran the check
on MY1 2020 and 2019 via src/tz_check.py. Wrote docs/ingest_checks.md §1.

**Result:** AURN timestamps are genuine UTC instants. No conversion applied
anywhere in the pipeline. Evidence: constant 3600 steps, no repeats, no negatives
across both 2020 transitions on two independent files; 24 rows and 0% nulls on
29 March, ruling out null-padding; NO2 steepest morning rise at 06->07 local in
both a GMT and a BST month, one hour apart in UTC.

**Understand:** March skips a label, October repeats one — only March can open a
7200 gap, so March is load-bearing and October corroborates. A silent step test is
a failure to discriminate, not a confirmation of UTC. The 0% null rate is what
actually closes it.

**Struggled:** Two diagnostics turned out underpowered before one worked — PM2.5
has almost no diurnal structure at MY1, and neither pollutant has a morning peak,
so the spec's "peak hour" statistic was reporting noise. Also repeatedly lost
track of which spec copy is authoritative (PyCharm = real, project knowledge =
photocopy).

**Incidental:** MY1 PM2.5 2020 = 78.7%, one contiguous Oct-Dec outage, not a
patchy site. Raw column drift 2019 = 51 vs 2020 = 49.

**Unresolved:** Two Part 6 amendments still to fold in — group hour-of-day in
Europe/London, and use NO2 + steepest morning rise for the alignment check. Both
are RESULTS-SEEN, unlike the earlier entries. Listed at the end of
docs/ingest_checks.md.

**Next:** Fold in those two amendments, then W1.4 — freshness measurement.

## 2026-08-26 — W1.3 closed, W1.4 complete

Folded two results-seen amendments into PROJECT_SPEC.md: Part 6's diurnal check
now specifies Europe/London hour grouping and NO2 steepest morning rise, with a
dated changelog entry tagged results seen. Fixed "Both" -> "All three" lines of
evidence in ingest_checks.md. Commit 99bf990.

docs/information_contract.md IS on disk and committed (8,968 bytes, commit
8df7e3e). Earlier belief that it was never saved was wrong. Content not yet
audited against the spec's "names every deferred feature" requirement.

W1.4 done. AURN publication lag at MY1 = 15.0 h, pinned 2026-08-26 14:00 UTC.
PM2.5 and NO2 identically stale, so pipeline not instrument. Confirmed daily
batch: re-run 37 min later, timestamp unchanged, lag grew by elapsed time only.
Worst case <= 39 h. Lag exceeds the 6 h horizon, so AURN cannot support a live
forward forecast (Part 12) — retrospective scoreboard still fine, LAQN needed
for the forward half. Recorded in ingest_checks.md §2. Commit 06a8593.

Also found: 2026 file is truncated not padded (5,688 rows = 237 x 24 exactly);
MY1 PM2.5 currently 97.1% over 30 days, so 2020's 78.7% was one outage not a bad
site — carry to W1.6; schema drift continues, 2026 = 43 cols.

Understood: results-seen tagging; last row != last reading; lag as a deployment
gap rather than leakage; batch vs rolling feed.
Unresolved: the "it" in the W1.2 entry below, still unidentified. Half-finished
W1.2 note still sitting uncommitted in this file.
Next: audit docs/information_contract.md against spec Part 10, then src/ingest.py
(Week 1 Day 2).

## 2026-09-02 — W1.6 station selection (+ contract audit gate)

**Done**
- Contract audit closed: `docs/information_contract.md` reconciled against Spec Part 10.
  Split ingest-time interpolation (fold-independent) from fitted imputation
  (fold-dependent); added the out-of-fold burn-in note — watcher's first scoreable
  fold is one behind F3's. Commit `3b84db5`.
- Coverage threshold pre-registered (`2be7350`, marked "coverage figures seen"),
  then denominator specified per-pollutant with a record-start condition
  (`be6da4c`, "no selection results seen").
- `src/select_stations.py` written and run. 18 London AURN candidates → 4 kept:
  MY1 (urban traffic), KC1 (urban background), BEX (suburban background),
  HRL (urban industrial). Every rejection carries a number in
  `docs/station_selection.md`. Commit `461cb20`, pushed.
- Start-date boundary corrected mid-run to match the pre-registered rule.
  Outcome-neutral: HP1 fails on NO2 regardless. Recorded as such.

**Understood**
- Why the coverage denominator must be a full `date_range`, not `len(mask)`:
  a missing row is missing data, and `len(mask)` silently counts it as absent
  rather than as a gap.
- Why the threshold entry had to say "coverage figures seen" — the decision
  came after the numbers, and a reconstructed timestamp is worthless.

**Consequence to carry forward**
- CA1 rejected (63.8% PM2.5 coverage, 2021). MY1 is therefore the *only* traffic
  site in the final set. H4's leave-one-station-out fold now tests transfer to an
  unseen *site type*, which is harder than the spec assumed. FLAG IN WEEK 4 WRITE-UP.

**Unresolved**
- Do any London AURN sites report only PM2.5 sub-fractions (V2.5/NV2.5/GR2.5)
  and no combined PM2.5? Such a site vanishes from candidates with no rejection row.
- HP1, LONM, TED2 show NO2 pooled at exactly 0.0%. Genuinely no NO2 column, or a bug?
- Portfolio vs. graded-dissertation framing; related-work section placement.
- `boundary_layer_height` null for H1 2024 at MY1 — deferred to W1.6, status unknown.

**Next**
- W1.7 `src/features.py`. Fold-independent features only; check every candidate
  against `docs/information_contract.md` §5 before writing it.

## 2026-09-04 — W1.7 features.py

**Done**
- `src/features.py` written and run. 49 columns from MY1's processed Parquet.
  Fold-independent only: lags (0/1/3/6/12/24h), rolling mean+sigma (3/6/12/24h),
  deltas (1/3/6h), ERA5 weather at t, wind u/v, calendar at t and t+6h.
- Canary self-test passes: poisoning pm2_5 at t*=2022-01-01 leaves all 35,064
  earlier feature rows bit-identical. No feature reaches backwards in time.
- Decisions taken: gapless hourly reindex before rolling (row-based windows);
  min_periods = 75% of window; AURN's own ws/wd excluded in favour of ERA5;
  calendar features derived in Europe/London, not UTC.
- Output: 70,128 rows, 62,879 with a target, 53,342 fully non-null.

**Found**
- `boundary_layer_height` 100% null for 2024-01 to 2024-06. Recorded in
  `docs/ingest_checks.md`. Handling deferred to W2 as a pre-registration item.

**Unresolved**
- KC1, BEX, HRL selected in W1.6 but not yet in `data/processed`. W1's done
  condition requires all four.
- Line-by-line review of `features.py` still owed.

**Next**
- Ingest KC1, BEX, HRL. Then W1.8 EDA notebook.

## 2026-09-04 (cont.) — W1: all four stations ingested and featurised

**Done**
- KC1, BEX, HRL ingested. All four stations now in data/processed and
  data/features. 49 feature columns each, all canaries pass.
- Coverage (PM2.5 / NO2): KC1 99.6/99.2, HRL 98.9/98.4, BEX 96.0/94.4,
  MY1 89.7/96.6. Fully non-null feature rows: KC1 91.0%, HRL 90.0%,
  BEX 82.2%, MY1 76.1%.
- `ingest.py` allowlist split into required (date, pm2_5, no2) vs optional.
  Triggered by BEX lacking PM10/O3 in 2018 and O3 in 2025. Required absent
  still fails loudly; optional absent is NaN plus a log line.
- BLH null is 93.8% coverage at ALL FOUR stations, identical. The
  2024-01..2024-06 hole is an Open-Meteo ERA5 archive gap, not
  station-specific. A re-pull elsewhere will not fix it.

**Note for W4 write-up**
- MY1 is both the anchor and the weakest site (76.1% clean rows vs KC1's
  91.0%), and post-CA1 it is the only traffic site. Flag alongside the H4
  transfer-difficulty note.

**Unresolved**
- BLH handling: keep with documented hole, or substitute. W2 pre-registration
  item, must be logged before F3 is fitted.
- Line-by-line review of features.py still owed.

**Next**
- W1.8 EDA notebook, then W1.9 sql/schema.sql.
## 2026-09-04 (close) — W1 gate met

All four stations (MY1, KC1, BEX, HRL) are ingested and featurised at 49
columns each with passing canaries. Week 1's done condition — one command
loads clean, UTC-correct, feature-engineered data for every surviving
station — is met. Commits e3b9c77, 4f0b73f, db4c4c7.

Remaining in W1: W1.8 EDA notebook, W1.9 sql/schema.sql. Neither blocks W2.

Carried into W2:
- BLH null for 2024-01..2024-06 at all four stations (Open-Meteo ERA5
  archive gap). Handling is a pre-registration item, must be logged
  before F3 is fitted.
- MY1 is anchor, only traffic site, and weakest on coverage (76.1% clean
  rows). Flag with the CA1 note in W4.
- Line-by-line review of src/features.py still owed.

Next session: W1.8 EDA notebook.