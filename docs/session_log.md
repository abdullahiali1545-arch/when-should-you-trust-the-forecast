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