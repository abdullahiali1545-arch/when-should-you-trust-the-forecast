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