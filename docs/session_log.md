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