# Station Selection (W1.6)

Run 2026-09-02 14:33 UTC. Thresholds pre-registered in `PROJECT_SPEC.md` changelog 2026-09-02, before this script was run.

**4 of 18 candidates kept.**

## Kept

| Site | Name | Type | PM2.5 pooled | PM2.5 worst year | NO2 pooled | Record |
|---|---|---|---|---|---|---|
| BEX | London Bexley | Suburban Background | 95.9% | 82.0% (2019) | 94.3% | 2018-01 to 2025-12 |
| HRL | London Harlington | Urban Industrial | 98.8% | 96.7% (2018) | 98.0% | 2018-01 to 2025-12 |
| KC1 | London N. Kensington | Urban Background | 99.6% | 98.4% (2022) | 99.0% | 2018-01 to 2025-12 |
| MY1 | London Marylebone Road | Urban Traffic | 89.4% | 78.7% (2020) | 96.3% | 2018-01 to 2025-12 |

## Rejected

| Site | Name | Reason |
|---|---|---|
| CA1 | Camden Kerbside | PM2.5 below 70% floor in 2021 63.8% |
| CLL2 | London Bloomsbury | PM2.5 below 70% floor in 2021 35.7%, 2022 60.4% |
| HG1 | Haringey Roadside | no PM2.5 readings 2018-2025 |
| HG4 | London Haringey Priory Park South | PM2.5 record starts 2025-03-04 15:00, after 2019-01-01 23:59 |
| HIL | London Hillingdon | PM2.5 record starts 2022-05-05 12:00, after 2019-01-01 23:59 |
| HORS | London Westminster | PM2.5 pooled 68.4% < 80%; PM2.5 below 70% floor in 2024 0.0%, 2025 11.6% |
| HP1 | London Honor Oak Park | NO2 pooled 0.0% < 70% |
| HR3 | London Harrow Stanmore | no AURN files 2018-2025 |
| LOFS | London Farringdon Street | PM2.5 record starts 2025-05-02 12:00, after 2019-01-01 23:59 |
| LON6 | London Eltham | PM2.5 below 70% floor in 2022 62.1% |
| LONC | London A406 N Circular | no AURN files 2018-2025 |
| LONM | London Norbury Manor School | PM2.5 record starts 2025-04-02 20:00, after 2019-01-01 23:59; NO2 pooled 0.0% < 70% |
| TED | London Teddington | no AURN files 2018-2025 |
| TED2 | London Teddington Bushy Park | PM2.5 below 70% floor in 2024 9.6%; NO2 pooled 0.0% < 70% |

## Method

Coverage is measured on **raw AURN data**, before the ingest-time
interpolation of sub-2-hour gaps. Measuring post-imputation would
inflate coverage unevenly: a station with many short gaps is
heavily patched and scores well, a station with one long outage is
not patched at all. That would favour intermittently broken sites
over briefly broken ones — the opposite of what is wanted, since
scattered gaps do more damage to lags and rolling windows than one
contiguous hole.

Denominator is the operational record of **that pollutant** at that
site, not the full 2018-2025 span. Scoring absent years as 0% would
reject every station commissioned after 2018 on grounds unrelated
to data quality.

A station whose PM2.5 record begins after 2019-01-01 is rejected
regardless of coverage: walk-forward fold 1 trains on 2018-2019, so
such a station contributes nothing to it and would fail silently in
Week 2 rather than here.
