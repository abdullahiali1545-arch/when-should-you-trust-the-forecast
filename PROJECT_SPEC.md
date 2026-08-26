# When Should You Trust the Forecast?

**Prospective detection of air-quality forecast unreliability in London, and whether routing distrusted forecasts to a simpler fallback produces a better system.**

`PROJECT_SPEC.md` — v2.0, 17 August 2026. Amended 20–26 August 2026; see the Changelog at the end. Commit this first. Do not edit it after Week 1 except to record decisions with dates.

---

## Part 0 — What this project is, in plain English

Air quality in London is measured hourly at fixed monitoring stations. You can build a model that looks at the last day of pollution and weather and predicts what the pollution will be six hours from now. That is ordinary and thousands of people have done it.

This project builds that model — and then builds a **second** model whose only job is to look at the first model and say *"I don't think you should trust this one."* It has to make that call **before** the real value arrives, using only information that exists at the moment of prediction.

Then comes the part that makes it an experiment rather than a demo. When the second model says "don't trust this," the system does not go silent — a real forecasting service still has to publish a number. Instead it **falls back** to a dumber, more robust method (e.g. "assume it stays the same as now"). The question is whether that switching behaviour makes the *whole system* better than always using the smart model.

And crucially: there is a boring explanation that would make all of this meaningless. Pollution episodes last for hours, so "the model was wrong an hour ago" already predicts "the model will be wrong now" — with no cleverness at all. **Most of the design below exists to rule out the boring explanation.**

### Who would use this

A six-hour PM2.5 forecast is consumed by someone making a threshold decision: a school deciding whether to hold PE outdoors, a person with asthma deciding whether to run, a local authority deciding whether to issue an alert. That framing appears in the optional decision layer (Part 11) and it is what makes the results legible to a non-technical reader.

---

## Part 1 — The research question and hypotheses

> **Can a system recognise, using only information available at prediction time, when its own forecast is about to be unusually wrong — and does routing those cases to a simpler fallback produce a better forecasting system overall?**

The forecaster is scaffolding. The watcher and the routing policy are the project.

| ID | Hypothesis | Status |
|---|---|---|
| **H1** | Forecast errors rise during the COVID lockdown period | Sanity check on the instrument. **Not a finding.** |
| **H2** | Past-only indicators predict elevated future error **beyond what R1 and R2 achieve** (Part 9) | **The core hypothesis** |
| **H3** | Routing distrusted forecasts to a fallback lowers total error at 100% coverage, vs always-ML and vs always-fallback | **The headline result** |
| **H4** | A watcher trained on some stations transfers to unseen stations (leave-one-station-out) | Generalisation |
| **H5** | Routing improves cost-weighted alert decisions under asymmetric costs | Optional extension |

Every one of these can fail informatively. Part 13 pre-registers what each failure would mean.

---

## Part 2 — Glossary (read once, then use freely)

| Term | Plain meaning |
|---|---|
| **Baseline** | The dumb method you must beat before anything you built counts as working. Here: "assume the pollution stays where it is." |
| **Persistence** | That specific dumb method: `ŷ(t+6h) = y(t)`. Surprisingly hard to beat at short horizons. |
| **Climatology** | The long-run average for this hour-of-day and month. The other dumb method. |
| **Residual** | The gap between what the model said and what actually happened: `y − ŷ`. |
| **Distribution shift** | The world changed after the model was trained, so patterns it learned no longer hold. April 2020 is the textbook case. |
| **Covariate shift** | A specific kind: the *inputs* now look unlike training inputs (unusual weather, unusual traffic), even if the underlying physics is unchanged. |
| **Data leakage** | Accidentally letting the model see information that would not have existed at prediction time. It makes results look excellent and be worthless. |
| **Walk-forward validation** | Train on the past, test on the future, roll forward, repeat. The only honest way to evaluate a time series. |
| **Selective prediction** | Letting a model decline to answer when it is unsure. Long-established idea (El-Yaniv & Wiener 2010; Geifman & El-Yaniv 2017). |
| **Routing / deferral** | This project's version: instead of declining, hand the case to a simpler model. |
| **Coverage** | The fraction of cases the main model is allowed to answer. |
| **Risk–coverage curve** | A plot of error-among-answered-cases against coverage. Lower and flatter is better. |
| **AURC** | Area Under the Risk–Coverage curve. One number summarising that plot. Lower is better. |
| **Calibration** | Whether a stated confidence matches reality — if it says 80% sure, it should be right 80% of the time. |
| **Ablation study** | Remove one component, re-run, see how much worse it gets. Tells you which parts actually mattered. |
| **Block bootstrap** | Estimating uncertainty by resampling *contiguous chunks* of time (e.g. whole weeks) many times. Blocks, not individual hours, because neighbouring hours are not independent. |
| **Reanalysis** | A reconstruction of past weather built afterwards using observations that were not available at the time. Not the same as a forecast. |

---

## Part 3 — What changed from v1, and why

**Review these nine changes before you start. If you disagree with any of them, argue it now, not in Week 4.**

| # | Change | Reason |
|---|---|---|
| 1 | **Abstention → routing.** Primary framing is now "switch to a fallback," not "output nothing" | "Abstain" is not a real action; a forecast service must publish something. Routing is operationally honest and gives a headline at 100% coverage, with no coverage caveat to explain |
| 2 | **Mode B redefined as past-only weather** (no future weather at all; climatology for anything forward-looking) | Archived *forecast* weather does not exist for the required period. Open-Meteo's historical forecast archive starts ~2021, and its fixed-lead-time archive mostly January 2024. April 2020 has none. Reanalysis at prediction time is a mild look-ahead |
| 3 | **Block-bootstrap confidence intervals on every headline comparison** | Without them, "the watcher beat R2 by 0.03 AURC" is unfalsifiable. The watcher's features are partly functions of the same autocorrelated signal R2 uses, so small wins are exactly what noise produces |
| 4 | **H4 becomes leave-one-station-out across 5–6 stations** | One held-out station is a sample of one. You cannot generalise from n=1 |
| 5 | **Timezone is verified empirically, not assumed** | v1 said AURN timestamps are local time and to convert. That is likely true for LAQN but probably wrong for AURN via `pyaurn`, which serves DEFRA's openair `.RData` files. Applying the "fix" to already-UTC data would create the exact silent one-hour bug it was meant to prevent |
| 6 | **Postgres and GitHub Actions moved from Week 1 to Week 5.** Parquet + `sql/schema.sql` first | Realistic Week 1 failure mode is four days lost to database setup with no science done. Same final repo, lower stalling risk. (PyCharm Professional's database tools reduce this pain but do not remove it) |
| 7 | **Final-year holdout, touched exactly once** | Otherwise you will tune the watcher against the same folds you report, and the result will be optimistic without you noticing |
| 8 | **COVID demoted from headline to validation section**; Mode A collapsed to a single "optimism gap" figure | A 2020 story told in 2026 is not a hook. The optimism-gap figure is more interesting than either evaluation mode alone |
| 9 | **Optional decision layer** (DAQI band crossings, asymmetric costs) and **optional live scoreboard** added as post-core extensions | MAE means nothing to a non-specialist. A live prospective log cannot be leaked into or quietly re-run — it is the one result that cannot be faked |

### Residual concerns I still have

Read these and accept them before starting.

1. **H2 has roughly a coin-flip chance of a null result.** R2 is a strong baseline for good physical reasons. This is fine — Part 13 makes the null publishable — but do not start expecting a win.
2. **Station coverage may force the plan down to four stations.** The Week 1 audit decides. Do not pre-commit to six.
3. **The live scoreboard depends on data freshness you have not yet measured.** `pyaurn` pulls yearly `.RData` files whose update lag is unknown to you. Measure it in Week 1 (Part 6). If the lag is long, the near-real-time fallback source is Imperial's LAQN API. Do not promise a live page before that check.
4. **The decision layer's threshold and cost ratio are arbitrary.** Sweep both and report across the sweep, or a reviewer will assume you picked the flattering values.

---

## Part 4 — System architecture

```
                    ┌──────────────────────────────────────┐
                    │  past pollution + past weather       │
                    │  (all strictly ≤ time t)             │
                    └───────────────┬──────────────────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             ▼                      ▼                      ▼
      ┌────────────┐        ┌──────────────┐       ┌──────────────┐
      │ FALLBACK   │        │ FORECASTER   │       │  WATCHER     │
      │ persistence│        │  LightGBM    │       │ features:    │
      │ climatology│        │              │       │ shift, vol., │
      └─────┬──────┘        └──────┬───────┘       │ disagreement,│
            │                      │               │ residuals    │
            │                      │               └──────┬───────┘
            │                      │                      │
            │                      │              trust / distrust
            │                      │                      │
            └──────────────┬───────┴──────────────────────┘
                           ▼
                 ┌───────────────────┐
                 │  ROUTING POLICY   │
                 │  trust → ML       │
                 │  distrust → fall  │
                 └─────────┬─────────┘
                           ▼
                  point forecast for t+6h
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │ EVALUATION                               │
        │ • MAE/RMSE at 100% coverage vs always-ML  │
        │ • risk–coverage curve + AURC              │
        │ • block-bootstrap CIs on all differences  │
        │ • (optional) cost-weighted alert decisions│
        └──────────────────────────────────────────┘
```

---

## Part 5 — Data

| Source | What | Access |
|---|---|---|
| AURN (via `rdata`) | Hourly NO₂, PM2.5, PM10, 2018–present | DEFRA openair `.RData` URLs, read directly |
| Open-Meteo archive (`/v1/archive`) | Hourly ERA5 weather matched by station lat/lon | Free, no key |

```python
# AURN ingestion via `rdata`. See the Changelog (2026-08-20) for why not pyaurn.
# Verified 2026-08-20: MY1_2024 returns 8,784 rows x 43 columns.
import urllib.request, pathlib, pandas as pd, rdata

def load_aurn(site: str, year: int) -> pd.DataFrame:
    url = f"https://uk-air.defra.gov.uk/openair/R_data/{site}_{year}.RData"
    tmp = pathlib.Path(f"{site}_{year}.RData")            # a plain Path, NOT
    tmp.write_bytes(urllib.request.urlopen(url).read())   # NamedTemporaryFile —
    df = rdata.read_rda(tmp)[f"{site}_{year}"]            # Windows holds an
    df["date"] = pd.to_datetime(df["date"], unit="s", utc=True)  # exclusive lock
    return df                                    # date arrives as float64 epoch s

# Site metadata — the importMeta() equivalent. Verified 2026-08-20: 3,076 rows
# x 13 cols. Columns: site_id, site_name, location_type, latitude, longitude,
# parameter, Parameter_name, start_date, end_date, ratified_to, zone, agglomeration
#   https://uk-air.defra.gov.uk/openair/R_data/AURN_metadata.RData
```

```
https://archive-api.open-meteo.com/v1/archive?latitude=51.52&longitude=-0.15
  &start_date=2018-01-01&end_date=2025-12-31
  &hourly=temperature_2m,relative_humidity_2m,pressure_msl,
          wind_speed_10m,wind_direction_10m,boundary_layer_height
  &timezone=UTC&wind_speed_unit=ms
```

**`wind_speed_unit=ms` is not optional.** Open-Meteo returns km/h by default. Omit it and every u/v component is 3.6× too large — nothing errors, F3 barely notices because trees are scale-invariant, but the distribution-distance features and every physical statement in the write-up are wrong. Assert on the returned `hourly_units` in `src/ingest.py` rather than trusting the default.

**Never branch on the API's `error` field.** Open-Meteo returns rate-limit failures and genuine bad-variable failures with the same `{"error": true, ...}` shape. Read `reason`: retry with backoff on a rate limit, fail loudly on anything else.

### Station selection is scripted, not chosen by hand

Do **not** hardcode a station list from memory. Write `src/select_stations.py` that:

1. Pulls `AURN_metadata.RData` (see above), filters to Greater London AURN sites.
2. Pulls 2018–2025 for each candidate.
3. Computes hourly coverage for PM2.5 and NO₂ per site.
4. Keeps sites with **≥80% coverage**; drops the rest.
5. Writes `docs/station_selection.md` — the list, the coverage numbers, and every site rejected with its reason.

Target 5–6 surviving stations spanning kerbside / urban background / suburban site types. MY1 (Marylebone Road) is your known kerbside anchor; take the others from the metadata, not from assumption. **A dropped station documented with a number is evidence of rigour. A dropped station documented with silence is a hole.**

---

## Part 6 — Ingest rules (Week 1, day one)

### Timezone — verify, do not assume

Pull one station across both 2020 clock changes — **29 March** (GMT→BST) and **25 October** (BST→GMT) — and inspect the **raw epoch-seconds `date` column**, not the rendered timestamps. Test whether consecutive raw values differ by a constant 3600 across each transition.

- **Constant 3600 throughout, no repeated or out-of-order values** → genuine UTC instants. Do nothing but `tz_localize('UTC')`.
- **A 7200 step at the March transition, or a repeated or out-of-order value at the October one** → local wall-clock time encoded as if UTC. Then and only then convert.

**March is the load-bearing test.** The two transitions are not symmetric. On 29 March a local-encoded series never carries the `01:00` label, because the clock jumps from 01:00 GMT to 02:00 BST — a *skipped* label, which opens a 7200 gap. On 25 October the `01:00` label is *repeated*, not skipped, so no gap can open there. March therefore fires whatever the provider does in October; October's repeat is corroboration, not a second independent test.

Two provider conventions produce no step signature at all, and both must be checked for rather than assumed absent. If October's repeated hour is **de-duplicated**, October's labels run 00, 01, 02 … at a constant 3600 with nothing to see. If March's missing hour is **padded with an all-null row** to preserve a regular grid, March reads 3600 throughout as well. So alongside the step diffs, record rows-per-day and the null rate for both transition dates. If the step test comes back silent on both, it has not confirmed UTC — it has failed to discriminate, and the diurnal-alignment check below is the only remaining evidence.

**Do not count rows in the local day.** The local day of 25 October 2020 is genuinely 25 hours long, so correctly-stored UTC data also yields 25 rows for it. That count cannot discriminate, and reading 25 as evidence of local storage would trigger the exact conversion this check exists to prevent. The annual row count is equally useless: a local-time year loses an hour in spring and gains one in autumn, netting to 8,784 in a leap year either way.

Control, measured 2026-08-20: Open-Meteo's archive with `timezone=UTC` returns exactly 24 rows for the UTC calendar date 2020-10-25, a single 01:00, spanning 00:00 to 23:00. The weather side is UTC-clean, and the slicing convention is by UTC calendar date.

Second, independent check — diurnal alignment. Group by **hour of day in `Europe/London`**, not UTC, for one GMT month and one BST month, and compare the hourly **NO₂** profile. The clock matters: human activity follows local time, so under correct UTC storage the local-hour profiles align while the UTC-hour profiles sit one hour apart. Applying the pass condition to UTC hours would fail correct data — the exact inversion this check exists to prevent.

Use the **steepest hour-on-hour morning rise**, not the morning peak. Neither pollutant peaks in the morning at MY1; both climb into the late afternoon, so an `idxmax` over a morning window returns the window edge rather than a feature. Use NO₂ rather than PM2.5: kerbside NO₂ is close to pure traffic exhaust and gives a sharp, human-scheduled ramp, whereas kerbside PM2.5 is dominated by non-exhaust wear and regional secondary aerosol and carries almost no diurnal structure. PM2.5 remains the forecasting target; this is the diagnostic instrument only.

Same rise hour in both months → aligned. One hour of misalignment → something shifted.

Write both results into `docs/ingest_checks.md` §1. Get this wrong and every diurnal feature silently shifts by an hour for half the year — the kind of bug that never announces itself and quietly poisons everything downstream.

### Data freshness — measure it now

Pull the current year and find the timestamp of the most recent non-null observation. Record the lag in `docs/ingest_checks.md`. This number decides whether the live scoreboard (Part 12) is feasible from AURN or needs the LAQN API instead. **Do this in Week 1, not Week 6.**

It is also an honesty constraint on the headline result: residual features assume the truth at `s+6h` is available at `s+6h`. If the measured publication lag is N hours, a live deployment carries N hours less residual information than the backtest. State that in the README.

### Imputation

- Gaps **under 2 hours**: linear interpolation, flagged in an `imputed` column.
- Gaps **of 2 hours or more**: leave as NaN. Never interpolate across a long gap and then compute a rolling statistic over it.

### Ratification

Reference data is ratified annually, usually the following April. Recent data is provisional and can change. **Pin a snapshot date, record it in the README, and re-pull only deliberately.** `AURN_metadata.RData` carries a `ratified_to` column per site — use it rather than guessing.

---

## Part 7 — The forecaster

Target: **PM2.5 at t+6h**, single horizon. Multi-horizon is phase two and stays cut.

| Model | Role |
|---|---|
| **F0 — Persistence** `ŷ(t+6) = y(t)` | The bar, and the routing fallback |
| **F1 — Climatology** (mean by hour-of-day × month) | The other fallback candidate |
| **F2 — Linear / autoregressive, weather-conditioned** | Statistical reference; the disagreement partner for the watcher |
| **F3 — LightGBM** | The ML model under scrutiny |

**No neural networks.** Gradient boosting is the right tool at this data size. Not reaching for a bigger hammer is a strength; say so in the README.

**Features:** lagged PM2.5 and NO₂ (t, t−1, t−3, t−6, t−12, t−24); rolling means, standard deviations, rates of change; temperature, humidity, pressure, boundary layer height; wind as **u/v components** (`u = speed·cos θ`, `v = speed·sin θ` — direction is circular, 359° and 1° are adjacent, so raw degrees are a trap); hour, day of week, month, weekend flag.

**F2 goes in during Week 2, not Week 4.** Much of day-to-day pollution variance is wind, boundary layer and temperature. When a watcher indicator fires, the first question a reviewer asks is whether it merely detected unusual weather. Without a weather-conditioned reference in place early, H2 stays ambiguous permanently.

---

## Part 8 — Defining "high error" (the decision that matters most)

**Do not rank by raw absolute error.** Absolute error scales with concentration: predicting 40 when truth is 55 gives an error of 15; predicting 12 when truth is 15 gives 3. Rank raw errors and your "unreliable" class silently becomes "high pollution episodes" — the watcher would then be solving a different, easier, already-solved problem, H2 would appear to succeed, and the success would be meaningless.

**Primary definition — stratified.** Bin cases by *predicted* concentration (deciles). Within each bin, the top 20% of absolute errors are labelled unreliable. Bins come from predicted, not actual, values — actual values are not available at prediction time.

**Thresholds are fitted parameters.** Both the decile edges and the within-bin 80th percentile are estimated from data, so both are fitted on training folds only and applied frozen to the test fold. See Part 10.

**Robustness check — relative error.** `|y − ŷ| / max(ŷ, floor)`, floor chosen to stop small denominators exploding. Re-run the headline result under this definition. If the conclusion flips, that is itself a finding and belongs in the README.

Document the reasoning. **This paragraph will be probed in interview, and catching the problem yourself is worth more than whatever the result turns out to be.**

---

## Part 9 — The watcher and the routing policy

### Watcher features (all computed from information available at time t)

| Family | Examples | What it is meant to catch |
|---|---|---|
| Distribution distance | PSI or Wasserstein distance, recent window vs training window | Covariate shift |
| Volatility | Rolling σ, rolling absolute change | Unstable regimes |
| Model disagreement | \|ŷ_F3 − ŷ_F2\| | Extrapolation. Needs no ground truth |
| Recent residuals | Rolling MAE, bias, residual volatility — **only from timestamps whose outcome has already arrived** | Strongest single signal; also exactly what R2 exploits |

The ablation study exists mainly to answer one question: **does anything beyond recent residuals contribute at all?**

### Routing rules — the whole point of the design

| ID | Rule |
|---|---|
| **R0** | Route at random, coverage-matched |
| **R1** | Route when predicted concentration is high |
| **R2** | Route when recent absolute error was high |
| **R3** | Route on the full watcher |

Plus the two trivial endpoints: **always-ML** and **always-fallback**.

> **The null hypothesis is that this entire project reduces to "pollution episodes are autocorrelated."**
> If R3 does not beat R1 and R2 with a confidence interval that excludes zero, the project's answer is negative — and that is a legitimate, reportable, interview-proof result.

---

## Part 10 — Evaluation protocol

### Walk-forward only

Train on 2018–2019, test on the next quarter, roll the window forward, refit every fold. **Never `train_test_split` on a time series.**

### Final holdout, touched once

Hold out the most recent full year entirely. Do not look at it, plot it, or compute anything on it while developing. Run it **once**, at the end, and report whatever it says — including if it disagrees with your walk-forward result. Record the date you touched it.

### Two evaluation modes

- **Mode A (hindcast)** — uses observed weather at prediction time. Diagnostic only. Produces exactly one figure: the **optimism gap**, i.e. how much better your numbers look when you cheat slightly. Nothing else.
- **Mode B (prospective)** — **no future weather of any kind.** Only weather observed up to *t*, plus climatological values for anything forward-looking. **This is the headline.**

### Metrics

| Question | Metric |
|---|---|
| Is the routed system better overall? | MAE and RMSE at 100% coverage vs always-ML and always-fallback |
| Is the watcher a good detector? | PR-AUC (not plain accuracy — the unreliable class is a minority) |
| Selective view | Risk–coverage curve and AURC |
| Is any difference real? | **Block bootstrap: week-long blocks, ≥1000 resamples, report the CI on every difference** |

**Win condition, stated in advance:** a difference counts only if its bootstrap confidence interval excludes zero. Anything else is reported as "no detectable difference." Write this sentence in the README before you have results.

### Leakage rules

Information used at time *t* ⊆ information available by time *t*. Specifically:

- Walk-forward evaluation only.
- Scalers and imputation fitted on the training window only, refitted each fold.
- No rolling statistic that reaches across the forecast origin.
- Distribution distances computed against the training window, not the full series.
- Residual features only from timestamps whose outcome has already occurred.
- No future weather in Mode B — including "today's" weather at hour t+3.
- **The watcher trains only on out-of-fold F3 residuals.** In-sample residuals are systematically too small and differently shaped. Consequence: the watcher's first scoreable fold is one behind F3's — one quarter of burn-in.
- **Stratified-label thresholds are fitted on training folds and applied frozen** to the test fold. The realised positive rate in a test fold will therefore drift from 20%; that drift is reported, not corrected — it measures shift in the error process.
- **The fold test.** Any feature whose value at a fixed timestamp would change if a fold boundary moved is built inside the walk-forward harness, not in `src/features.py`. This covers scalers, climatology means, model disagreement, residual features, distribution distances and label thresholds. See `docs/information_contract.md`.

### The canary test

Correct output and leaked output look identical — both produce a fold table with plausible MAEs, no error, no warning. So the harness cannot be validated by inspecting its output.

Corrupt one input series at a single timestamp `t*` with a sentinel value, rebuild the features, and assert that every row with origin earlier than `t*` is bit-identical to before. If an earlier row changed, information moved backwards in time. Run it on the label-construction function too, not just the features. Build it in the same session as the harness.

---

## Part 11 — Optional decision layer (H5)

Convert the forecast into the decision a real user makes: does PM2.5 cross a DAQI band boundary in six hours — alert or no alert?

Then weight the errors asymmetrically: a **missed** alert costs more than a false alarm, because the consequence of a miss is someone with asthma going for a run. Report the routed system's cost against always-ML **across a sweep of cost ratios and thresholds**, not at one flattering point.

This is what makes the result legible to a non-specialist reader, and it maps directly onto how threshold-and-cost problems work in energy demand and insurance risk.

---

## Part 12 — Optional live scoreboard

Conditional on the freshness check in Part 6.

A single page (Streamlit — you have done this before) showing:

- Today's forecast for each station
- The trust flag and which model was routed to
- **A rolling scoreboard of how the watcher's past calls actually turned out**, scored after the fact

Why this outranks any modelling addition: a backtest can be leaked into, tuned, or quietly re-run until it looks good. **A prospective public log cannot.** Three weeks of live scoring earns you a sentence nobody else has — *"in live operation since [date], hours the watcher flagged had N× the error of hours it didn't"* — built from data that did not exist when the model was trained.

It also gives you a URL, which matters in the one channel where portfolios actually get opened: cold email to a small company.

---

## Part 13 — What counts as success and failure

Success is **not** "the watcher works." Each of these is a complete, honest project:

| Outcome | What it means | Reportable? |
|---|---|---|
| R3 beats R1 and R2, CI excludes zero | Prospective reliability detection is feasible from surface data | Yes — positive result |
| R3 matches R1/R2, CI includes zero | Distributional unfamiliarity adds nothing beyond autocorrelation | **Yes — honest negative result** |
| Routing beats always-ML but only via R2 | The useful signal is recent error, full stop. Simpler than expected | Yes — and more useful than a complicated win |
| H1 fails; errors do not rise at the shocks | The assumed shifts do not damage this particular forecasting relationship | Yes — and genuinely surprising |
| Everything is null | The forecaster is well-behaved and there is nothing to detect | Yes, if you show the evaluation was capable of detecting an effect |

### What you may claim

A rigorous, leakage-controlled, baseline-anchored application of selective prediction and model routing to a real forecasting problem under documented distribution shifts.

### What you may **not** claim

A new method. A novel research question. "The first system to detect its own unreliability." Selective prediction is over fifteen years old and conformal prediction has already been applied to PM2.5. Overclaiming is the fastest way to lose a technically literate reader. **Say plainly in the README that the framework is established and your contribution is the rigour of the comparison.**

---

## Part 14 — Scope control

### CORE — required to answer the question

Ingest one station → walk-forward harness → F0/F1/F2/F3 → stratified error labels → watcher + R0/R1/R2/R3 → routed vs always-ML at 100% coverage → risk–coverage curve → **block-bootstrap intervals**.

This is shippable on its own and is already a strong project.

### EXTENSIONS — only if the core works

Leave-one-station-out (H4) · ablation study · COVID validation section · Postgres + GitHub Actions refresh · decision layer (H5) · live scoreboard.

### CUT — do not build

ULEZ stress test · multiple forecast horizons · neural networks · seven notebooks · PDF report · exploratory "identify periods of distribution shift" phase · AERONET/column data (see `docs/data_audit.md`) · night-time/stellar targets.

---

## Part 15 — Timeline

~3 hours/day. **Treat these as milestones, not calendar weeks.** Between final-year work and shifts, a 6-week plan that quietly becomes 10 is worse than a 3-week core that ships.

### Phase 0 — Setup (2 days)

- PyCharm installed, student Professional licence applied for, conda env set as project interpreter
- Repo created, this file committed **first**
- `pyaurn` and Open-Meteo pulls both returning data

### Week 1 — Data foundation

| Day | Work |
|---|---|
| 1 | Domain one-pager (see Part 16). Timezone check. Freshness check. Both written to `docs/ingest_checks.md` |
| 2 | `src/ingest.py`: AURN + Open-Meteo → Parquet. UTC throughout |
| 3 | `src/select_stations.py` → `docs/station_selection.md`. Coverage audit decides 4, 5 or 6 stations |
| 4 | `src/features.py`: lags, rolling stats, u/v wind, calendar features. **Fold-independent features only** — see Part 10 |
| 5 | `sql/schema.sql` written (not yet deployed). EDA notebook: diurnal cycles, seasonality, missingness map |

**Done when:** one command loads clean, UTC-correct, feature-engineered data for every surviving station, and `docs/information_contract.md` names every deferred feature with the fold dependency that deferred it.

### Week 2 — Forecaster and harness

- Walk-forward harness in `src/evaluate.py` — build this before any model, because everything downstream depends on it being right. Canary test first
- F0, F1, F2 implemented and scored
- F3 (LightGBM) implemented and scored
- Mode A vs Mode B comparison → the optimism-gap figure

**Done when:** you can state, with numbers, whether F3 beats persistence in Mode B — and if it doesn't, why.

### Week 3 — Watcher, routing, and the first shippable result

- Stratified error labelling (Part 8), plus the relative-error robustness check
- Watcher features and classifier
- R0/R1/R2/R3 implemented
- Routed vs always-ML vs always-fallback at 100% coverage
- Risk–coverage curve and AURC
- Block-bootstrap CIs on every difference

> ### ⏹ STOP HERE AND SHIP
> Write the README. Push it. This is a complete project. Everything below is an extension, and none of it is worth delaying this milestone for.

### Week 4 — Rigour

- Leave-one-station-out (H4)
- Ablation: watcher with and without each feature family
- COVID validation section (H1)
- Final holdout run — **once**

### Week 5 — Engineering and write-up

- Deploy Postgres, load from Parquet using `sql/schema.sql`
- GitHub Actions weekly refresh
- README rewritten properly (Part 16)
- All figures regenerable by one command

### Week 6 — Optional

- Live scoreboard (only if the Part 6 freshness check passed)
- Decision layer (H5)

---

## Part 16 — The README is the product

Reviewers spend minutes, not hours. Write the README in this order:

1. **One sentence of result, at the very top.** Not the question — the answer.
2. **One figure.** The routed-vs-always-ML comparison with confidence intervals.
3. **The question**, in three sentences.
4. **What I built**, with the architecture diagram.
5. **The hardest decision** — the error-definition problem in Part 8, and why the obvious approach would have produced a meaningless success.
6. **What I found, including what didn't work.** Negative results get their own heading, not a footnote.
7. **What I cannot claim** (Part 13).
8. **Reproduce it** — one command.
9. **What I'd do differently.**

### Week 1, Day 1 writing exercise

Before any code, write one page in your own words answering:

> **Why would a model trained on 2018–19 London air have been confused in April 2020 — and why did it fail in *opposite directions* for NO₂ and PM2.5?**

(Traffic fell sharply and surface NO₂ fell with it, so a model expecting a normal Tuesday over-predicted. Meanwhile PM2.5 rose, because easterly winds carried continental aerosol into the city. Same model, same month, wrong in both directions.)

If you can explain that unaided, you understand distribution shift better than most people who use the phrase — and this page becomes the domain-intuition section of your README.

---

## Part 17 — Repo structure

```
when-should-you-trust-the-forecast/
├── README.md                      ← the product
├── PROJECT_SPEC.md                ← this file, committed first
├── environment.yml                ← conda interpreter
├── requirements.txt               ← pip packages
├── docs/
│   ├── ingest_checks.md           ← timezone + freshness results
│   ├── information_contract.md    ← what each model may read at time t
│   ├── station_selection.md       ← coverage audit, including rejections
│   ├── session_log.md             ← continuity between working sessions
│   └── data_audit.md              ← the AERONET audit; a killed hypothesis is evidence
├── src/
│   ├── ingest.py
│   ├── select_stations.py
│   ├── features.py                ← fold-independent features only
│   ├── forecast.py                ← F0–F3
│   ├── watcher.py                 ← features + classifier
│   ├── routing.py                 ← R0–R3, fallback policy
│   └── evaluate.py                ← walk-forward, canary test, risk–coverage, bootstrap
├── notebooks/
│   ├── 01_exploration.ipynb       ← scratchpad, not a deliverable
│   ├── 02_baselines.ipynb
│   └── 03_results.ipynb
├── sql/schema.sql
├── .github/workflows/refresh.yml
└── results/figures/
```

Notebooks are where you think. `src/` is what you are judged on.

---

## Part 18 — Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| H2 returns null; watcher doesn't beat R2 | **~50%** | Pre-registered in Part 13 as a valid result. Accept before starting |
| Coverage audit leaves fewer stations than hoped | Medium | Plan works at 4 stations; LOSO just has fewer folds |
| Timeline slips past 6 weeks | High | Ship at the Week 3 gate regardless |
| Tuning against reported folds | Medium | Final-year holdout, touched once |
| Postgres setup consumes days | Medium | Deferred to Week 5; Parquet works throughout |
| Live scoreboard blocked by data lag | Unknown until Week 1 | Freshness check in Part 6; LAQN API as fallback source |
| Scope creep from something that "sounds impressive" | High | Part 14 CUT list. Adding to it requires deleting something else |
| Silent leakage in the harness | High | Canary test, Part 10. Correct and leaked output are visually identical |

---

## Part 19 — Settled. Do not reopen.

These were decided on evidence. Reopening them costs time and changes nothing.

- **No column/AERONET layer.** The London record does not exist for the required years. Numbers in `docs/data_audit.md`.
- **No stars.** No night-time target at usable cadence.
- **No deep learning.** Gradient boosting is correct here; a neural network would make this a worse project, not a more impressive one.
- **Single horizon (6h) for v1.**
- **Pollution is the domain.** Locked 10 August 2026, re-confirmed 17 August 2026. The discovery phase is closed.

---

## The one-sentence version

> I built a London air-quality forecaster and a second system that tries to recognise, before the truth arrives, when the first one is about to be wrong — then routed the distrusted cases to a simple fallback and tested, with confidence intervals, whether the combined system beats always using the model, and whether it beats the trivial rule of just switching whenever pollution is high or the last forecast was bad.

That last clause is the one that makes it an experiment.

---

## Changelog

Amendments after the initial commit (f198a05, 2026-08-17). Each entry states whether results bearing on the decision had been seen at the time.

Parts 3, 15 and the residual concerns still refer to `pyaurn`. That text is left standing deliberately: it records what was believed on 17 August and what Phase 0 actually did. Body text is amended where stale wording would cause the wrong action; it is left alone where it is history.

### 2026-08-20 — Data access (plumbing; no results seen)

Part 5's `pyaurn.importAURN` route did not work in this environment against the current DEFRA files. pyaurn is unmaintained: last release 0.1.21, 2023-03-10. Proximate cause: DEFRA serves RDX3-serialised `.RData`, which the reader in pyaurn's `pyreadr` dependency did not read here. Note this was pyreadr 0.5.6 (released 2026-04-13, the current version), so the failure is not a matter of a stale reader.

Ingestion reimplemented with the pure-Python `rdata` package (1.1.0) reading DEFRA URLs directly. Part 5's source table and code block updated to match. Verified 2026-08-20: `MY1_2024.RData` returns 8,784 rows × 43 columns; `AURN_metadata.RData` returns 3,076 rows × 13 columns and serves as the `importMeta()` replacement for `src/select_stations.py`. Hypotheses, metrics and analysis plan unchanged.

### 2026-08-20 — H1 narrowed (pre-registered content; no results seen)

ULEZ struck from H1. H1 now reads: *Forecast errors rise during the COVID lockdown period.*

Reasons: Part 14 already CUTs the ULEZ stress test, so the original wording contradicted the scope control. ULEZ's effect on PM2.5 specifically is weak and contested — non-exhaust brake and tyre wear dominates kerbside PM2.5 and a low-emission zone does little about it. Its effect also cannot easily be separated from concurrent trends. H1 exists as a sanity check on the instrument, and a sanity check with a weak expected signal cannot function as a sanity check.

### 2026-08-20 — Three leakage rules added to Part 10 (clarification; no results seen)

Part 10 previously mandated walk-forward evaluation and per-fold refitting but did not say where the watcher's training labels come from, nor where the stratified-label thresholds are fitted. Both gaps admit leakage that produces plausible output and no error.

(a) The watcher trains only on out-of-fold F3 residuals. Consequence: its first scoreable fold is one behind F3's — one quarter of burn-in.

(b) Stratified-label thresholds — decile edges of predicted concentration, and the within-bin 80th percentile of absolute error — are fitted on training folds and applied frozen to the test fold. The realised positive rate in a test fold will therefore drift from 20%; that drift is reported, not corrected.

(c) The fold test: any feature whose value at a fixed timestamp would change if a fold boundary moved is built inside the walk-forward harness, not in `src/features.py`. Three of the watcher's four feature families fall on the harness side. See `docs/information_contract.md`.

Also added to Part 10: the canary test, as the validation method for the harness itself.

### 2026-08-20 — Open-Meteo request parameters (plumbing; no results seen)

`wind_speed_unit=ms` added to the Part 5 request. Open-Meteo defaults to km/h; omitting the parameter makes every u/v component 3.6× too large with no error raised. `src/ingest.py` asserts on the returned `hourly_units` rather than trusting the default, and keys its retry logic on the `reason` field, since rate-limit and bad-variable failures share the same `{"error": true}` shape.

`boundary_layer_height` verified working on `/v1/archive` for 2018 at MY1's coordinates, returning metres. The earlier failure was a rate-limit response misread as an unsupported variable.

### 2026-08-21 — Part 6 timezone test rewritten (correction; no results seen)

The v2.0 test — "count rows in that local day; 24 → UTC, 25 → local" — cannot discriminate as worded. The local day of 25 October 2020 contains 25 real hours, so a correctly-stored UTC series sliced by that local day also returns 25 rows. Under the stated rule that reads as evidence of local storage, and would trigger the timezone conversion the check exists to prevent. The test is sound only if "that local day" is read as "that UTC calendar date", which the wording does not say.

Replaced with a test on the raw epoch-seconds column: consecutive differences constant at 3600 across both 2020 transitions → UTC; a repeated value at the October transition, or a 7200 step at the March one → local wall clock. This is unambiguous under either slicing convention. The diurnal-alignment comparison is added as a second, independent line of evidence, since the raw-step test establishes what the data is but not whether the handling of it is correct.

The Part 6 control paragraph is retained, with its slicing convention stated explicitly as UTC calendar date.

Written before the check was run. No AURN timezone result had been seen.

### 2026-08-21 — Part 6 failure conditions completed (correction; no results seen)

The entry above paired a repeated epoch value at the October transition with a 7200 step at the March one, and implicitly treated the two as symmetric tests. They are not, and two gaps followed.

First, only March can produce a gap. A local-encoded series skips the `01:00` label on 29 March, so a 7200 step appears whatever the provider does in October, where the label is repeated rather than skipped. March is load-bearing; October corroborates. An earlier draft of this entry proposed widening the October condition to include a 7200 step, on the reasoning that providers which drop the repeated hour would produce one. That reasoning is wrong: dropping the repeat yields a contiguous 24-label day at a constant 3600 and no signature whatsoever.

Second, and this is the real gap, two conventions defeat the step test entirely — de-duplicating October's repeated hour, and padding March's missing hour with an all-null row. Under either, the affected transition reads as clean UTC. Part 6 now requires rows-per-day and null rate on both transition dates alongside the step diffs, and states explicitly that a silent step test is a failure to discriminate rather than a confirmation of UTC, leaving the diurnal-alignment check as the only remaining evidence.

Out-of-order (negative) steps added as a third local-clock signature. A logger whose clock is set to local time can emit timestamps that run backwards at the October transition. `diff()` already computes this, so the check is free.

Written before the check was run.

### 2026-08-26 — Part 6 diurnal-alignment check respecified (correction; **results seen**)

Unlike every entry above, this one was written **after** the data had been inspected. The check ran on 21 August against MY1 2019 and 2020, and both amendments below were made *because* the profiles had been looked at. They change a diagnostic instrument, not a hypothesis, a label definition or a headline metric — but the distinction is tagged rather than glossed.

(a) **Hour of day is grouped in `Europe/London`, not UTC.** Part 6 said "group by hour of day" without naming the clock. AURN timestamps are true UTC instants (verified; `docs/ingest_checks.md` §1), while the signal under test is human activity, which follows the local clock. Under correct storage the local-hour profiles align across a GMT and a BST month and the UTC-hour profiles sit one hour apart — so applying Part 6's stated pass condition to UTC hours would fail correct data, the exact inversion the check exists to prevent. Consistent with `docs/information_contract.md`, which already derives calendar features in `Europe/London`.

(b) **The check uses NO₂ and the steepest hour-on-hour morning rise**, not PM2.5 and the morning peak. Neither pollutant peaks in the morning at MY1; both climb into the late afternoon (February tops out at 17:00, June at 16:00), so an `idxmax` over a morning window reports the window edge rather than a feature. PM2.5 is worse still — at this kerbside site it is dominated by non-exhaust wear and regional secondary aerosol, and the February/June profiles differed by under 1 µg/m³ across the whole morning. The original statistic was reporting noise and could not have separated a one-hour bug from a one-hour seasonal effect. PM2.5 remains the forecasting target throughout.

Result, for the record: steepest rise at 06→07 local in both months, magnitudes within 0.1 µg/m³; in UTC hours the two months sit one hour apart, as BST = UTC+1 requires. Verdict unchanged from the raw-step test — AURN timestamps are UTC.