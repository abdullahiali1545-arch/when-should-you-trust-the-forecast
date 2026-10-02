# When Should You Trust the Forecast?

A six-hour PM2.5 forecaster for London, plus a second model that tries to spot when the forecast is about to be badly wrong.

**[Live map](https://abdullahiali1545-arch.github.io/when-should-you-trust-the-forecast/)** · [Live log](https://github.com/abdullahiali1545-arch/when-should-you-trust-the-forecast/tree/live-log/data/live) · Abdulahi Ali, BSc Mathematics and Data Science, City St George's, University of London · Aug–Sep 2026

## Result

It didn't work, and the reason is clear.

I trained a "watcher" to flag hours where my LightGBM forecaster (F3) was likely to have a large error, and sent those hours to persistence instead (persistence assumes PM2.5 in six hours will equal PM2.5 now). Switching made the overall forecast worse at all four stations I tested and on a 2025 holdout year that I ran once.

The watcher did find F3's bad hours more often than chance, at 1.12–1.16× the base rate. The problem was where it sent them. Flagged hours tend to be volatile, and persistence does even worse than F3 when pollution is changing quickly. With perfect hindsight, switching 40% of hours would have cut MAE at Marylebone Road from 3.514 to 2.866, so useful routing exists. Persistence just wasn't the right fallback.

![Headline result](results/figures/MY1_headline.png)

*MY1 (Marylebone Road). Each point is a difference in mean absolute error (MAE) against always using F3, with a 95% block-bootstrap interval. Right of zero is worse. Hollow grey points have intervals that include zero, so there is no detectable difference.*

## The question

> Can a system recognise, using only information available at prediction time, when its own forecast is about to be unusually wrong, and does sending those cases to a simpler fallback give a better forecast overall?

There is a cheap explanation that could make any "unreliability detector" look good. Pollution episodes last for hours, so a forecast that was wrong recently will probably be wrong now. To rule that out, the watcher had to beat two simple switching rules, as well as random switching:

| Rule | Switches to persistence when… |
|---|---|
| R0 | chosen at random (control) |
| R1 | F3's predicted PM2.5 is high |
| R2 | F3's error over the last 24 hours is high |
| R3 | the watcher says an unusually large error is likely |

Every rule switches the same share of hours (20% per fold), so they are compared on equal terms. I fixed the success criterion before running the comparison: a difference only counts if its 95% interval excludes zero.

## Results

### Forecasters (MY1, 35,799 out-of-fold hours)

| Model | MAE (µg/m³) |
|---|---|
| F0 persistence | 4.33 |
| F1 climatology (average for that hour and month) | 6.12 |
| F2 ridge regression | 3.85 |
| **F3 LightGBM** | **3.58** |

F3 beat persistence by 17% and ridge by 7%, and beat persistence in every quarterly fold except 2020 Q1. At the other three stations it beat persistence by about 16%. So the forecaster being judged is a reasonable one.

### Routing (MY1, 31,572 hours)

| System | MAE | vs always F3 [95% interval] |
|---|---|---|
| **Always F3** | **3.514** | — |
| R0 random | 3.682 | +0.168 [+0.148, +0.190] |
| R3 watcher | 3.772 | +0.257 [+0.209, +0.310] |
| R2 recent error | 3.803 | +0.289 [+0.227, +0.361] |
| R1 high PM2.5 | 3.819 | +0.305 [+0.240, +0.373] |
| Always persistence | 4.367 | +0.852 [+0.778, +0.924] |

Against the simple rules, the watcher showed no detectable difference from R2 (−0.031 [−0.073, +0.010]). It was slightly better than R1 (−0.047 [−0.091, −0.003]), and worse than random (+0.090 [+0.047, +0.134]). As a detector on its own, its mean PR-AUC was 0.207 against a no-skill baseline of 0.186, ahead in 16 of 18 folds.

### 2025 holdout (MY1, run once)

| Comparison | Walk-forward 2020–24 | Holdout 2025 |
|---|---|---|
| R3 vs always F3 | +0.257 [+0.209, +0.310] | +0.283 [+0.153, +0.409] |
| R3 vs R2 | −0.031 [−0.073, +0.010] | +0.026 [−0.042, +0.097] |
| R3 vs R1 | −0.047 [−0.091, −0.003] | −0.004 [−0.074, +0.065] |
| R3 vs R0 | +0.090 [+0.047, +0.134] | +0.076 [−0.032, +0.188] |

The main result held. The small R3-over-R1 edge did not repeat, so I don't count it as a finding. With 53 weeks of data instead of 219, the intervals are wider.

### Other stations (walk-forward, same pipeline, no settings changed)

| Station | Always F3 MAE | R3 vs always F3 | R3 vs R2 | R3 vs R1 | R3 vs R0 |
|---|---|---|---|---|---|
| MY1 | 3.514 | +0.257 [+0.209, +0.310] | −0.031 [−0.073, +0.010] | −0.047 [−0.091, −0.003] | +0.090 [+0.047, +0.134] |
| KC1 | 2.669 | +0.192 [+0.146, +0.244] | −0.053 [−0.088, −0.018] | −0.050 [−0.085, −0.016] | +0.081 [+0.043, +0.122] |
| BEX | 3.084 | +0.199 [+0.156, +0.249] | −0.102 [−0.161, −0.050] | −0.112 [−0.165, −0.062] | +0.080 [+0.040, +0.123] |
| HRL | 2.515 | +0.162 [+0.124, +0.207] | −0.061 [−0.095, −0.028] | −0.053 [−0.084, −0.024] | +0.063 [+0.031, +0.098] |

At KC1, BEX and HRL the watcher beat R1 and R2, but every rule was still worse than always using F3, and random was the least bad. So at those stations "beat R2" means it did less damage, not that it helped. I hadn't decided in advance how to read a split like this, and the four stations share almost the same weather inputs, so I treat it cautiously.

More detail (risk–coverage curves, a relative-error version of the labels, why the watcher does less damage) is in [`docs/results_detail.md`](docs/results_detail.md).

## How it works

```
past pollution + past weather (all at or before time t)
        │
        ├──► forecasters F0–F3 ──► F3 forecast for t+6h
        ├──► watcher ──► trust / distrust F3 at this hour
        └──► routing:  trust → F3,  distrust → persistence
                     │
              forecast for t+6h ──► compared with always using F3
```

**Data.** Hourly PM2.5 and NO₂ from DEFRA's AURN network, 2018–2025, read from DEFRA's `.RData` files with the `rdata` package. Hourly weather from the Open-Meteo ERA5 archive at each station's coordinates. Stations were chosen by a script that kept London sites with at least 80% hourly coverage. MY1 (Marylebone Road, kerbside), KC1, BEX and HRL passed. CA1 failed at 63.8%.

**Forecaster features.** Lagged PM2.5 and NO₂ (t, t−1, t−3, t−6, t−12, t−24), rolling means and standard deviations, temperature, humidity, pressure, wind as east–west and north–south components, and calendar features. 46 in total.

**Watcher features.** 27, in four groups: how different recent inputs look from the training period (Wasserstein distance), recent volatility, disagreement between F3 and ridge, and F3's recent errors.

**Evaluation.** Walk-forward: train on everything before a quarter, test on that quarter, move forward, refit. That gives 20 test folds from 2020 Q1 to 2024 Q4. The watcher needs two folds of F3 errors to learn from, so routing is scored on folds 3–20. Intervals come from a block bootstrap that resamples whole weeks, because neighbouring hours are not independent (219 weeks at MY1, 2,000 resamples, paired). 2025 was kept aside until the end.

I used gradient boosting, not a neural network. With one target at hourly resolution over a few years, it's the sensible tool, and a bigger model wouldn't have answered the question any better.

## Decisions that mattered

**What counts as "unreliable".** The obvious choice is to label the 20% of hours with the biggest errors. But errors grow with pollution level. A miss of 15 µg/m³ when the truth is 55 is ordinary, and a miss of 3 when the truth is 15 may not be. Ranking raw errors would mostly flag smoggy hours, and the watcher would learn to spot pollution episodes, which is easier and not the question. So I split hours into ten bins by F3's *predicted* PM2.5 and labelled the worst 20% of errors within each bin. The bins come from predictions, not actual values, because actual values aren't known in advance. The bin edges and cut-offs are fitted on training data only, so the share labelled unreliable varies by fold (10–25%) instead of being forced to 20%.

**Leakage.** A leaky pipeline and an honest one both produce sensible-looking numbers with no error, so you can't catch leakage by looking at the output. I used a canary test instead: plant an extreme value at one time t*, rerun, and check that no prediction made before t* changes. It fires on a planted leak, stays quiet without one, and passed on the real pipeline at three positions. Anything that depends on the training window, like the recent-error features or the distribution distances, is built inside the walk-forward loop. Recent-error features only use hours whose true value was already published. [`docs/information_contract.md`](docs/information_contract.md) lists what each model can see at time t.

**Silent data bugs.** Two mistakes would have produced no error and wrong results. Open-Meteo returns wind in km/h by default, which would scale every wind feature by 3.6, so I request m/s and assert on the returned units. A timezone mix-up would shift every hour-of-day feature by an hour for half the year, so I checked the raw timestamps across both 2020 clock changes and confirmed AURN stores true UTC. Details are in [`docs/ingest_checks.md`](docs/ingest_checks.md).

## Live log

A backtest can be rerun until it looks good. So the frozen F3 now runs on a schedule and logs each forecast before DEFRA publishes the reading it predicts. The log is on a `live-log` branch that blocks force-pushes and deletion.

Since switching lost everywhere, the live map always shows F3. It adds a warning when recent forecasts have been poor, using rule R2 with fixed thresholds. I picked R2 with a rule I wrote down before the comparison. The full reasoning is in [`docs/results_detail.md`](docs/results_detail.md).

AURN publishes in a daily batch about 8–15 hours late, so the logger can only forecast about six hours a day per station. The log started on 30 September 2026. I won't make comparative claims from it until it has at least 26 weeks of scored forecasts, including a full January to March.

## Limitations

- **Four stations in one city.** They share almost the same weather, so they aren't independent tests. MY1 is the only kerbside site.
- **Holdout at one station.** The 2025 holdout was only run at MY1.
- **One fallback, one switching share.** This shows that switching 20% of hours to persistence doesn't help. It doesn't show that a watcher like this could never be useful.
- **Shortcuts.** The target column ended up in the feature table. It is excluded explicitly, with assertions in `src/forecast.py`, rather than rebuilt. Distribution distances update daily, not hourly.
- **Feature importance.** I haven't run an ablation, so I don't claim to know which watcher features matter.
- **Correlation only.** The results show what predicts F3's errors, not what causes them.
- **Not a new method.** Selective prediction and model routing are established ideas. What this project adds is a careful test on real data, with baselines, leakage checks and confidence intervals.

## What I'd do differently

The biggest change would be the watcher's question. It learns "will F3 be bad?", but switching only helps when "will persistence beat F3?". The results suggest those are different questions, and I'd train on the second one. I'd also try ridge as the fallback, since it beat persistence in every fold. After that I'd test whether a watcher trained at some stations transfers to a new one, and use stations outside London so the replications are actually independent.

## Reproduce

```
conda env create -f environment.yml
conda activate aq
```

`environment.yml` was built on Windows. The Linux live logger uses `requirements-live.txt`.

Main station (MY1), from the repo root:

```
python -m src.ingest --site MY1 --start 2018 --end 2025
python -m src.features --station MY1
python -m src.run_forecast
python -m src.watcher
python -m src.routing
python -m src.risk_coverage
python -m src.bootstrap
python -m src.plot_headline
```

For another station, run `ingest` and `features` with that site code, then add `--station KC1` (or `BEX`, `HRL`) to `run_forecast`, `watcher`, `routing` and `bootstrap`. The 2025 holdout is the same four scripts with `--holdout`. That run is designed to happen once. Data snapshots: MY1 pulled 2026-08-29, the other three 2026-09-04.

## Repo layout

```
PROJECT_SPEC.md     plan, hypotheses, and dated pre-registered changes
docs/               ingest checks, information contract, harness design,
                    station selection, results detail, live map (index.html)
src/                ingest → features → forecast → watcher → routing → bootstrap,
                    plus the live logger (live.py, freeze.py)
tests/              canary tests (pytest)
models/             frozen F3 per station
results/            CSVs and figures
```

Data files aren't committed. The commands above rebuild them.