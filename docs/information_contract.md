# Information Contract

What each component of the system may read at forecast origin `t`, targeting `t+6h`.
Check a feature against this before writing it. Companion to `PROJECT_SPEC.md` Part 10.

Written 20 August 2026, W1.1.

---

## 1. The rule

> Information used at time *t* ⊆ information available by time *t*.

In words: **nothing that could not have been known at `t`.**

Note this is not the same as "nothing after `t`". Some things about the future are
knowable in advance. The test that separates them:

> **Could I have written this number down at `t` without waiting?**

| At 09:00 | Write it down now? | |
|---|---|---|
| PM2.5 at 09:00 | Yes — it has been measured | ✅ |
| PM2.5 at 15:00 | No — wait six hours | ❌ |
| Wind speed at 15:00 | No — wait six hours | ❌ |
| "15:00 is a Tuesday afternoon in February" | Yes — calendar arithmetic | ✅ |

The rule includes `t` itself, not just the hours before it. The 09:00 reading has
arrived by 09:00 and may be used — F0 persistence is nothing but that value.

---

## 2. Consumers

| Consumer | May read | Latest timestamp | Why that timestamp |
|---|---|---|---|
| **F0** persistence | PM2.5 | `t` | The reading at `t` has arrived |
| **F1** climatology | (hour, month) means | end of **training window** | Averages over the test period would leak the future into the baseline |
| **F2** linear | PM2.5/NO₂ lags, ERA5 weather | `t` | Mode B forbids weather after `t` |
| **F3** LightGBM | as F2, plus rolling stats and calendar | `t` | Same; calendar features for `t+6h` are exempt (§4) |
| **Watcher** | `ŷ_F3`, `ŷ_F2`, volatility, distribution distances | `t` | All computable at `t` |
| | residual-derived features | **`t−6h`** | A residual needs an outcome (§3) |
| **R1** | `ŷ_F3(t+6h)` | — | A prediction, not a truth |
| **R2** | rolling MAE of scoreable residuals | **`t−6h`** | Same reason as the watcher's |
| **Label** | `y(t+6h)`, `ŷ_F3(t+6h)`; thresholds from train folds | `t+6h`, after the fact | Labels are built retrospectively and are never an input |

---

## 3. The two clocks

Two different cut-offs sit in the same feature row.

**Observed features stop at `t`.** PM2.5, NO₂, wind, temperature. These were
*measured*. Nothing to wait for.

**Residual-derived features stop at `t−6h`.** A residual is not observed, it is
computed:

```
residual = actual PM2.5 − predicted PM2.5
```

It needs the actual value, and the actual value arrives six hours after the forecast
was made. So a residual is only born six hours after its forecast.

At 09:00, walking backwards through F3's recent forecasts:

| Made at | Predicted | Truth arrives | Scoreable at 09:00? |
|---|---|---|---|
| 08:00 | 14:00 | 14:00 | ❌ still in flight |
| 07:00 | 13:00 | 13:00 | ❌ |
| 06:00 | 12:00 | 12:00 | ❌ |
| 05:00 | 11:00 | 11:00 | ❌ |
| 04:00 | 10:00 | 10:00 | ❌ |
| **03:00** | **09:00** | **09:00** | **✅ freshest available** |
| 02:00 | 08:00 | 08:00 | ✅ |

So "F3's last 24 forecasts" at 09:00 means those made **03:00 today back to 03:00
yesterday** — not 09:00 back to 09:00.

**You always carry a six-hour blind spot on your most recent errors.** This is not a
limitation to engineer around; it is the truth of the situation. A real forecaster
at 09:00 was equally blind.

In code, with `resid` indexed by the hour the forecast was *made*:

```python
mae_24h = resid.rolling(24).mean().shift(6)
```

- `.rolling(24).mean()` — averages 24 consecutive forecasts
- `.shift(6)` — moves that value six hours later, so the number sitting at 09:00
  averages forecasts made at 03:00 and earlier

Drop the `.shift(6)` and the watcher sees errors from forecasts still in flight.
Nothing errors. The result looks excellent and means nothing.

**Assumption to state in the README:** this treats the reading at `s+6h` as available
at `s+6h`. In live operation it would not be. The W1.4 freshness check measures the
real publication lag; a live deployment carries that much less residual information
than the backtest.

---

## 4. Legal, but looks suspicious

**Calendar features for the target hour** — `target_hour=15`, `target_dow=Tuesday`,
`target_month=February`, `is_weekend=False`.

Legal because calendars are deterministic. Standing at 09:00 you can compute
`09:00 + 6h = 15:00` and check today's date. No waiting.

These are the pointer, not the pattern: the model learns from training data how
mid-afternoon Februarys behave, but it has to be *told* which hour it is aiming at
before it can apply what it learned. Rule them out and F3 cannot tell a February
afternoon from 3am in July.

**F1's climatology prediction for `t+6h`** — legal, but for a different reason: it is
an average over past Februarys within the training window. Only the calendar and past
data are involved.

**`ŷ_F3(t+6h)` as a watcher input, and as R1's trigger** — legal. A prediction is not
a truth. It was produced at `t` from inputs at `t`.

**Model disagreement, `|ŷ_F3 − ŷ_F2|`** — legal, and needs no ground truth ever. Both
models run at `t` on inputs from `t` and earlier, so their gap is computable
immediately. This is the only watcher feature with no lag at all.

---

## 5. The fold test

Walk-forward means F3 is retrained repeatedly, each time on everything before the
fold under test:

| Fold | F3 trained on | Tested on |
|---|---|---|
| 1 | 2018–2019 | Q1 2020 |
| 2 | 2018–2019 + Q1 2020 | Q2 2020 |
| 3 | 2018–2019 + Q1–Q2 2020 | Q3 2020 |

A **fold boundary** is where a training window stops. It moves every fold. Which
gives the test:

> **Would this column's value at a fixed timestamp change if I moved a fold boundary?**
>
> **No** → precompute in `src/features.py`
> **Yes** → compute inside the walk-forward harness, per fold

### Which side each feature falls on

| Feature | Fold-dependent? | Why |
|---|---|---|
| `pm25_lag_6` | No | An observation at a fixed timestamp. Once known, never changes |
| `wind_u` | No | Same — measured at `t` |
| `target_hour` | No | Follows from the calendar and the 6h horizon by arithmetic |
| `pm25_std_6h` | No | Arithmetic on six observed readings from this row's own past |
| `mae_last_24h` | **Yes** | Needs F3's out-of-fold predictions; a retrained F3 makes different errors |
| F1 climatology | **Yes** | The (hour, month) means are averaged over the training window |
| `abs(ŷ_F3 − ŷ_F2)` | **Yes** | Both models are refitted each fold |
| `wasserstein_wind_vs_training` | **Yes** | The reference distribution *is* the training window |
| Scaler / fitted imputation parameters | **Yes** | Fitted on the training window |
| Ingest-time linear interpolation (gaps < 2 h) | No | Deterministic arithmetic on two observed neighbours; no fitted parameter. Applied once in `src/ingest.py`, already in the Parquet |
| Label decile edges, 80th pct cut | **Yes** | Fitted parameters, frozen and applied to the test fold |

**The pattern:** being *computed* does not make a feature fold-dependent.
`pm25_std_6h` is computed and safe. What matters is whether the computation touches a
**fitted model** or a **training-window statistic**.

### The two-halves case

`wasserstein_wind_vs_training` obeys opposite rules on its two inputs:

| Half | Rule |
|---|---|
| Recent 72h window ending at `t` | May reach back freely. All observed by `t` |
| Reference distribution | Bounded by this fold's training window. May not reach forward |

```python
# LEAKS — one global reference contaminates every value in the column
ref  = wind_all_years
feat = wasserstein(wind.loc[t-71h:t], ref)

# CORRECT — reference is this fold's training window, refit every fold
ref_k = wind.loc[fold_k_train_start : fold_k_train_end]
feat  = wasserstein(wind.loc[t-71h:t], ref_k)
```

### Consequence for W1.7
### The second lag: one fold of burn-in

§3's six-hour blind spot operates *within* a fold. A second, larger lag operates
*across* folds.

The watcher may train only on **out-of-fold** F3 residuals. In-sample residuals
are systematically too small and differently shaped, so a watcher trained on them
learns a distribution that never occurs at test time. F3's fold-1 residuals are
in-sample, so they are unusable.

Consequence: the watcher's first scoreable fold is fold 2 — one quarter of
burn-in behind F3. Spec Part 10. Budget for it when counting evaluation folds.
**Three of the watcher's four feature families are fold-dependent** — residuals,
disagreement, distribution distance. Only volatility is not. `src/features.py`
therefore builds much less of the watcher than Part 9's feature list suggests, and the
rest is deferred to the harness by design, not by omission.

---

## 6. The ERA5 caveat

ERA5 is **reanalysis**: past weather reconstructed afterwards using observations
nobody had at the time. It is not a forecast and not a raw station reading.

So "weather at `t` is known at `t`" is true for a station observation and mildly false
for reanalysis. The ERA5 value indexed to 09:00 was produced years later, informed by
data that arrived after 09:00.

Spec Part 3, change 2 accepted this: reanalysis at prediction time is a mild
look-ahead, tolerated because archived *forecast* weather does not exist for the
required period.

**What this means in practice:** the acceptance stands — do not try to engineer around
it. But do not write the unqualified sentence in the README. State that Mode B uses
reanalysis observed up to `t`, and that this is a known mild optimism, separate from
the larger optimism Mode A measures.