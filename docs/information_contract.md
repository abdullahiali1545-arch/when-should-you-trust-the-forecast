# Information Contract

What each component of the system may read at forecast origin `t`, targeting `t+6h`.
Check a feature against this before writing it. Companion to `PROJECT_SPEC.md` Part 10.

Written 20 August 2026, W1.1. Audited 9 September 2026, W2.1 — see §7.

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
| **F2** linear | PM2.5/NO₂ lags, ERA5 weather (temperature, humidity, pressure, wind u/v — **not** `boundary_layer_height`, dropped 2026-09-09) | `t` | Mode B forbids weather after `t` |
| **F3** LightGBM | as F2, plus rolling stats and calendar | `t` | Same; calendar features for `t+6h` are exempt (§4) |
| **Watcher** | `ŷ_F3`, `ŷ_F2`, volatility, distribution distances | `t` | All computable at `t` |
| | residual-derived features | **`t−6h`** | A residual needs an outcome (§3) |
| **R0** random, coverage-matched | nothing but a random draw and the coverage rate | — | Reads no data at all; the coverage rate is set by the policy being matched |
| **R1** | `ŷ_F3(t+6h)` | — | A prediction, not a truth |
| **R2** | rolling MAE of scoreable residuals | **`t−6h`** | Same reason as the watcher's |
| **R3** full watcher | whatever the watcher may read, above | `t` / `t−6h` | Inherits the watcher's two clocks; adds nothing of its own |
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
mae_24h  = resid.abs().rolling(24).mean().shift(6)   # magnitude of error
bias_24h = resid.rolling(24).mean().shift(6)         # direction of error
```

- `.abs()` — **required for MAE.** `resid` is signed (`actual − predicted`), so
  averaging it without `.abs()` gives bias, not error magnitude: a model alternating
  +20 and −20 would score ≈ 0. Both quantities are legitimate watcher features and
  they mean different things — "the model is running high" versus "the model is
  wrong". Name them apart and compute them apart.
- `.rolling(24).mean()` — averages 24 consecutive forecasts
- `.shift(6)` — moves that value six hours later, so the number sitting at 09:00
  averages forecasts made at 03:00 and earlier

Drop the `.shift(6)` and the watcher sees errors from forecasts still in flight.
Nothing errors. The result looks excellent and means nothing.

Note the asymmetry between the two mistakes. A missing `.abs()` makes the feature
*wrong* — the watcher gets worse and you notice. A missing `.shift(6)` makes it
*leaked* — the watcher gets better and you do not.

**Assumption to state in the README:** this treats the reading at `s+6h` as available
at `s+6h`. In live operation it would not be. The W1.4 freshness check measured a 15 h
AURN publication lag; a live deployment carries that much less residual information
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
| `pm2_5_lag_6` | No | An observation at a fixed timestamp. Once known, never changes |
| `wind_u` | No | Same — measured at `t` |
| `target_hour` | No | Follows from the calendar and the 6h horizon by arithmetic |
| `pm2_5_std_6h` | No | Arithmetic on six observed readings from this row's own past |
| `mae_last_24h` | **Yes** | Needs F3's out-of-fold predictions; a retrained F3 makes different errors |
| F1 climatology | **Yes** | The (hour, month) means are averaged over the training window |
| `abs(ŷ_F3 − ŷ_F2)` | **Yes** | Both models are refitted each fold |
| `wasserstein_wind_vs_training` | **Yes** | The reference distribution *is* the training window |
| Scaler / fitted imputation parameters | **Yes** | Fitted on the training window |
| Ingest-time linear interpolation (gaps < 2 h) | No | Deterministic arithmetic, no fitted parameter, applied once in `src/ingest.py` and already in the Parquet. **But it is a declared one-hour look-ahead:** a single missing hour is filled from the readings on *either* side, so the fill at 14:00 uses the 15:00 value, which then appears in `pm2_5_lag_0` at 14:00. Spec Part 6 mandates the rule; magnitude is 35 single-hour PM2.5 fills at MY1 out of ~70k rows. Accepted and disclosed, not engineered around — the `imputed` flag keeps affected rows auditable. State it in the README |
| Label decile edges, 80th pct cut | **Yes** | Fitted parameters, frozen and applied to the test fold |

**The pattern:** being *computed* does not make a feature fold-dependent.
`pm2_5_std_6h` is computed and safe. What matters is whether the computation touches a
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

**Three of the watcher's four feature families are fold-dependent** — residuals,
disagreement, distribution distance. Only volatility is not. `src/features.py`
therefore builds much less of the watcher than Part 9's feature list suggests, and the
rest is deferred to the harness by design, not by omission.

### The second lag: one fold of burn-in

§3's six-hour blind spot operates *within* a fold. A second, larger lag operates
*across* folds.

The watcher may train only on **out-of-fold** F3 residuals. In-sample residuals
are systematically too small and differently shaped, so a watcher trained on them
learns a distribution that never occurs at test time. F3's fold-1 residuals are
in-sample, so they are unusable.

Consequence: the watcher's first scoreable fold is fold 2 — one quarter of
burn-in behind F3. Spec Part 10. Budget for it when counting evaluation folds.

### How this document is enforced

Nothing above is self-enforcing. Correct output and leaked output are visually
identical — both produce a fold table with plausible MAEs, no error and no warning —
so the harness cannot be validated by inspecting its results.

The enforcement mechanism is the **canary test** (Spec Part 10): corrupt one input
series at a single timestamp `t*`, rebuild, and assert that every row with origin
earlier than `t*` is bit-identical to before. If an earlier row changed, information
moved backwards in time.

Two canaries exist, at different scopes:

| Canary | Scope | Status |
|---|---|---|
| `features.py --canary` | Fold-independent columns only | Passing at all four stations, 2026-09-09 |
| `evaluate.py` canary | The walk-forward harness: scalers, climatology, residuals, disagreement, distribution distances, label thresholds | **W2.3 — not yet written** |

The first cannot detect fold-level leakage. It is the second that protects the
headline result.

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

---

## 7. Declared look-aheads

Every look-ahead this project knowingly accepts, in one place. Nothing here is a bug;
each is a documented trade-off. The README must disclose all three.

| # | Look-ahead | Magnitude | Why accepted | Where |
|---|---|---|---|---|
| 1 | ERA5 reanalysis used as weather-at-`t` | Mild, unquantified | Archived forecast weather does not exist for 2018–2020 | §6, Spec Part 3 change 2 |
| 2 | Single-hour gaps interpolated from both neighbours | 1 hour, 35 PM2.5 rows at MY1 | Spec Part 6 imputation rule; alternative is discarding the rows | §5 table |
| 3 | Residual features assume truth at `s+6h` arrives at `s+6h` | 15 h in live operation | Backtest convention; live deployment carries less residual information | §3, W1.4 |

**The claim this supports:** "I found these, quantified them, and disclosed them"
is a stronger position than "there was no leakage." An examiner who finds an
undisclosed look-ahead has found a flaw. One who finds it already in the contract has
found rigour.

---

## Audit log

**2026-09-09 (W2.1)** — first audit against Spec Part 10. Changes:

1. §5 structural repair. `### Consequence for W1.7` had no body; its content had been
   spliced onto the end of the burn-in section. Both sections restored with their own
   text.
2. §3 `mae_24h` corrected. The example averaged signed residuals, computing bias while
   naming it MAE. `.abs()` added; `bias_24h` split out as a separate named feature.
   The `.shift(6)` was already correct.
3. §5 interpolation row expanded to declare the one-hour look-ahead. The
   fold-dependence answer was correct; the disclosure was missing.
4. §2 F2 row records `boundary_layer_height` as dropped (2026-09-09 changelog entry).
5. Column names corrected to match `features.py`: `pm25_*` → `pm2_5_*`.
6. §2 gained R0 and R3, previously absent.
7. §5 gained "How this document is enforced" — the canary test, and the scope
   difference between the W1.7 and W2.3 canaries.
8. §7 added: all declared look-aheads in one table.

No model results existed at the time of this audit.