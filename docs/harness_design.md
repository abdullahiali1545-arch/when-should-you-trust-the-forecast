# Harness design (W2.2)

Written before `src/evaluate.py` exists. Part one — time geometry and canary
placement. Part two (refit list, common scoring set, F3 hyperparameters, output
schema) is appended in the next session, before any model is fitted.

Horizon `h = 6` hours throughout. Station is a parameter; this geometry is
identical at MY1, KC1, BEX and HRL.

---

## 1. Every row has two timestamps

A training or scoring row is

    r_t = ( x_t , y_{t+h} )

- **Origin `t`** — the hour the forecast is made. `x_t` is built only from
  information at or before `t`.
- **Target time `t + h`** — the hour the answer exists. The value `y_τ` counts
  as known from `τ` onward.

Example: origin 14:00 uses PM2.5 at 14:00, 13:00, 11:00, …; its target is PM2.5
at 20:00, which does not exist until 20:00.

### Why the target is built in the harness, not in `src/features.py`

Every column in the feature table obeys one invariant: its value at origin `t`
depends only on information available at or before `t`. The W1 canary is a test
of exactly that invariant — poison the input series at `t*`, rebuild, and assert
that every row with origin earlier than `t*` is unchanged.

A target column would break it immediately. The row with origin `t* − 6` carries
`y_{t*}` as its target, so poisoning `t*` changes a row whose origin is earlier
than `t*`, and the canary fails. That failure would be correct behaviour, not a
bug: the target *is* information from after the origin. It is legitimate to use
as a label and illegitimate to use as a feature, and the two only stay separate
if they live in different places.

The alternative — keeping the target in `features.py` and exempting it from the
canary — means the canary needs an exemption list. Exemption lists are where
leaks hide, because every future column that looks a bit like the target has a
plausible argument for joining it. Keeping the invariant absolute means the
canary is a real test rather than a test of whatever wasn't excused from it.

---

## 2. Fold layout

Let `a_k` be the first hour of test quarter `k`, all times UTC.

    a_1  = 2020-01-01 00:00
    a_21 = H = 2025-01-01 00:00        (start of the sealed holdout)

    K = 20 folds: 2020Q1 … 2024Q4

**Test set for fold k:**

    S_k = { t : a_k <= t < a_{k+1}  AND  t + h < H }

The second condition is the holdout guard. Without it the last origins of 2024
would be scored against 2025 truth:

| Origin | Target time | Scored? |
|---|---|---|
| 2024-12-31 17:00 | 2024-12-31 23:00 | yes |
| 2024-12-31 18:00 | 2025-01-01 00:00 | **no** |
| 2024-12-31 23:00 | 2025-01-01 05:00 | **no** |

**Numbers.** 2020–2024 spans 1,827 days = 43,848 candidate origins. The holdout
guard removes the final 6, leaving at most 43,842 scored origins per station
before rows with a missing target are dropped. 2024Q4 has 2,208 candidate
origins and 2,202 scored.

The `S_k` do not overlap and together cover 2020–2024, so every origin is
predicted exactly once by a model that never trained on it. That is what makes
the output table out-of-fold.

### Why `origin < 2025` is the wrong guard

It filters the wrong clock. Six origins — 2024-12-31 18:00 through 23:00 — pass
an origin-based test while their targets fall on 2025-01-01 between 00:00 and
05:00. Scoring them reads six hours of holdout truth.

The cost is not accuracy. Six rows in 43,848 will not move an MAE to three
decimal places, and that is precisely why the mistake would never be noticed.
The cost is the claim. The holdout was sealed on 2026-08-29 with a
pre-registered undertaking that it would be touched once, at the end, with the
date recorded. An origin-based guard silently breaks that undertaking during
development, and there is no way to un-see the values afterwards. From that
point the project's central methodological claim — that the final number was
produced without the holdout influencing any earlier choice — is no longer
something that can be demonstrated, only asserted.

A reviewer cannot check what you looked at. They can only check whether the rule
you wrote makes looking impossible. The guard has to be stated on target time
because that is the clock on which the holdout is defined.

---

## 3. Window type — expanding

**Decision: expanding.** Training set grows each fold; nothing is dropped from
the front.

    Expanding:  T_k = { t : t_0 <= t , t + h <= a_k }      t_0 = 2018-01-01 00:00
    Sliding:    T_k = { t : a_k − W <= t , t + h <= a_k }

| | Expanding | Sliding (W = 2 years) |
|---|---|---|
| Fold 1 training origins | 17,515 | 17,515 |
| Fold 20 training origins | 59,155 | 17,539 |
| Free parameters | none | W |
| Adapts to a changed world | slowly | faster |

**Reasons.** No free parameter, so nothing to tune post hoc. It matches Part 10's
"train on 2018–2019" starting point. It is what an operational service would do
by default, absent evidence that old data hurts.

**Not** because more distribution shift gives the watcher more to detect. That
would be choosing a design to flatter H2, and it is the reason this decision is
pre-registered rather than settled later.

Note: Part 10 says "roll the window forward", which does not settle expanding vs
sliding. This document settles it.

---

## 4. The purge

A training row is admissible only if its answer had arrived by the refit moment
`a_k`:

    t + h <= a_k        equivalently    t <= a_k − 6h

Removing training rows whose target falls after the train/test boundary is called
**purging**. Convention chosen: `<=`, so a target landing exactly at `a_k` is
allowed. This removes `h − 1 = 5` rows per fold.

**Worked example, fold 2021Q2, `a_k` = 2021-04-01 00:00:**

| Origin | Target time | In T_k? |
|---|---|---|
| 03-31 17:00 | 03-31 23:00 | yes |
| 03-31 18:00 | 04-01 00:00 | yes — arrives exactly at refit |
| 03-31 19:00 | 04-01 01:00 | **purged** |
| 03-31 23:00 | 04-01 05:00 | **purged** |
| 04-01 00:00 | 04-01 06:00 | test row |

Without the purge the model trains on PM2.5 values from 01:00–05:00 on 1 April —
hours inside the quarter it is about to be scored on.

**Why 5 rows out of ~17,000 matters.** Not for the effect size. Because the
canary asserts bit-identical predictions, and a small leak fails that assert as
loudly as a large one — so "no leakage" becomes provable rather than
"probably negligible".

---

## 5. The same rule in W3

The watcher's recent-residual features use `e_τ = y_{τ+h} − ŷ_τ`. A residual is
usable at origin `s` only if

    τ + h <= s

At `s` = 14:00 the newest legal residual comes from origin 08:00. Using origin
13:00 would require PM2.5 at 19:00 — five hours in the future. Because pollution
is autocorrelated, that residual is close to the answer itself, so the leak would
inflate the watcher on every row, not just at fold boundaries.

One inequality — `origin + h <= now` — enforced in two places.

---

## 6. Canary design (specification for W2.3)

A canary tests one named pathway, not "leakage" in general. Aim it wrong and it
returns a green tick that means nothing. Three placements, three hypotheses:

| `t*` | Tests | Expected on a correct harness |
|---|---|---|
| `a_k + 5h` (just inside a test quarter) | the purge | no scored prediction with origin < `t*` moves |
| `H + 3h` (just inside the holdout) | the holdout guard | no scored 2024 origin moves |
| mid-quarter | features reaching forward | no prediction with origin < `t*` moves |

**Why the seam.** Poisoning `y_{t*}` reaches an earlier-origin prediction only
through the training set, via the row with origin `t* − 6` whose target is
`y_{t*}`. That row is illegal in `T_k` exactly when `a_k < t* < a_k + 6`. A
mid-quarter `t*` has nothing illegal for the poison to travel through: its
carrier row sits deep inside the test quarter, is not a candidate for `T_k`
either way, and enters later folds legitimately — where every changed prediction
has a *later* origin than `t*`.

**Rules for the assert:**

1. **Assert on predicted values only.** Poisoning `y_{t*}` legitimately changes
   `y_true` and the residual for origin `t* − 6`. Asserting on those fails for
   the wrong reason.
2. **Use an extreme sentinel (e.g. `1e6`).** Bit-identity is a sufficient test,
   not a complete one — a leak with no numerical effect is invisible, so the
   poison must be loud enough to move a tree split.
3. **Run it on F1 and F2 as well as F3.** F1's climatology means and F2's OLS
   coefficients are analytically guaranteed to move when a training target
   changes. F3 is the model under scrutiny; F1 and F2 are the sensitive
   instruments.

---

## 7. Open — settled in part two

- What is refitted inside each fold (F1 means, F2 scaler and coefficients, F3).
- The common scoring set: all four models scored on identical rows, since F2
  cannot accept NaN and LightGBM can.
- Fixed F3 hyperparameters, pre-registered rather than tuned.
- LightGBM determinism settings, so refits are bit-reproducible for the canary.
- Output schema: `station | origin | target_time | fold | y_true | yhat_F0..F3`.