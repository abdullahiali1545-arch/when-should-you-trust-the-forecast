# Domain notes — April 2020

**Why a model trained on 2018–19 London air was confused in April 2020, and why it failed in opposite directions for NO₂ and PM2.5.**

W1.2 — 20 August 2026

Written before looking at any MY1 data.

---

## The setup

A model trained on London air quality in 2018 and 2019 learned a relatively stable world. Traffic followed a reliable weekly and daily rhythm and  pollution levels tracked it. Two years of consistent behaviour allowed the model to learn associations that were useful under those conditions.

Then April 2020 arrived, and two things happened at once.

---

## NO₂ fell, and the model over-predicted

Nitrogen dioxide at a kerbside site is strongly influenced by traffic exhaust. Under lockdown, traffic collapsed — most cars stayed off the road and many people stayed indoors.

So NO₂ fell sharply.

The model, reasoning from two years of normal Tuesdays, expected levels closer to those seen under normal traffic conditions. It therefore over-predicted: it forecast more NO₂ than arrived.

This half is intuitive. Fewer cars, less exhaust; the model expected more than it got.

---

## PM2.5 rose, and the model under-predicted

This was the half that made April 2020 worth writing about.

Studies of London during this period reported elevated PM2.5 despite the traffic collapse.

The main explanation was meteorological rather than purely local. Spring 2020 included persistent easterly and south-easterly airflow over southern England, carrying aerosol from continental Europe towards London. Secondary formation also contributed — spring agricultural activity provides ammonia, which can react with atmospheric gases to form particulate matter.

So while London was emitting less from some local sources, particles could still arrive from elsewhere. A model relying strongly on the historical relationship between traffic-related conditions and PM2.5 could therefore under-predict PM2.5.

> Over-predicted NO₂. Under-predicted PM2.5. Same model, same month, wrong in opposite directions.

Whether MY1's 2020 record actually shows this pattern is tested later in the project's COVID validation. If it does not, that disagreement is a finding rather than something to be explained away.

### What this is not

At first I misunderstood and was tempted to explain the PM2.5 rise primarily as domestic emissions — people staying at home, burning wood, or heating their homes more. That is not a sufficient explanation for the April pattern.

Wrong seasonal emphasis. Heating demand generally falls as winter ends and temperatures rise. Spring 2020 in the UK was unusually warm and sunny, so a simple increase in domestic heating is not a convincing primary mechanism.

Wrong scale as a primary explanation. Domestic burning is a genuine PM2.5 source and can be important locally, but it is not enough by itself to explain the broad spring 2020 PM2.5 behaviour observed across London.

The distinction matters because the two explanations describe different kinds of failure:

| Explanation | What it says |

|---|---|

| Domestic heating rose | **London emitted more.** A local emissions story. |

| Continental transport | **London emitted less from some local sources, while PM2.5 could still rise because pollution was transported into the region.** |

The second mechanism is particularly useful for understanding the project because it shows how a model can receive plausible local information while the relationship between its inputs and the target changes.

---

## Why this is distribution shift and not simply a missing feature

The model's observed inputs could all be accurate. Traffic really had collapsed. NO₂ really was low. Its reasoning from those inputs could therefore be internally consistent while still producing the wrong PM2.5 forecast.

What changed was the relationship between the inputs and the target.

In 2018–19, lower traffic was associated with lower PM2.5 under the prevailing range of conditions. The model could learn that association correctly. In April 2020, low traffic still meant lower local emissions — but PM2.5 could remain high because transported pollution and atmospheric processes had become much more important.

The same input relationship was therefore less reliable under the new regime.

This is why the answer is not automatically a better feature. A missing input can be an engineering problem: add the relevant information, retrain, and potentially improve the forecast. A changed relationship is harder because the next shift may be caused by something different from the previous one.

> The model's inputs can be correct and its reasoning can be internally consistent, while the relationship it learned no longer holds. That is why the project asks whether we can recognise when the forecast should not be trusted.

---

## The model does not "break"

The word "break" is misleading here, and the correction is central to the project.

F3 does not stop working in April 2020. It carries on exactly as designed: it takes its inputs, applies the relationship it learned, and returns a number. No software error is required. No warning is necessarily produced. There may be no obvious internal sign of distress.

For example, F3 might predict 12 µg/m³ with no indication that the real value will be much higher.

Nothing inside the model necessarily has failed. The problem is the fit between the model and a world that has changed — and from inside the forecasting process, a shifted regime can look like a normal one.

> F3 learned a relationship that was useful under the conditions it was trained on. The world changed. F3 can keep applying that relationship confidently even when it has become unreliable.

That is why a second model is required. If F3 could reliably recognise its own periods of unreliability, a separate watcher would be unnecessary. The watcher instead looks for signals available at prediction time that indicate when F3's learned relationship may be less trustworthy.

---

## Why this matters for the project

If model failures were always caused by missing inputs, the solution would simply be better feature engineering.

Some failures cannot be completely anticipated by listing every possible future problem. The next shift may be caused by a different combination of emissions, weather, transport, behaviour, or other factors.

That is the argument for a second model. Rather than trying to enumerate every way F3 could fail, the watcher asks a different question:

> Given what we can observe at prediction time, does this look like a situation in which F3 is likely to be unreliable?

---

## Three components, three jobs

Keeping these separate matters because it is easy to say "the model" and mean two different things.

| Role | Job |

|---|---|

| **F3** | Learns a forecasting relationship from historical data and uses it to predict PM2.5 six hours ahead. |

| **Distribution shift** | A change in the data-generating environment or in the relationship between inputs and the target that can make F3's learned relationship less reliable. |

| **The watcher** | Uses information available at prediction time to estimate whether F3 is likely to be unreliable. |

Distribution shift is not something the watcher "suffers from" in the terminology of this project. Detecting conditions associated with unreliable forecasts is the watcher's job.

A note on precision: F3 never sees the PM2.5 value it is predicting. It uses lagged PM2.5 — for example, values at `t`, `t−1`, `t−3` and other permitted lags — and forecasts `t+6h`. Saying that F3 "takes PM2.5 into account" refers to its historical/lagged inputs and must not be interpreted as access to its future target.

April 2020 also shows why error direction deserves attention alongside error magnitude. The same broad shock can produce over-prediction for one pollutant and under-prediction for another. A system tracking only `|error|` sees that both periods were bad but loses information about the direction of the failure. This is why the watcher includes a signed-bias feature such as `bias_last_24h`, rather than relying only on error magnitude.

---

## Caveats

Continental transport was a major contributor to elevated spring 2020 London PM2.5, not necessarily the sole cause. Secondary formation from agricultural ammonia also contributed, and the relative contribution of different sources is not being established by this project.

This page was written from domain reading before examining MY1 data, so it records a prior rather than a result.

Whether MY1's 2020 record actually shows the expected pattern is tested in the project's COVID validation. If the data disagrees with this prior, that disagreement is a finding and will be reported rather than silently rewritten.

---

## W1.2 completion check

- ☑ Written before examining MY1 data.

- ☑ Explains the April 2020 domain mechanism.

- ☑ Explains why NO₂ and PM2.5 can move in opposite directions.

- ☑ Explains distribution shift without treating it as a software failure.

- ☑ Separates F3, distribution shift and the watcher.

- ☑ Distinguishes lagged PM2.5 inputs from the future target.

- ☑ Notes why signed error direction matters.

- ☑ Records uncertainty rather than presenting the COVID pattern as a MY1 result.