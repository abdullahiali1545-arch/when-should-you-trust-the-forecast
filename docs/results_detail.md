# Results detail

Material moved out of the README so the README stays short. Every number here is unchanged from the previous README.

## Exploratory analyses (not pre-registered)

**Why the watcher does less damage than R1 and R2.** At KC1, BEX and HRL, persistence beat F3 in about the same share of switched hours for every rule (40–42%). The difference was the size of the loss in the other switched hours. It was smaller for the watcher than for R1 and R2, and smallest for random switching. So the watcher's edge seems to come from avoiding the very worst hours, not from finding hours where persistence wins.

**Switching share (MY1).** I repeated the comparison for every share from 5% to 95%. No share beat always using F3, for any rule.

**Risk–coverage (MY1).** Area under the risk–coverage curve (AURC, lower is better): R1 2.703, R2 2.914, R3 3.270, R0 3.480. This measure favours R1 for the same reason the main labels are stratified. Dropping high-pollution hours lowers raw error whether or not those hours were unreliable, so R1's score here says little about spotting unreliability.

**Relative-error labels (MY1).** I repeated the analysis with "unreliable" defined as a large error relative to the prediction, |y − ŷ| / max(ŷ, floor), with the floor at the 10th percentile of predicted PM2.5. I fixed this definition after I had seen the main results, so it is a robustness check only. The labels agreed with the main labels on 90.6% of hours. Watcher PR-AUC was 0.198 against 0.183. R3's routed MAE was 3.718, still worse than always F3 and random. Under these labels R3 did beat R2 (−0.085 [−0.158, −0.023]), but I don't treat that as a finding, because the main comparison shows no difference.

## Live log: choosing the warning rule

I compared F3's error in flagged hours with its error in unflagged hours, for each rule, with week-block bootstrap intervals. On raw error the simple rules looked best: in flagged hours F3's error was 2.01–2.71× its unflagged error under R1, 1.70–2.16× under R2 and 1.32–1.53× under R3. Raw error rewards any rule that picks high-pollution hours, though. So before running it, I fixed the choice to depend on a level-adjusted comparison, where each hour's error is divided by the average error for its predicted-concentration decile:

| Station | R1 | R2 | R3 | R3 vs R2 |
|---|---|---|---|---|
| MY1 | 1.00 | 1.11 | 1.07 | no detectable difference |
| KC1 | 1.00 | 1.15 | 1.09 | no detectable difference |
| BEX | 1.00 | 1.15 | 1.11 | no detectable difference |
| HRL | 1.00 | 1.17 | 1.12 | no detectable difference |

R1 carries nothing beyond the pollution level itself. The watcher beat R1 and random but wasn't detectably different from R2, so under the pre-registered rule the map uses R2. It flags a site when F3's average error over the previous day is at or above the 80th percentile from the walk-forward folds: MY1 4.57, KC1 3.64, BEX 4.28, HRL 3.43 µg/m³. With these fixed thresholds the warning still held (level-adjusted 1.11–1.17) and flagged about 17% of hours, far more in January to March than in autumn. The threshold was fitted on walk-forward errors while the live model was trained on more data, so the live flag rate may drift. That drift is reported, not corrected.

## Live log: integrity and timing

There is one frozen F3 per station, trained once on data to the end of 2025. Before freezing, I checked the training recipe by refitting fold 20 and reproducing the committed out-of-fold forecasts exactly. Live forecasts match on Windows and Linux: 24 of 24 agreed to every decimal place in the first comparison.

The logger only forecasts from origins whose target reading hasn't been published yet, which are the last hours of each daily batch (evening origins). At those origins the model knows exactly what the backtest knew. So the publication delay limits which hours can be forecast, not how much the warning knows. I originally expected the delay to leave the live warning with less recent-error information than the backtest, and that was wrong.

GitHub's scheduler is best-effort, so runs are sometimes hours apart. No forecasts are lost, because each batch's origins stay unpublished until the next batch. Each commit records the code version that produced it.

## Data-quality notes

- Gaps under two hours are linearly interpolated and flagged. Longer gaps are left missing, so rolling statistics never run over invented values.
- MY1 had an instrument outage in late 2020 (November entirely missing), which leaves one test fold with only 386 usable hours.
- Boundary-layer height was dropped because the Open-Meteo archive has no values for January to June 2024 at any station.
- Calendar features use London local time, because traffic follows the local clock. The NO₂ rush-hour pattern was clearly sharper this way (correlation 0.925 vs 0.825).