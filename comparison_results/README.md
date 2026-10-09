# AdaCent rerun vs. old AdaCent — comparison

Comparison of the mean trajectories from our AdaCent reimplementation
(`AdaCent_(k=25)`, the rerun) against the old AdaCent runs (`KCESBatch_(k=25)`,
the earlier name for AdaCent), for the three main YAHPO suites: `lcbench`,
`rbv2_rpart`, and `rbv2_aknn`.

## Measures

- **Relative MAE (%)** — the average absolute pointwise difference between the
  two mean curves, as a percentage of the average level of the old AdaCent curve:

      mean(|rerun − old|) / mean(old) × 100

  Global form (ratio of means), not the pointwise MAPE (`mean |Δ|/old`),
  which is unstable where the old AdaCent curve is near zero early in the budget.

- **final Δ%** — the absolute difference of the last budget point as a percentage
  of the old AdaCent final value:

      |rerun_final − old_final| / old_final × 100

All curves are on the native 0–1 scale (lcbench `val_accuracy` divided by 100;
rbv2 `acc` is already a fraction). Each curve spans budget points 1 … 1040
(T = 52, batch 25).

## Results

| Suite | tasks | Relative MAE mean | Relative MAE median | final Δ% mean | final Δ% median |
| --- | ---: | ---: | ---: | ---: | ---: |
| lcbench | 34 | 0.502% | 0.428% | 0.116% | 0.088% |
| rbv2_rpart | 116 | 0.794% | 0.428% | 0.270% | 0.091% |
| rbv2_aknn | 118 | 0.454% | 0.226% | 0.110% | 0.031% |

## Files

Per suite:

- `per_task/*.png` — overlay of rerun vs. old AdaCent per task.
