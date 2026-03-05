# Appendix A — Phase 0 Neuromorphic Baselines

Neuromorphic efficiency metrics per model per scenario from Phase 0 pipeline evaluation.

## SyOps/step

| Model | step_high_speed_2500rpm_2A | step_low_speed_500rpm_2A | step_mid_speed_1500rpm_2A |
| --- | --- | --- | --- |
| best_incremental_snn | 9.018e+04 | 4.416e+04 | 6.631e+04 |
| intermediate_scheduled_sampling | 8.509e+04 | 7.093e+04 | 7.661e+04 |
| poor_no_tanh | 5.142e+04 | 4.093e+04 | 4.407e+04 |

## Sparsity

| Model | step_high_speed_2500rpm_2A | step_low_speed_500rpm_2A | step_mid_speed_1500rpm_2A |
| --- | --- | --- | --- |
| best_incremental_snn | 0.8792 | 0.9408 | 0.9112 |
| intermediate_scheduled_sampling | 0.886 | 0.905 | 0.8974 |
| poor_no_tanh | 0.9311 | 0.9452 | 0.941 |

## Spikes/step

| Model | step_high_speed_2500rpm_2A | step_low_speed_500rpm_2A | step_mid_speed_1500rpm_2A |
| --- | --- | --- | --- |
| best_incremental_snn | 1670 | 817.7 | 1228 |
| intermediate_scheduled_sampling | 1576 | 1314 | 1419 |
| poor_no_tanh | 952.2 | 758 | 816.1 |
