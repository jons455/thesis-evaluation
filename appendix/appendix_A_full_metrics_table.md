# Appendix A — Full Per-Scenario Metric Tables

Complete control (MAE, ITAE, settling time, overshoot) and neuromorphic (SyOps, Sparsity, Spikes) metrics for PI + 3 SNNs across all scenarios.

## Full metrics (4 agents × 6 scenarios)

| Agent | Scenario | MAE i_q (A) | MAE i_d (A) | ITAE i_q (A·s²) | ITAE i_d (A·s²) | Settling time (s) | Overshoot (%) | SyOps | Sparsity | Spikes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PI-baseline | step_low_speed_500rpm_2A | 0.003529 | 0.000465 | 0.000002 | 2.249e-07 | 0.004900 | 78.64 | 0.000000 | 0.000000 | 0.000000 |
| PI-baseline | step_mid_speed_1500rpm_2A | 0.007718 | 0.001498 | 0.000005 | 6.639e-07 | 0.007900 | 117.5 | 0.000000 | 0.000000 | 0.000000 |
| PI-baseline | step_high_speed_2500rpm_2A | 0.01199 | 0.002728 | 0.000009 | 0.000001 | 0.009300 | 156.3 | 0.000000 | 0.000000 | 0.000000 |
| PI-baseline | multi_step_bidirectional_1500rpm | 0.005228 | 0.002950 | 0.000005 | 4.218e-07 | 0.1006 | 118.2 | 0.000000 | 0.000000 | 0.000000 |
| PI-baseline | four_quadrant_transition_1500rpm | 0.004269 | 0.001935 | 0.000005 | 4.218e-07 | 0.1006 | 59.1 | 0.000000 | 0.000000 | 0.000000 |
| PI-baseline | field_weakening_2500rpm | 0.007161 | 0.002468 | 0.000009 | 0.000001 | 0.3511 | 59 | 0.000000 | 0.000000 | 0.000000 |
| best_incremental_snn | step_low_speed_500rpm_2A | 0.945 | 0.5491 | 0.000919 | 0.000402 | — | 70.95 | 1.325e+08 | 0.9408 | 2.453e+06 |
| best_incremental_snn | step_mid_speed_1500rpm_2A | 0.7844 | 0.238 | 0.000918 | 0.000431 | — | 117.8 | 1.989e+08 | 0.9112 | 3.684e+06 |
| best_incremental_snn | step_high_speed_2500rpm_2A | 1.221 | 0.5609 | 0.001482 | 0.001173 | — | 139.4 | 2.705e+08 | 0.8792 | 5.01e+06 |
| best_incremental_snn | multi_step_bidirectional_1500rpm | 0.8476 | 0.2349 | 0.001076 | 0.000470 | — | 78.06 | 6.038e+08 | 0.9191 | 1.118e+07 |
| best_incremental_snn | four_quadrant_transition_1500rpm | 0.8561 | 0.1988 | 0.001076 | 0.000470 | — | 56.43 | 5.168e+08 | 0.9231 | 9.57e+06 |
| best_incremental_snn | field_weakening_2500rpm | 1.118 | 0.4388 | 0.001621 | 0.001027 | — | 64.5 | 4.653e+08 | 0.8961 | 8.617e+06 |
| intermediate_scheduled_sampling | step_low_speed_500rpm_2A | 2.249 | 3.672 | 0.002789 | 0.004636 | — | 219 | 2.128e+08 | 0.905 | 3.941e+06 |
| intermediate_scheduled_sampling | step_mid_speed_1500rpm_2A | 2.439 | 3.576 | 0.003036 | 0.004479 | — | 201.5 | 2.298e+08 | 0.8974 | 4.256e+06 |
| intermediate_scheduled_sampling | step_high_speed_2500rpm_2A | 2.449 | 3.577 | 0.003049 | 0.004571 | — | 193.7 | 2.553e+08 | 0.886 | 4.727e+06 |
| intermediate_scheduled_sampling | multi_step_bidirectional_1500rpm | 2.39 | 2.998 | 0.002933 | 0.003592 | — | 204 | 7.46e+08 | 0.9001 | 1.381e+07 |
| intermediate_scheduled_sampling | four_quadrant_transition_1500rpm | 2.358 | 2.88 | 0.002933 | 0.003592 | — | 204 | 6.685e+08 | 0.9005 | 1.238e+07 |
| intermediate_scheduled_sampling | field_weakening_2500rpm | 2.294 | 3.045 | 0.002709 | 0.001553 | — | 191.1 | 4.594e+08 | 0.8974 | 8.508e+06 |
| poor_no_tanh | step_low_speed_500rpm_2A | 1.78 | 3.333 | 0.002219 | 0.004182 | — | 192.8 | 1.228e+08 | 0.9452 | 2.274e+06 |
| poor_no_tanh | step_mid_speed_1500rpm_2A | 2.406 | 14.69 | 0.003022 | 0.01868 | — | 188.9 | 1.322e+08 | 0.941 | 2.448e+06 |
| poor_no_tanh | step_high_speed_2500rpm_2A | 2.839 | 16.31 | 0.003528 | 0.02036 | — | 198.3 | 1.543e+08 | 0.9311 | 2.857e+06 |
| poor_no_tanh | multi_step_bidirectional_1500rpm | 2.689 | 15.02 | 0.003524 | 0.01934 | — | 154.4 | 4.587e+08 | 0.9386 | 8.494e+06 |
| poor_no_tanh | four_quadrant_transition_1500rpm | 2.747 | 15.17 | 0.003524 | 0.01934 | — | 154.4 | 4.157e+08 | 0.9381 | 7.699e+06 |
| poor_no_tanh | field_weakening_2500rpm | 2.862 | 14.48 | 0.003610 | 0.01995 | — | 198.8 | 3.095e+08 | 0.9309 | 5.732e+06 |
