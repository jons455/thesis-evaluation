# Appendix A — Full Per-Scenario Metric Tables

Complete MAE, ITAE, settling time, and overshoot for PI + 3 SNNs across all scenarios.

## Full metrics (4 agents × 6 scenarios)

| Agent | Scenario | MAE i_q (A) | ITAE i_q (A·s²) | Settling time (s) | Overshoot (%) |
| --- | --- | --- | --- | --- | --- |
| PI-baseline | step_low_speed_500rpm_2A | 0.003529 | 0.000002 | 0.004900 | 78.64 |
| PI-baseline | step_mid_speed_1500rpm_2A | 0.007718 | 0.000005 | 0.007900 | 117.5 |
| PI-baseline | step_high_speed_2500rpm_2A | 0.01199 | 0.000009 | 0.009300 | 156.3 |
| PI-baseline | multi_step_bidirectional_1500rpm | 0.005228 | 0.000005 | 0.1006 | 118.2 |
| PI-baseline | four_quadrant_transition_1500rpm | 0.004269 | 0.000005 | 0.1006 | 59.1 |
| PI-baseline | field_weakening_2500rpm | 0.007161 | 0.000009 | 0.3511 | 59 |
| best_incremental_snn | step_low_speed_500rpm_2A | 0.945 | 0.000919 | — | 70.95 |
| best_incremental_snn | step_mid_speed_1500rpm_2A | 0.7844 | 0.000918 | — | 117.8 |
| best_incremental_snn | step_high_speed_2500rpm_2A | 1.221 | 0.001482 | — | 139.4 |
| best_incremental_snn | multi_step_bidirectional_1500rpm | 0.8476 | 0.001076 | — | 78.06 |
| best_incremental_snn | four_quadrant_transition_1500rpm | 0.8561 | 0.001076 | — | 56.43 |
| best_incremental_snn | field_weakening_2500rpm | 1.118 | 0.001621 | — | 64.5 |
| intermediate_scheduled_sampling | step_low_speed_500rpm_2A | 2.249 | 0.002789 | — | 219 |
| intermediate_scheduled_sampling | step_mid_speed_1500rpm_2A | 2.439 | 0.003036 | — | 201.5 |
| intermediate_scheduled_sampling | step_high_speed_2500rpm_2A | 2.449 | 0.003049 | — | 193.7 |
| intermediate_scheduled_sampling | multi_step_bidirectional_1500rpm | 2.39 | 0.002933 | — | 204 |
| intermediate_scheduled_sampling | four_quadrant_transition_1500rpm | 2.358 | 0.002933 | — | 204 |
| intermediate_scheduled_sampling | field_weakening_2500rpm | 2.294 | 0.002709 | — | 191.1 |
| poor_no_tanh | step_low_speed_500rpm_2A | 1.78 | 0.002219 | — | 192.8 |
| poor_no_tanh | step_mid_speed_1500rpm_2A | 2.406 | 0.003022 | — | 188.9 |
| poor_no_tanh | step_high_speed_2500rpm_2A | 2.839 | 0.003528 | — | 198.3 |
| poor_no_tanh | multi_step_bidirectional_1500rpm | 2.689 | 0.003524 | — | 154.4 |
| poor_no_tanh | four_quadrant_transition_1500rpm | 2.747 | 0.003524 | — | 154.4 |
| poor_no_tanh | field_weakening_2500rpm | 2.862 | 0.003610 | — | 198.8 |
