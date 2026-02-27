# Appendix B — Supplementary Trajectory Plots

The chapter body keeps **S2** (step_mid_speed_1500rpm_2A) as the representative step response.
This appendix lists artifacts for step response overlays and relative-error (Plot E1) for all scenarios.

## Trajectory data (Phase 3)

| File | Model | Scenario |
| --- | --- | --- |
| trajectory_best_incremental_snn_field_weakening_2500rpm.json | best_incremental_snn | field_weakening_2500rpm |
| trajectory_best_incremental_snn_four_quadrant_transition_1500rpm.json | best_incremental_snn | four_quadrant_transition_1500rpm |
| trajectory_best_incremental_snn_multi_step_bidirectional_1500rpm.json | best_incremental_snn | multi_step_bidirectional_1500rpm |
| trajectory_best_incremental_snn_step_high_speed_2500rpm_2A.json | best_incremental_snn | step_high_speed_2500rpm_2A |
| trajectory_best_incremental_snn_step_low_speed_500rpm_2A.json | best_incremental_snn | step_low_speed_500rpm_2A |
| trajectory_best_incremental_snn_step_mid_speed_1500rpm_2A.json | best_incremental_snn | step_mid_speed_1500rpm_2A |
| trajectory_intermediate_scheduled_sampling_field_weakening_2500rpm.json | intermediate_scheduled_sampling | field_weakening_2500rpm |
| trajectory_intermediate_scheduled_sampling_four_quadrant_transition_1500rpm.json | intermediate_scheduled_sampling | four_quadrant_transition_1500rpm |
| trajectory_intermediate_scheduled_sampling_multi_step_bidirectional_1500rpm.json | intermediate_scheduled_sampling | multi_step_bidirectional_1500rpm |
| trajectory_intermediate_scheduled_sampling_step_high_speed_2500rpm_2A.json | intermediate_scheduled_sampling | step_high_speed_2500rpm_2A |
| trajectory_intermediate_scheduled_sampling_step_low_speed_500rpm_2A.json | intermediate_scheduled_sampling | step_low_speed_500rpm_2A |
| trajectory_intermediate_scheduled_sampling_step_mid_speed_1500rpm_2A.json | intermediate_scheduled_sampling | step_mid_speed_1500rpm_2A |
| trajectory_PI-baseline_field_weakening_2500rpm.json | PI-baseline | field_weakening_2500rpm |
| trajectory_PI-baseline_four_quadrant_transition_1500rpm.json | PI-baseline | four_quadrant_transition_1500rpm |
| trajectory_PI-baseline_multi_step_bidirectional_1500rpm.json | PI-baseline | multi_step_bidirectional_1500rpm |
| trajectory_PI-baseline_step_high_speed_2500rpm_2A.json | PI-baseline | step_high_speed_2500rpm_2A |
| trajectory_PI-baseline_step_low_speed_500rpm_2A.json | PI-baseline | step_low_speed_500rpm_2A |
| trajectory_PI-baseline_step_mid_speed_1500rpm_2A.json | PI-baseline | step_mid_speed_1500rpm_2A |
| trajectory_poor_no_tanh_field_weakening_2500rpm.json | poor_no_tanh | field_weakening_2500rpm |
| trajectory_poor_no_tanh_four_quadrant_transition_1500rpm.json | poor_no_tanh | four_quadrant_transition_1500rpm |
| trajectory_poor_no_tanh_multi_step_bidirectional_1500rpm.json | poor_no_tanh | multi_step_bidirectional_1500rpm |
| trajectory_poor_no_tanh_step_high_speed_2500rpm_2A.json | poor_no_tanh | step_high_speed_2500rpm_2A |
| trajectory_poor_no_tanh_step_low_speed_500rpm_2A.json | poor_no_tanh | step_low_speed_500rpm_2A |
| trajectory_poor_no_tanh_step_mid_speed_1500rpm_2A.json | poor_no_tanh | step_mid_speed_1500rpm_2A |

## Generating supplementary figures

- **Step response overlays (S1, S3–S6):** Use the same plotting logic as Plot 3.1, selecting one scenario per figure.
- **Relative error (Plot E1):** Already includes all scenarios in one figure; see `plots/utils/plot_phase3.py` → `plot_relative_error_vs_pi()`.
