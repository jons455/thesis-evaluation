# Appendix E — Logging and File Structure

## PVP logging checklist

| Run(s) | Logged content |
| --- | --- |
| R1, R2 | Full trajectory (t, i_q_ref, i_q, u_q) |
| R3, R4, R5 | Full trajectory |
| R12 | Full trajectory |
| R0a–R0c | Metrics (MAE) + neuromorphic baselines (SyOps/step, mean sparsity, spikes/step) |
| R6–R11, R13, R14 (opt) | Metrics only (MAE, ITAE, settling time, overshoot, SyOps, sparsity) |

After every run and every scenario: GEM environment, controller state, SNN hidden/membrane state, and metric accumulators are fully reset.

## JSON archive file naming conventions

| Pattern | Meaning |
| --- | --- |
| `phase0_ground_truth/phase0_rankings.json` | MAE_q per model per scenario (Phase 0) |
| `phase0_ground_truth/phase0_neuromorphic.json` | Neuromorphic baselines: SyOps/step, mean sparsity, spikes/step per model per scenario (Phase 0, wrapper-free) |
| `phase0_ground_truth/phase0_report.txt` | Human-readable Phase 0 report |
| `phase2_metric_validation/phase2_validation.json` | Control + neuromorphic metric deviations (manual vs pipeline); includes `neuromorphic_comparisons` array |
| `phase3_discriminative/trajectory_<model>_<scenario>.json` | Trajectory (t, i_q_ref, i_q, u_q) for one agent and one scenario |
| `phase3_discriminative/R2_PI_baseline.json`, `R3_*.json`, … | Full metrics per scenario for that run |
| `phase3_discriminative/phase3_summaries.json` | All agents’ scenario results combined |
| `phase3_discriminative/phase3_mae_table.json` | MAE i_q only, per model per scenario |
| `phase4_reproducibility/R6_best_incremental_snn.json`, R7, R8 | Per-repeat metrics (R6–R8) |
| `phase4_reproducibility/phase4_sigma_table.json` | Per-metric σ across repeats |
| `phase4_reproducibility/phase4_all_metrics.json` | All repeats’ raw metrics |
| `phase5_hil/R12_akida_hil.json`, `R13_akida_hil.json` | HIL run results |
| `phase5_hil/phase5_results.json` | R12 and R13 combined for plots |

## Version-control manifest (template)

Map each PVP run ID to exact artifacts for Open Science reproducibility:

| Run ID | Model weights | Motor/benchmark config | Script / commit |
| --- | --- | --- | --- |
| R0a–R0c | `models_for_evaluation/{best_incremental_snn,intermediate_scheduled_sampling,poor_no_tanh}/model.pt` | embark benchmark STANDARD_SCENARIOS | `pvp/phase0_ground_truth.py` |
| R1, R2 | — (PI) | same | `pvp/phase1_correctness.py` |
| R3–R5 | same as Phase 0 | same | `pvp/phase3_discriminative.py` |
| R6–R8 | `best_incremental_snn/model.pt` | same | `pvp/phase4_reproducibility.py` |
| R9–R11 | (Keras / Akida sim) | (Phase 5-Q) | — |
| R12–R13 | Akida .fbz from Phase 5-Q | same | `pvp/phase5_hil.py` |

Fill in commit hash or tag and config file paths when archiving a run.
