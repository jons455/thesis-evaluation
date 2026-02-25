# Appendix F — Motor Parameters and Training Configuration

## Motor electrical parameters (PMSM)

Used for PI gain derivation and benchmark simulation (gym-electric-motor / embark).

| Parameter | Value | Unit |
| --- | --- | --- |
| Pole pairs (p) | 3 | — |
| Stator resistance (R_s) | 0.543 | Ω |
| d-axis inductance (L_d) | 1.13 | mH |
| q-axis inductance (L_q) | 1.42 | mH |
| PM flux linkage (Ψ_PM) | 16.9 | mWb |
| Maximum current (I_max) | 10.8 | A |
| DC-link voltage (V_DC) | 48 | V |
| Maximum speed (n_max) | 3000 | RPM |

Source: README.md and embark/benchmark config; same values in `notebooks/train_snn_v12.py` (V12Config).

## SNN training hyperparameters (v12 — best_incremental_snn)

Reference from `notebooks/train_snn_v12.py` (V12Config). Other probe models (v9, v10) may differ; see their training notebooks.

| Parameter | Value |
| --- | --- |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Batch size (Phase 1) | 256 |
| Phase 1 epochs | 5 (imitation) |
| Phase 2 epochs | 15 (closed-loop BPTT) |
| Rollout length (Phase 2) | 10 → 100 (curriculum) |
| Phase 2 batch size | 64 |
| Phase 2 steps per epoch | 300 |
| Phase 2 LR factor | 0.1 |
| Gradient clip (Phase 1) | 2.0 |
| Phase 2 gradient clip | 0.5 |
| BPTT | Closed-loop through differentiable PMSM |
| Delta u max (output) | 0.2 |
| Hidden sizes | [128, 96, 64] |
| Betas (membrane) | [0.96, 0.90, 0.82] |
| Rate steps | 48 |
| Input size | 13 (refs, prev voltage, EMAs; no derivatives) |
| Output size | 2 (delta u_d, delta u_q) |

Sampling stride and windowing: see training data generation and dataset in `evaluation/pytorch_snn/utils/dataset.py` and training scripts.
