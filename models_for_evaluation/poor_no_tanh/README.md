# Poor no-tanh SNN controller (open-loop only)

**Original name:** v9_no_tanh (from `v9/v9_no_tanh.pt`)

## What this model is

Feed-forward rate-coded SNN for PMSM current control with **absolute voltage output**. Same 12-feature input and architecture as v10 (dual-population, [128, 96, 64], 48 rate steps, no tanh), but trained only with **open-loop** supervised imitation—no scheduled sampling or closed-loop phase.

- **Input:** 12 features (i_d, i_q, e_d, e_q, n, de_d, de_q, EMAs, dn)
- **Output:** Absolute voltage normalized ±1.0
- **Architecture:** Same as v10; ~22k parameters.

## How it was trained

- **Notebook:** `notebooks/train_snn_v9.ipynb`
- **Data:** v3 balanced H5 (300 episodes, ~597k samples)
- **Training:** 20 epochs, open-loop only. MSE loss + spike rate regularization. No closed-loop or scheduled sampling (exposure bias not addressed).
- **Constants:** `i_max=10.8 A`, `u_max=48 V`, `n_max=3000 rpm`, `error_gain=4.0`.

## Training / benchmark performance

- **Poor** of the three evaluation models; used to validate that the benchmark flags bad controllers.
- Quick benchmark: A_step_pos 1.03 A, **B_step_neg 8.57 A**, C_high_speed 2.08 A RMSE_q.
- **Severe instability on negative step** (B_step_neg); steady-state error ~8.5 A there.

## What it’s good at

- Validating that the benchmark **correctly flags controller failure** and instability.
- Showing the importance of closed-loop (or scheduled-sampling) training: v10 and v12 are much more stable.
- Serving as the “poor” baseline in the good / intermediate / poor ranking.

## Checkpoint

- **This folder:** `model.pt` (copy of original `evaluation/trained_models/v9/v9_no_tanh.pt`)
