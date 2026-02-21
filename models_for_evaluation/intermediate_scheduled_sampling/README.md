# Intermediate scheduled-sampling SNN controller

**Original name:** v10_scheduled_sampling (from `v10/v10_scheduled_sampling.pt`)

## What this model is

Feed-forward rate-coded SNN for PMSM current control with **absolute voltage output** (`u_d`, `u_q`). Uses the same 12-feature input as the v9 family (currents, errors, speed, derivatives, EMAs, speed derivative) but was trained with scheduled sampling and a closed-loop phase to reduce exposure bias.

- **Input:** 12 features (i_d, i_q, e_d, e_q, n, de_d, de_q, EMAs, dn)
- **Output:** Absolute voltage normalized ±1.0, denormalized to ±u_max
- **Architecture:** Dual-population encoding, hidden [128, 96, 64], 48 rate steps, linear readout (no tanh). ~22k parameters.

## How it was trained

- **Notebook:** `notebooks/train_snn_v10.ipynb`
- **Data:** H5 from Drive (e.g. `training_data_v3.h5`)
- **Three-stage scheduled sampling:**
  1. **Stage 1 (epochs 1–10):** Open-loop, teacher forcing ε = 1.0.
  2. **Stage 2 (epochs 11–30):** ε decays linearly 1.0 → 0.1.
  3. **Stage 3 (epochs 31–40):** Full closed-loop, ε = 0.0.
- **Constants:** `i_max=10.8 A`, `u_max=48 V`, `n_max=3000 rpm`, `error_gain=4.0`.

## Training / benchmark performance

- Intermediate of the three evaluation models (better than v9_no_tanh, worse than v12).
- Quick benchmark: A_step_pos 0.90 A, B_step_neg 1.54 A, C_high_speed 0.93 A RMSE_q.
- Noticeable degradation on negative step (B_step_neg) vs v12.

## What it’s good at

- Showing that the benchmark **distinguishes intermediate** performance.
- Demonstrating the benefit of scheduled sampling over pure open-loop (v9).
- Middle reference between best (v12) and poor (v9_no_tanh) controllers.

## Checkpoint

- **This folder:** `model.pt` (copy of original `evaluation/trained_models/v10/v10_scheduled_sampling.pt`)
