# Best incremental SNN controller

**Original name:** v12 best_model (from `v12/incremental/best_model.pt`)

## What this model is

Feed-forward rate-coded SNN for PMSM current control with **incremental voltage output**: it predicts *changes* to the voltage (`delta_u_d`, `delta_u_q`) which are then accumulated and clamped. This design improves stability and matches the PI controller’s incremental behaviour.

- **Input:** 13 features (currents, refs, errors, speed, previous voltage, EMAs; no derivatives)
- **Output:** Incremental voltage ±0.2 (normalized), accumulated as `u_new = clamp(u_prev + delta_u, -u_max, u_max)`
- **Architecture:** Dual-population encoding, hidden [128, 96, 64], 48 rate steps, LIF betas [0.96, 0.90, 0.82], linear readout (no tanh). ~22k parameters.

## How it was trained

- **Script:** `notebooks/train_snn_v12.py`
- **Data:** CSV from `evaluation/data/raw/train_v3`
- **Two-phase training:**
  1. **Phase 1 (epochs 1–5):** Supervised imitation on PI incremental targets (MSE + spike rate reg).
  2. **Phase 2 (epochs 6–20):** Closed-loop BPTT with differentiable PMSM simulator; curriculum 10→100 steps; imitation weight 0.7→0.2.
- **Constants:** `i_max=10.8 A`, `u_max=48 V`, `n_max=3000 rpm`, `error_gain=4.0`.

## Training / benchmark performance

- Best overall of the three evaluation models.
- Quick benchmark (3 runs × 500 steps): A_step_pos 0.72 A, B_step_neg 0.95 A, C_high_speed 1.17 A RMSE_q.
- Stable on both positive and negative steps; steady-state error ~0.75–0.95 A.

## What it’s good at

- Demonstrating that the benchmark correctly ranks a **good** controller.
- Stable behaviour across nominal positive, nominal negative, and high-speed scenarios.
- Reference for comparing intermediate and poor models.

## Checkpoint

- **This folder:** `model.pt` (copy of original `evaluation/trained_models/v12/incremental/best_model.pt`)
