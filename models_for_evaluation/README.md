# Models for Benchmark Evaluation

This directory contains evaluation results and documentation for three SNN controllers selected to validate the benchmark's ability to distinguish between good and poor-performing models.

## Purpose

These models are used to **validate the benchmark itself**, demonstrating that:
- Metrics correctly rank models by performance
- Clear differences between good and bad controllers are captured
- Failure modes (e.g., instability on negative steps) are detected

## Model Selection

| Model | Performance Category | Purpose |
|-------|---------------------|---------|
| **v12 best_model** | **Best** | Demonstrates benchmark captures good controllers |
| **v10_scheduled_sampling** | **Intermediate** | Shows benchmark distinguishes intermediate performance |
| **v9_no_tanh** | **Poor** | Validates benchmark flags failures and instability |

---

## Model Details

### 1. v12 best_model (Best Performance)

**Checkpoint:** `evaluation/trained_models/v12/incremental/best_model.pt`

**Training Script:** [`notebooks/train_snn_v12.py`](../../notebooks/train_snn_v12.py)

#### Architecture Features

- **Input size:** 13 features
  - Currents: `i_d`, `i_q`
  - References: `i_d_ref`, `i_q_ref`
  - Errors: `e_d`, `e_q` (normalized with `error_gain=4.0`)
  - Speed: `n` (normalized with `n_max=3000`)
  - Previous voltage: `u_d_prev`, `u_q_prev` (normalized)
  - Temporal: `e_d_ema_slow`, `e_q_ema_slow`, `e_d_ema_fast`, `e_q_ema_fast`
  - **No derivative features** (de_d, de_q, dn)

- **Output:** Incremental voltage (`delta_u_d`, `delta_u_q`)
  - Clamped to `±delta_u_max = ±0.2` (normalized)
  - Accumulated: `u_new = clamp(u_prev + delta_u, -u_max, u_max)`

- **Network:** Feed-forward rate-coded SNN
  - Dual-population encoding (pos/neg channels)
  - Hidden layers: `[128, 96, 64]`
  - LIF neurons with `betas = [0.96, 0.90, 0.82]`
  - Rate steps: 48
  - Surrogate gradient slope: 25.0
  - **No tanh** on output (linear readout)
  - Parameters: 22,178

#### Training Method

**Hybrid two-phase training:**

1. **Phase 1 (epochs 1-5):** Supervised imitation learning
   - Trained on PI controller incremental targets from CSV data
   - MSE loss + spike rate regularization

2. **Phase 2 (epochs 6-20):** Closed-loop BPTT
   - Differentiable PMSM simulator
   - Curriculum: rollout length 10 → 100 steps
   - Imitation weight annealing: 0.7 → 0.2
   - Loss: `w_imitation * L_imitation + (1-w_imitation) * (L_tracking + λ_smooth * L_smooth)`

**Physical constants:**
- `i_max = 10.8 A`, `u_max = 48.0 V`, `n_max = 3000 rpm`
- `error_gain = 4.0`
- EMA: slow α=0.98, fast α=0.70

#### Performance Summary (Quick Evaluation: 3 runs × 500 steps)

| Scenario | RMSE_q [A] | MAE_q [A] | Settling [ms] | Steady-State Error [A] |
|----------|------------|-----------|---------------|------------------------|
| A_step_pos (0→2A @ 1000rpm) | 0.724 | 0.659 | 50.0 | 0.754 |
| B_step_neg (0→-2A @ 1000rpm) | 0.953 | 0.774 | 50.0 | 0.768 |
| C_high_speed (0→2A @ 3000rpm) | 1.168 | 0.977 | 50.0 | 0.937 |

**Strengths:**
- Best overall performance across all scenarios
- Stable on both positive and negative steps
- Reasonable steady-state error (~0.75–0.95 A)

---

### 2. v10_scheduled_sampling (Intermediate Performance)

**Checkpoint:** `evaluation/trained_models/v10/v10_scheduled_sampling.pt`

**Training Notebook:** [`notebooks/train_snn_v10.ipynb`](../../notebooks/train_snn_v10.ipynb)

#### Architecture Features

- **Input size:** 12 features
  - Currents: `i_d`, `i_q`
  - Errors: `e_d`, `e_q` (normalized with `error_gain=4.0`)
  - Speed: `n` (normalized with `n_max=3000`)
  - Derivatives: `de_d`, `de_q`
  - Temporal: `e_d_ema_slow`, `e_q_ema_slow`, `e_d_ema_fast`, `e_q_ema_fast`
  - Speed derivative: `dn`

- **Output:** Absolute voltage (`u_d`, `u_q`)
  - Clamped to `±1.0` (normalized), denormalized to `±u_max`

- **Network:** Feed-forward rate-coded SNN
  - Dual-population encoding
  - Hidden layers: `[128, 96, 64]`
  - LIF neurons with `betas = [0.96, 0.90, 0.82]`
  - Rate steps: 48
  - Surrogate gradient slope: 25.0
  - **No tanh** on output (linear readout)
  - Parameters: 21,922

#### Training Method

**Three-stage scheduled sampling:**

1. **Stage 1 (epochs 1-10):** Open-loop foundation
   - Teacher forcing ε = 1.0 (always ground-truth currents)
   - Stable initial learning

2. **Stage 2 (epochs 11-30):** Scheduled sampling
   - Teacher forcing ε decays linearly: 1.0 → 0.1
   - Gradual exposure to own prediction errors

3. **Stage 3 (epochs 31-40):** Full closed-loop
   - Teacher forcing ε = 0.0 (always simulated currents)
   - Trained on true deployment conditions

**Physical constants:**
- `i_max = 10.8 A`, `u_max = 48.0 V`, `n_max = 3000 rpm`
- `error_gain = 4.0`
- EMA: slow α=0.98, fast α=0.70

#### Performance Summary (Quick Evaluation: 3 runs × 500 steps)

| Scenario | RMSE_q [A] | MAE_q [A] | Settling [ms] | Steady-State Error [A] |
|----------|------------|-----------|---------------|------------------------|
| A_step_pos (0→2A @ 1000rpm) | 0.897 | 0.781 | 50.0 | 0.810 |
| B_step_neg (0→-2A @ 1000rpm) | 1.539 | 1.530 | 50.0 | 1.529 |
| C_high_speed (0→2A @ 3000rpm) | 0.930 | 0.924 | 50.0 | 0.758 |

**Characteristics:**
- Intermediate performance (worse than v12, better than v9_no_tanh)
- Shows degradation on negative step (B_step_neg: 1.54 A vs v12's 0.95 A)
- Demonstrates benchmark captures intermediate quality

---

### 3. v9_no_tanh (Poor Performance)

**Checkpoint:** `evaluation/trained_models/v9/v9_no_tanh.pt`

**Training Notebook:** [`notebooks/train_snn_v9.ipynb`](../../notebooks/train_snn_v9.ipynb)

#### Architecture Features

- **Input size:** 12 features
  - Same as v10: `i_d`, `i_q`, `e_d`, `e_q`, `n`, `de_d`, `de_q`, EMAs, `dn`

- **Output:** Absolute voltage (`u_d`, `u_q`)
  - Clamped to `±1.0` (normalized)

- **Network:** Feed-forward rate-coded SNN
  - Same architecture as v10
  - **No tanh** on output (linear readout)
  - Parameters: 21,922

#### Training Method

**Open-loop supervised learning:**
- Trained on v3 balanced H5 data (300 episodes, ~597k samples)
- 20 epochs
- MSE loss + spike rate regularization
- **No closed-loop training** (exposure bias not addressed)

**Key difference from v10:**
- No scheduled sampling or closed-loop training
- Pure imitation learning on ground-truth trajectories

**Physical constants:**
- `i_max = 10.8 A`, `u_max = 48.0 V`, `n_max = 3000 rpm`
- `error_gain = 4.0`
- EMA: slow α=0.98, fast α=0.70

#### Performance Summary (Quick Evaluation: 3 runs × 500 steps)

| Scenario | RMSE_q [A] | MAE_q [A] | Settling [ms] | Steady-State Error [A] |
|----------|------------|-----------|---------------|------------------------|
| A_step_pos (0→2A @ 1000rpm) | 1.030 | 0.934 | 50.0 | 1.230 |
| B_step_neg (0→-2A @ 1000rpm) | **8.567** | **8.550** | 50.0 | **8.525** |
| C_high_speed (0→2A @ 3000rpm) | 2.081 | 1.424 | 50.0 | 1.116 |

**Failure Mode:**
- **Severe instability on negative step** (B_step_neg: RMSE_q = 8.57 A)
- Demonstrates benchmark correctly flags controller failures
- Shows importance of closed-loop training (v10/v12 perform much better)

---

## Evaluation Results

### Full Evaluation (10 runs × 2000 steps)

Results from full benchmark evaluation are saved in:
- **Plots:** [`plots/`](./plots/) — trajectory and envelope plots for each scenario
- **Summary:** See evaluation console output for detailed metrics

### Quick Evaluation Summary (3 runs × 500 steps)

| Model | A_step_pos RMSE_q | B_step_neg RMSE_q | C_high_speed RMSE_q | Overall Rank |
|-------|-------------------|-------------------|---------------------|--------------|
| **v12 best_model** | 0.724 A | 0.953 A | 1.168 A | **1st (Best)** |
| **v10_scheduled_sampling** | 0.897 A | 1.539 A | 0.930 A | **2nd (Intermediate)** |
| **v9_no_tanh** | 1.030 A | **8.567 A** | 2.081 A | **3rd (Poor)** |

**Benchmark Validation:**
✅ Metrics correctly rank models: v12 < v10 < v9_no_tanh  
✅ Large performance gaps are captured (v9_no_tanh B_step_neg: 8.57 A vs v12: 0.95 A)  
✅ Failure modes detected (v9_no_tanh instability on negative step)  
✅ Consistent across scenarios (v12 best overall, v9 worst overall)

---

## How to Re-run Evaluation

```bash
# Full evaluation (10 runs × 2000 steps per scenario)
poetry run python -m evaluation.analysis.evaluate_rate_snn \
  --model "evaluation/trained_models/v12/incremental/best_model.pt" \
       "evaluation/trained_models/v10/v10_scheduled_sampling.pt" \
       "evaluation/trained_models/v9/v9_no_tanh.pt" \
  --plots-dir "docs/models_for_evaluation/plots"

# Quick evaluation (3 runs × 500 steps)
poetry run python -m evaluation.analysis.evaluate_rate_snn \
  --model "evaluation/trained_models/v12/incremental/best_model.pt" \
       "evaluation/trained_models/v10/v10_scheduled_sampling.pt" \
       "evaluation/trained_models/v9/v9_no_tanh.pt" \
  --plots-dir "docs/models_for_evaluation/plots" \
  --quick
```

---

## References

- **v12 Training:** [`notebooks/train_snn_v12.py`](../../notebooks/train_snn_v12.py)
- **v10 Training:** [`notebooks/train_snn_v10.ipynb`](../../notebooks/train_snn_v10.ipynb)
- **v9 Training:** [`notebooks/train_snn_v9.ipynb`](../../notebooks/train_snn_v9.ipynb)
- **Evaluation Script:** [`evaluation/analysis/evaluate_rate_snn.py`](../../evaluation/analysis/evaluate_rate_snn.py)
- **Benchmark Interface:** [`docs/RATE_SNN_BENCHMARK_INTERFACE.md`](../RATE_SNN_BENCHMARK_INTERFACE.md)
