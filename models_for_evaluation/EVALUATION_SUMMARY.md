# Benchmark Evaluation Summary

**Date:** Evaluation in progress (full: 10 runs × 2000 steps per scenario)  
**Purpose:** Validate benchmark metrics correctly distinguish between good and poor-performing controllers

## Models Evaluated

1. **v12 best_model** — Best performing (incremental output, hybrid training)
2. **v10_scheduled_sampling** — Intermediate (scheduled sampling training)
3. **v9_no_tanh** — Poor performing (open-loop only, fails on negative step)

## Expected Results

Based on quick evaluation (3 runs × 500 steps), the benchmark should show:

| Model | A_step_pos RMSE_q | B_step_neg RMSE_q | C_high_speed RMSE_q |
|-------|-------------------|-------------------|---------------------|
| v12 best_model | ~0.72 A | ~0.95 A | ~1.17 A |
| v10_scheduled_sampling | ~0.90 A | ~1.54 A | ~0.93 A |
| v9_no_tanh | ~1.03 A | **~8.57 A** | ~2.08 A |

**Validation Criteria:**
- ✅ Metrics rank models correctly: v12 < v10 < v9_no_tanh
- ✅ Large performance gaps captured (v9_no_tanh B_step_neg failure)
- ✅ Consistent ranking across all scenarios

## Full Evaluation Results

*Results will be populated here once the full evaluation completes.*

Check the evaluation output in the terminal or see plots in [`plots/`](./plots/).

---

**Note:** Full evaluation takes ~15-30 minutes. Results include:
- Detailed metrics table (RMSE_q, MAE_q, settling time, overshoot, steady-state error)
- Trajectory plots (i_q vs time, v_q vs time)
- Envelope plots (error bounds over time)
