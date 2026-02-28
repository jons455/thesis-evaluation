# Appendix C — Phase 4 Reproducibility Raw Data

Per-run metric values for R6, R7, R8 and full hardware/software environment for reproducibility.

## Phase 4 report (excerpt)

```
PVP Phase 4 — Reproducibility
Model: best_incremental_snn
Repeats: 3, Seed: 42
Device: CPU
Python: 3.11.1
PyTorch: 2.4.1+cu124
OS: Windows 10
Generated: 2026-02-26 19:49:09

Notes:
  EXACT         — bitwise identical across all repeats
  sigma < 1e-10 — float64 rounding noise, functionally deterministic
  sigma > 1e-10 — genuine non-determinism
  sigma=nan     — all values non-finite (controller never settled)

Scenario: step_low_speed_500rpm_2A
  mae_i_q                       : sigma=1.11e-16
  mae_i_d                       : EXACT
  settling_time_i_q             : sigma=nan (all non-finite)
  overshoot                     : EXACT
  total_syops                   : EXACT
  mean_sparsity                 : EXACT

Scenario: step_mid_speed_1500rpm_2A
  mae_i_q                       : sigma=1.11e-16
  mae_i_d                       : EXACT
  settling_time_i_q             : sigma=nan (all non-finite)
  overshoot                     : sigma=1.42e-14
  total_syops                   : EXACT
  mean_sparsity                 : EXACT
...
```

## Raw per-run files

- `R6_best_incremental_snn.json` (present)
- `R7_best_incremental_snn.json` (present)
- `R8_best_incremental_snn.json` (present)

## Hardware and software environment (template)

Fill or verify when capturing for a specific PVP run:

| Item | Value |
| --- | --- |
| CPU | Intel64 Family 6 Model 151 Stepping 5, GenuineIntel |
| OS | Windows 10 |
| Kernel (if Linux) | (run `uname -r`) |
| Python | 3.11.1 |
| NumPy | 2.1.3 |
| PyTorch | 2.4.1+cu124 |
| Deterministic setting | `torch.use_deterministic_algorithms(True, warn_only=True)`; `torch.backends.cudnn.deterministic = True`; `torch.backends.cudnn.benchmark = False` |

See `embark-evaluation/plots/utils/common.py` → `setup_deterministic(seed)`.
