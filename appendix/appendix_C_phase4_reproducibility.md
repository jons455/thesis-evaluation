# Appendix C — Phase 4 Reproducibility Raw Data

Per-run metric values for R6, R7, R8 and full hardware/software environment for reproducibility.

(phase4_reproducibility/ not found.)

## Hardware and software environment (template)

Fill or verify when capturing for a specific PVP run:

| Item | Value |
| --- | --- |
| CPU | Intel64 Family 6 Model 126 Stepping 5, GenuineIntel |
| OS | Windows 10 |
| Kernel (if Linux) | (run `uname -r`) |
| Python | 3.11.9 |
| NumPy | 1.26.4 |
| PyTorch | 2.10.0+cpu |
| Deterministic setting | `torch.use_deterministic_algorithms(True, warn_only=True)`; `torch.backends.cudnn.deterministic = True`; `torch.backends.cudnn.benchmark = False` |

See `embark-evaluation/plots/utils/common.py` → `setup_deterministic(seed)`.
