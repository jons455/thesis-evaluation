# PVP Plots

Publication-ready figures generated from a completed PVP run. Each phase writes into its own folder under `embark-evaluation/plots/`.

## How to generate

From the repo root:

```bash
poetry run python embark-evaluation/plots/utils/plot_all.py --results-dir embark-evaluation/pvp/results/<run_name>
```

**Examples**

- **Thesis run** (canonical run for the thesis):
  ```bash
  poetry run python embark-evaluation/plots/utils/plot_all.py --results-dir embark-evaluation/pvp/results/thesis_final
  ```
- Full run (skip Phase 5 if you have no HIL data):
  ```bash
  poetry run python embark-evaluation/plots/utils/plot_all.py --results-dir embark-evaluation/pvp/results/pvp_run3 --skip 5
  ```
- **Phase 5 only** (e.g. from a run that has HIL results):
  ```bash
  poetry run python embark-evaluation/plots/utils/plot_all.py --results-dir embark-evaluation/pvp/results/20260222_110021 --only 5
  ```
  Plots are written to `embark-evaluation/plots/phase5/`.

---

## Phase 1 — Correctness (SC-1)

| File | Description |
|------|-------------|
| `phase1/p1_1_trajectory_overlay_<scenario>.png` | **Plot 1.1** — Trajectory overlay: R1 (PI native) vs R2 (PI via wrapper). One figure per scenario: \(i_q\), \(u_q\) vs time. |
| `phase1/p1_2_residual_trace.png` | **Plot 1.2** — Residual \|R1 − R2\| with symlog Y-axis; reference lines at ±10⁻¹² (pass) and ±10⁻⁶ (fail). |

---

## Phase 2 — Metric validation (SC-2)

| File | Description |
|------|-------------|
| `phase2/p2_1_deviation_histogram.png` | **Plot 2.1** — Histogram of deviations between manual NumPy and pipeline metrics (step_mid_speed_1500rpm_2A). |
| `phase2/p2_2_metric_validation_table.png` | **Plot 2.2** — Table: Manual vs pipeline value and deviation per metric. |
| `phase2/p2_3_deviation_lollipop.png` | **Plot 2.3** — Lollipop chart of deviations (log scale). |

---

## Phase 3 — Discriminative power (SC-3)

| File | Description |
|------|-------------|
| `phase3/p3_1_envelope_best_incremental_snn.png` | **Plot 3.1** — Envelope: v12 SNN vs PI; \(i_q\) and \(u_q\) for all scenarios in one figure. |
| `phase3/p3_E1_relative_error_vs_pi.png` | **Plot E1** — Relative MAE vs PI: (MAE_SNN − MAE_PI) / MAE_PI per scenario (horizontal bars). |
| `phase3/p3_2_mae_grouped_bar.png` | **Plot 3.2** — MAE \(i_q\) grouped bar chart (log scale), PI + SNNs. |
| `phase3/p3_3_itae_grouped_bar.png` | **Plot 3.3** — ITAE \(i_q\) grouped bar chart. |
| `phase3/p3_4_syops_vs_mae_pareto.png` | **Plot 3.4** — SyOps vs MAE Pareto scatter (one point per model, averaged). |
| `phase3/p3_V1_v12_vs_pi_normalized.png` | **Plot V1** — v12 vs PI: key metrics normalized (multi-panel). |
| `phase3/p3_V2_log_ratio_mae.png` | **Plot V2** — log₁₀(MAE / MAE_PI) per scenario. |
| `phase3/p3_V3_radar_summary.png` | **Plot V3** — Normalized multi-metric radar (PI + SNNs; outer = better). |
| `phase3/p3_V4_neuromorphic_radar_snn_vs_snn.png` | **Plot V4** — Neuromorphic radar: SNN vs SNN only (SyOps, sparsity, spikes; outer = better). |

---

## Phase 4 — Reproducibility (SC-4)

| File | Description |
|------|-------------|
| `phase4/p4_1_sigma_heatmap.png` | **Table 4.1** — Per-metric σ across R6–R8 repeats (rows = metrics, columns = scenarios). Shows actual σ (0, float noise e.g. 1e-16, or — for NaN). σ &lt; 1e-10 = deterministic (PASS). |
| `phase4/p4_2_repeat_deviation_bars.png` | **Plot 4.2** — Grouped bars of σ per metric per scenario. Only generated when some σ &gt; 1e-10 (e.g. GPU non-determinism). |

---

## Phase 5 — HIL deployment (SC-6)

*Requires `phase5_hil/` (R12, R13 and optionally phase5_results.json).*

| File | Description |
|------|-------------|
| `phase5/p5_1_latency_waterfall.png` | **Plot 5.1** — Latency waterfall: round-trip vs chip inference vs 0.1 ms control budget. |
| `phase5/p5_2_repeatability_table.png` | **Plot 5.2 (table)** — SC-6a repeatability: R12 vs R13 within tolerance (table figure). |
| `phase5/p5_2_sc6a_tolerance_comparison.png` | **Plot 5.2 (bars)** — Deviation vs tolerance per metric (normalized). |

---

## Phase 6 — Overhead (SC-7)

| File | Description |
|------|-------------|
| `phase6/p6_1_wall_time_breakdown.png` | **Plot 6.1** — Wall time per controller for the **Phase 6 run only** (one sweep: PI + 3 SNNs × scenarios). The total shown is *not* the full PVP time. The 2 h line is the SC-7 budget. When `pvp_summary.json` exists, the plot also shows **Full PVP (all phases)** from that run. |
| `phase6/p6_2_inference_speed.png` | **Plot 6.2** — Inference speed (µs per step) comparison. |

**Note:** If Phase 6 was run with `--quick` (2 scenarios), the total may be ~7–12 min; that is only this profiling run, not the full thesis evaluation. The full PVP (Phases 0–6 with STANDARD scenarios) typically takes ~2.5–3 h; see `pvp_summary.txt` for the actual total.

---

## Layout

```
embark-evaluation/plots/
├── README.md          ← this file
├── phase1/            ← Plot 1.1, 1.2
├── phase2/            ← Plot 2.1, 2.2, 2.3
├── phase3/            ← Plot 3.x, E1, V1–V4
├── phase4/            ← Table 4.1, Plot 4.2
├── phase5/            ← Plot 5.1, 5.2 (if HIL run exists)
├── phase6/            ← Plot 6.1, 6.2
└── utils/             ← plot_phase*.py, plot_all.py
```

Scripts live in `plots/utils/`. The orchestrator is `plot_all.py`; it reads from `pvp/results/<run_name>/phase*_*/` and writes into the `phaseN/` folders above.
