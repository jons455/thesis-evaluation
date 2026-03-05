"""
Phase 0 Plots — Ground Truth Neuromorphic Baselines.

Plot 0.1: Neuromorphic baselines across operating speeds — grouped bar chart
          showing SyOps/step, activation sparsity, and spikes/step for each
          SNN model at three motor speeds (500, 1500, 2500 rpm).

          Key insight: neuromorphic computational cost scales with motor speed
          (higher speed → more spikes needed for back-EMF compensation).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Reuse Phase 3 style constants for consistency across all PVP plots
_MODEL_STYLES = {
    "best_incremental_snn":            {"color": "#4477aa", "label": "best_incremental_snn"},
    "intermediate_scheduled_sampling": {"color": "#228833", "label": "intermediate_scheduled_sampling"},
    "poor_no_tanh":                    {"color": "#cc6677", "label": "poor_no_tanh"},
}

_SNN_MODEL_ORDER = [
    "best_incremental_snn",
    "intermediate_scheduled_sampling",
    "poor_no_tanh",
]

# Scenario display order (low -> mid -> high speed) and short labels
_SCENARIO_ORDER = [
    "step_low_speed_500rpm_2A",
    "step_mid_speed_1500rpm_2A",
    "step_high_speed_2500rpm_2A",
]

_SCENARIO_SHORT = {
    "step_low_speed_500rpm_2A":  "500 rpm",
    "step_mid_speed_1500rpm_2A": "1500 rpm",
    "step_high_speed_2500rpm_2A": "2500 rpm",
}

_SNN_SHORT_LABEL = {
    "best_incremental_snn":            "best_incr.",
    "intermediate_scheduled_sampling": "intermediate",
    "poor_no_tanh":                    "poor_no_tanh",
}


def _get_style(name: str) -> dict:
    return _MODEL_STYLES.get(name, {"color": "gray", "label": name})


def plot_neuromorphic_baselines(results_dir: Path, plots_dir: Path) -> None:
    """Plot 0.1 — Neuromorphic baselines across operating speeds.

    Three subplots (1 row x 3 cols):
      - SyOps/step       per scenario, grouped by model
      - Activation sparsity  per scenario, grouped by model
      - Spikes/step      per scenario, grouped by model

    Each subplot shows 3 speed groups (500/1500/2500 rpm) with 3 bars
    per group (one per SNN model).  Reveals how neuromorphic computational
    cost scales with motor operating speed.
    """
    neuro_path = results_dir / "phase0_neuromorphic.json"
    if not neuro_path.exists():
        print("  [skip] phase0_neuromorphic.json not found for Plot 0.1")
        return

    with open(neuro_path, encoding="utf-8") as f:
        neuro_data = json.load(f)

    models = [m for m in _SNN_MODEL_ORDER if m in neuro_data]
    if not models:
        print("  [skip] No SNN models in neuromorphic baselines for Plot 0.1")
        return

    scenarios = [s for s in _SCENARIO_ORDER if s in neuro_data[models[0]]]
    if not scenarios:
        print("  [skip] No matching scenarios for Plot 0.1")
        return

    x_labels = [_SCENARIO_SHORT.get(s, s) for s in scenarios]
    x = np.arange(len(scenarios))
    n_models = len(models)
    total_group_width = 0.72
    w = total_group_width / n_models

    # Metric definitions: (json_key, ylabel, use_scientific_notation)
    metric_defs = [
        ("syops_per_step",  "SyOps / step",           True),
        ("mean_sparsity",   "Activation Sparsity",     False),
        ("spikes_per_step", "Spikes / step",           False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax, (metric_key, ylabel, sci_notation) in zip(axes, metric_defs):
        for i, model in enumerate(models):
            style = _get_style(model)
            offset = (i - (n_models - 1) / 2) * w
            vals = [
                neuro_data[model].get(s, {}).get(metric_key, 0.0)
                for s in scenarios
            ]
            bars = ax.bar(
                x + offset, vals, w,
                color=style["color"], alpha=0.85,
                label=_SNN_SHORT_LABEL.get(model, model),
                zorder=3,
            )
            # Value labels on top of bars
            for bar_obj, val in zip(bars, vals):
                if metric_key == "mean_sparsity":
                    label_text = f"{val:.3f}"
                elif val >= 1000:
                    label_text = f"{val / 1000:.1f}k"
                else:
                    label_text = f"{val:.0f}"
                ax.text(
                    bar_obj.get_x() + bar_obj.get_width() / 2,
                    bar_obj.get_height(),
                    label_text,
                    ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.3, axis="y", zorder=0)
        ax.tick_params(axis="both", labelsize=8)

        if sci_notation:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        # Sparsity axis: fix range to show differences clearly
        if metric_key == "mean_sparsity":
            all_vals = [
                neuro_data[m].get(s, {}).get(metric_key, 0.0)
                for m in models for s in scenarios
            ]
            y_min = min(all_vals) - 0.02
            y_max = max(all_vals) + 0.02
            ax.set_ylim(max(0, y_min), min(1.0, y_max))

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=n_models,
        fontsize=8.5, bbox_to_anchor=(0.5, 1.02),
        frameon=True, framealpha=0.9,
    )

    fig.suptitle(
        "Phase 0 — Neuromorphic Baselines Across Operating Speeds",
        fontweight="bold", fontsize=11, y=1.08,
    )
    plt.tight_layout()

    out = plots_dir / "p0_1_neuromorphic_baselines.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def generate_phase0_plots(results_dir: Path, plots_dir: Path) -> None:
    """Entry point: generate all Phase 0 plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("Phase 0 plots:")
    plot_neuromorphic_baselines(results_dir, plots_dir)
