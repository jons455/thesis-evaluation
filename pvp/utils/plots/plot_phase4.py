"""
Phase 4 Plots — Reproducibility Validation.

Table 4.1: Per-metric σ heatmap across scenarios (should be zero on CPU).
Plot 4.1: Metric deviation across repeats — grouped bar showing σ values.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# Key metrics to show in the heatmap (skip rarely informative ones)
_KEY_METRICS = [
    "mae_i_q",
    "mae_i_d",
    "itae_i_q",
    "itae_i_d",
    "settling_time_i_q",
    "overshoot",
    "total_syops",
    "mean_sparsity",
]


def plot_sigma_heatmap(results_dir: Path, plots_dir: Path) -> None:
    """Table 4.1 — Per-metric σ heatmap across scenarios.

    Rows = metrics, columns = scenarios.
    Cell values: σ across N repeats (should all be 0.0 on CPU).
    """
    sigma_path = results_dir / "phase4_sigma_table.json"
    if not sigma_path.exists():
        print("  [skip] phase4_sigma_table.json not found for Table 4.1")
        return

    sigma_table = _load_json(sigma_path)
    scenario_names = sorted(sigma_table.keys())
    if not scenario_names:
        print("  [skip] No scenarios in sigma table")
        return

    # Filter to key metrics that actually exist in data
    available = set()
    for sn in scenario_names:
        available.update(sigma_table[sn].keys())
    metrics = [m for m in _KEY_METRICS if m in available]
    if not metrics:
        metrics = sorted(available)[:10]

    # Build matrix
    data = np.zeros((len(metrics), len(scenario_names)))
    for j, sn in enumerate(scenario_names):
        for i, mk in enumerate(metrics):
            data[i, j] = sigma_table[sn].get(mk, 0.0)

    all_zero = np.all(data == 0.0)

    # --- Heatmap ---
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(scenario_names)), max(4, 0.6 * len(metrics))))

    if all_zero:
        # All zeros: use a uniform green color
        cmap = plt.cm.Greens
        im = ax.imshow(
            np.zeros_like(data),
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
    else:
        cmap = plt.cm.YlOrRd
        im = ax.imshow(
            data,
            aspect="auto",
            cmap=cmap,
        )
        plt.colorbar(im, ax=ax, label="σ (standard deviation)", shrink=0.7)

    # Annotate cells
    for i in range(len(metrics)):
        for j in range(len(scenario_names)):
            val = data[i, j]
            if val == 0.0:
                text = "0"
                color = "white" if not all_zero else "darkgreen"
            else:
                text = f"{val:.2e}"
                color = "white" if val > data.max() * 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color, fontweight="bold")

    # Axis labels
    short_scenarios = [s.replace("_", "\n") for s in scenario_names]
    ax.set_xticks(np.arange(len(scenario_names)))
    ax.set_xticklabels(short_scenarios, fontsize=7, rotation=0)
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels([m.replace("_", " ") for m in metrics], fontsize=8)

    verdict = "σ = 0 for all metrics (PASS)" if all_zero else "Non-zero σ detected — check GPU non-determinism"
    ax.set_title(
        f"Reproducibility — Per-Metric Standard Deviation Across Repeats\n({verdict})",
        fontweight="bold",
        fontsize=10,
    )

    plt.tight_layout()
    out = plots_dir / "p4_1_sigma_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_repeat_deviation_bars(results_dir: Path, plots_dir: Path) -> None:
    """Plot 4.2 — Per-metric σ as grouped bars across scenarios.

    Useful when σ > 0 (GPU runs). Skipped if all σ = 0.
    """
    sigma_path = results_dir / "phase4_sigma_table.json"
    if not sigma_path.exists():
        print("  [skip] phase4_sigma_table.json not found for Plot 4.2")
        return

    sigma_table = _load_json(sigma_path)
    scenario_names = sorted(sigma_table.keys())

    available = set()
    for sn in scenario_names:
        available.update(sigma_table[sn].keys())
    metrics = [m for m in _KEY_METRICS if m in available]
    if not metrics:
        return

    # Check if anything is non-zero
    has_nonzero = any(
        sigma_table[sn].get(mk, 0.0) > 0.0
        for sn in scenario_names
        for mk in metrics
    )
    if not has_nonzero:
        print("  [skip] All σ = 0 — no bar chart needed for Plot 4.2")
        return

    x = np.arange(len(metrics))
    width = 0.7 / len(scenario_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(scenario_names)))

    for j, sn in enumerate(scenario_names):
        vals = [sigma_table[sn].get(mk, 0.0) for mk in metrics]
        short = sn.replace("step_", "").replace("_", " ")[:25]
        ax.bar(x + j * width, vals, width, color=colors[j], alpha=0.85, label=short)

    ax.set_xticks(x + width * (len(scenario_names) - 1) / 2)
    ax.set_xticklabels([m.replace("_", " ") for m in metrics], fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("σ (standard deviation)")
    ax.set_title("Metric Standard Deviation Across Repeats", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = plots_dir / "p4_2_repeat_deviation_bars.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def generate_phase4_plots(results_dir: Path, plots_dir: Path) -> None:
    """Entry point: generate all Phase 4 plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("Phase 4 plots:")
    plot_sigma_heatmap(results_dir, plots_dir)
    plot_repeat_deviation_bars(results_dir, plots_dir)
