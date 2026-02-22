"""
Phase 2 Plots — Metric Validation.

Plot 2.1: Histogram of deviations between manual NumPy and pipeline accumulator.
Plot 2.2: Clean white table — Manual vs pipeline metric comparison.
Plot 2.3: Lollipop chart of deviations on log scale.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_comparisons(results_dir: Path):
    val_path = results_dir / "phase2_validation.json"
    if not val_path.exists():
        print("  [skip] phase2_validation.json not found")
        return None, None, None
    with open(val_path) as f:
        data = json.load(f)
    return data["comparisons"], data["scenario"], data.get("step_onset", "?")


def plot_deviation_histogram(results_dir: Path, plots_dir: Path) -> None:
    """Plot 2.1 — Histogram of deviations between manual and pipeline metrics."""
    comparisons, scenario, step_onset = _load_comparisons(results_dir)
    if comparisons is None:
        return

    metrics = []
    deviations = []
    verdicts = []
    for c in comparisons:
        dev = c["deviation"]
        if dev is not None and not np.isnan(dev):
            metrics.append(c["metric"])
            deviations.append(dev)
            verdicts.append(c["verdict"])

    if not deviations:
        print("  [skip] No valid deviations for histogram")
        return

    verdict_colors = {
        "PASS": "#4CAF50",
        "INVESTIGATE": "#FFC107",
        "HARD FAIL": "#F44336",
        "NaN": "#9E9E9E",
    }
    colors = [verdict_colors.get(v, "#9E9E9E") for v in verdicts]

    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(metrics))
    bars = ax.barh(y, deviations, color=colors, edgecolor="gray", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=9)
    ax.set_xlabel("Absolute Deviation")
    ax.set_xscale("symlog", linthresh=1e-18)
    ax.set_title(
        f"Metric Validation — Manual vs Pipeline Deviations\n{scenario}, step onset k={step_onset}",
        fontweight="bold",
        fontsize=10,
    )
    ax.axvline(0, color="black", lw=0.5)
    ax.grid(alpha=0.3, axis="x")
    ax.invert_yaxis()

    # Annotate values
    for bar, dev in zip(bars, deviations):
        ax.text(
            bar.get_width(), bar.get_y() + bar.get_height() / 2,
            f"  {dev:.2e}", va="center", fontsize=7,
        )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="PASS"),
        Patch(facecolor="#FFC107", label="INVESTIGATE"),
        Patch(facecolor="#F44336", label="HARD FAIL"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    plt.tight_layout()
    out = plots_dir / "p2_1_deviation_histogram.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_metric_validation_table(results_dir: Path, plots_dir: Path) -> None:
    """Plot 2.2 — Clean white table: Manual vs pipeline metric comparison."""
    comparisons, scenario, step_onset = _load_comparisons(results_dir)
    if comparisons is None:
        return

    col_labels = ["Metric", "Manual", "Pipeline", "Deviation", "Verdict"]
    cell_text = []

    for c in comparisons:
        manual = f"{c['manual']:.10f}" if c["manual"] is not None and not np.isnan(c["manual"]) else "NaN"
        pipeline = f"{c['pipeline']:.10f}" if c["pipeline"] is not None and not np.isnan(c["pipeline"]) else "NaN"
        dev = f"{c['deviation']:.2e}" if c["deviation"] is not None and not np.isnan(c["deviation"]) else "NaN"
        verdict = c["verdict"]
        cell_text.append([c["metric"], manual, pipeline, dev, verdict])

    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.45 * len(cell_text)))
    ax.axis("off")
    ax.set_title(
        f"Table 2.2 — Metric Validation (SC-2)\n{scenario}, step onset k={step_onset}",
        fontweight="bold",
        fontsize=11,
        pad=12,
    )

    # White table with light grid
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Style: white background, light gray header
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.5)
        if key[0] == 0:
            cell.set_facecolor("#F5F5F5")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("white")

    plt.tight_layout()
    out = plots_dir / "p2_2_metric_validation_table.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_deviation_lollipop(results_dir: Path, plots_dir: Path) -> None:
    """Plot 2.3 — Lollipop chart of deviations on log scale."""
    comparisons, scenario, step_onset = _load_comparisons(results_dir)
    if comparisons is None:
        return

    metrics = []
    deviations = []
    verdicts = []
    for c in comparisons:
        dev = c["deviation"]
        if dev is not None and not np.isnan(dev) and dev > 0:
            metrics.append(c["metric"])
            deviations.append(dev)
            verdicts.append(c["verdict"])

    if not deviations:
        print("  [skip] No non-zero deviations for lollipop chart")
        return

    verdict_colors = {
        "PASS": "#4CAF50",
        "INVESTIGATE": "#FFC107",
        "HARD FAIL": "#F44336",
    }
    colors = [verdict_colors.get(v, "#9E9E9E") for v in verdicts]

    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(metrics))

    # Stems
    for i, (dev, color) in enumerate(zip(deviations, colors)):
        ax.plot([0, dev], [i, i], color=color, lw=2, zorder=2)
        ax.scatter(dev, i, color=color, s=80, zorder=3, edgecolors="black", linewidths=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Absolute Deviation (log scale)")
    ax.set_title(
        f"Metric Deviations — Lollipop Chart\n{scenario}, step onset k={step_onset}",
        fontweight="bold",
        fontsize=10,
    )
    ax.grid(alpha=0.3, axis="x")
    ax.invert_yaxis()

    plt.tight_layout()
    out = plots_dir / "p2_3_deviation_lollipop.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def generate_phase2_plots(results_dir: Path, plots_dir: Path) -> None:
    """Entry point: generate all Phase 2 plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("Phase 2 plots:")
    plot_deviation_histogram(results_dir, plots_dir)
    plot_metric_validation_table(results_dir, plots_dir)
    plot_deviation_lollipop(results_dir, plots_dir)
