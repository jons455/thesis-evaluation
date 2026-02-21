"""
Phase 2 Plots — Metric Validation.

Table 2.1: Manual NumPy vs. pipeline accumulator comparison (rendered as figure).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_metric_validation_table(results_dir: Path, plots_dir: Path) -> None:
    """Table 2.1 — Manual vs pipeline metric comparison as a publication figure."""
    val_path = results_dir / "phase2_validation.json"
    if not val_path.exists():
        print("  [skip] phase2_validation.json not found")
        return

    with open(val_path) as f:
        data = json.load(f)

    comparisons = data["comparisons"]
    scenario = data["scenario"]
    step_onset = data.get("step_onset", "?")

    # Build table data
    col_labels = ["Metric", "Manual", "Pipeline", "Deviation", "Verdict"]
    cell_text = []
    cell_colors = []

    verdict_colors = {
        "PASS": "#C8E6C9",
        "INVESTIGATE": "#FFF9C4",
        "HARD FAIL": "#FFCDD2",
        "NaN": "#E0E0E0",
    }

    for c in comparisons:
        manual = f"{c['manual']:.10f}" if c["manual"] is not None and not np.isnan(c["manual"]) else "NaN"
        pipeline = f"{c['pipeline']:.10f}" if c["pipeline"] is not None and not np.isnan(c["pipeline"]) else "NaN"
        dev = f"{c['deviation']:.2e}" if c["deviation"] is not None and not np.isnan(c["deviation"]) else "NaN"
        verdict = c["verdict"]

        cell_text.append([c["metric"], manual, pipeline, dev, verdict])
        row_color = verdict_colors.get(verdict, "#FFFFFF")
        cell_colors.append([row_color] * 5)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")
    ax.set_title(
        f"Table 2.1 — Metric Validation (SC-2)\n{scenario}, step onset k={step_onset}",
        fontweight="bold",
        fontsize=11,
        pad=12,
    )

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#E3F2FD"] * 5,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Bold header
    for j in range(len(col_labels)):
        table[0, j].set_text_props(fontweight="bold")

    plt.tight_layout()
    out = plots_dir / "p2_1_metric_validation_table.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def generate_phase2_plots(results_dir: Path, plots_dir: Path) -> None:
    """Entry point: generate all Phase 2 plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("Phase 2 plots:")
    plot_metric_validation_table(results_dir, plots_dir)
