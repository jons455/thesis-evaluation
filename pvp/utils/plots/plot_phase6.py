"""
Phase 6 Plots — Overhead Profiling.

Plot 6.1: Wall-time breakdown — stacked bar per controller.
Plot 6.2: Inference speed — µs per step comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Consistent coloring matching Phase 3
_CONTROLLER_COLORS = {
    "PI-baseline":                     "#212121",
    "best_incremental_snn":            "#2196F3",
    "intermediate_scheduled_sampling": "#FF9800",
    "poor_no_tanh":                    "#F44336",
}

_CONTROLLER_LABELS = {
    "PI-baseline":                     "PI Baseline",
    "best_incremental_snn":            "Best SNN (v12)",
    "intermediate_scheduled_sampling": "Intermediate SNN (v10)",
    "poor_no_tanh":                    "Poor SNN (v9)",
}


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _get_color(name: str) -> str:
    return _CONTROLLER_COLORS.get(name, "gray")


def _get_label(name: str) -> str:
    return _CONTROLLER_LABELS.get(name, name)


def plot_wall_time_bar(results_dir: Path, plots_dir: Path) -> None:
    """Plot 6.1 — Total wall-time per controller with 2-hour budget line.

    Simple horizontal bar chart showing how long each controller takes
    for the full scenario suite.
    """
    timing_path = results_dir / "phase6_timing.json"
    if not timing_path.exists():
        print("  [skip] phase6_timing.json not found for Plot 6.1")
        return

    data = _load_json(timing_path)
    timing = data.get("timing", [])
    if not timing:
        print("  [skip] No timing data for Plot 6.1")
        return

    names = [_get_label(t["name"]) for t in timing]
    wall_times = [t["wall_time_s"] for t in timing]
    colors = [_get_color(t["name"]) for t in timing]
    total_wall = data.get("total_wall_s", sum(wall_times))

    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(9, max(3, 0.8 * len(names))))
    bars = ax.barh(y, wall_times, color=colors, alpha=0.85, edgecolor="gray", linewidth=0.5)

    # Annotate each bar with time
    for i, (bar, wt) in enumerate(zip(bars, wall_times)):
        if wt >= 60:
            time_str = f"{wt / 60:.1f} min"
        else:
            time_str = f"{wt:.1f} s"
        ax.text(bar.get_width() + max(wall_times) * 0.02, bar.get_y() + bar.get_height() / 2,
                time_str, ha="left", va="center", fontsize=9, fontweight="bold")

    # Total annotation
    ax.text(
        0.98, 0.02,
        f"Total: {total_wall:.1f} s ({total_wall / 60:.1f} min)",
        transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", alpha=0.8),
    )

    # 2-hour budget (only show if scale makes sense)
    budget_s = 7200
    if total_wall > budget_s * 0.1:
        ax.axvline(budget_s, color="red", ls="--", lw=1.5, alpha=0.7, label="2-Hour Budget (SC-7)")
        ax.legend(fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Wall Time [s]")
    ax.set_title("Benchmark Execution Time per Controller", fontweight="bold")
    ax.grid(alpha=0.3, axis="x")
    ax.invert_yaxis()

    plt.tight_layout()
    out = plots_dir / "p6_1_wall_time_breakdown.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_inference_speed(results_dir: Path, plots_dir: Path) -> None:
    """Plot 6.2 — Inference speed (µs per step) comparison.

    Vertical bar chart with one bar per controller showing the mean
    microseconds per simulation step.
    """
    timing_path = results_dir / "phase6_timing.json"
    if not timing_path.exists():
        print("  [skip] phase6_timing.json not found for Plot 6.2")
        return

    data = _load_json(timing_path)
    timing = data.get("timing", [])

    # Filter to entries that have time_per_step_us
    entries = [(t["name"], t.get("time_per_step_us")) for t in timing if t.get("time_per_step_us")]
    if not entries:
        print("  [skip] No per-step timing data for Plot 6.2")
        return

    names = [_get_label(n) for n, _ in entries]
    us_per_step = [v for _, v in entries]
    colors = [_get_color(n) for n, _ in entries]

    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, us_per_step, color=colors, alpha=0.85, edgecolor="gray", linewidth=0.5)

    # Annotate bars
    for bar, val in zip(bars, us_per_step):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(us_per_step) * 0.02,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # Control timestep reference (100 µs = 0.1 ms at 10 kHz)
    timestep_us = 100.0
    if max(us_per_step) > timestep_us * 0.3:
        ax.axhline(timestep_us, color="red", ls="--", lw=1.5, alpha=0.7,
                    label=f"Control Timestep ({timestep_us:.0f} µs)")
        ax.legend(fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Time per Step [µs]")
    ax.set_title("Inference Speed — Simulation Steps per Microsecond", fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = plots_dir / "p6_2_inference_speed.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def generate_phase6_plots(results_dir: Path, plots_dir: Path) -> None:
    """Entry point: generate all Phase 6 plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("Phase 6 plots:")
    plot_wall_time_bar(results_dir, plots_dir)
    plot_inference_speed(results_dir, plots_dir)
