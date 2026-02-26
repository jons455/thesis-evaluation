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


# Professional palette (aligned with Phase 3 and Phase 5)
_CONTROLLER_COLORS = {
    "PI-baseline":                     "#2d2d2d",
    "best_incremental_snn":            "#4477aa",
    "intermediate_scheduled_sampling": "#228833",
    "poor_no_tanh":                    "#cc6677",
}

_CONTROLLER_LABELS = {
    "PI-baseline":                     "PI",
    "best_incremental_snn":            "SNN (v12)",
    "intermediate_scheduled_sampling": "SNN (v10)",
    "poor_no_tanh":                    "SNN (v9)",
}

_COLOR_REF = "#2d2d2d"


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _get_color(name: str) -> str:
    return _CONTROLLER_COLORS.get(name, "gray")


def _get_label(name: str) -> str:
    return _CONTROLLER_LABELS.get(name, name)


def plot_wall_time_bar(results_dir: Path, plots_dir: Path) -> None:
    """Plot 6.1 — Total wall-time per controller with 2-hour budget line.

    Shows Phase 6 run only (one sweep: PI + 3 models × scenarios). The 2 h line
    is the SC-7 budget. Full PVP (all phases 0–6) takes longer; see pvp_summary.txt.
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

    # Title: make clear this is Phase 6 run only, not full PVP
    ax.set_title("Phase 6 run: PI + 3 SNNs × scenarios (this run only; full PVP = all phases)", fontsize=10)

    bars = ax.barh(y, wall_times, color=colors, alpha=0.85, edgecolor="gray", linewidth=0.5)

    # Annotate each bar with time
    for i, (bar, wt) in enumerate(zip(bars, wall_times)):
        if wt >= 60:
            time_str = f"{wt / 60:.1f} min"
        else:
            time_str = f"{wt:.1f} s"
        ax.text(bar.get_width() + max(wall_times) * 0.02, bar.get_y() + bar.get_height() / 2,
                time_str, ha="left", va="center", fontsize=9, fontweight="bold")

    # Total annotation: Phase 6 total + optional full PVP from pvp_summary
    run_dir = results_dir.parent
    pvp_summary_path = run_dir / "pvp_summary.json"
    total_lines = [f"Phase 6 total: {total_wall:.1f} s ({total_wall / 60:.1f} min)"]
    try:
        if pvp_summary_path.exists():
            summary = _load_json(pvp_summary_path)
            full_pvp_s = summary.get("total_time_s")
            if isinstance(full_pvp_s, (int, float)) and full_pvp_s > 0:
                total_lines.append(f"Full PVP (all phases): {full_pvp_s / 60:.1f} min")
    except Exception:
        pass
    ax.text(
        0.98, 0.02,
        "\n".join(total_lines),
        transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8e8e8", alpha=0.9),
    )

    budget_s = 7200
    if total_wall > budget_s * 0.1:
        ax.axvline(budget_s, color=_COLOR_REF, ls="--", lw=1.5, alpha=0.8, label="SC-7 budget (2 h)")
        ax.legend(fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Wall Time [s]")
    ax.grid(alpha=0.3, axis="x")
    ax.invert_yaxis()

    plt.tight_layout()
    out = plots_dir / "p6_1_wall_time_breakdown.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
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

    timestep_us = 100.0
    if max(us_per_step) > timestep_us * 0.3:
        ax.axhline(timestep_us, color=_COLOR_REF, ls="--", lw=1.5, alpha=0.8,
                   label=f"Control timestep ({timestep_us:.0f} µs)")
        ax.legend(fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Time per Step [µs]")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = plots_dir / "p6_2_inference_speed.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def generate_phase6_plots(results_dir: Path, plots_dir: Path) -> None:
    """Entry point: generate all Phase 6 plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("Phase 6 plots:")
    plot_wall_time_bar(results_dir, plots_dir)
    plot_inference_speed(results_dir, plots_dir)
