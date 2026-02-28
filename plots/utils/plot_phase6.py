"""
Phase 6 Plots — Overhead Profiling.

Timing data (wall time per controller, µs per step) is tabular by nature;
no plot functions are defined here. The raw numbers are reported in
phase6_timing.json and surfaced in pvp_summary.txt.
"""

from __future__ import annotations

from pathlib import Path


def generate_phase6_plots(results_dir: Path, plots_dir: Path) -> None:
    """Entry point: generate Phase 6 plots (none currently — data is tabular)."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("Phase 6 plots: none (timing data reported in phase6_timing.json)")
