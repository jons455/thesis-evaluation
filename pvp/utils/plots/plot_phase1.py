"""
Phase 1 Plots — Correctness Probing.

Plot 1.1: Trajectory overlay (i_q and u_q) — R1 vs R2, per scenario.
Plot 1.2: Residual trace |R1 − R2| with symlog Y-axis.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_trajectory(path: Path) -> dict[str, np.ndarray]:
    with open(path, "r") as f:
        raw = json.load(f)
    return {k: np.array(v) for k, v in raw.items()}


def plot_trajectory_overlay(results_dir: Path, plots_dir: Path) -> None:
    """Plot 1.1 — R1 vs R2 trajectory overlay, one figure per scenario."""
    r1_files = sorted(results_dir.glob("R1_trajectory_*.json"))
    if not r1_files:
        print("  [skip] No R1 trajectory files found for Plot 1.1")
        return

    for r1_path in r1_files:
        scenario = r1_path.stem.replace("R1_trajectory_", "")
        r2_path = results_dir / f"R2_trajectory_{scenario}.json"
        if not r2_path.exists():
            continue

        r1 = _load_trajectory(r1_path)
        r2 = _load_trajectory(r2_path)
        t_ms = r1["t"] * 1000

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # i_q overlay
        axes[0].plot(t_ms, r1["i_q_ref"], "k--", lw=1.5, alpha=0.6, label="Reference")
        axes[0].plot(t_ms, r1["i_q"], color="#2196F3", lw=1.2, label="R1 (PI native)")
        axes[0].plot(t_ms, r2["i_q"], color="#FF5722", lw=1.0, ls="--", label="R2 (PI wrapper)")
        axes[0].set_ylabel("$i_q$ [A]")
        axes[0].legend(loc="best", fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[0].set_title(f"SC-1 Trajectory Overlay — {scenario}", fontweight="bold")

        # u_q overlay
        axes[1].plot(t_ms, r1["u_q"], color="#2196F3", lw=1.2, label="R1 (PI native)")
        axes[1].plot(t_ms, r2["u_q"], color="#FF5722", lw=1.0, ls="--", label="R2 (PI wrapper)")
        axes[1].set_ylabel("$v_q$ [V]")
        axes[1].set_xlabel("Time [ms]")
        axes[1].legend(loc="best", fontsize=8)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        out = plots_dir / f"p1_1_trajectory_overlay_{scenario}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.name}")


def plot_residual_trace(results_dir: Path, plots_dir: Path) -> None:
    """Plot 1.2 — Residual |R1 − R2| with symlog Y-axis, per scenario."""
    r1_files = sorted(results_dir.glob("R1_trajectory_*.json"))
    if not r1_files:
        print("  [skip] No R1 trajectory files found for Plot 1.2")
        return

    fig, axes = plt.subplots(len(r1_files), 1, figsize=(10, 3 * len(r1_files)), sharex=False)
    if len(r1_files) == 1:
        axes = [axes]

    for ax, r1_path in zip(axes, r1_files):
        scenario = r1_path.stem.replace("R1_trajectory_", "")
        r2_path = results_dir / f"R2_trajectory_{scenario}.json"
        if not r2_path.exists():
            continue

        r1 = _load_trajectory(r1_path)
        r2 = _load_trajectory(r2_path)
        n = min(len(r1["i_q"]), len(r2["i_q"]))
        t_ms = r1["t"][:n] * 1000
        residual = np.abs(r1["i_q"][:n] - r2["i_q"][:n])

        ax.plot(t_ms, residual, color="#1565C0", lw=0.8)
        ax.axhline(1e-12, color="green", ls="--", lw=0.8, alpha=0.7, label="Pass ($10^{-12}$ A)")
        ax.axhline(1e-6, color="red", ls="--", lw=0.8, alpha=0.7, label="Hard fail ($10^{-6}$ A)")
        ax.set_yscale("symlog", linthresh=1e-15)
        ax.set_ylabel("|R1 − R2| [A]")
        ax.set_title(scenario, fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time [ms]")
    fig.suptitle("SC-1 Residual Trace — PI Native vs. PI Wrapper", fontweight="bold", y=1.01)
    plt.tight_layout()
    out = plots_dir / "p1_2_residual_trace.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def generate_phase1_plots(results_dir: Path, plots_dir: Path) -> None:
    """Entry point: generate all Phase 1 plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("Phase 1 plots:")
    plot_trajectory_overlay(results_dir, plots_dir)
    plot_residual_trace(results_dir, plots_dir)
