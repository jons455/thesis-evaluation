"""
Phase 5 Plots — HIL Deployment Feasibility.

Plot 5.1: Latency waterfall — round-trip vs chip inference vs 0.1 ms budget.
Plot 5.2: SC-6a tolerance comparison — R12 vs R13 deviation per metric.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Tolerance bands matching Phase 5 script
_TOLERANCES = {
    "mae_i_q": {"type": "relative", "value": 0.01, "label": "MAE $i_q$"},
    "mae_i_d": {"type": "relative", "value": 0.01, "label": "MAE $i_d$"},
    "itae_i_q": {"type": "relative", "value": 0.01, "label": "ITAE $i_q$"},
    "settling_time_i_q": {"type": "absolute", "value": 0.0001, "label": "Settling Time"},
    "overshoot": {"type": "absolute", "value": 0.5, "label": "Overshoot"},
}


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_latency_waterfall(results_dir: Path, plots_dir: Path) -> None:
    """Plot 5.1 — Latency waterfall: round-trip vs chip inference time.

    Shows mean and p95 round-trip latency, chip inference time, and the
    0.1 ms control timestep budget as reference line.
    """
    data_path = results_dir / "phase5_results.json"
    if not data_path.exists():
        print("  [skip] phase5_results.json not found for Plot 5.1")
        return

    data = _load_json(data_path)
    r12 = data.get("R12", {})
    scenarios_data = r12.get("scenario_results", [])
    if not scenarios_data:
        print("  [skip] No scenario results in R12 for Plot 5.1")
        return

    scenario_names = []
    mean_roundtrip = []
    p95_roundtrip = []
    chip_time_ms = []

    for sr in scenarios_data:
        m = sr.get("metrics", {})
        mean_lat = m.get("mean_latency_ms")
        p95_lat = m.get("p95_latency_ms")
        chip_us = m.get("chip_mean_us")

        if mean_lat is None:
            continue

        scenario_names.append(sr.get("scenario_name", "unknown"))
        mean_roundtrip.append(float(mean_lat))
        p95_roundtrip.append(float(p95_lat) if p95_lat is not None else float(mean_lat))
        chip_time_ms.append(float(chip_us) / 1000.0 if chip_us is not None else 0.0)

    if not scenario_names:
        print("  [skip] No latency data available for Plot 5.1")
        return

    x = np.arange(len(scenario_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(scenario_names)), 5))

    # Stacked-style bars: chip inference (bottom) + overhead (rest of round-trip)
    overhead_ms = [rt - chip for rt, chip in zip(mean_roundtrip, chip_time_ms)]

    bars_chip = ax.bar(
        x - width / 2, chip_time_ms, width,
        color="#4CAF50", alpha=0.85, label="Chip Inference"
    )
    bars_overhead = ax.bar(
        x - width / 2, overhead_ms, width,
        bottom=chip_time_ms,
        color="#FF9800", alpha=0.85, label="Network + Host Overhead"
    )
    bars_p95 = ax.bar(
        x + width / 2, p95_roundtrip, width,
        color="#F44336", alpha=0.65, label="P95 Round-Trip"
    )

    # 0.1 ms budget line
    timestep_ms = 0.1
    ax.axhline(
        timestep_ms, color="red", ls="--", lw=1.5, alpha=0.8,
        label=f"Control Timestep ({timestep_ms} ms)"
    )

    # Annotate values
    for i, (mean_rt, p95, chip) in enumerate(zip(mean_roundtrip, p95_roundtrip, chip_time_ms)):
        ax.text(i - width / 2, mean_rt + 0.02 * max(mean_roundtrip),
                f"{mean_rt:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.text(i + width / 2, p95 + 0.02 * max(p95_roundtrip),
                f"{p95:.3f}", ha="center", va="bottom", fontsize=7, color="#F44336")

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenario_names], fontsize=7)
    ax.set_ylabel("Latency [ms]")
    ax.set_title("HIL Latency Breakdown — Chip Inference vs. Round-Trip", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = plots_dir / "p5_1_latency_waterfall.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_sc6a_tolerance_comparison(results_dir: Path, plots_dir: Path) -> None:
    """Plot 5.2 — SC-6a hardware repeatability: R12 vs R13 deviation vs tolerance band.

    Horizontal bar chart, one bar per metric×scenario, colored by pass/fail.
    """
    data_path = results_dir / "phase5_results.json"
    if not data_path.exists():
        print("  [skip] phase5_results.json not found for Plot 5.2")
        return

    data = _load_json(data_path)
    r12_scenarios = data.get("R12", {}).get("scenario_results", [])
    r13_scenarios = data.get("R13", {}).get("scenario_results", [])

    if not r12_scenarios or not r13_scenarios:
        print("  [skip] Need both R12 and R13 data for Plot 5.2")
        return

    labels = []
    deviations = []
    tolerances = []
    passes = []

    for sr12, sr13 in zip(r12_scenarios, r13_scenarios):
        sn = sr12.get("scenario_name", "?")
        m12 = sr12.get("metrics", {})
        m13 = sr13.get("metrics", {})

        for mk, tol_info in _TOLERANCES.items():
            v12 = m12.get(mk)
            v13 = m13.get(mk)
            if v12 is None or v13 is None:
                continue

            v12, v13 = float(v12), float(v13)
            dev = abs(v13 - v12)

            if tol_info["type"] == "relative":
                band = tol_info["value"] * abs(v12) if abs(v12) > 1e-12 else 1e-12
            else:
                band = tol_info["value"]

            passed = dev <= band
            short_scenario = sn.replace("step_", "").replace("_", " ")[:20]
            labels.append(f"{tol_info['label']}\n({short_scenario})")
            deviations.append(dev / band if band > 0 else 0.0)  # Normalized to tolerance
            tolerances.append(band)
            passes.append(passed)

    if not labels:
        print("  [skip] No metrics to compare for Plot 5.2")
        return

    y = np.arange(len(labels))
    colors = ["#4CAF50" if p else "#F44336" for p in passes]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(labels))))
    ax.barh(y, deviations, color=colors, alpha=0.8, edgecolor="gray", linewidth=0.5)

    # Tolerance boundary at 1.0 (normalized)
    ax.axvline(1.0, color="red", ls="--", lw=1.5, alpha=0.8, label="Tolerance Limit")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Deviation / Tolerance Band (normalized)")
    ax.set_title("SC-6a Hardware Repeatability — R12 vs R13", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")

    # Invert so first metric is at top
    ax.invert_yaxis()

    plt.tight_layout()
    out = plots_dir / "p5_2_sc6a_tolerance_comparison.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def generate_phase5_plots(results_dir: Path, plots_dir: Path) -> None:
    """Entry point: generate all Phase 5 plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("Phase 5 plots:")
    plot_latency_waterfall(results_dir, plots_dir)
    plot_sc6a_tolerance_comparison(results_dir, plots_dir)
