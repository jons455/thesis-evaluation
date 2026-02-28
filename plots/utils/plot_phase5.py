"""
Phase 5 Plots — HIL Deployment Feasibility.

Plot 5.1: Latency waterfall — round-trip vs chip inference vs 0.1 ms budget.
Plot 5.2: Hardware repeatability — deviation between two HIL runs per metric.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Professional palette (aligned with Phase 3) — neutral: within vs outside tolerance
_COLOR_WITHIN = "#228833"
_COLOR_OUTSIDE = "#cc6677"
_COLOR_CHIP = "#4477aa"
_COLOR_OVERHEAD = "#ee7733"
_COLOR_P95 = "#cc6677"
_COLOR_REF = "#2d2d2d"

# Tolerance bands matching Phase 5 script
_TOLERANCES = {
    "mae_i_q": {"type": "relative", "value": 0.01, "label": "MAE $i_q$"},
    "mae_i_d": {"type": "relative", "value": 0.01, "label": "MAE $i_d$"},
    "itae_i_q": {"type": "relative", "value": 0.01, "label": "ITAE $i_q$"},
    "settling_time_i_q": {"type": "absolute", "value": 0.0001, "label": "Settling Time"},
    "overshoot": {"type": "absolute", "value": 0.5, "label": "Overshoot"},
}


def _load_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    # Allow non-standard Infinity in JSON (from Phase 5 results)
    text = text.replace(": Infinity", ": 1e999").replace(": Infinity ", ": 1e999 ")
    return json.loads(text)


def _safe_float(x, default=None):
    """Convert to float; return default for inf/nan/non-numeric."""
    if x is None:
        return default
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if math.isfinite(v):
        return v
    return default


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
    scenarios_data = r12.get("scenario_results") or r12.get("scenarios", [])
    if not scenarios_data:
        print("  [skip] No scenario results for Plot 5.1")
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

        scenario_names.append(sr.get("scenario_name") or sr.get("name", "unknown"))
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
        color=_COLOR_CHIP, alpha=0.85, label="Chip Inference"
    )
    bars_overhead = ax.bar(
        x - width / 2, overhead_ms, width,
        bottom=chip_time_ms,
        color=_COLOR_OVERHEAD, alpha=0.85, label="Network + Host Overhead"
    )
    bars_p95 = ax.bar(
        x + width / 2, p95_roundtrip, width,
        color=_COLOR_P95, alpha=0.75, label="P95 Round-Trip"
    )

    timestep_ms = 0.1
    ax.axhline(
        timestep_ms, color=_COLOR_REF, ls="--", lw=1.5, alpha=0.8,
        label=f"Control Timestep ({timestep_ms} ms)"
    )

    for i, (mean_rt, p95, chip) in enumerate(zip(mean_roundtrip, p95_roundtrip, chip_time_ms)):
        ax.text(i - width / 2, mean_rt + 0.02 * max(mean_roundtrip),
                f"{mean_rt:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.text(i + width / 2, p95 + 0.02 * max(p95_roundtrip),
                f"{p95:.3f}", ha="center", va="bottom", fontsize=7, color=_COLOR_P95)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenario_names], fontsize=7)
    ax.set_ylabel("Latency [ms]")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = plots_dir / "p5_1_latency_waterfall.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_sc6a_tolerance_comparison(results_dir: Path, plots_dir: Path) -> None:
    """Plot 5.2 — Hardware repeatability: deviation between two HIL runs vs tolerance.

    Two outputs: a compact table (primary), and a zoomed bar chart when all values are small.
    """
    data_path = results_dir / "phase5_results.json"
    if not data_path.exists():
        print("  [skip] phase5_results.json not found for Plot 5.2")
        return

    data = _load_json(data_path)
    run1 = data.get("R12", {})
    run2 = data.get("R13", {})
    s1 = run1.get("scenario_results") or run1.get("scenarios", [])
    s2 = run2.get("scenario_results") or run2.get("scenarios", [])

    if not s1 or not s2:
        print("  [skip] Need both run data for Plot 5.2")
        return

    rows = []  # (metric_label, scenario_short, v1, v2, dev, band, passed, norm)

    for sr1, sr2 in zip(s1, s2):
        sn = sr1.get("scenario_name") or sr1.get("name", "?")
        m1 = sr1.get("metrics", {})
        m2 = sr2.get("metrics", {})
        short_scenario = sn.replace("step_", "").replace("_", " ")[:22]

        for mk, tol_info in _TOLERANCES.items():
            v1 = _safe_float(m1.get(mk))
            v2 = _safe_float(m2.get(mk))
            if v1 is None or v2 is None:
                continue

            dev = abs(v2 - v1)
            if tol_info["type"] == "relative":
                band = tol_info["value"] * abs(v1) if abs(v1) > 1e-12 else 1e-12
            else:
                band = tol_info["value"]

            within_tol = dev <= band
            norm = (dev / band) if band > 0 else 0.0
            rows.append((tol_info["label"], short_scenario, v1, v2, dev, band, within_tol, norm))

    if not rows:
        print("  [skip] No metrics to compare for Plot 5.2")
        return

    # ─── 1) Table figure (clear when all within tolerance) ───
    cell_text = []
    for (metric_label, scenario, v1, v2, dev, band, within_tol, _) in rows:
        status = "within" if within_tol else "outside"
        cell_text.append([
            metric_label,
            scenario,
            f"{v1:.4g}",
            f"{v2:.4g}",
            f"{dev:.4g}",
            f"{band:.4g}",
            status,
        ])
    col_labels = ["Metric", "Scenario", "Run 1", "Run 2", "|diff|", "Tol", "Within tol."]

    fig_table, ax_table = plt.subplots(figsize=(10, 0.4 + 0.32 * len(cell_text)))
    ax_table.axis("off")
    table = ax_table.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#ccc")
        cell.set_linewidth(0.5)
        if key[0] == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("white")
    plt.tight_layout()
    out_table = plots_dir / "p5_2_repeatability_table.png"
    fig_table.savefig(out_table, bbox_inches="tight", dpi=300)
    plt.close(fig_table)
    print(f"  Saved: {out_table.name}")


def generate_phase5_plots(results_dir: Path, plots_dir: Path) -> None:
    """Entry point: generate all Phase 5 plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("Phase 5 plots:")
    plot_latency_waterfall(results_dir, plots_dir)
    plot_sc6a_tolerance_comparison(results_dir, plots_dir)
