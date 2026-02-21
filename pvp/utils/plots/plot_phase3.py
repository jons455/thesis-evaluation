"""
Phase 3 Plots — Discriminative Power.

Plot 3.1: Step response grid (controllers x scenarios).
Plot E1:  Relative error vs PI — (MAE_snn - MAE_pi) / MAE_pi.
Plot 3.2: MAE grouped bar chart, log scale.
Plot 3.3: ITAE grouped bar chart.
Plot 3.4: SyOps vs MAE Pareto scatter.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Consistent model styling across all Phase 3 plots
MODEL_STYLES = {
    "PI-baseline":                     {"color": "#212121", "marker": "s", "label": "PI Baseline"},
    "best_incremental_snn":            {"color": "#2196F3", "marker": "o", "label": "Best SNN (v12)"},
    "intermediate_scheduled_sampling": {"color": "#FF9800", "marker": "^", "label": "Intermediate SNN (v10)"},
    "poor_no_tanh":                    {"color": "#F44336", "marker": "D", "label": "Poor SNN (v9)"},
}

MODEL_ORDER = ["PI-baseline", "best_incremental_snn", "intermediate_scheduled_sampling", "poor_no_tanh"]


def _get_style(name: str) -> dict:
    return MODEL_STYLES.get(name, {"color": "gray", "marker": "x", "label": name})


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _extract_metric_table(summaries: dict, metric_key: str) -> dict[str, dict[str, float]]:
    """Extract {model: {scenario: value}} from phase3_summaries.json."""
    table: dict[str, dict[str, float]] = {}
    for model_name, summary in summaries.items():
        table[model_name] = {}
        for sr in summary.get("scenarios", []):
            val = sr.get("metrics", {}).get(metric_key, float("nan"))
            table[model_name][sr["name"]] = val if val is not None else float("nan")
    return table


# ──────────────────────────────────────────────────────────────────────
# Plot 3.1: Step response grid
# ──────────────────────────────────────────────────────────────────────

def plot_step_response_grid(results_dir: Path, plots_dir: Path) -> None:
    """Plot 3.1 — Step response grid, one column per scenario, all controllers overlaid."""
    traj_files = sorted(results_dir.glob("trajectory_*.json"))
    if not traj_files:
        print("  [skip] No trajectory files for Plot 3.1")
        return

    # Group by scenario
    scenarios_data: dict[str, dict[str, dict]] = {}
    for tf in traj_files:
        # filename: trajectory_{model}_{scenario}.json
        parts = tf.stem.replace("trajectory_", "")
        # Find scenario by matching known scenario suffixes
        for model in MODEL_ORDER:
            if parts.startswith(model + "_"):
                scenario = parts[len(model) + 1:]
                scenarios_data.setdefault(scenario, {})[model] = tf
                break

    if not scenarios_data:
        print("  [skip] Could not parse trajectory filenames for Plot 3.1")
        return

    scenario_names = sorted(scenarios_data.keys())
    n_scen = len(scenario_names)

    fig, axes = plt.subplots(2, n_scen, figsize=(5 * n_scen, 6), sharex=False, squeeze=False)

    for col, scenario in enumerate(scenario_names):
        ref_plotted = False
        for model in MODEL_ORDER:
            if model not in scenarios_data[scenario]:
                continue
            traj_path = scenarios_data[scenario][model]
            with open(traj_path) as f:
                traj = json.load(f)
            t_ms = np.array(traj["t"]) * 1000
            i_q = np.array(traj["i_q"])
            i_q_ref = np.array(traj["i_q_ref"])
            u_q = np.array(traj["u_q"])
            style = _get_style(model)

            if not ref_plotted:
                axes[0, col].plot(t_ms, i_q_ref, "k--", lw=1.2, alpha=0.5, label="Reference")
                ref_plotted = True

            axes[0, col].plot(t_ms, i_q, color=style["color"], lw=1.0, alpha=0.85, label=style["label"])
            axes[1, col].plot(t_ms, u_q, color=style["color"], lw=0.8, alpha=0.85, label=style["label"])

        short_name = scenario.replace("_", " ").replace("rpm", " rpm").replace("2A", "2 A")
        axes[0, col].set_title(short_name, fontsize=8, fontweight="bold")
        axes[0, col].grid(alpha=0.3)
        axes[1, col].grid(alpha=0.3)
        axes[1, col].set_xlabel("Time [ms]")

    axes[0, 0].set_ylabel("$i_q$ [A]")
    axes[1, 0].set_ylabel("$v_q$ [V]")
    axes[0, 0].legend(fontsize=7, loc="best")

    fig.suptitle("Step Response Comparison — All Controllers", fontweight="bold", y=1.02)
    plt.tight_layout()
    out = plots_dir / "p3_1_step_response_grid.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ──────────────────────────────────────────────────────────────────────
# Plot E1: Relative error vs PI
# ──────────────────────────────────────────────────────────────────────

def plot_relative_error_vs_pi(results_dir: Path, plots_dir: Path) -> None:
    """Plot E1 — (MAE_snn - MAE_pi) / MAE_pi per scenario, horizontal bar."""
    mae_path = results_dir / "phase3_mae_table.json"
    if not mae_path.exists():
        print("  [skip] phase3_mae_table.json not found for Plot E1")
        return

    mae_table = _load_json(mae_path)
    pi_mae = mae_table.get("PI-baseline", {})
    if not pi_mae:
        return

    snn_models = [m for m in MODEL_ORDER if m != "PI-baseline" and m in mae_table]
    scenarios = sorted(pi_mae.keys())

    fig, ax = plt.subplots(figsize=(8, 4))
    y_positions = np.arange(len(scenarios))
    bar_height = 0.22

    for i, model in enumerate(snn_models):
        rel_errors = []
        for scen in scenarios:
            pi_val = pi_mae.get(scen, float("nan"))
            snn_val = mae_table[model].get(scen, float("nan"))
            if pi_val > 1e-12:
                rel_errors.append((snn_val - pi_val) / pi_val)
            else:
                rel_errors.append(0.0)

        style = _get_style(model)
        ax.barh(
            y_positions + i * bar_height,
            rel_errors,
            height=bar_height,
            color=style["color"],
            alpha=0.8,
            label=style["label"],
        )

    ax.axvline(0, color="black", lw=0.8, ls="-")
    ax.set_yticks(y_positions + bar_height)
    ax.set_yticklabels([s.replace("_", "\n") for s in scenarios], fontsize=7)
    ax.set_xlabel("Relative MAE Difference vs PI")
    ax.set_title("Relative Tracking Error vs. PI Baseline", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    out = plots_dir / "p3_E1_relative_error_vs_pi.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ──────────────────────────────────────────────────────────────────────
# Plot 3.2: MAE grouped bar chart (log scale)
# ──────────────────────────────────────────────────────────────────────

def plot_mae_grouped_bar(results_dir: Path, plots_dir: Path) -> None:
    """Plot 3.2 — MAE_i_q grouped bar chart, log scale."""
    mae_path = results_dir / "phase3_mae_table.json"
    if not mae_path.exists():
        print("  [skip] phase3_mae_table.json not found for Plot 3.2")
        return

    mae_table = _load_json(mae_path)
    models = [m for m in MODEL_ORDER if m in mae_table]
    scenarios = sorted(next(iter(mae_table.values())).keys())

    x = np.arange(len(scenarios))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        vals = [mae_table[model].get(s, float("nan")) for s in scenarios]
        style = _get_style(model)
        ax.bar(x + i * width, vals, width, color=style["color"], alpha=0.85, label=style["label"])

    ax.set_yscale("log")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=7)
    ax.set_ylabel("MAE $i_q$ [A] (log scale)")
    ax.set_title("Mean Absolute Error — All Controllers", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = plots_dir / "p3_2_mae_grouped_bar.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ──────────────────────────────────────────────────────────────────────
# Plot 3.3: ITAE grouped bar chart
# ──────────────────────────────────────────────────────────────────────

def plot_itae_grouped_bar(results_dir: Path, plots_dir: Path) -> None:
    """Plot 3.3 — ITAE_i_q grouped bar chart."""
    summ_path = results_dir / "phase3_summaries.json"
    if not summ_path.exists():
        print("  [skip] phase3_summaries.json not found for Plot 3.3")
        return

    summaries = _load_json(summ_path)
    itae_table = _extract_metric_table(summaries, "itae_i_q")

    models = [m for m in MODEL_ORDER if m in itae_table]
    if not models:
        print("  [skip] No ITAE data for Plot 3.3")
        return

    scenarios = sorted(next(iter(itae_table.values())).keys())
    x = np.arange(len(scenarios))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        vals = [itae_table[model].get(s, 0) or 0 for s in scenarios]
        style = _get_style(model)
        ax.bar(x + i * width, vals, width, color=style["color"], alpha=0.85, label=style["label"])

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=7)
    ax.set_ylabel("ITAE $i_q$")
    ax.set_title("Integral of Time-weighted Absolute Error", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = plots_dir / "p3_3_itae_grouped_bar.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ──────────────────────────────────────────────────────────────────────
# Plot 3.4: SyOps vs MAE Pareto scatter
# ──────────────────────────────────────────────────────────────────────

def plot_syops_vs_mae_pareto(results_dir: Path, plots_dir: Path) -> None:
    """Plot 3.4 — SyOps vs MAE_i_q Pareto scatter, one point per model (averaged)."""
    summ_path = results_dir / "phase3_summaries.json"
    if not summ_path.exists():
        print("  [skip] phase3_summaries.json not found for Plot 3.4")
        return

    summaries = _load_json(summ_path)
    mae_table = _extract_metric_table(summaries, "mae_i_q")
    syops_table = _extract_metric_table(summaries, "total_syops")

    fig, ax = plt.subplots(figsize=(7, 5))

    for model in MODEL_ORDER:
        if model not in mae_table or model not in syops_table:
            continue
        mae_vals = [v for v in mae_table[model].values() if v is not None and not np.isnan(v)]
        syops_vals = [v for v in syops_table[model].values() if v is not None and not np.isnan(v)]

        if not mae_vals or not syops_vals:
            continue

        avg_mae = np.mean(mae_vals)
        avg_syops = np.mean(syops_vals)
        style = _get_style(model)
        ax.scatter(avg_mae, avg_syops, s=120, color=style["color"],
                   marker=style["marker"], label=style["label"], zorder=5, edgecolors="black", linewidths=0.5)

    ax.set_xlabel("Mean MAE $i_q$ [A]")
    ax.set_ylabel("Mean SyOps per Episode")
    ax.set_title("Efficiency–Accuracy Trade-off (SyOps vs. MAE)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Note about rate-coded upper bound
    ax.text(
        0.02, 0.02,
        "SyOps are upper-bound estimates\nfor rate-coded models",
        transform=ax.transAxes, fontsize=7, fontstyle="italic", alpha=0.6,
        verticalalignment="bottom",
    )

    plt.tight_layout()
    out = plots_dir / "p3_4_syops_vs_mae_pareto.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def generate_phase3_plots(results_dir: Path, plots_dir: Path) -> None:
    """Entry point: generate all Phase 3 plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("Phase 3 plots:")
    plot_step_response_grid(results_dir, plots_dir)
    plot_relative_error_vs_pi(results_dir, plots_dir)
    plot_mae_grouped_bar(results_dir, plots_dir)
    plot_itae_grouped_bar(results_dir, plots_dir)
    plot_syops_vs_mae_pareto(results_dir, plots_dir)
