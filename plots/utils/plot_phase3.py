"""
Phase 3 Plots — Discriminative Power.

Plot 3.1: Envelope comparison — v12 vs PI baseline only (1 figure).
Plot E1:  Relative error vs PI — (MAE_snn - MAE_pi) / MAE_pi — v12 only.
Plot 3.2: MAE grouped bar chart, log scale — PI + v12 + v9.
Plot 3.4: SyOps vs MAE Pareto scatter — PI + v12 + v9.
Plot V1:  v12 vs PI — all metrics, normalized.
Plot V2:  Log-ratio — log10(MAE / MAE_PI) — v12 + v9.
Plot V3:  Radar / spider chart — PI + v12 + v9.
Plot V4:  Neuromorphic radar — SNN vs SNN only (SyOps, sparsity, spikes).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Professional, colorblind-friendly palette; neutral labels (no quality interpretation)
MODEL_STYLES = {
    "PI-baseline":                     {"color": "#2d2d2d", "marker": "s", "label": "PI"},
    "best_incremental_snn":            {"color": "#4477aa", "marker": "o", "label": "SNN (v12)"},
    "intermediate_scheduled_sampling": {"color": "#228833", "marker": "^", "label": "SNN (v10)"},
    "poor_no_tanh":                    {"color": "#cc6677", "marker": "D", "label": "SNN (v9)"},
}

# To include/exclude models: edit MODEL_ORDER (and optionally MODEL_STYLES for new models).
# Only models in this list are plotted; comment out or remove a name to drop it from all Phase 3 plots.
# v10 (intermediate_scheduled_sampling) excluded — v12 and v9 cover the relevant extremes.
MODEL_ORDER = ["PI-baseline", "best_incremental_snn", "poor_no_tanh"]
SNN_MODELS = [m for m in MODEL_ORDER if m != "PI-baseline"]

# v12 is the primary SNN for head-to-head PI comparison plots
V12_MODEL = "best_incremental_snn"


def _get_style(name: str) -> dict:
    return MODEL_STYLES.get(name, {"color": "gray", "marker": "x", "label": name})


def _load_json(path: Path) -> dict:
    """Load JSON; tolerate non-standard 'Infinity' in phase results."""
    import re
    text = path.read_text(encoding="utf-8")
    text = re.sub(r":\s*Infinity\b", ": null", text)
    return json.loads(text)


def _extract_metric_table(summaries: dict, metric_key: str) -> dict[str, dict[str, float]]:
    """Extract {model: {scenario: value}} from phase3_summaries.json.

    Supports both "scenarios" (with "name") and "scenario_results" (with "scenario_name").
    """
    table: dict[str, dict[str, float]] = {}
    for model_name, summary in summaries.items():
        table[model_name] = {}
        scenario_list = summary.get("scenarios") or summary.get("scenario_results") or []
        for sr in scenario_list:
            name = sr.get("name") or sr.get("scenario_name")
            if name is None:
                continue
            val = sr.get("metrics", {}).get(metric_key, float("nan"))
            table[model_name][name] = val if val is not None else float("nan")
    return table


# ──────────────────────────────────────────────────────────────────────
# Plot 3.1: Envelope comparison — one SNN vs PI per figure
# ──────────────────────────────────────────────────────────────────────

def plot_envelope_comparison(results_dir: Path, plots_dir: Path) -> None:
    """Plot 3.1 — Envelope plot: v12 SNN vs PI baseline, one figure with all scenarios.

    One column per scenario.
    Top row: i_q with reference (envelope = fill_between for error band).
    Bottom row: u_q comparison.
    """
    traj_files = sorted(results_dir.glob("trajectory_*.json"))
    if not traj_files:
        print("  [skip] No trajectory files for Plot 3.1")
        return

    # Group by scenario — include all models so we can look up paths
    _all_models = ["PI-baseline", "best_incremental_snn", "intermediate_scheduled_sampling", "poor_no_tanh"]
    scenarios_data: dict[str, dict[str, Path]] = {}
    for tf in traj_files:
        parts = tf.stem.replace("trajectory_", "")
        for model in _all_models:
            if parts.startswith(model + "_"):
                scenario = parts[len(model) + 1:]
                scenarios_data.setdefault(scenario, {})[model] = tf
                break

    if not scenarios_data:
        print("  [skip] Could not parse trajectory filenames for Plot 3.1")
        return

    scenario_names = sorted(scenarios_data.keys())
    n_scen = len(scenario_names)

    # Only v12 vs PI — single figure
    for snn_model in [V12_MODEL]:
        snn_style = _get_style(snn_model)
        pi_style = _get_style("PI-baseline")

        fig, axes = plt.subplots(2, n_scen, figsize=(4.5 * n_scen, 5), sharex=False, squeeze=False)

        has_data = False
        for col, scenario in enumerate(scenario_names):
            pi_path = scenarios_data.get(scenario, {}).get("PI-baseline")
            snn_path = scenarios_data.get(scenario, {}).get(snn_model)

            if pi_path is None or snn_path is None:
                continue
            has_data = True

            with open(pi_path) as f:
                pi_traj = json.load(f)
            with open(snn_path) as f:
                snn_traj = json.load(f)

            t_ms = np.array(pi_traj["t"]) * 1000
            pi_iq = np.array(pi_traj["i_q"])
            snn_iq = np.array(snn_traj["i_q"])
            ref_iq = np.array(pi_traj["i_q_ref"])
            pi_uq = np.array(pi_traj["u_q"])
            snn_uq = np.array(snn_traj["u_q"])

            ax_iq = axes[0, col]
            ax_uq = axes[1, col]

            # Reference
            ax_iq.plot(t_ms, ref_iq, "k--", lw=1.2, alpha=0.5, label="Reference")

            # PI baseline
            ax_iq.plot(t_ms, pi_iq, color=pi_style["color"], lw=1.0, alpha=0.7, label=pi_style["label"])

            # SNN model
            ax_iq.plot(t_ms, snn_iq, color=snn_style["color"], lw=1.0, alpha=0.85, label=snn_style["label"])

            # Envelope: fill between PI and SNN to highlight the difference
            ax_iq.fill_between(
                t_ms,
                np.minimum(pi_iq, snn_iq),
                np.maximum(pi_iq, snn_iq),
                alpha=0.15, color=snn_style["color"],
            )

            # Voltage comparison
            ax_uq.plot(t_ms, pi_uq, color=pi_style["color"], lw=0.8, alpha=0.7, label=pi_style["label"])
            ax_uq.plot(t_ms, snn_uq, color=snn_style["color"], lw=0.8, alpha=0.85, label=snn_style["label"])
            ax_uq.fill_between(
                t_ms,
                np.minimum(pi_uq, snn_uq),
                np.maximum(pi_uq, snn_uq),
                alpha=0.15, color=snn_style["color"],
            )

            short_name = scenario.replace("_", " ").replace("rpm", " rpm").replace("2A", "2 A")
            ax_iq.set_title(short_name, fontsize=8, fontweight="bold")
            ax_iq.grid(alpha=0.3)
            ax_uq.grid(alpha=0.3)
            ax_uq.set_xlabel("Time [ms]")

        if not has_data:
            plt.close(fig)
            continue

        axes[0, 0].set_ylabel("$i_q$ [A]")
        axes[1, 0].set_ylabel("$v_q$ [V]")
        axes[0, 0].legend(fontsize=7, loc="best")

        fig.suptitle(f"{snn_style['label']} vs PI", fontweight="bold", y=1.02)
        plt.tight_layout()

        safe_name = snn_model.replace(" ", "_")
        out = plots_dir / f"p3_1_envelope_{safe_name}.png"
        fig.savefig(out, bbox_inches="tight", dpi=200)
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

    # Only v12 for the head-to-head PI comparison
    snn_models = [V12_MODEL] if V12_MODEL in mae_table else []
    scenarios = sorted(pi_mae.keys())

    fig, ax = plt.subplots(figsize=(8, 4))
    y_positions = np.arange(len(scenarios))
    bar_height = 0.5

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
    ax.set_yticks(y_positions)
    ax.set_yticklabels([_SCENARIO_SHORT.get(s, s).replace("\n", " ") for s in scenarios], fontsize=8)
    ax.set_xlabel(r"Relative MAE$_{i_q}$ vs PI  $\left(\frac{\mathrm{MAE}_{SNN} - \mathrm{MAE}_{PI}}{\mathrm{MAE}_{PI}}\right)$", fontsize=9)
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    out = plots_dir / "p3_E1_relative_error_vs_pi.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
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
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = plots_dir / "p3_2_mae_grouped_bar.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
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
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = plots_dir / "p3_3_itae_grouped_bar.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
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
    ax.legend(
        fontsize=9,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.95,
        edgecolor="#ccc",
    )
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    out = plots_dir / "p3_4_syops_vs_mae_pareto.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ──────────────────────────────────────────────────────────────────────
# Plot V1: v12 vs PI — all metrics, normalized where possible
# ──────────────────────────────────────────────────────────────────────

# Physical limits used for normalization
_I_MAX = 10.8   # A  (maximum current)
_U_MAX = 48.0   # V  (maximum voltage)

# Scenario display names (short labels for x-axis)
_SCENARIO_SHORT = {
    "step_low_speed_500rpm_2A":        "Low\n500 rpm",
    "step_mid_speed_1500rpm_2A":       "Mid\n1500 rpm",
    "step_high_speed_2500rpm_2A":      "High\n2500 rpm",
    "multi_step_bidirectional_1500rpm":"Bidirect.\n1500 rpm",
    "four_quadrant_transition_1500rpm":"4-Quad\n1500 rpm",
    "field_weakening_2500rpm":         "Field-Wk.\n2500 rpm",
}


def _get_scenario_metrics(summaries: dict, model: str, metric: str) -> dict[str, float]:
    """Return {scenario_name: value} for one model and metric key.

    Supports both summary formats: "scenarios" (with "name") and "scenario_results"
    (with "scenario_name"), so phase3_summaries from either pipeline work.
    """
    result: dict[str, float] = {}
    model_data = summaries.get(model, {})
    scenario_list = model_data.get("scenarios") or model_data.get("scenario_results") or []
    for sr in scenario_list:
        name = sr.get("name") or sr.get("scenario_name")
        if name is None:
            continue
        val = sr.get("metrics", {}).get(metric)
        result[name] = float(val) if val is not None else float("nan")
    return result


def plot_v12_vs_pi_normalized(results_dir: Path, plots_dir: Path) -> None:
    """Plot V1 — v12 SNN vs PI baseline: all key metrics, normalized where possible.

    Produces a multi-panel figure with one subplot per metric:
      - MAE i_q  [% of i_max]          (normalized)
      - MAE i_d  [% of i_max]          (normalized)
      - RMS i_q  [% of i_max]          (normalized)
      - ITAE i_q [raw, log scale]
      - Max error i_q [% of i_max]     (normalized)
      - Settling time [ms]             (Inf shown as hatched bar at ceiling)

    Each subplot shows both controllers side-by-side per scenario.
    """
    summ_path = results_dir / "phase3_summaries.json"
    if not summ_path.exists():
        print("  [skip] phase3_summaries.json not found for Plot V1")
        return

    summaries = _load_json(summ_path)

    pi_key = "PI-baseline"
    snn_key = "best_incremental_snn"
    if pi_key not in summaries or snn_key not in summaries:
        print("  [skip] PI-baseline or best_incremental_snn missing from summaries")
        return

    pi_style = _get_style(pi_key)
    snn_style = _get_style(snn_key)

    # Ordered scenario names from PI data
    scenarios = [sr["name"] for sr in summaries[pi_key]["scenarios"]]
    x_labels = [_SCENARIO_SHORT.get(s, s) for s in scenarios]
    x = np.arange(len(scenarios))
    w = 0.32  # bar half-width

    # ── collect metrics ──────────────────────────────────────────────
    def _get(model: str, metric: str) -> np.ndarray:
        vals = _get_scenario_metrics(summaries, model, metric)
        return np.array([vals.get(s, float("nan")) for s in scenarios])

    pi_mae_q  = _get(pi_key,  "mae_i_q")
    snn_mae_q = _get(snn_key, "mae_i_q")
    pi_mae_d  = _get(pi_key,  "mae_i_d")
    snn_mae_d = _get(snn_key, "mae_i_d")
    pi_rms_q  = _get(pi_key,  "rms_i_q")
    snn_rms_q = _get(snn_key, "rms_i_q")
    pi_itae_q  = _get(pi_key,  "itae_i_q")
    snn_itae_q = _get(snn_key, "itae_i_q")
    pi_maxerr_q  = _get(pi_key,  "max_error_i_q")
    snn_maxerr_q = _get(snn_key, "max_error_i_q")
    pi_settle  = _get(pi_key,  "settling_time_i_q")
    snn_settle = _get(snn_key, "settling_time_i_q")

    # Normalize current metrics to % of i_max
    pi_mae_q_pct   = pi_mae_q   / _I_MAX * 100
    snn_mae_q_pct  = snn_mae_q  / _I_MAX * 100
    pi_mae_d_pct   = pi_mae_d   / _I_MAX * 100
    snn_mae_d_pct  = snn_mae_d  / _I_MAX * 100
    pi_rms_q_pct   = pi_rms_q   / _I_MAX * 100
    snn_rms_q_pct  = snn_rms_q  / _I_MAX * 100
    pi_maxerr_q_pct   = pi_maxerr_q   / _I_MAX * 100
    snn_maxerr_q_pct  = snn_maxerr_q  / _I_MAX * 100

    # Settling time: replace Inf with NaN for bar height, track which are Inf
    _SETTLE_CAP_MS = 500.0  # ms — ceiling for display when Inf
    pi_settle_ms   = np.where(np.isinf(pi_settle),   np.nan, pi_settle   * 1000)
    snn_settle_ms  = np.where(np.isinf(snn_settle),  np.nan, snn_settle  * 1000)
    pi_settle_inf  = np.isinf(pi_settle)
    snn_settle_inf = np.isinf(snn_settle)
    # Fill capped bar height for Inf cases
    pi_settle_plot  = np.where(pi_settle_inf,  _SETTLE_CAP_MS, pi_settle_ms)
    snn_settle_plot = np.where(snn_settle_inf, _SETTLE_CAP_MS, snn_settle_ms)

    # ── figure layout: 2 rows × 3 cols ──────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()

    subplot_defs = [
        # (ax_idx, pi_vals,          snn_vals,          ylabel,               log_scale, note)
        (0, pi_mae_q_pct,  snn_mae_q_pct,  r"MAE $i_q$ [% of $i_{max}$]",       False, None),
        (1, pi_mae_d_pct,  snn_mae_d_pct,  r"MAE $i_d$ [% of $i_{max}$]",       False, None),
        (2, pi_rms_q_pct,  snn_rms_q_pct,  r"RMS $i_q$ [% of $i_{max}$]",       False, None),
        (3, pi_itae_q,     snn_itae_q,     r"ITAE $i_q$ [A·s²]",                True,  None),
        (4, pi_maxerr_q_pct, snn_maxerr_q_pct, r"Max error $i_q$ [% of $i_{max}$]", False, None),
        (5, pi_settle_plot, snn_settle_plot, "Settling time [ms]",              False, "settle"),
    ]

    for ax_idx, pi_vals, snn_vals, ylabel, log_scale, note in subplot_defs:
        ax = axes[ax_idx]

        bars_pi  = ax.bar(x - w / 2, pi_vals,  w, color=pi_style["color"],
                          alpha=0.85, label=pi_style["label"], zorder=3)
        bars_snn = ax.bar(x + w / 2, snn_vals, w, color=snn_style["color"],
                          alpha=0.85, label=snn_style["label"], zorder=3)

        # Hatching for Inf settling bars
        if note == "settle":
            for i, (inf_pi, inf_snn) in enumerate(zip(pi_settle_inf, snn_settle_inf)):
                if inf_pi:
                    bars_pi[i].set_hatch("///")
                    bars_pi[i].set_edgecolor("white")
                if inf_snn:
                    bars_snn[i].set_hatch("///")
                    bars_snn[i].set_edgecolor("white")
            # Add a note about hatch = did not settle
            ax.text(0.98, 0.97, "/// = did not settle", transform=ax.transAxes,
                    fontsize=6.5, ha="right", va="top", color="#555")
            ax.set_ylim(0, _SETTLE_CAP_MS * 1.12)

        if log_scale:
            ax.set_yscale("log")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(alpha=0.3, axis="y", zorder=0)
        ax.tick_params(axis="x", labelsize=7)

        if ax_idx == 0:
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("SNN (v12) vs PI Baseline — Normalized Performance Metrics",
                 fontweight="bold", fontsize=11, y=1.01)
    plt.tight_layout()

    out = plots_dir / "p3_V1_v12_vs_pi_normalized.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ──────────────────────────────────────────────────────────────────────
# Plot V2: Log-ratio — log10(MAE_snn / MAE_pi) per scenario
# ──────────────────────────────────────────────────────────────────────

def plot_log_ratio_mae(results_dir: Path, plots_dir: Path) -> None:
    """Plot V2 — log10(MAE_snn / MAE_pi) for each SNN model per scenario.

    A value of 2 means the SNN is 100× worse than PI. Gives a compact,
    single-number summary of relative degradation across all scenarios.
    Both i_q and i_d axes are shown side by side.
    """
    summ_path = results_dir / "phase3_summaries.json"
    if not summ_path.exists():
        print("  [skip] phase3_summaries.json not found for Plot V2")
        return

    summaries = _load_json(summ_path)
    pi_key = "PI-baseline"
    if pi_key not in summaries:
        print("  [skip] PI-baseline missing for Plot V2")
        return

    scenarios = [sr["name"] for sr in summaries[pi_key]["scenarios"]]
    x_labels = [_SCENARIO_SHORT.get(s, s) for s in scenarios]
    x = np.arange(len(scenarios))
    snn_models = [m for m in SNN_MODELS if m in summaries]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    width = 0.22

    for ax, metric, ax_title in zip(
        axes,
        ["mae_i_q", "mae_i_d"],
        [r"$i_q$ axis", r"$i_d$ axis"],
    ):
        pi_vals = np.array([
            _get_scenario_metrics(summaries, pi_key, metric).get(s, float("nan"))
            for s in scenarios
        ])

        for i, model in enumerate(snn_models):
            snn_vals = np.array([
                _get_scenario_metrics(summaries, model, metric).get(s, float("nan"))
                for s in scenarios
            ])
            # log10 ratio; guard against zero
            with np.errstate(divide="ignore", invalid="ignore"):
                log_ratio = np.log10(np.where(pi_vals > 1e-15, snn_vals / pi_vals, np.nan))

            style = _get_style(model)
            offset = (i - (len(snn_models) - 1) / 2) * width
            ax.bar(x + offset, log_ratio, width, color=style["color"],
                   alpha=0.85, label=style["label"], zorder=3)

        ax.axhline(0, color="black", lw=0.8, ls="--", zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7)
        ax.set_ylabel(r"$\log_{10}$(MAE$_{SNN}$ / MAE$_{PI}$)", fontsize=9)
        ax.set_title(ax_title, fontsize=10)
        ax.grid(alpha=0.3, axis="y", zorder=0)

        # Annotate reference lines
        for exp, label in [(1, "10×"), (2, "100×")]:
            ax.axhline(exp, color="#aaa", lw=0.7, ls=":", zorder=2)
            ax.text(len(scenarios) - 0.05, exp + 0.04, label,
                    fontsize=6.5, ha="right", va="bottom", color="#888")

    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(r"Relative MAE vs PI Baseline — $\log_{10}$(SNN / PI)",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()

    out = plots_dir / "p3_V2_log_ratio_mae.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ──────────────────────────────────────────────────────────────────────
# Plot V3: Radar / spider chart — normalized multi-metric summary
# ──────────────────────────────────────────────────────────────────────

def plot_radar_summary(results_dir: Path, plots_dir: Path) -> None:
    """Plot V3 — Radar chart: normalized multi-metric summary per controller.

    Metrics are aggregated (mean across scenarios) and scaled so that
    the PI baseline sits near 1.0 on each axis. SNN performance is shown
    relative to PI. Lower = better (smaller polygon = worse; outer = best).

    Because PI is near-perfect and SNNs are ~100-200× worse on error metrics,
    we invert and cap each axis so the chart stays readable:
      score = 1 / (1 + value / pi_value)  →  PI ≈ 0.5, SNN closer to 0.
    Alternatively we show normalized absolute scores bounded [0, 1].
    """
    summ_path = results_dir / "phase3_summaries.json"
    if not summ_path.exists():
        print("  [skip] phase3_summaries.json not found for Plot V3")
        return

    summaries = _load_json(summ_path)
    pi_key = "PI-baseline"
    if pi_key not in summaries:
        print("  [skip] PI-baseline missing for Plot V3")
        return

    models_to_plot = [m for m in MODEL_ORDER if m in summaries]

    # Metric definitions: (key, display_label, normalize_by, lower_is_better)
    # normalize_by: float to divide raw value before scoring
    metric_defs = [
        ("mae_i_q",       r"MAE $i_q$",       _I_MAX,  True),
        ("mae_i_d",       r"MAE $i_d$",        _I_MAX,  True),
        ("rms_i_q",       r"RMS $i_q$",        _I_MAX,  True),
        ("max_error_i_q", r"Max error $i_q$",  _I_MAX,  True),
        ("itae_i_q",      r"ITAE $i_q$",       None,    True),   # log-normalized
        ("mean_sparsity", r"Sparsity",          1.0,     False),  # higher = better for SNN
    ]

    n_metrics = len(metric_defs)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    labels = [d[1] for d in metric_defs]

    # Collect mean values across scenarios for each model and metric
    def _mean_metric(model: str, key: str) -> float:
        vals = list(_get_scenario_metrics(summaries, model, key).values())
        finite = [v for v in vals if v is not None and not (np.isnan(v) or np.isinf(v))]
        return float(np.mean(finite)) if finite else float("nan")

    # Compute scores: map each metric to [0, 1] where 1 = best possible
    # Strategy: for lower-is-better metrics, score = exp(-value / normalizer)
    # For sparsity (higher-is-better), score = value directly
    all_scores: dict[str, list[float]] = {}
    for model in models_to_plot:
        scores = []
        for key, _, norm, lower_better in metric_defs:
            val = _mean_metric(model, key)
            if np.isnan(val):
                scores.append(0.0)
                continue
            if lower_better:
                if norm is None:
                    # ITAE: log-normalize relative to PI
                    pi_val = _mean_metric(pi_key, key)
                    if pi_val > 1e-20 and not np.isnan(pi_val):
                        # score = pi_val / val (capped at 1.0)
                        score = min(1.0, pi_val / val)
                    else:
                        score = 0.0
                else:
                    # Normalize to physical max, then invert: 0A error → score 1.0
                    score = max(0.0, 1.0 - val / norm)
            else:
                # sparsity: already in [0,1], higher is better
                score = float(np.clip(val, 0.0, 1.0))
            scores.append(score)
        all_scores[model] = scores

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    for model in models_to_plot:
        style = _get_style(model)
        scores = all_scores[model] + all_scores[model][:1]
        ax.plot(angles, scores, color=style["color"], lw=2,
                marker=style["marker"], markersize=6, label=style["label"])
        ax.fill(angles, scores, color=style["color"], alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="#888")
    ax.yaxis.set_tick_params(labelsize=7)
    ax.grid(color="#ccc", linestyle="--", linewidth=0.6, alpha=0.7)

    ax.legend(
        fontsize=9,
        loc="upper right",
        bbox_to_anchor=(1.35, 1.15),
        frameon=True,
        framealpha=0.95,
        edgecolor="#ccc",
    )
    ax.set_title("Normalized Multi-Metric Summary\n(outer = better)",
                 fontweight="bold", fontsize=11, pad=20)

    plt.tight_layout()
    out = plots_dir / "p3_V3_radar_summary.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ──────────────────────────────────────────────────────────────────────
# Plot V4: Neuromorphic metrics — SNN vs SNN radar (SyOps, sparsity, spikes)
# ──────────────────────────────────────────────────────────────────────

def plot_neuromorphic_radar_snn_vs_snn(results_dir: Path, plots_dir: Path) -> None:
    """Plot V4 — Radar chart of neuromorphic metrics, SNN vs SNN only.

    Compares SNN models on: total_syops (lower better), mean_sparsity (higher better),
    min_sparsity (higher better), total_spikes (lower better). All axes normalized to
    [0, 1] with outer = better. PI is excluded since it has no neuromorphic metrics.
    Tries fallback keys (e.g. syops, sparsity) when harness uses different names.
    """
    summ_path = results_dir / "phase3_summaries.json"
    if not summ_path.exists():
        print("  [skip] phase3_summaries.json not found for Plot V4 (neuromorphic radar)")
        return

    summaries = _load_json(summ_path)
    snn_models = [m for m in SNN_MODELS if m in summaries]
    if len(snn_models) < 2:
        print("  [skip] Need at least 2 SNN models for neuromorphic SNN vs SNN plot")
        return

    # Primary key and optional fallbacks (harness may use "syops" vs "total_syops", etc.)
    neuro_defs = [
        (["total_syops", "syops"], r"SyOps (eff.)", True),
        (["mean_sparsity", "sparsity"], r"Sparsity (mean)", False),
        (["min_sparsity"], r"Sparsity (min)", False),
        (["total_spikes", "spikes"], r"Spikes (eff.)", True),
    ]

    def _mean_metric_with_fallbacks(model: str, keys: list[str]) -> float:
        for key in keys:
            vals = list(_get_scenario_metrics(summaries, model, key).values())
            finite = [v for v in vals if v is not None and not (np.isnan(v) or np.isinf(v))]
            if finite:
                return float(np.mean(finite))
        return float("nan")

    # Collect raw values per model (one value per axis)
    raw: dict[str, list[float]] = {m: [] for m in snn_models}
    for model in snn_models:
        for keys, _, _ in neuro_defs:
            raw[model].append(_mean_metric_with_fallbacks(model, keys))

    # Normalize to scores in [0, 1], outer = better
    n_metrics = len(neuro_defs)
    scores: dict[str, list[float]] = {}
    for model in snn_models:
        scores[model] = []
    for j, (_, _, lower_better) in enumerate(neuro_defs):
        vals = [raw[m][j] for m in snn_models]
        valid = [v for v in vals if not (v is None or np.isnan(v) or np.isinf(v))]
        if not valid:
            for m in snn_models:
                scores[m].append(0.0)
            continue
        v_min, v_max = min(valid), max(valid)
        if v_max <= v_min:
            for m in snn_models:
                scores[m].append(1.0)
            continue
        for i, model in enumerate(snn_models):
            v = raw[model][j]
            if v is None or np.isnan(v) or np.isinf(v):
                scores[model].append(0.0)
            elif lower_better:
                # lower = better → score = 1 at min, 0 at max
                score = (v_max - v) / (v_max - v_min)
                scores[model].append(float(np.clip(score, 0.0, 1.0)))
            else:
                # higher = better (e.g. sparsity)
                score = (v - v_min) / (v_max - v_min)
                scores[model].append(float(np.clip(score, 0.0, 1.0)))

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]
    labels = [d[1] for d in neuro_defs]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    for model in snn_models:
        style = _get_style(model)
        s = scores[model] + scores[model][:1]
        ax.plot(angles, s, color=style["color"], lw=2,
                marker=style["marker"], markersize=8, label=style["label"])
        ax.fill(angles, s, color=style["color"], alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="#888")
    ax.yaxis.set_tick_params(labelsize=7)
    ax.grid(color="#ccc", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.legend(
        fontsize=9,
        loc="upper right",
        bbox_to_anchor=(1.32, 1.12),
        frameon=True,
        framealpha=0.95,
        edgecolor="#ccc",
    )
    ax.set_title("Neuromorphic Metrics — SNN vs SNN\n(outer = better)",
                 fontweight="bold", fontsize=11, pad=20)
    plt.tight_layout()
    out = plots_dir / "p3_V4_neuromorphic_radar_snn_vs_snn.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def generate_phase3_plots(results_dir: Path, plots_dir: Path) -> None:
    """Entry point: generate all Phase 3 plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("Phase 3 plots:")
    plot_envelope_comparison(results_dir, plots_dir)
    plot_relative_error_vs_pi(results_dir, plots_dir)
    plot_mae_grouped_bar(results_dir, plots_dir)
    plot_syops_vs_mae_pareto(results_dir, plots_dir)
    plot_v12_vs_pi_normalized(results_dir, plots_dir)
    plot_log_ratio_mae(results_dir, plots_dir)
    plot_radar_summary(results_dir, plots_dir)
    plot_neuromorphic_radar_snn_vs_snn(results_dir, plots_dir)
