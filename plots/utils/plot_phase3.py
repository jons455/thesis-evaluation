"""
Phase 3 Plots — Discriminative Power.

Plot 3.1: Envelope comparison — each SNN vs PI baseline (1 figure per SNN).
Plot 3.2: MAE grouped bar chart, log scale — PI + all 3 SNNs.
Plot 3.4: SyOps vs MAE Pareto scatter — PI + all 3 SNNs.
Plot V1:  All models — all key metrics, normalized.
Plot V2:  Log-ratio — log10(MAE / MAE_PI) — all 3 SNNs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Professional, colorblind-friendly palette; neutral labels (no quality interpretation)
MODEL_STYLES = {
    "PI-baseline":                     {"color": "#2d2d2d", "marker": "s", "label": "PI-baseline"},
    "best_incremental_snn":            {"color": "#4477aa", "marker": "o", "label": "best_incremental_snn"},
    "intermediate_scheduled_sampling": {"color": "#228833", "marker": "^", "label": "intermediate_scheduled_sampling"},
    "poor_no_tanh":                    {"color": "#cc6677", "marker": "D", "label": "poor_no_tanh"},
}

# To include/exclude models: edit MODEL_ORDER (and optionally MODEL_STYLES for new models).
# Only models in this list are plotted; comment out or remove a name to drop it from all Phase 3 plots.
MODEL_ORDER = ["PI-baseline", "best_incremental_snn", "intermediate_scheduled_sampling", "poor_no_tanh"]
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
    """Plot 3.1 — Envelope plot: each SNN vs PI baseline, one figure per SNN.

    One column per scenario (ordered easy→hard).
    Top row: i_q with reference step signal and fill_between error band.
    Bottom row: u_q voltage command comparison.
    Produces one figure per SNN model in SNN_MODELS.
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

    # Logical scenario order: simple steps (easy→hard speed) then complex manoeuvres
    _SCENARIO_ORDER = [
        "step_low_speed_500rpm_2A",
        "step_mid_speed_1500rpm_2A",
        "step_high_speed_2500rpm_2A",
        "multi_step_bidirectional_1500rpm",
        "four_quadrant_transition_1500rpm",
        "field_weakening_2500rpm",
    ]
    scenario_names = [s for s in _SCENARIO_ORDER if s in scenarios_data]
    # Append any scenarios not in the predefined order (future-proof)
    for s in sorted(scenarios_data.keys()):
        if s not in scenario_names:
            scenario_names.append(s)
    n_scen = len(scenario_names)

    # One figure per SNN model — each shows that SNN vs PI across all scenarios
    for snn_model in SNN_MODELS:
        snn_style = _get_style(snn_model)
        pi_style = _get_style("PI-baseline")

        fig, axes = plt.subplots(2, n_scen, figsize=(4.0 * n_scen, 5.5), sharex=False, squeeze=False)
        fig.subplots_adjust(hspace=0.35, wspace=0.3)

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

            # Reference step signal
            ax_iq.plot(t_ms, ref_iq, "k--", lw=1.2, alpha=0.75, label=r"Reference $i_q^*$")

            # PI baseline
            ax_iq.plot(t_ms, pi_iq, color=pi_style["color"], lw=1.2, alpha=0.85, label=pi_style["label"])

            # SNN model trajectory
            ax_iq.plot(t_ms, snn_iq, color=snn_style["color"], lw=1.0, alpha=0.9, label=snn_style["label"])

            # Shaded band: highlights steady-state offset and oscillatory switching
            ax_iq.fill_between(
                t_ms,
                np.minimum(pi_iq, snn_iq),
                np.maximum(pi_iq, snn_iq),
                alpha=0.18, color=snn_style["color"],
            )

            # Voltage command comparison (u_q)
            ax_uq.plot(t_ms, pi_uq, color=pi_style["color"], lw=0.9, alpha=0.85, label=pi_style["label"])
            ax_uq.plot(t_ms, snn_uq, color=snn_style["color"], lw=0.9, alpha=0.9, label=snn_style["label"])
            ax_uq.fill_between(
                t_ms,
                np.minimum(pi_uq, snn_uq),
                np.maximum(pi_uq, snn_uq),
                alpha=0.18, color=snn_style["color"],
            )

            col_title = _SCENARIO_SHORT.get(scenario, scenario.replace("_", " "))
            ax_iq.set_title(col_title.replace("\n", " "), fontsize=8, fontweight="bold")
            ax_iq.grid(alpha=0.3)
            ax_uq.grid(alpha=0.3)
            ax_uq.set_xlabel("Time (ms)", fontsize=8)
            ax_iq.tick_params(labelsize=7)
            ax_uq.tick_params(labelsize=7)

        if not has_data:
            plt.close(fig)
            continue

        axes[0, 0].set_ylabel("Current $i_q$ (A)", fontsize=9)
        axes[1, 0].set_ylabel("Voltage command $u_q$ (V)", fontsize=9)
        axes[0, 0].legend(fontsize=7, loc="best", framealpha=0.85)

        fig.suptitle(
            f"Qualitative Trajectory Overlay — Reference, PI-baseline, and {snn_style['label']}",
            fontweight="bold",
            fontsize=10,
        )

        safe_name = snn_model.replace(" ", "_")
        out = plots_dir / f"p3_1_envelope_{safe_name}.png"
        fig.savefig(out, bbox_inches="tight", dpi=300)
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
    fig.savefig(out, bbox_inches="tight", dpi=300)
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
    fig.savefig(out, bbox_inches="tight", dpi=300)
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
    """Plot V1 — All models: all key metrics, normalized where possible.

    Produces a 2×3 multi-panel figure with one subplot per metric:
      - MAE i_q  [% of i_max]          (normalized)
      - MAE i_d  [% of i_max]          (normalized)
      - RMS i_q  [% of i_max]          (normalized)
      - ITAE i_q [raw, log scale]
      - Max error i_q [% of i_max]     (normalized)
      - Settling time [ms]             (Inf shown as hatched bar at ceiling)

    Each subplot shows all models in MODEL_ORDER side-by-side per scenario.
    """
    summ_path = results_dir / "phase3_summaries.json"
    if not summ_path.exists():
        print("  [skip] phase3_summaries.json not found for Plot V1")
        return

    summaries = _load_json(summ_path)

    pi_key = "PI-baseline"
    if pi_key not in summaries:
        print("  [skip] PI-baseline missing from summaries for Plot V1")
        return

    # Only include models that are actually in the summaries
    models = [m for m in MODEL_ORDER if m in summaries]
    if len(models) < 2:
        print("  [skip] Need at least 2 models for Plot V1")
        return

    # Ordered scenario names from PI data (robust to both "name" and "scenario_name" keys)
    pi_data = summaries[pi_key]
    scenario_list = pi_data.get("scenarios") or pi_data.get("scenario_results") or []
    scenarios = [sr.get("name") or sr.get("scenario_name") for sr in scenario_list]
    scenarios = [s for s in scenarios if s]

    x_labels = [_SCENARIO_SHORT.get(s, s) for s in scenarios]
    x = np.arange(len(scenarios))

    # Bar width: distribute evenly, keep total group width ≤ 0.8
    n_models = len(models)
    total_group_width = 0.76
    w = total_group_width / n_models

    # ── helper ───────────────────────────────────────────────────────
    def _get(model: str, metric: str) -> np.ndarray:
        vals = _get_scenario_metrics(summaries, model, metric)
        return np.array([vals.get(s, float("nan")) for s in scenarios])

    # Settling time: replace Inf with capped value; track which are Inf
    _SETTLE_CAP_MS = 500.0
    settle_plot: dict[str, np.ndarray] = {}
    settle_inf:  dict[str, np.ndarray] = {}
    for m in models:
        raw = _get(m, "settling_time_i_q")
        settle_inf[m]  = np.isinf(raw)
        settle_plot[m] = np.where(settle_inf[m], _SETTLE_CAP_MS,
                                  np.where(np.isnan(raw), np.nan, raw * 1000))

    # ── subplot definitions ─────────────────────────────────────────
    # (ax_idx, metric_key_or_None, ylabel, log_scale, note, normalize_to_pct)
    subplot_defs = [
        (0, "mae_i_q",       r"MAE $i_q$ [% of $i_{max}$]",          False, None,     True),
        (1, "mae_i_d",       r"MAE $i_d$ [% of $i_{max}$]",          False, None,     True),
        (2, "rms_i_q",       r"RMS $i_q$ [% of $i_{max}$]",          False, None,     True),
        (3, "itae_i_q",      r"ITAE $i_q$ [A·s²]",                   True,  None,     False),
        (4, "max_error_i_q", r"Max error $i_q$ [% of $i_{max}$]",    False, None,     True),
        (5, None,            "Settling time [ms]",                    False, "settle", False),
    ]

    # ── figure layout: 2 rows × 3 cols ──────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    axes = axes.flatten()

    for ax_idx, metric_key, ylabel, log_scale, note, normalize in subplot_defs:
        ax = axes[ax_idx]

        for i, model in enumerate(models):
            style  = _get_style(model)
            offset = (i - (n_models - 1) / 2) * w

            if note == "settle":
                vals = settle_plot[model]
            elif normalize:
                vals = _get(model, metric_key) / _I_MAX * 100
            else:
                vals = _get(model, metric_key)

            bars = ax.bar(x + offset, vals, w, color=style["color"],
                          alpha=0.85, label=style["label"], zorder=3)

            # Hatching for bars where settling time did not converge (Inf)
            if note == "settle":
                for j, is_inf in enumerate(settle_inf[model]):
                    if is_inf:
                        bars[j].set_hatch("///")
                        bars[j].set_edgecolor("white")

        if note == "settle":
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
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("All Models — Normalized Performance Metrics (PI + 3 SNN variants)",
                 fontweight="bold", fontsize=11, y=1.01)
    plt.tight_layout()

    out = plots_dir / "p3_V1_v12_vs_pi_normalized.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
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
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out.name}")



def generate_phase3_plots(results_dir: Path, plots_dir: Path) -> None:
    """Entry point: generate all Phase 3 plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("Phase 3 plots:")
    plot_envelope_comparison(results_dir, plots_dir)
    plot_mae_grouped_bar(results_dir, plots_dir)
    plot_syops_vs_mae_pareto(results_dir, plots_dir)
    plot_v12_vs_pi_normalized(results_dir, plots_dir)
    plot_log_ratio_mae(results_dir, plots_dir)
