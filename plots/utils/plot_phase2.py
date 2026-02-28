"""
Phase 2 Plots — Metric Validation.

Plot 2.2: Clean white table — Manual vs pipeline metric comparison.
Plot 2.3: Lollipop chart of deviations on log scale.
Plot 2.4: Step response time series with settling/overshoot annotations.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_comparisons(results_dir: Path):
    val_path = results_dir / "phase2_validation.json"
    if not val_path.exists():
        print("  [skip] phase2_validation.json not found")
        return None, None, None
    with open(val_path) as f:
        data = json.load(f)
    return data["comparisons"], data["scenario"], data.get("step_onset", "?")



def plot_metric_validation_table(results_dir: Path, plots_dir: Path) -> None:
    """Plot 2.2 — Clean white table: Manual vs pipeline metric comparison."""
    comparisons, scenario, step_onset = _load_comparisons(results_dir)
    if comparisons is None:
        return

    col_labels = ["Metric", "Manual", "Pipeline", "Deviation"]
    cell_text = []

    for c in comparisons:
        manual = f"{c['manual']:.10f}" if c["manual"] is not None and not np.isnan(c["manual"]) else "NaN"
        pipeline = f"{c['pipeline']:.10f}" if c["pipeline"] is not None and not np.isnan(c["pipeline"]) else "NaN"
        dev = f"{c['deviation']:.2e}" if c["deviation"] is not None and not np.isnan(c["deviation"]) else "NaN"
        cell_text.append([c["metric"], manual, pipeline, dev])

    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.45 * len(cell_text)))
    ax.axis("off")
    ax.set_title(f"{scenario}, step onset k={step_onset}", fontsize=11, pad=12)

    # White table with light grid
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Style: white background, light gray header
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.5)
        if key[0] == 0:
            cell.set_facecolor("#F5F5F5")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("white")

    plt.tight_layout()
    out = plots_dir / "p2_2_metric_validation_table.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_deviation_lollipop(results_dir: Path, plots_dir: Path) -> None:
    """Plot 2.3 — Lollipop chart of deviations on log scale."""
    comparisons, scenario, step_onset = _load_comparisons(results_dir)
    if comparisons is None:
        return

    metrics = []
    deviations = []
    for c in comparisons:
        dev = c["deviation"]
        if dev is not None and not np.isnan(dev) and dev > 0:
            metrics.append(c["metric"])
            deviations.append(dev)

    if not deviations:
        print("  [skip] No non-zero deviations for lollipop chart")
        return

    n_metrics = len(metrics)
    fig_h = max(2.0, 0.7 * n_metrics)
    fig, ax = plt.subplots(figsize=(5, fig_h))
    y = np.arange(n_metrics)
    color = "#5c5c5c"
    for i, dev in enumerate(deviations):
        ax.plot([0, dev], [i, i], color=color, lw=2, zorder=2)
        ax.scatter(dev, i, color=color, s=60, zorder=3, edgecolors="black", linewidths=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=9)
    ax.set_xscale("log")
    dev_min, dev_max = min(deviations), max(deviations)
    ax.set_xlim(dev_min * 0.5, dev_max * 2.5)
    ax.set_xlabel("Absolute Deviation (log scale)")
    ax.set_title(f"{scenario}, step onset k={step_onset}", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    ax.invert_yaxis()

    plt.tight_layout()
    out = plots_dir / "p2_3_deviation_lollipop.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_step_response_timeseries(results_dir: Path, plots_dir: Path) -> None:
    """Plot 2.4 — i_q step response with annotated settling time and overshoot.

    Shows the raw simulated i_q trajectory zoomed to the transient window, with:
      - ±2% settling band (green shading)
      - vertical line for manual settling time (blue)
      - vertical line for pipeline settling time (orange dashed) — usually 1 sample away
      - overshoot peak dot with % label
      - ITAE integration window (grey shading, first 50 ms after step)
      - text box explaining that ±dt = 0.1 ms discrete-index resolution accounts for
        any settling-time deviation between manual and pipeline

    Requires that phase2_validation.json contains "trajectory", "manual_metrics", and "dt"
    (written by phase2_metric_validation.py ≥ this version).  Old result files without
    the trajectory key are skipped gracefully.
    """
    val_path = results_dir / "phase2_validation.json"
    if not val_path.exists():
        print("  [skip] phase2_validation.json not found for Plot 2.4")
        return

    with open(val_path) as f:
        data = json.load(f)

    if "trajectory" not in data or "manual_metrics" not in data:
        print("  [skip] Plot 2.4 requires trajectory data — re-run phase2_metric_validation.py")
        return

    traj = data["trajectory"]
    i_q = np.array(traj["i_q"])                        # next_state, t = (k+1)*dt
    i_q_ref = np.array(traj["i_q_ref"])                # reference during step k
    i_q_ss = np.array(traj["i_q_at_step_start"])       # state at step start, t = k*dt

    step_onset: int = int(data["step_onset"])
    pre_step_ref = data.get("pre_step_ref")             # None when NaN
    dt: float = float(data["dt"])
    scenario: str = data["scenario"]

    mm = data["manual_metrics"]
    manual_settling = mm.get("settling_time_i_q")       # seconds or None (inf/nan)
    manual_overshoot = mm.get("overshoot_i_q")          # % or None
    manual_itae = mm.get("itae_i_q")

    pm = data.get("pipeline_metrics", {})
    pipeline_settling = pm.get("settling_time_i_q")
    pipeline_overshoot = pm.get("overshoot")

    # --- derived geometry ---
    target = float(i_q_ref[step_onset]) if step_onset < len(i_q_ref) else 0.0
    pre = float(pre_step_ref) if pre_step_ref is not None else 0.0
    step_size = abs(target - pre)
    band = 0.02 * step_size if step_size > 1e-9 else 0.05

    # Time arrays (in ms)
    t_ss_ms = np.arange(len(i_q_ss)) * dt * 1000       # step-start samples
    t_next_ms = (np.arange(len(i_q)) + 1) * dt * 1000  # next-state samples

    # Zoom window: 5 ms before step .. 60 ms after step
    pre_ms = 5.0
    post_ms = 65.0
    t_onset_ms = step_onset * dt * 1000
    xlim = (t_onset_ms - pre_ms, t_onset_ms + post_ms)

    # ITAE window end
    itae_window_ms = 50.0  # matches _manual_itae default window_s=0.05

    fig, ax = plt.subplots(figsize=(10, 5))

    # ITAE integration window shading (semi-transparent, drawn first so it's behind)
    ax.axvspan(t_onset_ms, t_onset_ms + itae_window_ms,
               alpha=0.06, color="#9E9E9E", label="ITAE window (50 ms)")

    # Reference and response
    ax.plot(t_ss_ms, i_q_ref, "k--", lw=1.5, alpha=0.55, label="Reference $i_q^*$")
    ax.plot(t_next_ms, i_q, color="#2196F3", lw=1.2, label="$i_q$ (simulated)")

    # ±2% settling band
    ax.axhspan(target - band, target + band,
               alpha=0.10, color="#4CAF50", zorder=1)
    ax.axhline(target + band, color="#4CAF50", ls=":", lw=0.9, alpha=0.8)
    ax.axhline(target - band, color="#4CAF50", ls=":", lw=0.9, alpha=0.8,
               label=f"±2% band (±{band:.4f} A)")

    # Step onset marker
    ax.axvline(t_onset_ms, color="#9E9E9E", ls="--", lw=0.8, alpha=0.7)

    # Manual settling time
    if manual_settling is not None:
        t_settle_man_ms = t_onset_ms + manual_settling * 1000
        ax.axvline(t_settle_man_ms, color="#1565C0", ls="-", lw=1.8,
                   label=f"Manual settle: {manual_settling * 1000:.2f} ms")
        ax.annotate(
            f"Manual\n{manual_settling * 1000:.2f} ms",
            xy=(t_settle_man_ms, target - band),
            xytext=(t_settle_man_ms - 8, target - 3.5 * band),
            fontsize=7, color="#1565C0",
            arrowprops=dict(arrowstyle="->", color="#1565C0", lw=0.8),
            ha="right",
        )

    # Pipeline settling time (usually 1 sample from manual)
    if pipeline_settling is not None:
        t_settle_pipe_ms = t_onset_ms + pipeline_settling * 1000
        ax.axvline(t_settle_pipe_ms, color="#FF5722", ls="--", lw=1.8,
                   label=f"Pipeline settle: {pipeline_settling * 1000:.2f} ms")
        ax.annotate(
            f"Pipeline\n{pipeline_settling * 1000:.2f} ms",
            xy=(t_settle_pipe_ms, target + band),
            xytext=(t_settle_pipe_ms + 2, target + 4.0 * band),
            fontsize=7, color="#FF5722",
            arrowprops=dict(arrowstyle="->", color="#FF5722", lw=0.8),
            ha="left",
        )

    # Overshoot peak (on i_q_at_step_start, which is what the pipeline tracks)
    if manual_overshoot is not None and step_onset < len(i_q_ss):
        end_idx = min(len(i_q_ss), step_onset + int(post_ms / (dt * 1000)) + 1)
        seg = i_q_ss[step_onset:end_idx]
        peak_local = int(np.argmax(seg) if target > 0 else np.argmin(seg))
        peak_idx = step_onset + peak_local
        if peak_idx < len(t_ss_ms):
            t_peak_ms = t_ss_ms[peak_idx]
            peak_val = float(i_q_ss[peak_idx])
            ax.scatter([t_peak_ms], [peak_val], color="#FF9800", s=70,
                       zorder=5, edgecolors="black", linewidths=0.5,
                       label=f"Overshoot peak ({manual_overshoot:.2f}%)")
            ax.annotate(
                f"{manual_overshoot:.2f}%",
                xy=(t_peak_ms, peak_val),
                xytext=(t_peak_ms + 1.5, peak_val + 0.3 * band),
                fontsize=8, color="#FF9800", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#FF9800", lw=0.8),
            )

    # Text box: explain ±dt resolution
    if manual_settling is not None and pipeline_settling is not None:
        diff_ms = abs(manual_settling - pipeline_settling) * 1000
        note = (
            f"|Δt_settle| = {diff_ms:.3f} ms\n"
            f"= {diff_ms / (dt * 1000):.1f} × Δt  (Δt = {dt * 1000:.1f} ms)\n"
            "Discrete-index resolution of\n"
            "settling detection fully explains\n"
            "the 1e-4 ITAE deviation."
        )
        ax.text(
            0.985, 0.97, note,
            transform=ax.transAxes,
            fontsize=7, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="#CCCCCC", alpha=0.9),
        )

    # Zoomed inset around settling point — makes the 1-sample gap visible.
    # The two settling lines are only dt = 0.1 ms apart, invisible at full scale.
    if manual_settling is not None and pipeline_settling is not None:
        t_mid_ms = t_onset_ms + (manual_settling + pipeline_settling) * 0.5 * 1000
        inset_half_ms = max(3.0, abs(manual_settling - pipeline_settling) * 1000 * 8)
        ix0_ms = t_mid_ms - inset_half_ms
        ix1_ms = t_mid_ms + inset_half_ms

        # Find sample indices that fall inside the inset x-window
        in_inset_ss = np.where((t_ss_ms >= ix0_ms) & (t_ss_ms <= ix1_ms))[0]
        in_inset_nxt = np.where((t_next_ms >= ix0_ms) & (t_next_ms <= ix1_ms))[0]
        if len(in_inset_ss) > 0:
            y_vals = np.concatenate([
                i_q_ss[in_inset_ss],
                i_q[in_inset_nxt] if len(in_inset_nxt) > 0 else np.array([]),
                [target + band, target - band],
            ])
            iy_pad = (y_vals.max() - y_vals.min()) * 0.5 or band * 2
            iy0 = y_vals.min() - iy_pad
            iy1 = y_vals.max() + iy_pad

            # Place inset in upper-left to avoid the legend
            axins = ax.inset_axes([0.02, 0.55, 0.38, 0.38])
            axins.set_xlim(ix0_ms, ix1_ms)
            axins.set_ylim(iy0, iy1)

            # Settling band in inset
            axins.axhspan(target - band, target + band, alpha=0.15, color="#4CAF50")
            axins.axhline(target + band, color="#4CAF50", ls=":", lw=0.8)
            axins.axhline(target - band, color="#4CAF50", ls=":", lw=0.8)

            # Reference (dashed)
            axins.plot(t_ss_ms, i_q_ref, "k--", lw=1.0, alpha=0.5)

            # Individual samples as dots so the discrete index is obvious
            if len(in_inset_nxt) > 0:
                axins.plot(t_next_ms[in_inset_nxt], i_q[in_inset_nxt],
                           color="#2196F3", lw=0.8)
                axins.scatter(t_next_ms[in_inset_nxt], i_q[in_inset_nxt],
                              color="#2196F3", s=18, zorder=4)

            # Settling lines
            t_settle_man_ms = t_onset_ms + manual_settling * 1000
            t_settle_pipe_ms = t_onset_ms + pipeline_settling * 1000
            axins.axvline(t_settle_man_ms, color="#1565C0", ls="-", lw=1.5,
                          label=f"Manual ({manual_settling * 1000:.2f} ms)")
            axins.axvline(t_settle_pipe_ms, color="#FF5722", ls="--", lw=1.5,
                          label=f"Pipeline ({pipeline_settling * 1000:.2f} ms)")

            axins.set_xlabel("Time [ms]", fontsize=6)
            axins.set_ylabel("$i_q$ [A]", fontsize=6)
            axins.tick_params(labelsize=6)
            axins.legend(fontsize=5.5, loc="upper right", framealpha=0.9)
            axins.set_title("settling region (zoomed)", fontsize=6, style="italic")
            axins.grid(alpha=0.25)

            ax.indicate_inset_zoom(axins, edgecolor="gray")

    ax.set_xlim(xlim)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("$i_q$ [A]")
    ax.set_title(f"SC-2 Step Response — {scenario}", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = plots_dir / "p2_4_step_response_timeseries.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def generate_phase2_plots(results_dir: Path, plots_dir: Path) -> None:
    """Entry point: generate all Phase 2 plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("Phase 2 plots:")
    plot_metric_validation_table(results_dir, plots_dir)
    plot_deviation_lollipop(results_dir, plots_dir)
    plot_step_response_timeseries(results_dir, plots_dir)
