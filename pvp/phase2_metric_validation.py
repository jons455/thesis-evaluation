"""
PVP Phase 2 — Metric Validation (SC-2).

Part A — Control Metrics (PI controller):
Uses R2 data (PI via wrapper) on `step_mid_speed_1500rpm_2A`, float64.
Computes MAE, ITAE, Settling Time, Overshoot manually from the trajectory
using NumPy, and compares against the pipeline accumulator values.

Part B — Neuromorphic Metrics (SNN controller):
Runs best_incremental_snn on the same scenario.  Manually accumulates
total_spikes, syops, and activation_sparsity from raw per-step
controller_info dicts using NumPy, and compares against the pipeline
accumulator values (SpikeCount, SynapticOps, ActivationSparsity).

MAE alignment: the pipeline (TrackingMAE) receives update(state, reference, ...)
and uses (reference, state) at each step — i.e. ref_t vs state at step start.
The manual MAE is therefore computed as mean(|i_q_ref - i_q_at_step_start|)
so it matches the pipeline; using next_state caused ~6.7e-4 deviation.

This script is pure data collection — it records deviations between manual
and pipeline implementations without issuing any PASS/FAIL verdicts.
All interpretation is done in interpret_results.py.

Step onset: first sample k where |i_q_ref[k] - i_q_ref[k-1]| > 0.01 A.

Steady-state RMS: RMS of tracking error over the window from t >= steady_state_start_s
(default 0.05 s) to end of episode, i.e. sqrt(mean((i - i_ref)^2)) in that window.
Validates that the pipeline's rms_i_q / rms_i_d (if defined as steady-state) are correct.

Settling and overshoot are defined for the axis that has the step (i_q for step_*_2A
scenarios where i_d_ref = 0). For scenarios with i_d_ref steps, the same formulas
apply to i_d; the pipeline currently reports settling_time_i_q and overshoot (i_q) only.

Usage:
    poetry run python embark-evaluation/pvp/phase2_metric_validation.py
    poetry run python embark-evaluation/pvp/phase2_metric_validation.py --run pvp_run1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_repo_root = Path(__file__).resolve().parents[2]
_embark_eval_dir = Path(__file__).resolve().parents[1]
for _p in [str(_repo_root), str(_embark_eval_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pvp.utils.common import (
    MODELS,
    build_snn_controller,
    ensure_results_dir,
    save_json,
    save_text_report,
    setup_deterministic,
)


def _find_step_onset(i_q_ref: np.ndarray, threshold: float = 0.01) -> int:
    """Find the first index where the reference changes by more than threshold."""
    for k in range(1, len(i_q_ref)):
        if abs(i_q_ref[k] - i_q_ref[k - 1]) > threshold:
            return k
    return 0


def _get_step_onset_and_pre_ref(
    i_q_ref: np.ndarray, threshold: float = 0.01
) -> tuple[int, float]:
    """
    Return (step_onset_index, pre_step_ref_value) for settling/overshoot.

    If the reference never changes in the trajectory (step at t=0 before first sample),
    step_onset is 0 and we assume pre_step_ref = 0 so step_size = |i_q_ref[0]|.
    """
    step_onset = _find_step_onset(i_q_ref, threshold)
    if step_onset == 0:
        if len(i_q_ref) > 0 and abs(i_q_ref[0]) > threshold:
            return 0, 0.0  # step at start: assume 0 -> ref[0]
        return 0, float("nan")  # no step detected
    return step_onset, float(i_q_ref[step_onset - 1])


def _manual_mae(i_q: np.ndarray, i_q_ref: np.ndarray) -> float:
    """Full-episode MAE: mean(|i_q_ref - i_q|). Use i_q aligned with pipeline (see below)."""
    return float(np.mean(np.abs(i_q_ref - i_q)))


def _manual_itae(
    i_q: np.ndarray, i_q_ref: np.ndarray, dt: float, step_onset: int, window_s: float = 0.05
) -> float:
    """
    ITAE from step onset over transient window.

    tau[k] = (k - step_onset) * dt, integrated only over [step_onset, step_onset + window_steps].
    """
    window_steps = int(window_s / dt)
    end = min(step_onset + window_steps, len(i_q))
    if step_onset >= end:
        return 0.0

    total = 0.0
    for k in range(step_onset, end):
        tau = (k - step_onset) * dt
        total += tau * abs(i_q_ref[k] - i_q[k]) * dt
    return float(total)


def _manual_settling_time(
    i_q: np.ndarray,
    i_q_ref: np.ndarray,
    dt: float,
    step_onset: int,
    pre_step_ref: float,
    band_fraction: float = 0.02,
    dwell_s: float = 0.001,
) -> float:
    """
    Settling time: time from step onset to when error enters and stays in band.

    Band = band_fraction * |step_size|. Must stay in band for dwell_s.
    pre_step_ref: reference value before the step (use when step at index 0).
    """
    if step_onset >= len(i_q_ref) - 1:
        return float("inf")
    if np.isnan(pre_step_ref):
        return float("nan")

    target = i_q_ref[step_onset]
    step_size = abs(target - pre_step_ref)
    if step_size < 1e-12:
        return float("nan")

    band = band_fraction * step_size
    dwell_steps = max(1, int(dwell_s / dt))

    # Scan backward from end to find last violation
    last_violation = step_onset - 1
    for k in range(len(i_q) - 1, step_onset - 1, -1):
        if abs(i_q[k] - target) > band:
            last_violation = k
            break

    if last_violation < step_onset:
        # Settled immediately
        return 0.0

    settled_idx = last_violation + 1
    if settled_idx >= len(i_q):
        return float("inf")

    return float((settled_idx - step_onset) * dt)


def _manual_overshoot(
    i_q_at_step_start: np.ndarray,
    i_q_ref: np.ndarray,
    step_onset: int,
    pre_step_ref: float,
) -> float:
    """
    Overshoot (%) aligned with pipeline (embark.benchmark.metrics.accumulators.dynamics.Overshoot).

    Pipeline: latches first non-zero ref as step_ref; tracks max(meas) or min(meas) of *state*
    at each update (state at step start); overshoot = (peak - step_ref) / |step_ref| * 100
    (or (step_ref - trough) / |step_ref| * 100 for negative step). No smoothing.
    """
    if step_onset >= len(i_q_at_step_start):
        return 0.0
    if np.isnan(pre_step_ref):
        return float("nan")

    step_ref = i_q_ref[step_onset]
    if abs(step_ref) < 1e-12:
        return float("nan")

    segment = i_q_at_step_start[step_onset:]
    if step_ref > 0.0:
        peak = float(np.max(segment))
        return max(0.0, (peak - step_ref) / abs(step_ref) * 100.0)
    else:
        trough = float(np.min(segment))
        return max(0.0, (step_ref - trough) / abs(step_ref) * 100.0)


def _manual_rms_steady_state(
    i: np.ndarray,
    i_ref: np.ndarray,
    dt: float,
    start_s: float = 0.05,
) -> float:
    """
    Steady-state RMS of tracking error: sqrt(mean((i - i_ref)^2)) over samples from
    t >= start_s to end of episode. Matches documentation that RMS is "steady-state
    only (after 50 ms)".
    """
    start_k = max(0, int(start_s / dt))
    if start_k >= len(i):
        return float("nan")
    err = i[start_k:] - i_ref[start_k:]
    return float(np.sqrt(np.mean(err * err)))


# ---------------------------------------------------------------------------
# Neuromorphic manual computation (Part B)
# ---------------------------------------------------------------------------


def _manual_spike_count(per_step_spikes: list[float]) -> dict[str, float]:
    """Manual total spike count from raw per-step controller_info values."""
    total = sum(per_step_spikes)
    return {
        "total_spikes": float(total),
        "spikes_per_step": float(total) / len(per_step_spikes) if per_step_spikes else 0.0,
    }


def _manual_synaptic_ops(per_step_syops: list[float]) -> dict[str, float]:
    """Manual total synaptic operations from raw per-step controller_info values."""
    total = sum(per_step_syops)
    return {
        "total_syops": float(total),
        "syops_per_step": float(total) / len(per_step_syops) if per_step_syops else 0.0,
    }


def _manual_activation_sparsity(per_step_sparsity: list[float]) -> dict[str, float]:
    """Manual activation sparsity stats from raw per-step controller_info values."""
    if not per_step_sparsity:
        return {"mean_sparsity": 0.0, "min_sparsity": 0.0, "max_sparsity": 0.0}
    return {
        "mean_sparsity": sum(per_step_sparsity) / len(per_step_sparsity),
        "min_sparsity": min(per_step_sparsity),
        "max_sparsity": max(per_step_sparsity),
    }


def run_phase2(run_name: str | None = None, seed: int = 42) -> dict:
    """Execute Phase 2: manual NumPy metric validation against pipeline.

    Pure data collection — no PASS/FAIL verdicts issued here.
    All interpretation is in interpret_results.py.
    """
    setup_deterministic(seed)

    from embark.benchmark.agents import PIControllerAgent
    from embark.benchmark.harness import STANDARD_SCENARIOS, BenchmarkSuite, ClosedLoopHarness
    from embark.benchmark.metrics.neurobench_factory import create_metrics
    from embark.benchmark.physics.config import PMSMConfig

    results_dir = ensure_results_dir("phase2_metric_validation", run_name)

    # Find the primary reference scenario
    target_scenario = None
    for s in STANDARD_SCENARIOS:
        if s.name == "step_mid_speed_1500rpm_2A":
            target_scenario = s
            break

    if target_scenario is None:
        print("ERROR: step_mid_speed_1500rpm_2A not found in STANDARD_SCENARIOS")
        return {}

    print("=" * 70)
    print("  PVP Phase 2 — Metric Validation (SC-2)")
    print(f"  Scenario: {target_scenario.name}")
    print("=" * 70)

    pmsm_config = PMSMConfig()
    task = target_scenario.create_task(physics_config=pmsm_config)
    controller = PIControllerAgent.from_system_config(task.physics_engine.config)
    metrics = create_metrics(controller)
    dt = task.physics_engine.config.tau

    # Run step-by-step to capture trajectory AND pipeline metrics
    state, reference = task.reset()
    controller.reset()
    for m in metrics:
        m.reset()

    trajectory = {
        "i_q": [],
        "i_q_ref": [],
        "i_q_at_step_start": [],  # state["i_q"] at start of step (pipeline alignment)
        "i_d": [],
        "i_d_ref": [],
        "i_d_at_step_start": [],  # state["i_d"] at start of step (pipeline alignment)
        "u_q": [],
        "u_d": [],
    }
    step = 0
    done = False

    while not done and step < target_scenario.max_steps:
        # Pipeline metrics receive (state, reference, ...); they typically use ref vs state.
        trajectory["i_q_at_step_start"].append(float(state["i_q"]))
        trajectory["i_d_at_step_start"].append(float(state["i_d"]))

        action = controller(state, reference)
        controller_info = getattr(controller, "last_info", None)
        next_state, next_ref, done = task.step(action)

        for m in metrics:
            m.update(state, reference, action, next_state, controller_info)

        trajectory["i_q_ref"].append(float(reference["i_q_ref"]))
        trajectory["i_q"].append(float(next_state["i_q"]))
        trajectory["i_d_ref"].append(float(reference["i_d_ref"]))
        trajectory["i_d"].append(float(next_state["i_d"]))
        trajectory["u_q"].append(float(action["v_q"]))
        trajectory["u_d"].append(float(action["v_d"]))

        state, reference = next_state, next_ref
        step += 1

    # Pipeline metrics
    pipeline_results: dict[str, Any] = {"steps": step}
    for m in metrics:
        result = m.compute()
        if isinstance(result, dict):
            pipeline_results.update(result)
        else:
            pipeline_results[m.name] = result

    # Manual computation
    i_q = np.array(trajectory["i_q"])  # next_state["i_q"] after each step
    i_q_at_step_start = np.array(trajectory["i_q_at_step_start"])  # state["i_q"] at step start
    i_q_ref = np.array(trajectory["i_q_ref"])
    i_d = np.array(trajectory["i_d"])
    i_d_at_step_start = np.array(trajectory["i_d_at_step_start"])
    i_d_ref = np.array(trajectory["i_d_ref"])
    step_onset, pre_step_ref = _get_step_onset_and_pre_ref(i_q_ref)

    # Pipeline TrackingMAE uses (reference, state) at each update, i.e. ref_t vs state_t.
    # So we use state at step start for MAE to match pipeline; otherwise deviation ~6e-4.
    manual_mae = _manual_mae(i_q_at_step_start, i_q_ref)
    manual_mae_next = _manual_mae(i_q, i_q_ref)  # kept for diagnostics
    manual_mae_d = _manual_mae(i_d_at_step_start, i_d_ref)
    manual_itae = _manual_itae(i_q, i_q_ref, dt, step_onset)
    manual_settling = _manual_settling_time(i_q, i_q_ref, dt, step_onset, pre_step_ref)
    manual_overshoot = _manual_overshoot(i_q_at_step_start, i_q_ref, step_onset, pre_step_ref)
    # Steady-state RMS (from t >= 50 ms to end); pipeline may use same or full-episode.
    manual_rms_q_ss = _manual_rms_steady_state(i_q, i_q_ref, dt)
    manual_rms_d_ss = _manual_rms_steady_state(i_d, i_d_ref, dt)

    # Collect comparisons (raw data, no verdicts)
    comparisons: list[dict[str, Any]] = []
    report_lines: list[str] = [
        "PVP Phase 2 — Metric Validation",
        f"Scenario: {target_scenario.name}",
        f"Step onset index: {step_onset}, pre_step_ref: {pre_step_ref}",
        f"Steps: {step}, dt: {dt}",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"{'Metric':<25s} {'Manual':>14s} {'Pipeline':>14s} {'Deviation':>14s}",
        "-" * 70,
    ]

    metric_pairs = [
        ("MAE_i_q", manual_mae, pipeline_results.get("mae_i_q", float("nan"))),
        ("MAE_i_d", manual_mae_d, pipeline_results.get("mae_i_d", float("nan"))),
        ("ITAE_i_q", manual_itae, pipeline_results.get("itae_i_q", float("nan"))),
        ("Settling_i_q", manual_settling, pipeline_results.get("settling_time_i_q", float("nan"))),
        ("Overshoot_i_q", manual_overshoot, pipeline_results.get("overshoot", float("nan"))),
        ("RMS_i_q_steady_state", manual_rms_q_ss, pipeline_results.get("rms_i_q", float("nan"))),
        ("RMS_i_d_steady_state", manual_rms_d_ss, pipeline_results.get("rms_i_d", float("nan"))),
    ]

    for name, manual_val, pipeline_val in metric_pairs:
        if np.isnan(manual_val) or np.isnan(pipeline_val):
            dev = float("nan")
        else:
            dev = abs(manual_val - pipeline_val)

        comp = {
            "metric": name,
            "manual": manual_val,
            "pipeline": pipeline_val,
            "deviation": dev,
        }
        comparisons.append(comp)

        manual_str = "N/A" if np.isnan(manual_val) else f"{manual_val:14.10f}"
        pipe_str = "N/A" if np.isnan(pipeline_val) else f"{pipeline_val:14.10f}"
        dev_str = "N/A" if np.isnan(dev) else f"{dev:14.2e}"
        line = f"  {name:<25s} {manual_str:>14s} {pipe_str:>14s} {dev_str:>14s}"
        print(line)
        report_lines.append(line)

    report_lines.append("")

    # ===================================================================
    # Part B — Neuromorphic Metric Validation (SNN controller)
    # ===================================================================
    print("")
    print("  --- Part B: Neuromorphic Metric Validation ---")

    snn_spec = MODELS[0]  # best_incremental_snn
    snn_controller, _snn_meta = build_snn_controller(snn_spec)
    print(f"  SNN probe: {snn_spec.name} ({snn_spec.version})")

    # Fresh task + metrics for SNN run
    snn_task = target_scenario.create_task(physics_config=pmsm_config)
    snn_metrics = create_metrics(snn_controller)

    if hasattr(snn_controller, "configure"):
        snn_controller.configure(snn_task.physics_engine.config, snn_task)

    snn_state, snn_reference = snn_task.reset()
    snn_controller.reset()
    for m in snn_metrics:
        m.reset()

    # Per-step raw neuromorphic data
    raw_spikes: list[float] = []
    raw_syops: list[float] = []
    raw_sparsity: list[float] = []

    snn_step = 0
    snn_done = False

    while not snn_done and snn_step < target_scenario.max_steps:
        snn_action = snn_controller(snn_state, snn_reference)
        snn_info = getattr(snn_controller, "last_info", None)
        snn_next_state, snn_next_ref, snn_done = snn_task.step(snn_action)

        for m in snn_metrics:
            m.update(snn_state, snn_reference, snn_action, snn_next_state, snn_info)

        # Capture raw per-step neuromorphic values
        if snn_info:
            raw_spikes.append(float(snn_info.get("total_spikes", 0)))
            raw_syops.append(float(snn_info.get("syops", 0)))
            raw_sparsity.append(float(snn_info.get("sparsity", 0)))

        snn_state, snn_reference = snn_next_state, snn_next_ref
        snn_step += 1

    # Pipeline neuromorphic metrics
    snn_pipeline: dict[str, Any] = {"steps": snn_step}
    for m in snn_metrics:
        result = m.compute()
        if isinstance(result, dict):
            snn_pipeline.update(result)
        else:
            snn_pipeline[m.name] = result

    # Manual neuromorphic metrics
    manual_spikes = _manual_spike_count(raw_spikes)
    manual_syops = _manual_synaptic_ops(raw_syops)
    manual_sparsity = _manual_activation_sparsity(raw_sparsity)

    # Compare
    neuromorphic_comparisons: list[dict[str, Any]] = []
    neuro_pairs = [
        ("Total_spikes", manual_spikes["total_spikes"], snn_pipeline.get("total_spikes", float("nan"))),
        ("Spikes_per_step", manual_spikes["spikes_per_step"], snn_pipeline.get("spikes_per_step", float("nan"))),
        ("Total_syops", manual_syops["total_syops"], snn_pipeline.get("total_syops", float("nan"))),
        ("SyOps_per_step", manual_syops["syops_per_step"], snn_pipeline.get("syops_per_step", float("nan"))),
        ("Mean_sparsity", manual_sparsity["mean_sparsity"], snn_pipeline.get("mean_sparsity", float("nan"))),
        ("Min_sparsity", manual_sparsity["min_sparsity"], snn_pipeline.get("min_sparsity", float("nan"))),
    ]

    report_lines.append("--- Part B: Neuromorphic Metric Validation ---")
    report_lines.append(f"SNN probe: {snn_spec.name} ({snn_spec.version})")
    report_lines.append(f"Steps: {snn_step}")
    report_lines.append(f"Raw controller_info samples: {len(raw_spikes)}")
    report_lines.append("")
    report_lines.append(
        f"{'Metric':<25s} {'Manual':>14s} {'Pipeline':>14s} {'Deviation':>14s}"
    )
    report_lines.append("-" * 70)

    for name, manual_val, pipeline_val in neuro_pairs:
        if np.isnan(manual_val) or np.isnan(pipeline_val):
            dev = float("nan")
        else:
            dev = abs(manual_val - pipeline_val)

        comp = {
            "metric": name,
            "manual": manual_val,
            "pipeline": pipeline_val,
            "deviation": dev,
        }
        neuromorphic_comparisons.append(comp)

        manual_str = "N/A" if np.isnan(manual_val) else f"{manual_val:14.6f}"
        pipe_str = "N/A" if np.isnan(pipeline_val) else f"{pipeline_val:14.6f}"
        dev_str = "N/A" if np.isnan(dev) else f"{dev:14.2e}"
        line = f"  {name:<25s} {manual_str:>14s} {pipe_str:>14s} {dev_str:>14s}"
        print(line)
        report_lines.append(line)

    report_lines.append("")
    report_lines.append("(No pass/fail verdicts — see interpret_results.py)")

    # ===================================================================
    # Save all results
    # ===================================================================
    def _json_float(v: float):
        """Convert non-finite floats to None for JSON serialisation."""
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return v

    save_json(
        {
            "scenario": target_scenario.name,
            "step_onset": step_onset,
            "pre_step_ref": pre_step_ref if not np.isnan(pre_step_ref) else None,
            "dt": dt,
            "comparisons": comparisons,
            "pipeline_metrics": pipeline_results,
            "manual_mae_next_state": manual_mae_next,
            # Raw trajectory for Plot 2.4 (step response timeseries).
            "trajectory": {
                "i_q": trajectory["i_q"],
                "i_q_ref": trajectory["i_q_ref"],
                "i_q_at_step_start": trajectory["i_q_at_step_start"],
            },
            # Manual scalar metrics needed for plot annotations.
            "manual_metrics": {
                "settling_time_i_q": _json_float(manual_settling),
                "overshoot_i_q": _json_float(manual_overshoot),
                "itae_i_q": _json_float(manual_itae),
                "mae_i_q": _json_float(manual_mae),
            },
            # Part B: Neuromorphic metric validation
            "snn_model": snn_spec.name,
            "neuromorphic_comparisons": neuromorphic_comparisons,
            "neuromorphic_pipeline_metrics": {
                k: snn_pipeline.get(k)
                for k in [
                    "total_spikes", "spikes_per_step",
                    "total_syops", "syops_per_step",
                    "mean_sparsity", "min_sparsity", "max_sparsity",
                ]
            },
        },
        results_dir / "phase2_validation.json",
    )
    save_text_report(report_lines, results_dir / "phase2_report.txt")

    return {
        "comparisons": comparisons,
        "neuromorphic_comparisons": neuromorphic_comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PVP Phase 2 — Metric Validation")
    parser.add_argument("--run", type=str, default=None, help="Run name for results directory")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    run_phase2(run_name=args.run, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
