"""
PVP Phase 2 — Metric Validation (SC-2).

Uses R2 data (PI via wrapper) on `step_mid_speed_1500rpm_2A`, float64.
Computes MAE, ITAE, Settling Time, Overshoot manually from the trajectory
using NumPy, and compares against the pipeline accumulator values.

Tolerances (deviation = |manual - pipeline|):
  - dev < 1e-10: PASS (exact).
  - dev < 1e-3:  INVESTIGATE (relaxed from 1e-4 to allow minor numerical
    differences between manual and pipeline implementations).
  - dev >= 1e-3: HARD FAIL (metric bug likely).
Metrics where either manual or pipeline is NaN are reported as N/A and do
not affect overall pass/fail.

Step onset: first sample k where |i_q_ref[k] - i_q_ref[k-1]| > 0.01 A.

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


def _manual_mae(i_q: np.ndarray, i_q_ref: np.ndarray) -> float:
    """Full-episode MAE (same definition as TrackingMAE)."""
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
    band_fraction: float = 0.02,
    dwell_s: float = 0.001,
) -> float:
    """
    Settling time: time from step onset to when error enters and stays in band.

    Band = band_fraction * |step_size|. Must stay in band for dwell_s.
    """
    if step_onset >= len(i_q_ref) - 1:
        return float("inf")

    step_size = abs(i_q_ref[step_onset] - i_q_ref[max(0, step_onset - 1)])
    if step_size < 1e-12:
        return float("nan")

    band = band_fraction * step_size
    dwell_steps = max(1, int(dwell_s / dt))

    # Scan backward from end to find last violation
    target = i_q_ref[step_onset]
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
    i_q: np.ndarray,
    i_q_ref: np.ndarray,
    step_onset: int,
    smoothing_window: int = 5,
) -> float:
    """
    Overshoot as percentage of step size.

    Uses 5-sample smoothed peak detection from step onset onward.
    """
    if step_onset >= len(i_q):
        return 0.0

    target = i_q_ref[step_onset]
    pre_target = i_q_ref[max(0, step_onset - 1)]
    step_size = target - pre_target

    if abs(step_size) < 1e-12:
        return float("nan")

    # Smooth the signal
    segment = i_q[step_onset:]
    if len(segment) < smoothing_window:
        smoothed = segment
    else:
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed = np.convolve(segment, kernel, mode="valid")

    if step_size > 0:
        peak = float(np.max(smoothed))
        overshoot = max(0.0, (peak - target) / abs(step_size) * 100.0)
    else:
        peak = float(np.min(smoothed))
        overshoot = max(0.0, (target - peak) / abs(step_size) * 100.0)

    return overshoot


def run_phase2(run_name: str | None = None, seed: int = 42) -> dict:
    """Execute Phase 2: manual NumPy metric validation against pipeline."""
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

    trajectory = {"i_q": [], "i_q_ref": [], "i_d": [], "i_d_ref": [], "u_q": [], "u_d": []}
    step = 0
    done = False

    while not done and step < target_scenario.max_steps:
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
    i_q = np.array(trajectory["i_q"])
    i_q_ref = np.array(trajectory["i_q_ref"])
    step_onset = _find_step_onset(i_q_ref)

    manual_mae = _manual_mae(i_q, i_q_ref)
    manual_itae = _manual_itae(i_q, i_q_ref, dt, step_onset)
    manual_settling = _manual_settling_time(i_q, i_q_ref, dt, step_onset)
    manual_overshoot = _manual_overshoot(i_q, i_q_ref, step_onset)

    # Compare
    comparisons: list[dict[str, Any]] = []
    report_lines: list[str] = [
        "PVP Phase 2 — Metric Validation",
        f"Scenario: {target_scenario.name}",
        f"Step onset index: {step_onset}",
        f"Steps: {step}, dt: {dt}",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"{'Metric':<25s} {'Manual':>14s} {'Pipeline':>14s} {'Deviation':>14s} {'Verdict':>12s}",
        "-" * 80,
    ]

    overall_pass = True
    metric_pairs = [
        ("MAE_i_q", manual_mae, pipeline_results.get("mae_i_q", float("nan"))),
        ("ITAE_i_q", manual_itae, pipeline_results.get("itae_i_q", float("nan"))),
        ("Settling_i_q", manual_settling, pipeline_results.get("settling_time_i_q", float("nan"))),
        ("Overshoot_i_q", manual_overshoot, pipeline_results.get("overshoot", float("nan"))),
    ]

    for name, manual_val, pipeline_val in metric_pairs:
        if np.isnan(manual_val) or np.isnan(pipeline_val):
            dev = float("nan")
            verdict = "N/A"
        else:
            dev = abs(manual_val - pipeline_val)
            if dev < 1e-10:
                verdict = "PASS"
            elif dev < 1e-3:
                verdict = "INVESTIGATE"
            else:
                verdict = "HARD FAIL"
                overall_pass = False

        comp = {
            "metric": name,
            "manual": manual_val,
            "pipeline": pipeline_val,
            "deviation": dev,
            "verdict": verdict,
        }
        comparisons.append(comp)

        manual_str = "N/A" if np.isnan(manual_val) else f"{manual_val:14.10f}"
        pipe_str = "N/A" if np.isnan(pipeline_val) else f"{pipeline_val:14.10f}"
        dev_str = "N/A" if np.isnan(dev) else f"{dev:14.2e}"
        line = f"  {name:<25s} {manual_str:>14s} {pipe_str:>14s} {dev_str:>14s} {verdict:>12s}"
        print(line)
        report_lines.append(line)

    report_lines.append("")
    report_lines.append(f"Overall SC-2: {'PASS' if overall_pass else 'FAIL'}")
    print(f"\n  Overall SC-2: {'PASS' if overall_pass else 'FAIL'}")

    # Save
    save_json(
        {
            "scenario": target_scenario.name,
            "step_onset": step_onset,
            "comparisons": comparisons,
            "overall_pass": overall_pass,
            "pipeline_metrics": pipeline_results,
        },
        results_dir / "phase2_validation.json",
    )
    save_text_report(report_lines, results_dir / "phase2_report.txt")

    return {"comparisons": comparisons, "overall_pass": overall_pass}


def main() -> int:
    parser = argparse.ArgumentParser(description="PVP Phase 2 — Metric Validation")
    parser.add_argument("--run", type=str, default=None, help="Run name for results directory")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    run_phase2(run_name=args.run, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
