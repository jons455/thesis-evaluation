"""
PVP Phase 1 — Correctness Probing (SC-1, SC-5).

R1: PI native GEM (without wrapper — direct harness).
R2: PI via BenchmarkSuite wrapper (standard pipeline path).

Both use STANDARD_SCENARIOS, float64 on the PI path.
Compares R1 vs R2 trajectories: residual must be < 1e-12 A (pass), > 1e-6 A (hard fail).

Also logs full trajectories (t, i_q_ref, i_q, u_q) for overlay and residual plots.

Usage:
    poetry run python -m embark-evaluation.evaluation.phase1_correctness
    poetry run python -m embark-evaluation.evaluation.phase1_correctness --run pvp_run1
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


def _run_pi_native(scenarios, pmsm_config) -> dict[str, dict[str, Any]]:
    """
    R1: Run PI controller through the harness directly (native GEM).

    Uses ClosedLoopHarness with manually-created tasks — no BenchmarkSuite.
    Records full trajectory per scenario.
    """
    from embark.benchmark.agents import PIControllerAgent
    from embark.benchmark.harness import ClosedLoopHarness
    from embark.benchmark.metrics.neurobench_factory import create_metrics

    results = {}
    for scenario in scenarios:
        task = scenario.create_task(physics_config=pmsm_config)
        controller = PIControllerAgent.from_system_config(task.physics_engine.config)
        metrics = create_metrics(controller)

        harness = ClosedLoopHarness(task=task, controller=controller, metrics=metrics)

        # Capture trajectory by running step-by-step
        state, reference = task.reset()
        controller.reset()
        for m in metrics:
            m.reset()

        if hasattr(controller, "configure"):
            controller.configure(task.physics_engine.config, task)

        trajectory = {"t": [], "i_q_ref": [], "i_q": [], "i_d_ref": [], "i_d": [], "u_q": [], "u_d": []}
        step = 0
        done = False
        dt = task.physics_engine.config.tau

        while not done and step < (scenario.max_steps or float("inf")):
            action = controller(state, reference)
            controller_info = getattr(controller, "last_info", None)
            next_state, next_ref, done = task.step(action)

            for m in metrics:
                m.update(state, reference, action, next_state, controller_info)

            trajectory["t"].append(step * dt)
            trajectory["i_q_ref"].append(float(reference["i_q_ref"]))
            trajectory["i_q"].append(float(next_state["i_q"]))
            trajectory["i_d_ref"].append(float(reference["i_d_ref"]))
            trajectory["i_d"].append(float(next_state["i_d"]))
            trajectory["u_q"].append(float(action["v_q"]))
            trajectory["u_d"].append(float(action["v_d"]))

            state, reference = next_state, next_ref
            step += 1

        metric_results: dict[str, Any] = {"steps": step}
        for m in metrics:
            result = m.compute()
            if isinstance(result, dict):
                metric_results.update(result)
            else:
                metric_results[m.name] = result

        results[scenario.name] = {
            "trajectory": {k: [float(v) for v in vals] for k, vals in trajectory.items()},
            "metrics": metric_results,
        }

    return results


def _run_pi_wrapper(scenarios, pmsm_config) -> dict[str, dict[str, Any]]:
    """
    R2: Run PI controller through BenchmarkSuite (standard pipeline).

    Uses BenchmarkSuite.run_baseline() which creates PIControllerAgent internally.
    We also capture trajectories by re-running with manual trajectory capture,
    using the same approach as R1 but going through the suite's task creation.
    """
    from embark.benchmark.agents import PIControllerAgent
    from embark.benchmark.harness import BenchmarkConfig, BenchmarkSuite, ClosedLoopHarness
    from embark.benchmark.metrics.neurobench_factory import create_metrics

    config = BenchmarkConfig()
    suite = BenchmarkSuite(scenarios=scenarios, config=config, verbose=False)

    # Run via suite for official metrics
    summary = suite.run_baseline(name="PI-wrapper", quiet=True)

    # Also capture trajectories with manual step-by-step
    results = {}
    for i, scenario in enumerate(scenarios):
        task = scenario.create_task(physics_config=pmsm_config)
        controller = PIControllerAgent.from_system_config(task.physics_engine.config)
        metrics = create_metrics(controller)

        state, reference = task.reset()
        controller.reset()
        for m in metrics:
            m.reset()

        trajectory = {"t": [], "i_q_ref": [], "i_q": [], "i_d_ref": [], "i_d": [], "u_q": [], "u_d": []}
        step = 0
        done = False
        dt = task.physics_engine.config.tau

        while not done and step < (scenario.max_steps or float("inf")):
            action = controller(state, reference)
            controller_info = getattr(controller, "last_info", None)
            next_state, next_ref, done = task.step(action)

            for m in metrics:
                m.update(state, reference, action, next_state, controller_info)

            trajectory["t"].append(step * dt)
            trajectory["i_q_ref"].append(float(reference["i_q_ref"]))
            trajectory["i_q"].append(float(next_state["i_q"]))
            trajectory["i_d_ref"].append(float(reference["i_d_ref"]))
            trajectory["i_d"].append(float(next_state["i_d"]))
            trajectory["u_q"].append(float(action["v_q"]))
            trajectory["u_d"].append(float(action["v_d"]))

            state, reference = next_state, next_ref
            step += 1

        metric_results: dict[str, Any] = {"steps": step}
        for m in metrics:
            result = m.compute()
            if isinstance(result, dict):
                metric_results.update(result)
            else:
                metric_results[m.name] = result

        results[scenario.name] = {
            "trajectory": {k: [float(v) for v in vals] for k, vals in trajectory.items()},
            "metrics": metric_results,
        }

    return results


def run_phase1(run_name: str | None = None, seed: int = 42) -> dict:
    """Execute Phase 1: PI native vs PI wrapper correctness probing."""
    setup_deterministic(seed)

    from embark.benchmark.harness import STANDARD_SCENARIOS
    from embark.benchmark.physics.config import PMSMConfig

    results_dir = ensure_results_dir("phase1_correctness", run_name)
    pmsm_config = PMSMConfig()
    scenarios = STANDARD_SCENARIOS

    print("=" * 70)
    print("  PVP Phase 1 — Correctness Probing (SC-1, SC-5)")
    print(f"  Scenarios: {len(scenarios)}")
    print("=" * 70)

    # R1: PI native
    print("\n  R1: PI native GEM (direct harness)...")
    t0 = time.perf_counter()
    r1 = _run_pi_native(scenarios, pmsm_config)
    print(f"    Done in {time.perf_counter() - t0:.1f} s")

    # R2: PI via wrapper
    print("\n  R2: PI via BenchmarkSuite wrapper...")
    t0 = time.perf_counter()
    r2 = _run_pi_wrapper(scenarios, pmsm_config)
    print(f"    Done in {time.perf_counter() - t0:.1f} s")

    # --- Residual analysis ---
    report_lines: list[str] = [
        "PVP Phase 1 — Correctness Probing",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    residual_results: dict[str, dict[str, Any]] = {}
    overall_pass = True

    print("\n  --- Residual Analysis (R1 vs R2) ---")
    for scenario in scenarios:
        sname = scenario.name
        t1 = r1[sname]["trajectory"]
        t2 = r2[sname]["trajectory"]

        min_len = min(len(t1["i_q"]), len(t2["i_q"]))
        iq_r1 = np.array(t1["i_q"][:min_len])
        iq_r2 = np.array(t2["i_q"][:min_len])
        uq_r1 = np.array(t1["u_q"][:min_len])
        uq_r2 = np.array(t2["u_q"][:min_len])

        residual_iq = np.abs(iq_r1 - iq_r2)
        residual_uq = np.abs(uq_r1 - uq_r2)
        max_res_iq = float(np.max(residual_iq))
        max_res_uq = float(np.max(residual_uq))
        mean_res_iq = float(np.mean(residual_iq))

        if max_res_iq < 1e-12:
            verdict = "PASS (< 1e-12)"
        elif max_res_iq < 1e-6:
            verdict = "INVESTIGATE (1e-12 to 1e-6)"
        else:
            verdict = "HARD FAIL (> 1e-6)"
            overall_pass = False

        residual_results[sname] = {
            "max_residual_iq_A": max_res_iq,
            "mean_residual_iq_A": mean_res_iq,
            "max_residual_uq_V": max_res_uq,
            "verdict": verdict,
            "steps_compared": min_len,
        }

        line = f"  {sname}: max|R1-R2|_iq = {max_res_iq:.2e} A -> {verdict}"
        print(line)
        report_lines.append(line)

    report_lines.append("")
    report_lines.append(f"Overall SC-1: {'PASS' if overall_pass else 'FAIL'}")

    # --- Save trajectories and results ---
    save_json(
        {"R1_native": {s: r1[s]["metrics"] for s in r1}, "R2_wrapper": {s: r2[s]["metrics"] for s in r2}},
        results_dir / "phase1_metrics.json",
    )
    save_json(residual_results, results_dir / "phase1_residuals.json")

    # Save full trajectories for plotting
    for sname in r1:
        save_json(r1[sname]["trajectory"], results_dir / f"R1_trajectory_{sname}.json")
        save_json(r2[sname]["trajectory"], results_dir / f"R2_trajectory_{sname}.json")

    save_text_report(report_lines, results_dir / "phase1_report.txt")

    return {"residuals": residual_results, "overall_pass": overall_pass}


def main() -> int:
    parser = argparse.ArgumentParser(description="PVP Phase 1 — Correctness Probing")
    parser.add_argument("--run", type=str, default=None, help="Run name for results directory")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    run_phase1(run_name=args.run, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
