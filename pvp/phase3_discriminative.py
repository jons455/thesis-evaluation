"""
PVP Phase 3 — Discriminative Power (SC-3).

Runs R3–R5 (three SNN models via BenchmarkSuite wrapper) + R2 (PI reused).
All use STANDARD_SCENARIOS. Full state reset between each model run.

Pass/fail: ranking order matches Phase 0 ground truth per scenario.
Float32 precision floor applies; sub-1e-4 A MAE differences are not meaningful.

Usage:
    poetry run python embark-evaluation/pvp/phase3_discriminative.py
    poetry run python embark-evaluation/pvp/phase3_discriminative.py --run pvp_run1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parents[2]
_embark_eval_dir = Path(__file__).resolve().parents[1]
for _p in [str(_repo_root), str(_embark_eval_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pvp.utils.common import (
    MODELS,
    build_snn_controller,
    ensure_results_dir,
    load_phase0_rankings,
    save_json,
    save_text_report,
    setup_deterministic,
)


def _run_with_trajectories(controller, scenarios, pmsm_config, controller_name):
    """
    Run controller through scenarios, capturing both metrics and trajectories.

    Returns (per_scenario_metrics, per_scenario_trajectories).
    """
    from embark.benchmark.harness import ClosedLoopHarness
    from embark.benchmark.metrics.neurobench_factory import create_metrics

    all_metrics = {}
    all_trajectories = {}

    for scenario in scenarios:
        task = scenario.create_task(physics_config=pmsm_config)
        metrics = create_metrics(controller)

        if hasattr(controller, "configure"):
            controller.configure(task.physics_engine.config, task)

        state, reference = task.reset()
        controller.reset()
        for m in metrics:
            m.reset()

        traj = {"t": [], "i_q_ref": [], "i_q": [], "i_d_ref": [], "i_d": [], "u_q": [], "u_d": []}
        step = 0
        done = False
        dt = task.physics_engine.config.tau

        while not done and step < (scenario.max_steps or float("inf")):
            action = controller(state, reference)
            controller_info = getattr(controller, "last_info", None)
            next_state, next_ref, done = task.step(action)

            for m in metrics:
                m.update(state, reference, action, next_state, controller_info)

            traj["t"].append(step * dt)
            traj["i_q_ref"].append(float(reference["i_q_ref"]))
            traj["i_q"].append(float(next_state["i_q"]))
            traj["i_d_ref"].append(float(reference["i_d_ref"]))
            traj["i_d"].append(float(next_state["i_d"]))
            traj["u_q"].append(float(action["v_q"]))
            traj["u_d"].append(float(action["v_d"]))

            state, reference = next_state, next_ref
            step += 1

        metric_results: dict[str, Any] = {"steps": step}
        for m in metrics:
            result = m.compute()
            if isinstance(result, dict):
                metric_results.update(result)
            else:
                metric_results[m.name] = result

        all_metrics[scenario.name] = metric_results
        all_trajectories[scenario.name] = traj

    return all_metrics, all_trajectories


def run_phase3(
    run_name: str | None = None,
    seed: int = 42,
    quick: bool = False,
) -> dict:
    """Execute Phase 3: discriminative power validation."""
    setup_deterministic(seed)

    from embark.benchmark.harness import (
        QUICK_SCENARIOS,
        STANDARD_SCENARIOS,
        BenchmarkSuite,
    )
    from embark.benchmark.agents import PIControllerAgent
    from embark.benchmark.physics.config import PMSMConfig

    scenarios = QUICK_SCENARIOS if quick else STANDARD_SCENARIOS
    results_dir = ensure_results_dir("phase3_discriminative", run_name)
    pmsm_config = PMSMConfig()

    print("=" * 70)
    print("  PVP Phase 3 — Discriminative Power (SC-3)")
    print(f"  Scenarios: {len(scenarios)} ({'QUICK' if quick else 'STANDARD'})")
    print("=" * 70)

    suite = BenchmarkSuite(scenarios=scenarios, verbose=True)

    # --- R2: PI baseline with trajectory capture ---
    print("\n  R2: PI baseline (with trajectory capture)...")
    t0 = time.perf_counter()
    pi_controller = PIControllerAgent.from_system_config(pmsm_config)
    pi_metrics, pi_trajectories = _run_with_trajectories(
        pi_controller, scenarios, pmsm_config, "PI-baseline"
    )
    # Also run through suite for the official BenchmarkSummary
    pi_summary = suite.run_baseline(name="PI-baseline", quiet=True)
    print(f"    Done in {time.perf_counter() - t0:.1f} s")
    print(BenchmarkSuite.format_summary(pi_summary))

    # Save PI trajectories
    for sname, traj in pi_trajectories.items():
        save_json(traj, results_dir / f"trajectory_PI-baseline_{sname}.json")

    # --- R3–R5: SNN models with trajectory capture ---
    snn_summaries: dict[str, Any] = {}
    run_ids = {"best": "R3", "intermediate": "R4", "poor": "R5"}

    for spec in MODELS:
        rid = run_ids[spec.quality]
        print(f"\n  {rid}: {spec.name} ({spec.quality})...")
        t0 = time.perf_counter()

        controller, meta = build_snn_controller(spec, device="cpu")

        # Trajectory capture run
        snn_metrics, snn_trajectories = _run_with_trajectories(
            controller, scenarios, pmsm_config, spec.name
        )

        # Official BenchmarkSuite run
        controller2, _ = build_snn_controller(spec, device="cpu")
        summary = suite.run(controller=controller2, name=spec.name, quiet=True)
        elapsed = time.perf_counter() - t0

        print(BenchmarkSuite.format_summary(summary))
        print(f"    Time: {elapsed:.1f} s")

        snn_summaries[spec.name] = summary

        # Save SNN trajectories
        for sname, traj in snn_trajectories.items():
            save_json(traj, results_dir / f"trajectory_{spec.name}_{sname}.json")

    # --- Extract MAE_q per model per scenario ---
    all_mae: dict[str, dict[str, float]] = {}

    # PI
    all_mae["PI-baseline"] = {}
    for sr in pi_summary.scenario_results:
        all_mae["PI-baseline"][sr.scenario_name] = sr.metrics.get("mae_i_q", float("nan"))

    # SNNs
    for name, summary in snn_summaries.items():
        all_mae[name] = {}
        for sr in summary.scenario_results:
            all_mae[name][sr.scenario_name] = sr.metrics.get("mae_i_q", float("nan"))

    # --- Ranking comparison vs Phase 0 ---
    report_lines: list[str] = [
        "PVP Phase 3 — Discriminative Power",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # Try to load Phase 0 rankings
    try:
        p0_rankings = load_phase0_rankings(results_dir)
        has_p0 = True
    except FileNotFoundError:
        print("\n  WARNING: Phase 0 rankings not found. Skipping ranking comparison.")
        print("  Run Phase 0 first, then re-run Phase 3.")
        has_p0 = False
        p0_rankings = {}

    # Display MAE table
    report_lines.append("--- MAE_q per Model per Scenario ---")
    scenario_names = [s.name for s in scenarios]
    header = f"  {'Model':<35s} " + " ".join(f"{s:>20s}" for s in scenario_names)
    report_lines.append(header)
    report_lines.append("  " + "-" * (35 + 21 * len(scenario_names)))

    for model_name in ["PI-baseline"] + [m.name for m in MODELS]:
        vals = []
        for sn in scenario_names:
            v = all_mae.get(model_name, {}).get(sn, float("nan"))
            vals.append(f"{v:>20.6f}")
        line = f"  {model_name:<35s} " + " ".join(vals)
        report_lines.append(line)
        print(line)

    # --- Ranking check per scenario ---
    if has_p0:
        report_lines.append("")
        report_lines.append("--- Ranking Comparison (Phase 3 vs Phase 0) ---")
        snn_names = [m.name for m in MODELS]
        overall_pass = True

        for sn in scenario_names:
            # Phase 3 ranking (SNN models only)
            p3_ranked = sorted(snn_names, key=lambda n: all_mae.get(n, {}).get(sn, float("inf")))

            # Phase 0 ranking (may only cover single-step scenarios)
            if sn in p0_rankings.get(MODELS[0].name, {}):
                p0_ranked = sorted(snn_names, key=lambda n: p0_rankings.get(n, {}).get(sn, float("inf")))
                match = p3_ranked == p0_ranked
                if not match:
                    overall_pass = False
                verdict = "MATCH" if match else "MISMATCH"
                line = (
                    f"  {sn}: P0={' < '.join(p0_ranked)}"
                    f" | P3={' < '.join(p3_ranked)}"
                    f" -> {verdict}"
                )
            else:
                line = f"  {sn}: No Phase 0 data (multi-step scenario) — P3={' < '.join(p3_ranked)}"
            report_lines.append(line)
            print(line)

        report_lines.append("")
        report_lines.append(f"Overall SC-3: {'PASS' if overall_pass else 'PARTIAL FAIL'}")
        print(f"\n  Overall SC-3: {'PASS' if overall_pass else 'PARTIAL FAIL'}")
    else:
        overall_pass = None

    # --- Save results ---
    # Full summary dicts
    all_results = {
        "PI-baseline": pi_summary.to_dict(),
    }
    for name, summary in snn_summaries.items():
        all_results[name] = summary.to_dict()

    save_json(all_results, results_dir / "phase3_summaries.json")
    save_json(all_mae, results_dir / "phase3_mae_table.json")
    BenchmarkSuite.save_results(pi_summary, results_dir / "R2_PI_baseline.json")
    for spec in MODELS:
        rid = run_ids[spec.quality]
        BenchmarkSuite.save_results(
            snn_summaries[spec.name],
            results_dir / f"{rid}_{spec.name}.json",
        )
    save_text_report(report_lines, results_dir / "phase3_report.txt")

    return {"all_mae": all_mae, "overall_pass": overall_pass}


def main() -> int:
    parser = argparse.ArgumentParser(description="PVP Phase 3 — Discriminative Power")
    parser.add_argument("--run", type=str, default=None, help="Run name for results directory")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--quick", action="store_true", help="Use QUICK_SCENARIOS")
    args = parser.parse_args()

    run_phase3(run_name=args.run, seed=args.seed, quick=args.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
