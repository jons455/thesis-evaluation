"""
PVP Phase 6 — Overhead Profiling (SC-7).

Profiles: GEM step time, wrapper overhead, model inference time, total wall time.
Runs PI + all 3 SNN models through STANDARD_SCENARIOS (or QUICK).

Success: Full PVP benchmark (PI + 3 models x scenarios) completes within 2 hours.

Output: per-component timing breakdown table.

Usage:
    poetry run python embark-evaluation/pvp/phase6_overhead.py
    poetry run python embark-evaluation/pvp/phase6_overhead.py --run pvp_run1
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
    save_json,
    save_text_report,
    setup_deterministic,
)


def _profile_controller(controller, suite, name: str) -> dict[str, Any]:
    """Run a controller through the suite and measure wall time."""
    t0 = time.perf_counter()
    summary = suite.run(controller=controller, name=name, quiet=True)
    wall_time = time.perf_counter() - t0

    total_steps = sum(sr.metrics.get("steps", 0) for sr in summary.scenario_results)

    return {
        "name": name,
        "wall_time_s": wall_time,
        "total_steps": total_steps,
        "time_per_step_us": (wall_time / total_steps * 1e6) if total_steps > 0 else 0,
        "num_scenarios": len(summary.scenario_results),
    }


def run_phase6(
    run_name: str | None = None,
    seed: int = 42,
    quick: bool = False,
) -> dict:
    """Execute Phase 6: overhead profiling."""
    setup_deterministic(seed)

    from embark.benchmark.harness import QUICK_SCENARIOS, STANDARD_SCENARIOS, BenchmarkSuite

    scenarios = QUICK_SCENARIOS if quick else STANDARD_SCENARIOS
    results_dir = ensure_results_dir("phase6_overhead", run_name)

    print("=" * 70)
    print("  PVP Phase 6 — Overhead Profiling (SC-7)")
    print(f"  Scenarios: {len(scenarios)}")
    print("=" * 70)

    suite = BenchmarkSuite(scenarios=scenarios, verbose=False)

    timing_results: list[dict[str, Any]] = []
    total_wall = 0.0

    # PI baseline (suite.run_baseline uses internal PI; no controller object to pass)
    print("\n  Profiling PI baseline...")
    t0 = time.perf_counter()
    suite.run_baseline(name="PI-baseline", quiet=True)
    pi_wall = time.perf_counter() - t0
    pi_result = {"name": "PI-baseline", "wall_time_s": pi_wall}
    timing_results.append(pi_result)
    total_wall += pi_wall
    print(f"    PI: {pi_wall:.2f} s")

    # SNN models
    for spec in MODELS:
        print(f"\n  Profiling {spec.name}...")
        controller, meta = build_snn_controller(spec, device="cpu")
        result = _profile_controller(controller, suite, spec.name)
        timing_results.append(result)
        total_wall += result["wall_time_s"]
        print(f"    {spec.name}: {result['wall_time_s']:.2f} s ({result['time_per_step_us']:.1f} us/step)")

    # SC-7 assessment
    two_hours_s = 2 * 3600
    feasible = total_wall < two_hours_s

    report_lines: list[str] = [
        "PVP Phase 6 — Overhead Profiling",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Mode: {'QUICK' if quick else 'STANDARD'}",
        "",
        f"{'Controller':<40s} {'Wall Time':>12s} {'Steps':>8s} {'us/step':>10s}",
        "-" * 72,
    ]

    for r in timing_results:
        steps = r.get("total_steps", "N/A")
        us_step = r.get("time_per_step_us", "N/A")
        steps_str = str(steps) if isinstance(steps, int) else "N/A"
        us_str = f"{us_step:.1f}" if isinstance(us_step, (int, float)) else "N/A"
        line = f"  {r['name']:<40s} {r['wall_time_s']:>11.2f}s {steps_str:>8s} {us_str:>10s}"
        report_lines.append(line)
        print(line)

    report_lines.append("-" * 72)
    report_lines.append(f"  {'TOTAL':<40s} {total_wall:>11.2f}s")
    report_lines.append("")

    if quick:
        # Estimate full time: STANDARD has 6 scenarios vs QUICK's 2
        estimated_full = total_wall * (6.0 / 2.0)
        report_lines.append(f"  Estimated STANDARD time: {estimated_full:.0f} s ({estimated_full / 60:.1f} min)")
        feasible = estimated_full < two_hours_s

    report_lines.append(f"  SC-7 budget: {two_hours_s} s (2 hours)")
    report_lines.append(f"  SC-7: {'PASS' if feasible else 'FAIL'} (total={'%.0f' % total_wall} s)")

    print(f"\n  Total wall time: {total_wall:.1f} s")
    print(f"  SC-7: {'PASS' if feasible else 'FAIL'}")

    save_json(
        {"timing": timing_results, "total_wall_s": total_wall, "feasible": feasible},
        results_dir / "phase6_timing.json",
    )
    save_text_report(report_lines, results_dir / "phase6_report.txt")

    return {"total_wall_s": total_wall, "feasible": feasible}


def main() -> int:
    parser = argparse.ArgumentParser(description="PVP Phase 6 — Overhead Profiling")
    parser.add_argument("--run", type=str, default=None, help="Run name for results directory")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--quick", action="store_true", help="Use QUICK_SCENARIOS")
    args = parser.parse_args()

    run_phase6(run_name=args.run, seed=args.seed, quick=args.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
