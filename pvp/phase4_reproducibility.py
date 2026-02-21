"""
PVP Phase 4 — Reproducibility (SC-4).

R6–R8: 3 repeats of `best_incremental_snn` via BenchmarkSuite wrapper,
same seed, same config. STANDARD_SCENARIOS, metrics only.

Full state reset between every repeat and every scenario.

CPU-only pass: sigma = 0.000000 exactly.
GPU runs: report bounded sigma as documented limitation.

Usage:
    poetry run python -m embark-evaluation.evaluation.phase4_reproducibility
    poetry run python -m embark-evaluation.evaluation.phase4_reproducibility --run pvp_run1
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

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


def run_phase4(
    run_name: str | None = None,
    seed: int = 42,
    n_repeats: int = 3,
    quick: bool = False,
) -> dict:
    """Execute Phase 4: reproducibility validation."""
    from embark.benchmark.harness import (
        QUICK_SCENARIOS,
        STANDARD_SCENARIOS,
        BenchmarkSuite,
    )

    scenarios = QUICK_SCENARIOS if quick else STANDARD_SCENARIOS
    results_dir = ensure_results_dir("phase4_reproducibility", run_name)

    # Always test with the best (most complex) model
    best_spec = [m for m in MODELS if m.quality == "best"][0]

    print("=" * 70)
    print("  PVP Phase 4 — Reproducibility (SC-4)")
    print(f"  Model: {best_spec.name}")
    print(f"  Repeats: {n_repeats}, Scenarios: {len(scenarios)}")
    print("=" * 70)

    suite = BenchmarkSuite(scenarios=scenarios, verbose=False)

    # Collect metrics per repeat
    all_run_metrics: list[dict[str, dict[str, Any]]] = []

    for repeat in range(n_repeats):
        run_id = f"R{6 + repeat}"
        print(f"\n  {run_id}: Repeat {repeat + 1}/{n_repeats}...")

        # Full deterministic reset before each repeat
        setup_deterministic(seed)

        controller, meta = build_snn_controller(best_spec, device="cpu")
        t0 = time.perf_counter()
        summary = suite.run(controller=controller, name=f"{best_spec.name}_r{repeat}", quiet=True)
        elapsed = time.perf_counter() - t0
        print(f"    Done in {elapsed:.1f} s")

        run_metrics: dict[str, dict[str, Any]] = {}
        for sr in summary.scenario_results:
            run_metrics[sr.scenario_name] = sr.metrics

        all_run_metrics.append(run_metrics)
        BenchmarkSuite.save_results(summary, results_dir / f"{run_id}_{best_spec.name}.json")

    # --- Compute per-metric sigma across repeats ---
    scenario_names = [s.name for s in scenarios]

    # Discover all metric keys from the first run
    all_metric_keys = set()
    for sn in scenario_names:
        all_metric_keys.update(all_run_metrics[0].get(sn, {}).keys())
    # Filter to numeric metrics
    numeric_keys = sorted(
        k for k in all_metric_keys
        if isinstance(all_run_metrics[0].get(scenario_names[0], {}).get(k), (int, float))
        and k != "steps"
    )

    report_lines: list[str] = [
        "PVP Phase 4 — Reproducibility",
        f"Model: {best_spec.name}",
        f"Repeats: {n_repeats}, Seed: {seed}",
        f"Device: CPU",
        f"Python: {platform.python_version()}",
        f"PyTorch: {torch.__version__}",
        f"OS: {platform.system()} {platform.release()}",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    sigma_table: dict[str, dict[str, float]] = {}
    overall_pass = True

    for sn in scenario_names:
        report_lines.append(f"Scenario: {sn}")
        sigma_table[sn] = {}

        for mk in numeric_keys:
            values = []
            for run in all_run_metrics:
                v = run.get(sn, {}).get(mk)
                if v is not None and isinstance(v, (int, float)) and not np.isnan(v):
                    values.append(float(v))

            if len(values) < 2:
                sigma = 0.0
            else:
                sigma = float(np.std(values, ddof=0))

            sigma_table[sn][mk] = sigma
            if sigma > 0.0:
                overall_pass = False

        # Print key metrics
        key_metrics = ["mae_i_q", "mae_i_d", "settling_time_i_q", "overshoot", "total_syops", "mean_sparsity"]
        for mk in key_metrics:
            if mk in sigma_table[sn]:
                sigma = sigma_table[sn][mk]
                status = "EXACT" if sigma == 0.0 else f"sigma={sigma:.2e}"
                line = f"  {mk:<30s}: {status}"
                report_lines.append(line)
                print(f"  {sn} / {mk}: {status}")

        report_lines.append("")

    report_lines.append(f"Overall SC-4: {'PASS (sigma=0 for all)' if overall_pass else 'FAIL (non-zero sigma found)'}")
    print(f"\n  Overall SC-4: {'PASS' if overall_pass else 'FAIL'}")

    # Save
    save_json(sigma_table, results_dir / "phase4_sigma_table.json")
    save_json(
        {"repeats": [rm for rm in all_run_metrics]},
        results_dir / "phase4_all_metrics.json",
    )
    save_text_report(report_lines, results_dir / "phase4_report.txt")

    return {"sigma_table": sigma_table, "overall_pass": overall_pass}


def main() -> int:
    parser = argparse.ArgumentParser(description="PVP Phase 4 — Reproducibility")
    parser.add_argument("--run", type=str, default=None, help="Run name for results directory")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--n-repeats", type=int, default=3, help="Number of repeats")
    parser.add_argument("--quick", action="store_true", help="Use QUICK_SCENARIOS")
    args = parser.parse_args()

    run_phase4(run_name=args.run, seed=args.seed, n_repeats=args.n_repeats, quick=args.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
