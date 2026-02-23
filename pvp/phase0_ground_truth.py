"""
PVP Phase 0 — Ground Truth Calibration.

Establishes the MAE_q per model per scenario using the same BenchmarkSuite
code path as Phase 3. This ensures Phase 0 and Phase 3 are directly comparable
(no code-path mismatch between evaluate_rate_snn.py and the wrapper pipeline).

Runs: R0a (best), R0b (intermediate), R0c (poor) through STANDARD_SCENARIOS.
Output: phase0_rankings.json — MAE_q per scenario per model.
        phase0_report.txt  — raw data, no pass/fail verdicts.

Spread check and ranking verdicts are in interpret_results.py.

Usage:
    poetry run python embark-evaluation/pvp/phase0_ground_truth.py
    poetry run python embark-evaluation/pvp/phase0_ground_truth.py --run pvp_run1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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


def run_phase0(
    run_name: str | None = None,
    seed: int = 42,
    quick: bool = False,
) -> dict:
    """Execute Phase 0 and return the rankings dict.

    Uses BenchmarkSuite (same code path as Phase 3) so that Phase 0 ground
    truth values are directly comparable to Phase 3 results.
    """
    from embark.benchmark.harness import QUICK_SCENARIOS, STANDARD_SCENARIOS, BenchmarkSuite
    from embark.benchmark.physics.config import PMSMConfig

    setup_deterministic(seed)
    scenarios = QUICK_SCENARIOS if quick else STANDARD_SCENARIOS
    results_dir = ensure_results_dir("phase0_ground_truth", run_name)
    pmsm_config = PMSMConfig()

    print("=" * 70)
    print("  PVP Phase 0 — Ground Truth Calibration (BenchmarkSuite path)")
    print(f"  Scenarios: {len(scenarios)} ({'QUICK' if quick else 'STANDARD'}), Seed: {seed}")
    print("=" * 70)

    suite = BenchmarkSuite(scenarios=scenarios, verbose=False)
    rankings: dict[str, dict[str, float]] = {}
    report_lines: list[str] = [
        "PVP Phase 0 — Ground Truth Calibration",
        f"Code path: BenchmarkSuite (same as Phase 3)",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Scenarios: {len(scenarios)}, Seed: {seed}",
        "",
    ]

    run_ids = {"best": "R0a", "intermediate": "R0b", "poor": "R0c"}

    for spec in MODELS:
        run_id = run_ids[spec.quality]
        print(f"\n  {run_id}: {spec.name} ({spec.quality})")
        t0 = time.perf_counter()

        setup_deterministic(seed)
        controller, _ = build_snn_controller(spec, device="cpu")
        summary = suite.run(controller=controller, name=spec.name, quiet=True)
        elapsed = time.perf_counter() - t0

        mae_per_scenario: dict[str, float] = {}
        for sr in summary.scenario_results:
            mae_per_scenario[sr.scenario_name] = sr.metrics.get("mae_i_q", float("nan"))

        rankings[spec.name] = mae_per_scenario

        report_lines.append(f"{run_id}: {spec.name} ({spec.quality})  [{elapsed:.1f} s]")
        for sname, mae in mae_per_scenario.items():
            print(f"    {sname}: MAE_q = {mae:.6f} A")
            report_lines.append(f"  {sname}: MAE_q = {mae:.6f} A")
        report_lines.append("")

    # --- Spread data (raw numbers only, no verdict) ---
    report_lines.append("--- Discriminative Spread (raw data) ---")
    scenario_names = [s.name for s in scenarios]
    for sn in scenario_names:
        maes = {m.name: rankings[m.name].get(sn, float("nan")) for m in MODELS}
        valid = {k: v for k, v in maes.items() if not (v != v)}  # filter NaN
        if valid:
            spread = max(valid.values()) - min(valid.values())
            threshold_10pct = 0.10 * max(valid.values())
            vals_str = ", ".join(f"{k}={v:.4f}" for k, v in sorted(valid.items(), key=lambda x: x[1]))
            report_lines.append(f"  {sn}: spread={spread:.6f}, 10%_threshold={threshold_10pct:.6f}")
            report_lines.append(f"    values: {vals_str}")

    # --- Ranking per scenario (raw order, no verdict) ---
    report_lines.append("")
    report_lines.append("--- Ranking per Scenario (ascending MAE_q) ---")
    for sn in scenario_names:
        ranked = sorted(MODELS, key=lambda m: rankings[m.name].get(sn, float("inf")))
        ranking_str = " < ".join(
            f"{m.name}({rankings[m.name].get(sn, float('nan')):.4f})" for m in ranked
        )
        report_lines.append(f"  {sn}: {ranking_str}")

    # --- Save ---
    save_json(rankings, results_dir / "phase0_rankings.json")
    save_text_report(report_lines, results_dir / "phase0_report.txt")

    print(f"\n  Saved phase0_rankings.json to {results_dir}")
    return rankings


def main() -> int:
    parser = argparse.ArgumentParser(description="PVP Phase 0 — Ground Truth Calibration")
    parser.add_argument("--run", type=str, default=None, help="Run name for results directory")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--quick", action="store_true", help="Use QUICK_SCENARIOS")
    args = parser.parse_args()

    run_phase0(run_name=args.run, seed=args.seed, quick=args.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
