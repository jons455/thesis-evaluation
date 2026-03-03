"""
PVP Phase 0 — Ground Truth Calibration.

Establishes MAE_q per model per scenario using the wrapper-free
evaluate_rate_snn code path (PMSMCurrentControlTask + run_episode),
deliberately NOT using BenchmarkSuite. This makes Phase 0 genuinely
independent from Phase 3: a systematic bug in BenchmarkSuite cannot
simultaneously corrupt the ground-truth reference (Phase 0) and the
pipeline under test (Phase 3), so the discriminative test (SC-3) is
non-circular.

Scenarios mirror the three single-step STANDARD_SCENARIOS conditions
(500/1500/2500 rpm, 2 A) implemented directly via PMSMCurrentControlTask
instead of BenchmarkSuite.  Multi-step / four-quadrant / field-weakening
scenarios require the harness API and are covered in Phase 3 only.

Runs: R0a (best), R0b (intermediate), R0c (poor).
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
    ensure_results_dir,
    get_model_path,
    save_json,
    save_text_report,
    setup_deterministic,
)

# Benchmark-equivalent step scenarios: same physical conditions as
# BenchmarkSuite STANDARD_SCENARIOS step_*_2A entries, re-implemented
# via PMSMCurrentControlTask so no BenchmarkSuite code is involved.
PHASE0_SCENARIOS = [
    {
        "name": "step_low_speed_500rpm_2A",
        "n_rpm": 500,
        "i_d_ref": 0.0,
        "i_q_ref": 2.0,
        "desc": "Step 0→2 A @ 500 rpm",
    },
    {
        "name": "step_mid_speed_1500rpm_2A",
        "n_rpm": 1500,
        "i_d_ref": 0.0,
        "i_q_ref": 2.0,
        "desc": "Step 0→2 A @ 1500 rpm",
    },
    {
        "name": "step_high_speed_2500rpm_2A",
        "n_rpm": 2500,
        "i_d_ref": 0.0,
        "i_q_ref": 2.0,
        "desc": "Step 0→2 A @ 2500 rpm",
    },
]

# Match BenchmarkSuite step-scenario episode length: 3000 steps = 0.3 s at 10 kHz.
PHASE0_MAX_STEPS = 3000


def run_phase0(
    run_name: str | None = None,
    seed: int = 42,
    quick: bool = False,
) -> dict:
    """Execute Phase 0 and return the rankings dict.

    Uses evaluate_rate_snn (wrapper-free path) rather than BenchmarkSuite,
    so Phase 0 ground truth is independent of Phase 3's code path.
    """
    from evaluate_rate_snn import evaluate as wf_evaluate

    setup_deterministic(seed)
    max_steps = 500 if quick else PHASE0_MAX_STEPS
    # n_runs=1: the GEM environment is deterministic (no noise), so a single
    # seeded run is reproducible and sufficient for ground-truth ranking.
    n_runs = 1
    results_dir = ensure_results_dir("phase0_ground_truth", run_name)

    print("=" * 70)
    print("  PVP Phase 0 — Ground Truth Calibration (wrapper-free path)")
    print("  Code path: evaluate_rate_snn (NOT BenchmarkSuite)")
    print(f"  Scenarios: {len(PHASE0_SCENARIOS)}, max_steps={max_steps}, Seed: {seed}")
    print("=" * 70)

    models_for_eval = [(spec.name, get_model_path(spec)) for spec in MODELS]

    t0 = time.perf_counter()
    all_metrics = wf_evaluate(
        models=models_for_eval,
        scenarios=PHASE0_SCENARIOS,
        n_runs=n_runs,
        max_steps=max_steps,
        seed=seed,
        safety_limits=None,
        plot_dir=None,
    )
    elapsed = time.perf_counter() - t0

    rankings: dict[str, dict[str, float]] = {}
    report_lines: list[str] = [
        "PVP Phase 0 — Ground Truth Calibration",
        "Code path: evaluate_rate_snn (wrapper-free, NOT BenchmarkSuite)",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Scenarios: {len(PHASE0_SCENARIOS)} step scenarios (benchmark-equivalent)",
        f"Max steps per episode: {max_steps}, n_runs: {n_runs}, Seed: {seed}",
        f"Total elapsed: {elapsed:.1f} s",
        "",
    ]

    run_ids = {"best": "R0a", "intermediate": "R0b", "poor": "R0c"}

    for spec in MODELS:
        run_id = run_ids[spec.quality]
        model_metrics = all_metrics.get(spec.name, {})
        mae_per_scenario: dict[str, float] = {}

        for scen in PHASE0_SCENARIOS:
            sname = scen["name"]
            mlist = model_metrics.get(sname, [])
            mae_per_scenario[sname] = (
                float(sum(m.mae_q for m in mlist) / len(mlist))
                if mlist
                else float("nan")
            )

        rankings[spec.name] = mae_per_scenario

        report_lines.append(f"{run_id}: {spec.name} ({spec.quality})")
        for sname, mae in mae_per_scenario.items():
            print(f"    {sname}: MAE_q = {mae:.6f} A")
            report_lines.append(f"  {sname}: MAE_q = {mae:.6f} A")
        report_lines.append("")

    # --- Spread data (raw numbers only, no verdict) ---
    report_lines.append("--- Discriminative Spread (raw data) ---")
    for scen in PHASE0_SCENARIOS:
        sn = scen["name"]
        maes = {m.name: rankings[m.name].get(sn, float("nan")) for m in MODELS}
        valid = {k: v for k, v in maes.items() if v == v}  # filter NaN
        if valid:
            spread = max(valid.values()) - min(valid.values())
            threshold_10pct = 0.10 * max(valid.values())
            vals_str = ", ".join(
                f"{k}={v:.4f}" for k, v in sorted(valid.items(), key=lambda x: x[1])
            )
            report_lines.append(
                f"  {sn}: spread={spread:.6f}, 10%_threshold={threshold_10pct:.6f}"
            )
            report_lines.append(f"    values: {vals_str}")

    # --- Ranking per scenario (raw order, no verdict) ---
    report_lines.append("")
    report_lines.append("--- Ranking per Scenario (ascending MAE_q) ---")
    for scen in PHASE0_SCENARIOS:
        sn = scen["name"]
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
    parser.add_argument("--quick", action="store_true", help="Use 500 steps instead of 3000")
    args = parser.parse_args()

    run_phase0(run_name=args.run, seed=args.seed, quick=args.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
