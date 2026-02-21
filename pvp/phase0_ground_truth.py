"""
PVP Phase 0 — Independent Ground Truth Calibration.

Establishes the expected MAE_q ranking using the wrapper-free code path
(evaluate_rate_snn.py). Same GEM environment, benchmark-equivalent scenarios,
same MAE_q definition. Because this is a separate code path from the NeuroBench
wrapper pipeline, PVP Phase 3 discriminative test is non-circular.

Runs: R0a (best), R0b (intermediate), R0c (poor).
Output: reference ranking table (MAE_q per scenario per model).

Success gate: max_m(MAE_m) - min_m(MAE_m) >= 0.10 * max_m(MAE_m) on at least
one scenario. If not, halt — probes lack discriminative spread.

Usage:
    poetry run python -m embark-evaluation.evaluation.phase0_ground_truth
    poetry run python -m embark-evaluation.evaluation.phase0_ground_truth --run pvp_run1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Ensure repo root and embark-evaluation dir are importable
_repo_root = Path(__file__).resolve().parents[2]
_embark_eval_dir = Path(__file__).resolve().parents[1]
for _p in [str(_repo_root), str(_embark_eval_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pvp.utils.common import (
    MODELS,
    ModelSpec,
    ensure_results_dir,
    get_model_path,
    save_json,
    save_text_report,
    setup_deterministic,
)


def _run_phase0_model(
    spec: ModelSpec,
    scenarios: list[dict],
    n_runs: int,
    max_steps: int,
    seed: int,
) -> dict[str, float]:
    """Run one model through all scenarios without the wrapper. Returns {scenario: mae_q}."""
    from embark.benchmark.agents import PIControllerAgent
    from embark.benchmark.tasks.pmsm_current_control import PMSMCurrentControlTask
    from evaluation.analysis.evaluate_rate_snn import (
        TemporalFeatureBuilder,
        compute_metrics,
        load_rate_model,
        resolve_feature_params,
        run_episode,
    )

    checkpoint = get_model_path(spec)
    model, meta = load_rate_model(checkpoint, device="cpu")
    n_max, error_gain = resolve_feature_params(meta, None, None)

    is_v12 = spec.is_incremental
    delta_u_max = float(meta.get("delta_u_max", 0.2))

    results: dict[str, float] = {}

    for scen in scenarios:
        mae_values: list[float] = []
        for r in range(n_runs):
            task = PMSMCurrentControlTask.from_config(
                n_rpm=scen["n_rpm"],
                i_d_ref=scen["i_d_ref"],
                i_q_ref=scen["i_q_ref"],
                max_steps=max_steps,
            )
            i_max = task.physics_engine.config.i_max
            u_max = task.physics_engine.config.u_max

            fb = TemporalFeatureBuilder(
                i_max=i_max,
                n_max=n_max,
                error_gain=error_gain,
                input_size=int(meta.get("input_size", 12)),
                include_references=is_v12,
                include_prev_voltage=is_v12,
                include_derivatives=not is_v12,
            )

            ep = run_episode(
                model=model,
                feature_builder=fb,
                pi_agent=None,
                task=task,
                max_steps=max_steps,
                u_max=u_max,
                seed=seed + r,
                incremental_output=is_v12,
                delta_u_max=delta_u_max,
            )
            met = compute_metrics(ep)
            mae_values.append(met.mae_q)

        results[scen["name"]] = float(np.mean(mae_values))
    return results


def _benchmark_equivalent_scenarios() -> list[dict]:
    """
    Scenarios equivalent to STANDARD_SCENARIOS but expressed as simple dicts
    for the wrapper-free path. Uses single-step scenarios only (the wrapper-free
    path in evaluate_rate_snn uses from_config which creates step references).
    """
    return [
        {"name": "step_low_speed_500rpm_2A", "n_rpm": 500.0, "i_d_ref": 0.0, "i_q_ref": 2.0},
        {"name": "step_mid_speed_1500rpm_2A", "n_rpm": 1500.0, "i_d_ref": 0.0, "i_q_ref": 2.0},
        {"name": "step_high_speed_2500rpm_2A", "n_rpm": 2500.0, "i_d_ref": 0.0, "i_q_ref": 2.0},
    ]


def run_phase0(
    run_name: str | None = None,
    n_runs: int = 3,
    max_steps: int = 3000,
    seed: int = 42,
) -> dict:
    """Execute Phase 0 and return the rankings dict."""
    setup_deterministic(seed)
    results_dir = ensure_results_dir("phase0_ground_truth", run_name)
    scenarios = _benchmark_equivalent_scenarios()

    print("=" * 70)
    print("  PVP Phase 0 — Independent Ground Truth Calibration")
    print(f"  Scenarios: {len(scenarios)}, Runs per scenario: {n_runs}, Max steps: {max_steps}")
    print("=" * 70)

    rankings: dict[str, dict[str, float]] = {}
    report_lines: list[str] = [
        "PVP Phase 0 — Ground Truth Calibration",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Runs per scenario: {n_runs}, Max steps: {max_steps}, Seed: {seed}",
        "",
    ]

    for spec in MODELS:
        run_id = {"best": "R0a", "intermediate": "R0b", "poor": "R0c"}[spec.quality]
        print(f"\n  {run_id}: {spec.name} ({spec.quality})")
        t0 = time.perf_counter()
        mae_per_scenario = _run_phase0_model(spec, scenarios, n_runs, max_steps, seed)
        elapsed = time.perf_counter() - t0

        rankings[spec.name] = mae_per_scenario

        report_lines.append(f"{run_id}: {spec.name} ({spec.quality})")
        for sname, mae in mae_per_scenario.items():
            print(f"    {sname}: MAE_q = {mae:.6f} A")
            report_lines.append(f"  {sname}: MAE_q = {mae:.6f} A")
        report_lines.append(f"  Time: {elapsed:.1f} s")
        report_lines.append("")

    # --- Discriminative spread check ---
    report_lines.append("--- Discriminative Spread Check ---")
    spread_ok = False
    for scen in scenarios:
        maes = [rankings[m.name][scen["name"]] for m in MODELS]
        spread = max(maes) - min(maes)
        threshold = 0.10 * max(maes)
        ok = spread >= threshold
        if ok:
            spread_ok = True
        status = "PASS" if ok else "INCONCLUSIVE"
        line = f"  {scen['name']}: spread={spread:.6f}, threshold={threshold:.6f} -> {status}"
        print(line)
        report_lines.append(line)

    if not spread_ok:
        msg = "  HALT: No scenario has >= 10% spread. Select more degraded probe model."
        print(msg)
        report_lines.append(msg)
    else:
        report_lines.append("  At least one scenario has sufficient spread. PASS.")

    # --- Expected ranking per scenario ---
    report_lines.append("")
    report_lines.append("--- Expected Ranking (lowest MAE_q = best) ---")
    for scen in scenarios:
        ranked = sorted(MODELS, key=lambda m: rankings[m.name][scen["name"]])
        ranking_str = " < ".join(
            f"{m.name}({rankings[m.name][scen['name']]:.4f})" for m in ranked
        )
        report_lines.append(f"  {scen['name']}: {ranking_str}")

    # --- Save ---
    save_json(rankings, results_dir / "phase0_rankings.json")
    save_text_report(report_lines, results_dir / "phase0_report.txt")

    return rankings


def main() -> int:
    parser = argparse.ArgumentParser(description="PVP Phase 0 — Ground Truth Calibration")
    parser.add_argument("--run", type=str, default=None, help="Run name for results directory")
    parser.add_argument("--n-runs", type=int, default=3, help="Runs per model per scenario")
    parser.add_argument("--max-steps", type=int, default=3000, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    args = parser.parse_args()

    run_phase0(
        run_name=args.run,
        n_runs=args.n_runs,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
