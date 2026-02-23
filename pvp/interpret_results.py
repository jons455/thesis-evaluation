"""
PVP Interpret Results — centralized pass/fail verdict engine.

Reads the JSON output files produced by each phase script and applies
all PASS/FAIL/INVESTIGATE thresholds in one place.  Phase scripts
themselves are neutral data collectors; all interpretation lives here.

Thresholds applied:
  Phase 1 (SC-1):  max residual i_q < 1e-12 A → PASS
                   max residual i_q < 1e-6  A → INVESTIGATE
                   max residual i_q ≥ 1e-6  A → HARD FAIL
  Phase 2 (SC-2):  deviation < 1e-10 → EXACT
                   deviation < 1e-3  → INVESTIGATE (minor numerical diff)
                   deviation ≥ 1e-3  → HARD FAIL
                   N/A (nan) metrics → skipped (not a failure)
  Phase 3 (SC-3):  ranking order must match Phase 0 ground truth per scenario
                   MATCH → PASS,  MISMATCH → MISMATCH (informational)
  Phase 0 (SC-0):  discriminative spread ≥ 10 % of the worst model MAE per
                   scenario → PASS
  Phase 4 (SC-4):  sigma = 0.0 → EXACT
                   0 < sigma ≤ 1e-10 → float rounding noise (PASS)
                   sigma > 1e-10    → genuine non-determinism (FAIL)
                   sigma = nan      → all-non-finite (not counted as failure)

Usage:
    poetry run python embark-evaluation/pvp/interpret_results.py
    poetry run python embark-evaluation/pvp/interpret_results.py --run overnight_pvp
    poetry run python embark-evaluation/pvp/interpret_results.py --results-dir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parents[2]
_embark_eval_dir = Path(__file__).resolve().parents[1]
for _p in [str(_repo_root), str(_embark_eval_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pvp.utils.common import ensure_results_dir, save_json, save_text_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    """Load JSON or return None if file missing."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _nan(v: Any) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


# ---------------------------------------------------------------------------
# Per-phase interpretation functions
# ---------------------------------------------------------------------------

def interpret_phase0(results_dir: Path) -> dict:
    """SC-0: discriminative spread ≥ 10% of max MAE per scenario."""
    data = _load_json(results_dir / "phase0_ground_truth" / "phase0_rankings.json")
    if data is None:
        return {"status": "missing", "scenarios": {}}

    scenario_names = list(next(iter(data.values())).keys()) if data else []
    model_names = list(data.keys())

    scenario_verdicts: dict[str, dict] = {}
    overall_pass = True

    for sn in scenario_names:
        maes = {m: data[m].get(sn, float("nan")) for m in model_names}
        valid = {k: v for k, v in maes.items() if not _nan(v)}
        if len(valid) < 2:
            scenario_verdicts[sn] = {"verdict": "N/A", "spread": float("nan"), "threshold": float("nan")}
            continue

        max_mae = max(valid.values())
        min_mae = min(valid.values())
        spread = max_mae - min_mae
        threshold = 0.10 * max_mae

        if spread >= threshold:
            verdict = "PASS"
        else:
            verdict = "FAIL (spread < 10% of max)"
            overall_pass = False

        scenario_verdicts[sn] = {
            "spread": spread,
            "threshold_10pct": threshold,
            "values": dict(sorted(valid.items(), key=lambda x: x[1])),
            "verdict": verdict,
        }

    return {"status": "ok", "overall_pass": overall_pass, "scenarios": scenario_verdicts}


def interpret_phase1(results_dir: Path) -> dict:
    """SC-1: R1 vs R2 residuals must be < 1e-12 A (PASS) or < 1e-6 A (INVESTIGATE)."""
    data = _load_json(results_dir / "phase1_correctness" / "phase1_residuals.json")
    if data is None:
        return {"status": "missing", "scenarios": {}}

    scenario_verdicts: dict[str, dict] = {}
    overall_pass = True

    for sn, res in data.items():
        max_res = res.get("max_residual_iq_A", float("nan"))
        if _nan(max_res):
            verdict = "N/A"
        elif max_res < 1e-12:
            verdict = "PASS (< 1e-12 A)"
        elif max_res < 1e-6:
            verdict = "INVESTIGATE (1e-12 to 1e-6 A)"
        else:
            verdict = "HARD FAIL (≥ 1e-6 A)"
            overall_pass = False

        scenario_verdicts[sn] = {
            "max_residual_iq_A": max_res,
            "mean_residual_iq_A": res.get("mean_residual_iq_A", float("nan")),
            "max_residual_uq_V": res.get("max_residual_uq_V", float("nan")),
            "steps_compared": res.get("steps_compared"),
            "verdict": verdict,
        }

    return {"status": "ok", "overall_pass": overall_pass, "scenarios": scenario_verdicts}


def interpret_phase2(results_dir: Path) -> dict:
    """SC-2: manual vs pipeline deviations. N/A metrics are skipped."""
    data = _load_json(results_dir / "phase2_metric_validation" / "phase2_validation.json")
    if data is None:
        return {"status": "missing", "comparisons": []}

    comparisons = data.get("comparisons", [])
    overall_pass = True
    interpreted: list[dict] = []

    for comp in comparisons:
        dev = comp.get("deviation", float("nan"))
        manual = comp.get("manual", float("nan"))
        pipeline = comp.get("pipeline", float("nan"))

        if _nan(dev) or _nan(manual) or _nan(pipeline):
            verdict = "N/A (nan)"
        elif dev < 1e-10:
            verdict = "EXACT (< 1e-10)"
        elif dev < 1e-3:
            verdict = "INVESTIGATE (< 1e-3)"
        else:
            verdict = "HARD FAIL (≥ 1e-3)"
            overall_pass = False

        interpreted.append({**comp, "verdict": verdict})

    return {
        "status": "ok",
        "overall_pass": overall_pass,
        "scenario": data.get("scenario"),
        "step_onset": data.get("step_onset"),
        "comparisons": interpreted,
    }


def interpret_phase3(results_dir: Path) -> dict:
    """SC-3: ranking per scenario vs Phase 0 ground truth."""
    mae_data = _load_json(results_dir / "phase3_discriminative" / "phase3_mae_table.json")
    p0_data = _load_json(results_dir / "phase0_ground_truth" / "phase0_rankings.json")

    if mae_data is None:
        return {"status": "missing", "scenarios": {}}

    # SNN model names = all keys except PI-baseline
    snn_names = [n for n in mae_data.keys() if n != "PI-baseline"]
    scenario_names = list(next(iter(mae_data.values())).keys()) if mae_data else []

    scenario_verdicts: dict[str, dict] = {}
    match_count = 0
    comparable_count = 0
    overall_pass = True

    for sn in scenario_names:
        p3_ranked = sorted(snn_names, key=lambda n: mae_data.get(n, {}).get(sn, float("inf")))

        if p0_data and sn in (p0_data.get(snn_names[0], {}) if snn_names else {}):
            p0_ranked = sorted(snn_names, key=lambda n: p0_data.get(n, {}).get(sn, float("inf")))
            match = p3_ranked == p0_ranked
            comparable_count += 1
            if match:
                match_count += 1
            else:
                overall_pass = False
            verdict = "MATCH" if match else "MISMATCH"
            scenario_verdicts[sn] = {
                "p0_ranking": p0_ranked,
                "p3_ranking": p3_ranked,
                "verdict": verdict,
            }
        else:
            scenario_verdicts[sn] = {
                "p0_ranking": None,
                "p3_ranking": p3_ranked,
                "verdict": "NO_P0_DATA",
            }

    return {
        "status": "ok",
        "overall_pass": overall_pass if comparable_count > 0 else None,
        "matches": match_count,
        "comparable_scenarios": comparable_count,
        "scenarios": scenario_verdicts,
    }


def interpret_phase4(results_dir: Path) -> dict:
    """SC-4: sigma = 0 → EXACT; 0 < sigma ≤ 1e-10 → float noise (PASS); sigma > 1e-10 → FAIL."""
    data = _load_json(results_dir / "phase4_reproducibility" / "phase4_sigma_table.json")
    if data is None:
        return {"status": "missing", "scenarios": {}}

    SIGMA_NOISE_FLOOR = 1e-10  # float64 rounding noise threshold

    scenario_verdicts: dict[str, dict] = {}
    overall_pass = True

    for sn, metrics in data.items():
        metric_verdicts: dict[str, dict] = {}
        for mk, sigma in metrics.items():
            if _nan(sigma):
                verdict = "N/A (all non-finite)"
            elif sigma == 0.0:
                verdict = "EXACT"
            elif sigma <= SIGMA_NOISE_FLOOR:
                verdict = f"PASS (float noise, sigma={sigma:.2e})"
            else:
                verdict = f"FAIL (sigma={sigma:.2e} > 1e-10)"
                overall_pass = False

            metric_verdicts[mk] = {"sigma": sigma, "verdict": verdict}

        scenario_verdicts[sn] = metric_verdicts

    return {"status": "ok", "overall_pass": overall_pass, "scenarios": scenario_verdicts}


# ---------------------------------------------------------------------------
# Main interpreter
# ---------------------------------------------------------------------------

def interpret_all(results_dir: Path, run_name: str | None = None) -> dict:
    """Run all phase interpretations and emit a combined report."""
    out_dir = ensure_results_dir("interpretation", run_name)

    print("=" * 70)
    print("  PVP Interpret Results")
    print(f"  Source: {results_dir}")
    print(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    report_lines: list[str] = [
        "PVP Interpretation Report",
        f"Source: {results_dir}",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Thresholds:",
        "  Phase 0: spread >= 10% of max MAE per scenario",
        "  Phase 1: max|R1-R2|_iq < 1e-12 A  (INVESTIGATE < 1e-6 A)",
        "  Phase 2: deviation < 1e-10 (EXACT), < 1e-3 (INVESTIGATE), else HARD FAIL",
        "  Phase 3: P3 ranking per scenario must match P0 ground truth",
        "  Phase 4: sigma=0 (EXACT), sigma<=1e-10 (float noise/PASS), else FAIL",
        "",
    ]

    all_results: dict[str, Any] = {}

    # --- Phase 0 ---
    print("\n--- Phase 0: Discriminative Spread ---")
    p0 = interpret_phase0(results_dir)
    all_results["phase0"] = p0
    if p0["status"] == "missing":
        print("  Phase 0: MISSING (no phase0_rankings.json)")
        report_lines += ["Phase 0: MISSING", ""]
    else:
        overall = "PASS" if p0["overall_pass"] else "FAIL"
        print(f"  Overall Phase 0: {overall}")
        report_lines.append(f"Phase 0 — Discriminative Spread: {overall}")
        for sn, sv in p0["scenarios"].items():
            v = sv["verdict"]
            spread = sv.get("spread", float("nan"))
            spread_str = f"{spread:.6f}" if not _nan(spread) else "N/A"
            line = f"  {sn}: spread={spread_str}  -> {v}"
            print(f"  {line}")
            report_lines.append(f"  {line}")
        report_lines.append("")

    # --- Phase 1 ---
    print("\n--- Phase 1: Correctness Probing ---")
    p1 = interpret_phase1(results_dir)
    all_results["phase1"] = p1
    if p1["status"] == "missing":
        print("  Phase 1: MISSING")
        report_lines += ["Phase 1: MISSING", ""]
    else:
        overall = "PASS" if p1["overall_pass"] else "FAIL"
        print(f"  Overall Phase 1: {overall}")
        report_lines.append(f"Phase 1 — Correctness Probing (SC-1): {overall}")
        for sn, sv in p1["scenarios"].items():
            v = sv["verdict"]
            res = sv["max_residual_iq_A"]
            res_str = f"{res:.2e}" if not _nan(res) else "N/A"
            line = f"  {sn}: max|R1-R2|={res_str} A  -> {v}"
            print(f"  {line}")
            report_lines.append(f"  {line}")
        report_lines.append("")

    # --- Phase 2 ---
    print("\n--- Phase 2: Metric Validation ---")
    p2 = interpret_phase2(results_dir)
    all_results["phase2"] = p2
    if p2["status"] == "missing":
        print("  Phase 2: MISSING")
        report_lines += ["Phase 2: MISSING", ""]
    else:
        overall = "PASS" if p2["overall_pass"] else "FAIL"
        print(f"  Overall Phase 2: {overall}")
        report_lines.append(f"Phase 2 — Metric Validation (SC-2): {overall}")
        report_lines.append(f"  Scenario: {p2.get('scenario')}, step onset: {p2.get('step_onset')}")
        header = f"  {'Metric':<20s} {'Deviation':>14s} {'Verdict'}"
        report_lines.append(header)
        for comp in p2["comparisons"]:
            dev = comp.get("deviation", float("nan"))
            dev_str = f"{dev:.2e}" if not _nan(dev) else "N/A"
            line = f"  {comp['metric']:<20s} {dev_str:>14s}   {comp['verdict']}"
            print(f"  {line}")
            report_lines.append(f"  {line}")
        report_lines.append("")

    # --- Phase 3 ---
    print("\n--- Phase 3: Discriminative Power ---")
    p3 = interpret_phase3(results_dir)
    all_results["phase3"] = p3
    if p3["status"] == "missing":
        print("  Phase 3: MISSING")
        report_lines += ["Phase 3: MISSING", ""]
    else:
        overall_pass = p3["overall_pass"]
        overall_str = "PASS" if overall_pass else ("PARTIAL FAIL" if overall_pass is False else "N/A")
        matches = p3["matches"]
        comparable = p3["comparable_scenarios"]
        print(f"  Overall Phase 3: {overall_str} ({matches}/{comparable} scenarios match)")
        report_lines.append(
            f"Phase 3 — Discriminative Power (SC-3): {overall_str} "
            f"({matches}/{comparable} comparable scenarios match)"
        )
        for sn, sv in p3["scenarios"].items():
            v = sv["verdict"]
            p3r = " < ".join(sv["p3_ranking"]) if sv["p3_ranking"] else "—"
            if sv.get("p0_ranking"):
                p0r = " < ".join(sv["p0_ranking"])
                line = f"  {sn}: P0=[{p0r}] | P3=[{p3r}] -> {v}"
            else:
                line = f"  {sn}: P3=[{p3r}] (no P0 data) -> {v}"
            print(f"  {line}")
            report_lines.append(f"  {line}")
        report_lines.append("")

    # --- Phase 4 ---
    print("\n--- Phase 4: Reproducibility ---")
    p4 = interpret_phase4(results_dir)
    all_results["phase4"] = p4
    if p4["status"] == "missing":
        print("  Phase 4: MISSING")
        report_lines += ["Phase 4: MISSING", ""]
    else:
        overall = "PASS" if p4["overall_pass"] else "FAIL"
        print(f"  Overall Phase 4: {overall}")
        report_lines.append(f"Phase 4 — Reproducibility (SC-4): {overall}")
        key_metrics = ["mae_i_q", "mae_i_d", "settling_time_i_q", "overshoot", "total_syops", "mean_sparsity"]
        for sn, metrics in p4["scenarios"].items():
            report_lines.append(f"  Scenario: {sn}")
            for mk in key_metrics:
                if mk in metrics:
                    mv = metrics[mk]
                    line = f"    {mk:<30s}: {mv['verdict']}"
                    report_lines.append(line)
            report_lines.append("")

    # --- Summary ---
    phases_with_pass = ["phase0", "phase1", "phase2", "phase4"]
    phase_names = {"phase0": "P0", "phase1": "P1", "phase2": "P2", "phase3": "P3", "phase4": "P4"}
    summary_lines = []
    for pk in ["phase0", "phase1", "phase2", "phase3", "phase4"]:
        res = all_results.get(pk, {})
        if res.get("status") == "missing":
            summary_lines.append(f"  {phase_names[pk]}: MISSING")
        elif res.get("overall_pass") is True:
            summary_lines.append(f"  {phase_names[pk]}: PASS")
        elif res.get("overall_pass") is False:
            summary_lines.append(f"  {phase_names[pk]}: FAIL")
        else:
            summary_lines.append(f"  {phase_names[pk]}: N/A")

    print("\n" + "=" * 70)
    print("  SUMMARY")
    for sl in summary_lines:
        print(sl)
    print("=" * 70)

    report_lines += ["", "=" * 70, "SUMMARY", *summary_lines, "=" * 70]

    # Save
    save_json(all_results, out_dir / "interpretation.json")
    save_text_report(report_lines, out_dir / "interpretation_report.txt")
    print(f"\n  Saved to {out_dir}")

    return all_results


def main() -> int:
    parser = argparse.ArgumentParser(description="PVP Interpret Results")
    parser.add_argument("--run", type=str, default=None, help="Run name to load results from")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Explicit path to results directory (overrides --run)",
    )
    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        # Default: look in the standard results location
        base = Path(__file__).resolve().parent / "results"
        if args.run:
            results_dir = base / args.run
        else:
            # Find the most recent run directory
            candidates = sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            candidates = [c for c in candidates if c.is_dir()]
            if not candidates:
                print("ERROR: No results directory found. Run the PVP phases first.")
                return 1
            results_dir = candidates[0]
            print(f"  Using most recent results: {results_dir.name}")

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return 1

    interpret_all(results_dir, run_name=args.run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
