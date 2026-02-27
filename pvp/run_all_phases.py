"""
PVP — Run All Phases.

Orchestrator that runs Phases 0–4 and 6 sequentially (Phase 5 requires
hardware and is skipped unless --hil-host is provided).

Results are saved under:
    embark-evaluation/evaluation/results/<run_name>/
        phase0_ground_truth/
        phase1_correctness/
        phase2_metric_validation/
        phase3_discriminative/
        phase4_reproducibility/
        phase5_hil/            (only with --hil-host)
        phase6_overhead/
        pvp_summary.json
        pvp_summary.txt

Usage:
    poetry run python embark-evaluation/pvp/run_all_phases.py --run pvp_run1
    poetry run python embark-evaluation/pvp/run_all_phases.py --run pvp_run1 --quick
    poetry run python embark-evaluation/pvp/run_all_phases.py --run pvp_run1 --hil-host 10.42.0.1
    poetry run python embark-evaluation/pvp/run_all_phases.py --run pvp_gpu --phase4-device cuda --phase6-device cuda  # Phase 4+6 on GPU
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
_embark_eval_dir = Path(__file__).resolve().parents[1]
for _p in [str(_repo_root), str(_embark_eval_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pvp.utils.common import (
    RESULTS_BASE,
    save_json,
    save_text_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PVP — Run All Phases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", type=str, default=None, help="Run name (default: timestamp)")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--quick", action="store_true", help="Use QUICK_SCENARIOS where applicable")
    parser.add_argument("--hil-host", type=str, default=None, help="Akida board IP for Phase 5")
    parser.add_argument("--hil-port", type=int, default=5000, help="Akida server port")
    parser.add_argument(
        "--phase4-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for Phase 4 (cpu = deterministic; cuda = report bounded σ per SC-4)",
    )
    parser.add_argument(
        "--phase6-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for Phase 6 SNN inference (cuda = GPU timing/feasibility; PI baseline always CPU)",
    )
    parser.add_argument(
        "--skip",
        type=str,
        nargs="*",
        default=[],
        help="Phases to skip (e.g. --skip 0 5)",
    )
    args = parser.parse_args()

    run_name = args.run or time.strftime("%Y%m%d_%H%M%S")
    skip_phases = set(args.skip)

    print("=" * 70)
    print("  PVP — Pipeline Verification Procedure")
    print(f"  Run: {run_name}")
    print(f"  Seed: {args.seed}")
    print(f"  Mode: {'QUICK' if args.quick else 'STANDARD'}")
    print(f"  HIL: {'Yes (' + args.hil_host + ')' if args.hil_host else 'No (Phase 5 skipped)'}")
    print(f"  Skip: {skip_phases if skip_phases else 'None'}")
    print(f"  Phase 4 device: {args.phase4_device}  Phase 6 device: {args.phase6_device}")
    print("=" * 70)

    results_summary: dict[str, dict] = {}
    phase_times: dict[str, float] = {}
    total_start = time.perf_counter()

    # --- Phase 0 ---
    if "0" not in skip_phases:
        print("\n" + "=" * 70)
        print("  PHASE 0 — Ground Truth Calibration")
        print("=" * 70)
        try:
            from pvp.phase0_ground_truth import run_phase0

            t0 = time.perf_counter()
            result = run_phase0(run_name=run_name, seed=args.seed)
            phase_times["phase0"] = time.perf_counter() - t0
            results_summary["phase0"] = {"status": "completed", "time_s": phase_times["phase0"]}
        except Exception as e:
            print(f"  Phase 0 FAILED: {e}")
            traceback.print_exc()
            results_summary["phase0"] = {"status": "failed", "error": str(e)}
    else:
        print("\n  Phase 0: SKIPPED")
        results_summary["phase0"] = {"status": "skipped"}

    # --- Phase 1 ---
    if "1" not in skip_phases:
        print("\n" + "=" * 70)
        print("  PHASE 1 — Correctness Probing")
        print("=" * 70)
        try:
            from pvp.phase1_correctness import run_phase1

            t0 = time.perf_counter()
            result = run_phase1(run_name=run_name, seed=args.seed)
            phase_times["phase1"] = time.perf_counter() - t0
            results_summary["phase1"] = {
                "status": "completed",
                "overall_pass": result.get("overall_pass"),
                "time_s": phase_times["phase1"],
            }
        except Exception as e:
            print(f"  Phase 1 FAILED: {e}")
            traceback.print_exc()
            results_summary["phase1"] = {"status": "failed", "error": str(e)}
    else:
        print("\n  Phase 1: SKIPPED")
        results_summary["phase1"] = {"status": "skipped"}

    # --- Phase 2 ---
    if "2" not in skip_phases:
        print("\n" + "=" * 70)
        print("  PHASE 2 — Metric Validation")
        print("=" * 70)
        try:
            from pvp.phase2_metric_validation import run_phase2

            t0 = time.perf_counter()
            result = run_phase2(run_name=run_name, seed=args.seed)
            phase_times["phase2"] = time.perf_counter() - t0
            results_summary["phase2"] = {
                "status": "completed",
                "overall_pass": result.get("overall_pass"),
                "time_s": phase_times["phase2"],
            }
        except Exception as e:
            print(f"  Phase 2 FAILED: {e}")
            traceback.print_exc()
            results_summary["phase2"] = {"status": "failed", "error": str(e)}
    else:
        print("\n  Phase 2: SKIPPED")
        results_summary["phase2"] = {"status": "skipped"}

    # --- Phase 3 ---
    if "3" not in skip_phases:
        print("\n" + "=" * 70)
        print("  PHASE 3 — Discriminative Power")
        print("=" * 70)
        try:
            from pvp.phase3_discriminative import run_phase3

            t0 = time.perf_counter()
            result = run_phase3(run_name=run_name, seed=args.seed, quick=args.quick)
            phase_times["phase3"] = time.perf_counter() - t0
            results_summary["phase3"] = {
                "status": "completed",
                "overall_pass": result.get("overall_pass"),
                "time_s": phase_times["phase3"],
            }
        except Exception as e:
            print(f"  Phase 3 FAILED: {e}")
            traceback.print_exc()
            results_summary["phase3"] = {"status": "failed", "error": str(e)}
    else:
        print("\n  Phase 3: SKIPPED")
        results_summary["phase3"] = {"status": "skipped"}

    # --- Phase 4 ---
    if "4" not in skip_phases:
        print("\n" + "=" * 70)
        print("  PHASE 4 — Reproducibility")
        print("=" * 70)
        try:
            from pvp.phase4_reproducibility import run_phase4

            t0 = time.perf_counter()
            result = run_phase4(
                run_name=run_name,
                seed=args.seed,
                quick=args.quick,
                device=args.phase4_device,
            )
            phase_times["phase4"] = time.perf_counter() - t0
            results_summary["phase4"] = {
                "status": "completed",
                "overall_pass": result.get("overall_pass"),
                "time_s": phase_times["phase4"],
            }
        except Exception as e:
            print(f"  Phase 4 FAILED: {e}")
            traceback.print_exc()
            results_summary["phase4"] = {"status": "failed", "error": str(e)}
    else:
        print("\n  Phase 4: SKIPPED")
        results_summary["phase4"] = {"status": "skipped"}

    # --- Phase 5 (HIL) ---
    if "5" not in skip_phases and args.hil_host:
        print("\n" + "=" * 70)
        print("  PHASE 5 — HIL Deployment Feasibility")
        print("=" * 70)
        try:
            from pvp.phase5_hil import run_phase5

            t0 = time.perf_counter()
            result = run_phase5(
                host=args.hil_host,
                port=args.hil_port,
                run_name=run_name,
                seed=args.seed,
                quick=args.quick,
            )
            phase_times["phase5"] = time.perf_counter() - t0
            results_summary["phase5"] = {
                "status": "completed",
                "sc6a_pass": result.get("sc6a_pass"),
                "time_s": phase_times["phase5"],
            }
        except Exception as e:
            print(f"  Phase 5 FAILED: {e}")
            traceback.print_exc()
            results_summary["phase5"] = {"status": "failed", "error": str(e)}
    else:
        reason = "no --hil-host" if "5" not in skip_phases else "skipped"
        print(f"\n  Phase 5: SKIPPED ({reason})")
        results_summary["phase5"] = {"status": "skipped", "reason": reason}

    # --- Phase 6 ---
    if "6" not in skip_phases:
        print("\n" + "=" * 70)
        print("  PHASE 6 — Overhead Profiling")
        print("=" * 70)
        try:
            from pvp.phase6_overhead import run_phase6

            t0 = time.perf_counter()
            result = run_phase6(
                run_name=run_name,
                seed=args.seed,
                quick=args.quick,
                device=args.phase6_device,
            )
            phase_times["phase6"] = time.perf_counter() - t0
            results_summary["phase6"] = {
                "status": "completed",
                "feasible": result.get("feasible"),
                "time_s": phase_times["phase6"],
            }
        except Exception as e:
            print(f"  Phase 6 FAILED: {e}")
            traceback.print_exc()
            results_summary["phase6"] = {"status": "failed", "error": str(e)}
    else:
        print("\n  Phase 6: SKIPPED")
        results_summary["phase6"] = {"status": "skipped"}

    # --- Summary ---
    total_time = time.perf_counter() - total_start
    results_summary["total_time_s"] = total_time  # type: ignore[assignment]

    print("\n" + "=" * 70)
    print("  PVP SUMMARY")
    print("=" * 70)

    summary_lines: list[str] = [
        "PVP — Pipeline Verification Procedure Summary",
        f"Run: {run_name}",
        f"Total time: {total_time:.1f} s ({total_time / 60:.1f} min)",
        "",
        f"{'Phase':<30s} {'Status':<12s} {'Pass':>6s} {'Time':>10s}",
        "-" * 60,
    ]

    for phase_key in ["phase0", "phase1", "phase2", "phase3", "phase4", "phase5", "phase6"]:
        info = results_summary.get(phase_key, {})
        status = info.get("status", "unknown")
        pass_val = info.get("overall_pass", info.get("feasible", info.get("sc6a_pass", "")))
        time_val = info.get("time_s", "")
        pass_str = str(pass_val) if pass_val is not None and pass_val != "" else "-"
        time_str = f"{time_val:.1f}s" if isinstance(time_val, (int, float)) else "-"
        line = f"  {phase_key:<30s} {status:<12s} {pass_str:>6s} {time_str:>10s}"
        summary_lines.append(line)
        print(line)

    summary_lines.append("-" * 60)
    summary_lines.append(f"  Total: {total_time:.1f} s")

    results_dir = RESULTS_BASE / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(results_summary, results_dir / "pvp_summary.json")
    save_text_report(summary_lines, results_dir / "pvp_summary.txt")

    print(f"\n  Results: {results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
