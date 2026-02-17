"""
Run BenchmarkSuite with TD-HIL: remote Akida board as controller.

Uses the same evaluation/akida/server inference server on the board. The PC runs
the BenchmarkSuite (STANDARD_SCENARIOS or QUICK_SCENARIOS) and sends observations
to the board over TCP; the board runs the .fbz model and returns actions (time-dilated
HIL: simulation waits for each inference, so control quality is independent of latency).

Deployment:
  1. On board: copy inference_server.py and your .fbz to ~/akida_deployment (see AKIDA_BOARD_DEPLOY.md).
  2. On board: python3 server/inference_server.py --host 0.0.0.0 --port 5000 --model-path models/akida_model.fbz --input-shape "1,1,1,5"
  3. On PC: poetry run python embark-evaluation/run_models_benchmark_tdhil.py --host 10.42.0.1 --port 5000

Examples:
    poetry run python embark-evaluation/run_models_benchmark_tdhil.py --host 10.42.0.1 --port 5000
    poetry run python embark-evaluation/run_models_benchmark_tdhil.py --host 10.42.0.1 --port 5000 --quick --run tdhil_run1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Repo root on path for evaluation.* and embark
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BenchmarkSuite with remote Akida controller (TD-HIL).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        required=True,
        help="IP or hostname of Akida device (e.g. 10.42.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="TCP port of inference server",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Connection/timeout in seconds for each request",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use QUICK_SCENARIOS instead of STANDARD_SCENARIOS",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        metavar="NAME",
        help="Save under embark-evaluation/models_for_evaluation/results/NAME",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="embark-evaluation/results/akida",
        help="Results directory when --run is not set (default: embark-evaluation/results/akida)",
    )
    parser.add_argument(
        "--no-pwm",
        action="store_true",
        help="Disable PWM modulation (idealized voltage)",
    )
    parser.add_argument(
        "--error-gain",
        type=float,
        default=10.0,
        help="Error gain (must match training)",
    )
    parser.add_argument(
        "--n-max",
        type=float,
        default=4000.0,
        help="Max speed [RPM] for normalization",
    )
    parser.add_argument(
        "--input-shape",
        type=str,
        default="1,1,-1",
        help="Server input shape (e.g. '1,1,1,5')",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        default=True,
        help="Save JSON and benchmark_report.txt",
    )
    parser.add_argument(
        "--no-save-results",
        action="store_false",
        dest="save_results",
        help="Do not save results",
    )
    parser.add_argument(
        "--controller-name",
        type=str,
        default="akida_tdhil",
        help="Name used in report and output JSON",
    )
    return parser.parse_args()


def _build_remote_akida_controller(args: argparse.Namespace):
    """Build TensorControllerAdapter with RemoteAkidaPolicy and Akida state/action processors."""
    from embark.benchmark.adapters import TensorControllerAdapter
    from embark.benchmark.controllers.remote.akida_policy import RemoteAkidaPolicy
    from evaluation.akida.run_benchmark_remote import (
        AkidaActionProcessor,
        AkidaStateProcessor,
    )

    # Default physics; BenchmarkSuite will call configure() per scenario
    i_max = 10.0
    u_max = 24.0

    remote_policy = RemoteAkidaPolicy(
        host=args.host,
        port=args.port,
        timeout_s=args.timeout,
        output_shape=(2,),
    )
    state_processor = AkidaStateProcessor(
        i_max=i_max,
        n_max=args.n_max,
        error_gain=args.error_gain,
    )
    action_processor = AkidaActionProcessor(
        u_max=u_max,
        enable_pwm=not args.no_pwm,
    )
    controller = TensorControllerAdapter(
        controller=remote_policy,
        state_processor=state_processor,
        action_processor=action_processor,
    )
    return controller


def main() -> int:
    args = parse_args()

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    if args.run is not None:
        results_dir = (
            repo_root / "embark-evaluation/models_for_evaluation/results" / args.run
        ).resolve()
    else:
        results_dir = (repo_root / args.plots_dir).resolve()

    mode = "QUICK_SCENARIOS" if args.quick else "STANDARD_SCENARIOS"
    print("TD-HIL BenchmarkSuite (remote Akida)")
    print(f"  Host: {args.host}:{args.port}")
    print(f"  Mode: {mode}")
    print(f"  Controller name: {args.controller_name}")
    print(f"  Results: {results_dir}")
    print()

    try:
        from embark.benchmark import (
            BenchmarkSuite,
            QUICK_SCENARIOS,
            STANDARD_SCENARIOS,
        )
    except Exception as exc:
        print(f"Error: failed importing embark benchmark: {exc}")
        return 1

    scenarios = QUICK_SCENARIOS if args.quick else STANDARD_SCENARIOS
    suite = BenchmarkSuite(scenarios=scenarios, verbose=True)

    print("Building remote Akida controller...")
    try:
        controller = _build_remote_akida_controller(args)
    except Exception as exc:
        print(f"Error: could not build controller: {exc}")
        print("Ensure the inference server is running on the board (see AKIDA_BOARD_DEPLOY.md).")
        return 1

    results_dir.mkdir(parents=True, exist_ok=True)
    report_lines: list[str] = []
    if args.save_results:
        report_lines.append(f"TD-HIL Benchmark report — Mode: {mode}")
        report_lines.append(f"Host: {args.host}:{args.port}")
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

    print(f"Running BenchmarkSuite for: {args.controller_name}\n", flush=True)
    try:
        t0 = time.perf_counter()
        summary = suite.run(controller=controller, name=args.controller_name)
        elapsed = time.perf_counter() - t0
        print(suite.format_summary(summary))
        print(f"  Running time: {elapsed:.2f} s", flush=True)

        if args.save_results:
            report_lines.append(f">>> Controller: {args.controller_name}")
            report_lines.append(suite.format_summary(summary))
            report_lines.append(f"  Running time: {elapsed:.2f} s")
            report_lines.append("")
            out_path = results_dir / f"{args.controller_name}.json"
            suite.save_results(summary, out_path)
            print(f"  Saved: {out_path}", flush=True)

        # If summary contains latency keys, print a short note
        if isinstance(summary, dict):
            if summary.get("mean_latency_ms") or summary.get("chip_mean_us"):
                print("\n  (Round-trip and on-chip latency are in the summary/JSON.)")
        print(f"  Done: {args.controller_name}\n", flush=True)

    except Exception as exc:
        print(f"  [ERROR] Benchmark failed: {exc}", flush=True)
        if args.save_results:
            report_lines.append(f">>> Controller: {args.controller_name}")
            report_lines.append(f"  [ERROR] {exc}")
            report_lines.append("")
        return 1

    if args.save_results and report_lines:
        report_path = results_dir / "benchmark_report.txt"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"Report saved: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
