"""
PVP Phase 5 — HIL Deployment Feasibility (SC-6).

R12–R13: Akida model on hardware, 1–2 benchmark scenarios, two passes.
Full state reset between passes.

SC-6a: Hardware repeatability — R13 within tolerance of R12.
SC-6b: Hardware–software agreement — R12 within tolerance of R11 (Akida sim).
SC-6c: Timing characterization — measure and report (non-gating).

This script requires a running Akida inference server on the remote board.
It uses BenchmarkSuite with RemoteAkidaPolicy.

Per-metric tolerance bands:
  MAE:           ±1% of reference value
  ITAE:          ±1% of reference value
  Settling Time: ±0.1 ms absolute floor
  Overshoot:     ±0.5 pp absolute

Usage:
    poetry run python embark-evaluation/pvp/phase5_hil.py --host 10.42.0.1 --port 5000
    poetry run python embark-evaluation/pvp/phase5_hil.py --host 10.42.0.1 --port 5000 --run pvp_run1
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

# Tolerance bands per SC-6
TOLERANCES = {
    "mae_i_q": {"type": "relative", "value": 0.01},
    "mae_i_d": {"type": "relative", "value": 0.01},
    "itae_i_q": {"type": "relative", "value": 0.01},
    "itae_i_d": {"type": "relative", "value": 0.01},
    "settling_time_i_q": {"type": "absolute", "value": 0.0001},  # 0.1 ms
    "overshoot": {"type": "absolute", "value": 0.5},             # 0.5 pp
}


def _within_tolerance(val: float, ref: float, metric_name: str) -> tuple[bool, float]:
    """Check if val is within tolerance band of ref. Returns (pass, deviation)."""
    tol = TOLERANCES.get(metric_name)
    if tol is None:
        return True, 0.0

    if tol["type"] == "relative":
        if abs(ref) < 1e-12:
            return True, 0.0
        band = tol["value"] * abs(ref)
    else:
        band = tol["value"]

    dev = abs(val - ref)
    return dev <= band, dev


def _build_hil_controller(host: str, port: int, timeout: float, error_gain: float, n_max: float):
    """Build the remote Akida controller for HIL runs."""
    from embark.benchmark.adapters import TensorControllerAdapter
    from embark.benchmark.controllers.remote.akida_policy import RemoteAkidaPolicy
    from evaluation.akida.run_benchmark_remote import AkidaActionProcessor, AkidaStateProcessor

    remote_policy = RemoteAkidaPolicy(
        host=host,
        port=port,
        timeout_s=timeout,
        output_shape=(2,),
    )
    state_processor = AkidaStateProcessor(
        i_max=10.0,
        n_max=n_max,
        error_gain=error_gain,
    )
    action_processor = AkidaActionProcessor(u_max=24.0, enable_pwm=True)

    controller = TensorControllerAdapter(
        controller=remote_policy,
        state_processor=state_processor,
        action_processor=action_processor,
    )
    return controller


def run_phase5(
    host: str,
    port: int = 5000,
    timeout: float = 30.0,
    run_name: str | None = None,
    seed: int = 42,
    quick: bool = True,
    error_gain: float = 10.0,
    n_max: float = 4000.0,
) -> dict:
    """Execute Phase 5: HIL deployment feasibility."""
    setup_deterministic(seed)

    from embark.benchmark.harness import QUICK_SCENARIOS, STANDARD_SCENARIOS, BenchmarkSuite

    scenarios = QUICK_SCENARIOS if quick else STANDARD_SCENARIOS
    results_dir = ensure_results_dir("phase5_hil", run_name)

    print("=" * 70)
    print("  PVP Phase 5 — HIL Deployment Feasibility (SC-6)")
    print(f"  Host: {host}:{port}")
    print(f"  Scenarios: {len(scenarios)}")
    print("=" * 70)

    suite = BenchmarkSuite(scenarios=scenarios, verbose=True)

    # --- R12: First HIL pass ---
    print("\n  R12: Akida HIL pass 1...")
    controller_r12 = _build_hil_controller(host, port, timeout, error_gain, n_max)
    t0 = time.perf_counter()
    summary_r12 = suite.run(controller=controller_r12, name="akida_hil_r12", quiet=False)
    elapsed_r12 = time.perf_counter() - t0
    print(BenchmarkSuite.format_summary(summary_r12))
    print(f"    Time: {elapsed_r12:.1f} s")
    BenchmarkSuite.save_results(summary_r12, results_dir / "R12_akida_hil.json")

    # --- R13: Second HIL pass ---
    # Close R12's TCP connection so the server can exit its request loop, print summary, and accept R13.
    # (Otherwise the server stays blocked on recv() and never serves the second connection.)
    if hasattr(controller_r12.controller, "close"):
        controller_r12.controller.close()
    time.sleep(2)  # Let server print summary and return to accept()
    print("\n  R13: Akida HIL pass 2...")
    controller_r13 = _build_hil_controller(host, port, timeout, error_gain, n_max)
    t0 = time.perf_counter()
    summary_r13 = suite.run(controller=controller_r13, name="akida_hil_r13", quiet=False)
    elapsed_r13 = time.perf_counter() - t0
    print(BenchmarkSuite.format_summary(summary_r13))
    print(f"    Time: {elapsed_r13:.1f} s")
    BenchmarkSuite.save_results(summary_r13, results_dir / "R13_akida_hil.json")

    # --- SC-6a: Hardware repeatability (R12 vs R13) ---
    report_lines: list[str] = [
        "PVP Phase 5 — HIL Deployment Feasibility",
        f"Host: {host}:{port}",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "--- SC-6a: Hardware Repeatability (R12 vs R13) ---",
    ]

    sc6a_pass = True
    for sr12, sr13 in zip(summary_r12.scenario_results, summary_r13.scenario_results):
        report_lines.append(f"  Scenario: {sr12.scenario_name}")
        for mk in TOLERANCES:
            v12 = sr12.metrics.get(mk, float("nan"))
            v13 = sr13.metrics.get(mk, float("nan"))
            if np.isnan(v12) or np.isnan(v13):
                continue
            ok, dev = _within_tolerance(v13, v12, mk)
            if not ok:
                sc6a_pass = False
            status = "PASS" if ok else "FAIL"
            line = f"    {mk}: R12={v12:.6f}, R13={v13:.6f}, dev={dev:.6f} -> {status}"
            report_lines.append(line)
            print(line)

    report_lines.append(f"  SC-6a: {'PASS' if sc6a_pass else 'FAIL'}")

    # --- SC-6c: Timing characterization ---
    report_lines.append("")
    report_lines.append("--- SC-6c: Timing Characterization ---")
    for sr in summary_r12.scenario_results:
        m = sr.metrics
        mean_lat = m.get("mean_latency_ms", float("nan"))
        p95_lat = m.get("p95_latency_ms", float("nan"))
        chip_mean = m.get("chip_mean_us", float("nan"))

        line = (
            f"  {sr.scenario_name}: "
            f"round-trip={mean_lat:.3f} ms (p95={p95_lat:.3f}), "
            f"chip={chip_mean:.1f} us"
        )
        report_lines.append(line)
        print(line)

        timestep_ms = 0.1  # 10 kHz = 0.1 ms
        if not np.isnan(mean_lat) and mean_lat > timestep_ms:
            report_lines.append(f"    NOTE: round-trip ({mean_lat:.3f} ms) > control timestep ({timestep_ms} ms)")

    report_lines.append("  SC-6c: Reported (non-gating)")

    # --- Overall ---
    report_lines.append("")
    report_lines.append(
        "NOTE: SC-6b (hardware-software agreement) requires R11 (Akida sim) data. "
        "Compare R12 metrics against R11 manually or in a separate script."
    )
    report_lines.append(f"\nOverall Phase 5: SC-6a={'PASS' if sc6a_pass else 'FAIL'}, SC-6c=Reported")

    save_text_report(report_lines, results_dir / "phase5_report.txt")
    save_json(
        {
            "R12": summary_r12.to_dict(),
            "R13": summary_r13.to_dict(),
            "sc6a_pass": sc6a_pass,
        },
        results_dir / "phase5_results.json",
    )

    return {"sc6a_pass": sc6a_pass}


def main() -> int:
    parser = argparse.ArgumentParser(description="PVP Phase 5 — HIL Deployment Feasibility")
    parser.add_argument("--host", type=str, required=True, help="Akida board IP")
    parser.add_argument("--port", type=int, default=5000, help="Akida server port")
    parser.add_argument("--timeout", type=float, default=30.0, help="TCP timeout (s)")
    parser.add_argument("--run", type=str, default=None, help="Run name for results directory")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--quick", action="store_true", default=True, help="Use QUICK_SCENARIOS (default)")
    parser.add_argument("--full", action="store_true", help="Use STANDARD_SCENARIOS")
    parser.add_argument("--error-gain", type=float, default=10.0, help="Error gain for Akida")
    parser.add_argument("--n-max", type=float, default=4000.0, help="N_max for Akida normalization")
    args = parser.parse_args()

    run_phase5(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        run_name=args.run,
        seed=args.seed,
        quick=not args.full,
        error_gain=args.error_gain,
        n_max=args.n_max,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
