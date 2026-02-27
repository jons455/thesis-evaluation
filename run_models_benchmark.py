"""
Direct BenchmarkSuite runner for markdown-defined SNN checkpoints.

Runs the BenchmarkSuite strictly sequentially: one model at a time, all
scenarios for that model finish before the next model starts. Uses
STANDARD_SCENARIOS by default; pass --quick for a shorter validation run.

Speed: The SNN forward pass uses the selected device (CUDA if available).
Most runtime is in the physics simulation (CPU). Using --device cuda helps
when the model is on GPU; for faster iteration use --quick.

Examples:
    poetry run python embark-evaluation/run_models_benchmark.py
    poetry run python embark-evaluation/run_models_benchmark.py --quick
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

# Ensure repo root and embark-evaluation/scripts are importable.
_repo_root = Path(__file__).resolve().parents[1]
_scripts_dir = Path(__file__).resolve().parent / "scripts"
for _p in (str(_repo_root), str(_scripts_dir)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


CHECKPOINT_PATTERN = re.compile(r"\*\*Checkpoint:\*\*\s*`([^`]+)`")


def extract_checkpoints_from_markdown(md_path: Path) -> list[str]:
    if not md_path.exists():
        return []
    text = md_path.read_text(encoding="utf-8")
    return CHECKPOINT_PATTERN.findall(text)


def collect_models_from_docs(docs_dir: Path, repo_root: Path) -> list[Path]:
    model_paths: list[Path] = []
    seen: set[str] = set()

    for md_file in sorted(docs_dir.glob("*.md")):
        for checkpoint_str in extract_checkpoints_from_markdown(md_file):
            checkpoint_path = Path(checkpoint_str)
            if not checkpoint_path.is_absolute():
                checkpoint_path = repo_root / checkpoint_path
            checkpoint_path = checkpoint_path.resolve()
            key = str(checkpoint_path).lower()
            if key not in seen:
                seen.add(key)
                model_paths.append(checkpoint_path)

    return model_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run direct BenchmarkSuite evaluation for markdown-defined models."
    )
    parser.add_argument(
        "--models-docs-dir",
        type=str,
        default="embark-evaluation/models_for_evaluation",
        help="Directory that contains markdown files with **Checkpoint:** entries.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="embark-evaluation/models_for_evaluation/plots",
        help="Directory where JSON and report are saved (used when --run is not set).",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        metavar="NAME",
        help="Save results under models_for_evaluation/results/NAME (e.g. run1, run2). Overrides --plots-dir.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use QUICK_SCENARIOS instead of STANDARD_SCENARIOS.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Torch device for SNN checkpoints (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        default=True,
        help="Save JSON per controller and benchmark_report.txt (default: True).",
    )
    parser.add_argument(
        "--no-save-results",
        action="store_false",
        dest="save_results",
        help="Do not save JSON or report file.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        metavar="NAME[,NAME,...]",
        help="Run only these controller names (e.g. v9_v9_no_tanh). Comma-separated. Default: all from markdown.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered models and scenario mode without running benchmark.",
    )
    return parser.parse_args()


def _controller_name_from_path(path: Path) -> str:
    parent = path.parent.name
    stem = path.stem
    if parent.lower() == "incremental":
        grandparent = path.parent.parent.name
        return f"{grandparent}_{parent}_{stem}"
    return f"{parent}_{stem}"


class LocalSNNControllerWrapper:
    """Fallback wrapper when embark's SNNControllerWrapper is unavailable."""

    def __init__(self, model: torch.nn.Module):
        self._model = model
        self._last_info: dict[str, float] = {}

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def last_info(self) -> dict[str, float]:
        return self._last_info

    def reset(self) -> None:
        self._last_info = {}

    def configure(self, *_args, **_kwargs) -> None:
        return None

    def _build_neuromorphic_info_from_rate(
        self, spike_rate: float | torch.Tensor | int
    ) -> dict[str, float]:
        """Estimate benchmark neuromorphic metrics from scalar spike-rate output."""
        if isinstance(spike_rate, torch.Tensor):
            spike_rate_value = float(spike_rate.detach().float().mean().item())
        else:
            spike_rate_value = float(spike_rate)

        # Keep expected range stable even if model returns noisy values.
        spike_rate_value = max(0.0, min(1.0, spike_rate_value))

        fcs = list(getattr(self._model, "fcs", []))
        rate_steps = int(getattr(self._model, "rate_steps", 1))
        num_hidden_neurons = int(sum(int(fc.out_features) for fc in fcs))
        total_spikes = int(round(spike_rate_value * num_hidden_neurons * rate_steps))

        # Simple SyOps approximation: each spike triggers weighted fan-out work.
        fanouts: list[int] = [int(fc.out_features) for fc in fcs[1:]]
        readout = getattr(self._model, "readout", None)
        if readout is not None and hasattr(readout, "out_features"):
            fanouts.append(int(readout.out_features))
        mean_fanout = float(sum(fanouts)) / float(len(fanouts)) if fanouts else 1.0
        syops = int(round(total_spikes * mean_fanout))

        return {
            "mean_spike_rate": spike_rate_value,
            "total_spikes": float(total_spikes),
            "syops": float(syops),
            "sparsity": float(1.0 - spike_rate_value),
        }

    def _normalize_info_dict(self, info: dict[str, Any]) -> dict[str, float]:
        """Normalize various model-specific keys to benchmark accumulator keys."""
        normalized: dict[str, float] = {}

        if "total_spikes" in info:
            normalized["total_spikes"] = float(info["total_spikes"])
        if "syops" in info:
            normalized["syops"] = float(info["syops"])
        elif "total_operations" in info:
            normalized["syops"] = float(info["total_operations"])
        if "sparsity" in info:
            normalized["sparsity"] = float(info["sparsity"])
        elif "overall_sparsity" in info:
            normalized["sparsity"] = float(info["overall_sparsity"])
        if "mean_spike_rate" in info:
            normalized["mean_spike_rate"] = float(info["mean_spike_rate"])

        return normalized

    @torch.no_grad()
    def forward(self, observation: torch.Tensor):
        device = next(self._model.parameters()).device
        observation = observation.to(device)
        out = self._model(observation)
        if isinstance(out, tuple) and len(out) >= 2:
            action_tensor = out[0]
            info_like = out[1]
            if isinstance(info_like, dict):
                self._last_info = self._normalize_info_dict(info_like)
            else:
                self._last_info = self._build_neuromorphic_info_from_rate(info_like)
            return action_tensor
        self._last_info = {}
        return out


def _build_snn_controller(
    checkpoint_path: Path,
    device: str,
    n_max_override: float | None = None,
    error_gain_override: float | None = None,
):
    from embark.benchmark.adapters import TensorControllerAdapter
    from embark.benchmark.processors import RateSNNActionProcessor, RateSNNStateProcessor
    from evaluate_rate_snn import load_rate_model, resolve_feature_params

    model, meta = load_rate_model(checkpoint_path, device=device)
    n_max, error_gain = resolve_feature_params(meta, n_max_override, error_gain_override)

    is_incremental = bool(
        meta.get("incremental_output")
        or meta.get("version") == "v12"
        or int(meta.get("input_size", 12)) == 13
    )

    if is_incremental:
        state_processor = RateSNNStateProcessor(
            include_currents=True,
            include_references=True,
            include_errors=True,
            include_speed=True,
            include_prev_action=True,
            include_derivatives=False,
            include_ema_slow=True,
            include_ema_fast=True,
            error_gain=error_gain,
            n_max=n_max,
        )
        action_processor = RateSNNActionProcessor(
            incremental=True,
            delta_max=float(meta.get("delta_u_max", 0.2))
        )
    else:
        state_processor = RateSNNStateProcessor(
            include_currents=True,
            include_references=False,
            include_errors=True,
            include_speed=True,
            include_prev_action=False,
            include_derivatives=True,
            include_ema_slow=True,
            include_ema_fast=True,
            error_gain=error_gain,
            n_max=n_max,
        )
        action_processor = RateSNNActionProcessor(incremental=False)

    # Use our wrapper so observations are moved to the model device (fixes CPU/CUDA mismatch).
    wrapped = LocalSNNControllerWrapper(model=model)

    controller = TensorControllerAdapter(
        controller=wrapped,
        state_processor=state_processor,
        action_processor=action_processor,
    )
    return controller, meta


def main() -> int:
    args = parse_args()

    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    docs_dir = (repo_root / args.models_docs_dir).resolve()

    if not docs_dir.exists():
        print(f"Error: docs directory not found: {docs_dir}")
        return 1

    model_paths = collect_models_from_docs(docs_dir, repo_root)
    if not model_paths:
        print(f"Error: no checkpoint entries found in markdown files under: {docs_dir}")
        print("Expected markdown lines like: **Checkpoint:** `evaluation/trained_models/v12/incremental/best_model.pt`")
        return 1

    if args.only:
        only_names = {s.strip() for s in args.only.split(",") if s.strip()}
        model_paths = [p for p in model_paths if _controller_name_from_path(p) in only_names]
        if not model_paths:
            print(f"Error: --only {args.only!r} matched no models. Valid names: {[_controller_name_from_path(p) for p in collect_models_from_docs(docs_dir, repo_root)]}")
            return 1

    missing = [p for p in model_paths if not p.exists()]
    if missing:
        print("Error: some checkpoint paths from markdown do not exist:")
        for p in missing:
            print(f"  - {p}")
        print("\nUpdate the markdown checkpoint paths or add the missing model files, then retry.")
        return 1

    if args.run is not None:
        results_dir = (repo_root / "embark-evaluation/models_for_evaluation/results" / args.run).resolve()
    else:
        results_dir = (repo_root / args.plots_dir).resolve()

    print("Discovered checkpoint models:")
    for idx, model_path in enumerate(model_paths, start=1):
        print(f"  {idx}. {model_path}")

    mode = "QUICK_SCENARIOS" if args.quick else "STANDARD_SCENARIOS"
    cuda_available = torch.cuda.is_available()
    device = args.device or ("cuda" if cuda_available else "cpu")
    print(f"\nMode: {mode}")
    print(f"PyTorch CUDA available: {cuda_available}")
    if not cuda_available and args.device != "cpu":
        print("  (Install PyTorch with CUDA: https://pytorch.org — choose CUDA version.)")
    print(f"Device: {device}", flush=True)

    if args.dry_run:
        print("\nDry run complete. No benchmark executed.")
        return 0

    # Keep imports local so dry-run works even if benchmark deps are missing.
    try:
        from embark.benchmark import (
            BenchmarkSuite,
            QUICK_SCENARIOS,
            STANDARD_SCENARIOS,
        )
    except Exception as exc:
        print(f"Error: failed importing embark benchmark package: {exc}")
        return 1

    scenarios = QUICK_SCENARIOS if args.quick else STANDARD_SCENARIOS
    suite = BenchmarkSuite(scenarios=scenarios, verbose=True)

    results_dir.mkdir(parents=True, exist_ok=True)
    print("\nRunning BenchmarkSuite sequentially (one model at a time, all scenarios per model).\n", flush=True)

    report_lines: list[str] = []
    if args.save_results:
        report_lines.append(f"Benchmark report — Mode: {mode}, Device: {device}")
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

    for idx, model_path in enumerate(model_paths, start=1):
        controller_name = _controller_name_from_path(model_path)
        print(f"[{idx}/{len(model_paths)}] >>> SNN model: {controller_name}", flush=True)
        try:
            controller, meta = _build_snn_controller(model_path, device=device)
        except Exception as exc:
            print(f"  [ERROR] Could not build controller for {model_path.name}: {exc}", flush=True)
            continue

        meta_str = (
            f"  version={meta.get('version')}, "
            f"input={meta.get('input_size')}, "
            f"incremental={meta.get('incremental_output', False)}, "
            f"rate_steps={meta.get('rate_steps')}"
        )
        print(meta_str, flush=True)
        try:
            t0 = time.perf_counter()
            summary = suite.run(controller=controller, name=controller_name)
            elapsed = time.perf_counter() - t0
            print(suite.format_summary(summary))
            print(f"  Running time: {elapsed:.2f} s", flush=True)
            if args.save_results:
                report_lines.append(f">>> Running SNN model: {controller_name}")
                report_lines.append(meta_str)
                report_lines.append(suite.format_summary(summary))
                report_lines.append(f"  Running time: {elapsed:.2f} s")
                report_lines.append("")
                output_path = results_dir / f"{controller_name}.json"
                suite.save_results(summary, output_path)
                print(f"  Saved: {output_path}", flush=True)
            print(f"  Done: {controller_name}\n", flush=True)
        except Exception as exc:
            print(f"  [ERROR] Benchmark failed for {controller_name}: {exc}", flush=True)
            if args.save_results:
                report_lines.append(f">>> Running SNN model: {controller_name}")
                report_lines.append(meta_str)
                report_lines.append(f"  [ERROR] Benchmark failed: {exc}")
                report_lines.append("")
            print(f"  Skipped: {controller_name}\n", flush=True)

    if args.save_results and report_lines:
        report_path = results_dir / "benchmark_report.txt"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"Report saved: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
