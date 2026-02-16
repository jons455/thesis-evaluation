"""
Direct BenchmarkSuite runner for markdown-defined SNN checkpoints.

This script implements the benchmark flow directly (similar to the example):
    - BenchmarkSuite
    - TensorControllerAdapter
    - RateSNN state/action processors
    - One run per controller (PI optional + all SNN models from markdown docs)

Examples:
    python embark-evaluation/run_models_benchmark.py --quick
    poetry run python embark-evaluation/run_models_benchmark.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

# Ensure repo-root modules (e.g. evaluation.*) are importable.
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


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
        help="Directory where BenchmarkSuite JSON results are saved.",
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
        help="Torch device to load SNN checkpoints on (default: auto).",
    )
    parser.add_argument(
        "--skip-pi",
        action="store_true",
        help="Do not run PI baseline.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save one JSON file per controller summary.",
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

    @torch.no_grad()
    def forward(self, observation: torch.Tensor):
        out = self._model(observation)
        if isinstance(out, tuple) and len(out) == 2:
            action_tensor, spike_rate = out
            self._last_info = {"mean_spike_rate": float(spike_rate)}
            return action_tensor
        return out


def _build_snn_controller(
    checkpoint_path: Path,
    device: str,
    n_max_override: float | None = None,
    error_gain_override: float | None = None,
):
    from embark.benchmark.adapters import TensorControllerAdapter
    from embark.benchmark.processors import RateSNNActionProcessor, RateSNNStateProcessor
    from evaluation.analysis.evaluate_rate_snn import load_rate_model, resolve_feature_params

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

    try:
        from embark.benchmark.controllers.neural import SNNControllerWrapper

        wrapped = SNNControllerWrapper(model=model)
    except Exception:
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

    missing = [p for p in model_paths if not p.exists()]
    if missing:
        print("Error: some checkpoint paths from markdown do not exist:")
        for p in missing:
            print(f"  - {p}")
        print("\nUpdate the markdown checkpoint paths or add the missing model files, then retry.")
        return 1

    results_dir = (repo_root / args.plots_dir).resolve()

    print("Discovered checkpoint models:")
    for idx, model_path in enumerate(model_paths, start=1):
        print(f"  {idx}. {model_path}")

    mode = "QUICK_SCENARIOS" if args.quick else "STANDARD_SCENARIOS"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nMode: {mode}")
    print(f"Device: {device}")
    print(f"PI baseline: {'disabled' if args.skip_pi else 'enabled'}")

    if args.dry_run:
        print("\nDry run complete. No benchmark executed.")
        return 0

    # Keep imports local so dry-run works even if benchmark deps are missing.
    try:
        from embark.benchmark import (
            BenchmarkSuite,
            PIControllerAgent,
            PMSMConfig,
            QUICK_SCENARIOS,
            STANDARD_SCENARIOS,
        )
    except Exception as exc:
        print(f"Error: failed importing embark benchmark package: {exc}")
        return 1

    scenarios = QUICK_SCENARIOS if args.quick else STANDARD_SCENARIOS
    suite = BenchmarkSuite(scenarios=scenarios, verbose=True)

    results_dir.mkdir(parents=True, exist_ok=True)
    print("\nRunning direct BenchmarkSuite evaluation...\n")

    if not args.skip_pi:
        print(">>> Running PI baseline")
        pi = PIControllerAgent.from_system_config(PMSMConfig())
        pi_summary = suite.run(controller=pi, name="PI-baseline")
        suite.print_summary(pi_summary)
        if args.save_results:
            pi_path = results_dir / "PI-baseline.json"
            suite.save_results(pi_summary, pi_path)
            print(f"Saved: {pi_path}")
        print()

    for model_path in model_paths:
        controller_name = _controller_name_from_path(model_path)
        print(f">>> Running SNN model: {controller_name}")
        try:
            controller, meta = _build_snn_controller(model_path, device=device)
        except Exception as exc:
            print(f"  [ERROR] Could not build controller for {model_path.name}: {exc}")
            continue

        print(
            "  "
            f"version={meta.get('version')}, "
            f"input={meta.get('input_size')}, "
            f"incremental={meta.get('incremental_output', False)}, "
            f"rate_steps={meta.get('rate_steps')}"
        )
        try:
            summary = suite.run(controller=controller, name=controller_name)
            suite.print_summary(summary)
            if args.save_results:
                output_path = results_dir / f"{controller_name}.json"
                suite.save_results(summary, output_path)
                print(f"Saved: {output_path}")
        except Exception as exc:
            print(f"  [ERROR] Benchmark failed for {controller_name}: {exc}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
