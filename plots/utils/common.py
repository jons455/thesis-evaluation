"""
Shared constants, model loading, and result I/O for PVP evaluation.

All phases import from here so paths and model definitions are centralised.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]  # thesis-neuromorphic-controller-benchmark
EMBARK_EVAL_DIR = REPO_ROOT / "embark-evaluation"
RESULTS_BASE = EMBARK_EVAL_DIR / "pvp" / "results"
MODELS_DIR = EMBARK_EVAL_DIR / "models_for_evaluation"

# Ensure repo root and embark-evaluation are on sys.path
for _p in [str(REPO_ROOT), str(EMBARK_EVAL_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Specification for one PVP probe model."""
    name: str
    folder: str          # subfolder under models_for_evaluation/
    quality: str         # "best" | "intermediate" | "poor"
    is_incremental: bool
    version: str         # "v12", "v10", "v9"


MODELS: list[ModelSpec] = [
    ModelSpec(
        name="best_incremental_snn",
        folder="best_incremental_snn",
        quality="best",
        is_incremental=True,
        version="v12",
    ),
    ModelSpec(
        name="intermediate_scheduled_sampling",
        folder="intermediate_scheduled_sampling",
        quality="intermediate",
        is_incremental=False,
        version="v10",
    ),
    ModelSpec(
        name="poor_no_tanh",
        folder="poor_no_tanh",
        quality="poor",
        is_incremental=False,
        version="v9",
    ),
]


def get_model_path(spec: ModelSpec) -> Path:
    """Return the absolute path to a model checkpoint."""
    return MODELS_DIR / spec.folder / "model.pt"


# ---------------------------------------------------------------------------
# Deterministic setup
# ---------------------------------------------------------------------------

def setup_deterministic(seed: int = 42) -> None:
    """Configure PyTorch for deterministic, reproducible runs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# SNN controller builder (reuses run_models_benchmark._build_snn_controller)
# ---------------------------------------------------------------------------

def build_snn_controller(
    spec: ModelSpec,
    device: str = "cpu",
):
    """
    Build a TensorControllerAdapter for a given model spec.

    Returns (controller, metadata_dict).
    """
    from embark.benchmark.adapters import TensorControllerAdapter
    from embark.benchmark.processors import RateSNNActionProcessor, RateSNNStateProcessor
    from evaluation.analysis.evaluate_rate_snn import load_rate_model, resolve_feature_params

    checkpoint_path = get_model_path(spec)
    model, meta = load_rate_model(checkpoint_path, device=device)
    n_max, error_gain = resolve_feature_params(meta, None, None)

    is_incremental = spec.is_incremental

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
            delta_max=float(meta.get("delta_u_max", 0.2)),
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

    # Use the local wrapper from run_models_benchmark for device handling
    from run_models_benchmark import LocalSNNControllerWrapper

    wrapped = LocalSNNControllerWrapper(model=model)
    controller = TensorControllerAdapter(
        controller=wrapped,
        state_processor=state_processor,
        action_processor=action_processor,
    )
    return controller, meta


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def ensure_results_dir(phase_name: str, run_name: str | None = None) -> Path:
    """
    Create and return the results directory for a phase.

    Structure: results/<run_name>/<phase_name>/
    If run_name is None, uses a timestamp.
    """
    if run_name is None:
        run_name = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_BASE / run_name / phase_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _json_default(obj: Any) -> Any:
    """JSON serializer for special float values."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_json(data: dict, path: Path) -> None:
    """Save a dictionary as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)
    print(f"  Saved: {path}")


def save_text_report(lines: list[str], path: Path) -> None:
    """Save a list of lines as a text report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {path}")


# ---------------------------------------------------------------------------
# Phase 0 ranking loader
# ---------------------------------------------------------------------------

def load_phase0_rankings(results_dir: Path) -> dict[str, dict[str, float]]:
    """
    Load Phase 0 ground-truth MAE_q per model per scenario.

    Returns: {model_name: {scenario_name: mae_q_value}}
    """
    p0_path = results_dir.parent / "phase0_ground_truth" / "phase0_rankings.json"
    if not p0_path.exists():
        raise FileNotFoundError(
            f"Phase 0 rankings not found at {p0_path}. Run Phase 0 first."
        )
    with open(p0_path, "r", encoding="utf-8") as f:
        return json.load(f)
