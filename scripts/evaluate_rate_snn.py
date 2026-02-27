"""
Evaluate FeedForwardRate SNN models (v8/v9/v12) against PI baseline.

Canonical copy for embark-evaluation: wrapper-free evaluation used by PVP Phase 0
and for ad-hoc model checks. Feature builders match the training pipeline.
Uses benchmark rate-SNN state/action processors when evaluation.rate_interface
is available (same semantics as the harness).

Run from repo root:
    poetry run python embark-evaluation/scripts/evaluate_rate_snn.py --quick
    poetry run python embark-evaluation/scripts/evaluate_rate_snn.py \\
        --model embark-evaluation/models_for_evaluation/best_incremental_snn/model.pt \\
        embark-evaluation/models_for_evaluation/intermediate_scheduled_sampling/model.pt \\
        embark-evaluation/models_for_evaluation/poor_no_tanh/model.pt \\
        --plots-dir embark-evaluation/plots/rate_snn

See models_for_evaluation/README.md for full usage and model paths.
"""

from __future__ import annotations

import argparse
import math
import sys
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Project imports (repo root: this file lives in embark-evaluation/scripts/)
# ---------------------------------------------------------------------------
_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir.parents[2]  # repo root (scripts -> embark-evaluation -> repo)
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from embark.benchmark.tasks.pmsm_current_control import (  # noqa: E402
    PMSMCurrentControlTask,
    SafetyLimits,
)
# Local fallback for trained-models path (embark.utils.paths was removed in embark)
TRAINED_MODELS_DIR = _project_root / "evaluation" / "trained_models"

# Use benchmark rate-SNN state/action processors when available (same interface as harness)
try:
    from evaluation.rate_interface import (
        RATE_INTERFACE_AVAILABLE,
        get_action_processor_absolute,
        get_action_processor_incremental,
        get_state_processor_for_v9,
        get_state_processor_for_v12,
    )
except ImportError:
    RATE_INTERFACE_AVAILABLE = False
    get_state_processor_for_v9 = None
    get_state_processor_for_v12 = None
    get_action_processor_absolute = None
    get_action_processor_incremental = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Defaults per training version — overridden by checkpoint metadata if present
VERSION_DEFAULTS: dict[str, dict[str, Any]] = {
    "v8": {"n_max": 4000.0, "error_gain": 4.0},
    "v9": {"n_max": 3000.0, "error_gain": 4.0},
    "v10": {"n_max": 3000.0, "error_gain": 4.0},
    "v12": {"n_max": 3000.0, "error_gain": 4.0},
}


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS_FULL = [
    {
        "name": "A_step_pos",
        "n_rpm": 1000,
        "i_d_ref": 0.0,
        "i_q_ref": 2.0,
        "desc": "Step 0→2 A @ 1000 rpm",
    },
    {
        "name": "B_step_neg",
        "n_rpm": 1000,
        "i_d_ref": 0.0,
        "i_q_ref": -2.0,
        "desc": "Step 0→-2 A @ 1000 rpm",
    },
    {
        "name": "C_high_speed",
        "n_rpm": 3000,
        "i_d_ref": 0.0,
        "i_q_ref": 2.0,
        "desc": "Step 0→2 A @ 3000 rpm",
    },
]


# ---------------------------------------------------------------------------
# Feature builder — matches training exactly
# ---------------------------------------------------------------------------


class TemporalFeatureBuilder:
    """
    Builds temporal features for v8/v9 and v12 checkpoints.

    v8/v9 (12 features):
      [i_d, i_q, e_d, e_q, n, de_d, de_q, e_d_ema_s, e_q_ema_s, e_d_ema_f, e_q_ema_f, dn]

    v12 (13 features):
      [i_d, i_q, i_d_ref, i_q_ref, e_d, e_q, n, u_d_prev, u_q_prev, e_d_ema_s, e_q_ema_s, e_d_ema_f, e_q_ema_f]

    """

    def __init__(
        self,
        i_max: float = 10.8,
        n_max: float = 3000.0,
        error_gain: float = 4.0,
        ema_slow_alpha: float = 0.98,
        ema_fast_alpha: float = 0.70,
        clip_factor: float = 1.2,
        input_size: int = 12,
        include_references: bool = False,
        include_prev_voltage: bool = False,
        include_derivatives: bool = True,
    ):
        self.i_max = i_max
        self.n_max = n_max
        self.error_gain = error_gain
        self.ema_slow_alpha = ema_slow_alpha
        self.ema_fast_alpha = ema_fast_alpha
        self.clip_factor = clip_factor
        self.input_size = int(input_size)
        self.include_references = bool(include_references)
        self.include_prev_voltage = bool(include_prev_voltage)
        self.include_derivatives = bool(include_derivatives)
        self._state: dict[str, float] | None = None

    def reset(self) -> None:
        self._state = None

    def __call__(
        self,
        state: dict[str, float],
        reference: dict[str, float],
        prev_action_norm: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        i_d = float(state["i_d"])
        i_q = float(state["i_q"])
        e_d = float(reference["i_d_ref"]) - i_d
        e_q = float(reference["i_q_ref"]) - i_q

        # Clip then normalise (matches training pre-processing)
        clip = self.clip_factor * self.i_max
        i_d = max(-clip, min(clip, i_d))
        i_q = max(-clip, min(clip, i_q))
        e_d = max(-clip, min(clip, e_d))
        e_q = max(-clip, min(clip, e_q))

        i_d_n = i_d / self.i_max
        i_q_n = i_q / self.i_max
        e_d_n = (e_d * self.error_gain) / self.i_max
        e_q_n = (e_q * self.error_gain) / self.i_max

        omega = float(state.get("omega", 0.0))
        n_rpm = omega * 60.0 / (2.0 * math.pi)
        n_n = n_rpm / self.n_max

        if self._state is None:
            self._state = {
                "prev_e_d": e_d_n,
                "prev_e_q": e_q_n,
                "prev_n": n_n,
                "e_d_ema_s": e_d_n,
                "e_q_ema_s": e_q_n,
                "e_d_ema_f": e_d_n,
                "e_q_ema_f": e_q_n,
            }

        s = self._state
        de_d = e_d_n - s["prev_e_d"]
        de_q = e_q_n - s["prev_e_q"]
        dn = n_n - s["prev_n"]

        a_s = self.ema_slow_alpha
        a_f = self.ema_fast_alpha
        s["e_d_ema_s"] = a_s * s["e_d_ema_s"] + (1.0 - a_s) * e_d_n
        s["e_q_ema_s"] = a_s * s["e_q_ema_s"] + (1.0 - a_s) * e_q_n
        s["e_d_ema_f"] = a_f * s["e_d_ema_f"] + (1.0 - a_f) * e_d_n
        s["e_q_ema_f"] = a_f * s["e_q_ema_f"] + (1.0 - a_f) * e_q_n

        s["prev_e_d"] = e_d_n
        s["prev_e_q"] = e_q_n
        s["prev_n"] = n_n

        features: list[float] = [i_d_n, i_q_n]
        if self.include_references:
            features.extend(
                [
                    float(reference["i_d_ref"]) / self.i_max,
                    float(reference["i_q_ref"]) / self.i_max,
                ]
            )
        features.extend([e_d_n, e_q_n, n_n])
        if self.include_derivatives:
            features.extend([de_d, de_q])
        if self.include_prev_voltage:
            if prev_action_norm is None:
                u_d_prev_n = 0.0
                u_q_prev_n = 0.0
            else:
                u_d_prev_n = float(prev_action_norm[0])
                u_q_prev_n = float(prev_action_norm[1])
            features.extend([u_d_prev_n, u_q_prev_n])
        features.extend(
            [s["e_d_ema_s"], s["e_q_ema_s"], s["e_d_ema_f"], s["e_q_ema_f"]]
        )
        if self.include_derivatives:
            features.append(dn)

        if len(features) != self.input_size:
            raise ValueError(
                f"Feature size mismatch: built {len(features)} but expected {self.input_size}"
            )
        return torch.tensor(features, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Output EMA — post-model low-pass filter inspired by v5 multi-timescale readout
# ---------------------------------------------------------------------------


class OutputEMA:
    """
    First-order exponential moving average on the normalised action.

    Acts as a causal low-pass filter between the SNN readout and the plant,
    suppressing the high-frequency spike-noise that destabilises closed-loop
    control.  Equivalent to what the v5 ``MultiTimescaleReadout`` does inside
    the network, but applied as a zero-cost post-processing step so any
    checkpoint can benefit without retraining.

    Parameters
    ----------
    alpha : float
        Smoothing factor in [0, 1).
        0.0 = no smoothing (pass-through).
        0.9 = heavy smoothing (90 % previous output, 10 % new).

    """

    def __init__(self, alpha: float = 0.0):
        self.alpha = float(alpha)
        self._prev: torch.Tensor | None = None

    def reset(self) -> None:
        self._prev = None

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        if self.alpha <= 0.0 or self._prev is None:
            self._prev = action.detach().clone()
            return action
        smoothed = self.alpha * self._prev + (1.0 - self.alpha) * action
        self._prev = smoothed.detach().clone()
        return smoothed


# ---------------------------------------------------------------------------
# Model loading — loads the raw base model without the wrapper overhead
# ---------------------------------------------------------------------------


def load_rate_model(path: Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """
    Load a v8/v9/v12 FeedForwardRateSNN checkpoint.

    Returns (model, metadata) where metadata contains all checkpoint info needed to
    configure the feature builder.

    """
    import snntorch as snn
    from snntorch import surrogate

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]

    input_size = checkpoint.get("input_size", 12)
    hidden_sizes = checkpoint.get("hidden_sizes", [128, 96, 64])
    output_size = checkpoint.get("output_size", 2)
    betas = checkpoint.get("betas", [0.96, 0.90, 0.82])
    rate_steps = checkpoint.get("rate_steps", 20)
    slope = checkpoint.get("slope", 25.0)
    use_tanh = checkpoint.get("use_tanh", True)
    version = checkpoint.get("version", None)
    variant = checkpoint.get("variant", None)

    class _RateSNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.rate_steps = int(rate_steps)
            self.input_size = int(input_size)
            self.use_tanh = use_tanh

            self.fcs = nn.ModuleList()
            self.lifs = nn.ModuleList()
            prev = self.input_size * 2  # dual-population

            spike_grad = surrogate.fast_sigmoid(slope=slope)
            for hs, beta in zip(hidden_sizes, betas):
                self.fcs.append(nn.Linear(prev, hs))
                self.lifs.append(snn.Leaky(beta=beta, spike_grad=spike_grad))
                prev = hs

            self.readout = nn.Linear(prev, output_size)

        def _encode(self, x: torch.Tensor) -> torch.Tensor:
            x_clip = x.clamp(-1.0, 1.0)
            return torch.cat([torch.relu(x_clip), torch.relu(-x_clip)], dim=1)

        @torch.no_grad()
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
            if x.dim() == 1:
                x = x.unsqueeze(0)
            batch = x.shape[0]
            mems = [lif.init_leaky() for lif in self.lifs]
            acc = torch.zeros(batch, self.fcs[-1].out_features, device=x.device)
            spk_total = 0.0

            for _ in range(self.rate_steps):
                h = self._encode(x)
                for i, (fc, lif) in enumerate(zip(self.fcs, self.lifs)):
                    spk, mems[i] = lif(fc(h), mems[i])
                    h = spk
                acc += h
                spk_total += h.abs().mean().item()

            rate = acc / float(self.rate_steps)
            out = self.readout(rate)
            if self.use_tanh:
                out = torch.tanh(out)

            mean_spike_rate = spk_total / float(self.rate_steps)
            return out, mean_spike_rate

    model = _RateSNN()
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    meta = {
        "input_size": input_size,
        "hidden_sizes": hidden_sizes,
        "output_size": output_size,
        "betas": betas,
        "rate_steps": rate_steps,
        "slope": slope,
        "use_tanh": use_tanh,
        "version": version,
        "variant": variant,
        "n_params": sum(p.numel() for p in model.parameters()),
        "path": str(path),
    }

    # Try to read training constants stored in checkpoint
    meta["n_max"] = checkpoint.get("n_max", None)
    meta["error_gain"] = checkpoint.get("error_gain", None)
    meta["bias_diagnostics"] = checkpoint.get("bias_diagnostics", None)
    meta["delta_u_max"] = checkpoint.get("delta_u_max", 0.2)
    meta["incremental_output"] = checkpoint.get("incremental_output", False)

    return model, meta


# ---------------------------------------------------------------------------
# Denormalize action: [-1, 1] -> [-u_max, u_max] volts
# ---------------------------------------------------------------------------


def denormalize_action(action_tensor: torch.Tensor, u_max: float) -> dict[str, float]:
    a = action_tensor.detach().cpu().flatten()
    return {"v_d": float(a[0]) * u_max, "v_q": float(a[1]) * u_max}


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    time: np.ndarray
    i_d: np.ndarray
    i_q: np.ndarray
    i_d_ref: np.ndarray
    i_q_ref: np.ndarray
    v_d: np.ndarray
    v_q: np.ndarray
    terminated_early: bool = False
    violation_reason: str | None = None
    mean_spike_rate: float = 0.0


def run_episode(
    model: nn.Module | None,
    feature_builder: TemporalFeatureBuilder | None,
    pi_agent: Any | None,
    task: PMSMCurrentControlTask,
    max_steps: int,
    u_max: float,
    seed: int = 42,
    output_ema: OutputEMA | None = None,
    incremental_output: bool = False,
    delta_u_max: float = 0.2,
    state_proc: Any = None,
    action_proc: Any = None,
) -> EpisodeResult:
    """
    Run a single closed-loop episode.

    Either (model + feature_builder) for SNN, or pi_agent for PI baseline.
    When state_proc and action_proc are provided (benchmark rate interface),
    they are used instead of feature_builder and denormalize_action.
    ``output_ema``, when provided, smooths the normalised SNN action before
    denormalisation (has no effect on the PI agent).

    """
    is_snn = model is not None
    use_benchmark_processors = (
        is_snn and state_proc is not None and action_proc is not None
    )

    state, reference = task.reset(seed=seed)
    if feature_builder is not None:
        feature_builder.reset()
    if state_proc is not None and hasattr(state_proc, "reset"):
        state_proc.reset()
    if action_proc is not None and hasattr(action_proc, "reset"):
        action_proc.reset()
    if use_benchmark_processors:
        state_proc.configure(task.physics_engine.config, task)
        action_proc.configure(task.physics_engine.config, task)
    if output_ema is not None:
        output_ema.reset()

    times, ids, iqs, id_refs, iq_refs, vds, vqs = [], [], [], [], [], [], []
    spike_rates: list[float] = []
    dt = task.physics_engine.config.tau
    config = task.physics_engine.config
    done = False
    step = 0
    u_prev_d_n = 0.0
    u_prev_q_n = 0.0

    while not done and step < max_steps:
        if is_snn:
            if use_benchmark_processors:
                if incremental_output and hasattr(state_proc, "set_prev_action"):
                    state_proc.set_prev_action(u_prev_d_n, u_prev_q_n)
                obs = state_proc(state, reference)
                action_tensor, sr = model(obs)
                if incremental_output:
                    action_tensor = torch.clamp(action_tensor, -delta_u_max, delta_u_max)
                    if output_ema is not None:
                        action_tensor = output_ema(action_tensor)
                        action_tensor = torch.clamp(action_tensor, -delta_u_max, delta_u_max)
                else:
                    action_tensor = torch.clamp(action_tensor, -1.0, 1.0)
                    if output_ema is not None:
                        action_tensor = output_ema(action_tensor)
                        action_tensor = torch.clamp(action_tensor, -1.0, 1.0)
                action = action_proc(action_tensor, config)
                if incremental_output:
                    u_prev_d_n = action["v_d"] / u_max
                    u_prev_q_n = action["v_q"] / u_max
            else:
                obs = feature_builder(
                    state,
                    reference,
                    prev_action_norm=(u_prev_d_n, u_prev_q_n) if incremental_output else None,
                )
                action_tensor, sr = model(obs)
                if incremental_output:
                    action_tensor = torch.clamp(action_tensor, -delta_u_max, delta_u_max)
                    if output_ema is not None:
                        action_tensor = output_ema(action_tensor)
                        action_tensor = torch.clamp(action_tensor, -delta_u_max, delta_u_max)
                    a = action_tensor.detach().cpu().flatten()
                    u_prev_d_n = float(np.clip(u_prev_d_n + float(a[0]), -1.0, 1.0))
                    u_prev_q_n = float(np.clip(u_prev_q_n + float(a[1]), -1.0, 1.0))
                    action = {"v_d": u_prev_d_n * u_max, "v_q": u_prev_q_n * u_max}
                else:
                    action_tensor = torch.clamp(action_tensor, -1.0, 1.0)
                    if output_ema is not None:
                        action_tensor = output_ema(action_tensor)
                        action_tensor = torch.clamp(action_tensor, -1.0, 1.0)
                    action = denormalize_action(action_tensor, u_max)
            spike_rates.append(sr)
        else:
            action = pi_agent(state, reference)

        next_state, next_ref, done = task.step(action)

        t = next_state.get("time", (step + 1) * dt)
        times.append(t)
        ids.append(next_state["i_d"])
        iqs.append(next_state["i_q"])
        id_refs.append(next_ref["i_d_ref"])
        iq_refs.append(next_ref["i_q_ref"])
        vds.append(action["v_d"])
        vqs.append(action["v_q"])

        state, reference = next_state, next_ref
        step += 1

    terminated_early = (
        task.terminated_by_safety if hasattr(task, "terminated_by_safety") else False
    )
    violation = (
        task.last_violation_reason if hasattr(task, "last_violation_reason") else None
    )

    return EpisodeResult(
        time=np.array(times),
        i_d=np.array(ids),
        i_q=np.array(iqs),
        i_d_ref=np.array(id_refs),
        i_q_ref=np.array(iq_refs),
        v_d=np.array(vds),
        v_q=np.array(vqs),
        terminated_early=terminated_early,
        violation_reason=violation,
        mean_spike_rate=float(np.mean(spike_rates)) if spike_rates else 0.0,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class Metrics:
    rmse_q: float = 0.0
    rmse_d: float = 0.0
    mae_q: float = 0.0
    mae_d: float = 0.0
    max_err_q: float = 0.0
    max_err_d: float = 0.0
    settling_ms: float = 0.0
    overshoot_pct: float = 0.0
    ss_error_q: float = 0.0
    num_steps: int = 0
    mean_spike_rate: float = 0.0


def compute_metrics(ep: EpisodeResult) -> Metrics:
    eq = ep.i_q_ref - ep.i_q
    ed = ep.i_d_ref - ep.i_d

    rmse_q = float(np.sqrt(np.mean(eq**2)))
    rmse_d = float(np.sqrt(np.mean(ed**2)))
    mae_q = float(np.mean(np.abs(eq)))
    mae_d = float(np.mean(np.abs(ed)))
    max_err_q = float(np.max(np.abs(eq)))
    max_err_d = float(np.max(np.abs(ed)))

    # Steady-state error (last 10% of episode)
    n_ss = max(1, len(eq) // 10)
    ss_error_q = float(np.mean(np.abs(eq[-n_ss:])))

    # Settling time (5% band around final reference)
    final_ref = ep.i_q_ref[-1] if len(ep.i_q_ref) > 0 else 0.0
    threshold = 0.05 * abs(final_ref) if abs(final_ref) > 0.01 else 0.05
    settling_step = 0
    for i in range(len(eq) - 1, -1, -1):
        if abs(eq[i]) > threshold:
            settling_step = i + 1
            break
    dt = ep.time[1] - ep.time[0] if len(ep.time) > 1 else 1e-4
    settling_ms = settling_step * dt * 1000.0

    # Overshoot
    overshoot = 0.0
    if abs(final_ref) > 0.01:
        if final_ref > 0:
            overshoot = max(0.0, (np.max(ep.i_q) - final_ref) / abs(final_ref) * 100.0)
        else:
            overshoot = max(0.0, (final_ref - np.min(ep.i_q)) / abs(final_ref) * 100.0)

    return Metrics(
        rmse_q=rmse_q,
        rmse_d=rmse_d,
        mae_q=mae_q,
        mae_d=mae_d,
        max_err_q=max_err_q,
        max_err_d=max_err_d,
        settling_ms=settling_ms,
        overshoot_pct=overshoot,
        ss_error_q=ss_error_q,
        num_steps=len(ep.time),
        mean_spike_rate=ep.mean_spike_rate,
    )


# ---------------------------------------------------------------------------
# Envelope: sliding-window max/min of error (removes high-freq oscillation noise)
# ---------------------------------------------------------------------------


def sliding_envelope(signal: np.ndarray, window_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute upper and lower envelope of a 1D signal using sliding-window max/min.
    Reduces high-frequency visual noise and shows error boundaries.
    """
    n = len(signal)
    if n == 0:
        return np.array([]), np.array([])
    half = max(0, (window_steps - 1) // 2)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        upper[i] = np.nanmax(signal[lo:hi])
        lower[i] = np.nanmin(signal[lo:hi])
    return upper, lower


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_scenario(
    scenario_name: str,
    results: dict[str, list[EpisodeResult]],
    plot_dir: Path,
) -> None:
    """Plot i_q trajectories and voltage outputs for one scenario."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    colors = {"PI": "black"}
    snn_names = [n for n in results if n != "PI"]
    cmap = plt.cm.tab10
    for i, name in enumerate(snn_names):
        colors[name] = cmap(i % 10)

    ref_plotted = False

    for name, episodes in results.items():
        if not episodes:
            continue
        color = colors.get(name, "gray")

        # Stack all runs
        all_iq = np.array([ep.i_q for ep in episodes])
        all_vq = np.array([ep.v_q for ep in episodes])
        t = episodes[0].time

        # Truncate to shortest run length
        min_len = min(len(ep.time) for ep in episodes)
        t = t[:min_len]
        all_iq = all_iq[:, :min_len]
        all_vq = all_vq[:, :min_len]

        mean_iq = np.mean(all_iq, axis=0)
        std_iq = np.std(all_iq, axis=0)
        mean_vq = np.mean(all_vq, axis=0)

        # Plot reference once
        if not ref_plotted:
            ref = episodes[0].i_q_ref[:min_len]
            axes[0].plot(t * 1000, ref, "k--", lw=2, alpha=0.6, label="reference")
            ref_plotted = True

        # i_q trajectory with std band
        axes[0].fill_between(
            t * 1000,
            mean_iq - std_iq,
            mean_iq + std_iq,
            color=color,
            alpha=0.2,
        )
        axes[0].plot(t * 1000, mean_iq, color=color, lw=1.5, label=name)

        # Voltage
        axes[1].plot(t * 1000, mean_vq, color=color, lw=1.0, alpha=0.8, label=name)

    axes[0].set_ylabel("$i_q$ [A]")
    axes[0].set_title(scenario_name, fontweight="bold")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_ylabel("$v_q$ [V]")
    axes[1].set_xlabel("Time [ms]")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = plot_dir / f"{scenario_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


def plot_scenario_envelope(
    scenario_name: str,
    results: dict[str, list[EpisodeResult]],
    plot_dir: Path,
    window_steps: int = 51,
) -> None:
    """
    Plot upper/lower envelope of i_q error (e_q = i_q_ref - i_q) per controller.

    Sliding-window max/min removes high-frequency oscillation noise and shows
    error boundaries, making controllers comparable and highlighting whether
    oscillations are bounded and converging vs growing.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    colors = {"PI": "black"}
    snn_names = [n for n in results if n != "PI"]
    cmap = plt.cm.tab10
    for i, name in enumerate(snn_names):
        colors[name] = cmap(i % 10)

    for name, episodes in results.items():
        if not episodes:
            continue
        color = colors.get(name, "gray")
        min_len = min(len(ep.time) for ep in episodes)
        # e_q = i_q_ref - i_q (per run, then mean across runs)
        all_eq = np.array([ep.i_q_ref[:min_len] - ep.i_q[:min_len] for ep in episodes])
        mean_eq = np.mean(all_eq, axis=0)
        t = episodes[0].time[:min_len]
        upper, lower = sliding_envelope(mean_eq, window_steps)
        ax.fill_between(
            t * 1000,
            lower,
            upper,
            color=color,
            alpha=0.25,
        )
        ax.plot(t * 1000, upper, color=color, lw=1.2, label=name)
        ax.plot(t * 1000, lower, color=color, lw=1.2, linestyle="--", label="_nolegend_")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
    ax.set_ylabel("$e_q$ envelope (i_q,ref − i_q) [A]")
    ax.set_xlabel("Time [ms]")
    ax.set_title(f"{scenario_name} — error envelope (window={window_steps} steps)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_path = plot_dir / f"{scenario_name}_envelope.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


# ---------------------------------------------------------------------------
# Find models
# ---------------------------------------------------------------------------


def find_rate_models(version_dir: Path) -> list[tuple[str, Path]]:
    """Find all .pt files in a version directory (and version/incremental for v12)."""
    if not version_dir.exists():
        return []
    models = []
    seen = set()
    for pt in sorted(version_dir.glob("*.pt")):
        key = (pt.stem, str(pt.resolve()))
        if key not in seen:
            seen.add(key)
            models.append((pt.stem, pt))
    # v12 saves to version_dir/incremental/
    incremental_dir = version_dir / "incremental"
    if incremental_dir.is_dir():
        for pt in sorted(incremental_dir.glob("*.pt")):
            key = (pt.stem, str(pt.resolve()))
            if key not in seen:
                seen.add(key)
                models.append((pt.stem, pt))
    return models


# ---------------------------------------------------------------------------
# Resolve n_max / error_gain for a model
# ---------------------------------------------------------------------------


def resolve_feature_params(
    meta: dict, cli_n_max: float | None, cli_error_gain: float | None
) -> tuple[float, float]:
    """
    Determine n_max and error_gain from (in priority order):

    1. CLI override
    2. Checkpoint metadata
    3. Version defaults
    4. Fallback (3000, 4.0)

    """
    version = meta.get("version")
    defaults = VERSION_DEFAULTS.get(version, {}) if version else {}

    n_max = cli_n_max or meta.get("n_max") or defaults.get("n_max", 3000.0)
    error_gain = (
        cli_error_gain or meta.get("error_gain") or defaults.get("error_gain", 4.0)
    )
    return float(n_max), float(error_gain)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate(
    models: list[tuple[str, Path]],
    scenarios: list[dict],
    n_runs: int = 10,
    max_steps: int = 2000,
    seed: int = 42,
    safety_limits: SafetyLimits | None = None,
    plot_dir: Path | None = None,
    cli_n_max: float | None = None,
    cli_error_gain: float | None = None,
    device: str = "cpu",
    output_ema_alphas: list[float] | None = None,
) -> dict[str, dict[str, list[Metrics]]]:
    """
    Run evaluation for all models across all scenarios.

    Returns: {model_name: {scenario_name: [Metrics, ...]}}

    If output_ema_alphas is provided, each SNN model is evaluated with each
    alpha value, producing separate entries like "model_name (EMA=0.8)".

    """
    from embark.benchmark.agents import PIControllerAgent

    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)

    # Storage for results
    all_metrics: dict[str, dict[str, list[Metrics]]] = {}
    # For plotting: {scenario: {model_name: [EpisodeResult]}}
    plot_data: dict[str, dict[str, list[EpisodeResult]]] = {
        s["name"]: {} for s in scenarios
    }

    # --- PI baseline ---
    print("\n  PI Baseline")
    all_metrics["PI"] = {}
    for scen in scenarios:
        task = PMSMCurrentControlTask.from_config(
            n_rpm=scen["n_rpm"],
            i_d_ref=scen["i_d_ref"],
            i_q_ref=scen["i_q_ref"],
            max_steps=max_steps,
            safety_limits=safety_limits,
        )
        pi = PIControllerAgent.from_system_config(task.physics_engine.config)
        u_max = task.physics_engine.config.u_max

        eps_list = []
        met_list = []
        for r in range(n_runs):
            ep = run_episode(
                model=None,
                feature_builder=None,
                pi_agent=pi,
                task=task,
                max_steps=max_steps,
                u_max=u_max,
                seed=seed + r,
            )
            eps_list.append(ep)
            met_list.append(compute_metrics(ep))

        all_metrics["PI"][scen["name"]] = met_list
        plot_data[scen["name"]]["PI"] = eps_list

        avg = _avg_metrics(met_list)
        print(
            f"    {scen['name']:20s}  RMSE_q={avg.rmse_q:.4f}A  "
            f"MAE_q={avg.mae_q:.4f}A  settle={avg.settling_ms:.1f}ms  "
            f"steps={avg.num_steps}"
        )

    # --- SNN models ---
    # Determine EMA alpha values to sweep (None means no EMA)
    ema_alphas: list[float | None] = [None]
    if output_ema_alphas:
        ema_alphas = list(output_ema_alphas)

    for model_name, model_path in models:
        print(f"\n  {model_name}  ({model_path.name})")
        t0 = _time.perf_counter()

        model, meta = load_rate_model(model_path, device=device)
        n_max, error_gain = resolve_feature_params(meta, cli_n_max, cli_error_gain)
        is_v12 = bool(
            meta.get("incremental_output")
            or meta.get("version") == "v12"
            or int(meta.get("input_size", 12)) == 13
        )
        delta_u_max = float(meta.get("delta_u_max", 0.2))

        diag = meta.get("bias_diagnostics")
        info_parts = [
            f"version={meta.get('version')}",
            f"use_tanh={meta.get('use_tanh')}",
            f"rate_steps={meta.get('rate_steps')}",
            f"n_max={n_max:.0f}",
            f"error_gain={error_gain:.1f}",
            f"incremental={is_v12}",
            f"params={meta['n_params']:,}",
        ]
        if is_v12:
            info_parts.append(f"delta_u_max={delta_u_max:.3f}")
        if diag:
            info_parts.append(
                f"bias_ud={diag.get('bias_ud', 0):.4f} bias_uq={diag.get('bias_uq', 0):.4f}"
            )
        print(f"    Config: {', '.join(info_parts)}")

        # Use benchmark rate-SNN processors when available (same semantics as harness)
        use_benchmark_processors = (
            RATE_INTERFACE_AVAILABLE
            and get_state_processor_for_v9
            and get_state_processor_for_v12
            and get_action_processor_absolute
            and get_action_processor_incremental
        )
        if use_benchmark_processors:
            state_proc = (
                get_state_processor_for_v12(n_max=n_max, error_gain=error_gain)
                if is_v12
                else get_state_processor_for_v9(n_max=n_max, error_gain=error_gain)
            )
            action_proc = (
                get_action_processor_incremental(delta_max=delta_u_max)
                if is_v12
                else get_action_processor_absolute()
            )
        else:
            state_proc = action_proc = None

        # Evaluate with each EMA alpha
        for alpha in ema_alphas:
            if alpha is None:
                variant_name = model_name
                ema_obj = None
            else:
                variant_name = f"{model_name} (EMA={alpha})"
                ema_obj = OutputEMA(alpha=alpha)
                print(f"    [EMA alpha={alpha}]")

            all_metrics[variant_name] = {}

            for scen in scenarios:
                task = PMSMCurrentControlTask.from_config(
                    n_rpm=scen["n_rpm"],
                    i_d_ref=scen["i_d_ref"],
                    i_q_ref=scen["i_q_ref"],
                    max_steps=max_steps,
                    safety_limits=safety_limits,
                )
                i_max = task.physics_engine.config.i_max
                u_max = task.physics_engine.config.u_max

                fb = (
                    None
                    if use_benchmark_processors
                    else TemporalFeatureBuilder(
                        i_max=i_max,
                        n_max=n_max,
                        error_gain=error_gain,
                        input_size=int(meta.get("input_size", 12)),
                        include_references=is_v12,
                        include_prev_voltage=is_v12,
                        include_derivatives=not is_v12,
                    )
                )

                eps_list = []
                met_list = []
                for r in range(n_runs):
                    ep = run_episode(
                        model=model,
                        feature_builder=fb,
                        pi_agent=None,
                        task=task,
                        max_steps=max_steps,
                        u_max=u_max,
                        seed=seed + r,
                        output_ema=ema_obj,
                        incremental_output=is_v12,
                        delta_u_max=delta_u_max,
                        state_proc=state_proc,
                        action_proc=action_proc,
                    )
                    eps_list.append(ep)
                    met_list.append(compute_metrics(ep))

                all_metrics[variant_name][scen["name"]] = met_list
                plot_data[scen["name"]][variant_name] = eps_list

                avg = _avg_metrics(met_list)
                early = sum(1 for m in met_list if m.num_steps < max_steps)
                extra = f"  ({early}/{n_runs} early)" if early > 0 else ""
                print(
                    f"    {scen['name']:20s}  RMSE_q={avg.rmse_q:.4f}A  "
                    f"MAE_q={avg.mae_q:.4f}A  settle={avg.settling_ms:.1f}ms  "
                    f"spike_rate={avg.mean_spike_rate:.3f}  "
                    f"steps={avg.num_steps}{extra}"
                )

        elapsed = _time.perf_counter() - t0
        print(f"    ({elapsed:.1f}s)")

    # --- Plots ---
    if plot_dir is not None:
        print("\n  Generating plots...")
        for scen in scenarios:
            plot_scenario(scen["name"], plot_data[scen["name"]], plot_dir)
            plot_scenario_envelope(
                scen["name"], plot_data[scen["name"]], plot_dir, window_steps=51
            )

    return all_metrics


def _avg_metrics(mlist: list[Metrics]) -> Metrics:
    if not mlist:
        return Metrics()
    return Metrics(
        rmse_q=float(np.mean([m.rmse_q for m in mlist])),
        rmse_d=float(np.mean([m.rmse_d for m in mlist])),
        mae_q=float(np.mean([m.mae_q for m in mlist])),
        mae_d=float(np.mean([m.mae_d for m in mlist])),
        max_err_q=float(np.mean([m.max_err_q for m in mlist])),
        max_err_d=float(np.mean([m.max_err_d for m in mlist])),
        settling_ms=float(np.mean([m.settling_ms for m in mlist])),
        overshoot_pct=float(np.mean([m.overshoot_pct for m in mlist])),
        ss_error_q=float(np.mean([m.ss_error_q for m in mlist])),
        num_steps=int(np.mean([m.num_steps for m in mlist])),
        mean_spike_rate=float(np.mean([m.mean_spike_rate for m in mlist])),
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary(all_metrics: dict[str, dict[str, list[Metrics]]]) -> None:
    """Print a compact comparison table."""
    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)

    scenarios = set()
    for scen_dict in all_metrics.values():
        scenarios.update(scen_dict.keys())
    scenarios = sorted(scenarios)

    for scen in scenarios:
        print(f"\n  {scen}")
        print(
            f"  {'Model':<28s} {'RMSE_q':>8s} {'MAE_q':>8s} {'MAE_d':>8s} "
            f"{'MaxE_q':>8s} {'Settle':>8s} {'OS%':>6s} {'SS_e':>7s} {'Steps':>6s}"
        )
        print("  " + "-" * 96)

        for model_name in all_metrics:
            mlist = all_metrics[model_name].get(scen, [])
            if not mlist:
                continue
            avg = _avg_metrics(mlist)
            print(
                f"  {model_name:<28s} {avg.rmse_q:>8.4f} {avg.mae_q:>8.4f} "
                f"{avg.mae_d:>8.4f} {avg.max_err_q:>8.4f} "
                f"{avg.settling_ms:>7.1f}ms {avg.overshoot_pct:>5.1f}% "
                f"{avg.ss_error_q:>7.4f} {avg.num_steps:>6d}"
            )

    print("\n" + "=" * 100)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate FeedForwardRate SNN (v8/v9/v12) against PI baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="*",
        default=None,
        help="One or more .pt files to evaluate. If omitted, evaluates all .pt in --version dir.",
    )
    parser.add_argument(
        "--all-rate-models",
        action="store_true",
        help="Evaluate all rate-based checkpoints (v8, v9, v10, v12) from this repo's evaluation/trained_models/.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v9",
        help="Version subfolder under evaluation/trained_models/ (default: v9).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 3 runs x 500 steps (default: 10 runs x 2000 steps).",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
        help="Override number of runs per scenario.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max steps per episode.",
    )
    parser.add_argument(
        "--n-max",
        type=float,
        default=None,
        help="Override N_MAX for speed normalization (default: auto from checkpoint/version).",
    )
    parser.add_argument(
        "--error-gain",
        type=float,
        default=None,
        help="Override error gain (default: auto from checkpoint/version).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Base RNG seed (default: 42)."
    )
    parser.add_argument(
        "--disable-safety",
        action="store_true",
        help="Disable safety limits (allows dangerous currents).",
    )
    parser.add_argument(
        "--max-current",
        type=float,
        default=None,
        help="Override max current limit in Amperes (default: 20A).",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Override plot output directory.",
    )
    parser.add_argument(
        "--output-ema",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Apply output EMA smoothing with given alpha(s). "
            "alpha=0.0 is pass-through, alpha=0.9 is heavy smoothing. "
            "Provide multiple values to sweep (e.g., --output-ema 0.0 0.5 0.8)."
        ),
    )
    args = parser.parse_args()

    # --all-rate-models: discover v8, v9, v10, v12 checkpoints in this repo
    if args.all_rate_models:
        base = _project_root / "evaluation" / "trained_models"
        all_pt: list[Path] = []
        for sub in ("v8", "v9", "v10"):
            d = base / sub
            if d.is_dir():
                all_pt.extend(sorted(d.glob("*.pt")))
        v12_inc = base / "v12" / "incremental"
        if v12_inc.is_dir():
            all_pt.extend(sorted(v12_inc.glob("*.pt")))
        if not all_pt:
            print("Error: --all-rate-models found no .pt under evaluation/trained_models/{v8,v9,v10,v12/incremental}")
            return 1
        args.model = [str(p) for p in all_pt]
        print(f"  --all-rate-models: found {len(args.model)} checkpoint(s)")

    # Resolve mode
    if args.quick:
        n_runs = args.n_runs or 3
        max_steps = args.max_steps or 500
    else:
        n_runs = args.n_runs or 10
        max_steps = args.max_steps or 2000

    # Safety
    if args.disable_safety:
        safety = SafetyLimits(
            max_current_a=None, max_voltage_v=None, max_speed_rpm=None
        )
    elif args.max_current is not None:
        safety = SafetyLimits(max_current_a=args.max_current)
    else:
        safety = None  # default 20A

    # Find models
    if args.model and len(args.model) > 0:
        models = []
        for m in args.model:
            model_path = Path(m)
            if not model_path.is_absolute():
                model_path = Path.cwd() / model_path
            if not model_path.exists():
                print(f"Error: model not found: {model_path}")
                return 1
            stem = model_path.stem
            if any(stem == name for name, _ in models):
                stem = f"{stem}_{model_path.parent.name}"
            models.append((stem, model_path))
    else:
        version_dir = TRAINED_MODELS_DIR / args.version
        models = find_rate_models(version_dir)
        if not models:
            print(f"Error: no .pt files in {version_dir}")
            return 1

    # Plot dir: default to this repo's docs/plots; infer version when --model is used
    plot_version = args.version
    if args.model and len(args.model) > 0:
        if len(args.model) > 1:
            plot_version = "comparison"
        else:
            for part in Path(args.model[0]).parts:
                if len(part) > 1 and part[0] == "v" and part[1:].isdigit():
                    plot_version = part
                    break
    if args.no_plots:
        plot_dir = None
    elif args.plots_dir:
        plot_dir = Path(args.plots_dir)
    else:
        plot_dir = _project_root / "docs" / "plots" / f"rate_snn_{plot_version}"

    mode = "QUICK" if args.quick else "FULL"
    print("=" * 70)
    print(f"  Rate SNN Evaluation ({mode}: {n_runs} runs x {max_steps} steps)")
    print(f"  Models: {len(models)} from {args.version}")
    for name, path in models:
        print(f"    - {name}: {path}")
    if args.output_ema:
        print(f"  Output EMA alphas: {args.output_ema}")
    print("=" * 70)

    all_metrics = evaluate(
        models=models,
        scenarios=SCENARIOS_FULL,
        n_runs=n_runs,
        max_steps=max_steps,
        seed=args.seed,
        safety_limits=safety,
        plot_dir=plot_dir,
        cli_n_max=args.n_max,
        cli_error_gain=args.error_gain,
        output_ema_alphas=args.output_ema,
    )

    print_summary(all_metrics)

    if plot_dir:
        print(f"\n  Plots saved to: {plot_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
