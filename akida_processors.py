"""
Akida state and action processors for Phase 5 HIL and run_models_benchmark_tdhil.

Self-contained copy used by embark-evaluation only; no dependency on evaluation.akida.
Embark-evaluation uses these for remote Akida (TensorControllerAdapter + RemoteAkidaPolicy).
"""

from __future__ import annotations

import numpy as np
import torch

from embark.benchmark.interfaces import (
    ActionDict,
    ActionProcessor,
    ClosedLoopTask,
    ReferenceDict,
    StateDict,
    StateProcessor,
    SystemConfig,
)
from embark.benchmark.processors.pwm import PWMConverter


def state_reference_to_input(
    i_d: float,
    i_q: float,
    i_d_ref: float,
    i_q_ref: float,
    n_rpm: float,
    i_max: float,
    n_max: float = 4000.0,
    error_gain: float = 10.0,
) -> np.ndarray:
    """
    Convert physical state + reference to Akida model input vector.

    Returns shape (5,) array: [i_d_norm, i_q_norm, e_d_norm, e_q_norm, n_norm].
    """
    e_d = i_d_ref - i_d
    e_q = i_q_ref - i_q
    i_d_norm = i_d / i_max
    i_q_norm = i_q / i_max
    e_d_norm = np.clip((e_d / i_max) * error_gain, -1.0, 1.0)
    e_q_norm = np.clip((e_q / i_max) * error_gain, -1.0, 1.0)
    n_norm = n_rpm / n_max
    return np.array([i_d_norm, i_q_norm, e_d_norm, e_q_norm, n_norm], dtype=np.float32)


class AkidaStateProcessor(StateProcessor):
    """
    State processor that matches Akida model input format.
    """

    def __init__(self, i_max: float, n_max: float = 4000.0, error_gain: float = 10.0):
        self.i_max = i_max
        self.n_max = n_max
        self.error_gain = error_gain
        self._output_dim = 5

    def configure(
        self, physics_config: SystemConfig, task: ClosedLoopTask
    ) -> None:  # noqa: ARG002
        """Configure with physics limits."""
        self.i_max = getattr(physics_config, "i_max", self.i_max)
        omega_max = getattr(physics_config, "omega_max", None)
        if omega_max is not None:
            self.n_max = omega_max * 60.0 / (2 * np.pi)

    def __call__(self, state: StateDict, reference: ReferenceDict) -> torch.Tensor:
        """Convert state dict to Akida input tensor."""
        i_d = float(state["i_d"])
        i_q = float(state["i_q"])
        i_d_ref = float(reference["i_d_ref"])
        i_q_ref = float(reference["i_q_ref"])
        if "omega" in state:
            n_rpm = float(state["omega"]) * 60.0 / (2 * np.pi)
        else:
            n_rpm = float(state.get("n_rpm", 0.0))
        input_vec = state_reference_to_input(
            i_d,
            i_q,
            i_d_ref,
            i_q_ref,
            n_rpm,
            self.i_max,
            self.n_max,
            self.error_gain,
        )
        return torch.tensor(input_vec, dtype=torch.float32)

    @property
    def output_dim(self) -> int:
        return self._output_dim


class AkidaActionProcessor(ActionProcessor):
    """
    Action processor with PWM modulation for Akida hardware-in-the-loop.

    Converts Akida normalized outputs to physical voltages, then applies
    PWM duty-cycle conversion and dead-time compensation via embark's PWMConverter.
    """

    def __init__(
        self,
        u_max: float,
        v_dc: float | None = None,
        dead_time: float | None = None,
        pwm_frequency: float | None = None,
        enable_pwm: bool = True,
    ):
        self.u_max = u_max
        self._v_dc = v_dc
        self._dead_time = dead_time
        self._pwm_frequency = pwm_frequency
        self.enable_pwm = enable_pwm
        self._pwm: PWMConverter | None = None
        self._last_i_d: float = 0.0
        self._last_i_q: float = 0.0

    def configure(self, physics_config: SystemConfig) -> None:
        """Configure with physics limits and build PWM converter."""
        self.u_max = getattr(physics_config, "u_max", self.u_max)

        if self.enable_pwm:
            v_dc = self._v_dc or getattr(physics_config, "v_dc", self.u_max)
            dead_time = (
                self._dead_time
                if self._dead_time is not None
                else getattr(physics_config, "dead_time", 2.0e-6)
            )
            pwm_freq = (
                self._pwm_frequency
                if self._pwm_frequency is not None
                else getattr(physics_config, "pwm_frequency", 10_000.0)
            )
            self._pwm = PWMConverter(
                v_dc=v_dc,
                pwm_frequency=pwm_freq,
                dead_time=dead_time,
            )

    def set_currents(self, i_d: float, i_q: float) -> None:
        """Feed latest current measurements for dead-time direction."""
        self._last_i_d = i_d
        self._last_i_q = i_q

    def __call__(
        self, action: torch.Tensor, physics_config: SystemConfig
    ) -> ActionDict:  # noqa: ARG002
        """Convert action tensor to physical voltages and PWM duty cycles."""
        action_np = action.detach().cpu().flatten().numpy()

        v_d = float(action_np[0] * self.u_max)
        v_q = float(action_np[1] * self.u_max)

        if self.enable_pwm and self._pwm is not None:
            pwm_result = self._pwm.convert_dq(
                v_d=v_d,
                v_q=v_q,
                i_d=self._last_i_d,
                i_q=self._last_i_q,
            )
            return {
                "v_d": pwm_result["v_d"],
                "v_q": pwm_result["v_q"],
                "duty_d": pwm_result["duty_d"],
                "duty_q": pwm_result["duty_q"],
            }

        return {"v_d": v_d, "v_q": v_q}
