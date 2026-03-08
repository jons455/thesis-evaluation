"""
Rate-SNN state/action processors for embark-evaluation (self-contained, no evaluation package).

Uses embark.benchmark.processors when available (RateSNNStateProcessor, RateSNNActionProcessor,
create_v9_processor, create_v12_processor). If only legacy processor names exist in embark,
uses those. No dependency on the evaluation package.
"""

from __future__ import annotations

from typing import Any

RATE_INTERFACE_AVAILABLE = False
create_v5_processor: Any = None
create_v9_processor: Any = None
create_v12_processor: Any = None
RateSNNStateProcessor: Any = None
RateSNNActionProcessor: Any = None
# Legacy fallback (embark.benchmark.processors.normalizers / .decoders)
_LegacySNNStateProcessor: Any = None
_LegacyLinearActionProcessor: Any = None

try:
    from embark.benchmark.processors import (
        RateSNNActionProcessor as _RateSNNActionProcessor,
        RateSNNStateProcessor as _RateSNNStateProcessor,
        create_v5_processor as _create_v5,
        create_v9_processor as _create_v9,
        create_v12_processor as _create_v12,
    )
    RATE_INTERFACE_AVAILABLE = True
    create_v5_processor = _create_v5
    create_v9_processor = _create_v9
    create_v12_processor = _create_v12
    RateSNNStateProcessor = _RateSNNStateProcessor
    RateSNNActionProcessor = _RateSNNActionProcessor
except ImportError:
    try:
        from embark.benchmark.processors.normalizers import SNNStateProcessor as _LegacySNN
        from embark.benchmark.processors.decoders import LinearActionProcessor as _LegacyLinear
        _LegacySNNStateProcessor = _LegacySNN
        _LegacyLinearActionProcessor = _LegacyLinear
    except ImportError:
        pass


def get_state_processor_for_v5(
    error_gain: float = 3.0,
    n_max: float = 3000.0,
):
    """Return a v5-style state processor (5 features: i_d, i_q, e_d, e_q, n)."""
    if RATE_INTERFACE_AVAILABLE and create_v5_processor is not None:
        return create_v5_processor()
    if _LegacySNNStateProcessor is not None:
        return _LegacySNNStateProcessor(error_gain=error_gain, n_max=n_max)
    raise RuntimeError("No state processor available: install embark with benchmark.processors or processors.normalizers")


def get_state_processor_for_v9(
    error_gain: float = 4.0,
    n_max: float = 3000.0,
):
    """Return a v9-style state processor (12 features, with derivatives and EMAs)."""
    if RATE_INTERFACE_AVAILABLE and create_v9_processor is not None:
        return create_v9_processor()
    if RateSNNStateProcessor is not None:
        return RateSNNStateProcessor(
            include_currents=True,
            include_errors=True,
            include_speed=True,
            include_derivatives=True,
            include_ema_slow=True,
            include_ema_fast=True,
            include_references=False,
            include_prev_action=False,
            error_gain=error_gain,
            n_max=n_max,
        )
    if _LegacySNNStateProcessor is not None:
        return _LegacySNNStateProcessor(error_gain=error_gain, n_max=n_max)
    raise RuntimeError("No state processor available: install embark with benchmark.processors or processors.normalizers")


def get_state_processor_for_v12(
    error_gain: float = 4.0,
    n_max: float = 3000.0,
):
    """Return a v12-style state processor (13 features, refs + prev_action, no derivatives)."""
    if RATE_INTERFACE_AVAILABLE and create_v12_processor is not None:
        return create_v12_processor()
    if RateSNNStateProcessor is not None:
        return RateSNNStateProcessor(
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
    if _LegacySNNStateProcessor is not None:
        return _LegacySNNStateProcessor(error_gain=error_gain, n_max=n_max)
    raise RuntimeError("No state processor available: install embark with benchmark.processors or processors.normalizers")


def get_action_processor_absolute(u_max: float | None = None):
    """Return an action processor for absolute voltage output (v5/v9 style)."""
    if RATE_INTERFACE_AVAILABLE and RateSNNActionProcessor is not None:
        kwargs = {} if u_max is None else {"u_max": u_max}
        return RateSNNActionProcessor(incremental=False, **kwargs)
    if _LegacyLinearActionProcessor is not None:
        return _LegacyLinearActionProcessor(output_keys=["v_d", "v_q"])
    raise RuntimeError("No action processor available: install embark with benchmark.processors or processors.decoders")


def get_action_processor_incremental(delta_max: float = 0.2, u_max: float | None = None):
    """Return an action processor for incremental output (v12 style)."""
    if RATE_INTERFACE_AVAILABLE and RateSNNActionProcessor is not None:
        kwargs = {"incremental": True, "delta_max": delta_max}
        if u_max is not None:
            kwargs["u_max"] = u_max
        return RateSNNActionProcessor(**kwargs)
    if _LegacyLinearActionProcessor is not None:
        return _LegacyLinearActionProcessor(output_keys=["v_d", "v_q"])
    raise RuntimeError("No action processor available: install embark with benchmark.processors or processors.decoders")
