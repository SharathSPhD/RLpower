"""DynamicMockFMU — first-order lag extension of MockFMU for frequency analysis.

Adds realistic dynamic (time-constant) behaviour to the static linearised
MockFMU so that PRBS-based frequency response estimation yields meaningful
Bode plots with rolloff and phase lag rather than a flat constant response.

This module is for analysis / demonstration only.  Do NOT use in integration
tests (RULE-C1 still applies to MockFMU).
"""
from __future__ import annotations

import math
from typing import Sequence

from sco2rl.simulation.fmu.mock_fmu import MockFMU


# Default first-order time constants (seconds) for sCO₂ cycle variables.
# Based on approximate dynamic models from literature (simple recuperated cycle,
# 10 MW rated, 5 s communication step).
_DEFAULT_TAU: dict[str, float] = {
    "T_turbine_inlet": 60.0,       # Turbine inlet T — slow thermal lag
    "T_compressor_inlet": 30.0,    # Compressor inlet — faster cooling loop
    "P_high": 30.0,                # High-side pressure — inventory dynamics
    "P_low": 30.0,
    "mdot_turbine": 20.0,          # Mass flow — fast IGV response
    "W_turbine": 20.0,
    "W_main_compressor": 20.0,
    "W_net": 25.0,                 # Net power — combination of above
    "eta_thermal": 60.0,           # Efficiency — follows temperature changes
    "surge_margin_main": 20.0,
    "surge_margin_recomp": 20.0,
    "T_exhaust_source": 120.0,
    "mdot_exhaust_source": 60.0,
}


class DynamicMockFMU(MockFMU):
    """MockFMU with first-order lag dynamics for frequency response analysis.

    Each output variable y obeys::

        y[k] = α · y[k-1] + (1 − α) · y_ss[k]

    where ``y_ss[k]`` is the static (instantaneous) response from the parent
    sensitivity matrix, and ``α = exp(−Δt / τ)``.

    Parameters
    ----------
    obs_vars, action_vars, design_point, seed, ...:
        Forwarded to ``MockFMU.__init__``.
    time_constants:
        Optional dict mapping variable name → time constant in seconds.
        Merged with ``_DEFAULT_TAU`` (user values override defaults).
    """

    def __init__(
        self,
        obs_vars: Sequence[str],
        action_vars: Sequence[str],
        design_point: dict[str, float],
        seed: int = 42,
        fail_at_step: int | None = None,
        inlet_temp_drift: bool = False,
        obs_noise_std: float = 0.0,
        time_constants: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            obs_vars=obs_vars,
            action_vars=action_vars,
            design_point=design_point,
            seed=seed,
            fail_at_step=fail_at_step,
            inlet_temp_drift=inlet_temp_drift,
            obs_noise_std=obs_noise_std,
        )
        self._tau: dict[str, float] = {**_DEFAULT_TAU, **(time_constants or {})}
        # Tracks the *dynamic* state (y[k-1]) separately from parent's _state
        self._dynamic_state: dict[str, float] = {}

    def initialize(self, start_time: float, stop_time: float, step_size: float) -> None:
        super().initialize(start_time, stop_time, step_size)
        # Initialise dynamic state at design point
        self._dynamic_state = dict(self._state)

    def reset(self) -> None:
        super().reset()
        self._dynamic_state = dict(self._state)

    def do_step(self, current_time: float, step_size: float) -> bool:
        # Save dynamic state before parent overwrites self._state with y_ss
        prev_dynamic = dict(self._dynamic_state)

        # Parent computes y_ss (static sensitivity-based response)
        result = super().do_step(current_time, step_size)

        if result:
            for var in self._obs_vars:
                if var not in self._state:
                    continue
                tau = self._tau.get(var, 20.0)
                alpha = math.exp(-step_size / max(tau, 1e-9))
                y_ss = self._state[var]          # static target
                y_prev = prev_dynamic.get(var, y_ss)
                lagged = alpha * y_prev + (1.0 - alpha) * y_ss
                self._state[var] = lagged
                self._dynamic_state[var] = lagged

        return result
