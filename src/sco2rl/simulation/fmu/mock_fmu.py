"""MockFMU — deterministic linearized sCO₂ physics for unit testing.

Implements FMUInterface with:
- Linearized design-point sensitivities (no real thermodynamics)
- Fixed RNG seed for deterministic tests
- Configurable failure injection: fail_at_step, inlet_temp_drift, obs_noise_std
- Call tracking: n_steps, n_resets

RULE-C1: MockFMU MUST NOT be used in integration tests.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from sco2rl.simulation.fmu.interface import FMUInterface


class MockFMU(FMUInterface):
    """Linearized sCO₂ physics mock for isolated unit testing.

    All physics is first-order linearized around the design point.
    Sensitivities encode qualitative direction and order of magnitude only —
    sufficient to test control logic, not for thermodynamic accuracy.
    """

    # Linearized sensitivities: action → output effect per unit action change
    # Rows: [T_turbine_inlet, T_compressor_inlet, P_high, P_low,
    #        mdot_turbine, W_turbine, W_main_compressor, W_net,
    #        eta_thermal, surge_margin_main, T_exhaust_source, mdot_exhaust_source]
    # Cols: [bypass_valve, igv, inventory_valve, cooling_flow]
    _SENSITIVITY = np.array([
        # bypass  igv    inv   cool
        [-10.0,   5.0,   2.0,  0.0],   # T_turbine_inlet (°C)
        [  0.0,   0.0,   0.5, -3.0],   # T_compressor_inlet (°C)  cool↑ → T↓
        [ -1.0,  -0.5,  -0.8,  0.0],   # P_high (MPa)
        [ -0.2,  -0.1,  -0.3,  0.0],   # P_low (MPa)
        [ -8.0,   4.0,   1.0,  0.0],   # mdot_turbine (kg/s)
        [ -4.0,   2.5,   0.5,  0.0],   # W_turbine (MW)
        [  0.0,   0.5,   0.0,  0.0],   # W_main_compressor (MW)
        [ -4.0,   2.0,   0.5,  0.0],   # W_net (MW)  ≈ W_turb - W_comp
        [ -0.05,  0.02,  0.01, 0.0],   # eta_thermal
        [ -0.05,  0.03,  0.01, 0.0],   # surge_margin_main
        [  0.0,   0.0,   0.0,  0.0],   # T_exhaust_source (disturbance, unaffected)
        [  0.0,   0.0,   0.0,  0.0],   # mdot_exhaust_source (disturbance)
    ], dtype=np.float64)

    def __init__(
        self,
        obs_vars: Sequence[str],
        action_vars: Sequence[str],
        design_point: dict[str, float],
        seed: int = 42,
        fail_at_step: int | None = None,
        inlet_temp_drift: bool = False,
        obs_noise_std: float = 0.0,
    ) -> None:
        self._obs_vars: list[str] = list(obs_vars)
        self._action_vars: list[str] = list(action_vars)
        self._design_point: dict[str, float] = dict(design_point)
        self._seed = seed
        self._fail_at_step = fail_at_step
        self._inlet_temp_drift = inlet_temp_drift
        self._obs_noise_std = obs_noise_std

        # Validate that sensitivity matrix columns cover action_vars
        self._n_obs = len(self._obs_vars)
        self._n_actions = len(self._action_vars)

        # Current state (reset to design point on each reset)
        self._state: dict[str, float] = {}
        self._pending_inputs: dict[str, float] = {}

        # Tracking counters
        self.n_steps: int = 0
        self.n_resets: int = 0
        self.current_time: float = 0.0
        self.step_size: float = 5.0

        self._accumulated_drift: dict[str, float] = {}  # persistent across steps
        self._rng = np.random.default_rng(seed)
        self._initialized = False

    # ── FMUInterface implementation ────────────────────────────────────────────

    def initialize(self, start_time: float, stop_time: float, step_size: float) -> None:
        self.step_size = step_size
        self.current_time = start_time
        self._rng = np.random.default_rng(self._seed)
        self._state = dict(self._design_point)
        self._pending_inputs = {}
        self._initialized = True

    def set_inputs(self, inputs: dict[str, float]) -> None:
        for key in inputs:
            if key not in self._action_vars:
                raise KeyError(
                    f"Unknown action variable '{key}'. "
                    f"Valid: {self._action_vars}"
                )
        self._pending_inputs.update(inputs)

    def do_step(self, current_time: float, step_size: float) -> bool:
        self.n_steps += 1

        # Failure injection
        if self._fail_at_step is not None and self.n_steps >= self._fail_at_step:
            return False

        # Apply linearized sensitivities
        action_vec = np.array(
            [self._pending_inputs.get(v, 0.0) for v in self._action_vars],
            dtype=np.float64,
        )
        # Slice sensitivity to match obs_vars and action_vars
        sens = self._get_sensitivity_slice()
        deltas = sens @ action_vec  # shape: (n_obs,)

        for i, var in enumerate(self._obs_vars):
            if var in self._design_point:
                self._state[var] = self._design_point[var] + deltas[i]

        # Inlet temperature drift injection (accumulates across steps)
        if self._inlet_temp_drift and "T_compressor_inlet" in self._obs_vars:
            drift_per_step = 0.15  # °C per step → 20 steps = 3°C total drop
            self._accumulated_drift["T_compressor_inlet"] = (
                self._accumulated_drift.get("T_compressor_inlet", 0.0) - drift_per_step
            )
            self._state["T_compressor_inlet"] = (
                self._design_point.get("T_compressor_inlet", 33.0)
                + self._accumulated_drift["T_compressor_inlet"]
            )

        # Observation noise
        if self._obs_noise_std > 0.0:
            for var in self._obs_vars:
                if var in self._state:
                    self._state[var] += float(
                        self._rng.normal(0.0, self._obs_noise_std)
                    )

        self._pending_inputs = {}
        self.current_time = current_time + step_size
        return True

    def get_outputs(self) -> dict[str, float]:
        return {var: self._state.get(var, self._design_point.get(var, 0.0))
                for var in self._obs_vars}

    def get_outputs_as_array(self) -> np.ndarray:
        outputs = self.get_outputs()
        return np.array([outputs[v] for v in self._obs_vars], dtype=np.float32)

    def reset(self) -> None:
        self.n_resets += 1
        self.n_steps = 0
        self.current_time = 0.0
        self._rng = np.random.default_rng(self._seed)
        self._state = dict(self._design_point)
        self._pending_inputs = {}
        self._accumulated_drift = {}

    def close(self) -> None:
        pass  # No resources to release for mock

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_sensitivity_slice(self) -> np.ndarray:
        """Return the (n_obs, n_actions) sensitivity submatrix."""
        known_obs = [
            "T_turbine_inlet", "T_compressor_inlet", "P_high", "P_low",
            "mdot_turbine", "W_turbine", "W_main_compressor", "W_net",
            "eta_thermal", "surge_margin_main", "T_exhaust_source", "mdot_exhaust_source",
        ]
        known_actions = ["bypass_valve_opening", "igv_angle_normalized",
                         "inventory_valve_opening", "cooling_flow_normalized"]

        obs_idx = [known_obs.index(v) if v in known_obs else None
                   for v in self._obs_vars]
        act_idx = [known_actions.index(v) if v in known_actions else None
                   for v in self._action_vars]

        n_obs = len(self._obs_vars)
        n_act = len(self._action_vars)
        result = np.zeros((n_obs, n_act), dtype=np.float64)

        for row_i, oi in enumerate(obs_idx):
            if oi is None:
                continue
            for col_j, aj in enumerate(act_idx):
                if aj is None:
                    continue
                result[row_i, col_j] = self._SENSITIVITY[oi, aj]

        return result
