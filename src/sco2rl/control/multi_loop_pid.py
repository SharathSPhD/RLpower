"""Multi-loop PID controller for SCO2FMUEnv.

Provides a properly-tuned multi-channel PID baseline using the full
PIDController (with anti-windup and derivative) as a drop-in replacement for
the legacy PIDBaseline in deployment/inference/pid_baseline.py.

Configuration
-------------
Accepts the same config dict format as PIDBaseline (for backward-compatible
evaluation harness reuse), extended with optional ``kd``,
``anti_windup_gain``, and ``derivative_filter_tau`` keys per channel.

Measurement pairing (default for simple recuperated cycle):
    bypass_valve      → W_net              (MW)
    igv               → T_turbine_inlet    (°C)
    inventory_valve   → P_high             (MPa)
    cooling_flow      → T_compressor_inlet (°C)

Gains are IMC-inspired estimates valid for MockFMU linearised physics.
Re-tune for real FMU via ``scripts/tune_pid.py``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from sco2rl.control.interfaces import Controller
from sco2rl.control.pid import PIDController


# ---------------------------------------------------------------------------
# Default gains — IMC-tuned estimates for simple recuperated cycle
# (MockFMU linearised sensitivities; re-tune for real FMU)
# ---------------------------------------------------------------------------

_DEFAULT_GAINS: dict[str, dict[str, float]] = {
    # bypass_valve → W_net (sensitivity ≈ -4 MW/unit_action, τ≈25s)
    # Kp=0.25: 1 MW error → 0.25 unit action; Ki=Kp/τ
    "bypass_valve_opening": {
        "kp": 0.25, "ki": 0.010, "kd": 0.50,
        "anti_windup_gain": 0.10, "derivative_filter_tau": 10.0,
    },
    # igv → T_turbine_inlet (sensitivity ≈ +5 °C/unit, τ≈60s)
    "igv_angle_normalized": {
        "kp": 0.010, "ki": 0.0002, "kd": 0.05,
        "anti_windup_gain": 0.10, "derivative_filter_tau": 15.0,
    },
    # inventory_valve → P_high (sensitivity ≈ -0.8 MPa/unit, τ≈30s)
    "inventory_valve_opening": {
        "kp": 0.30, "ki": 0.010, "kd": 0.50,
        "anti_windup_gain": 0.10, "derivative_filter_tau": 10.0,
    },
    # cooling_flow → T_compressor_inlet (sensitivity ≈ -3 °C/unit, τ≈25s)
    "cooling_flow_normalized": {
        "kp": 0.20, "ki": 0.008, "kd": 0.40,
        "anti_windup_gain": 0.10, "derivative_filter_tau": 10.0,
    },
    # FMU-path action names (fmu_var names from env.yaml)
    "regulator.T_init": {
        "kp": 0.25, "ki": 0.010, "kd": 0.50,
        "anti_windup_gain": 0.10, "derivative_filter_tau": 10.0,
    },
    "regulator.m_flow_init": {
        "kp": 0.010, "ki": 0.0002, "kd": 0.05,
        "anti_windup_gain": 0.10, "derivative_filter_tau": 15.0,
    },
    "turbine.p_out": {
        "kp": 0.30, "ki": 0.010, "kd": 0.50,
        "anti_windup_gain": 0.10, "derivative_filter_tau": 10.0,
    },
    "precooler.T_output": {
        "kp": 0.20, "ki": 0.008, "kd": 0.40,
        "anti_windup_gain": 0.10, "derivative_filter_tau": 10.0,
    },
}

_DEFAULT_SETPOINTS: dict[str, float] = {
    "W_net": 10.0,           # MW
    "T_turbine_inlet": 750.0,  # °C
    "P_high": 20.0,          # MPa
    "T_compressor_inlet": 33.0,  # °C
    # FMU-path obs variable aliases
    "turbine.T_inlet_rt": 750.0,
    "main_compressor.T_inlet_rt": 33.0,
    "main_compressor.p_outlet": 20.0,
    "precooler.T_outlet_rt": 33.0,
}


class MultiLoopPID(Controller):
    """Multi-channel full PID controller compatible with SCO2FMUEnv.

    Drop-in replacement for ``PIDBaseline`` with improved gains and proper
    derivative / anti-windup handling.  Accepts the same config dict format
    so existing evaluation scripts need no changes.

    Parameters
    ----------
    config:
        Must contain:
        - ``obs_vars``: list of observation variable names
        - ``action_vars``: list of action variable names
        - ``measurement_indices``: dict mapping action_var → obs_var index
        - ``setpoints``: dict of setpoint values in physical units
        - ``gains``: optional dict overriding per-channel gains
        - ``history_steps``: int (default 1)
        - ``n_obs``: int (default len(obs_vars))
        - ``dt``: float simulation step size in seconds (default 5.0)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = config
        self._obs_vars: list[str] = config["obs_vars"]
        self._action_vars: list[str] = config["action_vars"]
        self._n_obs: int = int(config.get("n_obs", len(self._obs_vars)))
        self._history_steps: int = int(config.get("history_steps", 1))
        self._dt: float = float(config.get("dt", 5.0))

        gains: dict[str, dict] = config.get("gains", {})
        setpoints: dict[str, float] = config.get("setpoints", {})
        meas_idx: dict[str, int] = config.get("measurement_indices", {})

        self._controllers: dict[str, PIDController] = {}
        for act_name in self._action_vars:
            g = self._resolve_gains(act_name, gains)
            sp = self._resolve_setpoint(act_name, meas_idx, setpoints)
            self._controllers[act_name] = PIDController(
                kp=g["kp"],
                ki=g["ki"],
                kd=g.get("kd", 0.0),
                setpoint=sp,
                output_limits=(-1.0, 1.0),
                anti_windup_gain=g.get("anti_windup_gain", 0.1),
                derivative_filter_tau=g.get("derivative_filter_tau", 0.0),
                dt=self._dt,
            )
        self._meas_idx = meas_idx

    # ── Controller interface ───────────────────────────────────────────────────

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        if obs.ndim == 2:
            obs = obs[0]

        # Extract latest observation window (last history_steps slice)
        latest_offset = max(self._history_steps - 1, 0) * self._n_obs
        actions: list[float] = []

        for act_name in self._action_vars:
            meas_obs_idx = self._meas_idx.get(act_name, 0)
            idx = latest_offset + meas_obs_idx
            if idx >= len(obs):
                idx = min(meas_obs_idx, len(obs) - 1)
            measurement = float(obs[idx])
            actions.append(self._controllers[act_name].compute(measurement))

        return np.array(actions, dtype=np.float32), None

    def reset(self) -> None:
        for ctrl in self._controllers.values():
            ctrl.reset()

    @property
    def name(self) -> str:
        return "MultiLoopPID"

    # ── Setpoint control ───────────────────────────────────────────────────────

    def update_setpoint(self, action_var: str, setpoint: float) -> None:
        """Update the setpoint for a single action channel."""
        if action_var in self._controllers:
            self._controllers[action_var].set_setpoint(setpoint)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _resolve_gains(
        self, act_name: str, gains: dict[str, dict]
    ) -> dict[str, float]:
        """Return gains for act_name: user-provided first, then defaults."""
        if act_name in gains:
            return gains[act_name]
        if act_name in _DEFAULT_GAINS:
            return _DEFAULT_GAINS[act_name]
        # Fallback: conservative gains similar to legacy PIDBaseline
        return {"kp": 0.02, "ki": 0.001, "kd": 0.0, "anti_windup_gain": 0.1}

    def _resolve_setpoint(
        self,
        act_name: str,
        meas_idx: dict[str, int],
        setpoints: dict[str, float],
    ) -> float:
        """Map action name → measurement obs name → setpoint value."""
        obs_idx = meas_idx.get(act_name, 0)
        if obs_idx < len(self._obs_vars):
            obs_name = self._obs_vars[obs_idx]
            # Try direct match then substring match
            if obs_name in setpoints:
                return float(setpoints[obs_name])
            for sp_key, sp_val in setpoints.items():
                if sp_key.lower() in obs_name.lower() or obs_name.lower() in sp_key.lower():
                    return float(sp_val)
            # Check _DEFAULT_SETPOINTS
            if obs_name in _DEFAULT_SETPOINTS:
                return float(_DEFAULT_SETPOINTS[obs_name])
            for sp_key, sp_val in _DEFAULT_SETPOINTS.items():
                if sp_key.lower() in obs_name.lower() or obs_name.lower() in sp_key.lower():
                    return float(sp_val)
        return 0.0

    @classmethod
    def from_pid_config(cls, config: dict[str, Any]) -> "MultiLoopPID":
        """Construct from the same config dict format as PIDBaseline."""
        return cls(config)
