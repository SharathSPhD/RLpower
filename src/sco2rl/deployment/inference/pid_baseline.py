"""PID baseline policy for SCO2FMUEnv comparison."""
from __future__ import annotations
import numpy as np


class PIDController:
    def __init__(self, kp: float, ki: float, setpoint: float,
                 output_limits: tuple):
        self._kp = kp
        self._ki = ki
        self._setpoint = setpoint
        self._lo, self._hi = output_limits
        self._integral = 0.0

    def compute(self, measurement: float, dt: float) -> float:
        error = self._setpoint - measurement
        self._integral += error * dt
        output = self._kp * error + self._ki * self._integral
        return float(np.clip(output, self._lo, self._hi))

    def reset(self) -> None:
        self._integral = 0.0


class PIDBaseline:
    """Multi-channel PID policy for SCO2FMUEnv.
    predict() returns (action, None) -- compatible with PolicyEvaluator.
    """
    def __init__(self, config: dict):
        self._cfg = config
        self._obs_vars = config["obs_vars"]
        self._action_vars = config["action_vars"]
        self._n_obs = int(config.get("n_obs", len(self._obs_vars)))
        self._history_steps = int(config.get("history_steps", 1))
        gains = config["gains"]
        setpoints = config["setpoints"]
        self._meas_idx = config["measurement_indices"]
        self._dt = config.get("dt", 5.0)
        self._controllers = {}
        for act_name in self._action_vars:
            g = gains[act_name]
            meas_obs_idx = self._meas_idx[act_name]
            meas_obs_name = self._obs_vars[meas_obs_idx]
            sp_key = self._find_setpoint_key(meas_obs_name, setpoints)
            sp = setpoints.get(sp_key, 0.0)
            self._controllers[act_name] = PIDController(
                kp=g["kp"], ki=g["ki"],
                setpoint=sp, output_limits=(-1.0, 1.0)
            )

    def _find_setpoint_key(self, obs_name: str, setpoints: dict) -> str:
        if obs_name in setpoints:
            return obs_name
        for k in setpoints:
            if k.lower() in obs_name.lower() or obs_name.lower() in k.lower():
                return k
        return obs_name

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        if obs.ndim == 2:
            obs = obs[0]
        latest_offset = max(self._history_steps - 1, 0) * self._n_obs
        actions = []
        for act_name in self._action_vars:
            meas_idx = self._meas_idx[act_name]
            idx = latest_offset + meas_idx
            if idx >= len(obs):
                idx = min(meas_idx, len(obs) - 1)
            measurement = float(obs[idx])
            ctrl = self._controllers[act_name]
            actions.append(ctrl.compute(measurement, self._dt))
        return np.array(actions, dtype=np.float32), None

    def reset(self) -> None:
        for ctrl in self._controllers.values():
            ctrl.reset()
