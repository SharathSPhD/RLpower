"""PID baseline policy for SCO2FMUEnv comparison.

Implements a full PID controller (proportional + integral + derivative) with
a first-order low-pass filter on the derivative term to suppress noise
amplification. The filter time constant `derivative_filter_tau` (0 < tau <= 1)
controls the blend: tau=1 disables filtering (pure raw derivative), tau→0
approaches a no-derivative (PI) controller.
"""
from __future__ import annotations
import numpy as np


class PIDController:
    """Full PID controller with filtered derivative term.

    Parameters
    ----------
    kp:
        Proportional gain.
    ki:
        Integral gain.
    kd:
        Derivative gain.
    setpoint:
        Target value for the controlled variable.
    output_limits:
        (lo, hi) tuple clipping the controller output.
    derivative_filter_tau:
        Low-pass filter coefficient for the derivative term (0 < tau <= 1).
        tau=1.0 → no filtering (pure derivative).
        tau=0.1 → strong smoothing (retains 10% new, 90% previous value).
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        setpoint: float,
        output_limits: tuple,
        kd: float = 0.0,
        derivative_filter_tau: float = 0.1,
    ) -> None:
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._setpoint = setpoint
        self._lo, self._hi = output_limits
        self._tau = float(np.clip(derivative_filter_tau, 1e-6, 1.0))
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._filtered_d: float = 0.0

    def compute(self, measurement: float, dt: float) -> float:
        """Compute PID output for the current measurement.

        Parameters
        ----------
        measurement:
            Current process variable reading.
        dt:
            Time step in seconds.

        Returns
        -------
        float
            Clipped controller output in [lo, hi].
        """
        error = self._setpoint - measurement
        self._integral += error * dt

        # Filtered derivative: low-pass filter on error rate to suppress noise.
        raw_d = (error - self._prev_error) / max(dt, 1e-9)
        self._filtered_d = (
            self._tau * raw_d + (1.0 - self._tau) * self._filtered_d
        )
        self._prev_error = error

        output = (
            self._kp * error
            + self._ki * self._integral
            + self._kd * self._filtered_d
        )
        return float(np.clip(output, self._lo, self._hi))

    def reset(self) -> None:
        """Reset integrator, derivative state, and previous error."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._filtered_d = 0.0


class PIDBaseline:
    """Multi-channel PID policy for SCO2FMUEnv.

    Each action channel has its own PIDController tracking a single process
    variable (temperature, pressure, or mass flow). Gains are specified
    per-channel via the ``gains`` config key.

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
        self._controllers: dict[str, PIDController] = {}
        for act_name in self._action_vars:
            g = gains[act_name]
            meas_obs_idx = self._meas_idx[act_name]
            meas_obs_name = self._obs_vars[meas_obs_idx]
            sp_key = self._find_setpoint_key(meas_obs_name, setpoints)
            sp = setpoints.get(sp_key, 0.0)
            self._controllers[act_name] = PIDController(
                kp=float(g["kp"]),
                ki=float(g["ki"]),
                kd=float(g.get("kd", 0.0)),
                derivative_filter_tau=float(g.get("derivative_filter_tau", 0.1)),
                setpoint=sp,
                output_limits=(-1.0, 1.0),
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
