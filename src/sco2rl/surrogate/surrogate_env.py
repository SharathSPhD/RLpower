"""SurrogateEnv: Gymnasium environment backed by FNO1d surrogate model.

Replaces the FMU with a trained surrogate for GPU-native vectorized training.
Compatible observation/action space with SCO2FMUEnv for cross-validation.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from sco2rl.surrogate.fno_model import FNO1d


class SurrogateEnv(gym.Env):
    """Gymnasium env backed by FNO1d surrogate instead of FMU.

    Provides the same observation/action interface as SCO2FMUEnv but uses the
    FNO1d model for state transition, enabling fast GPU-native training.

    Parameters
    ----------
    model : FNO1d
        Trained (or randomly initialized) FNO1d surrogate model.
    config : dict
        Environment configuration matching SCO2FMUEnv config schema.
    device : str
        Torch device for surrogate inference, e.g. "cpu" or "cuda".
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        model: FNO1d,
        config: dict,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self._model = model
        self._config = config
        self._device = torch.device(device)
        self._model.to(self._device)
        self._model.eval()

        # ── Parse config ──────────────────────────────────────────────────────
        self._obs_vars: list[str] = config["obs_vars"]
        self._action_vars: list[str] = config["action_vars"]
        self._n_obs = len(self._obs_vars)
        self._n_act = len(self._action_vars)
        self._history_steps: int = config.get("history_steps", 1)
        self._max_steps: int = config.get("episode_max_steps", 200)
        self._step_size: float = config.get("step_size", 5.0)

        # Reward weights
        rw = config.get("reward", {})
        self._w_tracking: float = rw.get("w_tracking", 1.0)
        self._w_efficiency: float = rw.get("w_efficiency", 0.3)
        self._w_smoothness: float = rw.get("w_smoothness", 0.1)
        self._rated_power: float = rw.get("rated_power_mw", 10.0)
        self._design_eta: float = rw.get("design_efficiency", 0.40)
        self._terminal_reward: float = rw.get("terminal_failure_reward", -100.0)

        # Safety thresholds
        safety = config.get("safety", {})
        self._T_comp_min: float = safety.get("T_comp_inlet_min_c", safety.get("T_compressor_inlet_min", 32.2))
        self._surge_min: float = safety.get("surge_margin_min", 0.05)

        # Setpoints
        sp = config.get("setpoint", {})
        self._W_net_setpoint: float = sp.get("W_net_setpoint", sp.get("W_net", 10.0))

        # Obs bounds for normalization (default 0..1)
        obs_bounds = config.get("obs_bounds", {})
        self._obs_lo = np.array(
            [obs_bounds.get(v, (0.0, 1.0))[0] for v in self._obs_vars], dtype=np.float32
        )
        self._obs_hi = np.array(
            [obs_bounds.get(v, (0.0, 1.0))[1] for v in self._obs_vars], dtype=np.float32
        )
        self._obs_range = np.where(
            (self._obs_hi - self._obs_lo) > 0,
            self._obs_hi - self._obs_lo,
            1.0,
        )

        # Optional z-score stats from train_surrogate.py
        norm_cfg = config.get("normalization", {})
        self._obs_mean = np.asarray(norm_cfg.get("obs_mean", []), dtype=np.float32)
        self._obs_std = np.asarray(norm_cfg.get("obs_std", []), dtype=np.float32)
        self._act_mean = np.asarray(norm_cfg.get("act_mean", []), dtype=np.float32)
        self._act_std = np.asarray(norm_cfg.get("act_std", []), dtype=np.float32)
        self._next_obs_mean = np.asarray(norm_cfg.get("next_obs_mean", []), dtype=np.float32)
        self._next_obs_std = np.asarray(norm_cfg.get("next_obs_std", []), dtype=np.float32)
        self._has_zscore = (
            self._obs_mean.shape[0] == self._n_obs
            and self._obs_std.shape[0] == self._n_obs
            and self._act_mean.shape[0] == self._n_act
            and self._act_std.shape[0] == self._n_act
            and self._next_obs_mean.shape[0] == self._n_obs
            and self._next_obs_std.shape[0] == self._n_obs
        )

        # Action config (physical bounds + rate limit)
        action_cfg = config.get("action_config", {})
        self._act_phys_min = np.array(
            [action_cfg.get(v, {}).get("phys_min", action_cfg.get(v, {}).get("min", 0.0)) for v in self._action_vars],
            dtype=np.float32,
        )
        self._act_phys_max = np.array(
            [action_cfg.get(v, {}).get("phys_max", action_cfg.get(v, {}).get("max", 1.0)) for v in self._action_vars],
            dtype=np.float32,
        )
        self._act_range = self._act_phys_max - self._act_phys_min
        self._rate_limits = np.array(
            [
                action_cfg.get(v, {}).get(
                    "rate_limit",
                    action_cfg.get(v, {}).get("rate", 0.05),
                )
                for v in self._action_vars
            ],
            dtype=np.float32,
        )

        # Design-point values for initial state
        dp = config.get("obs_design_point", {})
        self._design_point = np.array(
            [dp.get(v, 0.5) for v in self._obs_vars], dtype=np.float32
        )

        # ── Gymnasium spaces ──────────────────────────────────────────────────
        obs_dim = self._n_obs * self._history_steps
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._n_act,), dtype=np.float32
        )

        # ── Internal state ────────────────────────────────────────────────────
        self._state: np.ndarray = self._design_point.copy()
        self._history: np.ndarray = np.tile(self._design_point, self._history_steps)
        self._prev_action: np.ndarray = np.zeros(self._n_act, dtype=np.float32)
        self._prev_phys_action: np.ndarray = self._act_phys_min.copy()
        self._step_count: int = 0
        self._current_phys_action: np.ndarray = (
            self._act_phys_min + self._act_range * 0.5
        )
        self._episode_constraint_violations: int = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        return self._device

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # Start near design point with small Gaussian perturbation (1% of range)
        perturbation = rng.standard_normal(self._n_obs).astype(np.float32) * 0.01
        self._state = np.clip(
            self._design_point + perturbation,
            self._obs_lo,
            self._obs_hi,
        )

        # Initialize history buffer with replicated initial state
        self._history = np.tile(self._state, self._history_steps).astype(np.float32)
        self._prev_action = np.zeros(self._n_act, dtype=np.float32)
        self._prev_phys_action = self._act_phys_min.copy()
        self._current_phys_action = self._act_phys_min + self._act_range * 0.5
        self._step_count = 0
        self._episode_constraint_violations = 0

        obs = self._history.copy()
        info: dict[str, Any] = {"step": 0}
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one environment step.

        Parameters
        ----------
        action : np.ndarray
            Shape (n_act,), normalized to [-1, 1].

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Scale to physical range: [-1,1] -> [phys_min, phys_max]
        phys_action = self._act_phys_min + (action + 1.0) * 0.5 * self._act_range

        # Apply rate limiting
        delta = phys_action - self._current_phys_action
        rate_limited_delta = np.clip(delta, -self._rate_limits, self._rate_limits)
        phys_action = self._current_phys_action + rate_limited_delta
        previous_phys_action = self._current_phys_action.copy()
        self._current_phys_action = phys_action.copy()

        # Normalize state/action to the model feature space.
        if self._has_zscore:
            state_model = (self._state - self._obs_mean) / np.maximum(self._obs_std, 1e-6)
            act_model = (action - self._act_mean) / np.maximum(self._act_std, 1e-6)
        else:
            state_model = self._normalize_state(self._state)
            # Backward-compat mode: dataset actions are normalized to [-1, 1].
            act_model = action

        # Build model input tensors
        state_t = torch.tensor(state_model, dtype=torch.float32, device=self._device).unsqueeze(0)
        act_t = torch.tensor(act_model, dtype=torch.float32, device=self._device).unsqueeze(0)

        # One-step surrogate prediction
        with torch.no_grad():
            next_state_t = self._model.predict_next_state(state_t, act_t)
        next_state_model = next_state_t.squeeze(0).cpu().numpy().astype(np.float32)
        if self._has_zscore:
            next_state = (
                next_state_model * np.maximum(self._next_obs_std, 1e-6) + self._next_obs_mean
            )
        else:
            next_state = self._obs_lo + next_state_model * self._obs_range
        next_state = np.clip(next_state, self._obs_lo, self._obs_hi)

        # Update history buffer (FIFO: drop oldest, append newest)
        if self._history_steps > 1:
            self._history = np.roll(self._history, -self._n_obs)
            self._history[-self._n_obs:] = next_state
        else:
            self._history = next_state.copy()

        # Compute reward
        reward = self._compute_reward(next_state, phys_action, previous_phys_action)

        # Check termination conditions
        terminated, term_reason = self._check_terminated(next_state)
        violation_values = self._compute_constraint_violations(next_state)
        step_has_violation = float(any(v > 0.0 for v in violation_values.values()))
        self._episode_constraint_violations += int(step_has_violation > 0.0)
        if terminated:
            reward = self._terminal_reward

        # Check truncation
        self._step_count += 1
        truncated = self._step_count >= self._max_steps

        # Update state
        self._state = next_state
        self._prev_action = action.copy()
        self._prev_phys_action = phys_action.copy()

        obs = self._history.copy()
        info: dict[str, Any] = {
            "step": self._step_count,
            "phys_action": phys_action,
            "terminated_reason": term_reason,
            "constraint_violation_step": step_has_violation,
            "constraint_violations": violation_values,
            "raw_obs": {v: float(next_state[i]) for i, v in enumerate(self._obs_vars)},
        }
        if terminated or truncated:
            info["constraint_violation"] = (
                float(self._episode_constraint_violations) / max(self._step_count, 1)
            )
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state from physical range to [0, 1] per variable."""
        return ((state - self._obs_lo) / self._obs_range).astype(np.float32)

    def _compute_reward(
        self,
        state: np.ndarray,
        phys_action: np.ndarray,
        previous_phys_action: np.ndarray,
    ) -> float:
        """Compute scalar reward from current state and action.

        Reward components:
        - Tracking: penalize deviation from W_net setpoint
        - Efficiency: encourage high thermal efficiency
        - Smoothness: penalize large action changes
        """
        w_net = self._derived_w_net(state)
        dev = abs(w_net - self._W_net_setpoint) / max(self._rated_power, 1e-6)
        r_tracking = max(0.0, 1.0 - dev**2)

        eta = self._derived_eta(state, w_net)
        eta_ratio = np.clip(eta / max(self._design_eta, 1e-6), 0.0, 2.0)
        r_efficiency = float(eta_ratio)

        # Smoothness penalty aligned with SCO2FMUEnv:
        # mean((delta_physical_action / action_range)^2)
        action_delta = phys_action - previous_phys_action
        normalized_delta = action_delta / np.maximum(self._act_range, 1e-9)
        r_smoothness = -float(np.mean(normalized_delta**2))

        reward = (
            self._w_tracking * r_tracking
            + self._w_efficiency * r_efficiency
            + self._w_smoothness * r_smoothness
        )
        return float(reward)

    def _check_terminated(self, state: np.ndarray) -> tuple[bool, str]:
        """Return (terminated, reason) given current surrogate state."""
        violation_values = self._compute_constraint_violations(state)
        if violation_values["T_comp_min"] > 0.0:
            return True, "T_compressor_inlet_violation"
        if violation_values["surge_margin_main"] > 0.0:
            return True, "surge_margin_main_violation"
        if violation_values["surge_margin_recomp"] > 0.0:
            return True, "surge_margin_recomp_violation"
        return False, ""

    def _compute_constraint_violations(self, state: np.ndarray) -> dict[str, float]:
        """Compute non-negative constraint violations matching SCO2FMUEnv."""
        t_idx = self._index_of(
            ["T_compressor_inlet", "main_compressor.T_inlet_rt", "precooler.T_outlet_rt"]
        )
        sm_main_idx = self._index_of(["surge_margin_main"])
        sm_recomp_idx = self._index_of(["surge_margin_recomp"])

        t_comp = float(state[t_idx]) if t_idx is not None else None
        sm_main = float(state[sm_main_idx]) if sm_main_idx is not None else None
        sm_recomp = float(state[sm_recomp_idx]) if sm_recomp_idx is not None else None

        return {
            "T_comp_min": max(0.0, self._T_comp_min - t_comp) if t_comp is not None else 0.0,
            "surge_margin_main": (
                max(0.0, self._surge_min - sm_main) if sm_main is not None else 0.0
            ),
            "surge_margin_recomp": (
                max(0.0, self._surge_min - sm_recomp) if sm_recomp is not None else 0.0
            ),
        }

    def _index_of(self, candidates: list[str]) -> int | None:
        for name in candidates:
            if name in self._obs_vars:
                return self._obs_vars.index(name)
        return None

    def _derived_w_net(self, state: np.ndarray) -> float:
        idx = self._index_of(["W_net"])
        if idx is not None:
            return float(state[idx])
        w_t_idx = self._index_of(["W_turbine", "turbine.W_turbine"])
        w_c_idx = self._index_of(["W_main_compressor", "main_compressor.W_comp"])
        if w_t_idx is not None and w_c_idx is not None:
            return float(state[w_t_idx] - state[w_c_idx])
        return 0.0

    def _derived_eta(self, state: np.ndarray, w_net: float) -> float:
        idx = self._index_of(["eta_thermal"])
        if idx is not None:
            return float(state[idx])
        q_idx = self._index_of(["Q_recuperator", "recuperator.Q_actual"])
        if q_idx is not None:
            q_val = float(state[q_idx])
            if q_val > 1e-6:
                return float(np.clip(w_net / q_val, 0.0, 1.5))
        return 0.0
