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
        self._T_comp_min: float = safety.get("T_comp_inlet_min_c", 32.2)
        self._surge_min: float = safety.get("surge_margin_min", 0.05)

        # Setpoints
        sp = config.get("setpoint", {})
        self._W_net_setpoint: float = sp.get("W_net_setpoint", 10.0)

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

        # Action config (physical bounds + rate limit)
        action_cfg = config.get("action_config", {})
        self._act_phys_min = np.array(
            [action_cfg.get(v, {}).get("phys_min", 0.0) for v in self._action_vars],
            dtype=np.float32,
        )
        self._act_phys_max = np.array(
            [action_cfg.get(v, {}).get("phys_max", 1.0) for v in self._action_vars],
            dtype=np.float32,
        )
        self._act_range = self._act_phys_max - self._act_phys_min
        self._rate_limits = np.array(
            [action_cfg.get(v, {}).get("rate_limit", 0.05) for v in self._action_vars],
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
        self._step_count: int = 0
        self._current_phys_action: np.ndarray = (
            self._act_phys_min + self._act_range * 0.5
        )

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
        self._current_phys_action = self._act_phys_min + self._act_range * 0.5
        self._step_count = 0

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
        self._current_phys_action = phys_action.copy()

        # Normalize state and action for model input
        state_norm = self._normalize_state(self._state)
        act_norm = (phys_action - self._act_phys_min) / np.where(
            self._act_range > 0, self._act_range, 1.0
        )

        # Build model input tensors
        state_t = torch.tensor(state_norm, dtype=torch.float32, device=self._device).unsqueeze(0)
        act_t = torch.tensor(act_norm, dtype=torch.float32, device=self._device).unsqueeze(0)

        # One-step surrogate prediction
        with torch.no_grad():
            next_state_t = self._model.predict_next_state(state_t, act_t)
        next_state_norm = next_state_t.squeeze(0).cpu().numpy().astype(np.float32)

        # Denormalize
        next_state = self._obs_lo + next_state_norm * self._obs_range

        # Update history buffer (FIFO: drop oldest, append newest)
        if self._history_steps > 1:
            self._history = np.roll(self._history, -self._n_obs)
            self._history[-self._n_obs:] = next_state_norm
        else:
            self._history = next_state_norm.copy()

        # Compute reward
        reward = self._compute_reward(next_state, phys_action, action)

        # Check termination conditions
        terminated = self._is_terminated(next_state)
        if terminated:
            reward = self._terminal_reward

        # Check truncation
        self._step_count += 1
        truncated = self._step_count >= self._max_steps

        # Update state
        self._state = next_state
        self._prev_action = action.copy()

        obs = self._history.copy()
        info: dict[str, Any] = {
            "step": self._step_count,
            "phys_action": phys_action,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state from physical range to [0, 1] per variable."""
        return ((state - self._obs_lo) / self._obs_range).astype(np.float32)

    def _compute_reward(
        self,
        state: np.ndarray,
        phys_action: np.ndarray,
        norm_action: np.ndarray,
    ) -> float:
        """Compute scalar reward from current state and action.

        Reward components:
        - Tracking: penalize deviation from W_net setpoint
        - Efficiency: encourage high thermal efficiency
        - Smoothness: penalize large action changes
        """
        # Find W_net and eta_thermal indices (if present)
        w_net_idx = self._obs_vars.index("W_net") if "W_net" in self._obs_vars else None
        eta_idx = self._obs_vars.index("eta_thermal") if "eta_thermal" in self._obs_vars else None

        # Tracking reward: normalized power deviation
        r_tracking = 0.0
        if w_net_idx is not None:
            w_net = state[w_net_idx]
            dev = abs(w_net - self._W_net_setpoint) / max(self._rated_power, 1e-6)
            r_tracking = -dev

        # Efficiency reward
        r_efficiency = 0.0
        if eta_idx is not None:
            eta = state[eta_idx]
            r_efficiency = (eta - self._design_eta) / max(self._design_eta, 1e-6)

        # Smoothness penalty: L2 norm of action change
        action_delta = norm_action - self._prev_action
        r_smoothness = -float(np.dot(action_delta, action_delta))

        reward = (
            self._w_tracking * r_tracking
            + self._w_efficiency * r_efficiency
            + self._w_smoothness * r_smoothness
        )
        return float(reward)

    def _is_terminated(self, state: np.ndarray) -> bool:
        """Check safety-critical termination conditions."""
        # T_compressor_inlet check
        t_idx = (
            self._obs_vars.index("T_compressor_inlet")
            if "T_compressor_inlet" in self._obs_vars
            else None
        )
        if t_idx is not None:
            T_comp = state[t_idx]
            if T_comp < self._T_comp_min:
                return True

        # Surge margin check
        sm_idx = (
            self._obs_vars.index("surge_margin_main")
            if "surge_margin_main" in self._obs_vars
            else None
        )
        if sm_idx is not None:
            sm = state[sm_idx]
            if sm < self._surge_min:
                return True

        return False
