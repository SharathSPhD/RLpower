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
        self._obs_range = np.maximum(self._obs_hi - self._obs_lo, 1e-6)

        # Z-score normalization stats (optional; injected from norm_stats)
        norm = config.get("normalization", {})
        self._has_zscore = bool(norm)
        if self._has_zscore:
            self._obs_mean = np.array(norm["obs_mean"], dtype=np.float32)
            self._obs_std = np.maximum(np.array(norm["obs_std"], dtype=np.float32), 1e-6)
            self._act_mean = np.array(norm["act_mean"], dtype=np.float32)
            self._act_std = np.maximum(np.array(norm["act_std"], dtype=np.float32), 1e-6)
            self._next_obs_mean = np.array(norm["next_obs_mean"], dtype=np.float32)
            self._next_obs_std = np.maximum(np.array(norm["next_obs_std"], dtype=np.float32), 1e-6)
            self._has_zscore = (
                self._obs_mean.shape[0] == self._n_obs
                and self._obs_std.shape[0] == self._n_obs
                and self._act_mean.shape[0] == self._n_act
                and self._act_std.shape[0] == self._n_act
                and self._next_obs_mean.shape[0] == self._n_obs
                and self._next_obs_std.shape[0] == self._n_obs
            )
        # Min-max normalization of history observations returned to the RL policy.
        # Uses obs_bounds range; avoids the zero-std problem for constant variables
        # (e.g. eta, p_outlet which don't vary during FNO training data collection).
        self._hist_lo = np.tile(self._obs_lo, self._history_steps)
        self._hist_range = np.tile(self._obs_range, self._history_steps)

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

        # Design-point values for initial state.
        # Default to physical midpoint of each variable's range so that the
        # initial state is well inside the safety envelope regardless of obs_lo.
        dp = config.get("obs_design_point", {})
        self._design_point = np.array(
            [
                float(dp.get(v, (lo + hi) * 0.5))
                for v, lo, hi in zip(self._obs_vars, self._obs_lo, self._obs_hi)
            ],
            dtype=np.float32,
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
        perturbation = rng.standard_normal(self._n_obs).astype(np.float32) * 0.01 * self._obs_range
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

        obs = self._normalized_history()
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

        obs = self._normalized_history()
        info: dict[str, Any] = {
            "step": self._step_count,
            "phys_action": phys_action,
            "terminated_reason": term_reason,
            "constraint_violation_step": step_has_violation,
            "constraint_violations": violation_values,
            "episode_constraint_violations": self._episode_constraint_violations,
        }
        return obs, reward, terminated, truncated, info

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize physical state to [0, 1] range."""
        return (state - self._obs_lo) / self._obs_range

    def _normalized_history(self) -> np.ndarray:
        """Return history buffer normalized to [-1, 1] via obs_bounds min-max."""
        return (2.0 * (self._history - self._hist_lo) / self._hist_range - 1.0).astype(np.float32)

    def _compute_reward(
        self,
        next_state: np.ndarray,
        phys_action: np.ndarray,
        prev_phys_action: np.ndarray,
    ) -> float:
        """Compute reward signal."""
        # W_net is the first observation variable in the dataset
        # Find index of W_net or use position 0 as proxy
        w_net_idx = 0
        for i, v in enumerate(self._obs_vars):
            if "W_net" in v or "w_net" in v or "net_power" in v or "W_turbine" in v:
                w_net_idx = i
                break

        w_net = float(next_state[w_net_idx])
        tracking_error = abs(w_net - self._W_net_setpoint) / max(self._W_net_setpoint, 1e-6)
        r_tracking = self._w_tracking * max(0.0, 1.0 - tracking_error)

        # Smoothness penalty: penalize large action changes
        action_delta = np.sum(np.abs(phys_action - prev_phys_action) / np.maximum(self._act_range, 1e-6))
        r_smooth = -self._w_smoothness * float(action_delta)

        return float(r_tracking + r_smooth)

    def _check_terminated(self, state: np.ndarray) -> tuple[bool, str]:
        """Check hard safety constraints."""
        # T_comp_inlet must stay above minimum
        for i, v in enumerate(self._obs_vars):
            if "T_compressor_inlet" in v or "T_comp" in v:
                if float(state[i]) < self._T_comp_min:
                    return True, "T_compressor_inlet_violation"
                break

        # Check for NaN/Inf
        if not np.all(np.isfinite(state)):
            return True, "nan_state"

        return False, ""

    def _compute_constraint_violations(self, state: np.ndarray) -> dict[str, float]:
        """Return per-constraint violation magnitudes (0 = no violation)."""
        violations: dict[str, float] = {}
        for i, v in enumerate(self._obs_vars):
            if "T_compressor_inlet" in v or "T_comp" in v:
                violations["T_comp_inlet"] = max(0.0, self._T_comp_min - float(state[i]))
                break
        return violations

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
