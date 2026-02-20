"""GPU-native batched surrogate environment for SKRL PPO.

This environment keeps all vectorized state on a single device tensor and runs
one batched FNO forward pass per `step()` call.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import batch_space

from sco2rl.surrogate.fno_model import FNO1d


class TorchBatchedSurrogateEnv(gym.vector.VectorEnv):
    """Vectorized surrogate environment with batched device-side dynamics."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        model: FNO1d | Any,
        config: dict[str, Any],
        n_envs: int,
        device: str = "cpu",
    ) -> None:
        self._model = model
        self._config = config
        self._device = torch.device(device)
        self._n_envs = int(n_envs)

        if hasattr(self._model, "to"):
            self._model = self._model.to(self._device)
        if hasattr(self._model, "train"):
            self._model.train(False)

        # Parse config
        self._obs_vars: list[str] = list(config["obs_vars"])
        self._action_vars: list[str] = list(config["action_vars"])
        self._n_obs = len(self._obs_vars)
        self._n_act = len(self._action_vars)
        self._history_steps: int = int(config.get("history_steps", 1))
        self._max_steps: int = int(config.get("episode_max_steps", 200))

        rw = config.get("reward", {})
        self._w_tracking: float = float(rw.get("w_tracking", 1.0))
        self._w_efficiency: float = float(rw.get("w_efficiency", 0.3))
        self._w_smoothness: float = float(rw.get("w_smoothness", 0.1))
        self._rated_power: float = float(rw.get("rated_power_mw", 10.0))
        self._design_eta: float = float(rw.get("design_efficiency", 0.40))
        self._terminal_reward: float = float(rw.get("terminal_failure_reward", -100.0))

        safety = config.get("safety", {})
        self._T_comp_min: float = float(
            safety.get("T_comp_inlet_min_c", safety.get("T_compressor_inlet_min", 32.2))
        )
        self._surge_min: float = float(safety.get("surge_margin_min", 0.05))

        sp = config.get("setpoint", {})
        self._W_net_setpoint: float = float(sp.get("W_net_setpoint", sp.get("W_net", 10.0)))

        obs_bounds = config.get("obs_bounds", {})
        obs_lo = np.array(
            [obs_bounds.get(v, (0.0, 1.0))[0] for v in self._obs_vars],
            dtype=np.float32,
        )
        obs_hi = np.array(
            [obs_bounds.get(v, (0.0, 1.0))[1] for v in self._obs_vars],
            dtype=np.float32,
        )
        obs_range = np.where((obs_hi - obs_lo) > 0, obs_hi - obs_lo, 1.0).astype(np.float32)

        self._obs_lo = torch.tensor(obs_lo, device=self._device)
        self._obs_hi = torch.tensor(obs_hi, device=self._device)
        self._obs_range = torch.tensor(obs_range, device=self._device)

        norm_cfg = config.get("normalization", {})
        obs_mean = np.asarray(norm_cfg.get("obs_mean", []), dtype=np.float32)
        obs_std = np.asarray(norm_cfg.get("obs_std", []), dtype=np.float32)
        act_mean = np.asarray(norm_cfg.get("act_mean", []), dtype=np.float32)
        act_std = np.asarray(norm_cfg.get("act_std", []), dtype=np.float32)
        next_obs_mean = np.asarray(norm_cfg.get("next_obs_mean", []), dtype=np.float32)
        next_obs_std = np.asarray(norm_cfg.get("next_obs_std", []), dtype=np.float32)
        self._has_zscore = (
            obs_mean.shape[0] == self._n_obs
            and obs_std.shape[0] == self._n_obs
            and act_mean.shape[0] == self._n_act
            and act_std.shape[0] == self._n_act
            and next_obs_mean.shape[0] == self._n_obs
            and next_obs_std.shape[0] == self._n_obs
        )

        if self._has_zscore:
            self._obs_mean = torch.tensor(obs_mean, device=self._device)
            self._obs_std = torch.tensor(obs_std, device=self._device).clamp_min(1e-6)
            self._act_mean = torch.tensor(act_mean, device=self._device)
            self._act_std = torch.tensor(act_std, device=self._device).clamp_min(1e-6)
            self._next_obs_mean = torch.tensor(next_obs_mean, device=self._device)
            self._next_obs_std = torch.tensor(next_obs_std, device=self._device).clamp_min(1e-6)
        else:
            self._obs_mean = None
            self._obs_std = None
            self._act_mean = None
            self._act_std = None
            self._next_obs_mean = None
            self._next_obs_std = None

        action_cfg = config.get("action_config", {})
        act_phys_min = np.array(
            [action_cfg.get(v, {}).get("phys_min", action_cfg.get(v, {}).get("min", 0.0)) for v in self._action_vars],
            dtype=np.float32,
        )
        act_phys_max = np.array(
            [action_cfg.get(v, {}).get("phys_max", action_cfg.get(v, {}).get("max", 1.0)) for v in self._action_vars],
            dtype=np.float32,
        )
        rate_limits = np.array(
            [
                action_cfg.get(v, {}).get(
                    "rate_limit",
                    action_cfg.get(v, {}).get("rate", 0.05),
                )
                for v in self._action_vars
            ],
            dtype=np.float32,
        )
        self._act_phys_min = torch.tensor(act_phys_min, device=self._device)
        self._act_phys_max = torch.tensor(act_phys_max, device=self._device)
        self._act_range = (self._act_phys_max - self._act_phys_min).clamp_min(1e-9)
        self._rate_limits = torch.tensor(rate_limits, device=self._device)

        # Design-point defaults to physical midpoint so initial state is well
        # inside the safety envelope regardless of obs_lo magnitudes.
        dp = config.get("obs_design_point", {})
        design_point = np.array(
            [float(dp.get(v, (lo + hi) * 0.5))
             for v, lo, hi in zip(self._obs_vars, obs_lo, obs_hi)],
            dtype=np.float32,
        )
        self._design_point = torch.tensor(design_point, device=self._device)

        single_obs_dim = self._n_obs * self._history_steps
        single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(single_obs_dim,),
            dtype=np.float32,
        )
        single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._n_act,),
            dtype=np.float32,
        )
        self.num_envs = self._n_envs
        self.single_observation_space = single_observation_space
        self.single_action_space = single_action_space
        self.observation_space = batch_space(single_observation_space, self._n_envs)
        self.action_space = batch_space(single_action_space, self._n_envs)

        # Internal vectorized state
        self._state = torch.zeros((self._n_envs, self._n_obs), device=self._device)
        self._history = torch.zeros((self._n_envs, single_obs_dim), device=self._device)
        self._current_phys_action = torch.zeros((self._n_envs, self._n_act), device=self._device)
        self._prev_phys_action = torch.zeros((self._n_envs, self._n_act), device=self._device)
        self._step_count = torch.zeros((self._n_envs,), dtype=torch.int32, device=self._device)
        self._episode_constraint_violations = torch.zeros((self._n_envs,), dtype=torch.int32, device=self._device)

    @property
    def device(self) -> torch.device:
        return self._device

    def reset(self, *, seed: int | list[int] | None = None, options: dict[str, Any] | None = None):
        del options
        self._reset_indices(torch.arange(self._n_envs, device=self._device), seed=seed)
        obs = self._history.detach().cpu().numpy().astype(np.float32)
        info = {"step": np.zeros((self._n_envs,), dtype=np.int32)}
        return obs, info

    def step(self, actions: np.ndarray):
        action_t = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        if action_t.ndim == 1:
            action_t = action_t.unsqueeze(0)
        if action_t.shape != (self._n_envs, self._n_act):
            raise ValueError(
                f"Expected actions shape {(self._n_envs, self._n_act)}, got {tuple(action_t.shape)}"
            )
        action_t = torch.clamp(action_t, -1.0, 1.0)

        phys_action = self._act_phys_min + (action_t + 1.0) * 0.5 * self._act_range
        delta = phys_action - self._current_phys_action
        rate_limited_delta = torch.clamp(delta, -self._rate_limits, self._rate_limits)
        phys_action = self._current_phys_action + rate_limited_delta
        previous_phys_action = self._current_phys_action.clone()
        self._current_phys_action = phys_action

        if self._has_zscore:
            state_model = (self._state - self._obs_mean) / self._obs_std
            act_model = (action_t - self._act_mean) / self._act_std
        else:
            state_model = (self._state - self._obs_lo) / self._obs_range
            act_model = action_t

        with torch.no_grad():
            next_state_model = self._model.predict_next_state(state_model, act_model)

        if self._has_zscore:
            next_state = next_state_model * self._next_obs_std + self._next_obs_mean
        else:
            next_state = self._obs_lo + next_state_model * self._obs_range
        next_state = torch.clamp(next_state, self._obs_lo, self._obs_hi)

        if self._history_steps > 1:
            self._history = torch.roll(self._history, shifts=-self._n_obs, dims=1)
            self._history[:, -self._n_obs:] = next_state
        else:
            self._history = next_state.clone()

        reward = self._compute_reward(next_state, phys_action, previous_phys_action)

        violations = self._compute_constraint_violations(next_state)
        terminated, term_reason = self._check_terminated(violations)
        step_has_violation = (
            (violations["T_comp_min"] > 0.0)
            | (violations["surge_margin_main"] > 0.0)
            | (violations["surge_margin_recomp"] > 0.0)
        ).to(torch.float32)
        self._episode_constraint_violations += step_has_violation.to(torch.int32)

        reward = torch.where(
            terminated,
            torch.full_like(reward, self._terminal_reward),
            reward,
        )

        self._step_count += 1
        truncated = self._step_count >= self._max_steps

        self._state = next_state
        self._prev_phys_action = phys_action

        obs = self._history.detach().cpu().numpy().astype(np.float32)
        rewards_np = reward.detach().cpu().numpy().astype(np.float32)
        terminated_np = terminated.detach().cpu().numpy().astype(bool)
        truncated_np = truncated.detach().cpu().numpy().astype(bool)
        step_np = self._step_count.detach().cpu().numpy().astype(np.int32)

        t_comp_violation = violations["T_comp_min"].detach().cpu().numpy().astype(np.float32)
        surge_main_violation = violations["surge_margin_main"].detach().cpu().numpy().astype(np.float32)
        surge_recomp_violation = violations["surge_margin_recomp"].detach().cpu().numpy().astype(np.float32)
        step_violation_np = step_has_violation.detach().cpu().numpy().astype(np.float32)
        phys_action_np = phys_action.detach().cpu().numpy().astype(np.float32)

        done = terminated_np | truncated_np
        ratio = (
            self._episode_constraint_violations.detach().cpu().numpy().astype(np.float32)
            / np.maximum(step_np.astype(np.float32), 1.0)
        )

        info: dict[str, Any] = {
            "step": step_np,
            "phys_action": phys_action_np,
            "terminated_reason": np.array(term_reason, dtype=object),
            "constraint_violation_step": step_violation_np,
            "constraint_violations": {
                "T_comp_min": t_comp_violation,
                "surge_margin_main": surge_main_violation,
                "surge_margin_recomp": surge_recomp_violation,
            },
            "constraint_violation": ratio,
        }

        if np.any(done):
            done_idx = np.where(done)[0]
            final_obs = np.empty((self._n_envs,), dtype=object)
            final_obs[:] = None
            final_info = np.empty((self._n_envs,), dtype=object)
            final_info[:] = None
            for i in done_idx:
                final_obs[i] = obs[i].copy()
                final_info[i] = {
                    "terminated_reason": term_reason[i],
                    "constraint_violation": float(ratio[i]),
                }
            info["final_observation"] = final_obs
            info["final_info"] = final_info
            self._reset_indices(torch.as_tensor(done_idx, device=self._device), seed=None)
            obs = self._history.detach().cpu().numpy().astype(np.float32)

        return obs, rewards_np, terminated_np, truncated_np, info

    def close(self) -> None:
        return None

    def _reset_indices(self, indices: torch.Tensor, seed: int | list[int] | None) -> None:
        if indices.numel() == 0:
            return
        idx_np = indices.detach().cpu().numpy().astype(int)
        k = len(idx_np)

        if isinstance(seed, list):
            if len(seed) != self._n_envs:
                raise ValueError(f"seed list length must be {self._n_envs}, got {len(seed)}")
            seeds = [seed[i] for i in idx_np]
        elif seed is None:
            seeds = [None] * k
        else:
            seeds = [int(seed) + int(i) for i in idx_np]

        init_states = np.zeros((k, self._n_obs), dtype=np.float32)
        design_point = self._design_point.detach().cpu().numpy()
        obs_lo = self._obs_lo.detach().cpu().numpy()
        obs_hi = self._obs_hi.detach().cpu().numpy()
        for j, s in enumerate(seeds):
            rng = np.random.default_rng(s)
            perturbation = rng.standard_normal(self._n_obs).astype(np.float32) * 0.01
            init_states[j] = np.clip(design_point + perturbation, obs_lo, obs_hi)

        init_t = torch.tensor(init_states, dtype=torch.float32, device=self._device)
        self._state[indices] = init_t
        if self._history_steps > 1:
            self._history[indices] = init_t.repeat(1, self._history_steps)
        else:
            self._history[indices] = init_t
        mid_action = (self._act_phys_min + self._act_phys_max) * 0.5
        self._current_phys_action[indices] = mid_action
        self._prev_phys_action[indices] = self._act_phys_min
        self._step_count[indices] = 0
        self._episode_constraint_violations[indices] = 0

    def _compute_reward(
        self,
        state: torch.Tensor,
        phys_action: torch.Tensor,
        previous_phys_action: torch.Tensor,
    ) -> torch.Tensor:
        w_net = self._derived_w_net(state)
        dev = torch.abs(w_net - self._W_net_setpoint) / max(self._rated_power, 1e-6)
        r_tracking = torch.clamp(1.0 - dev * dev, min=0.0)

        eta = self._derived_eta(state, w_net)
        eta_ratio = torch.clamp(eta / max(self._design_eta, 1e-6), min=0.0, max=2.0)

        action_delta = phys_action - previous_phys_action
        normalized_delta = action_delta / self._act_range
        r_smoothness = -torch.mean(normalized_delta * normalized_delta, dim=1)

        return (
            self._w_tracking * r_tracking
            + self._w_efficiency * eta_ratio
            + self._w_smoothness * r_smoothness
        )

    def _check_terminated(self, violations: dict[str, torch.Tensor]) -> tuple[torch.Tensor, list[str]]:
        terminated = torch.zeros((self._n_envs,), dtype=torch.bool, device=self._device)
        reasons = [""] * self._n_envs

        mask_t = violations["T_comp_min"] > 0.0
        terminated = terminated | mask_t
        for i in torch.where(mask_t)[0].tolist():
            reasons[i] = "T_compressor_inlet_violation"

        mask_m = (violations["surge_margin_main"] > 0.0) & (~terminated)
        terminated = terminated | mask_m
        for i in torch.where(mask_m)[0].tolist():
            reasons[i] = "surge_margin_main_violation"

        mask_r = (violations["surge_margin_recomp"] > 0.0) & (~terminated)
        terminated = terminated | mask_r
        for i in torch.where(mask_r)[0].tolist():
            reasons[i] = "surge_margin_recomp_violation"

        return terminated, reasons

    def _compute_constraint_violations(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        t_idx = self._index_of(
            ["T_compressor_inlet", "main_compressor.T_inlet_rt", "precooler.T_outlet_rt"]
        )
        sm_main_idx = self._index_of(["surge_margin_main"])
        sm_recomp_idx = self._index_of(["surge_margin_recomp"])

        t_comp = state[:, t_idx] if t_idx is not None else None
        sm_main = state[:, sm_main_idx] if sm_main_idx is not None else None
        sm_recomp = state[:, sm_recomp_idx] if sm_recomp_idx is not None else None

        zeros = torch.zeros((self._n_envs,), dtype=torch.float32, device=self._device)
        return {
            "T_comp_min": torch.clamp(self._T_comp_min - t_comp, min=0.0) if t_comp is not None else zeros,
            "surge_margin_main": torch.clamp(self._surge_min - sm_main, min=0.0) if sm_main is not None else zeros,
            "surge_margin_recomp": torch.clamp(self._surge_min - sm_recomp, min=0.0) if sm_recomp is not None else zeros,
        }

    def _index_of(self, candidates: list[str]) -> int | None:
        for name in candidates:
            if name in self._obs_vars:
                return self._obs_vars.index(name)
        return None

    def _derived_w_net(self, state: torch.Tensor) -> torch.Tensor:
        idx = self._index_of(["W_net"])
        if idx is not None:
            return state[:, idx]
        w_t_idx = self._index_of(["W_turbine", "turbine.W_turbine"])
        w_c_idx = self._index_of(["W_main_compressor", "main_compressor.W_comp"])
        if w_t_idx is not None and w_c_idx is not None:
            return state[:, w_t_idx] - state[:, w_c_idx]
        return torch.zeros((self._n_envs,), dtype=torch.float32, device=self._device)

    def _derived_eta(self, state: torch.Tensor, w_net: torch.Tensor) -> torch.Tensor:
        idx = self._index_of(["eta_thermal"])
        if idx is not None:
            return state[:, idx]
        q_idx = self._index_of(["Q_recuperator", "recuperator.Q_actual"])
        if q_idx is not None:
            q_val = state[:, q_idx]
            ratio = torch.zeros_like(w_net)
            mask = q_val > 1e-6
            ratio[mask] = torch.clamp(w_net[mask] / q_val[mask], min=0.0, max=1.5)
            return ratio
        return torch.zeros((self._n_envs,), dtype=torch.float32, device=self._device)
