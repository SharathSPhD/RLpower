"""PolicyEvaluator -- evaluate a trained policy on any Gymnasium environment.

Works with SCO2FMUEnv and SurrogateEnv.
Does NOT modify VecNormalize stats (sets training=False during eval).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EvaluationMetrics:
    """Aggregate metrics from N evaluation episodes."""

    mean_reward: float
    std_reward: float
    mean_episode_length: float
    violation_rate: float           # fraction of steps with safety violations
    T_comp_inlet_min: float         # min observed T_comp_inlet across all episodes
    T_comp_inlet_mean: float        # mean observed T_comp_inlet
    n_episodes: int
    phase: int                      # curriculum phase at evaluation time
    per_episode_rewards: list[float] = field(default_factory=list)


class PolicyEvaluator:
    """Evaluate a policy on any Gymnasium environment.

    Parameters
    ----------
    env:
        A Gymnasium-compatible environment (SCO2FMUEnv, SurrogateEnv, etc.).
    config:
        Dict with keys:
          - n_eval_episodes (int): number of episodes to roll out
          - T_comp_inlet_var (str): name of T_compressor_inlet in obs_vars
          - deterministic (bool): whether to use deterministic policy
    """

    def __init__(self, env, config: dict) -> None:
        self._env = env
        self._n_episodes: int = int(config["n_eval_episodes"])
        self._T_var: str = config.get("T_comp_inlet_var", "T_compressor_inlet")
        self._deterministic: bool = bool(config.get("deterministic", True))

        # Determine the index of T_comp_inlet in obs_vars (for fallback)
        obs_vars = getattr(env, "_obs_vars", [])
        self._T_idx: int | None = (
            obs_vars.index(self._T_var) if self._T_var in obs_vars else None
        )

    def _get_T_comp(self, obs: np.ndarray, info: dict) -> float | None:
        """Extract T_compressor_inlet from step info or obs."""
        # Prefer raw_obs dict from info (most accurate)
        raw_obs = info.get("raw_obs", {})
        if self._T_var in raw_obs:
            return float(raw_obs[self._T_var])
        # Fallback: read from flattened obs at _T_idx
        if self._T_idx is not None and self._T_idx < len(obs):
            return float(obs[self._T_idx])
        return None

    def _is_violation(self, terminated: bool, info: dict) -> bool:
        """Return True if the step resulted in a safety violation."""
        reason = info.get("terminated_reason", None)
        if reason is not None and "violation" in str(reason):
            return True
        # Also treat early termination (terminated before truncated) as violation
        # but only if terminated_reason is non-None (not None == truncated)
        return False

    def evaluate(self, model, phase: int = 0) -> EvaluationMetrics:
        """Run n_eval_episodes and return EvaluationMetrics.

        Parameters
        ----------
        model:
            Any object with ``predict(obs, deterministic) -> (action, _state)``.
        phase:
            Curriculum phase to record in the metrics.
        """
        episode_rewards: list[float] = []
        episode_lengths: list[float] = []
        violation_steps: int = 0
        total_steps: int = 0
        T_values: list[float] = []

        env = self._env

        for ep_idx in range(self._n_episodes):
            # Deterministic per-episode seed ensures RL and PID face identical
            # initial conditions when both are evaluated on the same environment.
            episode_seed = ep_idx
            obs, info = env.reset(seed=episode_seed)

            # Reset stateful policies (e.g. PID integrator) between episodes.
            if hasattr(model, "reset"):
                model.reset()

            ep_reward = 0.0
            ep_steps = 0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=self._deterministic)
                obs, reward, terminated, truncated, step_info = env.step(action)
                ep_reward += float(reward)
                ep_steps += 1
                total_steps += 1

                # Track T_comp_inlet
                T_val = self._get_T_comp(obs, step_info)
                if T_val is not None:
                    T_values.append(T_val)

                # Safety violations
                if self._is_violation(terminated, step_info):
                    violation_steps += 1

                done = terminated or truncated

            episode_rewards.append(ep_reward)
            episode_lengths.append(float(ep_steps))

        rewards_arr = np.array(episode_rewards, dtype=np.float64)
        T_arr = np.array(T_values, dtype=np.float64) if T_values else np.array([np.nan])

        violation_rate = violation_steps / total_steps if total_steps > 0 else 0.0

        return EvaluationMetrics(
            mean_reward=float(np.mean(rewards_arr)),
            std_reward=float(np.std(rewards_arr)),
            mean_episode_length=float(np.mean(episode_lengths)),
            violation_rate=float(violation_rate),
            T_comp_inlet_min=float(np.nanmin(T_arr)),
            T_comp_inlet_mean=float(np.nanmean(T_arr)),
            n_episodes=self._n_episodes,
            phase=phase,
            per_episode_rewards=episode_rewards,
        )
