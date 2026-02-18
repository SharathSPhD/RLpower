"""LagrangianPPO -- SB3 PPO with trainable Lagrange constraint multipliers.

For each constraint c_i(s, a) <= 0, maintains lambda_i >= 0.
Lagrangian update: lambda_i <- max(0, lambda_i + alpha_lambda * mean_violation_i)

This is the dual-ascent (multiplier gradient) approach, NOT the penalty method.
Multipliers adapt automatically based on observed constraint violations.

Key design:
- Multipliers are stored as plain Python floats (not torch tensors) so they
  can be serialised without torch state dicts.
- The inner SB3 PPO handles all policy/value network training; LagrangianPPO
  adds constraint violation tracking on top.
- update_multipliers() is called by FMUTrainer at the end of each rollout.
"""
from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


class _LagrangianPenaltyVecEnv(VecEnvWrapper):
    """VecEnv wrapper that injects Lagrangian penalties into step rewards.

    The wrapped environment is expected to emit, per env step, an info dict key
    ``constraint_violations`` containing a mapping:

        {constraint_name: non_negative_violation_magnitude}

    Reward shaping applied at each step:

        r_lagrangian = r_env - sum_i(lambda_i * max(0, violation_i))

    Notes
    -----
    - ``multipliers`` is passed by reference so updates from dual-ascent are
      reflected immediately in reward shaping (no sync call required).
    - Unknown constraint names are ignored.
    """

    def __init__(self, venv: VecEnv, multipliers: dict[str, float]) -> None:
        super().__init__(venv)
        self._multipliers = multipliers

    def reset(self) -> np.ndarray:
        return self.venv.reset()

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        obs, rewards, dones, infos = self.venv.step_wait()
        shaped_rewards = rewards.astype(np.float32, copy=True)
        for env_idx, info in enumerate(infos):
            violations = info.get("constraint_violations", {})
            if not isinstance(violations, dict):
                continue
            penalty = 0.0
            for name, violation in violations.items():
                lam = float(self._multipliers.get(str(name), 0.0))
                if lam <= 0.0:
                    continue
                penalty += lam * max(0.0, float(violation))
            shaped_rewards[env_idx] = float(shaped_rewards[env_idx] - penalty)
            info["lagrangian_penalty"] = float(penalty)
        return obs, shaped_rewards, dones, infos


class LagrangianPPO:
    """SB3 PPO wrapper with trainable Lagrange constraint multipliers.

    Parameters
    ----------
    env:
        A VecEnv (or single Gymnasium env) for training.
    multiplier_lr:
        Learning rate for the dual variable (lambda) update.
    constraint_names:
        Names of constraints to track (e.g. ["T_comp", "surge_margin"]).
        A multiplier lambda_i is initialised to 0.0 for each.
    **ppo_kwargs:
        Passed directly to stable_baselines3.PPO.__init__.
    """

    def __init__(
        self,
        env: VecEnv | Any,
        multiplier_lr: float = 1e-3,
        constraint_names: list[str] | None = None,
        **ppo_kwargs: Any,
    ) -> None:
        self._multiplier_lr = multiplier_lr
        self._constraint_names: list[str] = list(constraint_names or [])
        # Initialise all multipliers at 0.0 (Rule: non-negative, start zero)
        self._multipliers: dict[str, float] = {
            name: 0.0 for name in self._constraint_names
        }
        self._penalty_env: VecEnv | Any
        if isinstance(env, VecEnv):
            self._penalty_env = _LagrangianPenaltyVecEnv(venv=env, multipliers=self._multipliers)
        else:
            # Fallback for unusual call-sites that pass a non-VecEnv env.
            self._penalty_env = env
        # Build inner SB3 PPO (receives Lagrangian-shaped rewards from wrapper)
        self._ppo = PPO(env=self._penalty_env, **ppo_kwargs)

    # -- Multiplier management ------------------------------------------------

    def update_multipliers(self, violations: dict[str, float]) -> None:
        """Update each lambda_i based on mean constraint violation.

        lambda_i <- max(0, lambda_i + alpha * violation_i)

        Parameters
        ----------
        violations:
            Mapping constraint_name -> mean violation value over latest rollout.
            Positive = constraint violated; zero or negative = satisfied.
            Unknown keys are silently ignored.
        """
        for name, violation in violations.items():
            if name not in self._multipliers:
                continue
            updated = self._multipliers[name] + self._multiplier_lr * violation
            self._multipliers[name] = max(0.0, updated)

    def get_multipliers(self) -> dict[str, float]:
        """Return a copy of current lambda values."""
        return dict(self._multipliers)

    def compute_lagrangian_penalty(self, violations: dict[str, float]) -> float:
        """Compute ``sum_i(lambda_i * max(0, violation_i))`` for one step."""
        penalty = 0.0
        for name, violation in violations.items():
            lam = float(self._multipliers.get(name, 0.0))
            if lam <= 0.0:
                continue
            penalty += lam * max(0.0, float(violation))
        return float(penalty)

    # -- Training / inference --------------------------------------------------

    def learn(self, total_timesteps: int, callback: Any = None) -> "LagrangianPPO":
        """Run PPO training for total_timesteps.

        Returns self for chaining.
        """
        self._ppo.learn(total_timesteps=total_timesteps, callback=callback)
        return self

    def predict(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, Any]:
        """Delegate to inner PPO.predict."""
        return self._ppo.predict(obs, deterministic=deterministic)

    # -- Serialisation ---------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights (SB3 zip) and multiplier state.

        Writes two files:
          <path>.zip  -- SB3 PPO model weights
          <path>_multipliers.pkl  -- Lagrange multiplier dict
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._ppo.save(path)
        multiplier_path = path + "_multipliers.pkl"
        with open(multiplier_path, "wb") as f:
            pickle.dump(
                {
                    "multipliers": self._multipliers,
                    "multiplier_lr": self._multiplier_lr,
                    "constraint_names": self._constraint_names,
                },
                f,
            )

    @classmethod
    def load(cls, path: str, env: Any = None) -> "LagrangianPPO":
        """Load a saved LagrangianPPO.

        Parameters
        ----------
        path:
            Path prefix (without .zip extension) used when save() was called.
        env:
            Environment to attach (optional; needed for further training).
        """
        multiplier_path = path + "_multipliers.pkl"
        if os.path.isfile(multiplier_path):
            with open(multiplier_path, "rb") as f:
                meta = pickle.load(f)
        else:
            # Pre-fix checkpoint: multipliers pkl was not written (inner PPO
            # was saved instead of LagrangianPPO). Resume with fresh multipliers.
            meta = {"multiplier_lr": 1e-3, "constraint_names": [], "multipliers": {}}

        obj = cls.__new__(cls)
        obj._multiplier_lr = meta["multiplier_lr"]
        obj._constraint_names = meta["constraint_names"]
        obj._multipliers = meta["multipliers"]
        if isinstance(env, VecEnv):
            obj._penalty_env = _LagrangianPenaltyVecEnv(venv=env, multipliers=obj._multipliers)
        else:
            obj._penalty_env = env
        obj._ppo = PPO.load(path, env=obj._penalty_env)
        return obj

    # -- Accessors for CheckpointManager / FMUTrainer -------------------------

    @property
    def policy(self):
        """Delegate policy access to inner PPO."""
        return self._ppo.policy

    @property
    def num_timesteps(self) -> int:
        """Total timesteps trained so far (from inner PPO)."""
        return self._ppo.num_timesteps
