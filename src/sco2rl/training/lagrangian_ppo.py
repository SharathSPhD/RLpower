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
from stable_baselines3.common.vec_env import VecEnv


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
        # Build inner SB3 PPO
        self._ppo = PPO(env=env, **ppo_kwargs)

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
        with open(multiplier_path, "rb") as f:
            meta = pickle.load(f)

        obj = cls.__new__(cls)
        obj._multiplier_lr = meta["multiplier_lr"]
        obj._constraint_names = meta["constraint_names"]
        obj._multipliers = meta["multipliers"]
        obj._ppo = PPO.load(path, env=env)
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
