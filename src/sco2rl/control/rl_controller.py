"""RLController — wraps a Stable-Baselines3 / LagrangianPPO policy as a Controller.

Provides a uniform interface so that RL policies can be used interchangeably
with PID controllers in analysis and evaluation scripts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sco2rl.control.interfaces import Controller


class RLController(Controller):
    """Wraps an SB3-compatible policy as a ``Controller``.

    The policy must implement ``predict(obs, deterministic) → (action, state)``,
    which is satisfied by any SB3 ``BaseAlgorithm`` or the project's
    ``LagrangianPPO``.

    Parameters
    ----------
    policy:
        Any object with a ``predict(obs, deterministic=True)`` method.
    controller_name:
        Human-readable name shown in plot legends.
    """

    def __init__(self, policy: Any, controller_name: str = "RL") -> None:
        self._policy = policy
        self._name = str(controller_name)

    # ── Controller interface ───────────────────────────────────────────────────

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        action, state = self._policy.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32), None

    def reset(self) -> None:
        """No-op for stateless RL policies (LSTM policies should override)."""

    @property
    def name(self) -> str:
        return self._name

    # ── Factory methods ────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        env: Any = None,
        algorithm_cls: Any = None,
        controller_name: str = "RL",
    ) -> "RLController":
        """Load a saved SB3 checkpoint and wrap it as an RLController.

        Parameters
        ----------
        checkpoint_path:
            Path to a saved model directory or ``.zip`` file.
        env:
            Optional environment for observation/action space validation.
            Pass ``None`` to skip env validation (useful for analysis without
            a live FMU).
        algorithm_cls:
            SB3 algorithm class (e.g. ``PPO``).  If ``None``, tries to import
            ``LagrangianPPO`` from the project, then falls back to ``PPO``.
        controller_name:
            Name shown in plot legends.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        if algorithm_cls is None:
            try:
                from sco2rl.training.lagrangian_ppo import LagrangianPPO  # type: ignore

                algorithm_cls = LagrangianPPO
            except ImportError:
                try:
                    from stable_baselines3 import PPO  # type: ignore

                    algorithm_cls = PPO
                except ImportError as exc:
                    raise ImportError(
                        "Cannot load checkpoint: neither LagrangianPPO nor "
                        "stable_baselines3.PPO is available."
                    ) from exc

        policy = algorithm_cls.load(str(path), env=env)
        return cls(policy, controller_name=controller_name)
