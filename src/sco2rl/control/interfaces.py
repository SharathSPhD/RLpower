"""Abstract Controller interface for all sco2rl controllers.

All concrete controllers (PID, RL, MPC) implement this interface so that
analysis tooling, evaluation harnesses, and notebooks can treat them
uniformly without caring about implementation details.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Controller(ABC):
    """Standard interface for all controllers used with SCO2FMUEnv.

    The interface is intentionally minimal and compatible with the Stable-
    Baselines3 policy API so that any controller can be passed to the
    existing PolicyEvaluator and ScenarioRunner infrastructure.

    ``predict()`` returns ``(action, state)`` where ``state`` is always
    ``None`` for non-recurrent controllers. This matches the SB3 convention.
    """

    @abstractmethod
    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        """Compute action for the given observation.

        Parameters
        ----------
        obs:
            Flat observation vector of shape ``(obs_dim,)`` or
            ``(1, obs_dim)`` (batched).  Physical units as returned by
            ``SCO2FMUEnv`` — no VecNormalize applied.
        deterministic:
            Ignored by deterministic controllers; forwarded to stochastic
            policies (RL).

        Returns
        -------
        action : np.ndarray
            Action vector in ``[-1, 1]`` of shape ``(action_dim,)``.
        state : None
            Always ``None`` for non-recurrent controllers.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset all internal state (integral accumulator, history, etc.).

        Must be called between episodes.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable controller name (used in plot legends and logs)."""
