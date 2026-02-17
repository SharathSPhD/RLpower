"""MetricsObserver — tracks episode rewards and constraint violations.

Maintains a rolling window and fires advancement signals when thresholds are met.
All config via constructor dict (RULE-C2).
"""
from __future__ import annotations

from collections import deque
from typing import Any

from sco2rl.curriculum.phase import CurriculumPhase


class MetricsObserver:
    """Tracks episode reward and violation history; computes advancement readiness.

    Parameters
    ----------
    config:
        Dict with keys: window_size, advance_threshold, violation_rate_limit,
        min_episodes.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._window_size: int = int(config["window_size"])
        self._advance_threshold: float = float(config["advance_threshold"])
        self._violation_rate_limit: float = float(config["violation_rate_limit"])
        self._min_episodes: int = int(config["min_episodes"])

        self._rewards: deque[float] = deque(maxlen=self._window_size)
        self._violations: deque[float] = deque(maxlen=self._window_size)
        self._total_episodes: int = 0

    def record_episode(self, reward: float, violation_fraction: float) -> None:
        """Add one episode's metrics to the rolling window."""
        self._rewards.append(float(reward))
        self._violations.append(float(violation_fraction))
        self._total_episodes += 1

    def should_advance(self, current_phase: CurriculumPhase) -> bool:
        """True if ready to advance: window full, reward ≥ threshold, violations ≤ limit."""
        if self._total_episodes < self._min_episodes:
            return False
        if len(self._rewards) < self._window_size:
            return False
        if self.get_mean_reward() < self._advance_threshold:
            return False
        if self.get_violation_rate() > self._violation_rate_limit:
            return False
        return True

    def get_mean_reward(self) -> float:
        """Rolling mean reward over the current window."""
        if not self._rewards:
            return 0.0
        return sum(self._rewards) / len(self._rewards)

    def get_violation_rate(self) -> float:
        """Mean constraint violation fraction over the current window."""
        if not self._violations:
            return 0.0
        return sum(self._violations) / len(self._violations)

    def set_advance_threshold(self, threshold: float) -> None:
        """Update the advancement threshold (called by scheduler on phase advance)."""
        self._advance_threshold = float(threshold)

    def reset(self) -> None:
        """Clear history (called on phase advance)."""
        self._rewards.clear()
        self._violations.clear()
        self._total_episodes = 0

    @property
    def n_episodes(self) -> int:
        """Total episodes recorded since last reset."""
        return self._total_episodes
