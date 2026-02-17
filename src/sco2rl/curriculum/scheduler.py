"""CurriculumScheduler — state machine tracking phase and advancement.

No backward transitions (allow_regression=False per config).
All state is serialisable for RULE-C4 checkpointing.
"""
from __future__ import annotations

from sco2rl.curriculum.phase import CurriculumPhase, PhaseConfig
from sco2rl.curriculum.metrics_observer import MetricsObserver


class CurriculumScheduler:
    """Tracks current curriculum phase; advances when MetricsObserver signals ready.

    Parameters
    ----------
    phase_configs:
        One PhaseConfig per phase, in order (index 0 = STEADY_STATE, ...).
    observer:
        MetricsObserver instance; reset() is called automatically on advance.
    """

    def __init__(
        self,
        phase_configs: list[PhaseConfig],
        observer: MetricsObserver,
    ) -> None:
        if len(phase_configs) != len(CurriculumPhase):
            raise ValueError(
                f"Expected {len(CurriculumPhase)} phase configs, "
                f"got {len(phase_configs)}"
            )
        self._configs: dict[CurriculumPhase, PhaseConfig] = {
            cfg.phase: cfg for cfg in phase_configs
        }
        self._observer = observer
        self._phase = CurriculumPhase.STEADY_STATE

    # ── Public API ─────────────────────────────────────────────────────────────

    def record_episode(self, reward: float, violation_fraction: float) -> bool:
        """Record episode metrics and check for advancement.

        Returns True if the phase was advanced, False otherwise.
        """
        self._observer.record_episode(reward, violation_fraction)
        if self._observer.should_advance(self._phase):
            return self.advance()
        return False

    def get_phase(self) -> CurriculumPhase:
        """Current curriculum phase."""
        return self._phase

    def get_phase_config(self) -> PhaseConfig:
        """Configuration for the current phase."""
        return self._configs[self._phase]

    def advance(self) -> bool:
        """Advance to the next phase.

        Returns False if already at EMERGENCY_TRIP (no regression, no wrap).
        Updates observer's advance_threshold to the new phase's configured value.
        """
        if self._phase == CurriculumPhase.EMERGENCY_TRIP:
            return False
        self._phase = CurriculumPhase(int(self._phase) + 1)
        # Update threshold to the new phase's configured value so per-phase
        # advancement thresholds from curriculum.yaml are respected.
        new_cfg = self._configs[self._phase]
        self._observer.set_advance_threshold(new_cfg.advance_threshold)
        self._observer.reset()
        return True

    def state_dict(self) -> dict:
        """Serialisable state for RULE-C4 checkpointing."""
        return {
            "phase": int(self._phase),
            "n_episodes": self._observer.n_episodes,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from a checkpoint state dict."""
        self._phase = CurriculumPhase(int(state["phase"]))
        # Replay n_episodes worth of neutral records to restore episode count
        n = int(state["n_episodes"])
        self._observer.reset()
        for _ in range(n):
            # Use neutral values: below threshold so no spurious advance
            self._observer.record_episode(reward=0.0, violation_fraction=0.0)
