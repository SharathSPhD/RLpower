"""Curriculum scheduling: phase definitions, metrics tracking, state machine."""
from sco2rl.curriculum.phase import CurriculumPhase, PhaseConfig
from sco2rl.curriculum.metrics_observer import MetricsObserver
from sco2rl.curriculum.scheduler import CurriculumScheduler

__all__ = ["CurriculumPhase", "PhaseConfig", "MetricsObserver", "CurriculumScheduler"]

