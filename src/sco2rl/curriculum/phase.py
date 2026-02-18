"""Curriculum phase definitions for sCO₂ RL training.

7 progressive scenarios from steady-state to emergency turbine trip.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class CurriculumPhase(IntEnum):
    """Training difficulty phases — ordered from easiest (0) to hardest (6)."""
    STEADY_STATE = 0    # Fixed heat source, no load changes
    LOAD_FOLLOW = 1     # ±30% gradual load following
    AMBIENT_TEMP = 2    # ±10°C ambient temperature disturbance
    EAF_TRANSIENTS = 3  # EAF heat source 200–1200°C cycles
    LOAD_REJECTION = 4  # 50% rapid load rejection (30 seconds)
    COLD_STARTUP = 5    # Cold startup through CO₂ critical region
    EMERGENCY_TRIP = 6  # Emergency turbine trip recovery


@dataclass
class PhaseConfig:
    """Configuration for a single curriculum phase."""
    phase: CurriculumPhase
    advance_threshold: float    # Mean reward over window required to advance
    min_episodes: int           # Minimum episodes before advancing
    window_size: int            # Rolling window size for mean reward
    violation_rate_limit: float # Max fraction of episodes with constraint violations
    disturbance_amplitude: float  # Disturbance magnitude for this phase
    episode_length_steps: int = 720  # Per-phase episode length (FMU control steps)
