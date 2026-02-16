"""Deterministic seed management for reproducible RL training.

SeedManager.set_all(seed=42) must be called at the start of every unit test
(via conftest fixture) and at the start of every training run.

Tests run inside Docker on DGX Spark (ARM64). See RULE-D1, RULE-D2.
"""

from __future__ import annotations

import random

import numpy as np

GLOBAL_SEED: int = 42
"""Default seed used across all unit tests and reproducibility checkpoints."""


class SeedManager:
    """Sets random seeds for numpy, stdlib random, and optionally torch.

    Usage::

        SeedManager.set_all(42)

    This class has only class-level methods and no instance state; it acts as
    a namespace for seed-setting operations.
    """

    @classmethod
    def set_all(cls, seed: int = GLOBAL_SEED) -> None:
        """Set seeds for numpy, stdlib random, and torch (if available).

        Args:
            seed: Integer seed. Defaults to GLOBAL_SEED (42).
        """
        random.seed(seed)
        np.random.seed(seed)
        cls._set_torch_seed(seed)

    @staticmethod
    def _set_torch_seed(seed: int) -> None:
        """Set torch seed if torch is installed (optional dependency at Stage 0)."""
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            # torch is an optional dependency at Stage 0; skip silently
            pass
