"""Unit tests for SeedManager.

Tests run inside Docker on DGX Spark (ARM64). See RULE-D1, RULE-D2.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from sco2rl.utils.seeds import GLOBAL_SEED, SeedManager


class TestSeedManager:
    def test_global_seed_is_42(self) -> None:
        """GLOBAL_SEED constant is 42 as documented."""
        assert GLOBAL_SEED == 42

    def test_set_all_makes_numpy_deterministic(self) -> None:
        """Two calls to set_all(42) produce the same numpy random output."""
        SeedManager.set_all(42)
        a = np.random.rand(10)
        SeedManager.set_all(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_set_all_makes_stdlib_random_deterministic(self) -> None:
        """Two calls to set_all(42) produce the same stdlib random output."""
        SeedManager.set_all(42)
        a = [random.random() for _ in range(10)]
        SeedManager.set_all(42)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_set_all_different_seeds_different_outputs(self) -> None:
        """Different seeds produce different random outputs."""
        SeedManager.set_all(42)
        a = np.random.rand(10)
        SeedManager.set_all(99)
        b = np.random.rand(10)
        assert not np.array_equal(a, b)

    def test_set_all_default_seed_is_global_seed(self) -> None:
        """set_all() with no argument uses GLOBAL_SEED."""
        SeedManager.set_all(GLOBAL_SEED)
        a = np.random.rand(10)
        SeedManager.set_all()
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)
