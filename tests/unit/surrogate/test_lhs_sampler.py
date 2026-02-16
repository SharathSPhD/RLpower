"""Unit tests for LatinHypercubeSampler -- TDD RED phase."""
from __future__ import annotations
import numpy as np
import pytest
from scipy.stats import qmc

LHS_CONFIG = {
    "parameter_ranges": {
        "T_exhaust_c": {"min": 200.0, "max": 1200.0},
        "mdot_exhaust_kg_s": {"min": 10.0, "max": 100.0},
        "W_net_setpoint_mw": {"min": 2.0, "max": 12.0},
    }
}

@pytest.fixture
def sampler():
    from sco2rl.surrogate.lhs_sampler import LatinHypercubeSampler
    return LatinHypercubeSampler(config=LHS_CONFIG, seed=42)

class TestLatinHypercubeSamplerShape:
    def test_sample_returns_correct_shape(self, sampler):
        samples = sampler.sample(100)
        assert samples.shape == (100, 3)

    def test_sample_small_n(self, sampler):
        assert sampler.sample(1).shape == (1, 3)

    def test_sample_large_n(self, sampler):
        assert sampler.sample(1000).shape == (1000, 3)

class TestLatinHypercubeSamplerBounds:
    def test_sample_within_bounds(self, sampler):
        samples = sampler.sample(500)
        lower, upper = sampler.get_bounds()
        for dim in range(3):
            assert np.all(samples[:, dim] >= lower[dim])
            assert np.all(samples[:, dim] <= upper[dim])

    def test_get_bounds_returns_correct_values(self, sampler):
        lower, upper = sampler.get_bounds()
        assert lower.shape == (3,)
        assert upper.shape == (3,)
        assert lower[0] == pytest.approx(200.0)
        assert upper[0] == pytest.approx(1200.0)
        assert lower[1] == pytest.approx(10.0)
        assert upper[1] == pytest.approx(100.0)
        assert lower[2] == pytest.approx(2.0)
        assert upper[2] == pytest.approx(12.0)

    def test_get_bounds_returns_numpy_arrays(self, sampler):
        lower, upper = sampler.get_bounds()
        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)

class TestLatinHypercubeSamplerCoverage:
    def test_lhs_better_coverage_than_random(self):
        from sco2rl.surrogate.lhs_sampler import LatinHypercubeSampler
        n = 200
        sampler = LatinHypercubeSampler(config=LHS_CONFIG, seed=42)
        lhs_samples = sampler.sample(n)
        lower, upper = sampler.get_bounds()
        lhs_unit = (lhs_samples - lower) / (upper - lower)
        rng = np.random.default_rng(42)
        random_unit = rng.uniform(0, 1, size=(n, 3))
        lhs_disc = qmc.discrepancy(lhs_unit)
        rand_disc = qmc.discrepancy(random_unit)
        assert lhs_disc < rand_disc

class TestLatinHypercubeSamplerReproducibility:
    def test_reproducible_with_same_seed(self):
        from sco2rl.surrogate.lhs_sampler import LatinHypercubeSampler
        s1 = LatinHypercubeSampler(config=LHS_CONFIG, seed=42)
        s2 = LatinHypercubeSampler(config=LHS_CONFIG, seed=42)
        np.testing.assert_array_equal(s1.sample(50), s2.sample(50))

    def test_different_seeds_differ(self):
        from sco2rl.surrogate.lhs_sampler import LatinHypercubeSampler
        s1 = LatinHypercubeSampler(config=LHS_CONFIG, seed=1)
        s2 = LatinHypercubeSampler(config=LHS_CONFIG, seed=2)
        assert not np.allclose(s1.sample(50), s2.sample(50))
