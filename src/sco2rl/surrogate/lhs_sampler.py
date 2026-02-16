"""LatinHypercubeSampler -- space-filling parameter sampling for surrogate training.

Uses scipy.stats.qmc.LatinHypercube to generate samples that cover the
parameter space more uniformly than random sampling.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc


class LatinHypercubeSampler:
    """Generate Latin Hypercube Samples for surrogate training data collection.

    Parameters
    ----------
    config:
        Dict with key "parameter_ranges" mapping parameter names to
        {"min": float, "max": float} bounds. Parameter order is preserved
        as insertion order (Python 3.7+).
    seed:
        Random seed for reproducibility.
    """

    def __init__(self, config: dict, seed: int = 42) -> None:
        self._config = config
        self._seed = seed

        param_ranges = config["parameter_ranges"]
        self._param_names: list[str] = list(param_ranges.keys())
        self._n_dims = len(self._param_names)

        self._lower = np.array(
            [param_ranges[k]["min"] for k in self._param_names], dtype=np.float64
        )
        self._upper = np.array(
            [param_ranges[k]["max"] for k in self._param_names], dtype=np.float64
        )

    def sample(self, n: int) -> np.ndarray:
        """Generate n Latin Hypercube samples.

        Parameters
        ----------
        n:
            Number of samples to generate.

        Returns
        -------
        np.ndarray
            Array of shape (n, n_dims) with values scaled to [min, max] bounds.
        """
        sampler = qmc.LatinHypercube(d=self._n_dims, seed=self._seed)
        unit_samples = sampler.random(n)  # shape (n, n_dims) in [0, 1]
        scaled = qmc.scale(unit_samples, self._lower, self._upper)
        return scaled

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) bounds arrays of shape (n_dims,).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (lower_bounds, upper_bounds) each of shape (n_dims,).
        """
        return self._lower.copy(), self._upper.copy()
