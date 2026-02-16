"""FMU interface — abstract base class for all FMU adapters.

Both MockFMU (unit tests) and FMPyAdapter (real FMU) implement this contract.
SCO2FMUEnv depends only on FMUInterface, never on concrete implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FMUInterface(ABC):
    """Contract for step-based FMU adapters used by SCO2FMUEnv."""

    @abstractmethod
    def initialize(self, start_time: float, stop_time: float, step_size: float) -> None:
        """Allocate resources and set the FMU to initialized state."""

    @abstractmethod
    def set_inputs(self, inputs: dict[str, float]) -> None:
        """Write action values into the FMU before the next do_step call."""

    @abstractmethod
    def do_step(self, current_time: float, step_size: float) -> bool:
        """Advance simulation by step_size seconds.

        Returns True on success, False on solver failure.
        """

    @abstractmethod
    def get_outputs(self) -> dict[str, float]:
        """Return the current output variables as a name→value mapping."""

    @abstractmethod
    def get_outputs_as_array(self) -> np.ndarray:
        """Return outputs in obs_vars order as a float32 numpy array."""

    @abstractmethod
    def reset(self) -> None:
        """Re-initialise the FMU to the design-point state (t=0)."""

    @abstractmethod
    def close(self) -> None:
        """Release any resources (file handles, processes, sockets)."""
