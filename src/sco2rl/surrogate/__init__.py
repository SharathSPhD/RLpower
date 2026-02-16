"""Surrogate model layer: FNO training via PhysicsNeMo on GPU-vectorized environments."""

from sco2rl.surrogate.fno_model import FNO1d, FNOBlock, SpectralConv1d
from sco2rl.surrogate.surrogate_env import SurrogateEnv
from sco2rl.surrogate.fidelity_gate import FidelityGate, FidelityReport

__all__ = [
    "FNO1d",
    "FNOBlock",
    "SpectralConv1d",
    "SurrogateEnv",
    "FidelityGate",
    "FidelityReport",
]
