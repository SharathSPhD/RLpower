"""Surrogate model layer: FNO training via PhysicsNeMo on GPU-vectorized environments.

Heavy torch/gymnasium imports are lazy to keep this package importable in
lightweight environments (e.g. unit tests without full ML stack).
"""

# Only truly lightweight modules are imported eagerly
from sco2rl.surrogate.lhs_sampler import LatinHypercubeSampler
from sco2rl.surrogate.trajectory_collector import TrajectoryCollector


def __getattr__(name: str):
    if name in ("FNO1d", "FNOBlock", "SpectralConv1d"):
        from sco2rl.surrogate.fno_model import FNO1d, FNOBlock, SpectralConv1d
        globals().update({"FNO1d": FNO1d, "FNOBlock": FNOBlock, "SpectralConv1d": SpectralConv1d})
        return globals()[name]
    if name == "SurrogateEnv":
        from sco2rl.surrogate.surrogate_env import SurrogateEnv
        globals()["SurrogateEnv"] = SurrogateEnv
        return SurrogateEnv
    if name in ("FidelityGate", "FidelityReport"):
        from sco2rl.surrogate.fidelity_gate import FidelityGate, FidelityReport
        globals().update({"FidelityGate": FidelityGate, "FidelityReport": FidelityReport})
        return globals()[name]
    if name == "TrajectoryDataset":
        from sco2rl.surrogate.trajectory_dataset import TrajectoryDataset
        globals()["TrajectoryDataset"] = TrajectoryDataset
        return TrajectoryDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LatinHypercubeSampler",
    "TrajectoryCollector",
    "TrajectoryDataset",
    "FNO1d",
    "FNOBlock",
    "SpectralConv1d",
    "SurrogateEnv",
    "FidelityGate",
    "FidelityReport",
]
