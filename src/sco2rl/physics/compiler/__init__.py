"""sco2rl.physics.compiler — OMPython compiler pipeline for sCO₂ FMU export.

Public API:
    OMCSessionWrapper     — context-managed OMPython session with retry
    FMUExporter           — CycleModel → .mo → .fmu via OMC
    FMUExportError        — raised on OMC compilation failure
    CyclePhysicsValidator — validates RULE-P1, RULE-P5, energy balance
    PhysicsViolation      — raised on physics constraint violation
"""
from sco2rl.physics.compiler.fmu_exporter import FMUExportError, FMUExporter
from sco2rl.physics.compiler.omc_session import OMCSessionWrapper
from sco2rl.physics.compiler.physics_validator import CyclePhysicsValidator, PhysicsViolation

__all__ = [
    "OMCSessionWrapper",
    "FMUExporter",
    "FMUExportError",
    "CyclePhysicsValidator",
    "PhysicsViolation",
]
