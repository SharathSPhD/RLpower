"""sco2rl.control — Reusable controller library for sCO₂ Brayton cycle.

Public API
----------
Controller
    Abstract base class for all controllers.
PIDController
    Single-loop full PID with anti-windup and derivative filter.
MultiLoopPID
    Multi-channel PID for the 4-actuator simple recuperated cycle.
RLController
    Thin wrapper making any SB3/LagrangianPPO policy a Controller.

Quick start::

    from sco2rl.control import MultiLoopPID, RLController

    # Use improved PID baseline
    pid = MultiLoopPID(config=pid_config)

    # Load trained RL policy
    rl = RLController.from_checkpoint("artifacts/checkpoints/run01/final")

    # Both satisfy the Controller interface
    action, _ = pid.predict(obs)
    action, _ = rl.predict(obs)
"""
from sco2rl.control.interfaces import Controller
from sco2rl.control.pid import PIDController
from sco2rl.control.multi_loop_pid import MultiLoopPID
from sco2rl.control.rl_controller import RLController

__all__ = [
    "Controller",
    "PIDController",
    "MultiLoopPID",
    "RLController",
]
