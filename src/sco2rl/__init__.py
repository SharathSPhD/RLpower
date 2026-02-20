"""sco2rl — Deep RL for autonomous control of sCO₂ WHR Brayton cycle."""

__version__ = "0.1.0"

# Control library (reusable SCOPE-compatible controllers)
from sco2rl.control.interfaces import Controller
from sco2rl.control.pid import PIDController
from sco2rl.control.multi_loop_pid import MultiLoopPID
from sco2rl.control.rl_controller import RLController

# Analysis library (step response, frequency response, scenario runner)
from sco2rl.analysis.metrics import (
    StepResponseResult,
    FrequencyResponseResult,
    ControlMetricsSummary,
)
from sco2rl.analysis.scenario_runner import ScenarioRunner, ControlScenario

__all__ = [
    # Control
    "Controller",
    "PIDController",
    "MultiLoopPID",
    "RLController",
    # Analysis
    "StepResponseResult",
    "FrequencyResponseResult",
    "ControlMetricsSummary",
    "ScenarioRunner",
    "ControlScenario",
]
