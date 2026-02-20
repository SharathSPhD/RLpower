"""sco2rl.analysis — Control system analysis tools for sCO₂ Brayton cycle.

Provides step-response and frequency-domain analysis for comparing PID and
RL controllers across the 7-phase curriculum.

Public API
----------
StepResponseResult
    Dataclass with transient response metrics (overshoot, settling time, IAE…).
FrequencyResponseResult
    Dataclass with Bode data and stability margins (gain/phase margin).
ControlMetricsSummary
    Aggregated metrics for one (phase, scenario) PID vs. RL comparison.
ScenarioRunner
    Orchestrator that runs experiments across phases and serialises results.
ControlScenario
    Enum of available test scenarios.
build_mock_env
    Factory for MockFMU-backed SCO2FMUEnv (Colab-compatible, no real FMU).
build_mock_pid
    Factory for MultiLoopPID with MockFMU variable names.

Quick start::

    from sco2rl.analysis import ScenarioRunner, build_mock_env, build_mock_pid

    runner = ScenarioRunner(n_seeds=3)
    results = runner.run_all(
        env_factory=build_mock_env,
        pid_policy=build_mock_pid(),
        rl_policy=None,
        phases=[0, 1, 2],
    )
    ScenarioRunner.save(results, "data/control_analysis_all_phases.json")
"""
from sco2rl.analysis.metrics import (
    StepResponseResult,
    FrequencyResponseResult,
    ControlMetricsSummary,
)
from sco2rl.analysis.scenario_runner import (
    ScenarioRunner,
    ControlScenario,
    build_mock_env,
    build_mock_pid,
)

__all__ = [
    "StepResponseResult",
    "FrequencyResponseResult",
    "ControlMetricsSummary",
    "ScenarioRunner",
    "ControlScenario",
    "build_mock_env",
    "build_mock_pid",
]
