"""Unit tests for CrossValidator.

TDD: Written before implementation.
Uses MockEvaluator to inject controlled EvaluationMetrics.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime

import numpy as np
import pytest

from sco2rl.simulation.fmu.mock_fmu import MockFMU
from sco2rl.environment.sco2_env import SCO2FMUEnv

# ── Shared config (reuse from policy_evaluator tests) ─────────────────────────

OBS_VARS = [
    "T_compressor_inlet", "T_turbine_inlet", "P_high", "P_low",
    "mdot_turbine", "mdot_main_compressor", "W_turbine", "W_main_compressor",
    "W_net", "eta_thermal", "surge_margin_main",
]
ACTION_VARS = [
    "bypass_valve_opening", "igv_angle_normalized",
    "inventory_valve_opening", "cooling_flow_normalized",
]
DESIGN_POINT = {
    "T_compressor_inlet": 33.0,
    "T_turbine_inlet": 750.0,
    "P_high": 20.0,
    "P_low": 7.7,
    "mdot_turbine": 70.0,
    "mdot_main_compressor": 70.0,
    "W_turbine": 14.5,
    "W_main_compressor": 4.0,
    "W_net": 10.0,
    "eta_thermal": 0.40,
    "surge_margin_main": 0.20,
}
ENV_CONFIG = {
    "obs_vars": OBS_VARS,
    "obs_bounds": {v: (0.0, 1500.0) for v in OBS_VARS},
    "action_vars": ACTION_VARS,
    "action_config": {
        v: {"min": 0.0, "max": 1.0, "rate": 0.05}
        for v in ACTION_VARS
    },
    "history_steps": 1,
    "step_size": 5.0,
    "episode_max_steps": 20,
    "reward": {
        "w_tracking": 1.0, "w_efficiency": 0.3, "w_smoothness": 0.1,
        "rated_power_mw": 10.0, "design_efficiency": 0.40,
        "terminal_failure_reward": -100.0,
    },
    "safety": {"T_compressor_inlet_min": 32.2, "surge_margin_min": 0.05},
    "setpoint": {"W_net": 5.0},
}
CV_CONFIG = {
    "n_eval_episodes": 3,
    "T_comp_inlet_var": "T_compressor_inlet",
    "deterministic": True,
    "selection_metric": "mean_reward",
}


# ── MockEvaluator for injecting controlled metrics ────────────────────────────

class MockEvaluator:
    """Injects pre-built EvaluationMetrics, one per evaluate() call."""

    def __init__(self, metrics_sequence):
        self._iter = iter(metrics_sequence)

    def evaluate(self, model):
        return next(self._iter)


def make_metrics(mean_reward: float, violation_rate: float = 0.0, phase: int = 0):
    from sco2rl.training.policy_evaluator import EvaluationMetrics
    return EvaluationMetrics(
        mean_reward=mean_reward,
        std_reward=0.1,
        mean_episode_length=20.0,
        violation_rate=violation_rate,
        T_comp_inlet_min=33.0,
        T_comp_inlet_mean=33.5,
        n_episodes=3,
        phase=phase,
        per_episode_rewards=[mean_reward] * 3,
    )


class DummyModel:
    def predict(self, obs, deterministic=True):
        return np.zeros(4, dtype=np.float32), None


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    fmu = MockFMU(
        obs_vars=OBS_VARS,
        action_vars=ACTION_VARS,
        design_point=DESIGN_POINT,
        seed=42,
    )
    fmu.initialize(start_time=0.0, stop_time=10000.0, step_size=5.0)
    return SCO2FMUEnv(fmu=fmu, config=ENV_CONFIG)


@pytest.fixture
def cross_validator_factory(env):
    """Returns a factory that builds CrossValidator with injected evaluator."""
    from sco2rl.training.cross_validator import CrossValidator

    def _make(metrics_a, metrics_b):
        evaluator = MockEvaluator([metrics_a, metrics_b])
        return CrossValidator(env=env, config=CV_CONFIG, evaluator=evaluator)

    return _make


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_compare_returns_cross_validation_report(cross_validator_factory):
    from sco2rl.training.cross_validator import CrossValidationReport
    cv = cross_validator_factory(make_metrics(5.0), make_metrics(3.0))
    report = cv.compare(DummyModel(), DummyModel())
    assert isinstance(report, CrossValidationReport)


def test_selected_path_is_path_a_or_b(cross_validator_factory):
    cv = cross_validator_factory(make_metrics(5.0), make_metrics(3.0))
    report = cv.compare(DummyModel(), DummyModel())
    assert report.selected_path in {"path_a", "path_b"}


def test_selection_reason_is_non_empty_string(cross_validator_factory):
    cv = cross_validator_factory(make_metrics(5.0), make_metrics(3.0))
    report = cv.compare(DummyModel(), DummyModel())
    assert isinstance(report.selection_reason, str)
    assert len(report.selection_reason) > 0


def test_both_metrics_present(cross_validator_factory):
    from sco2rl.training.policy_evaluator import EvaluationMetrics
    cv = cross_validator_factory(make_metrics(5.0), make_metrics(3.0))
    report = cv.compare(DummyModel(), DummyModel())
    assert isinstance(report.path_a_metrics, EvaluationMetrics)
    assert isinstance(report.path_b_metrics, EvaluationMetrics)


def test_higher_reward_selected(cross_validator_factory):
    """path_a has higher mean_reward → must be selected."""
    cv = cross_validator_factory(make_metrics(10.0), make_metrics(1.0))
    report = cv.compare(DummyModel(), DummyModel())
    assert report.selected_path == "path_a"


def test_save_and_load_report(cross_validator_factory):
    from sco2rl.training.cross_validator import CrossValidator
    cv = cross_validator_factory(make_metrics(7.0), make_metrics(4.0))
    report = cv.compare(DummyModel(), DummyModel())

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "report.json")
        cv.save_report(report, path)
        loaded = CrossValidator.load_report(path)

    assert loaded.selected_path == report.selected_path


def test_timestamp_is_iso_format(cross_validator_factory):
    cv = cross_validator_factory(make_metrics(5.0), make_metrics(3.0))
    report = cv.compare(DummyModel(), DummyModel())
    # Should not raise
    dt = datetime.fromisoformat(report.timestamp)
    assert dt is not None
