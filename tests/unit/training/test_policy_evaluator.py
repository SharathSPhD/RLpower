"""Unit tests for PolicyEvaluator.

TDD: Written before implementation. Uses MockFMU + SCO2FMUEnv.
"""
from __future__ import annotations

import numpy as np
import pytest

from sco2rl.simulation.fmu.mock_fmu import MockFMU
from sco2rl.environment.sco2_env import SCO2FMUEnv

# ── Observation / action config ────────────────────────────────────────────────

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

EVAL_CONFIG = {
    "n_eval_episodes": 3,
    "T_comp_inlet_var": "T_compressor_inlet",
    "deterministic": True,
}


class ConstantModel:
    """Always predicts zero action (design-point normalized)."""

    def predict(self, obs, deterministic=True):
        n_act = 4
        return np.zeros(n_act, dtype=np.float32), None


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
def evaluator(env):
    from sco2rl.training.policy_evaluator import PolicyEvaluator
    return PolicyEvaluator(env=env, config=EVAL_CONFIG)


@pytest.fixture
def model():
    return ConstantModel()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_evaluate_returns_evaluation_metrics(evaluator, model):
    from sco2rl.training.policy_evaluator import EvaluationMetrics
    result = evaluator.evaluate(model)
    assert isinstance(result, EvaluationMetrics)


def test_n_episodes_matches_config(evaluator, model):
    result = evaluator.evaluate(model)
    assert result.n_episodes == EVAL_CONFIG["n_eval_episodes"]


def test_per_episode_rewards_length(evaluator, model):
    result = evaluator.evaluate(model)
    assert len(result.per_episode_rewards) == EVAL_CONFIG["n_eval_episodes"]


def test_mean_reward_in_reasonable_range(evaluator, model):
    result = evaluator.evaluate(model)
    assert np.isfinite(result.mean_reward)


def test_violation_rate_in_zero_one(evaluator, model):
    result = evaluator.evaluate(model)
    assert 0.0 <= result.violation_rate <= 1.0


def test_T_comp_inlet_min_is_float(evaluator, model):
    result = evaluator.evaluate(model)
    assert np.isfinite(result.T_comp_inlet_min)
    assert isinstance(result.T_comp_inlet_min, float)
