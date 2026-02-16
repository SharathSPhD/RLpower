"""Unit tests for FineTuner.

TDD: Written before implementation.
Uses MockFMU + SCO2FMUEnv with minimal steps for fast tests.
"""
from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from sco2rl.simulation.fmu.mock_fmu import MockFMU
from sco2rl.environment.sco2_env import SCO2FMUEnv

# ── Shared env config ─────────────────────────────────────────────────────────

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
    "episode_max_steps": 10,
    "reward": {
        "w_tracking": 1.0, "w_efficiency": 0.3, "w_smoothness": 0.1,
        "rated_power_mw": 10.0, "design_efficiency": 0.40,
        "terminal_failure_reward": -100.0,
    },
    "safety": {"T_compressor_inlet_min": 32.2, "surge_margin_min": 0.05},
    "setpoint": {"W_net": 5.0},
}


def make_test_env():
    """Build SCO2FMUEnv with MockFMU for unit tests."""
    fmu = MockFMU(
        obs_vars=OBS_VARS,
        action_vars=ACTION_VARS,
        design_point=DESIGN_POINT,
        seed=42,
    )
    fmu.initialize(start_time=0.0, stop_time=10000.0, step_size=5.0)
    return SCO2FMUEnv(fmu=fmu, config=ENV_CONFIG)


def make_dummy_checkpoint(tmp_path: str, curriculum_phase: int = 0) -> str:
    """Create a minimal RULE-C4 checkpoint for testing.

    Returns path to the checkpoint JSON metadata file.
    """
    env = make_test_env()
    vec_env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=16, batch_size=8)

    weights_path = os.path.join(tmp_path, "model.zip")
    model.save(weights_path)

    meta = {
        "model_path": weights_path,
        "vecnorm_stats": None,
        "curriculum_phase": curriculum_phase,
        "lagrange_multipliers": {},
        "total_timesteps": 1000,
        "step": 1000,
    }
    meta_path = os.path.join(tmp_path, "checkpoint.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    return meta_path


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def checkpoint_path(tmp_dir):
    return make_dummy_checkpoint(tmp_dir)


@pytest.fixture
def finetune_config(tmp_dir):
    return {
        "finetune_steps": 64,
        "finetune_lr": 5e-5,
        "checkpoint_dir": os.path.join(tmp_dir, "finetune"),
        "eval_freq": 32,
        "eval_episodes": 2,
        "seed": 42,
    }


@pytest.fixture
def fine_tuner(finetune_config):
    from sco2rl.training.fine_tuner import FineTuner

    def env_factory():
        return make_test_env()

    return FineTuner(env_factory=env_factory, config=finetune_config)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_finetune_returns_metrics_dict(fine_tuner, checkpoint_path):
    result = fine_tuner.finetune(checkpoint_path)
    assert isinstance(result, dict)


def test_metrics_has_total_timesteps(fine_tuner, checkpoint_path):
    result = fine_tuner.finetune(checkpoint_path)
    assert "total_timesteps" in result


def test_metrics_has_final_mean_reward(fine_tuner, checkpoint_path):
    result = fine_tuner.finetune(checkpoint_path)
    assert "final_mean_reward" in result
    assert math.isfinite(result["final_mean_reward"])


def test_metrics_has_checkpoint_path(fine_tuner, checkpoint_path):
    result = fine_tuner.finetune(checkpoint_path)
    assert "checkpoint_path" in result
    assert isinstance(result["checkpoint_path"], str)
    assert os.path.isfile(result["checkpoint_path"])


def test_finetune_increases_total_timesteps(fine_tuner, checkpoint_path):
    result = fine_tuner.finetune(checkpoint_path)
    assert result["total_timesteps"] > 1000  # original checkpoint had 1000


def test_model_property_not_none_after_finetune(fine_tuner, checkpoint_path):
    fine_tuner.finetune(checkpoint_path)
    assert fine_tuner.model is not None
