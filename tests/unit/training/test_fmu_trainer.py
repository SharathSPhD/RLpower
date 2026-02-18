"""Unit tests for FMUTrainer.

All tests use MockFMU (RULE-C1).
Tests verify setup(), evaluate() return types, and deterministic evaluation.
"""
from __future__ import annotations

import numpy as np
import pytest

from sco2rl.simulation.fmu.mock_fmu import MockFMU

# ---- Constants --------------------------------------------------------------

OBS_VARS = [
    "T_turbine_inlet", "T_compressor_inlet", "P_high", "P_low",
    "W_turbine", "W_main_compressor", "W_net", "eta_thermal", "surge_margin_main",
]
ACTION_VARS = [
    "bypass_valve_opening", "igv_angle_normalized",
    "inventory_valve_opening", "cooling_flow_normalized",
]
DESIGN_POINT = {
    "T_turbine_inlet": 750.0, "T_compressor_inlet": 33.0,
    "P_high": 20.0, "P_low": 7.7, "W_turbine": 14.5,
    "W_main_compressor": 4.0, "W_net": 10.0, "eta_thermal": 0.47,
    "surge_margin_main": 0.20,
}
OBS_BOUNDS = {
    "T_turbine_inlet": (600.0, 850.0), "T_compressor_inlet": (30.0, 45.0),
    "P_high": (14.0, 26.0), "P_low": (6.5, 9.5),
    "W_turbine": (0.0, 25.0), "W_main_compressor": (0.0, 15.0),
    "W_net": (0.0, 15.0), "eta_thermal": (0.0, 0.60),
    "surge_margin_main": (0.0, 0.60),
}
ACTION_CONFIG = {
    "bypass_valve_opening":    {"min": 0.0, "max": 1.0, "rate": 0.05},
    "igv_angle_normalized":    {"min": 0.0, "max": 1.0, "rate": 0.05},
    "inventory_valve_opening": {"min": 0.0, "max": 1.0, "rate": 0.02},
    "cooling_flow_normalized": {"min": 0.0, "max": 1.0, "rate": 0.05},
}

TRAINER_CONFIG = {
    # Env config
    "obs_vars": OBS_VARS,
    "obs_bounds": OBS_BOUNDS,
    "action_vars": ACTION_VARS,
    "action_config": ACTION_CONFIG,
    "history_steps": 3,
    "step_size": 5.0,
    "episode_max_steps": 15,
    "reward": {
        "w_tracking": 1.0, "w_efficiency": 0.3, "w_smoothness": 0.1,
        "rated_power_mw": 10.0, "design_efficiency": 0.47,
        "terminal_failure_reward": -100.0,
    },
    "safety": {
        "compressor_inlet_temp_min_c": 32.2,
        "compressor_inlet_temp_catastrophic_c": 31.5,
    },
    "setpoint": {"W_net": 10.0},
    "normalization": {
        "norm_obs": True, "norm_reward": True,
        "clip_obs": 10.0, "clip_reward": 10.0, "gamma": 0.99,
    },
    # PPO config (small for unit tests)
    "ppo": {"n_steps": 16, "batch_size": 8, "n_epochs": 1,
            "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2,
            "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5},
    "network": {"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
    # Constraint config
    "constraint_names": ["T_comp_min", "surge_margin_main"],
    "multiplier_lr": 1e-2,
    # Checkpoint config
    "checkpoint_dir": "/tmp/sco2rl_test_checkpoints",
    "run_name": "unit_test_run",
    "checkpoint_freq": 100_000,
    "initial_curriculum_phase": 0,
}


# ---- Fixtures ---------------------------------------------------------------

def _make_fmu_factory(seed: int = 42):
    """Return a factory function that creates a MockFMU (for SubprocVecEnv compatibility)."""
    def factory():
        fmu = MockFMU(
            obs_vars=OBS_VARS,
            action_vars=ACTION_VARS,
            design_point=DESIGN_POINT,
            seed=seed,
        )
        fmu.initialize(start_time=0.0, stop_time=1000.0, step_size=5.0)
        return fmu
    return factory


@pytest.fixture
def mock_fmu():
    fmu = MockFMU(
        obs_vars=OBS_VARS,
        action_vars=ACTION_VARS,
        design_point=DESIGN_POINT,
        seed=42,
    )
    fmu.initialize(start_time=0.0, stop_time=1000.0, step_size=5.0)
    return fmu


@pytest.fixture
def trainer():
    from sco2rl.training.fmu_trainer import FMUTrainer
    return FMUTrainer(config=TRAINER_CONFIG)


@pytest.fixture
def trainer_setup(trainer):
    """Set up trainer with factory-based API (new interface)."""
    trainer.setup(fmu_factory=_make_fmu_factory(), n_envs=1)
    return trainer


# ---- Tests ------------------------------------------------------------------

class TestFMUTrainerSetup:
    def test_setup_with_factory_completes_without_error(self, trainer):
        """setup(fmu_factory=..., n_envs=1) should complete without raising."""
        trainer.setup(fmu_factory=_make_fmu_factory(), n_envs=1)

    def test_setup_creates_policy(self, trainer_setup):
        assert trainer_setup._policy is not None

    def test_setup_creates_env(self, trainer_setup):
        assert trainer_setup._env is not None

    def test_setup_creates_checkpoint_manager(self, trainer_setup):
        assert trainer_setup._checkpoint_mgr is not None

    def test_train_before_setup_raises(self, trainer):
        with pytest.raises(RuntimeError, match="setup"):
            trainer.train(total_timesteps=10)

    def test_evaluate_before_setup_raises(self, trainer):
        with pytest.raises(RuntimeError, match="setup"):
            trainer.evaluate(n_episodes=1)

    def test_setup_creates_curriculum_callback(self, trainer_setup):
        """setup() must create a CurriculumCallback (no longer dead code)."""
        assert trainer_setup._curriculum_callback is not None

    def test_setup_factory_called_n_envs_times(self, trainer):
        """setup(fmu_factory, n_envs=1) calls factory once."""
        call_count = {"n": 0}

        def counting_factory():
            call_count["n"] += 1
            fmu = MockFMU(
                obs_vars=OBS_VARS,
                action_vars=ACTION_VARS,
                design_point=DESIGN_POINT,
                seed=0,
            )
            fmu.initialize(start_time=0.0, stop_time=1000.0, step_size=5.0)
            return fmu

        trainer.setup(fmu_factory=counting_factory, n_envs=1)
        assert call_count["n"] == 1


class TestFMUTrainerEvaluate:
    def test_evaluate_returns_dict(self, trainer_setup):
        result = trainer_setup.evaluate(n_episodes=2)
        assert isinstance(result, dict)

    def test_evaluate_has_mean_reward_key(self, trainer_setup):
        result = trainer_setup.evaluate(n_episodes=2)
        assert "mean_reward" in result

    def test_evaluate_has_violation_rate_key(self, trainer_setup):
        result = trainer_setup.evaluate(n_episodes=2)
        assert "violation_rate" in result

    def test_evaluate_mean_reward_is_float(self, trainer_setup):
        result = trainer_setup.evaluate(n_episodes=2)
        assert isinstance(result["mean_reward"], float)

    def test_evaluate_violation_rate_between_0_and_1(self, trainer_setup):
        result = trainer_setup.evaluate(n_episodes=2)
        vr = result["violation_rate"]
        assert 0.0 <= vr <= 1.0

    def test_evaluate_deterministic_gives_same_result_twice(self, trainer_setup):
        """Deterministic evaluation must be reproducible."""
        r1 = trainer_setup.evaluate(n_episodes=2)
        r2 = trainer_setup.evaluate(n_episodes=2)
        assert r1["mean_reward"] == pytest.approx(r2["mean_reward"], abs=1e-4)
