"""Unit tests for LagrangianPPO.

All tests use MockFMU + SCO2FMUEnv (RULE-C1: no real FMU in unit tests).
Tests verify multiplier initialisation, update logic, and basic training.
"""
from __future__ import annotations

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from sco2rl.simulation.fmu.mock_fmu import MockFMU

# ---- Shared constants -------------------------------------------------------

OBS_VARS = [
    "T_turbine_inlet", "T_compressor_inlet", "P_high", "P_low",
    "mdot_turbine", "W_turbine", "W_main_compressor", "W_net",
    "eta_thermal", "surge_margin_main",
]

ACTION_VARS = [
    "bypass_valve_opening", "igv_angle_normalized",
    "inventory_valve_opening", "cooling_flow_normalized",
]

DESIGN_POINT = {
    "T_turbine_inlet": 750.0,
    "T_compressor_inlet": 33.0,
    "P_high": 20.0,
    "P_low": 7.7,
    "mdot_turbine": 130.0,
    "W_turbine": 14.5,
    "W_main_compressor": 4.0,
    "W_net": 10.0,
    "eta_thermal": 0.47,
    "surge_margin_main": 0.20,
}

OBS_BOUNDS = {
    "T_turbine_inlet":   (600.0, 850.0),
    "T_compressor_inlet":(30.0, 45.0),
    "P_high":            (14.0, 26.0),
    "P_low":             (6.5, 9.5),
    "mdot_turbine":      (40.0, 220.0),
    "W_turbine":         (0.0, 25.0),
    "W_main_compressor": (0.0, 15.0),
    "W_net":             (0.0, 15.0),
    "eta_thermal":       (0.0, 0.60),
    "surge_margin_main": (0.0, 0.60),
}

ACTION_CONFIG = {
    "bypass_valve_opening":    {"min": 0.0, "max": 1.0, "rate": 0.05},
    "igv_angle_normalized":    {"min": 0.0, "max": 1.0, "rate": 0.05},
    "inventory_valve_opening": {"min": 0.0, "max": 1.0, "rate": 0.02},
    "cooling_flow_normalized": {"min": 0.0, "max": 1.0, "rate": 0.05},
}

ENV_CONFIG = {
    "obs_vars": OBS_VARS,
    "obs_bounds": OBS_BOUNDS,
    "action_vars": ACTION_VARS,
    "action_config": ACTION_CONFIG,
    "history_steps": 3,
    "step_size": 5.0,
    "episode_max_steps": 20,
    "reward": {
        "w_tracking": 1.0,
        "w_efficiency": 0.3,
        "w_smoothness": 0.1,
        "rated_power_mw": 10.0,
        "design_efficiency": 0.47,
        "terminal_failure_reward": -100.0,
    },
    "safety": {
        "compressor_inlet_temp_min_c": 32.2,
        "compressor_inlet_temp_catastrophic_c": 31.5,
    },
    "setpoint": {"W_net": 10.0},
}

CONSTRAINT_NAMES = ["T_comp_min", "surge_margin_main"]


# ---- Fixtures ---------------------------------------------------------------

@pytest.fixture
def mock_fmu():
    fmu = MockFMU(
        obs_vars=OBS_VARS,
        action_vars=ACTION_VARS,
        design_point=DESIGN_POINT,
        seed=42,
    )
    fmu.initialize(start_time=0.0, stop_time=100.0, step_size=5.0)
    return fmu


@pytest.fixture
def vec_env(mock_fmu):
    from sco2rl.environment.sco2_env import SCO2FMUEnv
    def make():
        fmu = MockFMU(
            obs_vars=OBS_VARS,
            action_vars=ACTION_VARS,
            design_point=DESIGN_POINT,
            seed=42,
        )
        fmu.initialize(start_time=0.0, stop_time=100.0, step_size=5.0)
        return SCO2FMUEnv(fmu=fmu, config=ENV_CONFIG)
    return DummyVecEnv([make])


@pytest.fixture
def lagrangian_ppo(vec_env):
    from sco2rl.training.lagrangian_ppo import LagrangianPPO
    return LagrangianPPO(
        env=vec_env,
        multiplier_lr=1e-2,
        constraint_names=CONSTRAINT_NAMES,
        policy="MlpPolicy",
        n_steps=32,
        batch_size=16,
        n_epochs=2,
        verbose=0,
    )


# ---- Tests ------------------------------------------------------------------

class TestLagrangianPPOInit:
    def test_multipliers_initialised_at_zero(self, lagrangian_ppo):
        mults = lagrangian_ppo.get_multipliers()
        for name in CONSTRAINT_NAMES:
            assert mults[name] == pytest.approx(0.0), f"{name} not 0.0"

    def test_all_constraint_names_present(self, lagrangian_ppo):
        mults = lagrangian_ppo.get_multipliers()
        assert set(mults.keys()) == set(CONSTRAINT_NAMES)

    def test_get_multipliers_returns_copy(self, lagrangian_ppo):
        m1 = lagrangian_ppo.get_multipliers()
        m1["T_comp_min"] = 999.0
        m2 = lagrangian_ppo.get_multipliers()
        assert m2["T_comp_min"] == pytest.approx(0.0)


class TestMultiplierUpdate:
    def test_update_increases_multiplier_when_violation_positive(self, lagrangian_ppo):
        lagrangian_ppo.update_multipliers({"T_comp_min": 1.0})
        mults = lagrangian_ppo.get_multipliers()
        assert mults["T_comp_min"] > 0.0

    def test_update_does_not_change_multiplier_when_violation_zero(self, lagrangian_ppo):
        lagrangian_ppo.update_multipliers({"T_comp_min": 0.0})
        mults = lagrangian_ppo.get_multipliers()
        assert mults["T_comp_min"] == pytest.approx(0.0)

    def test_multiplier_never_goes_negative(self, lagrangian_ppo):
        """Even large negative violation cannot make lambda < 0."""
        lagrangian_ppo.update_multipliers({"T_comp_min": -1000.0})
        mults = lagrangian_ppo.get_multipliers()
        assert mults["T_comp_min"] >= 0.0

    def test_update_ignores_unknown_constraint_names(self, lagrangian_ppo):
        # Should not raise
        lagrangian_ppo.update_multipliers({"unknown_constraint": 5.0})
        # Known constraint unaffected
        mults = lagrangian_ppo.get_multipliers()
        assert "unknown_constraint" not in mults

    def test_update_accumulates_over_multiple_calls(self, lagrangian_ppo):
        for _ in range(5):
            lagrangian_ppo.update_multipliers({"T_comp_min": 1.0})
        mults = lagrangian_ppo.get_multipliers()
        # After 5 updates with lr=0.01, violation=1.0: lambda = 5 * 0.01 = 0.05
        assert mults["T_comp_min"] == pytest.approx(0.05, abs=1e-9)

    def test_multiplier_update_correct_formula(self, lagrangian_ppo):
        """lambda <- max(0, lambda + lr * violation)."""
        lr = 1e-2
        violation = 2.5
        lagrangian_ppo.update_multipliers({"surge_margin_main": violation})
        mults = lagrangian_ppo.get_multipliers()
        expected = max(0.0, 0.0 + lr * violation)
        assert mults["surge_margin_main"] == pytest.approx(expected, abs=1e-9)


class TestLagrangianPPOLearn:
    def test_learn_runs_without_error(self, lagrangian_ppo):
        lagrangian_ppo.learn(total_timesteps=100)

    def test_learn_returns_self(self, lagrangian_ppo):
        result = lagrangian_ppo.learn(total_timesteps=64)
        assert result is lagrangian_ppo

    def test_num_timesteps_increases_after_learn(self, lagrangian_ppo):
        lagrangian_ppo.learn(total_timesteps=64)
        assert lagrangian_ppo.num_timesteps >= 64


class TestLagrangianPPOPredict:
    def test_predict_returns_action_array(self, lagrangian_ppo, vec_env):
        obs = vec_env.reset()
        action, _ = lagrangian_ppo.predict(obs, deterministic=True)
        assert isinstance(action, np.ndarray)
        assert action.shape == (1, len(ACTION_VARS))


class TestLagrangianPPOSaveLoad:
    def test_save_creates_zip_file(self, lagrangian_ppo, tmp_path):
        path = str(tmp_path / "test_model")
        lagrangian_ppo.save(path)
        import os
        assert os.path.exists(path + ".zip")

    def test_save_creates_multiplier_file(self, lagrangian_ppo, tmp_path):
        path = str(tmp_path / "test_model")
        lagrangian_ppo.save(path)
        import os
        assert os.path.exists(path + "_multipliers.pkl")

    def test_load_restores_multipliers(self, lagrangian_ppo, tmp_path, vec_env):
        from sco2rl.training.lagrangian_ppo import LagrangianPPO
        lagrangian_ppo.update_multipliers({"T_comp_min": 0.5, "surge_margin_main": 0.3})
        path = str(tmp_path / "test_model")
        lagrangian_ppo.save(path)
        loaded = LagrangianPPO.load(path, env=vec_env)
        mults = loaded.get_multipliers()
        assert mults["T_comp_min"] == pytest.approx(0.005, abs=1e-9)
        assert mults["surge_margin_main"] == pytest.approx(0.003, abs=1e-9)
