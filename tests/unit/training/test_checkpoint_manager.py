"""Unit tests for CheckpointManager.

Tests verify RULE-C4: checkpoint must contain ALL 5 required fields:
  model_path, vecnorm_stats, curriculum_phase, lagrange_multipliers, total_timesteps

MockFMU + SCO2FMUEnv used for model creation.
"""
from __future__ import annotations

import json
import os
import pytest

from sco2rl.simulation.fmu.mock_fmu import MockFMU
from stable_baselines3.common.vec_env import DummyVecEnv

# ---- Constants --------------------------------------------------------------

OBS_VARS = [
    "T_turbine_inlet", "T_compressor_inlet", "P_high", "W_net",
    "eta_thermal", "surge_margin_main",
]
ACTION_VARS = ["bypass_valve_opening", "igv_angle_normalized"]
DESIGN_POINT = {
    "T_turbine_inlet": 750.0, "T_compressor_inlet": 33.0,
    "P_high": 20.0, "W_net": 10.0, "eta_thermal": 0.47, "surge_margin_main": 0.20,
}
OBS_BOUNDS = {
    "T_turbine_inlet": (600.0, 850.0), "T_compressor_inlet": (30.0, 45.0),
    "P_high": (14.0, 26.0), "W_net": (0.0, 15.0),
    "eta_thermal": (0.0, 0.60), "surge_margin_main": (0.0, 0.60),
}
ACTION_CONFIG = {
    "bypass_valve_opening": {"min": 0.0, "max": 1.0, "rate": 0.05},
    "igv_angle_normalized": {"min": 0.0, "max": 1.0, "rate": 0.05},
}
ENV_CONFIG = {
    "obs_vars": OBS_VARS, "obs_bounds": OBS_BOUNDS,
    "action_vars": ACTION_VARS, "action_config": ACTION_CONFIG,
    "history_steps": 3, "step_size": 5.0, "episode_max_steps": 10,
    "reward": {"w_tracking": 1.0, "w_efficiency": 0.3, "w_smoothness": 0.1,
               "rated_power_mw": 10.0, "design_efficiency": 0.47,
               "terminal_failure_reward": -100.0},
    "safety": {"compressor_inlet_temp_min_c": 32.2,
                "compressor_inlet_temp_catastrophic_c": 31.5},
    "setpoint": {"W_net": 10.0},
}
CONSTRAINT_NAMES = ["T_comp_min", "surge_margin_main"]
VECNORM_STATS = {"obs_rms_mean": [0.5, 0.3], "obs_rms_var": [1.0, 0.8]}
LAGRANGE_MULTS = {"T_comp_min": 0.05, "surge_margin_main": 0.12}


# ---- Fixtures ---------------------------------------------------------------

def _make_vec_env():
    from sco2rl.environment.sco2_env import SCO2FMUEnv
    def make():
        fmu = MockFMU(obs_vars=OBS_VARS, action_vars=ACTION_VARS,
                      design_point=DESIGN_POINT, seed=42)
        fmu.initialize(start_time=0.0, stop_time=100.0, step_size=5.0)
        return SCO2FMUEnv(fmu=fmu, config=ENV_CONFIG)
    return DummyVecEnv([make])


@pytest.fixture
def vec_env():
    return _make_vec_env()


@pytest.fixture
def lagrangian_ppo(vec_env):
    from sco2rl.training.lagrangian_ppo import LagrangianPPO
    return LagrangianPPO(
        env=vec_env,
        multiplier_lr=1e-2,
        constraint_names=CONSTRAINT_NAMES,
        policy="MlpPolicy",
        n_steps=16,
        batch_size=8,
        n_epochs=1,
        verbose=0,
    )


@pytest.fixture
def checkpoint_mgr(tmp_path):
    from sco2rl.training.checkpoint_manager import CheckpointManager
    return CheckpointManager(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        run_name="test_run",
    )


# ---- Tests ------------------------------------------------------------------

class TestCheckpointManagerInit:
    def test_run_dir_is_created(self, tmp_path):
        from sco2rl.training.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager(
            checkpoint_dir=str(tmp_path / "ckpts"),
            run_name="my_run",
        )
        assert (tmp_path / "ckpts" / "my_run").is_dir()


class TestCheckpointManagerSave:
    def test_save_creates_json_file(self, checkpoint_mgr, lagrangian_ppo):
        path = checkpoint_mgr.save(
            model=lagrangian_ppo,
            vecnorm_stats=VECNORM_STATS,
            curriculum_phase=1,
            lagrange_multipliers=LAGRANGE_MULTS,
            total_timesteps=10000,
            step=100,
        )
        assert os.path.exists(path)
        assert path.endswith(".json")

    def test_save_creates_model_zip(self, checkpoint_mgr, lagrangian_ppo, tmp_path):
        path = checkpoint_mgr.save(
            model=lagrangian_ppo,
            vecnorm_stats=VECNORM_STATS,
            curriculum_phase=0,
            lagrange_multipliers=LAGRANGE_MULTS,
            total_timesteps=500,
            step=50,
        )
        # The JSON path contains the checkpoint; model zip is a sibling
        json_dir = os.path.dirname(path)
        zip_files = [f for f in os.listdir(json_dir) if f.endswith(".zip")]
        assert len(zip_files) >= 1

    def test_save_filename_includes_step(self, checkpoint_mgr, lagrangian_ppo):
        path = checkpoint_mgr.save(
            model=lagrangian_ppo,
            vecnorm_stats=VECNORM_STATS,
            curriculum_phase=2,
            lagrange_multipliers=LAGRANGE_MULTS,
            total_timesteps=200,
            step=12345,
        )
        assert "12345" in os.path.basename(path)

    def test_save_filename_includes_curriculum_phase(self, checkpoint_mgr, lagrangian_ppo):
        path = checkpoint_mgr.save(
            model=lagrangian_ppo,
            vecnorm_stats=VECNORM_STATS,
            curriculum_phase=3,
            lagrange_multipliers=LAGRANGE_MULTS,
            total_timesteps=300,
            step=300,
        )
        assert "phase_3" in os.path.basename(path) or "3" in os.path.basename(path)

    def test_save_json_contains_all_rule_c4_fields(self, checkpoint_mgr, lagrangian_ppo):
        path = checkpoint_mgr.save(
            model=lagrangian_ppo,
            vecnorm_stats=VECNORM_STATS,
            curriculum_phase=1,
            lagrange_multipliers=LAGRANGE_MULTS,
            total_timesteps=1000,
            step=100,
        )
        with open(path) as f:
            data = json.load(f)
        required_fields = {
            "model_path", "vecnorm_stats", "curriculum_phase",
            "lagrange_multipliers", "total_timesteps",
        }
        assert required_fields.issubset(set(data.keys()))


class TestCheckpointManagerLoad:
    def test_load_returns_all_rule_c4_fields(self, checkpoint_mgr, lagrangian_ppo):
        path = checkpoint_mgr.save(
            model=lagrangian_ppo,
            vecnorm_stats=VECNORM_STATS,
            curriculum_phase=1,
            lagrange_multipliers=LAGRANGE_MULTS,
            total_timesteps=1000,
            step=100,
        )
        data = checkpoint_mgr.load(path)
        assert "model_path" in data
        assert "vecnorm_stats" in data
        assert "curriculum_phase" in data
        assert "lagrange_multipliers" in data
        assert "total_timesteps" in data

    def test_load_round_trip_vecnorm_stats(self, checkpoint_mgr, lagrangian_ppo):
        path = checkpoint_mgr.save(
            model=lagrangian_ppo,
            vecnorm_stats=VECNORM_STATS,
            curriculum_phase=0,
            lagrange_multipliers=LAGRANGE_MULTS,
            total_timesteps=100,
            step=100,
        )
        data = checkpoint_mgr.load(path)
        assert data["vecnorm_stats"] == VECNORM_STATS

    def test_load_round_trip_curriculum_phase(self, checkpoint_mgr, lagrangian_ppo):
        path = checkpoint_mgr.save(
            model=lagrangian_ppo,
            vecnorm_stats=VECNORM_STATS,
            curriculum_phase=4,
            lagrange_multipliers=LAGRANGE_MULTS,
            total_timesteps=100,
            step=100,
        )
        data = checkpoint_mgr.load(path)
        assert data["curriculum_phase"] == 4

    def test_load_round_trip_lagrange_multipliers(self, checkpoint_mgr, lagrangian_ppo):
        path = checkpoint_mgr.save(
            model=lagrangian_ppo,
            vecnorm_stats=VECNORM_STATS,
            curriculum_phase=0,
            lagrange_multipliers=LAGRANGE_MULTS,
            total_timesteps=100,
            step=100,
        )
        data = checkpoint_mgr.load(path)
        assert data["lagrange_multipliers"] == LAGRANGE_MULTS

    def test_load_round_trip_total_timesteps(self, checkpoint_mgr, lagrangian_ppo):
        path = checkpoint_mgr.save(
            model=lagrangian_ppo,
            vecnorm_stats=VECNORM_STATS,
            curriculum_phase=0,
            lagrange_multipliers=LAGRANGE_MULTS,
            total_timesteps=999_999,
            step=100,
        )
        data = checkpoint_mgr.load(path)
        assert data["total_timesteps"] == 999_999

    def test_load_raises_value_error_for_missing_field(self, tmp_path):
        from sco2rl.training.checkpoint_manager import CheckpointManager
        # Write a malformed checkpoint missing 'lagrange_multipliers'
        bad_path = str(tmp_path / "bad_checkpoint.json")
        import json
        with open(bad_path, "w") as f:
            json.dump({
                "model_path": "/fake/model",
                "vecnorm_stats": {},
                "curriculum_phase": 0,
                # lagrange_multipliers is missing
                "total_timesteps": 0,
            }, f)
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path), run_name="r")
        with pytest.raises(ValueError, match="lagrange_multipliers"):
            mgr.load(bad_path)


class TestCheckpointManagerDiscovery:
    def test_get_latest_returns_none_when_empty(self, checkpoint_mgr):
        assert checkpoint_mgr.get_latest() is None

    def test_get_latest_returns_most_recent(self, checkpoint_mgr, lagrangian_ppo):
        path1 = checkpoint_mgr.save(
            model=lagrangian_ppo, vecnorm_stats={}, curriculum_phase=0,
            lagrange_multipliers={}, total_timesteps=100, step=100,
        )
        path2 = checkpoint_mgr.save(
            model=lagrangian_ppo, vecnorm_stats={}, curriculum_phase=0,
            lagrange_multipliers={}, total_timesteps=200, step=200,
        )
        latest = checkpoint_mgr.get_latest()
        assert latest == path2

    def test_list_checkpoints_sorted_oldest_first(self, checkpoint_mgr, lagrangian_ppo):
        path1 = checkpoint_mgr.save(
            model=lagrangian_ppo, vecnorm_stats={}, curriculum_phase=0,
            lagrange_multipliers={}, total_timesteps=100, step=100,
        )
        path2 = checkpoint_mgr.save(
            model=lagrangian_ppo, vecnorm_stats={}, curriculum_phase=0,
            lagrange_multipliers={}, total_timesteps=200, step=200,
        )
        path3 = checkpoint_mgr.save(
            model=lagrangian_ppo, vecnorm_stats={}, curriculum_phase=1,
            lagrange_multipliers={}, total_timesteps=300, step=300,
        )
        paths = checkpoint_mgr.list_checkpoints()
        assert paths == [path1, path2, path3]

    def test_list_checkpoints_empty_when_none_saved(self, checkpoint_mgr):
        assert checkpoint_mgr.list_checkpoints() == []
