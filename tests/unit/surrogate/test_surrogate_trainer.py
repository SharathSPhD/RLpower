"""
Tests for SurrogateTrainer -- SKRL PPO agent training on FNO surrogate envs.

TDD: These tests are written FIRST and must fail (RED) before implementation.
"""

import os
import json
import pytest
import numpy as np
import torch
import gymnasium as gym

# ---------------------------------------------------------------------------
# Stub classes (substitutes for the parallel exp/stage3-fno-a branch)
# ---------------------------------------------------------------------------

N_OBS = 8
N_ACT = 4
OBS_VARS = [f"obs_{i}" for i in range(N_OBS)]
ACTION_VARS = [f"act_{i}" for i in range(N_ACT)]


class StubSurrogateModel:
    """Minimal surrogate model stub for testing."""

    def predict_next_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return state + 0.01 * torch.randn_like(state)


class StubSurrogateEnv(gym.Env):
    """Minimal SurrogateEnv stub for SKRL trainer tests."""

    def __init__(self, model, config, device="cpu"):
        super().__init__()
        n_obs = N_OBS * config.get("history_steps", 1)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (n_obs,), np.float32
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (N_ACT,), np.float32)
        self._step_count = 0
        self._max_steps = config.get("episode_max_steps", 50)

    def reset(self, *, seed=None, options=None):
        self._step_count = 0
        return np.zeros(self.observation_space.shape, np.float32), {}

    def step(self, action):
        self._step_count += 1
        obs = np.random.randn(*self.observation_space.shape).astype(np.float32)
        reward = float(np.random.randn())
        terminated = False
        truncated = self._step_count >= self._max_steps
        return obs, reward, terminated, truncated, {}


# ---------------------------------------------------------------------------
# Shared test config (small values for fast tests)
# ---------------------------------------------------------------------------

TEST_CONFIG = {
    "n_envs": 2,
    "rollout_steps": 16,
    "learning_epochs": 2,
    "mini_batches": 2,
    "discount_factor": 0.99,
    "lambda_gae": 0.95,
    "policy_learning_rate": 3e-4,
    "value_learning_rate": 3e-4,
    "clip_param": 0.2,
    "entropy_loss_scale": 0.01,
    "value_loss_scale": 0.5,
    "grad_norm_clip": 0.5,
    "total_timesteps": 64,
    "checkpoint_dir": "artifacts/checkpoints/surrogate_ppo",
    "tensorboard_log": "artifacts/logs/surrogate_ppo",
    "seed": 42,
    "env_config": {
        "obs_vars": OBS_VARS,
        "obs_bounds": {v: (0.0, 1.0) for v in OBS_VARS},
        "action_vars": ACTION_VARS,
        "action_config": {
            v: {"phys_min": 0.0, "phys_max": 1.0, "rate_limit": 0.05}
            for v in ACTION_VARS
        },
        "obs_design_point": {v: 0.5 for v in OBS_VARS},
        "reward": {
            "w_tracking": 1.0,
            "w_efficiency": 0.3,
            "w_smoothness": 0.1,
            "rated_power_mw": 10.0,
            "design_efficiency": 0.40,
            "terminal_failure_reward": -100.0,
        },
        "safety": {
            "T_comp_inlet_min_c": -999.0,
            "surge_margin_min": -999.0,
        },
        "setpoint": {"W_net_setpoint": 5.0},
        "history_steps": 1,
        "episode_max_steps": 20,
    },
}

ENV_CLASS = StubSurrogateEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_model():
    return StubSurrogateModel()


@pytest.fixture
def trainer(stub_model, tmp_path):
    from sco2rl.surrogate.surrogate_trainer import SurrogateTrainer

    cfg = dict(TEST_CONFIG)
    cfg["checkpoint_dir"] = str(tmp_path / "checkpoints")
    cfg["tensorboard_log"] = str(tmp_path / "logs")

    t = SurrogateTrainer(
        surrogate_model=stub_model,
        config=cfg,
        env_class=ENV_CLASS,
        device="cpu",
    )
    return t


@pytest.fixture
def built_trainer(trainer):
    trainer.build_envs()
    trainer.build_agent()
    return trainer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildEnvs:
    def test_build_envs_creates_wrapped_envs(self, trainer):
        """After build_envs(), trainer.envs must not be None."""
        assert trainer.envs is None, "envs should be None before build_envs()"
        trainer.build_envs()
        assert trainer.envs is not None, "envs must be set after build_envs()"


class TestBuildAgent:
    def test_build_agent_creates_ppo_agent(self, trainer):
        """After build_agent(), trainer.agent must not be None."""
        trainer.build_envs()
        assert trainer.agent is None, "agent should be None before build_agent()"
        trainer.build_agent()
        assert trainer.agent is not None, "agent must be set after build_agent()"


class TestCheckpoint:
    def test_save_checkpoint_creates_file(self, built_trainer, tmp_path):
        """save_checkpoint() must create a JSON file at the given path."""
        ckpt_path = str(tmp_path / "test_ckpt.json")
        built_trainer.save_checkpoint(ckpt_path)
        assert os.path.isfile(ckpt_path), f"Checkpoint file not found: {ckpt_path}"

    def test_checkpoint_has_rule_c4_fields(self, built_trainer, tmp_path):
        """Saved checkpoint must contain all 5 RULE-C4 required fields."""
        ckpt_path = str(tmp_path / "test_ckpt.json")
        built_trainer.save_checkpoint(ckpt_path)
        with open(ckpt_path) as f:
            data = json.load(f)
        required_fields = {
            "model_weights",
            "vec_normalize_stats",
            "curriculum_phase",
            "lagrange_multipliers",
            "total_timesteps",
        }
        missing = required_fields - set(data.keys())
        assert not missing, f"Checkpoint missing RULE-C4 fields: {missing}"

    def test_load_checkpoint_restores_total_timesteps(self, built_trainer, tmp_path):
        """Loading a checkpoint must restore total_timesteps."""
        built_trainer._total_timesteps = 100
        ckpt_path = str(tmp_path / "restore_ckpt.json")
        built_trainer.save_checkpoint(ckpt_path)

        built_trainer._total_timesteps = 0
        built_trainer.load_checkpoint(ckpt_path)
        assert built_trainer._total_timesteps == 100, (
            f"Expected total_timesteps=100, got {built_trainer._total_timesteps}"
        )


class TestTrain:
    def test_train_returns_metrics_dict(self, built_trainer):
        """train(timesteps=64) must return dict with mean_reward and total_timesteps."""
        metrics = built_trainer.train(timesteps=64)
        assert isinstance(metrics, dict), "train() must return a dict"
        assert "mean_reward" in metrics, "metrics dict must contain mean_reward"
        assert "total_timesteps" in metrics, "metrics dict must contain total_timesteps"
        assert metrics["total_timesteps"] > 0, "total_timesteps must be > 0"

    def test_train_converts_transition_budget_to_trainer_steps(self, stub_model, tmp_path):
        """Default path (batched env) interprets timesteps as transitions."""
        from sco2rl.surrogate.surrogate_trainer import SurrogateTrainer

        cfg = dict(TEST_CONFIG)
        cfg["n_envs"] = 4
        cfg["total_timesteps"] = 64
        cfg["rollout_steps"] = 8
        cfg["learning_epochs"] = 1
        cfg["mini_batches"] = 1
        cfg["checkpoint_dir"] = str(tmp_path / "checkpoints_batched")
        cfg["tensorboard_log"] = str(tmp_path / "logs_batched")

        trainer = SurrogateTrainer(
            surrogate_model=stub_model,
            config=cfg,
            env_class=None,  # use TorchBatchedSurrogateEnv path
            device="cpu",
        )
        trainer.build_envs()
        trainer.build_agent()
        metrics = trainer.train(timesteps=64)
        assert metrics["total_timesteps"] == 64
