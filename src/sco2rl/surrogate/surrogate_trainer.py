"""
SurrogateTrainer: SKRL PPO agent training on GPU-vectorized surrogate envs.

This trainer:
- Creates n_envs surrogate environments wrapped for SKRL
- Builds a SKRL PPO agent with separate policy and value networks
- Saves/loads checkpoints in RULE-C4 format (5 required fields)
- Logs to TensorBoard
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Type

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy and Value network definitions
# ---------------------------------------------------------------------------

class _Policy(GaussianMixin, Model):
    """Gaussian stochastic policy for PPO."""

    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 hidden_size: int = 256):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, self.num_actions),
        )
        self.log_std = nn.Parameter(-0.5 * torch.ones(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std, {}


class _Value(DeterministicMixin, Model):
    """Deterministic value network for PPO critic."""

    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 hidden_size: int = 256):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# ---------------------------------------------------------------------------
# SurrogateTrainer
# ---------------------------------------------------------------------------

class SurrogateTrainer:
    """Train SKRL PPO agent on vectorized SurrogateEnv (FNO-backed).

    Architecture:
    - n_envs GPU-vectorized SurrogateEnv instances (using skrl.envs.wrappers.torch)
    - SKRL PPO agent with separate policy and value networks
    - Saves checkpoints compatible with RULE-C4 format (5 required fields)
    - Logs to TensorBoard

    Args:
        surrogate_model: any object with predict_next_state(state, action) -> next_state
        config: dict from skrl_ppo section of fno_surrogate.yaml
        env_class: gymnasium.Env subclass to instantiate for each parallel env.
            Defaults to importing sco2rl.surrogate.surrogate_env.SurrogateEnv.
        device: torch device string ("cpu" or "cuda"). Note: SKRL wrap_env may
            override this to the available accelerator; build_agent() will use
            the device that wrap_env actually selected.
    """

    def __init__(
        self,
        surrogate_model: Any,
        config: Dict,
        env_class: Optional[Type[gym.Env]] = None,
        device: str = "cuda",
    ):
        self._surrogate_model = surrogate_model
        self._config = config
        self._requested_device = device
        self._env_class = env_class  # None => use TorchBatchedSurrogateEnv
        self._total_timesteps: int = 0

        # Resolved after build_envs() — may differ from _requested_device
        # because SKRL wrap_env auto-selects the accelerator
        self._device: str = device

        # Built lazily
        self._envs = None
        self._agent = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def envs(self):
        return self._envs

    @property
    def agent(self):
        return self._agent

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _get_env_class(self) -> Type[gym.Env]:
        if self._env_class is not None:
            return self._env_class
        from sco2rl.surrogate.surrogate_env import SurrogateEnv
        return SurrogateEnv

    def build_envs(self) -> None:
        """Create and wrap environment(s) for SKRL.

        Default behavior (env_class=None): use TorchBatchedSurrogateEnv with one
        batched forward pass per step for all environments.

        Compatibility behavior (env_class provided): keep legacy SyncVectorEnv
        path, used by unit tests that inject stub env classes.
        """
        n_envs: int = self._config.get("n_envs", 2)
        env_config: dict = self._config.get("env_config", {})
        surrogate_model = self._surrogate_model
        requested_device = self._requested_device

        if self._env_class is None:
            from sco2rl.surrogate.batched_env import TorchBatchedSurrogateEnv
            raw_env = TorchBatchedSurrogateEnv(
                model=surrogate_model,
                config=env_config,
                n_envs=n_envs,
                device=requested_device,
            )
            env_type = "TorchBatchedSurrogateEnv"
        else:
            EnvClass = self._get_env_class()

            def _make_env():
                return EnvClass(
                    model=surrogate_model,
                    config=env_config,
                    device=requested_device,
                )

            if n_envs == 1:
                raw_env = _make_env()
            else:
                raw_env = gym.vector.SyncVectorEnv([_make_env for _ in range(n_envs)])
            env_type = getattr(EnvClass, "__name__", "custom_env")

        self._envs = wrap_env(raw_env, wrapper="gymnasium")

        # Resolve the actual device SKRL picked
        self._device = str(self._envs.device)
        logger.info(
            "Built %d %s environment(s) wrapped for SKRL (device=%s).",
            n_envs,
            env_type,
            self._device,
        )

    def build_agent(self) -> None:
        """Create SKRL PPO agent with actor/critic networks.

        Networks are placed on the device that SKRL's wrap_env resolved, which
        ensures tensor device consistency during training.
        """
        if self._envs is None:
            raise RuntimeError("Call build_envs() before build_agent().")

        obs_space = self._envs.observation_space
        act_space = self._envs.action_space
        device = self._device  # Use the resolved device

        cfg = dict(PPO_DEFAULT_CONFIG)
        cfg.update({
            "rollouts": self._config.get("rollout_steps", 16),
            "learning_epochs": self._config.get("learning_epochs", 10),
            "mini_batches": self._config.get("mini_batches", 8),
            "discount_factor": self._config.get("discount_factor", 0.99),
            "lambda": self._config.get("lambda_gae", 0.95),
            "policy_learning_rate": self._config.get("policy_learning_rate", 3e-4),
            "value_learning_rate": self._config.get("value_learning_rate", 3e-4),
            "clip_ratio": self._config.get("clip_param", 0.2),
            "entropy_loss_scale": self._config.get("entropy_loss_scale", 0.01),
            "value_loss_scale": self._config.get("value_loss_scale", 0.5),
            "grad_norm_clip": self._config.get("grad_norm_clip", 0.5),
            "experiment": {
                "write_interval": 0,      # disable default SKRL file writes
                "checkpoint_interval": 0,
                "directory": self._config.get("tensorboard_log", "/tmp/skrl_logs"),
            },
        })

        policy = _Policy(obs_space, act_space, device)
        value = _Value(obs_space, act_space, device)

        memory_size = max(
            self._config.get("rollout_steps", 16),
            self._config.get("n_envs", 2),
        )
        memory = RandomMemory(
            memory_size=memory_size,
            num_envs=self._envs.num_envs,
            device=device,
        )

        self._agent = PPO(
            models={"policy": policy, "value": value},
            memory=memory,
            cfg=cfg,
            observation_space=obs_space,
            action_space=act_space,
            device=device,
        )
        logger.info("Built SKRL PPO agent on device=%s.", device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, timesteps: int = None) -> Dict:
        """Run SKRL training loop.

        Returns:
            dict with keys: 'mean_reward', 'total_timesteps'
        """
        if self._agent is None or self._envs is None:
            raise RuntimeError("Call build_envs() and build_agent() before train().")

        if timesteps is None:
            timesteps = int(self._config.get("total_timesteps", 5_000_000))

        # By default, config total_timesteps is interpreted as desired transitions.
        # For vectorized envs, SequentialTrainer timesteps counts env.step() calls.
        treat_as_transitions = bool(self._config.get("timesteps_are_transitions", True))
        trainer_timesteps = int(timesteps)
        if treat_as_transitions and getattr(self._envs, "num_envs", 1) > 1:
            trainer_timesteps = max(1, trainer_timesteps // int(self._envs.num_envs))
            logger.info(
                "Converting requested transitions=%d to trainer timesteps=%d (num_envs=%d).",
                timesteps,
                trainer_timesteps,
                self._envs.num_envs,
            )

        trainer_cfg = {
            "timesteps": trainer_timesteps,
            "headless": True,
        }
        trainer = SequentialTrainer(
            cfg=trainer_cfg,
            env=self._envs,
            agents=self._agent,
        )
        trainer.train()

        if treat_as_transitions:
            self._total_timesteps += trainer_timesteps * int(self._envs.num_envs)
        else:
            self._total_timesteps += trainer_timesteps

        # Collect mean reward from agent tracking data (may be NaN early on)
        mean_reward = float("nan")
        try:
            tracking = getattr(self._agent, "tracking_data", {})
            if tracking:
                key = "Reward / Total reward (mean)"
                if key in tracking and tracking[key]:
                    mean_reward = float(tracking[key][-1])
        except Exception:
            pass

        return {
            "mean_reward": mean_reward,
            "total_timesteps": self._total_timesteps,
        }

    # ------------------------------------------------------------------
    # Checkpoint (RULE-C4)
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save checkpoint in RULE-C4 format (5 required fields).

        Saves model weights to a companion .pt file alongside the JSON manifest.
        The 5 RULE-C4 required fields are:
            model_weights, vec_normalize_stats, curriculum_phase,
            lagrange_multipliers, total_timesteps
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Save PyTorch weights to a .pt sidecar file next to the JSON
        weights_path = path.replace(".json", "") + "_weights.pt"
        if self._agent is not None:
            state = {
                "policy": self._agent.policy.state_dict(),
                "value": self._agent.value.state_dict(),
            }
            torch.save(state, weights_path)
        else:
            torch.save({}, weights_path)

        manifest = {
            "model_weights": weights_path,
            "vec_normalize_stats": None,   # No VecNormalize for surrogate trainer
            "curriculum_phase": 0,         # 0 = STEADY_STATE
            "lagrange_multipliers": {},    # Empty initially
            "total_timesteps": self._total_timesteps,
        }
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Checkpoint saved to %s (weights: %s)", path, weights_path)

    def load_checkpoint(self, path: str) -> None:
        """Load RULE-C4 checkpoint, restoring total_timesteps and model weights."""
        with open(path) as f:
            manifest = json.load(f)

        self._total_timesteps = manifest.get("total_timesteps", 0)

        weights_path = manifest.get("model_weights", "")
        if weights_path and os.path.isfile(weights_path) and self._agent is not None:
            state = torch.load(weights_path, map_location=self._device)
            if "policy" in state:
                self._agent.policy.load_state_dict(state["policy"])
            if "value" in state:
                self._agent.value.load_state_dict(state["value"])
            logger.info("Loaded model weights from %s", weights_path)

        logger.info(
            "Checkpoint loaded from %s (total_timesteps=%d)",
            path, self._total_timesteps
        )
