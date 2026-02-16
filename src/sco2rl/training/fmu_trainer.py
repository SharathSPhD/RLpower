"""FMUTrainer -- orchestrates the full FMU-based PPO training loop.

Wires together:
  SCO2FMUEnv (with injected FMUInterface)
  VecNormalize
  LagrangianPPO
  CheckpointManager

Unit tests use MockFMU; integration tests use FMPyAdapter with real FMU.
RULE-C1: FMUInterface is injected via setup() -- no direct FMU construction here.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sco2rl.environment.sco2_env import SCO2FMUEnv
from sco2rl.simulation.fmu.interface import FMUInterface
from sco2rl.training.lagrangian_ppo import LagrangianPPO
from sco2rl.training.checkpoint_manager import CheckpointManager


class FMUTrainer:
    """Full FMU-based PPO training loop.

    Parameters
    ----------
    config:
        Dict combining ppo_fmu.yaml, env.yaml, and constraints.yaml settings.
        Required keys: obs_vars, obs_bounds, action_vars, action_config,
        history_steps, step_size, episode_max_steps, reward, safety,
        checkpoint_dir, run_name, constraint_names, multiplier_lr.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._env: VecNormalize | None = None
        self._policy: LagrangianPPO | None = None
        self._checkpoint_mgr: CheckpointManager | None = None
        self._fmu: FMUInterface | None = None

    def setup(self, fmu: FMUInterface) -> None:
        """Wire up environment, policy, and checkpoint manager.

        Parameters
        ----------
        fmu:
            Injected FMUInterface (MockFMU for unit tests, FMPyAdapter for prod).
        """
        self._fmu = fmu
        cfg = self._config

        # Build env config from training config dict
        env_config = {
            "obs_vars": cfg["obs_vars"],
            "obs_bounds": cfg["obs_bounds"],
            "action_vars": cfg["action_vars"],
            "action_config": cfg["action_config"],
            "history_steps": cfg.get("history_steps", 5),
            "step_size": cfg.get("step_size", 5.0),
            "episode_max_steps": cfg.get("episode_max_steps", 720),
            "reward": cfg.get("reward", {}),
            "safety": cfg.get("safety", {}),
            "setpoint": cfg.get("setpoint", {}),
        }

        # Wrap in DummyVecEnv (single env for unit tests; SubprocVecEnv for prod)
        def make_env():
            return SCO2FMUEnv(fmu=fmu, config=env_config)

        vec_env = DummyVecEnv([make_env])

        # VecNormalize for observation and reward normalisation
        norm_cfg = cfg.get("normalization", {})
        self._env = VecNormalize(
            vec_env,
            norm_obs=norm_cfg.get("norm_obs", True),
            norm_reward=norm_cfg.get("norm_reward", True),
            clip_obs=norm_cfg.get("clip_obs", 10.0),
            clip_reward=norm_cfg.get("clip_reward", 10.0),
            gamma=norm_cfg.get("gamma", 0.99),
        )

        # LagrangianPPO
        ppo_cfg = cfg.get("ppo", {})
        constraint_names = cfg.get("constraint_names", [])
        multiplier_lr = cfg.get("multiplier_lr", 1e-3)

        net_arch = cfg.get("network", {}).get("net_arch", {"pi": [256, 256], "vf": [256, 256]})
        policy_kwargs = {"net_arch": net_arch}

        self._policy = LagrangianPPO(
            env=self._env,
            multiplier_lr=multiplier_lr,
            constraint_names=constraint_names,
            policy="MlpPolicy",
            n_steps=ppo_cfg.get("n_steps", 2048),
            batch_size=ppo_cfg.get("batch_size", 256),
            n_epochs=ppo_cfg.get("n_epochs", 10),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_range=ppo_cfg.get("clip_range", 0.2),
            ent_coef=ppo_cfg.get("ent_coef", 0.01),
            vf_coef=ppo_cfg.get("vf_coef", 0.5),
            max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
            policy_kwargs=policy_kwargs,
            verbose=0,
        )

        # CheckpointManager
        checkpoint_dir = cfg.get("checkpoint_dir", "artifacts/checkpoints/fmu_direct")
        run_name = cfg.get("run_name", "run")
        self._checkpoint_mgr = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            run_name=run_name,
        )

    def train(self, total_timesteps: int) -> LagrangianPPO:
        """Main training loop with curriculum and checkpointing.

        Note: For unit tests, total_timesteps should be small (e.g. 100).
        Full training uses 5_000_000 steps on real FMU with SubprocVecEnv.
        """
        if self._policy is None or self._env is None:
            raise RuntimeError("Call setup() before train().")

        checkpoint_freq = self._config.get("checkpoint_freq", 100_000)
        curriculum_phase = self._config.get("initial_curriculum_phase", 0)

        self._policy.learn(total_timesteps=total_timesteps)

        # Save final checkpoint
        self._checkpoint_mgr.save(
            model=self._policy,
            vecnorm_stats=self._env.get_attr("obs_rms", indices=[0]),
            curriculum_phase=curriculum_phase,
            lagrange_multipliers=self._policy.get_multipliers(),
            total_timesteps=self._policy.num_timesteps,
            step=self._policy.num_timesteps,
        )

        return self._policy

    def evaluate(self, n_episodes: int = 10) -> dict:
        """Run n_episodes deterministically, return mean reward and violation rate.

        Returns
        -------
        dict with keys:
            "mean_reward": float -- mean episode reward across n_episodes
            "violation_rate": float -- fraction of steps with constraint violations
        """
        if self._policy is None or self._env is None:
            raise RuntimeError("Call setup() before evaluate().")

        # Freeze VecNormalize stats during evaluation (deterministic mode)
        self._env.training = False

        episode_rewards = []
        violation_count = 0
        total_steps = 0

        for _ in range(n_episodes):
            obs = self._env.reset()
            done = np.array([False])
            ep_reward = 0.0

            while not done[0]:
                action, _ = self._policy.predict(obs, deterministic=True)
                obs, reward, done, info = self._env.step(action)
                ep_reward += float(reward[0])
                total_steps += 1

                # Count violations from info dict if present
                if info and "violations" in info[0]:
                    violation_count += int(info[0]["violations"] > 0)

            episode_rewards.append(ep_reward)

        # Restore VecNormalize to training mode
        self._env.training = True

        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        violation_rate = violation_count / total_steps if total_steps > 0 else 0.0

        return {
            "mean_reward": mean_reward,
            "violation_rate": violation_rate,
        }
