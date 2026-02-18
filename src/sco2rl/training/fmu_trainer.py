"""FMUTrainer -- orchestrates the full FMU-based PPO training loop.

Wires together:
  SCO2FMUEnv (with injected FMUInterface)
  VecNormalize
  LagrangianPPO
  CheckpointManager
  CurriculumCallback (wires MetricsObserver + CurriculumScheduler into training)

Unit tests use MockFMU; integration tests use FMPyAdapter with real FMU.
RULE-C1: FMUInterface is injected via fmu_factory callable -- no direct FMU construction.

ADR: setup(fmu_factory: Callable[[], FMUInterface], n_envs: int) creates one FMU
instance per VecEnv worker via the factory, enabling SubprocVecEnv parallelism.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from sco2rl.environment.sco2_env import SCO2FMUEnv
from sco2rl.simulation.fmu.interface import FMUInterface
from sco2rl.training.lagrangian_ppo import LagrangianPPO
from sco2rl.training.checkpoint_manager import CheckpointManager
from sco2rl.training.curriculum_callback import CurriculumCallback
from sco2rl.curriculum.scheduler import CurriculumScheduler
from sco2rl.curriculum.metrics_observer import MetricsObserver
from sco2rl.curriculum.phase import CurriculumPhase, PhaseConfig


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
        self._curriculum_callback: CurriculumCallback | None = None

    def setup(
        self,
        fmu_factory: Callable[[], FMUInterface],
        n_envs: int = 1,
    ) -> None:
        """Wire up environment, policy, curriculum callback, and checkpoint manager.

        Parameters
        ----------
        fmu_factory:
            Callable that returns a new FMUInterface instance. Called once per
            VecEnv worker so each process gets its own independent FMU.
        n_envs:
            Number of parallel FMU environments. n_envs=1 uses DummyVecEnv
            (synchronous, no subprocess overhead — good for unit tests).
            n_envs>1 uses SubprocVecEnv with spawn start method.
        """
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

        # Monitor wrapper is required for CurriculumCallback to read episode stats
        # via info["episode"]["r"] (populated by SB3's Monitor, not VecNormalize).
        def make_single_env():
            return Monitor(SCO2FMUEnv(fmu=fmu_factory(), config=env_config))

        env_fns = [make_single_env for _ in range(n_envs)]
        vec_env = (
            DummyVecEnv(env_fns)
            if n_envs == 1
            else SubprocVecEnv(env_fns, start_method="spawn")
        )

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

        # CurriculumCallback: wires MetricsObserver + CurriculumScheduler into learn()
        # curriculum.yaml has nested structure: advancement.* and phases[i].*
        curriculum_cfg = cfg.get("curriculum", {})
        advancement_cfg = curriculum_cfg.get("advancement", {})
        phases_cfg = curriculum_cfg.get("phases", [])
        phase_0_cfg = next((p for p in phases_cfg if p.get("id") == 0), {})
        require_zero_violations = advancement_cfg.get("require_zero_constraint_violations", False)
        # violation_rate_limit_pct (0–100) takes precedence over require_zero_violations
        viol_pct = advancement_cfg.get("violation_rate_limit_pct", None)
        if viol_pct is not None:
            violation_rate_limit = float(viol_pct) / 100.0
        elif require_zero_violations:
            violation_rate_limit = 0.0
        else:
            violation_rate_limit = 0.05
        observer_cfg = {
            "window_size": advancement_cfg.get("window_episodes", 50),
            "advance_threshold": phase_0_cfg.get("advancement_threshold", 8.0),
            "violation_rate_limit": violation_rate_limit,
            "min_episodes": advancement_cfg.get("window_episodes", 50),
        }
        observer = MetricsObserver(config=observer_cfg)

        phase_configs = self._build_default_phase_configs(curriculum_cfg)
        scheduler = CurriculumScheduler(phase_configs=phase_configs, observer=observer)

        checkpoint_freq = cfg.get("checkpoint_freq", 100_000)
        self._curriculum_callback = CurriculumCallback(
            scheduler=scheduler,
            observer=observer,
            checkpoint_mgr=self._checkpoint_mgr,
            checkpoint_freq=checkpoint_freq,
            vecnorm=self._env,
            lagrangian_model=self._policy,
            verbose=cfg.get("verbose", 0),
        )

    def train(self, total_timesteps: int) -> LagrangianPPO:
        """Main training loop with curriculum and checkpointing.

        Note: For unit tests, total_timesteps should be small (e.g. 100).
        Full training uses 5_000_000 steps on real FMU with SubprocVecEnv.
        """
        if self._policy is None or self._env is None:
            raise RuntimeError("Call setup() before train().")

        # Pass CurriculumCallback to learn() so curriculum advances during training
        self._policy.learn(
            total_timesteps=total_timesteps,
            callback=self._curriculum_callback,
        )

        # Save final checkpoint (RULE-C4: 5 required fields + VecNormalize stats)
        final_phase = int(self._curriculum_callback.scheduler.get_phase())
        ts = self._policy.num_timesteps
        self._checkpoint_mgr.save(
            model=self._policy,
            vecnorm_stats={"obs_rms": None},  # legacy field kept for schema compat
            curriculum_phase=final_phase,
            lagrange_multipliers=self._policy.get_multipliers(),
            total_timesteps=ts,
            step=ts,
            vecnorm=self._env,  # persist running normalization stats
        )

        return self._policy

    def evaluate(self, n_episodes: int = 10) -> dict:
        """Run n_episodes deterministically; return raw mean reward and violation rate.

        The raw episodic return is read from ``info["episode"]["r"]`` populated by
        the SB3 Monitor wrapper (which records pre-VecNormalize rewards).  This is
        the same reward signal that the CurriculumCallback/MetricsObserver uses for
        advancement decisions, so the reported value is directly comparable to the
        curriculum advancement thresholds.

        Returns
        -------
        dict with keys:
            "mean_reward": float -- raw mean episode reward across n_episodes
            "violation_rate": float -- fraction of steps with constraint violations
        """
        if self._policy is None or self._env is None:
            raise RuntimeError("Call setup() before evaluate().")

        # Freeze VecNormalize stats during evaluation (deterministic mode)
        self._env.training = False

        episode_rewards: list[float] = []
        violation_count = 0
        total_steps = 0

        for _ in range(n_episodes):
            obs = self._env.reset()
            done = np.array([False])

            while not done[0]:
                action, _ = self._policy.predict(obs, deterministic=True)
                obs, _norm_reward, done, info = self._env.step(action)
                total_steps += 1

                if info and "constraint_violation" in info[0]:
                    violation_count += int(info[0]["constraint_violation"] > 0)

            # Read raw episodic return recorded by Monitor (pre-VecNormalize).
            # Mirrors the reward used by MetricsObserver for curriculum advancement.
            raw_ep_reward = float(info[0].get("episode", {}).get("r", 0.0))
            episode_rewards.append(raw_ep_reward)

        # Restore VecNormalize to training mode
        self._env.training = True

        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        violation_rate = violation_count / total_steps if total_steps > 0 else 0.0

        return {
            "mean_reward": mean_reward,
            "violation_rate": violation_rate,
        }

    # -- Internal ---------------------------------------------------------------

    @staticmethod
    def _build_default_phase_configs(curriculum_cfg: dict) -> list[PhaseConfig]:
        """Build default PhaseConfig list for all 7 curriculum phases.

        Uses per-phase thresholds matching curriculum.yaml if present,
        otherwise applies sensible defaults. One PhaseConfig per CurriculumPhase.
        """
        phases_from_cfg = curriculum_cfg.get("phases", [])
        defaults = [
            # phase,                           thresh,  min_ep, window, viol_lim, ampl
            (CurriculumPhase.STEADY_STATE,      0.85,     50,     50,   0.02, 0.0),
            (CurriculumPhase.LOAD_FOLLOW,       0.80,     50,     50,   0.05, 0.3),
            (CurriculumPhase.AMBIENT_TEMP,      0.75,     50,     50,   0.05, 10.0),
            (CurriculumPhase.EAF_TRANSIENTS,    0.70,     50,     50,   0.10, 200.0),
            (CurriculumPhase.LOAD_REJECTION,    0.65,     50,     50,   0.10, 0.5),
            (CurriculumPhase.COLD_STARTUP,      0.60,     50,     50,   0.15, 300.0),
            (CurriculumPhase.EMERGENCY_TRIP,    0.55,     50,     50,   0.20, 400.0),
        ]
        cfg_by_id = {
            int(phase_cfg.get("id", idx)): phase_cfg
            for idx, phase_cfg in enumerate(phases_from_cfg)
        }
        configs = []
        for i, (phase, thresh, min_ep, window, viol_lim, ampl) in enumerate(defaults):
            override = cfg_by_id.get(int(phase), {})
            configs.append(PhaseConfig(
                phase=phase,
                advance_threshold=override.get("advancement_threshold", thresh),
                min_episodes=override.get("min_episodes", min_ep),
                window_size=override.get("window_size", window),
                violation_rate_limit=override.get("violation_rate_limit", viol_lim),
                disturbance_amplitude=override.get("disturbance_amplitude", ampl),
                episode_length_steps=override.get("episode_length_steps", 720),
            ))
        return configs
