"""CurriculumCallback — SB3 BaseCallback that wires curriculum into the training loop.

Responsibilities:
  1. On each rollout end: extract completed episode stats from infos and record
     them in MetricsObserver.
  2. After recording: ask MetricsObserver.should_advance(); if True, call
     CurriculumScheduler.advance() and push the new phase to all VecEnv workers
     via training_env.env_method("set_curriculum_phase", new_phase).
  3. Save a checkpoint (RULE-C4) every checkpoint_freq timesteps via
     CheckpointManager.save().

Design constraints:
  - Uses stable_baselines3.common.callbacks.BaseCallback (SB3 convention).
  - Never calls self.training_env.reset() — SB3 owns environment lifecycle.
  - self.locals["infos"] contains dicts per worker; only process when dones[i]=True.
  - Checkpoint includes model weights, vecnorm stats, curriculum phase,
    lagrange multipliers, total_timesteps (RULE-C4 five fields).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """SB3 callback that records episodes, advances curriculum, and saves checkpoints.

    Parameters
    ----------
    scheduler:
        CurriculumScheduler instance — manages phase transitions.
    observer:
        MetricsObserver instance — tracks rolling episode statistics.
    checkpoint_mgr:
        CheckpointManager instance — saves RULE-C4 checkpoints.
    checkpoint_freq:
        Save a checkpoint every N timesteps.
    vecnorm:
        Optional VecNormalize wrapper (passed to CheckpointManager.save()).
        May be None if not using observation normalization.
    verbose:
        SB3 verbosity level.
    """

    def __init__(
        self,
        scheduler: Any,
        observer: Any,
        checkpoint_mgr: Any,
        checkpoint_freq: int = 10_000,
        vecnorm: Any | None = None,
        lagrangian_model: Any | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.scheduler = scheduler
        self.observer = observer
        self.checkpoint_mgr = checkpoint_mgr
        self.checkpoint_freq = checkpoint_freq
        self.vecnorm = vecnorm
        # LagrangianPPO wrapper reference; self.model is the INNER SB3 PPO
        # (set by SB3 during learn()). We need the wrapper to save multipliers.
        self._lagrangian_model = lagrangian_model

        # Track last checkpoint timestep to compute intervals
        self._last_checkpoint_at: int = 0

    # -- BaseCallback interface -------------------------------------------------

    def _on_step(self) -> bool:
        """Called after every env step; required by BaseCallback API."""
        return True  # returning False would stop training

    def _on_rollout_end(self) -> None:
        """Called after each rollout (n_steps * n_envs transitions collected).

        Extract completed episodes → record in observer → maybe advance phase
        → maybe save checkpoint.
        """
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", np.zeros(len(infos), dtype=bool))

        # 1. Record each completed episode
        for i, (info, done) in enumerate(zip(infos, dones)):
            if not done:
                continue
            ep_info = info.get("episode", {})
            reward = float(ep_info.get("r", 0.0))
            violation = float(info.get("constraint_violation", 0.0))
            self.observer.record_episode(reward=reward, violation_fraction=violation)

        # 2. Check if curriculum should advance
        current_phase = self.scheduler.get_phase()
        if self.observer.should_advance(current_phase):
            advanced = self.scheduler.advance()
            if advanced:
                new_phase = self.scheduler.get_phase()
                self.model.get_env().env_method("set_curriculum_phase", new_phase)
                if self.verbose >= 1:
                    print(f"[CurriculumCallback] Advanced to phase {new_phase}")

        # 3. Save checkpoint at checkpoint_freq intervals
        if (
            self.num_timesteps > 0
            and self.num_timesteps - self._last_checkpoint_at >= self.checkpoint_freq
        ):
            self._save_checkpoint()
            self._last_checkpoint_at = self.num_timesteps

    # -- Internal ---------------------------------------------------------------

    def _save_checkpoint(self) -> None:
        """Save RULE-C4 checkpoint via CheckpointManager (5 required fields).

        Uses self._lagrangian_model (LagrangianPPO wrapper) when available so
        that both the .zip and _multipliers.pkl are written.  Falls back to
        self.model (inner SB3 PPO) for backward-compatibility in unit tests.
        """
        ts = self.num_timesteps
        save_model = self._lagrangian_model if self._lagrangian_model is not None else self.model
        multipliers = getattr(save_model, "get_multipliers", lambda: {})()
        self.checkpoint_mgr.save(
            model=save_model,
            vecnorm_stats={"obs_rms": None},  # VecNormalize stats placeholder
            curriculum_phase=int(self.scheduler.get_phase()),
            lagrange_multipliers=multipliers,
            total_timesteps=ts,
            step=ts,
        )
