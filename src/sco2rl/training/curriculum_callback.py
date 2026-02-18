"""CurriculumCallback — SB3 BaseCallback that wires curriculum into the training loop.

Responsibilities:
  1. On each env step (_on_step): extract completed episode stats from infos and
     record them in MetricsObserver.  Episode counting MUST happen in _on_step
     (not _on_rollout_end) because episode boundaries rarely coincide with the
     rollout boundary: with episode_max_steps=120 and n_steps=2048 per rollout,
     2048 % 120 = 8, so the last step of a rollout is almost never an episode
     terminal.  _on_rollout_end only sees infos/dones from the final step of the
     rollout, missing all mid-rollout episode completions.
  2. On rollout end (_on_rollout_end): update Lagrange multipliers, check phase
     advancement, save checkpoints.
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

import random
from collections import defaultdict
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
    interleave_ratio:
        When the curriculum is at Phase 6, fraction of workers that will be
        redirected to a randomly sampled earlier phase (0–5) after each episode
        completion.  0.0 disables interleaving (default / backward-compatible).
        Only Phase-6 episodes (not replay episodes) are recorded in the
        MetricsObserver so advancement accounting is unaffected.
    phase_steps:
        Mapping of phase_id → episode_length_steps, required when
        interleave_ratio > 0 so replay episodes use the correct episode length.
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
        interleave_ratio: float = 0.0,
        phase_steps: dict[int, int] | None = None,
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

        # Interleaved curriculum replay
        self.interleave_ratio: float = float(interleave_ratio)
        self._phase_steps: dict[int, int] = phase_steps or {}
        # Per-worker tracking: which phase is the worker currently running?
        # Populated lazily on first done signal (n_envs not known until learn()).
        self._worker_phase: dict[int, int] = {}

        # Track last checkpoint timestep to compute intervals
        self._last_checkpoint_at: int = 0
        # Accumulate per-rollout violation stats for Lagrange multiplier update
        self._rollout_violation_sums: dict[str, float] = defaultdict(float)
        self._rollout_violation_count: int = 0
        # Diagnostic counters
        self._log_episode_interval: int = 100  # print observer stats every N episodes
        self._last_logged_episodes: int = 0

    # -- BaseCallback interface -------------------------------------------------

    def _on_step(self) -> bool:
        """Called after every env step.

        Records completed episodes into MetricsObserver here (not in
        _on_rollout_end) to avoid missing episodes whose boundaries do not
        coincide with the rollout boundary.  Also accumulates per-step
        constraint violations for the Lagrange multiplier update.

        When interleave_ratio > 0 and the curriculum is at Phase 6, workers
        that finish an episode are probabilistically assigned to replay an
        earlier phase (0–5) for their next episode.  Only Phase-6 episodes
        are fed to MetricsObserver so advancement thresholds are evaluated
        against the target phase, not the replay phases.
        """
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", np.zeros(len(infos), dtype=bool))
        current_phase = self.scheduler.get_phase()

        for idx, (info, done) in enumerate(zip(infos, dones)):
            if not done:
                continue

            ep_info = info.get("episode", {})
            reward = float(ep_info.get("r", 0.0))
            violation = float(info.get("constraint_violation", 0.0))

            # Determine which phase this worker was running
            ep_phase = self._worker_phase.get(idx, current_phase)

            # Only count target-phase episodes in MetricsObserver so that
            # replay episodes don't distort the advancement window.
            if ep_phase == current_phase:
                self.observer.record_episode(reward=reward, violation_fraction=violation)

            # Accumulate per-episode constraint violations for Lagrange update
            # (include all episodes regardless of phase — Lagrange safety is global)
            violations = info.get("constraint_violations", {})
            if isinstance(violations, dict):
                for key, value in violations.items():
                    self._rollout_violation_sums[str(key)] += float(value)
                self._rollout_violation_count += 1

            # ── Interleaved replay: reassign this worker's next episode phase ──
            if (
                current_phase == 6
                and self.interleave_ratio > 0.0
                and self._phase_steps
            ):
                if random.random() < self.interleave_ratio:
                    replay_ph = random.randint(0, 5)
                    replay_steps = self._phase_steps.get(replay_ph, 360)
                    try:
                        self.training_env.env_method(
                            "set_curriculum_phase",
                            int(replay_ph),
                            int(replay_steps),
                            indices=[idx],
                        )
                        self._worker_phase[idx] = replay_ph
                    except Exception:
                        pass  # non-fatal: worker stays on current phase
                else:
                    # Restore worker to current phase if it was on a replay phase
                    if self._worker_phase.get(idx, current_phase) != current_phase:
                        current_steps = self._phase_steps.get(current_phase, 360)
                        try:
                            self.training_env.env_method(
                                "set_curriculum_phase",
                                int(current_phase),
                                int(current_steps),
                                indices=[idx],
                            )
                        except Exception:
                            pass
                    self._worker_phase[idx] = current_phase
            else:
                self._worker_phase[idx] = current_phase

        # Periodic diagnostic logging every _log_episode_interval episodes
        n_ep = self.observer.n_episodes
        if n_ep > 0 and n_ep - self._last_logged_episodes >= self._log_episode_interval:
            mean_r = self.observer.get_mean_reward()
            viol_r = self.observer.get_violation_rate()
            phase = current_phase
            print(
                f"[CurriculumCallback] step={self.num_timesteps} phase={phase} "
                f"episodes={n_ep} mean_reward={mean_r:.2f} "
                f"violation_rate={viol_r:.3f}",
                flush=True,
            )
            self._last_logged_episodes = n_ep

        return True  # returning False would stop training

    def _on_rollout_end(self) -> None:
        """Called after each rollout (n_steps * n_envs transitions collected).

        Updates Lagrange multipliers from rollout-accumulated violations,
        checks curriculum advancement, and saves checkpoints.
        """
        # Keep Lagrange multipliers synchronized with observed rollout violations.
        if self._lagrangian_model is not None and self._rollout_violation_count > 0:
            mean_violations = {
                key: value / self._rollout_violation_count
                for key, value in self._rollout_violation_sums.items()
            }
            self._lagrangian_model.update_multipliers(mean_violations)

        # Reset rollout accumulators
        self._rollout_violation_sums = defaultdict(float)
        self._rollout_violation_count = 0

        # 2. Check if curriculum should advance
        current_phase = self.scheduler.get_phase()
        mean_r = self.observer.get_mean_reward()
        viol_r = self.observer.get_violation_rate()
        if self.verbose >= 1:
            print(
                f"[CurriculumCallback] rollout_end step={self.num_timesteps} "
                f"phase={current_phase} episodes={self.observer.n_episodes} "
                f"mean_r={mean_r:.2f} viol_rate={viol_r:.3f}",
                flush=True,
            )
        if self.observer.should_advance(current_phase):
            advanced = self.scheduler.advance()
            if advanced:
                new_phase = self.scheduler.get_phase()
                phase_cfg = self.scheduler.get_phase_config()
                self.model.get_env().env_method(
                    "set_curriculum_phase",
                    int(new_phase),
                    int(phase_cfg.episode_length_steps),
                )
                print(
                    f"[CurriculumCallback] *** ADVANCED to phase {new_phase} at "
                    f"step={self.num_timesteps} ***",
                    flush=True,
                )

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

        When self.vecnorm is set, the VecNormalize running stats are persisted
        alongside the model so that observation normalisation is correctly
        restored on resume (fixes curriculum-stuck-at-phase-0 bug).
        """
        ts = self.num_timesteps
        save_model = self._lagrangian_model if self._lagrangian_model is not None else self.model
        multipliers = getattr(save_model, "get_multipliers", lambda: {})()
        self.checkpoint_mgr.save(
            model=save_model,
            vecnorm_stats={"obs_rms": None},  # legacy field kept for schema compat
            curriculum_phase=int(self.scheduler.get_phase()),
            lagrange_multipliers=multipliers,
            total_timesteps=ts,
            step=ts,
            vecnorm=self.vecnorm,  # persist running stats; None is safe (no-op)
        )
