"""Unit tests for CurriculumCallback — SB3 BaseCallback wiring curriculum.

TDD RED: written BEFORE implementation.
All tests MUST fail with ImportError until GREEN phase.

Design:
- CurriculumCallback(BaseCallback) records episode rewards + violations from rollout
- On rollout_end: calls MetricsObserver.should_advance() → if True, advances phase
  and calls env.env_method("set_curriculum_phase", new_phase) on all VecEnv workers
- Saves checkpoints at checkpoint_freq intervals via CheckpointManager

SB3 note: BaseCallback.training_env is a read-only property returning
self.model.get_env(). Tests must set cb.model = MagicMock() with
cb.model.get_env.return_value = mock_vecenv.
"""
from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_mock_scheduler(current_phase: int = 0, should_advance: bool = False):
    scheduler = MagicMock()
    scheduler.get_phase.return_value = current_phase
    scheduler.advance.return_value = should_advance
    return scheduler


def _make_mock_observer(should_advance: bool = False):
    observer = MagicMock()
    observer.should_advance.return_value = should_advance
    observer.n_episodes = 0
    return observer


def _make_mock_checkpoint_mgr():
    return MagicMock()


def _make_mock_vecenv(n_envs: int = 1):
    vec = MagicMock()
    vec.num_envs = n_envs
    return vec


def _make_callback(
    scheduler=None,
    observer=None,
    checkpoint_mgr=None,
    checkpoint_freq: int = 10_000,
    vecnorm=None,
    verbose: int = 0,
):
    from sco2rl.training.curriculum_callback import CurriculumCallback

    if scheduler is None:
        scheduler = _make_mock_scheduler()
    if observer is None:
        observer = _make_mock_observer()
    if checkpoint_mgr is None:
        checkpoint_mgr = _make_mock_checkpoint_mgr()

    return CurriculumCallback(
        scheduler=scheduler,
        observer=observer,
        checkpoint_mgr=checkpoint_mgr,
        checkpoint_freq=checkpoint_freq,
        vecnorm=vecnorm,
        verbose=verbose,
    )


def _wire_cb(cb, vec_env, num_timesteps: int = 100):
    """Wire a callback with a mock model + vecenv (SB3 requires model.get_env())."""
    cb.model = MagicMock()
    cb.model.get_env.return_value = vec_env
    cb.num_timesteps = num_timesteps
    return cb


def _make_locals(infos, dones):
    return {"infos": infos, "dones": np.array(dones, dtype=bool)}


# ── Import + class contract ────────────────────────────────────────────────────

class TestCurriculumCallbackImport:
    def test_module_importable(self):
        from sco2rl.training import curriculum_callback  # noqa: F401

    def test_class_importable(self):
        from sco2rl.training.curriculum_callback import CurriculumCallback  # noqa: F401

    def test_is_sb3_base_callback(self):
        from stable_baselines3.common.callbacks import BaseCallback
        from sco2rl.training.curriculum_callback import CurriculumCallback
        assert issubclass(CurriculumCallback, BaseCallback)

    def test_constructor_stores_scheduler(self):
        scheduler = _make_mock_scheduler()
        cb = _make_callback(scheduler=scheduler)
        assert cb.scheduler is scheduler

    def test_constructor_stores_observer(self):
        observer = _make_mock_observer()
        cb = _make_callback(observer=observer)
        assert cb.observer is observer

    def test_constructor_stores_checkpoint_mgr(self):
        ckpt = _make_mock_checkpoint_mgr()
        cb = _make_callback(checkpoint_mgr=ckpt)
        assert cb.checkpoint_mgr is ckpt

    def test_constructor_stores_checkpoint_freq(self):
        cb = _make_callback(checkpoint_freq=5000)
        assert cb.checkpoint_freq == 5000


# ── Episode recording ──────────────────────────────────────────────────────────

class TestEpisodeRecording:
    def test_on_rollout_end_calls_record_episode_for_each_done(self):
        """_on_rollout_end records one entry per done env."""
        observer = _make_mock_observer()
        scheduler = _make_mock_scheduler()
        cb = _make_callback(observer=observer, scheduler=scheduler)
        vec_env = _make_mock_vecenv(n_envs=2)
        _wire_cb(cb, vec_env, num_timesteps=100)
        cb.locals = _make_locals(
            infos=[
                {"episode": {"r": 5.0, "l": 10}, "constraint_violation": 0.1},
                {"episode": {"r": -2.0, "l": 8}, "constraint_violation": 0.0},
            ],
            dones=[True, True],
        )

        cb._on_rollout_end()
        assert observer.record_episode.call_count == 2

    def test_on_rollout_end_skips_non_done_envs(self):
        """_on_rollout_end skips infos for non-done envs."""
        observer = _make_mock_observer()
        scheduler = _make_mock_scheduler()
        cb = _make_callback(observer=observer, scheduler=scheduler)
        vec_env = _make_mock_vecenv(n_envs=2)
        _wire_cb(cb, vec_env, num_timesteps=100)
        cb.locals = _make_locals(
            infos=[
                {"episode": {"r": 5.0, "l": 10}, "constraint_violation": 0.0},
                {"episode": {"r": -2.0, "l": 8}, "constraint_violation": 0.0},
            ],
            dones=[True, False],
        )

        cb._on_rollout_end()
        assert observer.record_episode.call_count == 1

    def test_record_episode_passes_reward_and_violation(self):
        """record_episode called with correct (reward, violation_fraction) args."""
        observer = _make_mock_observer()
        scheduler = _make_mock_scheduler()
        cb = _make_callback(observer=observer, scheduler=scheduler)
        vec_env = _make_mock_vecenv(n_envs=1)
        _wire_cb(cb, vec_env, num_timesteps=50)
        cb.locals = _make_locals(
            infos=[{"episode": {"r": 7.5, "l": 20}, "constraint_violation": 0.15}],
            dones=[True],
        )

        cb._on_rollout_end()
        observer.record_episode.assert_called_once_with(reward=7.5, violation_fraction=0.15)


# ── Phase advancement ──────────────────────────────────────────────────────────

class TestPhaseAdvancement:
    def test_advances_phase_when_observer_says_so(self):
        """When observer.should_advance() is True, scheduler.advance() is called."""
        observer = _make_mock_observer(should_advance=True)
        scheduler = _make_mock_scheduler(current_phase=0)
        cb = _make_callback(observer=observer, scheduler=scheduler)
        vec_env = _make_mock_vecenv(n_envs=1)
        _wire_cb(cb, vec_env, num_timesteps=100)
        cb.locals = _make_locals(
            infos=[{"episode": {"r": 5.0, "l": 10}, "constraint_violation": 0.0}],
            dones=[True],
        )

        cb._on_rollout_end()
        scheduler.advance.assert_called_once()

    def test_does_not_advance_when_not_ready(self):
        """When observer.should_advance() is False, scheduler.advance() not called."""
        observer = _make_mock_observer(should_advance=False)
        scheduler = _make_mock_scheduler(current_phase=0)
        cb = _make_callback(observer=observer, scheduler=scheduler)
        vec_env = _make_mock_vecenv(n_envs=1)
        _wire_cb(cb, vec_env, num_timesteps=50)
        cb.locals = _make_locals(
            infos=[{"episode": {"r": 2.0, "l": 5}, "constraint_violation": 0.3}],
            dones=[True],
        )

        cb._on_rollout_end()
        scheduler.advance.assert_not_called()

    def test_calls_env_method_on_phase_advance(self):
        """On phase advance, env.env_method('set_curriculum_phase', new_phase) called."""
        observer = _make_mock_observer(should_advance=True)
        scheduler = _make_mock_scheduler(current_phase=0)
        scheduler.get_phase.side_effect = [0, 1]  # before advance=0, after=1
        scheduler.advance.return_value = True
        cb = _make_callback(observer=observer, scheduler=scheduler)
        vec_env = _make_mock_vecenv(n_envs=2)
        _wire_cb(cb, vec_env, num_timesteps=200)
        cb.locals = _make_locals(
            infos=[{"episode": {"r": 8.0, "l": 15}, "constraint_violation": 0.0}],
            dones=[True],
        )

        cb._on_rollout_end()
        vec_env.env_method.assert_called_once_with("set_curriculum_phase", 1)


# ── Checkpointing ──────────────────────────────────────────────────────────────

class TestCheckpointing:
    def test_saves_checkpoint_at_freq_interval(self):
        """Checkpoint saved when num_timesteps crosses checkpoint_freq boundary."""
        ckpt_mgr = _make_mock_checkpoint_mgr()
        observer = _make_mock_observer(should_advance=False)
        scheduler = _make_mock_scheduler(current_phase=0)
        cb = _make_callback(
            observer=observer,
            scheduler=scheduler,
            checkpoint_mgr=ckpt_mgr,
            checkpoint_freq=1000,
        )
        vec_env = _make_mock_vecenv(n_envs=1)
        _wire_cb(cb, vec_env, num_timesteps=1000)
        cb.locals = _make_locals(
            infos=[{"episode": {"r": 3.0, "l": 10}, "constraint_violation": 0.0}],
            dones=[True],
        )

        cb._on_rollout_end()
        ckpt_mgr.save.assert_called_once()

    def test_no_checkpoint_before_freq(self):
        """No checkpoint saved when num_timesteps < checkpoint_freq."""
        ckpt_mgr = _make_mock_checkpoint_mgr()
        observer = _make_mock_observer(should_advance=False)
        scheduler = _make_mock_scheduler(current_phase=0)
        cb = _make_callback(
            observer=observer,
            scheduler=scheduler,
            checkpoint_mgr=ckpt_mgr,
            checkpoint_freq=1000,
        )
        vec_env = _make_mock_vecenv(n_envs=1)
        _wire_cb(cb, vec_env, num_timesteps=500)
        cb.locals = _make_locals(
            infos=[{"episode": {"r": 1.0, "l": 5}, "constraint_violation": 0.0}],
            dones=[True],
        )

        cb._on_rollout_end()
        ckpt_mgr.save.assert_not_called()
