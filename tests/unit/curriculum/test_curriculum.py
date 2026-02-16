"""Unit tests for curriculum scheduling — CurriculumPhase, MetricsObserver, CurriculumScheduler.

TDD RED: written before implementation. All must fail until GREEN phase.
"""
from __future__ import annotations

import pytest


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _make_phase_configs(window=5, threshold=5.0, min_ep=3, viol_limit=0.0):
    """Return a list of 7 PhaseConfigs with overridable defaults."""
    from sco2rl.curriculum.phase import CurriculumPhase, PhaseConfig
    return [
        PhaseConfig(
            phase=phase,
            advance_threshold=threshold,
            min_episodes=min_ep,
            window_size=window,
            violation_rate_limit=viol_limit,
            disturbance_amplitude=float(phase),
        )
        for phase in CurriculumPhase
    ]


def _make_observer(window=5, threshold=5.0, viol_limit=0.0, min_ep=3):
    from sco2rl.curriculum.metrics_observer import MetricsObserver
    return MetricsObserver(
        config={
            "window_size": window,
            "advance_threshold": threshold,
            "violation_rate_limit": viol_limit,
            "min_episodes": min_ep,
        }
    )


def _make_scheduler(**kwargs):
    from sco2rl.curriculum.scheduler import CurriculumScheduler
    configs = _make_phase_configs(**kwargs)
    observer = _make_observer(
        window=kwargs.get("window", 5),
        threshold=kwargs.get("threshold", 5.0),
        viol_limit=kwargs.get("viol_limit", 0.0),
        min_ep=kwargs.get("min_ep", 3),
    )
    return CurriculumScheduler(phase_configs=configs, observer=observer)


# ── CurriculumPhase ────────────────────────────────────────────────────────────

class TestCurriculumPhase:
    def test_has_seven_members(self):
        from sco2rl.curriculum.phase import CurriculumPhase
        assert len(CurriculumPhase) == 7

    def test_is_int_enum(self):
        from enum import IntEnum
        from sco2rl.curriculum.phase import CurriculumPhase
        assert issubclass(CurriculumPhase, IntEnum)

    def test_values_zero_through_six(self):
        from sco2rl.curriculum.phase import CurriculumPhase
        assert list(CurriculumPhase) == [0, 1, 2, 3, 4, 5, 6]

    def test_comparable_with_less_than(self):
        from sco2rl.curriculum.phase import CurriculumPhase
        assert CurriculumPhase.STEADY_STATE < CurriculumPhase.LOAD_FOLLOW
        assert CurriculumPhase.EAF_TRANSIENTS < CurriculumPhase.EMERGENCY_TRIP

    def test_named_members(self):
        from sco2rl.curriculum.phase import CurriculumPhase
        assert CurriculumPhase.STEADY_STATE == 0
        assert CurriculumPhase.LOAD_FOLLOW == 1
        assert CurriculumPhase.AMBIENT_TEMP == 2
        assert CurriculumPhase.EAF_TRANSIENTS == 3
        assert CurriculumPhase.LOAD_REJECTION == 4
        assert CurriculumPhase.COLD_STARTUP == 5
        assert CurriculumPhase.EMERGENCY_TRIP == 6

    def test_can_cast_from_int(self):
        from sco2rl.curriculum.phase import CurriculumPhase
        assert CurriculumPhase(3) == CurriculumPhase.EAF_TRANSIENTS


# ── PhaseConfig ────────────────────────────────────────────────────────────────

class TestPhaseConfig:
    def test_fields_accessible(self):
        from sco2rl.curriculum.phase import CurriculumPhase, PhaseConfig
        cfg = PhaseConfig(
            phase=CurriculumPhase.STEADY_STATE,
            advance_threshold=8.0,
            min_episodes=50,
            window_size=50,
            violation_rate_limit=0.0,
            disturbance_amplitude=0.0,
        )
        assert cfg.phase == CurriculumPhase.STEADY_STATE
        assert cfg.advance_threshold == pytest.approx(8.0)
        assert cfg.min_episodes == 50
        assert cfg.window_size == 50
        assert cfg.violation_rate_limit == pytest.approx(0.0)
        assert cfg.disturbance_amplitude == pytest.approx(0.0)

    def test_is_dataclass(self):
        import dataclasses
        from sco2rl.curriculum.phase import PhaseConfig
        assert dataclasses.is_dataclass(PhaseConfig)


# ── MetricsObserver ────────────────────────────────────────────────────────────

class TestMetricsObserver:
    def test_initial_n_episodes_is_zero(self):
        obs = _make_observer()
        assert obs.n_episodes == 0

    def test_record_episode_increments_n_episodes(self):
        obs = _make_observer()
        obs.record_episode(reward=5.0, violation_fraction=0.0)
        assert obs.n_episodes == 1

    def test_should_advance_false_before_window_full(self):
        from sco2rl.curriculum.phase import CurriculumPhase
        obs = _make_observer(window=5, threshold=5.0)
        # Only 3 episodes — window needs 5
        for _ in range(3):
            obs.record_episode(reward=9.0, violation_fraction=0.0)
        assert obs.should_advance(CurriculumPhase.STEADY_STATE) is False

    def test_should_advance_false_before_min_episodes(self):
        from sco2rl.curriculum.phase import CurriculumPhase
        obs = _make_observer(window=3, threshold=5.0)
        # window=3, but min_episodes=10
        from sco2rl.curriculum.metrics_observer import MetricsObserver
        obs2 = MetricsObserver(config={
            "window_size": 3,
            "advance_threshold": 5.0,
            "violation_rate_limit": 0.0,
            "min_episodes": 10,
        })
        for _ in range(3):
            obs2.record_episode(reward=9.0, violation_fraction=0.0)
        assert obs2.should_advance(CurriculumPhase.STEADY_STATE) is False

    def test_should_advance_true_when_reward_above_threshold(self):
        from sco2rl.curriculum.phase import CurriculumPhase
        obs = _make_observer(window=3, threshold=5.0)
        for _ in range(3):
            obs.record_episode(reward=6.0, violation_fraction=0.0)
        assert obs.should_advance(CurriculumPhase.STEADY_STATE) is True

    def test_should_advance_false_when_reward_below_threshold(self):
        from sco2rl.curriculum.phase import CurriculumPhase
        obs = _make_observer(window=3, threshold=5.0)
        for _ in range(3):
            obs.record_episode(reward=4.0, violation_fraction=0.0)
        assert obs.should_advance(CurriculumPhase.STEADY_STATE) is False

    def test_should_advance_false_when_violation_rate_too_high(self):
        from sco2rl.curriculum.phase import CurriculumPhase
        # viol_limit=0.0 → zero tolerance
        obs = _make_observer(window=3, threshold=5.0, viol_limit=0.0)
        for _ in range(3):
            obs.record_episode(reward=9.0, violation_fraction=0.1)  # violations present
        assert obs.should_advance(CurriculumPhase.STEADY_STATE) is False

    def test_should_advance_true_with_permissive_violation_limit(self):
        from sco2rl.curriculum.phase import CurriculumPhase
        obs = _make_observer(window=3, threshold=5.0, viol_limit=0.5)
        for _ in range(3):
            obs.record_episode(reward=6.0, violation_fraction=0.3)  # below limit
        assert obs.should_advance(CurriculumPhase.STEADY_STATE) is True

    def test_get_mean_reward_correct(self):
        obs = _make_observer(window=3)
        obs.record_episode(reward=3.0, violation_fraction=0.0)
        obs.record_episode(reward=6.0, violation_fraction=0.0)
        obs.record_episode(reward=9.0, violation_fraction=0.0)
        assert obs.get_mean_reward() == pytest.approx(6.0)

    def test_get_mean_reward_rolling_window(self):
        obs = _make_observer(window=3)
        for r in [1.0, 2.0, 3.0, 10.0, 10.0, 10.0]:
            obs.record_episode(reward=r, violation_fraction=0.0)
        # Only last 3 in window: [10, 10, 10]
        assert obs.get_mean_reward() == pytest.approx(10.0)

    def test_get_violation_rate_correct(self):
        obs = _make_observer(window=4)
        obs.record_episode(reward=1.0, violation_fraction=0.1)
        obs.record_episode(reward=1.0, violation_fraction=0.0)
        obs.record_episode(reward=1.0, violation_fraction=0.2)
        obs.record_episode(reward=1.0, violation_fraction=0.0)
        # Mean violation fraction: (0.1+0.0+0.2+0.0)/4 = 0.075
        assert obs.get_violation_rate() == pytest.approx(0.075)

    def test_reset_clears_history(self):
        from sco2rl.curriculum.phase import CurriculumPhase
        obs = _make_observer(window=3, threshold=5.0)
        for _ in range(3):
            obs.record_episode(reward=9.0, violation_fraction=0.0)
        assert obs.should_advance(CurriculumPhase.STEADY_STATE) is True
        obs.reset()
        assert obs.n_episodes == 0
        assert obs.should_advance(CurriculumPhase.STEADY_STATE) is False

    def test_get_mean_reward_zero_when_no_episodes(self):
        obs = _make_observer()
        assert obs.get_mean_reward() == pytest.approx(0.0)


# ── CurriculumScheduler ────────────────────────────────────────────────────────

class TestCurriculumScheduler:
    def test_initial_phase_is_steady_state(self):
        from sco2rl.curriculum.phase import CurriculumPhase
        sched = _make_scheduler()
        assert sched.get_phase() == CurriculumPhase.STEADY_STATE

    def test_record_episode_returns_false_before_window_full(self):
        sched = _make_scheduler(window=5, threshold=5.0, min_ep=5)
        for i in range(4):
            advanced = sched.record_episode(reward=9.0, violation_fraction=0.0)
        assert advanced is False

    def test_record_episode_returns_true_on_advance(self):
        sched = _make_scheduler(window=3, threshold=5.0, min_ep=3, viol_limit=0.0)
        advanced = False
        for _ in range(3):
            advanced = sched.record_episode(reward=9.0, violation_fraction=0.0)
        assert advanced is True

    def test_phase_advances_after_threshold_met(self):
        from sco2rl.curriculum.phase import CurriculumPhase
        sched = _make_scheduler(window=3, threshold=5.0, min_ep=3, viol_limit=0.0)
        for _ in range(3):
            sched.record_episode(reward=9.0, violation_fraction=0.0)
        assert sched.get_phase() == CurriculumPhase.LOAD_FOLLOW

    def test_no_advance_when_reward_too_low(self):
        from sco2rl.curriculum.phase import CurriculumPhase
        sched = _make_scheduler(window=3, threshold=5.0, min_ep=3)
        for _ in range(10):
            sched.record_episode(reward=3.0, violation_fraction=0.0)
        assert sched.get_phase() == CurriculumPhase.STEADY_STATE

    def test_stays_at_emergency_trip_when_at_last_phase(self):
        from sco2rl.curriculum.phase import CurriculumPhase
        sched = _make_scheduler(window=1, threshold=0.0, min_ep=1, viol_limit=1.0)
        # Advance through all phases rapidly
        for _ in range(7 * 2):
            sched.record_episode(reward=10.0, violation_fraction=0.0)
        assert sched.get_phase() == CurriculumPhase.EMERGENCY_TRIP
        result = sched.advance()
        assert result is False
        assert sched.get_phase() == CurriculumPhase.EMERGENCY_TRIP

    def test_advance_manually_increments_phase(self):
        from sco2rl.curriculum.phase import CurriculumPhase
        sched = _make_scheduler()
        result = sched.advance()
        assert result is True
        assert sched.get_phase() == CurriculumPhase.LOAD_FOLLOW

    def test_phases_advance_sequentially(self):
        from sco2rl.curriculum.phase import CurriculumPhase
        sched = _make_scheduler(window=1, threshold=0.0, min_ep=1, viol_limit=1.0)
        expected = list(CurriculumPhase)[1:]  # phases 1–6 after initial
        for expected_phase in expected:
            sched.record_episode(reward=10.0, violation_fraction=0.0)
            assert sched.get_phase() == expected_phase

    def test_observer_reset_on_phase_advance(self):
        """After phase advance, observer history must be cleared."""
        sched = _make_scheduler(window=3, threshold=5.0, min_ep=3, viol_limit=0.0)
        for _ in range(3):
            sched.record_episode(reward=9.0, violation_fraction=0.0)
        # Verify observer was reset: n_episodes back to 0 via internal access
        assert sched._observer.n_episodes == 0

    def test_get_phase_config_matches_current_phase(self):
        from sco2rl.curriculum.phase import CurriculumPhase
        sched = _make_scheduler()
        cfg = sched.get_phase_config()
        assert cfg.phase == CurriculumPhase.STEADY_STATE

    def test_state_dict_round_trip(self):
        """state_dict() / load_state_dict() preserves exact phase and episode count."""
        from sco2rl.curriculum.phase import CurriculumPhase
        sched = _make_scheduler(window=3, threshold=5.0, min_ep=3)
        sched.advance()  # phase → LOAD_FOLLOW
        sched.record_episode(reward=4.0, violation_fraction=0.0)

        state = sched.state_dict()
        assert "phase" in state
        assert "n_episodes" in state

        # Create fresh scheduler and restore
        sched2 = _make_scheduler(window=3, threshold=5.0, min_ep=3)
        sched2.load_state_dict(state)
        assert sched2.get_phase() == CurriculumPhase.LOAD_FOLLOW
        assert sched2._observer.n_episodes == 1

    def test_no_regression_stays_at_higher_phase(self):
        """allow_regression=False: can never go backwards (enforced in advance)."""
        sched = _make_scheduler()
        sched.advance()  # → LOAD_FOLLOW
        sched.advance()  # → AMBIENT_TEMP
        # There's no public "retreat" — just confirm advance doesn't wrap
        from sco2rl.curriculum.phase import CurriculumPhase
        assert sched.get_phase() == CurriculumPhase.AMBIENT_TEMP
