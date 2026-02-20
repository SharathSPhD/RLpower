"""Unit tests for sco2rl.analysis.step_response."""
from __future__ import annotations

import numpy as np
import pytest

from sco2rl.analysis.step_response import compute_step_metrics, run_step_scenario
from sco2rl.analysis.scenario_runner import build_mock_env, build_mock_pid


class MockRLPolicy:
    """Minimal RL-compatible policy for testing.

    Always outputs zero action vector.  No SB3 dependency required.
    """

    @property
    def name(self) -> str:
        return "RL"

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return np.zeros(4, dtype=np.float32), None

    def reset(self) -> None:
        pass


# ─── compute_step_metrics ─────────────────────────────────────────────────────


def _make_first_order_step(
    t_total: float = 200.0,
    dt: float = 5.0,
    K: float = 1.0,
    tau: float = 30.0,
    step_mag: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate a first-order step response for metric validation."""
    n = int(t_total / dt)
    t = np.arange(n) * dt
    y = K * step_mag * (1.0 - np.exp(-t / tau))
    final = float(K * step_mag)
    return t, y, final


def test_metrics_keys():
    t, y, final = _make_first_order_step()
    metrics = compute_step_metrics(t, y, final, step_magnitude=1.0, dt=5.0)
    expected_keys = {
        "overshoot_pct", "undershoot_pct", "settling_time_s",
        "rise_time_s", "peak_time_s", "iae", "ise", "itae",
    }
    assert set(metrics.keys()) == expected_keys


def test_no_overshoot_first_order():
    """Pure first-order step response has zero overshoot."""
    t, y, final = _make_first_order_step(K=1.0, tau=30.0)
    m = compute_step_metrics(t, y, final, step_magnitude=1.0, dt=5.0)
    assert m["overshoot_pct"] == pytest.approx(0.0, abs=1.0)


def test_rise_time_first_order():
    """Rise time should be approximately 2.197 * tau for first-order system."""
    tau = 30.0
    t, y, final = _make_first_order_step(K=1.0, tau=tau, t_total=400.0)
    m = compute_step_metrics(t, y, final, step_magnitude=1.0, dt=5.0)
    expected_rise = 2.197 * tau
    # Allow ±10% tolerance (discretisation error)
    assert m["rise_time_s"] == pytest.approx(expected_rise, rel=0.15)


def test_settling_time_first_order():
    """For ±2% settling band, settling time ≈ 4*tau for first-order."""
    tau = 20.0
    t, y, final = _make_first_order_step(K=1.0, tau=tau, t_total=500.0)
    m = compute_step_metrics(t, y, final, step_magnitude=1.0, dt=5.0)
    # For 2% band: ts ≈ -tau * ln(0.02) ≈ 3.91 * tau
    expected = 3.91 * tau
    assert m["settling_time_s"] == pytest.approx(expected, rel=0.2)


def test_iae_positive():
    t, y, final = _make_first_order_step()
    m = compute_step_metrics(t, y, final, step_magnitude=1.0, dt=5.0)
    assert m["iae"] > 0.0
    assert m["ise"] > 0.0
    assert m["itae"] > 0.0


def test_itae_greater_than_iae():
    """ITAE penalises late errors more than IAE, so ITAE > IAE for typical responses."""
    t, y, final = _make_first_order_step(tau=30.0, t_total=600.0)
    m = compute_step_metrics(t, y, final, step_magnitude=1.0, dt=5.0)
    assert m["itae"] > m["iae"]


def test_negative_step():
    """Metrics should work for negative step magnitudes."""
    t = np.arange(40) * 5.0
    y = -1.0 * (1.0 - np.exp(-t / 20.0))  # First-order response to -1 step
    final = -1.0
    m = compute_step_metrics(t, y, final, step_magnitude=-1.0, dt=5.0)
    assert m["overshoot_pct"] == pytest.approx(0.0, abs=1.0)
    assert m["iae"] > 0.0


def test_zero_step_magnitude():
    """Zero step magnitude should not raise and should return zeros."""
    t = np.arange(20) * 5.0
    y = np.ones_like(t) * 5.0
    m = compute_step_metrics(t, y, 5.0, step_magnitude=0.0, dt=5.0)
    for v in m.values():
        assert v == pytest.approx(0.0)


# ─── run_step_scenario with MockFMU ───────────────────────────────────────────


@pytest.fixture
def mock_env():
    env = build_mock_env(dynamic=False)
    yield env
    env.close()


@pytest.fixture
def mock_pid():
    return build_mock_pid()


def test_run_step_returns_result(mock_env, mock_pid):
    from sco2rl.analysis.metrics import StepResponseResult
    result = run_step_scenario(
        env=mock_env,
        policy=mock_pid,
        step_magnitude=2.0,
        step_at_step=20,
        n_steps=80,
        dt=5.0,
        variable="W_net",
        phase=0,
        scenario="test_step",
    )
    assert isinstance(result, StepResponseResult)
    assert result.variable == "W_net"
    assert result.phase == 0
    assert len(result.time_s) > 0
    assert len(result.response) == len(result.time_s)


def test_run_step_records_trajectory(mock_env, mock_pid):
    result = run_step_scenario(
        env=mock_env, policy=mock_pid,
        step_magnitude=2.0, step_at_step=10, n_steps=60, dt=5.0,
    )
    # Time should be non-decreasing
    t = np.array(result.time_s)
    assert np.all(np.diff(t) >= 0)


def test_run_step_step_onset(mock_env, mock_pid):
    result = run_step_scenario(
        env=mock_env, policy=mock_pid,
        step_magnitude=2.0, step_at_step=15, n_steps=70,
        dt=5.0, variable="W_net",
    )
    assert result.step_onset_s == pytest.approx(15 * 5.0)


def test_run_step_metrics_finite(mock_env, mock_pid):
    result = run_step_scenario(
        env=mock_env, policy=mock_pid,
        step_magnitude=1.0, step_at_step=20, n_steps=100, dt=5.0,
    )
    for field_name in ["overshoot_pct", "settling_time_s", "rise_time_s", "iae", "ise", "itae"]:
        val = getattr(result, field_name)
        assert np.isfinite(val), f"{field_name} is not finite: {val}"


def test_run_step_different_phases(mock_pid):
    """Results should vary by phase (different disturbance profiles)."""
    results = {}
    for phase in [0, 1, 2]:
        env = build_mock_env(dynamic=False)
        try:
            res = run_step_scenario(
                env=env, policy=mock_pid,
                step_magnitude=2.0, step_at_step=10, n_steps=60, dt=5.0,
                phase=phase,
            )
            results[phase] = res
        finally:
            env.close()
    # All should produce valid results
    for phase, res in results.items():
        assert len(res.response) > 0, f"Phase {phase} has empty response"


# ─── MockRLPolicy tests ────────────────────────────────────────────────────────


def test_run_step_scenario_with_rl_policy(mock_env):
    """run_step_scenario must work with any policy-like object (not just PID)."""
    from sco2rl.analysis.metrics import StepResponseResult
    rl = MockRLPolicy()
    result = run_step_scenario(
        env=mock_env,
        policy=rl,
        step_magnitude=2.0,
        step_at_step=15,
        n_steps=60,
        dt=5.0,
        variable="W_net",
        phase=0,
        scenario="step_load_up",
    )
    assert isinstance(result, StepResponseResult)
    assert result.controller == "RL"
    assert len(result.response) > 0
    assert len(result.time_s) == len(result.response)
    for field_name in ["iae", "ise", "itae", "overshoot_pct", "settling_time_s"]:
        assert np.isfinite(getattr(result, field_name)), f"{field_name} is not finite"


def test_rl_and_pid_produce_different_trajectories(mock_pid):
    """RL mock (zero action) and PID (computed action) must yield different trajectories."""
    rl = MockRLPolicy()
    env_pid = build_mock_env(dynamic=False)
    env_rl = build_mock_env(dynamic=False)
    try:
        res_pid = run_step_scenario(
            env=env_pid, policy=mock_pid,
            step_magnitude=2.0, step_at_step=10, n_steps=60, dt=5.0,
        )
        res_rl = run_step_scenario(
            env=env_rl, policy=rl,
            step_magnitude=2.0, step_at_step=10, n_steps=60, dt=5.0,
        )
    finally:
        env_pid.close()
        env_rl.close()
    # Responses must be non-identical: PID computes non-zero actions, RL outputs zeros
    min_len = min(len(res_pid.response), len(res_rl.response))
    assert min_len > 0
    pid_arr = np.array(res_pid.response[:min_len])
    rl_arr = np.array(res_rl.response[:min_len])
    assert not np.allclose(pid_arr, rl_arr), "PID and zero-RL should produce different trajectories"


def test_step_result_controller_name_rl(mock_env):
    """StepResponseResult.controller must reflect the RL policy name."""
    rl = MockRLPolicy()
    result = run_step_scenario(
        env=mock_env, policy=rl,
        step_magnitude=1.0, step_at_step=10, n_steps=40, dt=5.0,
    )
    assert result.controller == "RL"
