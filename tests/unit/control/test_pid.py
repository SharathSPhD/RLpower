"""Unit tests for sco2rl.control.pid.PIDController and sco2rl.control.multi_loop_pid.MultiLoopPID."""
from __future__ import annotations

import numpy as np
import pytest

from sco2rl.control.pid import PIDController
from sco2rl.control.multi_loop_pid import MultiLoopPID


# ─── PIDController ─────────────────────────────────────────────────────────────


def test_proportional_only():
    ctrl = PIDController(kp=2.0, ki=0.0, kd=0.0, setpoint=10.0, output_limits=(-10.0, 10.0))
    out = ctrl.compute(8.0)   # error = 2.0 → output = 4.0
    assert out == pytest.approx(4.0, abs=1e-6)


def test_integral_accumulates():
    ctrl = PIDController(kp=0.0, ki=1.0, kd=0.0, setpoint=1.0, output_limits=(-100.0, 100.0), dt=1.0)
    outs = [ctrl.compute(0.0) for _ in range(5)]
    # Each step: integral += 1.0*1 → output = 1, 2, 3, 4, 5
    assert outs[0] == pytest.approx(1.0, abs=1e-6)
    assert outs[4] == pytest.approx(5.0, abs=1e-6)
    # Strictly increasing
    assert all(outs[i + 1] > outs[i] for i in range(4))


def test_derivative_on_measurement_no_kick():
    """Derivative-on-measurement must not produce a spike on setpoint change."""
    ctrl = PIDController(
        kp=0.0, ki=0.0, kd=1.0, setpoint=0.0,
        output_limits=(-10.0, 10.0), dt=1.0,
    )
    # First call: no previous measurement → d_term = 0
    out0 = ctrl.compute(5.0)
    assert out0 == pytest.approx(0.0, abs=1e-6)

    # Measurement unchanged → derivative = 0
    out1 = ctrl.compute(5.0)
    assert out1 == pytest.approx(0.0, abs=1e-6)


def test_derivative_nonzero_on_measurement_change():
    ctrl = PIDController(
        kp=0.0, ki=0.0, kd=2.0, setpoint=0.0,
        output_limits=(-100.0, 100.0), dt=1.0,
    )
    ctrl.compute(5.0)        # prime the filter
    out = ctrl.compute(7.0)  # measurement increased by 2 → d_term = -Kd * dm/dt = -4
    assert out == pytest.approx(-4.0, abs=1e-6)


def test_output_saturated():
    ctrl = PIDController(kp=100.0, ki=0.0, setpoint=10.0, output_limits=(-1.0, 1.0))
    out = ctrl.compute(0.0)  # Large proportional output → saturation
    assert out == pytest.approx(1.0, abs=1e-6)


def test_anti_windup_prevents_unbounded_integral():
    """When output is saturated, integral must converge (not grow to +infinity).

    Back-calculation equilibrium: integral_eq ≈ output_sat/ki + error/aw_gain = 1 + 200 = 201
    Without anti-windup: integral = error * N_steps = 100,000 after 1000 steps.
    With anti-windup: integral converges to a finite bounded value.
    """
    ctrl = PIDController(
        kp=0.0, ki=1.0, kd=0.0, setpoint=100.0,
        output_limits=(-1.0, 1.0), anti_windup_gain=0.5, dt=1.0,
    )
    for _ in range(1000):
        ctrl.compute(0.0)  # Constant large positive error

    # Without anti-windup: integral would be ~100,000 (diverges).
    # With back-calculation: converges to ≈ 201 (finite and stable).
    assert np.isfinite(ctrl.integral), "Integral must be finite"
    assert abs(ctrl.integral) < 5000, (
        f"Integral {ctrl.integral:.1f} should be much smaller than unbounded 100,000"
    )


def test_reset_clears_state():
    ctrl = PIDController(kp=1.0, ki=1.0, kd=1.0, setpoint=5.0, dt=1.0)
    for _ in range(10):
        ctrl.compute(0.0)

    ctrl.reset()
    assert ctrl.integral == pytest.approx(0.0)
    # After reset, first call should give same result as fresh controller
    ctrl2 = PIDController(kp=1.0, ki=1.0, kd=1.0, setpoint=5.0, dt=1.0)
    assert ctrl.compute(2.0) == pytest.approx(ctrl2.compute(2.0), abs=1e-9)


def test_set_setpoint():
    ctrl = PIDController(kp=1.0, ki=0.0, setpoint=5.0, output_limits=(-10.0, 10.0))
    ctrl.set_setpoint(10.0)
    out = ctrl.compute(8.0)  # error = 2.0
    assert out == pytest.approx(2.0, abs=1e-6)


def test_derivative_filter():
    """Derivative filter should smooth out sudden changes."""
    ctrl_filtered = PIDController(
        kp=0.0, ki=0.0, kd=1.0, setpoint=0.0,
        output_limits=(-100.0, 100.0), derivative_filter_tau=5.0, dt=1.0,
    )
    ctrl_unfiltered = PIDController(
        kp=0.0, ki=0.0, kd=1.0, setpoint=0.0,
        output_limits=(-100.0, 100.0), derivative_filter_tau=0.0, dt=1.0,
    )
    # Prime both
    ctrl_filtered.compute(0.0)
    ctrl_unfiltered.compute(0.0)

    # Sudden step change
    out_f = ctrl_filtered.compute(10.0)
    out_u = ctrl_unfiltered.compute(10.0)

    # Filtered should have smaller absolute derivative response
    assert abs(out_f) < abs(out_u)


# ─── MultiLoopPID ──────────────────────────────────────────────────────────────

OBS_VARS = [
    "T_turbine_inlet", "T_compressor_inlet", "P_high", "P_low",
    "mdot_turbine", "W_turbine", "W_main_compressor", "W_net",
    "eta_thermal", "surge_margin_main",
]
ACTION_VARS = [
    "bypass_valve_opening", "igv_angle_normalized",
    "inventory_valve_opening", "cooling_flow_normalized",
]
N_OBS = len(OBS_VARS)
N_ACT = len(ACTION_VARS)

_PID_CONFIG = {
    "obs_vars": OBS_VARS,
    "action_vars": ACTION_VARS,
    "n_obs": N_OBS,
    "history_steps": 1,
    "dt": 5.0,
    "setpoints": {
        "W_net": 10.0,
        "T_turbine_inlet": 750.0,
        "P_high": 20.0,
        "T_compressor_inlet": 33.0,
    },
    "measurement_indices": {
        "bypass_valve_opening": 7,
        "igv_angle_normalized": 0,
        "inventory_valve_opening": 2,
        "cooling_flow_normalized": 1,
    },
}


@pytest.fixture
def pid():
    return MultiLoopPID(config=_PID_CONFIG)


def test_predict_correct_shape(pid):
    obs = np.zeros(N_OBS, dtype=np.float32)
    action, state = pid.predict(obs)
    assert action.shape == (N_ACT,)
    assert state is None


def test_predict_within_bounds(pid):
    rng = np.random.default_rng(0)
    for _ in range(50):
        obs = rng.standard_normal(N_OBS).astype(np.float32)
        action, _ = pid.predict(obs)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_reset_clears_pid_state(pid):
    obs = np.ones(N_OBS, dtype=np.float32) * 100.0
    for _ in range(10):
        pid.predict(obs)
    pid.reset()

    pid2 = MultiLoopPID(config=_PID_CONFIG)
    obs0 = np.zeros(N_OBS, dtype=np.float32)
    a1, _ = pid.predict(obs0)
    a2, _ = pid2.predict(obs0)
    np.testing.assert_array_almost_equal(a1, a2)


def test_predict_with_history(pid):
    """Latest history window should be used for measurement."""
    cfg = dict(_PID_CONFIG)
    cfg["history_steps"] = 2
    model = MultiLoopPID(config=cfg)

    old_obs = np.zeros(N_OBS, dtype=np.float32)
    new_obs = np.array(
        [750.0, 33.0, 20.0, 7.5, 95.0, 12.0, 2.0, 10.0, 0.4, 0.15],
        dtype=np.float32,
    )
    stacked = np.concatenate([old_obs, new_obs])  # old first, new last

    action_hist, _ = model.predict(stacked)
    model.reset()
    action_new_only, _ = model.predict(new_obs)
    np.testing.assert_allclose(action_hist, action_new_only, atol=1e-5)


def test_name(pid):
    assert pid.name == "MultiLoopPID"


def test_improved_gains_vs_legacy():
    """MultiLoopPID with IMC gains should produce larger action than legacy kp=0.02."""
    # Near-setpoint: W_net = 8 (error = 2 MW on channel 0 → bypass_valve)
    obs = np.array([750.0, 33.0, 20.0, 7.5, 95.0, 12.0, 2.0, 8.0, 0.4, 0.15], dtype=np.float32)

    pid_improved = MultiLoopPID(config=_PID_CONFIG)
    action_imp, _ = pid_improved.predict(obs)

    # Legacy config uses uniform kp=0.02, ki=0.001 for all channels
    legacy_config = dict(_PID_CONFIG)
    legacy_config["gains"] = {
        av: {"kp": 0.02, "ki": 0.001, "kd": 0.0} for av in ACTION_VARS
    }
    pid_legacy = MultiLoopPID(config=legacy_config)
    action_legacy, _ = pid_legacy.predict(obs)

    # For channel 0 (bypass_valve → W_net, error = 2 MW):
    # Improved Kp=0.25 → action ≈ 0.5 (more responsive)
    # Legacy Kp=0.02 → action ≈ 0.04 (weak response)
    assert abs(action_imp[0]) > abs(action_legacy[0]), (
        f"Improved PID ({abs(action_imp[0]):.4f}) should produce larger action than "
        f"legacy ({abs(action_legacy[0]):.4f}) on W_net error of 2 MW"
    )
