"""Tests for PIDBaseline."""
import numpy as np
import pytest

OBS_VARS = ["T_compressor_inlet", "T_turbine_inlet", "P_high", "P_low",
            "mdot_turbine", "mdot_main_compressor", "W_turbine", "W_main_compressor",
            "W_net", "eta_thermal", "surge_margin_main"]
ACTION_VARS = ["bypass_valve_opening", "igv_angle_normalized",
               "inventory_valve_opening", "cooling_flow_normalized"]
OBS_DIM = len(OBS_VARS)
ACT_DIM = len(ACTION_VARS)

PID_CONFIG = {
    "obs_vars": OBS_VARS,
    "action_vars": ACTION_VARS,
    "gains": {
        "bypass_valve_opening":    {"kp": 0.5, "ki": 0.1},
        "igv_angle_normalized":    {"kp": 0.3, "ki": 0.05},
        "inventory_valve_opening": {"kp": 0.2, "ki": 0.02},
        "cooling_flow_normalized": {"kp": 0.4, "ki": 0.08},
    },
    "setpoints": {
        "W_net":                  5.0,
        "T_turbine_inlet":       700.0,
        "P_high":                 18.0,
        "T_compressor_inlet":     33.0,
    },
    "measurement_indices": {
        "bypass_valve_opening":    8,
        "igv_angle_normalized":    1,
        "inventory_valve_opening": 2,
        "cooling_flow_normalized": 0,
    },
    "dt": 5.0,
}

@pytest.fixture
def pid():
    from sco2rl.deployment.inference.pid_baseline import PIDBaseline
    return PIDBaseline(PID_CONFIG)

def test_predict_returns_correct_shape(pid):
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    action, _ = pid.predict(obs)
    assert action.shape == (ACT_DIM,)

def test_action_within_bounds(pid):
    rng = np.random.default_rng(42)
    for _ in range(20):
        obs = rng.standard_normal(OBS_DIM).astype(np.float32)
        action, _ = pid.predict(obs)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)

def test_reset_clears_state(pid):
    obs = np.ones(OBS_DIM, dtype=np.float32) * 10.0
    for _ in range(5):
        pid.predict(obs)
    pid.reset()
    obs2 = np.zeros(OBS_DIM, dtype=np.float32)
    action_after_reset, _ = pid.predict(obs2)
    from sco2rl.deployment.inference.pid_baseline import PIDBaseline
    fresh_pid = PIDBaseline(PID_CONFIG)
    action_fresh, _ = fresh_pid.predict(obs2)
    np.testing.assert_array_almost_equal(action_after_reset, action_fresh)

def test_pid_controller_proportional():
    from sco2rl.deployment.inference.pid_baseline import PIDController
    ctrl = PIDController(kp=2.0, ki=0.0, setpoint=10.0, output_limits=(-1.0, 1.0))
    out = ctrl.compute(measurement=8.0, dt=1.0)
    assert out == pytest.approx(1.0)

def test_pid_controller_integral_accumulates():
    from sco2rl.deployment.inference.pid_baseline import PIDController
    ctrl = PIDController(kp=0.0, ki=1.0, setpoint=1.0, output_limits=(-10.0, 10.0))
    out1 = ctrl.compute(measurement=0.0, dt=1.0)
    out2 = ctrl.compute(measurement=0.0, dt=1.0)
    out3 = ctrl.compute(measurement=0.0, dt=1.0)
    assert out3 > out2 > out1

def test_predict_deterministic(pid):
    obs = np.array([33.0, 700.0, 18.0, 7.5, 95.0, 95.0, 12.0, 7.0, 5.0, 0.4, 0.3],
                   dtype=np.float32)
    a1, _ = pid.predict(obs, deterministic=True)
    pid.reset()
    a2, _ = pid.predict(obs, deterministic=True)
    np.testing.assert_array_equal(a1, a2)
