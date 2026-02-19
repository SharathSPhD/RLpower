"""Tests for PIDController and PIDBaseline."""
import numpy as np
import pytest

OBS_VARS = [
    "main_compressor.T_inlet_rt",   # index 0 — compressor inlet °C
    "turbine.T_inlet_rt",           # index 1 — turbine inlet °C
    "main_compressor.p_outlet",     # index 2 — high-side pressure MPa
    "precooler.T_outlet_rt",        # index 3 — precooler outlet °C
    "turbine.W_turbine",
    "main_compressor.W_comp",
    "main_compressor.eta",
    "recuperator.eta",
    "recuperator.Q_actual",
    "recuperator.T_hot_in",
    "recuperator.T_cold_in",
    "precooler.T_inlet_rt",
    "main_compressor.T_outlet_rt",
    "turbine.T_outlet_rt",
]

ACTION_VARS = [
    "regulator.T_init",        # bypass_valve → measures turbine.T_inlet_rt
    "regulator.m_flow_init",   # igv          → measures main_compressor.T_inlet_rt
    "turbine.p_out",           # inventory_valve → measures main_compressor.p_outlet
    "precooler.T_output",      # cooling_flow → measures precooler.T_outlet_rt
]

OBS_DIM = len(OBS_VARS)
ACT_DIM = len(ACTION_VARS)

# Per-channel gains matching cross_validate_and_export.py defaults
PID_GAINS = {
    "regulator.T_init":      {"kp": 0.0015, "ki": 0.00015, "kd": 0.0003, "derivative_filter_tau": 0.2},
    "regulator.m_flow_init": {"kp": 0.06,   "ki": 0.006,   "kd": 0.012,  "derivative_filter_tau": 0.15},
    "turbine.p_out":         {"kp": 0.04,   "ki": 0.004,   "kd": 0.008,  "derivative_filter_tau": 0.15},
    "precooler.T_output":    {"kp": 0.08,   "ki": 0.008,   "kd": 0.016,  "derivative_filter_tau": 0.1},
}

# Setpoints keyed by exact obs var name for exact matching in _find_setpoint_key
PID_CONFIG = {
    "obs_vars": OBS_VARS,
    "action_vars": ACTION_VARS,
    "gains": PID_GAINS,
    "setpoints": {
        "turbine.T_inlet_rt":           550.0,   # bypass_valve setpoint (°C)
        "main_compressor.T_inlet_rt":   33.0,    # igv setpoint (°C)
        "main_compressor.p_outlet":     20.0,    # inventory_valve setpoint (MPa)
        "precooler.T_outlet_rt":        33.0,    # cooling_flow setpoint (°C)
    },
    "measurement_indices": {
        "regulator.T_init":      1,   # turbine.T_inlet_rt at index 1
        "regulator.m_flow_init": 0,   # main_compressor.T_inlet_rt at index 0
        "turbine.p_out":         2,   # main_compressor.p_outlet at index 2
        "precooler.T_output":    3,   # precooler.T_outlet_rt at index 3
    },
    "dt": 5.0,
    "n_obs": OBS_DIM,
    "history_steps": 1,
}


@pytest.fixture
def pid():
    from sco2rl.deployment.inference.pid_baseline import PIDBaseline
    return PIDBaseline(PID_CONFIG)


# ── PIDController unit tests ────────────────────────────────────────────────

class TestPIDController:
    def test_proportional_only(self):
        from sco2rl.deployment.inference.pid_baseline import PIDController
        ctrl = PIDController(kp=2.0, ki=0.0, kd=0.0, setpoint=10.0, output_limits=(-10.0, 10.0))
        out = ctrl.compute(measurement=8.0, dt=1.0)
        # error = 10 - 8 = 2; kp * error = 4
        assert out == pytest.approx(4.0)

    def test_integral_accumulates(self):
        from sco2rl.deployment.inference.pid_baseline import PIDController
        ctrl = PIDController(kp=0.0, ki=1.0, kd=0.0, setpoint=1.0, output_limits=(-100.0, 100.0))
        out1 = ctrl.compute(measurement=0.0, dt=1.0)
        out2 = ctrl.compute(measurement=0.0, dt=1.0)
        out3 = ctrl.compute(measurement=0.0, dt=1.0)
        assert out3 > out2 > out1

    def test_derivative_responds_to_error_change(self):
        from sco2rl.deployment.inference.pid_baseline import PIDController
        # kd only, tau=1 (no filtering), setpoint=10
        ctrl = PIDController(kp=0.0, ki=0.0, kd=1.0, setpoint=10.0,
                             output_limits=(-100.0, 100.0), derivative_filter_tau=1.0)
        # First call: error=5, prev_error=0 → d_error=5/1s=5 → out=5
        out1 = ctrl.compute(measurement=5.0, dt=1.0)
        assert out1 == pytest.approx(5.0, abs=1e-4)
        # Second call: error still 5, prev_error=5 → d_error=0 → out=0
        out2 = ctrl.compute(measurement=5.0, dt=1.0)
        assert out2 == pytest.approx(0.0, abs=1e-4)

    def test_derivative_filter_smooths(self):
        from sco2rl.deployment.inference.pid_baseline import PIDController
        # tau=0.5 means 50% new, 50% old → filtered_d after step
        ctrl = PIDController(kp=0.0, ki=0.0, kd=1.0, setpoint=0.0,
                             output_limits=(-100.0, 100.0), derivative_filter_tau=0.5)
        # Step: measurement goes from 0 to -10 suddenly (error jumps from 0 to 10)
        ctrl.compute(measurement=0.0, dt=1.0)    # initialize
        out = ctrl.compute(measurement=-10.0, dt=1.0)
        # raw_d = (10 - 0) / 1 = 10; filtered_d = 0.5 * 10 + 0.5 * 0 = 5 → kd*5=5
        assert out == pytest.approx(5.0, abs=0.5)

    def test_output_clipping(self):
        from sco2rl.deployment.inference.pid_baseline import PIDController
        ctrl = PIDController(kp=100.0, ki=0.0, kd=0.0, setpoint=10.0, output_limits=(-1.0, 1.0))
        assert ctrl.compute(measurement=0.0, dt=1.0) == pytest.approx(1.0)
        assert ctrl.compute(measurement=20.0, dt=1.0) == pytest.approx(-1.0)

    def test_reset_clears_all_state(self):
        from sco2rl.deployment.inference.pid_baseline import PIDController
        ctrl = PIDController(kp=1.0, ki=1.0, kd=1.0, setpoint=5.0, output_limits=(-100.0, 100.0))
        for _ in range(10):
            ctrl.compute(measurement=0.0, dt=1.0)
        ctrl.reset()
        # After reset, integral=0, prev_error=0, filtered_d=0
        # Next compute should behave like first call
        fresh = PIDController(kp=1.0, ki=1.0, kd=1.0, setpoint=5.0, output_limits=(-100.0, 100.0))
        obs = 3.0
        assert ctrl.compute(obs, dt=1.0) == pytest.approx(fresh.compute(obs, dt=1.0), abs=1e-6)

    def test_legacy_signature_no_kd(self):
        """PIDController without kd arg behaves as PI (kd defaults to 0)."""
        from sco2rl.deployment.inference.pid_baseline import PIDController
        ctrl = PIDController(kp=1.0, ki=0.5, setpoint=10.0, output_limits=(-10.0, 10.0))
        out = ctrl.compute(measurement=8.0, dt=1.0)
        # error=2; kp*2 + ki*2*1 = 2 + 1 = 3
        assert out == pytest.approx(3.0, abs=0.01)


# ── PIDBaseline unit tests ──────────────────────────────────────────────────

class TestPIDBaseline:
    def test_predict_returns_correct_shape(self, pid):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        action, state = pid.predict(obs)
        assert action.shape == (ACT_DIM,)
        assert state is None

    def test_action_within_bounds(self, pid):
        rng = np.random.default_rng(42)
        for _ in range(50):
            obs = rng.standard_normal(OBS_DIM).astype(np.float32)
            action, _ = pid.predict(obs)
            assert np.all(action >= -1.0), "action below -1"
            assert np.all(action <= 1.0), "action above +1"

    def test_reset_clears_state(self, pid):
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

    def test_predict_deterministic_across_calls(self, pid):
        obs = np.array([33.0, 550.0, 20.0, 33.0, 12.0, 7.0, 0.88, 0.90,
                        50.0, 110.0, 37.0, 400.0, 75.0, 650.0], dtype=np.float32)
        a1, _ = pid.predict(obs, deterministic=True)
        pid.reset()
        a2, _ = pid.predict(obs, deterministic=True)
        np.testing.assert_array_equal(a1, a2)

    def test_all_four_setpoints_resolve(self, pid):
        """Each of the 4 controllers must have a non-zero setpoint."""
        for act_name, ctrl in pid._controllers.items():
            assert ctrl._setpoint != 0.0, (
                f"Controller for {act_name} has setpoint=0.0 — setpoint matching failed"
            )

    def test_cooling_flow_setpoint_above_critical(self, pid):
        """Cooling flow (precooler.T_outlet_rt) setpoint must be above CO2 critical (31.1°C)."""
        ctrl = pid._controllers.get("precooler.T_output")
        assert ctrl is not None
        assert ctrl._setpoint > 31.1, "Cooling flow setpoint must be above CO2 critical point"

    def test_per_channel_gains_differ(self, pid):
        """Different channels must have different kp values (not uniform defaults)."""
        kp_values = [ctrl._kp for ctrl in pid._controllers.values()]
        assert len(set(kp_values)) > 1, "All PID channels have identical kp — gains are not per-channel"

    def test_history_slice_uses_latest_frame(self):
        from sco2rl.deployment.inference.pid_baseline import PIDBaseline
        cfg = dict(PID_CONFIG)
        cfg["history_steps"] = 2
        cfg["n_obs"] = OBS_DIM
        model = PIDBaseline(cfg)
        latest = np.array([33.0, 550.0, 20.0, 33.0, 12.0, 7.0, 0.88, 0.90,
                           50.0, 110.0, 37.0, 400.0, 75.0, 650.0], dtype=np.float32)
        obs_hist = np.concatenate([np.zeros_like(latest), latest])
        action_hist, _ = model.predict(obs_hist)
        model.reset()
        action_single, _ = model.predict(latest)
        np.testing.assert_allclose(action_hist, action_single, atol=1e-6)

    def test_integrator_resets_between_predict_calls_via_evaluator(self):
        """PolicyEvaluator calls model.reset() — verify PID integral clears."""
        from sco2rl.deployment.inference.pid_baseline import PIDBaseline, PIDController
        pid = PIDBaseline(PID_CONFIG)
        obs_error = np.zeros(OBS_DIM, dtype=np.float32)
        # Drive integral up
        for _ in range(20):
            pid.predict(obs_error)
        action_before_reset, _ = pid.predict(obs_error)
        pid.reset()
        action_after_reset, _ = pid.predict(obs_error)
        # After reset, integral is 0 → output should be smaller magnitude
        assert np.all(np.abs(action_after_reset) <= np.abs(action_before_reset) + 1e-6)
