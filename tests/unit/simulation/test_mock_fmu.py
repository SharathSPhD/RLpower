"""Unit tests for FMUInterface ABC and MockFMU.

TDD RED: these tests were written BEFORE any implementation.
All tests in this file must fail (ImportError / AttributeError) until
the GREEN phase produces the implementation.
"""
from __future__ import annotations

import numpy as np
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

OBS_VARS = [
    "T_turbine_inlet",
    "T_compressor_inlet",
    "P_high",
    "P_low",
    "mdot_turbine",
    "W_turbine",
    "W_main_compressor",
    "W_net",
    "eta_thermal",
    "surge_margin_main",
    "T_exhaust_source",
    "mdot_exhaust_source",
]

ACTION_VARS = [
    "bypass_valve_opening",
    "igv_angle_normalized",
    "inventory_valve_opening",
    "cooling_flow_normalized",
]

DESIGN_POINT = {
    "T_turbine_inlet": 750.0,       # °C
    "T_compressor_inlet": 33.0,     # °C — above RULE-P1 threshold
    "P_high": 20.0,                 # MPa
    "P_low": 7.7,                   # MPa
    "mdot_turbine": 130.0,          # kg/s
    "W_turbine": 14.5,              # MW
    "W_main_compressor": 4.0,       # MW
    "W_net": 10.0,                  # MW (design target)
    "eta_thermal": 0.47,
    "surge_margin_main": 0.20,
    "T_exhaust_source": 800.0,      # °C
    "mdot_exhaust_source": 50.0,    # kg/s
}


@pytest.fixture
def mock_fmu():
    from sco2rl.simulation.fmu.mock_fmu import MockFMU
    fmu = MockFMU(obs_vars=OBS_VARS, action_vars=ACTION_VARS,
                  design_point=DESIGN_POINT, seed=42)
    fmu.initialize(start_time=0.0, stop_time=3600.0, step_size=5.0)
    return fmu


# ── FMUInterface ABC contract ─────────────────────────────────────────────────

class TestFMUInterfaceContract:
    """MockFMU must satisfy the FMUInterface ABC contract."""

    def test_mock_fmu_is_fmu_interface_subclass(self):
        from sco2rl.simulation.fmu.interface import FMUInterface
        from sco2rl.simulation.fmu.mock_fmu import MockFMU
        assert issubclass(MockFMU, FMUInterface)

    def test_fmu_interface_cannot_be_instantiated_directly(self):
        from sco2rl.simulation.fmu.interface import FMUInterface
        with pytest.raises(TypeError):
            FMUInterface()  # type: ignore[abstract]

    def test_interface_has_required_methods(self):
        from sco2rl.simulation.fmu.interface import FMUInterface
        for method in ("initialize", "set_inputs", "do_step", "get_outputs",
                       "reset", "close"):
            assert hasattr(FMUInterface, method), f"Missing: {method}"


# ── MockFMU initialization ─────────────────────────────────────────────────────

class TestMockFMUInitialization:
    def test_initialize_sets_time_to_zero(self, mock_fmu):
        assert mock_fmu.current_time == pytest.approx(0.0)

    def test_initialize_sets_step_size(self, mock_fmu):
        assert mock_fmu.step_size == pytest.approx(5.0)

    def test_initial_outputs_match_design_point(self, mock_fmu):
        outputs = mock_fmu.get_outputs()
        assert outputs["T_compressor_inlet"] == pytest.approx(
            DESIGN_POINT["T_compressor_inlet"], abs=2.0
        )
        assert outputs["W_net"] == pytest.approx(DESIGN_POINT["W_net"], abs=1.0)

    def test_initial_outputs_satisfy_rule_p1(self, mock_fmu):
        """T_compressor_inlet must be ≥ 32.2°C at design point."""
        outputs = mock_fmu.get_outputs()
        assert outputs["T_compressor_inlet"] >= 32.2


# ── MockFMU step execution ─────────────────────────────────────────────────────

class TestMockFMUStep:
    def test_do_step_returns_true_on_success(self, mock_fmu):
        result = mock_fmu.do_step(current_time=0.0, step_size=5.0)
        assert result is True

    def test_do_step_advances_time(self, mock_fmu):
        mock_fmu.do_step(current_time=0.0, step_size=5.0)
        assert mock_fmu.current_time == pytest.approx(5.0)

    def test_set_inputs_and_step_changes_outputs(self, mock_fmu):
        before = mock_fmu.get_outputs()["W_net"]
        # Open bypass valve fully → reduce W_net
        mock_fmu.set_inputs({"bypass_valve_opening": 0.8})
        mock_fmu.do_step(current_time=0.0, step_size=5.0)
        after = mock_fmu.get_outputs()["W_net"]
        # With bypass open, net power should decrease
        assert after < before

    def test_set_inputs_with_unknown_var_raises(self, mock_fmu):
        with pytest.raises((KeyError, ValueError)):
            mock_fmu.set_inputs({"nonexistent_variable": 1.0})

    def test_n_steps_counter_increments(self, mock_fmu):
        assert mock_fmu.n_steps == 0
        mock_fmu.do_step(current_time=0.0, step_size=5.0)
        assert mock_fmu.n_steps == 1
        mock_fmu.do_step(current_time=5.0, step_size=5.0)
        assert mock_fmu.n_steps == 2


# ── MockFMU reset ──────────────────────────────────────────────────────────────

class TestMockFMUReset:
    def test_reset_restores_design_point(self, mock_fmu):
        # Perturb state
        mock_fmu.set_inputs({"bypass_valve_opening": 0.9})
        mock_fmu.do_step(0.0, 5.0)
        mock_fmu.do_step(5.0, 5.0)
        # Reset
        mock_fmu.reset()
        outputs = mock_fmu.get_outputs()
        assert outputs["T_compressor_inlet"] == pytest.approx(
            DESIGN_POINT["T_compressor_inlet"], abs=2.0
        )

    def test_reset_zeroes_time_and_step_counter(self, mock_fmu):
        mock_fmu.do_step(0.0, 5.0)
        mock_fmu.reset()
        assert mock_fmu.current_time == pytest.approx(0.0)
        assert mock_fmu.n_steps == 0

    def test_n_resets_counter_increments(self, mock_fmu):
        assert mock_fmu.n_resets == 0
        mock_fmu.reset()
        assert mock_fmu.n_resets == 1
        mock_fmu.reset()
        assert mock_fmu.n_resets == 2

    def test_reset_is_deterministic_with_fixed_seed(self, mock_fmu):
        """Two resets with the same seed must produce identical first observations."""
        mock_fmu.reset()
        obs_a = mock_fmu.get_outputs()
        mock_fmu.reset()
        obs_b = mock_fmu.get_outputs()
        for key in obs_a:
            assert obs_a[key] == pytest.approx(obs_b[key], abs=1e-10)


# ── MockFMU failure injection ──────────────────────────────────────────────────

class TestMockFMUFailureInjection:
    def test_fail_at_step_returns_false(self):
        from sco2rl.simulation.fmu.mock_fmu import MockFMU
        fmu = MockFMU(obs_vars=OBS_VARS, action_vars=ACTION_VARS,
                      design_point=DESIGN_POINT, seed=42, fail_at_step=3)
        fmu.initialize(0.0, 3600.0, 5.0)
        assert fmu.do_step(0.0, 5.0) is True   # step 1
        assert fmu.do_step(5.0, 5.0) is True   # step 2
        assert fmu.do_step(10.0, 5.0) is False  # step 3 — fails

    def test_inlet_temp_drift_violates_rule_p1(self):
        """With inlet_temp_drift=True, T_compressor_inlet must drop below 32.2°C."""
        from sco2rl.simulation.fmu.mock_fmu import MockFMU
        fmu = MockFMU(obs_vars=OBS_VARS, action_vars=ACTION_VARS,
                      design_point=DESIGN_POINT, seed=42, inlet_temp_drift=True)
        fmu.initialize(0.0, 3600.0, 5.0)
        # Run for enough steps to trigger drift
        for i in range(20):
            fmu.do_step(i * 5.0, 5.0)
        outputs = fmu.get_outputs()
        assert outputs["T_compressor_inlet"] < 32.2

    def test_obs_noise_adds_variation(self):
        """With obs_noise_std > 0, consecutive steps should differ slightly."""
        from sco2rl.simulation.fmu.mock_fmu import MockFMU
        fmu = MockFMU(obs_vars=OBS_VARS, action_vars=ACTION_VARS,
                      design_point=DESIGN_POINT, seed=42, obs_noise_std=0.01)
        fmu.initialize(0.0, 3600.0, 5.0)
        fmu.do_step(0.0, 5.0)
        obs1 = fmu.get_outputs()["W_net"]
        fmu.do_step(5.0, 5.0)
        obs2 = fmu.get_outputs()["W_net"]
        # Noise should make successive observations differ
        assert obs1 != pytest.approx(obs2, abs=1e-10)


# ── MockFMU close ──────────────────────────────────────────────────────────────

class TestMockFMUClose:
    def test_close_does_not_raise(self, mock_fmu):
        mock_fmu.close()  # Should be a no-op for MockFMU, not raise

    def test_get_outputs_as_numpy_array(self, mock_fmu):
        """get_outputs_as_array returns a float32 numpy array in obs_vars order."""
        arr = mock_fmu.get_outputs_as_array()
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert arr.shape == (len(OBS_VARS),)
