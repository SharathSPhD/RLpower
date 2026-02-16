"""Integration smoke test: FMPyAdapter + real SCO2RecuperatedCycle.fmu.

Run inside Docker with real FMU:
    pytest tests/integration/test_fmpy_adapter_real_fmu.py -v --run-integration

ADR-S2-1: Actions map to SCOPE component parameters via setReal() between steps.
  bypass_valve → regulator.T_init     (K, heat source temperature)
  igv          → regulator.m_flow_init (kg/s, mass flow)
  inventory_valve → turbine.p_out     (Pa, low-side pressure)
  cooling_flow → precooler.T_output   (K, precooler target outlet T)

All verified via setReal() experiment 2026-02-16:
  regulator.T_init +100K → dW_turbine = +1.454 MW
  regulator.m_flow_init +25 kg/s → dW_turbine = +3.532 MW, dW_comp = +0.695 MW
  turbine.p_out +1 MPa → dW_turbine = -1.784 MW
  precooler.T_output +4.35K → dT_comp_inlet = +4.35°C
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# Skip unless --run-integration flag is passed (real FMU required)
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION"),
    reason="Set RUN_INTEGRATION=1 to run integration tests against real FMU",
)

FMU_PATH = "/workspace/artifacts/fmu_build/SCO2RecuperatedCycle.fmu"

# FMU variable names for simple_recuperated cycle
OBS_VARS = [
    "main_compressor.T_inlet_rt",   # K → °C
    "main_compressor.T_outlet_rt",  # K → °C
    "turbine.T_inlet_rt",           # K → °C
    "turbine.T_outlet_rt",          # K → °C
    "recuperator.T_hot_in",         # K → °C
    "recuperator.T_cold_in",        # K → °C
    "precooler.T_inlet_rt",         # K → °C
    "precooler.T_outlet_rt",        # K → °C
    "turbine.W_turbine",            # W → MW
    "main_compressor.W_comp",       # W → MW
    "main_compressor.eta",          # dimensionless
    "recuperator.eta",              # dimensionless
    "recuperator.Q_actual",         # W → MW
    "main_compressor.p_outlet",     # Pa → MPa
]

ACTION_VARS = [
    "regulator.T_init",       # bypass_valve: heat source T (K)
    "regulator.m_flow_init",  # igv: mass flow (kg/s)
    "turbine.p_out",          # inventory_valve: low-side pressure (Pa)
    "precooler.T_output",     # cooling_flow: precooler target T (K)
]

# Design-point values (before actions applied)
DESIGN_POINT_K = {
    "main_compressor.T_inlet_rt":  305.65,   # 32.5°C
    "turbine.T_inlet_rt":          973.15,   # 700°C
    "turbine.W_turbine":           13.42e6,  # 13.42 MW in W
    "main_compressor.W_comp":      2.64e6,   # 2.64 MW in W
    "main_compressor.p_outlet":    18.0e6,   # 18 MPa in Pa
}


@pytest.fixture(scope="module")
def fmpy_adapter():
    """Create FMPyAdapter with default_scale_offset for real FMU."""
    from sco2rl.simulation.fmu.fmpy_adapter import FMPyAdapter

    adapter = FMPyAdapter(
        fmu_path=FMU_PATH,
        obs_vars=OBS_VARS,
        action_vars=ACTION_VARS,
        instance_name="integration_test",
        scale_offset=FMPyAdapter.default_scale_offset(),
    )
    adapter.initialize(start_time=0.0, stop_time=3600.0, step_size=5.0)
    yield adapter
    adapter.close()


class TestFMPyAdapterInitialization:
    def test_fmu_file_exists(self):
        assert Path(FMU_PATH).exists(), f"FMU not found at {FMU_PATH}"

    def test_all_obs_vars_in_fmu(self, fmpy_adapter):
        """All requested obs variables exist in model description."""
        outputs = fmpy_adapter.get_outputs()
        for var in OBS_VARS:
            assert var in outputs, f"Missing obs var: {var}"

    def test_design_point_temperatures_in_celsius(self, fmpy_adapter):
        """After K→°C conversion: T_comp_inlet ≈ 32.5°C, T_turbine_inlet ≈ 700°C."""
        outputs = fmpy_adapter.get_outputs()
        T_comp = outputs["main_compressor.T_inlet_rt"]
        T_turb = outputs["turbine.T_inlet_rt"]
        assert 31.0 <= T_comp <= 35.0, f"T_comp_inlet out of range: {T_comp:.2f}°C"
        assert 698.0 <= T_turb <= 702.0, f"T_turbine_inlet out of range: {T_turb:.2f}°C"

    def test_design_point_power_in_mw(self, fmpy_adapter):
        """After W→MW conversion: W_turbine ≈ 13.4 MW, W_comp ≈ 2.6 MW."""
        outputs = fmpy_adapter.get_outputs()
        W_t = outputs["turbine.W_turbine"]
        W_c = outputs["main_compressor.W_comp"]
        assert 12.0 <= W_t <= 15.0, f"W_turbine out of range: {W_t:.3f}MW"
        assert 2.0 <= W_c <= 3.5, f"W_comp out of range: {W_c:.3f}MW"

    def test_design_point_pressure_in_mpa(self, fmpy_adapter):
        """After Pa→MPa conversion: P_high ≈ 18 MPa."""
        outputs = fmpy_adapter.get_outputs()
        P = outputs["main_compressor.p_outlet"]
        assert 17.5 <= P <= 18.5, f"P_high out of range: {P:.2f}MPa"

    def test_get_outputs_as_array_shape(self, fmpy_adapter):
        """get_outputs_as_array returns float32 array of length len(OBS_VARS)."""
        arr = fmpy_adapter.get_outputs_as_array()
        assert arr.dtype == np.float32
        assert arr.shape == (len(OBS_VARS),)


class TestFMPyAdapterStepping:
    def test_do_step_returns_true(self, fmpy_adapter):
        """One simulation step succeeds (CVODE converges)."""
        ok = fmpy_adapter.do_step(current_time=0.0, step_size=5.0)
        assert ok is True

    def test_cooling_flow_action_changes_comp_inlet(self, fmpy_adapter):
        """Setting precooler.T_output via set_inputs() changes T_comp_inlet.

        ADR-S2-1: setReal() on parameter variables works between CVODE steps.
        precooler.T_output +4.35K → T_comp_inlet +4.35°C (verified 2026-02-16).
        """
        # Baseline
        before = fmpy_adapter.get_outputs()
        T_comp_before = before["main_compressor.T_inlet_rt"]  # °C

        # Raise cooling target by 4 K (less cooling → higher compressor inlet T)
        fmpy_adapter.set_inputs({"precooler.T_output": 309.65})  # 305.65 + 4 K
        fmpy_adapter.do_step(current_time=5.0, step_size=5.0)
        after = fmpy_adapter.get_outputs()
        T_comp_after = after["main_compressor.T_inlet_rt"]  # °C

        # Should increase by ~4°C (1:1 relationship since precooler sets outlet exactly)
        delta = T_comp_after - T_comp_before
        assert 3.0 <= delta <= 5.0, f"T_comp_inlet delta={delta:.2f}°C, expected ~4°C"

        # Restore
        fmpy_adapter.set_inputs({"precooler.T_output": 305.65})

    def test_heat_source_action_changes_turbine_power(self, fmpy_adapter):
        """Raising regulator.T_init increases turbine power output.

        ADR-S2-1: regulator.T_init +100K → dW_turbine ≈ +1.45 MW (verified 2026-02-16).
        """
        fmpy_adapter.reset()
        fmpy_adapter.do_step(current_time=0.0, step_size=5.0)
        W_t_before = fmpy_adapter.get_outputs()["turbine.W_turbine"]  # MW

        # Raise heat source T by 100 K (= 100°C)
        fmpy_adapter.set_inputs({"regulator.T_init": 1073.15})  # 700°C → 800°C
        fmpy_adapter.do_step(current_time=5.0, step_size=5.0)
        W_t_after = fmpy_adapter.get_outputs()["turbine.W_turbine"]  # MW

        delta_mw = W_t_after - W_t_before
        assert 1.0 <= delta_mw <= 2.5, f"W_turbine delta={delta_mw:.3f}MW, expected ~1.45 MW"

    def test_mass_flow_action_scales_power(self, fmpy_adapter):
        """Raising mass flow (regulator.m_flow_init) proportionally increases W_turbine.

        ADR-S2-1: m_flow +25 kg/s (95→120) → dW_turbine ≈ +3.5 MW (verified 2026-02-16).
        """
        fmpy_adapter.reset()
        fmpy_adapter.do_step(current_time=0.0, step_size=5.0)
        W_t_before = fmpy_adapter.get_outputs()["turbine.W_turbine"]  # MW

        fmpy_adapter.set_inputs({"regulator.m_flow_init": 120.0})
        fmpy_adapter.do_step(current_time=5.0, step_size=5.0)
        W_t_after = fmpy_adapter.get_outputs()["turbine.W_turbine"]  # MW

        delta_mw = W_t_after - W_t_before
        assert 3.0 <= delta_mw <= 4.5, f"W_turbine delta={delta_mw:.3f}MW, expected ~3.5 MW"

    def test_pressure_action_changes_turbine_power(self, fmpy_adapter):
        """Raising turbine.p_out (low-side P) reduces turbine expansion work.

        ADR-S2-1: p_out +1 MPa → dW_turbine ≈ -1.78 MW (verified 2026-02-16).
        """
        fmpy_adapter.reset()
        fmpy_adapter.do_step(current_time=0.0, step_size=5.0)
        W_t_before = fmpy_adapter.get_outputs()["turbine.W_turbine"]  # MW

        fmpy_adapter.set_inputs({"turbine.p_out": 8.5e6})  # 7.5 → 8.5 MPa
        fmpy_adapter.do_step(current_time=5.0, step_size=5.0)
        W_t_after = fmpy_adapter.get_outputs()["turbine.W_turbine"]  # MW

        delta_mw = W_t_after - W_t_before
        assert -2.5 <= delta_mw <= -1.0, f"W_turbine delta={delta_mw:.3f}MW, expected ~-1.78 MW"

    def test_reset_restores_design_point(self, fmpy_adapter):
        """reset() returns FMU to design-point conditions."""
        # Apply some changes
        fmpy_adapter.set_inputs({"regulator.T_init": 1100.0})
        fmpy_adapter.do_step(current_time=0.0, step_size=5.0)

        # Reset
        fmpy_adapter.reset()
        fmpy_adapter.do_step(current_time=0.0, step_size=5.0)
        outputs = fmpy_adapter.get_outputs()

        # Check design point
        T_comp = outputs["main_compressor.T_inlet_rt"]
        assert 31.0 <= T_comp <= 35.0, f"After reset: T_comp_inlet={T_comp:.2f}°C (expected ~32.5)"


class TestSCO2FMUEnvIntegration:
    """End-to-end test: SCO2FMUEnv + real FMPyAdapter."""

    def test_env_step_loop_100_steps(self):
        """100 steps with real FMU: no crashes, rewards finite."""
        from sco2rl.simulation.fmu.fmpy_adapter import FMPyAdapter
        from sco2rl.environment.sco2_env import SCO2FMUEnv

        adapter = FMPyAdapter(
            fmu_path=FMU_PATH,
            obs_vars=OBS_VARS,
            action_vars=ACTION_VARS,
            instance_name="env_test",
            scale_offset=FMPyAdapter.default_scale_offset(),
        )

        # Minimal env config for real FMU
        config = {
            "obs_vars": OBS_VARS,
            "obs_bounds": {
                "main_compressor.T_inlet_rt":  (31.0, 43.0),
                "main_compressor.T_outlet_rt": (37.0, 110.0),
                "turbine.T_inlet_rt":          (527.0, 930.0),
                "turbine.T_outlet_rt":         (477.0, 780.0),
                "recuperator.T_hot_in":        (477.0, 780.0),
                "recuperator.T_cold_in":       (37.0, 110.0),
                "precooler.T_inlet_rt":        (127.0, 430.0),
                "precooler.T_outlet_rt":       (31.0, 43.0),
                "turbine.W_turbine":           (0.0, 25.0),
                "main_compressor.W_comp":      (0.0, 8.0),
                "main_compressor.eta":         (0.70, 0.92),
                "recuperator.eta":             (0.70, 0.98),
                "recuperator.Q_actual":        (0.0, 100.0),
                "main_compressor.p_outlet":    (14.0, 24.0),
            },
            "action_vars": ACTION_VARS,
            "action_config": {
                "regulator.T_init":      {"min": 800.0,  "max": 1200.0, "rate": 20.0},
                "regulator.m_flow_init": {"min": 60.0,   "max": 130.0,  "rate": 5.0},
                "turbine.p_out":         {"min": 7.0e6,  "max": 9.0e6,  "rate": 2.0e5},
                "precooler.T_output":    {"min": 305.65, "max": 315.0,  "rate": 0.5},
            },
            "history_steps": 1,
            "step_size": 5.0,
            "episode_max_steps": 100,
            "reward": {
                "w_tracking": 1.0,
                "w_efficiency": 0.3,
                "w_smoothness": 0.1,
                "rated_power_mw": 10.0,
                "design_efficiency": 0.40,
                "terminal_failure_reward": -100.0,
            },
            "safety": {
                "T_comp_min": 32.2,      # °C (RULE-P1)
                "surge_margin_min": 0.05,
            },
            "setpoint": {"W_net": 10.0},
        }

        env = SCO2FMUEnv(fmu=adapter, config=config)
        obs, _ = env.reset()
        assert obs.shape == (len(OBS_VARS),), f"Unexpected obs shape: {obs.shape}"

        total_reward = 0.0
        done = False
        for step_i in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            assert np.isfinite(reward), f"Non-finite reward at step {step_i}"
            assert np.all(np.isfinite(obs)), f"Non-finite obs at step {step_i}"
            if terminated or truncated:
                break

        assert step_i > 0, "Episode terminated on first step — likely solver failure"
