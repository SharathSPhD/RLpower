"""Integration-style tests for LHS-to-FMU coupling (Phase 2C-D).

Tests prove that:
1. SCO2FMUEnv.reset(options=...) applies T_exhaust_K / mdot_exhaust_kgs to the FMU.
2. TrajectoryCollector passes LHS samples as options to env.reset().
3. LHS samples cover the parameter space uniformly.

Uses MockFMU configured with FMU-realistic variable names so the fragment
matching logic in SCO2FMUEnv._OPTIONS_TO_FMU resolves correctly.
"""
from __future__ import annotations

import numpy as np
import pytest

# ── FMU-realistic variable names (matching actual SCO2 model) ─────────────
FMU_OBS_VARS = [
    "main_compressor.T_inlet_rt",
    "turbine.T_inlet_rt",
    "main_compressor.p_outlet",
    "precooler.T_outlet_rt",
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

FMU_ACTION_VARS = [
    "regulator.T_init",        # heat source temperature (K) ← T_exhaust_K
    "regulator.m_flow_init",   # heat source mass flow (kg/s) ← mdot_exhaust_kgs
    "turbine.p_out",           # inventory valve (Pa)
    "precooler.T_output",      # cooling flow (K)
]

FMU_DESIGN_POINT = {
    "main_compressor.T_inlet_rt":  33.0,
    "turbine.T_inlet_rt":         750.0,
    "main_compressor.p_outlet":    20.0,
    "precooler.T_outlet_rt":       33.0,
    "turbine.W_turbine":           14.0,
    "main_compressor.W_comp":       4.0,
    "main_compressor.eta":          0.85,
    "recuperator.eta":              0.90,
    "recuperator.Q_actual":         30.0,
    "recuperator.T_hot_in":        200.0,
    "recuperator.T_cold_in":        80.0,
    "precooler.T_inlet_rt":        100.0,
    "main_compressor.T_outlet_rt":  90.0,
    "turbine.T_outlet_rt":         400.0,
}

ENV_CONFIG = {
    "obs_vars": FMU_OBS_VARS,
    "obs_bounds": {v: (-500.0, 2000.0) for v in FMU_OBS_VARS},
    "action_vars": FMU_ACTION_VARS,
    "action_config": {
        "regulator.T_init":      {"min": 800.0,  "max": 1473.0, "rate": 50.0},
        "regulator.m_flow_init": {"min":  10.0,  "max":  50.0,  "rate":  5.0},
        "turbine.p_out":         {"min": 7.0e6,  "max": 9.0e6,  "rate": 2e5},
        "precooler.T_output":    {"min": 305.65, "max": 316.0,  "rate":  1.0},
    },
    "history_steps": 1,
    "step_size": 5.0,
    "episode_max_steps": 50,
    "reward": {
        "w_tracking": 1.0, "w_efficiency": 0.3, "w_smoothness": 0.1,
        "rated_power_mw": 10.0, "design_efficiency": 0.47,
        "terminal_failure_reward": -100.0,
    },
    "safety": {"T_compressor_inlet_min": 32.2, "surge_margin_min": 0.05},
    "setpoint": {"W_net": 10.0},
}


def _make_fmu_env(design_point=None, **kwargs):
    """Create env with MockFMU that has FMU-realistic variable names."""
    from sco2rl.simulation.fmu.mock_fmu import MockFMU
    from sco2rl.environment.sco2_env import SCO2FMUEnv
    dp = design_point or FMU_DESIGN_POINT
    fmu = MockFMU(
        obs_vars=FMU_OBS_VARS,
        action_vars=FMU_ACTION_VARS,
        design_point=dp,
        seed=42,
        **kwargs,
    )
    return SCO2FMUEnv(fmu=fmu, config=ENV_CONFIG)


# ── Phase 2A: env.reset(options=...) applies initial inputs ───────────────

class TestEnvResetOptions:
    def test_reset_without_options_uses_base_setpoint(self):
        env = _make_fmu_env()
        obs, info = env.reset()
        assert env._setpoint["W_net"] == pytest.approx(10.0)
        env.close()

    def test_reset_with_W_setpoint_MW_updates_setpoint(self):
        env = _make_fmu_env()
        obs, info = env.reset(options={"W_setpoint_MW": 8.5})
        assert env._setpoint["W_net"] == pytest.approx(8.5)
        env.close()

    def test_reset_with_T_exhaust_K_calls_set_inputs(self):
        """Options with T_exhaust_K must reach the FMU set_inputs call."""
        from sco2rl.simulation.fmu.mock_fmu import MockFMU
        from sco2rl.environment.sco2_env import SCO2FMUEnv
        fmu = MockFMU(
            obs_vars=FMU_OBS_VARS,
            action_vars=FMU_ACTION_VARS,
            design_point=FMU_DESIGN_POINT,
            seed=42,
        )
        recorded_inputs: list[dict] = []
        _orig_set_inputs = fmu.set_inputs

        def _spy_set_inputs(inputs):
            recorded_inputs.append(dict(inputs))
            _orig_set_inputs(inputs)

        fmu.set_inputs = _spy_set_inputs

        env = SCO2FMUEnv(fmu=fmu, config=ENV_CONFIG)
        env.reset(options={"T_exhaust_K": 900.0, "mdot_exhaust_kgs": 35.0})

        # At least one set_inputs call must contain regulator.T_init
        assert any(
            "regulator.T_init" in call for call in recorded_inputs
        ), f"No set_inputs call with regulator.T_init found. Calls: {recorded_inputs}"

        # The value must match what was requested
        init_call = next(c for c in recorded_inputs if "regulator.T_init" in c)
        assert init_call["regulator.T_init"] == pytest.approx(900.0)
        assert init_call["regulator.m_flow_init"] == pytest.approx(35.0)
        env.close()

    def test_reset_options_info_contains_initial_inputs(self):
        env = _make_fmu_env()
        _, info = env.reset(options={"T_exhaust_K": 800.0})
        assert "initial_inputs" in info
        assert "regulator.T_init" in info["initial_inputs"]
        assert info["initial_inputs"]["regulator.T_init"] == pytest.approx(800.0)
        env.close()

    def test_reset_options_fragment_matching_works_for_both_vars(self):
        """Both T_init and m_flow_init fragments must resolve."""
        from sco2rl.simulation.fmu.mock_fmu import MockFMU
        from sco2rl.environment.sco2_env import SCO2FMUEnv
        fmu = MockFMU(
            obs_vars=FMU_OBS_VARS,
            action_vars=FMU_ACTION_VARS,
            design_point=FMU_DESIGN_POINT,
            seed=42,
        )
        calls = []
        _orig = fmu.set_inputs
        fmu.set_inputs = lambda inp: (calls.append(dict(inp)), _orig(inp))

        env = SCO2FMUEnv(fmu=fmu, config=ENV_CONFIG)
        env.reset(options={"T_exhaust_K": 1100.0, "mdot_exhaust_kgs": 25.0})

        merged = {}
        for c in calls:
            merged.update(c)
        assert "regulator.T_init" in merged, "T_exhaust_K must map to regulator.T_init"
        assert "regulator.m_flow_init" in merged, "mdot_exhaust_kgs must map to regulator.m_flow_init"
        env.close()

    def test_reset_with_no_matching_vars_does_not_crash(self):
        """Options that don't match any action var should not raise exceptions."""
        from sco2rl.simulation.fmu.mock_fmu import MockFMU
        from sco2rl.environment.sco2_env import SCO2FMUEnv
        fmu = MockFMU(
            obs_vars=FMU_OBS_VARS[:4],
            action_vars=["valve_A", "valve_B"],   # no T_init or m_flow_init
            design_point={v: FMU_DESIGN_POINT[v] for v in FMU_OBS_VARS[:4]},
            seed=42,
        )
        cfg = dict(ENV_CONFIG)
        cfg["action_vars"] = ["valve_A", "valve_B"]
        cfg["action_config"] = {
            "valve_A": {"min": 0.0, "max": 1.0, "rate": 0.1},
            "valve_B": {"min": 0.0, "max": 1.0, "rate": 0.1},
        }
        env = SCO2FMUEnv(fmu=fmu, config=cfg)
        obs, info = env.reset(options={"T_exhaust_K": 900.0})
        assert obs is not None
        assert info["initial_inputs"] == {}
        env.close()


# ── Phase 2B: TrajectoryCollector passes samples as options ───────────────

class TestTrajectoryCollectorPassesOptions:
    def test_sample_is_passed_as_options_to_env_reset(self):
        """collect_trajectory must call env.reset(options=sample_dict)."""
        from sco2rl.simulation.fmu.mock_fmu import MockFMU
        from sco2rl.environment.sco2_env import SCO2FMUEnv
        from sco2rl.surrogate.trajectory_collector import TrajectoryCollector

        reset_options_received: list = []

        fmu = MockFMU(
            obs_vars=FMU_OBS_VARS,
            action_vars=FMU_ACTION_VARS,
            design_point=FMU_DESIGN_POINT,
            seed=42,
        )
        env = SCO2FMUEnv(fmu=fmu, config=ENV_CONFIG)

        # Patch env.reset to capture options
        _orig_reset = env.reset
        def _spy_reset(*args, **kwargs):
            reset_options_received.append(kwargs.get("options"))
            return _orig_reset(*args, **kwargs)
        env.reset = _spy_reset

        collector = TrajectoryCollector(
            env=env,
            config={
                "trajectory_length_steps": 10,
                "action_perturbation": {"type": "random_walk", "step_std": 0.02, "clip": 0.1},
            },
            seed=42,
            raw_obs_dim=len(FMU_OBS_VARS),
        )
        sample = np.array([900.0, 30.0, 9.0])
        collector.collect_trajectory(sample)

        assert len(reset_options_received) == 1
        opts = reset_options_received[0]
        assert opts is not None, "env.reset was not called with options"
        assert opts["T_exhaust_K"] == pytest.approx(900.0)
        assert opts["mdot_exhaust_kgs"] == pytest.approx(30.0)
        assert opts["W_setpoint_MW"] == pytest.approx(9.0)
        env.close()

    def test_different_samples_produce_different_initial_setpoints(self):
        """Two different W_setpoint_MW values must produce different env._setpoint["W_net"]."""
        from sco2rl.surrogate.trajectory_collector import TrajectoryCollector

        env1 = _make_fmu_env()
        env2 = _make_fmu_env()
        cfg = {
            "trajectory_length_steps": 5,
            "action_perturbation": {"type": "random_walk", "step_std": 0.01, "clip": 0.05},
        }
        c1 = TrajectoryCollector(env=env1, config=cfg, seed=42, raw_obs_dim=len(FMU_OBS_VARS))
        c2 = TrajectoryCollector(env=env2, config=cfg, seed=42, raw_obs_dim=len(FMU_OBS_VARS))

        c1.collect_trajectory(np.array([900.0, 30.0, 7.0]))  # W_setpoint = 7 MW
        c2.collect_trajectory(np.array([900.0, 30.0, 12.0])) # W_setpoint = 12 MW

        # After collect_trajectory, setpoint should differ
        assert env1._setpoint["W_net"] == pytest.approx(7.0), \
            f"Expected W_net=7.0 but got {env1._setpoint['W_net']}"
        assert env2._setpoint["W_net"] == pytest.approx(12.0), \
            f"Expected W_net=12.0 but got {env2._setpoint['W_net']}"
        env1.close()
        env2.close()


# ── Phase 2D: LHS coverage validation ─────────────────────────────────────

class TestLHSCoverage:
    """Validate that the LHS sampler produces samples with good space coverage."""

    def test_lhs_samples_cover_full_range(self):
        """Each dimension's samples must span at least 90% of the declared range."""
        from sco2rl.surrogate.lhs_sampler import LatinHypercubeSampler
        lhs_config = {
            "parameter_ranges": {
                "T_exhaust_K":      {"min": 473.0,  "max": 1473.0},
                "mdot_exhaust_kgs": {"min":  10.0,  "max":   50.0},
                "W_setpoint_MW":    {"min":   7.0,  "max":   12.0},
            }
        }
        sampler = LatinHypercubeSampler(config=lhs_config, seed=42)
        samples = sampler.sample(n=1000)

        assert samples.shape == (1000, 3)

        low, high = sampler.get_bounds()
        ranges = high - low
        actual_ranges = samples.max(axis=0) - samples.min(axis=0)
        coverage_fractions = actual_ranges / ranges
        assert np.all(coverage_fractions >= 0.90), \
            f"LHS coverage < 90% in at least one dim: {coverage_fractions}"

    def test_lhs_samples_within_bounds(self):
        from sco2rl.surrogate.lhs_sampler import LatinHypercubeSampler
        lhs_config = {
            "parameter_ranges": {
                "T_exhaust_K":      {"min": 473.0,  "max": 1473.0},
                "mdot_exhaust_kgs": {"min":  10.0,  "max":   50.0},
                "W_setpoint_MW":    {"min":   7.0,  "max":   12.0},
            }
        }
        sampler = LatinHypercubeSampler(config=lhs_config, seed=99)
        samples = sampler.sample(n=500)
        low, high = sampler.get_bounds()
        assert np.all(samples >= low), "LHS samples below lower bound"
        assert np.all(samples <= high), "LHS samples above upper bound"

    def test_lhs_no_duplicate_rows(self):
        """LHS samples must all be unique (no repeated parameter combinations)."""
        from sco2rl.surrogate.lhs_sampler import LatinHypercubeSampler
        lhs_config = {
            "parameter_ranges": {
                "T_exhaust_K":      {"min": 473.0,  "max": 1473.0},
                "mdot_exhaust_kgs": {"min":  10.0,  "max":   50.0},
                "W_setpoint_MW":    {"min":   7.0,  "max":   12.0},
            }
        }
        sampler = LatinHypercubeSampler(config=lhs_config, seed=7)
        samples = sampler.sample(n=100)
        # Round to 4 decimal places then check uniqueness
        rounded = np.round(samples, 4)
        unique_rows = np.unique(rounded, axis=0)
        assert len(unique_rows) == len(samples), \
            f"Duplicate LHS samples found: {len(samples) - len(unique_rows)} duplicates"

    def test_lhs_uniform_coverage_per_dimension(self):
        """LHS guarantees ~n samples per n equal bins per dimension."""
        from sco2rl.surrogate.lhs_sampler import LatinHypercubeSampler
        lhs_config = {
            "parameter_ranges": {
                "T_exhaust_K":      {"min": 473.0,  "max": 1473.0},
                "mdot_exhaust_kgs": {"min":  10.0,  "max":   50.0},
                "W_setpoint_MW":    {"min":   7.0,  "max":   12.0},
            }
        }
        sampler = LatinHypercubeSampler(config=lhs_config, seed=0)
        n = 100
        samples = sampler.sample(n=n)
        low, high = sampler.get_bounds()

        # Each dimension should have exactly 1 sample per n equal-width bin
        n_bins = n
        for dim in range(samples.shape[1]):
            bin_edges = np.linspace(low[dim], high[dim], n_bins + 1)
            counts, _ = np.histogram(samples[:, dim], bins=bin_edges)
            # Every bin must have exactly 1 sample (LHS guarantee)
            assert np.all(counts == 1), \
                f"Dim {dim}: LHS uniformity violation, bin counts: {counts}"

    def test_different_seeds_produce_different_samples(self):
        from sco2rl.surrogate.lhs_sampler import LatinHypercubeSampler
        cfg = {
            "parameter_ranges": {
                "T_exhaust_K": {"min": 473.0, "max": 1473.0},
                "mdot_exhaust_kgs": {"min": 10.0, "max": 50.0},
                "W_setpoint_MW": {"min": 7.0, "max": 12.0},
            }
        }
        s1 = LatinHypercubeSampler(config=cfg, seed=1).sample(n=10)
        s2 = LatinHypercubeSampler(config=cfg, seed=2).sample(n=10)
        assert not np.allclose(s1, s2), "Different seeds should produce different samples"
