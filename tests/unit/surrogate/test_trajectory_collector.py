"""Unit tests for TrajectoryCollector -- TDD RED phase."""
from __future__ import annotations
import numpy as np
import pytest

OBS_VARS = [
    "T_compressor_inlet", "T_turbine_inlet", "P_high", "P_low", "mdot_turbine",
    "mdot_main_compressor", "W_turbine", "W_main_compressor", "W_net",
    "eta_thermal", "surge_margin_main",
]
ACTION_VARS = [
    "bypass_valve_opening", "igv_angle_normalized",
    "inventory_valve_opening", "cooling_flow_normalized",
]
DESIGN_POINT = {
    "T_compressor_inlet": 33.0,
    "T_turbine_inlet": 750.0,
    "P_high": 20.0,
    "P_low": 7.7,
    "mdot_turbine": 130.0,
    "mdot_main_compressor": 130.0,
    "W_turbine": 14.5,
    "W_main_compressor": 4.0,
    "W_net": 10.0,
    "eta_thermal": 0.47,
    "surge_margin_main": 0.20,
}
ENV_CONFIG = {
    "obs_vars": OBS_VARS,
    "obs_bounds": {v: (0.0, 1500.0) for v in OBS_VARS},
    "action_vars": ACTION_VARS,
    "action_config": {
        "bypass_valve_opening": {"min": 0.0, "max": 1.0, "rate": 0.05},
        "igv_angle_normalized": {"min": 0.0, "max": 1.0, "rate": 0.05},
        "inventory_valve_opening": {"min": 0.0, "max": 1.0, "rate": 0.02},
        "cooling_flow_normalized": {"min": 0.0, "max": 1.0, "rate": 0.05},
    },
    "history_steps": 1,
    "step_size": 5.0,
    "episode_max_steps": 200,
    "reward": {
        "w_tracking": 1.0, "w_efficiency": 0.3, "w_smoothness": 0.1,
        "rated_power_mw": 10.0, "design_efficiency": 0.40,
        "terminal_failure_reward": -100.0,
    },
    "safety": {"T_compressor_inlet_min": 32.2, "surge_margin_min": 0.05},
    "setpoint": {"W_net": 5.0},
}
COLLECTOR_CONFIG = {
    "trajectory_length_steps": 200,
    "action_perturbation": {
        "type": "random_walk",
        "step_std": 0.02,
        "clip": 0.1,
    },
}
SAMPLE = np.array([600.0, 55.0, 7.0])  # [T_exhaust, mdot, W_setpoint]

N_OBS = len(OBS_VARS)
N_ACT = len(ACTION_VARS)
TRAJ_LEN = 200


def _make_env():
    from sco2rl.simulation.fmu.mock_fmu import MockFMU
    from sco2rl.environment.sco2_env import SCO2FMUEnv
    fmu = MockFMU(obs_vars=OBS_VARS, action_vars=ACTION_VARS,
                  design_point=DESIGN_POINT, seed=42)
    return SCO2FMUEnv(fmu=fmu, config=ENV_CONFIG)


@pytest.fixture
def collector():
    from sco2rl.surrogate.trajectory_collector import TrajectoryCollector
    env = _make_env()
    return TrajectoryCollector(env=env, config=COLLECTOR_CONFIG, seed=42)


class TestTrajectoryCollectorKeys:
    def test_collect_trajectory_returns_dict_with_correct_keys(self, collector):
        traj = collector.collect_trajectory(SAMPLE)
        assert set(traj.keys()) == {"states", "actions", "metadata"}


class TestTrajectoryCollectorShapes:
    def test_states_shape(self, collector):
        traj = collector.collect_trajectory(SAMPLE)
        assert traj["states"].shape == (TRAJ_LEN, N_OBS), (
            f"Expected ({TRAJ_LEN}, {N_OBS}), got {traj['states'].shape}"
        )

    def test_actions_shape(self, collector):
        traj = collector.collect_trajectory(SAMPLE)
        assert traj["actions"].shape == (TRAJ_LEN - 1, N_ACT), (
            f"Expected ({TRAJ_LEN - 1}, {N_ACT}), got {traj['actions'].shape}"
        )

    def test_metadata_shape(self, collector):
        traj = collector.collect_trajectory(SAMPLE)
        assert traj["metadata"].shape == (3,)


class TestTrajectoryCollectorMetadata:
    def test_metadata_values_match_sample(self, collector):
        traj = collector.collect_trajectory(SAMPLE)
        np.testing.assert_array_almost_equal(traj["metadata"], SAMPLE)


class TestTrajectoryCollectorActionPerturbation:
    def test_action_perturbation_within_clip(self, collector):
        traj = collector.collect_trajectory(SAMPLE)
        actions = traj["actions"]  # (T-1, N_ACT), values in [-1, 1] normalized
        clip = COLLECTOR_CONFIG["action_perturbation"]["clip"]
        tol = 1e-6
        # Actions should be within [-clip - tol, clip + tol] range of initial zero
        # Actually actions accumulate via random walk, so we check consecutive diffs
        if actions.shape[0] > 1:
            diffs = np.diff(actions, axis=0)
            assert np.all(np.abs(diffs) <= clip + tol), (
                f"Max action step {np.abs(diffs).max():.4f} exceeds clip {clip}"
            )


class TestTrajectoryCollectorBatch:
    def test_collect_batch_returns_list_of_correct_length(self, collector):
        from sco2rl.surrogate.lhs_sampler import LatinHypercubeSampler
        lhs_config = {
            "parameter_ranges": {
                "T_exhaust_c": {"min": 200.0, "max": 1200.0},
                "mdot_exhaust_kg_s": {"min": 10.0, "max": 100.0},
                "W_net_setpoint_mw": {"min": 2.0, "max": 12.0},
            }
        }
        sampler = LatinHypercubeSampler(config=lhs_config, seed=42)
        samples = sampler.sample(5)
        batch = collector.collect_batch(samples)
        assert isinstance(batch, list)
        assert len(batch) == 5

    def test_collect_batch_each_dict_has_correct_keys(self, collector):
        samples = np.array([SAMPLE] * 3)
        batch = collector.collect_batch(samples)
        for traj in batch:
            assert set(traj.keys()) == {"states", "actions", "metadata"}


class TestTrajectoryCollectorPadding:
    def test_short_episode_padded_to_trajectory_length(self):
        """Even if env terminates early, states shape == (TRAJ_LEN, N_OBS)."""
        from sco2rl.simulation.fmu.mock_fmu import MockFMU
        from sco2rl.environment.sco2_env import SCO2FMUEnv
        from sco2rl.surrogate.trajectory_collector import TrajectoryCollector
        # fail_at_step=10 forces episode termination at step 10
        fmu = MockFMU(obs_vars=OBS_VARS, action_vars=ACTION_VARS,
                      design_point=DESIGN_POINT, seed=42, fail_at_step=10)
        env = SCO2FMUEnv(fmu=fmu, config=ENV_CONFIG)
        collector = TrajectoryCollector(env=env, config=COLLECTOR_CONFIG, seed=42)
        traj = collector.collect_trajectory(SAMPLE)
        assert traj["states"].shape == (TRAJ_LEN, N_OBS), (
            f"Expected ({TRAJ_LEN}, {N_OBS}) after padding, got {traj['states'].shape}"
        )
        assert traj["actions"].shape == (TRAJ_LEN - 1, N_ACT)
