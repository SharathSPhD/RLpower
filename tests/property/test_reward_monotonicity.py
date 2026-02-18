from __future__ import annotations

import numpy as np
from hypothesis import given, settings, strategies as st

from sco2rl.environment.sco2_env import SCO2FMUEnv
from sco2rl.simulation.fmu.mock_fmu import MockFMU


def _env() -> SCO2FMUEnv:
    obs_vars = ["W_turbine", "W_main_compressor", "eta_thermal", "T_compressor_inlet"]
    action_vars = ["a0", "a1", "a2", "a3"]
    fmu = MockFMU(
        obs_vars=obs_vars,
        action_vars=action_vars,
        design_point={
            "W_turbine": 14.0,
            "W_main_compressor": 4.0,
            "eta_thermal": 0.4,
            "T_compressor_inlet": 33.0,
        },
    )
    cfg = {
        "obs_vars": obs_vars,
        "obs_bounds": {v: (0.0, 100.0) for v in obs_vars},
        "action_vars": action_vars,
        "action_config": {a: {"min": 0.0, "max": 1.0, "rate": 0.1} for a in action_vars},
        "history_steps": 1,
        "step_size": 5.0,
        "episode_max_steps": 20,
        "reward": {
            "w_tracking": 1.0,
            "w_efficiency": 0.3,
            "w_smoothness": 0.1,
            "rated_power_mw": 10.0,
            "design_efficiency": 0.4,
            "terminal_failure_reward": -100.0,
        },
        "safety": {"T_compressor_inlet_min": 10.0, "surge_margin_min": -1.0},
        "setpoint": {"W_net": 10.0},
    }
    return SCO2FMUEnv(fmu=fmu, config=cfg)


@settings(max_examples=60, database=None)
@given(st.floats(min_value=0.0, max_value=8.0))
def test_reward_decreases_when_tracking_error_increases(delta_power: float) -> None:
    env = _env()
    try:
        env.reset(seed=0)
        action = np.zeros(4, dtype=np.float32)
        near = np.array([14.0, 4.0, 0.4, 33.0], dtype=np.float32)  # W_net=10
        far = np.array([14.0 - delta_power, 4.0, 0.4, 33.0], dtype=np.float32)  # lower W_net
        r_near = env._compute_reward(near, action)
        r_far = env._compute_reward(far, action)
        assert r_near >= r_far
    finally:
        env.close()
