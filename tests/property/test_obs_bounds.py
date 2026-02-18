from __future__ import annotations

import numpy as np
from hypothesis import given, settings, strategies as st

from sco2rl.environment.sco2_env import SCO2FMUEnv
from sco2rl.simulation.fmu.mock_fmu import MockFMU


def _env() -> SCO2FMUEnv:
    obs_vars = ["T_compressor_inlet", "W_turbine", "W_main_compressor", "W_net"]
    action_vars = ["a0", "a1", "a2", "a3"]
    fmu = MockFMU(
        obs_vars=obs_vars,
        action_vars=action_vars,
        design_point={
            "T_compressor_inlet": 33.0,
            "W_turbine": 14.0,
            "W_main_compressor": 4.0,
            "W_net": 10.0,
        },
    )
    cfg = {
        "obs_vars": obs_vars,
        "obs_bounds": {
            "T_compressor_inlet": (20.0, 50.0),
            "W_turbine": (0.0, 30.0),
            "W_main_compressor": (0.0, 15.0),
            "W_net": (-5.0, 20.0),
        },
        "action_vars": action_vars,
        "action_config": {a: {"min": 0.0, "max": 1.0, "rate": 0.1} for a in action_vars},
        "history_steps": 3,
        "step_size": 5.0,
        "episode_max_steps": 25,
        "reward": {"terminal_failure_reward": -100.0},
        "safety": {"T_compressor_inlet_min": 10.0, "surge_margin_min": -1.0},
        "setpoint": {"W_net": 10.0},
    }
    return SCO2FMUEnv(fmu=fmu, config=cfg)


@settings(max_examples=40, database=None)
@given(st.integers(min_value=1, max_value=15))
def test_observation_vector_remains_inside_observation_space(n_steps: int) -> None:
    env = _env()
    try:
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        for _ in range(n_steps):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                break
    finally:
        env.close()
