from __future__ import annotations

import numpy as np
from hypothesis import given, settings, strategies as st

from sco2rl.environment.sco2_env import SCO2FMUEnv
from sco2rl.simulation.fmu.mock_fmu import MockFMU


def _make_env() -> SCO2FMUEnv:
    obs_vars = ["T_compressor_inlet", "W_turbine", "W_main_compressor"]
    action_vars = ["a0", "a1", "a2", "a3"]
    fmu = MockFMU(
        obs_vars=obs_vars,
        action_vars=action_vars,
        design_point={
            "T_compressor_inlet": 33.0,
            "W_turbine": 14.0,
            "W_main_compressor": 4.0,
        },
        seed=42,
    )
    cfg = {
        "obs_vars": obs_vars,
        "obs_bounds": {v: (0.0, 100.0) for v in obs_vars},
        "action_vars": action_vars,
        "action_config": {
            "a0": {"min": 0.0, "max": 1.0, "rate": 0.1},
            "a1": {"min": -2.0, "max": 2.0, "rate": 0.2},
            "a2": {"min": 5.0, "max": 9.0, "rate": 0.1},
            "a3": {"min": 10.0, "max": 20.0, "rate": 1.0},
        },
        "history_steps": 1,
        "step_size": 5.0,
        "episode_max_steps": 30,
        "reward": {"terminal_failure_reward": -100.0},
        "safety": {"T_compressor_inlet_min": 10.0, "surge_margin_min": -1.0},
        "setpoint": {"W_net": 10.0},
    }
    return SCO2FMUEnv(fmu=fmu, config=cfg)


@settings(max_examples=80, database=None)
@given(st.lists(st.floats(min_value=-2.0, max_value=2.0), min_size=4, max_size=4))
def test_applied_physical_action_stays_within_declared_bounds(action_values: list[float]) -> None:
    env = _make_env()
    try:
        env.reset(seed=0)
        action = np.asarray(action_values, dtype=np.float32)
        env.step(action)
        assert np.all(env.last_physical_action >= env._act_phys_min - 1e-6)
        assert np.all(env.last_physical_action <= env._act_phys_max + 1e-6)
    finally:
        env.close()
