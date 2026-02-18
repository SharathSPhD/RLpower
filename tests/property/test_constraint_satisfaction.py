from __future__ import annotations

import numpy as np
from hypothesis import given, settings, strategies as st

from sco2rl.environment.sco2_env import SCO2FMUEnv
from sco2rl.simulation.fmu.mock_fmu import MockFMU


def _env() -> SCO2FMUEnv:
    obs_vars = ["T_compressor_inlet", "surge_margin_main"]
    action_vars = ["a0", "a1", "a2", "a3"]
    fmu = MockFMU(
        obs_vars=obs_vars,
        action_vars=action_vars,
        design_point={"T_compressor_inlet": 33.0, "surge_margin_main": 0.2},
    )
    cfg = {
        "obs_vars": obs_vars,
        "obs_bounds": {"T_compressor_inlet": (20.0, 50.0), "surge_margin_main": (0.0, 1.0)},
        "action_vars": action_vars,
        "action_config": {a: {"min": 0.0, "max": 1.0, "rate": 0.1} for a in action_vars},
        "history_steps": 1,
        "step_size": 5.0,
        "episode_max_steps": 10,
        "reward": {"terminal_failure_reward": -100.0},
        "safety": {"T_compressor_inlet_min": 32.2, "surge_margin_min": 0.05},
        "setpoint": {"W_net": 10.0},
    }
    return SCO2FMUEnv(fmu=fmu, config=cfg)


@settings(max_examples=80, database=None)
@given(st.floats(min_value=20.0, max_value=32.19))
def test_low_compressor_inlet_always_triggers_termination(temp_value: float) -> None:
    env = _env()
    try:
        terminated, reason = env._check_terminated(
            fmu_success=True,
            raw_obs=np.array([temp_value, 0.2], dtype=np.float32),
        )
        assert terminated is True
        assert "T_compressor_inlet" in reason
    finally:
        env.close()
