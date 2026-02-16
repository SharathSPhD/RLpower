"""Unit tests for SurrogateEnv -- TDD RED phase."""

import numpy as np
import pytest
import torch

from sco2rl.surrogate.fno_model import FNO1d
from sco2rl.surrogate.surrogate_env import SurrogateEnv


OBS_VARS = [
    "T_compressor_inlet", "T_turbine_inlet", "P_high", "P_low",
    "mdot_turbine", "W_net", "eta_thermal", "surge_margin_main",
]
ACTION_VARS = [
    "bypass_valve_opening", "igv_angle_normalized",
    "inventory_valve_opening", "cooling_flow_normalized",
]
N_OBS = len(OBS_VARS)   # 8
N_ACT = len(ACTION_VARS)  # 4
INPUT_DIM = N_OBS + N_ACT  # 12
HISTORY_STEPS = 1

SMALL_FNO_CONFIG = {
    "modes": 4, "width": 16, "n_layers": 2,
    "input_dim": INPUT_DIM, "output_dim": N_OBS,
}

ENV_CONFIG = {
    "obs_vars": OBS_VARS,
    "obs_bounds": {v: (0.0, 1.0) for v in OBS_VARS},
    "action_vars": ACTION_VARS,
    "action_config": {
        v: {"phys_min": 0.0, "phys_max": 1.0, "rate_limit": 0.05}
        for v in ACTION_VARS
    },
    "history_steps": HISTORY_STEPS,
    "step_size": 5.0,
    "episode_max_steps": 50,
    "reward": {
        "w_tracking": 1.0,
        "w_efficiency": 0.3,
        "w_smoothness": 0.1,
        "rated_power_mw": 10.0,
        "design_efficiency": 0.40,
        "terminal_failure_reward": -100.0,
    },
    "safety": {
        "T_comp_inlet_min_c": -999.0,   # disabled so unit tests dont terminate immediately
        "surge_margin_min": -999.0,  # disabled so unit tests dont terminate immediately
    },
    "setpoint": {"W_net_setpoint": 5.0},
    "obs_design_point": {v: 0.5 for v in OBS_VARS},
}


@pytest.fixture
def tiny_model():
    return FNO1d(**SMALL_FNO_CONFIG)


@pytest.fixture
def env(tiny_model):
    return SurrogateEnv(model=tiny_model, config=ENV_CONFIG, device="cpu")


# ─── Space shape tests ────────────────────────────────────────────────────────

def test_surrogate_env_obs_space_shape(env):
    """Observation space shape must be (N_OBS * history_steps,)."""
    expected = (N_OBS * HISTORY_STEPS,)
    assert env.observation_space.shape == expected, (
        f"Expected {expected}, got {env.observation_space.shape}"
    )


def test_surrogate_env_action_space_shape(env):
    """Action space shape must be (N_ACT,)."""
    expected = (N_ACT,)
    assert env.action_space.shape == expected, (
        f"Expected {expected}, got {env.action_space.shape}"
    )


# ─── Reset tests ──────────────────────────────────────────────────────────────

def test_reset_returns_correct_obs_shape(env):
    """reset() returns obs with shape matching obs_space."""
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape, (
        f"Expected {env.observation_space.shape}, got {obs.shape}"
    )


# ─── Step tests ───────────────────────────────────────────────────────────────

def test_step_returns_correct_types(env):
    """step() returns (ndarray, float, bool, bool, dict)."""
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(obs, np.ndarray), f"obs type: {type(obs)}"
    assert isinstance(reward, float), f"reward type: {type(reward)}"
    assert isinstance(terminated, bool), f"terminated type: {type(terminated)}"
    assert isinstance(truncated, bool), f"truncated type: {type(truncated)}"
    assert isinstance(info, dict), f"info type: {type(info)}"


def test_step_obs_shape(env):
    """Obs from step() matches obs_space.shape."""
    env.reset()
    action = env.action_space.sample()
    obs, _, _, _, _ = env.step(action)
    assert obs.shape == env.observation_space.shape, (
        f"Expected {env.observation_space.shape}, got {obs.shape}"
    )


def test_max_steps_truncates(env):
    """After episode_max_steps steps, truncated=True."""
    env.reset()
    max_steps = ENV_CONFIG["episode_max_steps"]
    truncated = False
    terminated = False
    # Take exactly max_steps steps (termination disabled in ENV_CONFIG safety settings)
    for _ in range(max_steps):
        if terminated or truncated:
            break
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
    assert truncated or terminated, (
        f"Expected truncated=True (or terminated) after {max_steps} steps, "
        f"got truncated={truncated}, terminated={terminated}"
    )
    # Specifically: if not terminated early, must be truncated
    if not terminated:
        assert truncated, "Expected truncated=True after max_steps (no early termination)"


def test_action_scaling_applied(env):
    """Stepping with all-minus-one action vs all-plus-one gives different state."""
    env.reset(seed=42)
    action_low = np.full(N_ACT, -1.0)
    obs_low, _, _, _, _ = env.step(action_low)

    env.reset(seed=42)
    action_high = np.full(N_ACT, 1.0)
    obs_high, _, _, _, _ = env.step(action_high)

    assert not np.allclose(obs_low, obs_high), (
        "action=-1 and action=+1 produced identical observations — scaling not applied"
    )
