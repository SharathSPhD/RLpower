"""Unit tests for TorchBatchedSurrogateEnv."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sco2rl.surrogate.batched_env import TorchBatchedSurrogateEnv
from sco2rl.surrogate.fno_model import FNO1d
from sco2rl.surrogate.surrogate_env import SurrogateEnv

OBS_VARS = [
    "T_compressor_inlet",
    "T_turbine_inlet",
    "P_high",
    "P_low",
    "mdot_turbine",
    "W_net",
    "eta_thermal",
    "surge_margin_main",
]
ACTION_VARS = [
    "bypass_valve_opening",
    "igv_angle_normalized",
    "inventory_valve_opening",
    "cooling_flow_normalized",
]

N_OBS = len(OBS_VARS)
N_ACT = len(ACTION_VARS)
HISTORY_STEPS = 1
INPUT_DIM = N_OBS + N_ACT

SMALL_FNO_CONFIG = {
    "modes": 4,
    "width": 16,
    "n_layers": 2,
    "input_dim": INPUT_DIM,
    "output_dim": N_OBS,
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
        "T_comp_inlet_min_c": -999.0,
        "surge_margin_min": -999.0,
    },
    "setpoint": {"W_net_setpoint": 5.0},
    "obs_design_point": {v: 0.5 for v in OBS_VARS},
}


@pytest.fixture
def tiny_model():
    return FNO1d(**SMALL_FNO_CONFIG)


def test_reset_and_step_shapes(tiny_model):
    env = TorchBatchedSurrogateEnv(
        model=tiny_model,
        config=ENV_CONFIG,
        n_envs=4,
        device="cpu",
    )
    obs, info = env.reset(seed=42)
    assert obs.shape == (4, N_OBS * HISTORY_STEPS)
    assert isinstance(info, dict)
    assert info["step"].shape == (4,)

    action = np.zeros((4, N_ACT), dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (4, N_OBS * HISTORY_STEPS)
    assert reward.shape == (4,)
    assert terminated.shape == (4,)
    assert truncated.shape == (4,)
    assert "constraint_violations" in info
    assert set(info["constraint_violations"].keys()) == {
        "T_comp_min",
        "surge_margin_main",
        "surge_margin_recomp",
    }


def test_single_env_parity_with_surrogate_env(tiny_model):
    batched = TorchBatchedSurrogateEnv(
        model=tiny_model,
        config=ENV_CONFIG,
        n_envs=1,
        device="cpu",
    )
    single = SurrogateEnv(model=tiny_model, config=ENV_CONFIG, device="cpu")

    batched_obs, _ = batched.reset(seed=123)
    single_obs, _ = single.reset(seed=123)
    assert np.allclose(batched_obs[0], single_obs, atol=1e-6)

    action = np.array([0.2, -0.5, 0.1, 0.8], dtype=np.float32)
    b_obs, b_reward, b_term, b_trunc, _ = batched.step(action[None, :])
    s_obs, s_reward, s_term, s_trunc, _ = single.step(action)

    assert np.allclose(b_obs[0], s_obs, atol=1e-5)
    assert b_reward[0] == pytest.approx(s_reward, abs=1e-5)
    assert bool(b_term[0]) == bool(s_term)
    assert bool(b_trunc[0]) == bool(s_trunc)


def test_gpu_state_residency_when_cuda_available(tiny_model):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    env = TorchBatchedSurrogateEnv(
        model=tiny_model,
        config=ENV_CONFIG,
        n_envs=2,
        device="cuda",
    )
    env.reset(seed=7)
    assert env.device.type == "cuda"
    assert env._state.device.type == "cuda"
    assert env._history.device.type == "cuda"
