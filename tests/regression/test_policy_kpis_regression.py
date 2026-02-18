from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sco2rl.environment.sco2_env import SCO2FMUEnv
from sco2rl.simulation.fmu.mock_fmu import MockFMU
from sco2rl.training.policy_evaluator import PolicyEvaluator


GOLDEN_PATH = Path(__file__).parent / "golden" / "policy_kpis_phase0.json"


class _ZeroPolicy:
    def __init__(self, action_dim: int) -> None:
        self._action_dim = action_dim

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        del obs, deterministic
        return np.zeros(self._action_dim, dtype=np.float32), None


def _build_env() -> SCO2FMUEnv:
    obs_vars = [
        "T_compressor_inlet",
        "W_turbine",
        "W_main_compressor",
        "W_net",
        "eta_thermal",
        "surge_margin_main",
    ]
    action_vars = [
        "bypass_valve_opening",
        "igv_angle_normalized",
        "inventory_valve_opening",
        "cooling_flow_normalized",
    ]
    fmu = MockFMU(
        obs_vars=obs_vars,
        action_vars=action_vars,
        design_point={
            "T_compressor_inlet": 33.0,
            "W_turbine": 14.5,
            "W_main_compressor": 4.0,
            "W_net": 10.0,
            "eta_thermal": 0.47,
            "surge_margin_main": 0.2,
        },
        seed=42,
    )
    config = {
        "obs_vars": obs_vars,
        "obs_bounds": {name: (0.0, 1500.0) for name in obs_vars},
        "action_vars": action_vars,
        "action_config": {name: {"min": 0.0, "max": 1.0, "rate": 0.05} for name in action_vars},
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
        "safety": {"T_compressor_inlet_min": 32.2, "surge_margin_min": 0.05},
        "setpoint": {"W_net": 10.0},
    }
    return SCO2FMUEnv(fmu=fmu, config=config)


def test_phase0_kpis_match_golden_baseline() -> None:
    golden = json.loads(GOLDEN_PATH.read_text())
    env = _build_env()
    try:
        evaluator = PolicyEvaluator(
            env,
            {"n_eval_episodes": int(golden["n_eval_episodes"]), "deterministic": True},
        )
        metrics = evaluator.evaluate(_ZeroPolicy(action_dim=4), phase=0)
    finally:
        env.close()

    expected = golden["expected"]
    assert metrics.mean_reward == pytest.approx(expected["mean_reward"], rel=1e-8, abs=1e-8)
    assert metrics.std_reward == pytest.approx(expected["std_reward"], rel=1e-8, abs=1e-8)
    assert metrics.mean_episode_length == pytest.approx(
        expected["mean_episode_length"], rel=1e-8, abs=1e-8
    )
    assert metrics.violation_rate == pytest.approx(expected["violation_rate"], rel=1e-8, abs=1e-8)
