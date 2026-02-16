"""Unit tests for SCO2FMUEnv — Gymnasium environment wrapping FMUInterface.

TDD RED: written BEFORE implementation.
All tests MUST fail with ImportError/AttributeError until GREEN phase.

Design assumptions tested here:
- SCO2FMUEnv accepts a FMUInterface + config dict via constructor
- obs space: Box(low, high, shape=(n_obs_vars * history_steps,), float32)
- action space: Box(-1, 1, shape=(n_actions,), float32)  [normalized]
- reset() → (obs: np.ndarray, info: dict)
- step(action) → (obs, reward, terminated, truncated, info)
- ActionScaler maps [-1, 1] → [physical_min, physical_max]
- RateLimiter enforces max change per step
- Episode terminates on: (a) FMU solver failure, (b) T_compressor_inlet < 32.2°C
- gymnasium.utils.env_checker.check_env() must pass (Gymnasium compliance)
"""
from __future__ import annotations

import numpy as np
import pytest

# ── Shared test constants ──────────────────────────────────────────────────────

OBS_VARS = [
    "T_turbine_inlet",
    "T_compressor_inlet",
    "P_high",
    "P_low",
    "mdot_turbine",
    "W_turbine",
    "W_main_compressor",
    "W_net",
    "eta_thermal",
    "surge_margin_main",
    "T_exhaust_source",
    "mdot_exhaust_source",
]

ACTION_VARS = [
    "bypass_valve_opening",
    "igv_angle_normalized",
    "inventory_valve_opening",
    "cooling_flow_normalized",
]

DESIGN_POINT = {
    "T_turbine_inlet": 750.0,
    "T_compressor_inlet": 33.0,
    "P_high": 20.0,
    "P_low": 7.7,
    "mdot_turbine": 130.0,
    "W_turbine": 14.5,
    "W_main_compressor": 4.0,
    "W_net": 10.0,
    "eta_thermal": 0.47,
    "surge_margin_main": 0.20,
    "T_exhaust_source": 800.0,
    "mdot_exhaust_source": 50.0,
}

# Observation bounds for the 12 variables (min, max)
OBS_BOUNDS = {
    "T_turbine_inlet":      (600.0, 850.0),
    "T_compressor_inlet":   (30.0, 45.0),
    "P_high":               (14.0, 26.0),
    "P_low":                (6.5, 9.5),
    "mdot_turbine":         (40.0, 220.0),
    "W_turbine":            (0.0, 25.0),
    "W_main_compressor":    (0.0, 15.0),
    "W_net":                (0.0, 15.0),
    "eta_thermal":          (0.0, 0.60),
    "surge_margin_main":    (0.0, 0.60),
    "T_exhaust_source":     (150.0, 1300.0),
    "mdot_exhaust_source":  (5.0, 120.0),
}

# Action physical bounds: physical_min, physical_max, rate_limit_per_step
ACTION_CONFIG = {
    "bypass_valve_opening":    {"min": 0.0, "max": 1.0, "rate": 0.05},
    "igv_angle_normalized":    {"min": 0.0, "max": 1.0, "rate": 0.05},
    "inventory_valve_opening": {"min": 0.0, "max": 1.0, "rate": 0.02},
    "cooling_flow_normalized": {"min": 0.0, "max": 1.0, "rate": 0.05},
}

HISTORY_STEPS = 3  # smaller than production (5) for faster unit tests

ENV_CONFIG = {
    "obs_vars": OBS_VARS,
    "obs_bounds": OBS_BOUNDS,
    "action_vars": ACTION_VARS,
    "action_config": ACTION_CONFIG,
    "history_steps": HISTORY_STEPS,
    "step_size": 5.0,
    "episode_max_steps": 50,
    "reward": {
        "w_tracking": 1.0,
        "w_efficiency": 0.3,
        "w_smoothness": 0.1,
        "rated_power_mw": 10.0,
        "design_efficiency": 0.47,
        "terminal_failure_reward": -100.0,
    },
    "safety": {
        "T_compressor_inlet_min": 32.2,
        "surge_margin_min": 0.05,
    },
    "setpoint": {
        "W_net": 10.0,
    },
}


def _make_mock_fmu(**kwargs):
    from sco2rl.simulation.fmu.mock_fmu import MockFMU
    return MockFMU(
        obs_vars=OBS_VARS,
        action_vars=ACTION_VARS,
        design_point=DESIGN_POINT,
        seed=42,
        **kwargs,
    )


def _make_env(fmu=None, **config_overrides):
    from sco2rl.environment.sco2_env import SCO2FMUEnv
    if fmu is None:
        fmu = _make_mock_fmu()
    cfg = dict(ENV_CONFIG)
    cfg.update(config_overrides)
    return SCO2FMUEnv(fmu=fmu, config=cfg)


# ── Class + import contract ────────────────────────────────────────────────────

class TestSCO2FMUEnvImport:
    def test_module_importable(self):
        from sco2rl.environment import sco2_env  # noqa: F401

    def test_class_importable(self):
        from sco2rl.environment.sco2_env import SCO2FMUEnv  # noqa: F401

    def test_is_gymnasium_env(self):
        import gymnasium as gym
        from sco2rl.environment.sco2_env import SCO2FMUEnv
        assert issubclass(SCO2FMUEnv, gym.Env)


# ── Spaces ─────────────────────────────────────────────────────────────────────

class TestSpaces:
    def test_observation_space_shape(self):
        env = _make_env()
        expected_dim = len(OBS_VARS) * HISTORY_STEPS
        assert env.observation_space.shape == (expected_dim,)

    def test_observation_space_dtype_float32(self):
        env = _make_env()
        assert env.observation_space.dtype == np.float32

    def test_observation_space_is_box(self):
        import gymnasium as gym
        env = _make_env()
        assert isinstance(env.observation_space, gym.spaces.Box)

    def test_action_space_shape(self):
        env = _make_env()
        assert env.action_space.shape == (len(ACTION_VARS),)

    def test_action_space_dtype_float32(self):
        env = _make_env()
        assert env.action_space.dtype == np.float32

    def test_action_space_bounds_normalized(self):
        """Action space must be [-1, 1] for all dimensions (normalized)."""
        env = _make_env()
        np.testing.assert_array_equal(env.action_space.low, -np.ones(len(ACTION_VARS)))
        np.testing.assert_array_equal(env.action_space.high, np.ones(len(ACTION_VARS)))


# ── Reset ──────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_returns_tuple(self):
        env = _make_env()
        result = env.reset()
        assert isinstance(result, tuple) and len(result) == 2

    def test_reset_obs_is_ndarray_float32(self):
        env = _make_env()
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32

    def test_reset_obs_shape(self):
        env = _make_env()
        obs, _ = env.reset()
        expected_dim = len(OBS_VARS) * HISTORY_STEPS
        assert obs.shape == (expected_dim,)

    def test_reset_obs_in_observation_space(self):
        env = _make_env()
        obs, _ = env.reset()
        assert env.observation_space.contains(obs), (
            f"reset() obs not in observation_space: "
            f"low={env.observation_space.low[:4]}, "
            f"high={env.observation_space.high[:4]}, "
            f"obs={obs[:4]}"
        )

    def test_reset_info_is_dict(self):
        env = _make_env()
        _, info = env.reset()
        assert isinstance(info, dict)

    def test_reset_is_deterministic_with_seed(self):
        env = _make_env()
        obs_a, _ = env.reset(seed=0)
        obs_b, _ = env.reset(seed=0)
        np.testing.assert_array_equal(obs_a, obs_b)

    def test_reset_clears_step_counter(self):
        env = _make_env()
        env.reset()
        env.step(env.action_space.sample())
        env.reset()
        assert env._step_count == 0

    def test_reset_history_buffer_filled_with_design_point(self):
        """After reset, all history steps should reflect the design-point obs."""
        env = _make_env()
        obs, _ = env.reset()
        # obs is shape (n_obs * history_steps,) in order [t-H+1 ... t]
        # All history steps filled with same values → first and last block equal
        n = len(OBS_VARS)
        first_block = obs[:n]
        last_block = obs[-n:]
        np.testing.assert_array_almost_equal(first_block, last_block, decimal=4)


# ── Step ───────────────────────────────────────────────────────────────────────

class TestStep:
    def test_step_returns_5_tuple(self):
        env = _make_env()
        env.reset()
        result = env.step(np.zeros(len(ACTION_VARS), dtype=np.float32))
        assert len(result) == 5

    def test_step_obs_shape(self):
        env = _make_env()
        env.reset()
        obs, _, _, _, _ = env.step(np.zeros(len(ACTION_VARS), dtype=np.float32))
        assert obs.shape == (len(OBS_VARS) * HISTORY_STEPS,)

    def test_step_obs_dtype_float32(self):
        env = _make_env()
        env.reset()
        obs, _, _, _, _ = env.step(np.zeros(len(ACTION_VARS), dtype=np.float32))
        assert obs.dtype == np.float32

    def test_step_reward_is_float(self):
        env = _make_env()
        env.reset()
        _, reward, _, _, _ = env.step(np.zeros(len(ACTION_VARS), dtype=np.float32))
        assert isinstance(reward, float)

    def test_step_terminated_and_truncated_are_bool(self):
        env = _make_env()
        env.reset()
        _, _, terminated, truncated, _ = env.step(
            np.zeros(len(ACTION_VARS), dtype=np.float32)
        )
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_info_is_dict(self):
        env = _make_env()
        env.reset()
        _, _, _, _, info = env.step(np.zeros(len(ACTION_VARS), dtype=np.float32))
        assert isinstance(info, dict)

    def test_step_increments_step_count(self):
        env = _make_env()
        env.reset()
        assert env._step_count == 0
        env.step(np.zeros(len(ACTION_VARS), dtype=np.float32))
        assert env._step_count == 1

    def test_step_obs_in_observation_space(self):
        env = _make_env()
        env.reset()
        obs, _, _, _, _ = env.step(np.zeros(len(ACTION_VARS), dtype=np.float32))
        assert env.observation_space.contains(obs)

    def test_history_rolls_forward(self):
        """After 2 steps, the observation should differ from after 1 step."""
        env = _make_env()
        env.reset()
        action = np.array([0.5, -0.3, 0.0, 0.2], dtype=np.float32)
        obs1, _, _, _, _ = env.step(action)
        obs2, _, _, _, _ = env.step(action)
        # History shifts → the two observations must differ (new step entered buffer)
        assert not np.allclose(obs1, obs2)


# ── Termination conditions ─────────────────────────────────────────────────────

class TestTermination:
    def test_terminated_on_fmu_solver_failure(self):
        """do_step returning False → terminated=True, reward = terminal_failure_reward."""
        fmu = _make_mock_fmu(fail_at_step=1)
        env = _make_env(fmu=fmu)
        env.reset()
        _, reward, terminated, _, _ = env.step(
            np.zeros(len(ACTION_VARS), dtype=np.float32)
        )
        assert terminated is True
        assert reward == pytest.approx(ENV_CONFIG["reward"]["terminal_failure_reward"])

    def test_terminated_on_compressor_inlet_violation(self):
        """T_compressor_inlet dropping below 32.2°C → terminated=True."""
        fmu = _make_mock_fmu(inlet_temp_drift=True)
        env = _make_env(fmu=fmu)
        env.reset()
        # Run until drift triggers the safety constraint
        terminated = False
        for _ in range(60):
            _, _, terminated, _, info = env.step(
                np.zeros(len(ACTION_VARS), dtype=np.float32)
            )
            if terminated:
                break
        assert terminated is True

    def test_truncated_at_max_steps(self):
        """Episode truncates (not terminates) after episode_max_steps."""
        env = _make_env()
        env.reset()
        # Design-point action (physical=0 = normalized=-1) keeps T_comp at 33.0°C
        design_action = np.full(len(ACTION_VARS), -1.0, dtype=np.float32)
        truncated = False
        for _ in range(ENV_CONFIG["episode_max_steps"] + 5):
            _, _, terminated, truncated, _ = env.step(design_action)
            if terminated or truncated:
                break
        assert truncated is True


# ── Rate limiting ──────────────────────────────────────────────────────────────

class TestRateLimiting:
    def test_rate_limiter_clamps_large_action(self):
        """A max-magnitude action from [-1,1] must be rate-limited on first step."""
        env = _make_env()
        env.reset()
        # Bypass valve: rate 0.05 normalized means 0.05 in [0,1] physical range.
        # Starting from 0.0 (design point assumption), a +1.0 action (physical +1.0)
        # must be clamped to at most 0.05 after the first step.
        big_action = np.ones(len(ACTION_VARS), dtype=np.float32)  # all max
        env.step(big_action)
        # The FMU received clamped inputs — check via env's last_physical_action attr
        last = env.last_physical_action
        # bypass_valve_opening: max change per step = 0.05 → must be ≤ 0.05
        assert last[0] <= 0.05 + 1e-6, f"bypass not rate-limited: {last[0]}"

    def test_rate_limiter_allows_small_action(self):
        """A small action within the rate limit must not be clipped."""
        env = _make_env()
        env.reset()
        # bypass physical range [0,1], rate=0.05, normalized [-1,1] → physical:
        # a normalized=0.05 → physical = (0.05+1)/2 * 1.0 = 0.525
        # But from 0.0, that's +0.525 which exceeds rate. Use a tiny normalized action.
        # normalized 0.01 → physical ≈ (0.01+1)/2 = 0.505 from 0... still big.
        # Actually: from initial position 0.0, any positive action will be rate-limited.
        # Test: action that maps to Δ < rate (e.g., rate is 0.05 in physical units).
        # normalized -0.9 → physical = (-0.9+1)/2 * 1.0 = 0.05 → Δ=0.05 from 0.0
        small_action = np.array([-0.9, -0.9, -0.96, -0.9], dtype=np.float32)
        env.step(small_action)
        last = env.last_physical_action
        # bypass: physical = 0.05, rate = 0.05 → at the limit, should not be clipped
        assert last[0] == pytest.approx(0.05, abs=0.01), f"small action wrongly clipped: {last[0]}"


# ── Reward shape ───────────────────────────────────────────────────────────────

class TestReward:
    def test_reward_at_design_point_is_nonnegative(self):
        """At design point (physical=0 = normalized=-1), reward ≥ -0.5 (already at setpoint)."""
        env = _make_env()
        env.reset()
        # Design point = all physical actions at 0.0 = normalized -1.0 for all
        design_action = np.full(len(ACTION_VARS), -1.0, dtype=np.float32)
        _, reward, _, _, _ = env.step(design_action)
        # With W_net=10.0 matching setpoint=10.0, tracking penalty ≈ 0 → reward ≥ -0.5
        assert reward >= -0.5, f"Unexpected negative reward at design point: {reward}"

    def test_reward_worse_when_far_from_setpoint(self):
        """Forcing W_net far from setpoint should yield lower reward."""
        env_near = _make_env()
        env_far = _make_env()
        env_near.reset()
        env_far.reset()

        # Near setpoint: zero action (W_net ≈ 10.0)
        _, r_near, _, _, _ = env_near.step(
            np.zeros(len(ACTION_VARS), dtype=np.float32)
        )
        # Far from setpoint: open bypass fully (W_net drops significantly)
        _, r_far, _, _, _ = env_far.step(
            np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        )
        assert r_near >= r_far, (
            f"Reward near setpoint ({r_near:.3f}) must be ≥ reward far ({r_far:.3f})"
        )


# ── Gymnasium compliance ───────────────────────────────────────────────────────

class TestGymnasiumCompliance:
    def test_check_env_passes(self):
        """gymnasium.utils.env_checker.check_env() must not raise an Error."""
        import warnings
        from gymnasium.utils.env_checker import check_env
        env = _make_env()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            check_env(env)

    def test_env_has_render_mode_attribute(self):
        env = _make_env()
        assert hasattr(env, "render_mode")

    def test_env_has_spec_or_none(self):
        """spec may be None for unregistered envs — that's acceptable."""
        env = _make_env()
        assert hasattr(env, "spec")  # attribute must exist

    def test_multiple_resets_stable(self):
        """Ten consecutive resets without errors."""
        env = _make_env()
        for _ in range(10):
            obs, info = env.reset()
            assert obs.shape == (len(OBS_VARS) * HISTORY_STEPS,)

    def test_full_episode_loop(self):
        """Run a complete episode (reset → step loop → done) without error."""
        env = _make_env()
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            assert steps <= ENV_CONFIG["episode_max_steps"] + 2
        assert steps > 0
