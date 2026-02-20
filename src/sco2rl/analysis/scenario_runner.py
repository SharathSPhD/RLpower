"""ScenarioRunner — orchestrates control analysis across phases and controllers.

Runs a battery of step-response and frequency-response experiments for both
PID and RL controllers across all curriculum phases.  Results are serialised
to JSON so notebooks can load them without requiring a live FMU.

Typical usage::

    from sco2rl.analysis.scenario_runner import ScenarioRunner, build_mock_env
    from sco2rl.control import MultiLoopPID

    env_cfg = _load_env_config()
    pid = MultiLoopPID(config=pid_config)

    runner = ScenarioRunner()
    results = runner.run_all(
        env_factory=lambda: build_mock_env(env_cfg),
        pid_policy=pid,
        rl_policy=None,   # skip RL if no checkpoint available
        phases=[0, 1, 2, 3],
    )
    runner.save(results, "data/control_analysis_all_phases.json")
"""
from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sco2rl.analysis.metrics import ControlMetricsSummary, StepResponseResult, FrequencyResponseResult
from sco2rl.analysis.step_response import run_step_scenario
from sco2rl.analysis.frequency_analysis import estimate_frequency_response

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore


# ---------------------------------------------------------------------------
# Env config helpers
# ---------------------------------------------------------------------------

# Short variable names used by MockFMU / DynamicMockFMU
_MOCK_OBS_VARS = [
    "T_turbine_inlet", "T_compressor_inlet", "P_high", "P_low",
    "mdot_turbine", "W_turbine", "W_main_compressor", "W_net",
    "eta_thermal", "surge_margin_main",
]
_MOCK_ACTION_VARS = [
    "bypass_valve_opening", "igv_angle_normalized",
    "inventory_valve_opening", "cooling_flow_normalized",
]
_MOCK_DESIGN_POINT = {
    "T_turbine_inlet": 750.0,
    "T_compressor_inlet": 33.0,
    "P_high": 20.0,
    "P_low": 7.5,
    "mdot_turbine": 95.0,
    "W_turbine": 12.0,
    "W_main_compressor": 2.0,
    "W_net": 10.0,
    "eta_thermal": 0.40,
    "surge_margin_main": 0.15,
}

_MOCK_ENV_CONFIG: dict[str, Any] = {
    "obs_vars": _MOCK_OBS_VARS,
    "obs_bounds": {
        "T_turbine_inlet": (500.0, 950.0),
        "T_compressor_inlet": (31.0, 43.0),
        "P_high": (14.0, 24.0),
        "P_low": (6.5, 10.0),
        "mdot_turbine": (50.0, 150.0),
        "W_turbine": (0.0, 20.0),
        "W_main_compressor": (0.0, 8.0),
        "W_net": (-5.0, 20.0),
        "eta_thermal": (0.2, 0.6),
        "surge_margin_main": (0.0, 0.5),
    },
    "action_vars": _MOCK_ACTION_VARS,
    "action_config": {
        "bypass_valve_opening": {"min": -1.0, "max": 1.0, "rate": 0.1},
        "igv_angle_normalized": {"min": -1.0, "max": 1.0, "rate": 0.1},
        "inventory_valve_opening": {"min": -1.0, "max": 1.0, "rate": 0.05},
        "cooling_flow_normalized": {"min": -1.0, "max": 1.0, "rate": 0.1},
    },
    "history_steps": 5,
    "step_size": 5.0,
    "episode_max_steps": 300,
    "reward": {
        "w_tracking": 1.0,
        "w_efficiency": 0.3,
        "w_smoothness": 0.1,
        "rated_power_mw": 10.0,
        "design_efficiency": 0.40,
        "terminal_failure_reward": -100.0,
        "w_net_unit_scale": 1.0,
    },
    "safety": {
        "T_compressor_inlet_min": 32.2,
        "surge_margin_min": 0.05,
    },
    "setpoint": {"W_net": 10.0},
}

# PID config for MultiLoopPID with MockFMU obs/action variable names
_MOCK_PID_CONFIG: dict[str, Any] = {
    "obs_vars": _MOCK_OBS_VARS,
    "action_vars": _MOCK_ACTION_VARS,
    "n_obs": len(_MOCK_OBS_VARS),
    "history_steps": 5,
    "dt": 5.0,
    "gains": {
        "bypass_valve_opening": {
            "kp": 0.25, "ki": 0.010, "kd": 0.50,
            "anti_windup_gain": 0.10, "derivative_filter_tau": 10.0,
        },
        "igv_angle_normalized": {
            "kp": 0.010, "ki": 0.0002, "kd": 0.05,
            "anti_windup_gain": 0.10, "derivative_filter_tau": 15.0,
        },
        "inventory_valve_opening": {
            "kp": 0.30, "ki": 0.010, "kd": 0.50,
            "anti_windup_gain": 0.10, "derivative_filter_tau": 10.0,
        },
        "cooling_flow_normalized": {
            "kp": 0.20, "ki": 0.008, "kd": 0.40,
            "anti_windup_gain": 0.10, "derivative_filter_tau": 10.0,
        },
    },
    "setpoints": {
        "W_net": 10.0,
        "T_turbine_inlet": 750.0,
        "P_high": 20.0,
        "T_compressor_inlet": 33.0,
    },
    "measurement_indices": {
        "bypass_valve_opening": 7,     # W_net (index 7 in _MOCK_OBS_VARS)
        "igv_angle_normalized": 0,     # T_turbine_inlet
        "inventory_valve_opening": 2,  # P_high
        "cooling_flow_normalized": 1,  # T_compressor_inlet
    },
}


def build_mock_env(dynamic: bool = False, seed: int = 42) -> Any:
    """Build a SCO2FMUEnv backed by MockFMU (or DynamicMockFMU).

    Parameters
    ----------
    dynamic:
        If True, use DynamicMockFMU (adds lag dynamics for frequency response).
    seed:
        MockFMU RNG seed.
    """
    from sco2rl.environment.sco2_env import SCO2FMUEnv

    if dynamic:
        from sco2rl.analysis._dynamic_mock import DynamicMockFMU
        fmu = DynamicMockFMU(
            obs_vars=_MOCK_OBS_VARS,
            action_vars=_MOCK_ACTION_VARS,
            design_point=_MOCK_DESIGN_POINT,
            seed=seed,
        )
    else:
        from sco2rl.simulation.fmu.mock_fmu import MockFMU
        fmu = MockFMU(
            obs_vars=_MOCK_OBS_VARS,
            action_vars=_MOCK_ACTION_VARS,
            design_point=_MOCK_DESIGN_POINT,
            seed=seed,
        )

    return SCO2FMUEnv(fmu=fmu, config=_MOCK_ENV_CONFIG)


def build_mock_pid() -> Any:
    """Build a MultiLoopPID with MockFMU obs/action variable names."""
    from sco2rl.control.multi_loop_pid import MultiLoopPID
    return MultiLoopPID(config=_MOCK_PID_CONFIG)


# ---------------------------------------------------------------------------
# MLP surrogate env
# ---------------------------------------------------------------------------

_MLP_OBS_VARS = [
    "T_compressor_inlet", "P_high", "T_turbine_inlet", "T_hot_in", "T_hot_out",
    "P_low", "T_regen", "W_net", "T_comp_out", "W_turbine", "W_main_compressor",
    "eta_thermal", "p_outlet", "Q_in", "eta_2",
]
_MLP_STATE_BOUNDS = [
    (31.5, 42.0), (70.0, 120.0), (300.0, 1100.0), (200.0, 1100.0), (200.0, 1100.0),
    (70.0, 120.0), (100.0, 175.0), (31.5, 42.0), (5.0, 25.0), (0.5, 8.0),
    (0.85, 0.92), (0.90, 0.95), (5.0, 120.0), (15.0, 21.0),
]
_MLP_ACTION_BOUNDS = [(0.0, 1.0)] * 4
_MLP_ACTION_VARS = [
    "bypass_valve_opening", "igv_angle_normalized",
    "inventory_valve_opening", "cooling_flow_normalized",
]
_MLP_PID_CONFIG: dict[str, Any] = {
    "obs_vars": _MLP_OBS_VARS,
    "action_vars": _MLP_ACTION_VARS,
    "n_obs": len(_MLP_OBS_VARS),
    "history_steps": 1,
    "dt": 5.0,
    "gains": {
        "bypass_valve_opening": {"kp": 0.25, "ki": 0.010, "kd": 0.50,
            "anti_windup_gain": 0.10, "derivative_filter_tau": 10.0},
        "igv_angle_normalized": {"kp": 0.010, "ki": 0.0002, "kd": 0.05,
            "anti_windup_gain": 0.10, "derivative_filter_tau": 15.0},
        "inventory_valve_opening": {"kp": 0.30, "ki": 0.010, "kd": 0.50,
            "anti_windup_gain": 0.10, "derivative_filter_tau": 10.0},
        "cooling_flow_normalized": {"kp": 0.20, "ki": 0.008, "kd": 0.40,
            "anti_windup_gain": 0.10, "derivative_filter_tau": 10.0},
    },
    "setpoints": {"W_net": 10.0, "T_turbine_inlet": 750.0, "P_high": 95.0, "T_compressor_inlet": 36.75},
    "measurement_indices": {
        "bypass_valve_opening": 7, "igv_angle_normalized": 2,
        "inventory_valve_opening": 1, "cooling_flow_normalized": 0,
    },
}


class MLPStepEnv:
    """Gymnasium-compatible env wrapping the MLP step predictor."""

    def __init__(self, model: Any, norm: dict[str, np.ndarray], seed: int = 42) -> None:
        import torch
        self._model = model
        self._norm = norm
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        n_s = int(norm["s_mean"].shape[0])
        n_a = int(norm["a_mean"].shape[0])
        self._n_s = n_s
        self._n_a = n_a
        self._s_mean = np.array(norm["s_mean"], dtype=np.float32)
        self._s_std = np.array(norm["s_std"], dtype=np.float32)
        self._a_mean = np.array(norm["a_mean"], dtype=np.float32)
        self._a_std = np.array(norm["a_std"], dtype=np.float32)
        self._sp_mean = np.array(norm["sp_mean"], dtype=np.float32)
        self._sp_std = np.array(norm["sp_std"], dtype=np.float32)
        lo = np.array([b[0] for b in _MLP_STATE_BOUNDS[:n_s]], dtype=np.float32)
        hi = np.array([b[1] for b in _MLP_STATE_BOUNDS[:n_s]], dtype=np.float32)
        self._design_pt = 0.5 * (lo + hi)
        self._state = np.zeros(n_s, dtype=np.float32)
        self._prev_action = np.zeros(n_a, dtype=np.float32)
        self._episode_max_steps = 200
        self._step_count = 0
        self._obs_vars = _MLP_OBS_VARS
        self._history_steps = 1
        self._base_setpoint = {"W_net": 10.0}
        self._setpoint = {"W_net": 10.0}
        self._curriculum_phase = 0
        if gym is not None:
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(n_a,), dtype=np.float32
            )
        else:
            self.observation_space = None
            self.action_space = None

    def _state_to_obs(self, state: np.ndarray) -> np.ndarray:
        w_net = float(state[8] - state[9])
        return np.concatenate([state[:7], np.array([w_net], dtype=np.float32), state[7:14]]).astype(np.float32)

    def set_curriculum_phase(self, phase: int, episode_max_steps: int = 200) -> None:
        self._curriculum_phase = phase
        self._episode_max_steps = episode_max_steps

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = np.clip(
            self._design_pt + self._rng.standard_normal(self._n_s).astype(np.float32) * 0.01,
            [b[0] for b in _MLP_STATE_BOUNDS[:self._n_s]],
            [b[1] for b in _MLP_STATE_BOUNDS[:self._n_s]],
        )
        self._prev_action[:] = 0.0
        self._step_count = 0
        return self._state_to_obs(self._state), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        import torch
        action = np.clip(np.asarray(action, dtype=np.float32).flatten(), -1.0, 1.0)
        a_phys = np.array([0.5 * (lo + hi) for (lo, hi) in _MLP_ACTION_BOUNDS], dtype=np.float32)
        a_phys = a_phys + 0.5 * (action + 1.0) * np.array([hi - lo for (lo, hi) in _MLP_ACTION_BOUNDS], dtype=np.float32)
        a_phys = np.clip(a_phys, 0.0, 1.0)
        s_n = (self._state - self._s_mean) / self._s_std
        a_n = (a_phys - self._a_mean) / self._a_std
        with torch.no_grad():
            s_t = torch.from_numpy(s_n).float().unsqueeze(0)
            a_t = torch.from_numpy(a_n).float().unsqueeze(0)
            sp_n = self._model(s_t, a_t).squeeze(0).numpy()
        next_state = np.clip(sp_n * self._sp_std + self._sp_mean,
            [b[0] for b in _MLP_STATE_BOUNDS[:self._n_s]],
            [b[1] for b in _MLP_STATE_BOUNDS[:self._n_s]])
        self._state = next_state
        self._prev_action[:] = action
        self._step_count += 1
        terminated = float(next_state[0]) < 32.2
        truncated = self._step_count >= self._episode_max_steps
        W_net = float(next_state[8] - next_state[9])
        target = self._setpoint.get("W_net", 10.0)
        reward = -abs(W_net - target) / max(target, 1.0) - (100.0 if terminated else 0.0)
        return self._state_to_obs(next_state), float(reward), bool(terminated), bool(truncated), {}

    def close(self) -> None:
        pass


def build_mlp_env(
    mlp_model_path: str | Path = "artifacts/surrogate/mlp_step.pt",
    norm_path: str | Path = "artifacts/surrogate/mlp_step_norm.npz",
    seed: int = 42,
) -> MLPStepEnv:
    """Build MLPStepEnv with lazily loaded MLP model."""
    import torch
    path = Path(mlp_model_path)
    norm_path = Path(norm_path)
    proj = Path(__file__).resolve().parent.parent.parent.parent
    if not path.is_absolute():
        path = proj / path
    if not norm_path.is_absolute():
        norm_path = proj / norm_path
    if not path.exists():
        raise FileNotFoundError(f"MLP model not found: {path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"Norm file not found: {norm_path}")
    norm = dict(np.load(norm_path))
    n_s, n_a = int(norm["s_mean"].shape[0]), int(norm["a_mean"].shape[0])

    class _MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layers = [torch.nn.Linear(n_s + n_a, 512), torch.nn.SiLU()]
            for _ in range(3):
                layers += [torch.nn.Linear(512, 512), torch.nn.SiLU()]
            layers.append(torch.nn.Linear(512, n_s))
            self.net = torch.nn.Sequential(*layers)
        def forward(self, s, a):
            return s + self.net(torch.cat([s, a], dim=-1))

    model = _MLP()
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    model.eval()
    return MLPStepEnv(model=model, norm=norm, seed=seed)


def build_mlp_pid() -> Any:
    """Build MultiLoopPID with MLP obs/action variable names."""
    from sco2rl.control.multi_loop_pid import MultiLoopPID
    return MultiLoopPID(config=_MLP_PID_CONFIG)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


class ControlScenario(str, Enum):
    STEP_LOAD_UP_20 = "step_load_+20pct"
    STEP_LOAD_DOWN_20 = "step_load_-20pct"
    STEP_LOAD_UP_30 = "step_load_+30pct"
    LOAD_REJECTION_50 = "load_rejection_-50pct"
    FREQUENCY_RESPONSE = "frequency_response"


_RATED_POWER_MW = 10.0

_STEP_SCENARIOS: dict[ControlScenario, dict[str, Any]] = {
    ControlScenario.STEP_LOAD_UP_20: {
        "step_magnitude": +2.0,
        "step_at_step": 50,
        "n_steps": 250,
    },
    ControlScenario.STEP_LOAD_DOWN_20: {
        "step_magnitude": -2.0,
        "step_at_step": 50,
        "n_steps": 250,
    },
    ControlScenario.STEP_LOAD_UP_30: {
        "step_magnitude": +3.0,
        "step_at_step": 50,
        "n_steps": 250,
    },
    ControlScenario.LOAD_REJECTION_50: {
        "step_magnitude": -5.0,
        "step_at_step": 50,
        "n_steps": 250,
    },
}


# ---------------------------------------------------------------------------
# ScenarioRunner
# ---------------------------------------------------------------------------


class ScenarioRunner:
    """Orchestrates control analysis across phases and controllers.

    Results are averaged over ``n_seeds`` episodes to reduce noise from
    stochastic disturbance profiles.
    """

    def __init__(self, n_seeds: int = 3, dt: float = 5.0, verbose: bool = True) -> None:
        self._n_seeds = n_seeds
        self._dt = dt
        self._verbose = verbose

    def run_step(
        self,
        env_factory: Callable[[], Any],
        policy: Any,
        scenario: ControlScenario,
        phase: int,
    ) -> StepResponseResult:
        """Run a step scenario averaged over multiple seeds."""
        cfg = _STEP_SCENARIOS[scenario]
        results: list[StepResponseResult] = []

        for seed in range(self._n_seeds):
            env = env_factory()
            try:
                res = run_step_scenario(
                    env=env,
                    policy=policy,
                    step_magnitude=cfg["step_magnitude"],
                    step_at_step=cfg["step_at_step"],
                    n_steps=cfg["n_steps"],
                    dt=self._dt,
                    variable="W_net",
                    phase=phase,
                    scenario=scenario.value,
                    seed=seed,
                )
                results.append(res)
            finally:
                env.close()

        return _average_step_results(results)

    def run_frequency(
        self,
        env_factory: Callable[[], Any],
        policy: Any,
        phase: int,
        channel_idx: int = 0,
    ) -> FrequencyResponseResult:
        """Run PRBS frequency response estimation."""
        env = env_factory()
        try:
            result = estimate_frequency_response(
                env=env,
                policy=policy,
                channel_idx=channel_idx,
                output_variable="W_net",
                prbs_amplitude=0.05,
                n_bits=9,
                n_periods=4,
                dt=self._dt,
                warmup_steps=80,
                phase=phase,
            )
        finally:
            env.close()
        return result

    def run_all(
        self,
        env_factory: Callable[[], Any],
        pid_policy: Any | None,
        rl_policy: Any | None,
        phases: list[int] | None = None,
        scenarios: list[ControlScenario] | None = None,
        run_frequency: bool = True,
        freq_env_factory: Callable[[], Any] | None = None,
    ) -> list[ControlMetricsSummary]:
        """Run the full analysis battery.

        Parameters
        ----------
        env_factory:
            Callable returning a fresh SCO2FMUEnv (called for each run).
            Use ``build_mock_env()`` for MockFMU-based analysis.
        pid_policy:
            PID controller (or None to skip PID).
        rl_policy:
            RL controller (or None to skip RL).
        phases:
            List of curriculum phases to evaluate (default: 0–6).
        scenarios:
            List of ControlScenario to evaluate (default: step +20%, −20%).
        run_frequency:
            If True, also run PRBS frequency response (requires DynamicMockFMU
            or real FMU for meaningful results).

        Returns
        -------
        list[ControlMetricsSummary]
        """
        if phases is None:
            phases = list(range(7))
        if scenarios is None:
            scenarios = [
                ControlScenario.STEP_LOAD_UP_20,
                ControlScenario.STEP_LOAD_DOWN_20,
                ControlScenario.LOAD_REJECTION_50,
            ]

        summaries: list[ControlMetricsSummary] = []

        for phase in phases:
            for scenario in scenarios:
                if self._verbose:
                    print(f"  Phase {phase} | {scenario.value} ...", flush=True)

                summary = ControlMetricsSummary(phase=phase, scenario=scenario.value)

                if pid_policy is not None:
                    summary.pid_step = self.run_step(env_factory, pid_policy, scenario, phase)

                if rl_policy is not None:
                    summary.rl_step = self.run_step(env_factory, rl_policy, scenario, phase)

                summaries.append(summary)

        if run_frequency:
            freq_factory = freq_env_factory if freq_env_factory is not None else (
                lambda: build_mock_env(dynamic=True)
            )
            for phase in phases:
                if self._verbose:
                    print(f"  Phase {phase} | frequency response ...", flush=True)

                freq_summary = ControlMetricsSummary(
                    phase=phase, scenario=ControlScenario.FREQUENCY_RESPONSE.value
                )

                if pid_policy is not None:
                    freq_summary.pid_freq = self.run_frequency(freq_factory, pid_policy, phase)

                if rl_policy is not None:
                    freq_summary.rl_freq = self.run_frequency(freq_factory, rl_policy, phase)

                summaries.append(freq_summary)

        return summaries

    @staticmethod
    def save(
        results: list[ControlMetricsSummary],
        output_path: str | Path,
        env_type: str = "MockFMU",
    ) -> None:
        """Serialise results to JSON.

        Parameters
        ----------
        results:
            List of ControlMetricsSummary from ``run_all()``.
        output_path:
            Destination path (parent directories are created if needed).
        env_type:
            Environment type for metadata (e.g. "MockFMU", "MLP").
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        serialisable = {
            "version": "1.0",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generated_by": "sco2rl.analysis.scenario_runner.ScenarioRunner",
            "env": env_type,
            "results": [dataclasses.asdict(r) for r in results],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2)

        print(f"Saved {len(results)} results to {path}")

    @staticmethod
    def load(input_path: str | Path) -> list[ControlMetricsSummary]:
        """Load results from a previously saved JSON file."""
        path = Path(input_path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        summaries = []
        for raw in data.get("results", []):
            def _make_step(d: dict | None) -> StepResponseResult | None:
                if d is None:
                    return None
                return StepResponseResult(**d)

            def _make_freq(d: dict | None) -> FrequencyResponseResult | None:
                if d is None:
                    return None
                return FrequencyResponseResult(**d)

            summaries.append(ControlMetricsSummary(
                phase=raw["phase"],
                scenario=raw["scenario"],
                pid_step=_make_step(raw.get("pid_step")),
                rl_step=_make_step(raw.get("rl_step")),
                pid_freq=_make_freq(raw.get("pid_freq")),
                rl_freq=_make_freq(raw.get("rl_freq")),
            ))
        return summaries


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _average_step_results(results: list[StepResponseResult]) -> StepResponseResult:
    """Average numeric metrics over multiple seeds; keep first time series."""
    if not results:
        raise ValueError("Cannot average empty results list")
    if len(results) == 1:
        return results[0]

    base = results[0]
    fields_to_avg = [
        "overshoot_pct", "undershoot_pct", "settling_time_s",
        "rise_time_s", "peak_time_s", "steady_state_error",
        "iae", "ise", "itae", "final_value",
    ]

    averaged = dataclasses.replace(base)
    for field_name in fields_to_avg:
        vals = [getattr(r, field_name) for r in results]
        setattr(averaged, field_name, float(np.mean(vals)))

    averaged.seed = -1  # Indicates averaged result
    return averaged
