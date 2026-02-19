"""SCO2FMUEnv — Gymnasium environment wrapping FMUInterface.

Architecture:
  action (normalized) → ActionScaler → RateLimiter → FMU.set_inputs()
  FMU.do_step() → get_outputs() → history buffer → flattened obs

Dependency injection: FMUInterface passed at construction time.
All config via constructor dict — no hardcoded paths or global state (RULE-C2, C3).
"""
from __future__ import annotations

from typing import Any

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from sco2rl.simulation.fmu.interface import FMUInterface


class SCO2FMUEnv(gym.Env):
    """Gymnasium environment wrapping any FMUInterface implementation.

    Parameters
    ----------
    fmu:
        Injected FMUInterface (MockFMU for unit tests, FMPyAdapter for training).
    config:
        Dict with keys: obs_vars, obs_bounds, action_vars, action_config,
        history_steps, step_size, episode_max_steps, reward, safety, setpoint.
    """

    metadata = {"render_modes": []}

    def __init__(self, fmu: FMUInterface, config: dict[str, Any]) -> None:
        super().__init__()

        self._fmu = fmu
        self._cfg = config

        self._obs_vars: list[str] = config["obs_vars"]
        self._action_vars: list[str] = config["action_vars"]
        self._action_cfg: dict[str, dict] = config["action_config"]
        self._history_steps: int = config["history_steps"]
        self._step_size: float = float(config["step_size"])
        self._max_steps: int = int(config["episode_max_steps"])

        self._reward_cfg: dict = config["reward"]
        self._safety_cfg: dict = config["safety"]
        self._base_setpoint: dict[str, float] = dict(config.get("setpoint", {}))
        self._setpoint: dict[str, float] = dict(self._base_setpoint)

        n_obs = len(self._obs_vars)
        n_act = len(self._action_vars)
        obs_dim = n_obs * self._history_steps

        # ── Observation space: per-variable bounds tiled over history ──────────
        bounds = config["obs_bounds"]
        obs_low = np.array(
            [bounds[v][0] for v in self._obs_vars] * self._history_steps,
            dtype=np.float32,
        )
        obs_high = np.array(
            [bounds[v][1] for v in self._obs_vars] * self._history_steps,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, shape=(obs_dim,), dtype=np.float32
        )

        # ── Action space: normalized [-1, 1] ──────────────────────────────────
        self.action_space = spaces.Box(
            low=-np.ones(n_act, dtype=np.float32),
            high=np.ones(n_act, dtype=np.float32),
            dtype=np.float32,
        )

        # ── Action scaling & rate-limiting arrays ──────────────────────────────
        self._act_phys_min = np.array(
            [self._action_cfg[v]["min"] for v in self._action_vars], dtype=np.float64
        )
        self._act_phys_max = np.array(
            [self._action_cfg[v]["max"] for v in self._action_vars], dtype=np.float64
        )
        self._act_rate = np.array(
            [self._action_cfg[v]["rate"] for v in self._action_vars], dtype=np.float64
        )

        # ── State ──────────────────────────────────────────────────────────────
        self._history: np.ndarray = np.zeros(
            (self._history_steps, len(self._obs_vars)), dtype=np.float32
        )
        self._current_physical_action: np.ndarray = self._act_phys_min.copy()
        self._previous_physical_action: np.ndarray = self._act_phys_min.copy()
        self._step_count: int = 0
        self._current_time: float = 0.0
        self._episode_constraint_violations: int = 0

        # Curriculum phase (set by CurriculumCallback; persists across resets)
        self._curriculum_phase: int = 0
        self._disturbance_profile: dict[str, Any] = {}

        # Public attribute for test assertions
        self.last_physical_action: np.ndarray = np.zeros(n_act, dtype=np.float64)
        self.render_mode = None
        self.spec = None

    # ── Gymnasium interface ────────────────────────────────────────────────────

    # Mapping from options keys to FMU input variable name fragments.
    # When options contains "T_exhaust_K" the code searches action_vars for a
    # var whose name contains "T_init" (e.g. "regulator.T_init") and sets it.
    _OPTIONS_TO_FMU: dict[str, str] = {
        "T_exhaust_K":      "T_init",       # heat source temperature (K)
        "mdot_exhaust_kgs": "m_flow_init",  # heat source mass flow (kg/s)
    }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment, optionally with LHS-sampled operating conditions.

        Parameters
        ----------
        seed:
            RNG seed forwarded to Gymnasium base class.
        options:
            Dict with any of:
            - ``T_exhaust_K`` (float): heat source temperature in Kelvin.
              Applied to the FMU action variable whose name contains "T_init"
              (i.e. ``regulator.T_init``).
            - ``mdot_exhaust_kgs`` (float): heat source mass flow in kg/s.
              Applied to the FMU action variable whose name contains "m_flow_init".
            - ``W_setpoint_MW`` (float): power demand setpoint in MW.
              Overrides the ``W_net`` entry in ``self._setpoint`` for this episode.
        """
        super().reset(seed=seed)

        self._fmu.reset()
        self._fmu.initialize(
            start_time=0.0,
            stop_time=self._max_steps * self._step_size,
            step_size=self._step_size,
        )

        self._step_count = 0
        self._current_time = 0.0
        self._current_physical_action = self._act_phys_min.copy()
        self._previous_physical_action = self._act_phys_min.copy()
        self._episode_constraint_violations = 0
        self._setpoint = dict(self._base_setpoint)

        # Apply LHS-sampled operating conditions before reading initial obs.
        # This ensures the FMU starts from a diverse operating point for each
        # trajectory rather than always at the default initial condition.
        if options:
            initial_inputs: dict[str, float] = {}
            for opt_key, fmu_fragment in self._OPTIONS_TO_FMU.items():
                if opt_key in options:
                    # Find the matching action variable by name fragment
                    matched = next(
                        (v for v in self._action_vars if fmu_fragment in v),
                        None,
                    )
                    if matched is not None:
                        initial_inputs[matched] = float(options[opt_key])
            if "W_setpoint_MW" in options:
                self._setpoint["W_net"] = float(options["W_setpoint_MW"])
            if initial_inputs:
                self._fmu.set_inputs(initial_inputs)

        self._disturbance_profile = self._build_disturbance_profile()

        # Fill history buffer with the FMU's initial output
        raw_obs = self._get_raw_obs()
        for i in range(self._history_steps):
            self._history[i] = raw_obs

        obs = self._build_obs()
        info = {
            "step": 0,
            "time": 0.0,
            "initial_inputs": initial_inputs if options else {},
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # 1. Scale action from [-1, 1] to physical range
        action_f64 = action.astype(np.float64)
        physical = self._scale_action(action_f64)

        # 2. Apply rate limiting
        previous_physical_action = self._current_physical_action.copy()
        physical = self._apply_rate_limit(physical)
        self._previous_physical_action = previous_physical_action
        self._current_physical_action = physical
        self.last_physical_action = physical.copy()

        # 3. Build FMU inputs and then apply structured curriculum disturbance overrides.
        inputs = {v: float(physical[i]) for i, v in enumerate(self._action_vars)}
        disturbance_applied, disturbance_inputs = self._apply_curriculum_disturbance()
        if disturbance_inputs:
            inputs.update(disturbance_inputs)

        # 4. Send merged inputs to FMU
        self._fmu.set_inputs(inputs)

        # 5. Advance FMU by one step
        success = self._fmu.do_step(
            current_time=self._current_time, step_size=self._step_size
        )
        self._current_time += self._step_size
        self._step_count += 1

        # 6. Get outputs (fall back to last valid obs if FMU is in error state)
        if success:
            raw_obs = self._get_raw_obs()
        else:
            raw_obs = self._history[-1].copy()  # use last valid observation

        # 7. Check termination conditions
        terminated, term_reason = self._check_terminated(success, raw_obs)
        truncated = (not terminated) and (self._step_count >= self._max_steps)
        violation_values = self._compute_constraint_violations(raw_obs)
        step_has_violation = float(any(v > 0.0 for v in violation_values.values()))
        self._episode_constraint_violations += int(step_has_violation > 0.0)

        # 8. Update history (only on non-terminal or on terminal to get last obs)
        self._update_history(raw_obs)
        obs = self._build_obs()

        # 9. Compute reward
        if terminated and not success:
            reward = float(self._reward_cfg["terminal_failure_reward"])
            reward_components = {
                "r_tracking": 0.0,
                "r_efficiency": 0.0,
                "r_smoothness": 0.0,
                "tracking_error_norm_sq": 0.0,
                "w_net": 0.0,
                "w_net_setpoint": float(self._setpoint.get("W_net", self._reward_cfg.get("rated_power_mw", 10.0))),
                "reward_total": reward,
            }
        else:
            reward_components = self._compute_reward_components(raw_obs, physical)
            reward = float(reward_components["reward_total"])
            if terminated:
                reward = float(self._reward_cfg["terminal_failure_reward"])
                reward_components = dict(reward_components)
                reward_components["reward_total"] = reward

        info = {
            "step": self._step_count,
            "time": self._current_time,
            "terminated_reason": term_reason,
            "disturbance_applied": disturbance_applied,
            "disturbance_inputs": disturbance_inputs,
            "curriculum_phase": self._curriculum_phase,
            "constraint_violation_step": step_has_violation,
            "constraint_violations": violation_values,
            "raw_obs": {v: float(raw_obs[i]) for i, v in enumerate(self._obs_vars)},
            "reward_components": reward_components,
        }
        if terminated or truncated:
            info["constraint_violation"] = (
                float(self._episode_constraint_violations) / max(self._step_count, 1)
            )
        return obs, reward, terminated, truncated, info

    def set_curriculum_phase(self, phase: int, episode_max_steps: int | None = None) -> None:
        """Set the active curriculum phase.

        Called by CurriculumCallback to activate phase-appropriate disturbances.
        Persists across episode resets (curriculum outlasts individual episodes).

        Parameters
        ----------
        phase:
            Integer phase index (0 = STEADY_STATE, 3 = EAF_TRANSIENTS, etc.)
        """
        self._curriculum_phase = phase
        if episode_max_steps is not None:
            self._max_steps = int(episode_max_steps)
        # Rebuild disturbance profile for the new phase so that any step taken
        # after set_curriculum_phase (within the same episode) uses the correct profile.
        self._disturbance_profile = self._build_disturbance_profile()

    def render(self) -> None:
        return None

    def close(self) -> None:
        self._fmu.close()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_raw_obs(self) -> np.ndarray:
        """Return current FMU outputs as float32 array in obs_vars order."""
        outputs = self._fmu.get_outputs()
        return np.array(
            [outputs.get(v, 0.0) for v in self._obs_vars], dtype=np.float32
        )

    def _scale_action(self, action_norm: np.ndarray) -> np.ndarray:
        """Map action from [-1, 1] to [physical_min, physical_max]."""
        t = (action_norm + 1.0) / 2.0  # [0, 1]
        return self._act_phys_min + t * (self._act_phys_max - self._act_phys_min)

    def _apply_rate_limit(self, target: np.ndarray) -> np.ndarray:
        """Clamp change from current position to ±rate per variable."""
        delta = target - self._current_physical_action
        delta_clamped = np.clip(delta, -self._act_rate, self._act_rate)
        new_action = self._current_physical_action + delta_clamped
        return np.clip(new_action, self._act_phys_min, self._act_phys_max)

    def _update_history(self, raw_obs: np.ndarray) -> None:
        """Shift history buffer and insert new observation at the end."""
        self._history = np.roll(self._history, shift=-1, axis=0)
        self._history[-1] = raw_obs

    def _build_obs(self) -> np.ndarray:
        """Flatten history buffer to (n_obs * history_steps,) float32 array."""
        obs = self._history.flatten().astype(np.float32)
        # Clip to observation space bounds
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def _check_terminated(
        self, fmu_success: bool, raw_obs: np.ndarray
    ) -> tuple[bool, str]:
        """Return (terminated, reason) given FMU success and current observation."""
        if not fmu_success:
            return True, "fmu_solver_failure"

        violation_values = self._compute_constraint_violations(raw_obs)
        if violation_values["T_comp_min"] > 0.0:
            return True, "T_compressor_inlet_violation"
        if violation_values["surge_margin_main"] > 0.0:
            return True, "surge_margin_main_violation"
        if violation_values["surge_margin_recomp"] > 0.0:
            return True, "surge_margin_recomp_violation"

        return False, ""

    def _compute_reward(
        self, raw_obs: np.ndarray, physical_action: np.ndarray
    ) -> float:
        """Backward-compatible scalar reward helper."""
        return float(self._compute_reward_components(raw_obs, physical_action)["reward_total"])

    def _compute_reward_components(
        self, raw_obs: np.ndarray, physical_action: np.ndarray
    ) -> dict[str, float]:
        """Composite reward: tracking + efficiency + smoothness.

        The reward is intentionally shaped with a positive baseline so curriculum
        thresholds can be interpreted directly as episode returns.
        """
        obs_dict = {v: float(raw_obs[i]) for i, v in enumerate(self._obs_vars)}
        cfg = self._reward_cfg

        # Prefer direct W_net if available; otherwise derive from turbine/compressor work.
        w_net = self._first_present(obs_dict, ["W_net", "net_power"])
        if w_net is None:
            w_turb = self._first_present(obs_dict, ["W_turbine", "turbine.W_turbine"])
            w_comp = self._first_present(
                obs_dict, ["W_main_compressor", "main_compressor.W_comp"]
            )
            if w_turb is not None and w_comp is not None:
                w_net = w_turb - w_comp
            else:
                w_net = 0.0

        # FMPyAdapter.default_scale_offset() converts W_turbine/W_comp W→MW before
        # reaching this env, so w_net is already in MW.  MockFMU also returns MW.
        # w_net_unit_scale defaults to 1.0 (no conversion needed); set to 1e-6
        # only if using a raw-SI adapter that returns power in Watts.
        w_net_scale = float(cfg.get("w_net_unit_scale", 1.0))
        w_net_mw = w_net * w_net_scale

        w_net_sp = self._setpoint.get("W_net", cfg.get("rated_power_mw", 10.0))
        rated = max(float(cfg.get("rated_power_mw", 10.0)), 1e-6)
        tracking_err = ((w_net_mw - w_net_sp) / rated) ** 2
        r_tracking = cfg.get("w_tracking", 1.0) * max(0.0, 1.0 - tracking_err)

        # Efficiency bonus:
        # - use eta_thermal when available (MockFMU path),
        # - otherwise approximate via W_net / Q_recuperator (real FMU path).
        eta = self._first_present(obs_dict, ["eta_thermal", "cycle.eta_thermal"])
        if eta is None:
            q_rec = self._first_present(obs_dict, ["Q_recuperator", "recuperator.Q_actual"])
            if q_rec is not None and q_rec > 1e-6:
                eta = float(np.clip(w_net / q_rec, 0.0, 1.5))
            else:
                eta = 0.0
        eta_design = max(float(cfg.get("design_efficiency", 0.47)), 1e-6)
        eta_ratio = float(np.clip(eta / eta_design, 0.0, 2.0))
        r_efficiency = cfg.get("w_efficiency", 0.3) * eta_ratio

        # Smoothness penalty based on actual applied delta (normalized by action ranges).
        delta = physical_action - self._previous_physical_action
        act_range = np.maximum(self._act_phys_max - self._act_phys_min, 1e-9)
        normalized_delta = delta / act_range
        r_smooth = -cfg.get("w_smoothness", 0.1) * float(np.mean(normalized_delta**2))
        reward_total = float(r_tracking + r_efficiency + r_smooth)
        return {
            "r_tracking": float(r_tracking),
            "r_efficiency": float(r_efficiency),
            "r_smoothness": float(r_smooth),
            "tracking_error_norm_sq": float(tracking_err),
            "w_net": float(w_net_mw),  # MW units (after w_net_unit_scale applied)
            "w_net_setpoint": float(w_net_sp),
            "reward_total": reward_total,
        }

    def _compute_constraint_violations(self, raw_obs: np.ndarray) -> dict[str, float]:
        """Compute non-negative violation magnitudes for constrained variables."""
        obs_dict = {v: float(raw_obs[i]) for i, v in enumerate(self._obs_vars)}
        t_comp = self._first_present(
            obs_dict,
            ["T_compressor_inlet", "main_compressor.T_inlet_rt", "precooler.T_outlet_rt"],
        )
        surge_main = self._first_present(obs_dict, ["surge_margin_main"])
        surge_recomp = self._first_present(obs_dict, ["surge_margin_recomp"])

        t_min = float(self._safety_cfg.get("T_compressor_inlet_min", 32.2))
        sm_min = float(self._safety_cfg.get("surge_margin_min", 0.05))
        return {
            "T_comp_min": max(0.0, t_min - t_comp) if t_comp is not None else 0.0,
            "surge_margin_main": (
                max(0.0, sm_min - surge_main) if surge_main is not None else 0.0
            ),
            "surge_margin_recomp": (
                max(0.0, sm_min - surge_recomp) if surge_recomp is not None else 0.0
            ),
        }

    @staticmethod
    def _first_present(values: dict[str, float], keys: list[str]) -> float | None:
        """Return the first matching value for alias keys."""
        for key in keys:
            if key in values:
                return values[key]
        return None

    def _build_disturbance_profile(self) -> dict[str, Any]:
        """Create a deterministic per-episode disturbance profile from RNG seed."""
        phase = int(self._curriculum_phase)
        max_steps = max(1, int(self._max_steps))
        base_w_net = float(
            self._base_setpoint.get("W_net", self._reward_cfg.get("rated_power_mw", 10.0))
        )
        profile: dict[str, Any] = {
            "phase": phase,
            "max_steps": max_steps,
            "base_w_net": base_w_net,
        }
        if phase == 0:
            return profile

        rng = self.np_random

        if phase == 1:
            n_knots = int(rng.integers(4, 7))
            knot_times = np.linspace(0, max_steps - 1, n_knots, dtype=np.int32).tolist()
            knot_values = (base_w_net * rng.uniform(0.7, 1.3, size=n_knots)).tolist()
            profile["load_follow_knot_times"] = [int(t) for t in knot_times]
            profile["load_follow_knot_values"] = [float(v) for v in knot_values]

        elif phase == 2:
            cooling_idx = 3 if len(self._action_vars) > 3 else None
            if cooling_idx is not None:
                cooling_span = float(
                    max(1e-6, self._act_phys_max[cooling_idx] - self._act_phys_min[cooling_idx])
                )
            else:
                cooling_span = 1.0
            profile["ambient_amplitude"] = float(rng.uniform(0.1, 0.35) * cooling_span)
            profile["ambient_period_steps"] = int(rng.integers(300, 601))
            profile["ambient_phase"] = float(rng.uniform(0.0, 2.0 * math.pi))

        elif phase == 3:
            profile["eaf_low"] = float(self._act_phys_min[0]) if len(self._action_vars) > 0 else 0.0
            profile["eaf_high"] = float(self._act_phys_max[0]) if len(self._action_vars) > 0 else 0.0
            profile["eaf_ramp_up_steps"] = int(rng.integers(80, 121))
            profile["eaf_hold_high_steps"] = int(rng.integers(40, 81))
            profile["eaf_ramp_down_steps"] = int(rng.integers(80, 121))
            profile["eaf_hold_low_steps"] = int(rng.integers(20, 61))

        elif phase == 4:
            step_low = min(50, max_steps - 1)
            step_high = min(150, max_steps - 1)
            profile["load_reject_step"] = int(
                rng.integers(step_low, max(step_low + 1, step_high + 1))
            )
            profile["load_reject_pre_factor"] = float(rng.uniform(0.95, 1.05))
            profile["load_reject_post_factor"] = float(rng.uniform(0.45, 0.55))

        elif phase == 5:
            startup_steps = min(max_steps, int(rng.integers(120, 241)))
            profile["cold_startup_steps"] = max(1, startup_steps)
            profile["cold_start_w_factor"] = float(rng.uniform(0.15, 0.35))
            profile["cold_target_w_factor"] = float(rng.uniform(0.95, 1.05))
            if len(self._action_vars) > 0:
                profile["cold_heat_low"] = float(self._act_phys_min[0])
                profile["cold_heat_high"] = float(
                    self._act_phys_min[0] + 0.9 * (self._act_phys_max[0] - self._act_phys_min[0])
                )

        else:  # phase >= 6 emergency trip
            trip_low = min(100, max_steps - 1)
            trip_high = min(220, max_steps - 1)
            profile["trip_step"] = int(
                rng.integers(trip_low, max(trip_low + 1, trip_high + 1))
            )
            profile["pre_trip_w_factor"] = float(rng.uniform(0.95, 1.05))
            profile["post_trip_w_factor"] = float(rng.uniform(0.40, 0.55))
            if len(self._action_vars) > 0:
                profile["trip_heat_normal"] = float(
                    self._act_phys_min[0] + 0.75 * (self._act_phys_max[0] - self._act_phys_min[0])
                )
                profile["trip_heat_min"] = float(self._act_phys_min[0])

        return profile

    @staticmethod
    def _trapezoid_value(
        step: int,
        low: float,
        high: float,
        ramp_up_steps: int,
        hold_high_steps: int,
        ramp_down_steps: int,
        hold_low_steps: int,
    ) -> float:
        """Compute periodic trapezoid waveform value at `step`."""
        cycle = max(1, ramp_up_steps + hold_high_steps + ramp_down_steps + hold_low_steps)
        t = int(step % cycle)
        if t < ramp_up_steps:
            frac = t / max(ramp_up_steps, 1)
            return float(low + frac * (high - low))
        t -= ramp_up_steps
        if t < hold_high_steps:
            return float(high)
        t -= hold_high_steps
        if t < ramp_down_steps:
            frac = t / max(ramp_down_steps, 1)
            return float(high - frac * (high - low))
        return float(low)

    def _apply_curriculum_disturbance(self) -> tuple[bool, dict[str, float]]:
        """Apply phase-specific structured disturbance.

        Returns
        -------
        tuple[bool, dict[str, float]]
            disturbance_applied flag and FMU input overrides to merge into base inputs.
        """
        phase = int(self._curriculum_phase)
        if phase == 0:
            return False, {}

        t = int(self._step_count)
        profile = self._disturbance_profile
        base_w = float(profile.get("base_w_net", self._setpoint.get("W_net", 10.0)))
        overrides: dict[str, float] = {}
        applied = False

        if phase == 1:
            times = np.asarray(profile.get("load_follow_knot_times", [0, max(self._max_steps - 1, 1)]), dtype=np.float64)
            values = np.asarray(profile.get("load_follow_knot_values", [base_w, base_w]), dtype=np.float64)
            w_target = float(np.interp(float(t), times, values))
            self._setpoint["W_net"] = w_target
            applied = True

        elif phase == 2:
            if len(self._action_vars) > 3:
                amp = float(profile["ambient_amplitude"])
                period = max(1.0, float(profile["ambient_period_steps"]))
                phi = float(profile["ambient_phase"])
                center = float(0.5 * (self._act_phys_min[3] + self._act_phys_max[3]))
                value = center + amp * math.sin((2.0 * math.pi * t / period) + phi)
                value = float(np.clip(value, self._act_phys_min[3], self._act_phys_max[3]))
                overrides[self._action_vars[3]] = value
                applied = True

        elif phase == 3:
            if len(self._action_vars) > 0:
                value = self._trapezoid_value(
                    step=t,
                    low=float(profile["eaf_low"]),
                    high=float(profile["eaf_high"]),
                    ramp_up_steps=int(profile["eaf_ramp_up_steps"]),
                    hold_high_steps=int(profile["eaf_hold_high_steps"]),
                    ramp_down_steps=int(profile["eaf_ramp_down_steps"]),
                    hold_low_steps=int(profile["eaf_hold_low_steps"]),
                )
                overrides[self._action_vars[0]] = value
                applied = True

        elif phase == 4:
            step_at = int(profile["load_reject_step"])
            pre = float(profile["load_reject_pre_factor"])
            post = float(profile["load_reject_post_factor"])
            self._setpoint["W_net"] = base_w * (post if t >= step_at else pre)
            applied = True

        elif phase == 5:
            startup_steps = int(profile["cold_startup_steps"])
            w0 = base_w * float(profile["cold_start_w_factor"])
            w1 = base_w * float(profile["cold_target_w_factor"])
            frac = float(np.clip(t / max(startup_steps, 1), 0.0, 1.0))
            self._setpoint["W_net"] = float(w0 + frac * (w1 - w0))
            if len(self._action_vars) > 0:
                heat = float(profile["cold_heat_low"] + frac * (profile["cold_heat_high"] - profile["cold_heat_low"]))
                overrides[self._action_vars[0]] = heat
            applied = True

        else:  # phase >= 6
            trip_step = int(profile["trip_step"])
            self._setpoint["W_net"] = base_w * (
                float(profile["post_trip_w_factor"])
                if t >= trip_step
                else float(profile["pre_trip_w_factor"])
            )
            if len(self._action_vars) > 0:
                overrides[self._action_vars[0]] = (
                    float(profile["trip_heat_min"])
                    if t >= trip_step
                    else float(profile["trip_heat_normal"])
                )
            applied = True

        return applied, overrides
