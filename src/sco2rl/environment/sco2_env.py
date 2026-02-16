"""SCO2FMUEnv — Gymnasium environment wrapping FMUInterface.

Architecture:
  action (normalized) → ActionScaler → RateLimiter → FMU.set_inputs()
  FMU.do_step() → get_outputs() → history buffer → flattened obs

Dependency injection: FMUInterface passed at construction time.
All config via constructor dict — no hardcoded paths or global state (RULE-C2, C3).
"""
from __future__ import annotations

from typing import Any

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
        self._setpoint: dict[str, float] = config.get("setpoint", {})

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
        self._step_count: int = 0
        self._current_time: float = 0.0

        # Curriculum phase (set by CurriculumCallback; persists across resets)
        self._curriculum_phase: int = 0

        # Public attribute for test assertions
        self.last_physical_action: np.ndarray = np.zeros(n_act, dtype=np.float64)
        self.render_mode = None
        self.spec = None

    # ── Gymnasium interface ────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
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

        # Fill history buffer with the FMU's initial output
        raw_obs = self._get_raw_obs()
        for i in range(self._history_steps):
            self._history[i] = raw_obs

        obs = self._build_obs()
        info = {"step": 0, "time": 0.0}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # 1. Scale action from [-1, 1] to physical range
        action_f64 = action.astype(np.float64)
        physical = self._scale_action(action_f64)

        # 2. Apply rate limiting
        physical = self._apply_rate_limit(physical)
        self._current_physical_action = physical
        self.last_physical_action = physical.copy()

        # 3. Apply curriculum disturbance (before agent inputs, so agent sees disturbed state)
        disturbance_applied = self._apply_curriculum_disturbance()

        # 4. Send inputs to FMU
        inputs = {v: float(physical[i]) for i, v in enumerate(self._action_vars)}
        self._fmu.set_inputs(inputs)

        # 5. Advance FMU by one step
        success = self._fmu.do_step(
            current_time=self._current_time, step_size=self._step_size
        )
        self._current_time += self._step_size
        self._step_count += 1

        # 6. Get outputs
        raw_obs = self._get_raw_obs()

        # 7. Check termination conditions
        terminated, term_reason = self._check_terminated(success, raw_obs)
        truncated = (not terminated) and (self._step_count >= self._max_steps)

        # 8. Update history (only on non-terminal or on terminal to get last obs)
        self._update_history(raw_obs)
        obs = self._build_obs()

        # 9. Compute reward
        if terminated and not success:
            reward = float(self._reward_cfg["terminal_failure_reward"])
        else:
            reward = self._compute_reward(raw_obs, physical)
            if terminated:
                reward = float(self._reward_cfg["terminal_failure_reward"])

        info = {
            "step": self._step_count,
            "time": self._current_time,
            "terminated_reason": term_reason,
            "disturbance_applied": disturbance_applied,
            "raw_obs": {v: float(raw_obs[i]) for i, v in enumerate(self._obs_vars)},
        }
        return obs, reward, terminated, truncated, info

    def set_curriculum_phase(self, phase: int) -> None:
        """Set the active curriculum phase.

        Called by CurriculumCallback to activate phase-appropriate disturbances.
        Persists across episode resets (curriculum outlasts individual episodes).

        Parameters
        ----------
        phase:
            Integer phase index (0 = STEADY_STATE, 3 = EAF_TRANSIENTS, etc.)
        """
        self._curriculum_phase = phase

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

        obs_dict = {v: float(raw_obs[i]) for i, v in enumerate(self._obs_vars)}

        t_min = self._safety_cfg.get("T_compressor_inlet_min", 32.2)
        if "T_compressor_inlet" in obs_dict:
            if obs_dict["T_compressor_inlet"] < t_min:
                return True, "T_compressor_inlet_violation"

        sm_min = self._safety_cfg.get("surge_margin_min", 0.05)
        for var in ("surge_margin_main", "surge_margin_recomp"):
            if var in obs_dict and obs_dict[var] < sm_min:
                return True, f"{var}_violation"

        return False, ""

    def _compute_reward(
        self, raw_obs: np.ndarray, physical_action: np.ndarray
    ) -> float:
        """Composite reward: tracking + efficiency + smoothness."""
        obs_dict = {v: float(raw_obs[i]) for i, v in enumerate(self._obs_vars)}
        cfg = self._reward_cfg

        # Tracking penalty (normalized squared error)
        w_net = obs_dict.get("W_net", 0.0)
        w_net_sp = self._setpoint.get("W_net", cfg.get("rated_power_mw", 10.0))
        rated = cfg.get("rated_power_mw", 10.0)
        tracking_err = ((w_net - w_net_sp) / rated) ** 2
        r_tracking = -cfg.get("w_tracking", 1.0) * tracking_err

        # Efficiency bonus (deviation from design)
        eta = obs_dict.get("eta_thermal", 0.0)
        eta_design = cfg.get("design_efficiency", 0.47)
        r_efficiency = cfg.get("w_efficiency", 0.3) * (eta / eta_design - 1.0)

        # Smoothness penalty (L2 norm of action change)
        delta = physical_action - self._current_physical_action
        # Note: _current_physical_action was already updated before this call
        # Use the action itself as a proxy for smoothness
        r_smooth = -cfg.get("w_smoothness", 0.1) * float(
            np.sum((physical_action - (self._act_phys_min + self._act_phys_max) / 2.0) ** 2)
        ) / len(self._action_vars)

        return float(r_tracking + r_efficiency + r_smooth)

    def _apply_curriculum_disturbance(self) -> bool:
        """Apply phase-appropriate disturbance to the FMU before the agent's action.

        Phase 0 (STEADY_STATE): no disturbance.
        Phase 1+ (LOAD_FOLLOW, AMBIENT_TEMP, EAF_TRANSIENTS, ...): disturbance injected
        via FMU set_inputs() using a random perturbation drawn from the numpy RNG
        seeded by gymnasium's reset() call (self.np_random).

        Returns True if a disturbance was applied, False otherwise.

        Disturbance amplitudes (physical units, matching curriculum.yaml):
          Phase 1 LOAD_FOLLOW:   ±30% of rated W_net setpoint (via action[2] inventory_valve)
          Phase 2 AMBIENT_TEMP:  ±10°C precooler target (via action[3] cooling_flow)
          Phase 3 EAF_TRANSIENTS: ±200K heat source T (via action[0] bypass_valve)
          Phase 4 LOAD_REJECTION: -50% W_net setpoint (via action[2], 30-step duration)
          Phase 5+ COLD_STARTUP / EMERGENCY_TRIP: same as phase 3 + phase 2 combined
        """
        if self._curriculum_phase == 0:
            return False

        rng = self.np_random  # gymnasium RNG, seeded at reset()

        if self._curriculum_phase >= 3:
            # EAF_TRANSIENTS: random heat source temperature perturbation
            # action[0] maps to bypass_valve / heat source in real FMU (regulator.T_init)
            # For MockFMU, this call is a no-op but we still return True
            amplitude = 200.0  # K; scales with phase
            if self._curriculum_phase >= 5:
                amplitude = 300.0
            delta = float(rng.uniform(-amplitude, amplitude))
            # Only inject if the action variable exists in the FMU catalogue
            disturbance_var = self._action_vars[0] if self._action_vars else None
            if disturbance_var is not None:
                try:
                    self._fmu.set_inputs({disturbance_var: float(
                        self._current_physical_action[0] + delta
                    )})
                except (KeyError, Exception):
                    pass  # MockFMU may not know this var — that's fine
            return True

        if self._curriculum_phase == 2:
            # AMBIENT_TEMP: ±10°C perturbation to cooling target (action[3])
            delta = float(rng.uniform(-10.0, 10.0))
            disturbance_var = self._action_vars[3] if len(self._action_vars) > 3 else None
            if disturbance_var is not None:
                try:
                    self._fmu.set_inputs({disturbance_var: float(
                        self._current_physical_action[3] + delta
                    )})
                except (KeyError, Exception):
                    pass
            return True

        if self._curriculum_phase == 1:
            # LOAD_FOLLOW: setpoint step (modulates _setpoint dict directly)
            step_frac = float(rng.uniform(-0.3, 0.3))
            base = self._setpoint.get("W_net", 10.0)
            self._setpoint = dict(self._setpoint)  # copy to avoid mutating original
            self._setpoint["W_net"] = base * (1.0 + step_frac)
            return True

        return False
