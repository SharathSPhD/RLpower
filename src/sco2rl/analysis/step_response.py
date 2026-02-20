"""Step-response analysis for SCO2FMUEnv controllers.

Runs a controlled step-input experiment by:
1. Running the environment + policy to steady state.
2. Applying a setpoint step at a configurable time.
3. Recording the controlled variable trajectory.
4. Computing classical transient metrics (overshoot, settling time, rise time,
   IAE, ITAE, ISE).

All functions are pure (no side-effects beyond the provided env) and work
with any Controller-compatible policy, so both PID and RL policies can be
compared with identical experimental conditions.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.integrate import trapezoid

from sco2rl.analysis.metrics import StepResponseResult


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_step_scenario(
    env: Any,
    policy: Any,
    step_magnitude: float,
    step_at_step: int = 50,
    n_steps: int = 250,
    dt: float = 5.0,
    variable: str = "W_net",
    phase: int = 0,
    scenario: str = "step",
    seed: int = 0,
) -> StepResponseResult:
    """Run a step-input experiment and return control metrics.

    The experiment protocol:
    - Steps 0 … step_at_step-1: run under baseline setpoint (warm-up).
    - Step step_at_step: increase W_net setpoint by ``step_magnitude``.
    - Steps step_at_step … n_steps-1: record response.

    Parameters
    ----------
    env:
        ``SCO2FMUEnv`` instance (not wrapped with VecNormalize).
    policy:
        Any object with ``predict(obs, deterministic=True) → (action, None)``
        and ``reset() → None``.  Compatible with both ``MultiLoopPID`` and
        ``RLController``.
    step_magnitude:
        Signed change in W_net setpoint in MW.  E.g. ``+2.0`` for +20 % on a
        10 MW design point.
    step_at_step:
        Simulation step index at which the setpoint step is applied.
    n_steps:
        Total episode length including warm-up.
    dt:
        Simulation step size in seconds (must match env.step_size).
    variable:
        Observation variable to track.  Typically ``"W_net"``.
    phase:
        Curriculum phase for labelling; also used to set the env phase.
    scenario:
        Descriptive label (e.g. ``"step_load_+20pct"``).
    seed:
        Episode seed.

    Returns
    -------
    StepResponseResult
        Populated with time series and computed metrics.
    """
    controller_name = getattr(policy, "name", type(policy).__name__)

    env.set_curriculum_phase(phase, episode_max_steps=n_steps)
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()

    # Record trajectory
    time_s: list[float] = []
    response: list[float] = []
    setpoints: list[float] = []

    base_setpoint = float(env._base_setpoint.get("W_net", 10.0))
    current_setpoint = base_setpoint
    step_applied = False

    for step_idx in range(n_steps):
        t = step_idx * dt

        # Apply setpoint step
        if step_idx == step_at_step and not step_applied:
            current_setpoint = base_setpoint + step_magnitude
            env._setpoint["W_net"] = current_setpoint
            step_applied = True

        # Record BEFORE step (pre-action observation)
        info_raw = {v: float(obs[i]) for i, v in enumerate(env._obs_vars)} if hasattr(env, "_obs_vars") else {}
        var_val = _extract_variable(obs, env, variable)

        time_s.append(t)
        response.append(var_val)
        setpoints.append(current_setpoint)

        action, _ = policy.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, _info = env.step(action)

        if terminated or truncated:
            break

    time_arr = np.array(time_s)
    resp_arr = np.array(response)
    sp_arr = np.array(setpoints)

    # Split into warm-up and post-step
    step_onset_s = step_at_step * dt
    post_mask = time_arr >= step_onset_s

    result = StepResponseResult(
        variable=variable,
        controller=controller_name,
        phase=phase,
        scenario=scenario,
        seed=seed,
        time_s=time_s,
        setpoint=setpoints,
        response=response,
        step_onset_s=step_onset_s,
        initial_value=float(resp_arr[min(max(step_at_step - 1, 0), len(resp_arr) - 1)]) if len(resp_arr) > 0 else 0.0,
        step_magnitude=step_magnitude,
    )

    if post_mask.sum() < 5:
        return result

    post_time = time_arr[post_mask] - step_onset_s
    post_resp = resp_arr[post_mask]
    post_sp = sp_arr[post_mask]

    # Final (asymptotic) value = mean of last 10 %
    n_tail = max(1, int(0.1 * len(post_resp)))
    final_val = float(np.mean(post_resp[-n_tail:]))
    result.final_value = final_val

    # Steady-state error
    final_sp = float(post_sp[-1])
    result.steady_state_error = final_val - final_sp

    # Metrics
    metrics = compute_step_metrics(
        post_time, post_resp, final_val, step_magnitude, dt
    )
    result.overshoot_pct = metrics["overshoot_pct"]
    result.undershoot_pct = metrics["undershoot_pct"]
    result.settling_time_s = metrics["settling_time_s"]
    result.rise_time_s = metrics["rise_time_s"]
    result.peak_time_s = metrics["peak_time_s"]
    result.iae = metrics["iae"]
    result.ise = metrics["ise"]
    result.itae = metrics["itae"]

    return result


def compute_step_metrics(
    post_time: np.ndarray,
    post_resp: np.ndarray,
    final_val: float,
    step_magnitude: float,
    dt: float,
    settle_band: float = 0.02,
) -> dict[str, float]:
    """Compute classical step-response metrics from a post-step trajectory.

    Parameters
    ----------
    post_time:
        Time array in seconds, starting at 0 (t=0 is the step onset).
    post_resp:
        Response (controlled variable) array.
    final_val:
        Asymptotic final value (e.g. mean of last 10 %).
    step_magnitude:
        Requested setpoint change in physical units.
    dt:
        Step size in seconds (used for integration).
    settle_band:
        Settling band as a fraction of ``|step_magnitude|``  (default 2 %).

    Returns
    -------
    dict with keys: overshoot_pct, undershoot_pct, settling_time_s,
        rise_time_s, peak_time_s, iae, ise, itae.
    """
    n = len(post_resp)
    if n == 0 or abs(step_magnitude) < 1e-12:
        return {k: 0.0 for k in [
            "overshoot_pct", "undershoot_pct", "settling_time_s",
            "rise_time_s", "peak_time_s", "iae", "ise", "itae"]}

    initial_val = float(post_resp[0])
    step_abs = abs(step_magnitude)
    sign = 1.0 if step_magnitude >= 0 else -1.0

    # Normalised deviation above/below final
    deviation = (post_resp - final_val) * sign  # positive = overshoot direction

    # Overshoot
    peak_dev = float(np.max(deviation))
    overshoot_pct = max(0.0, peak_dev / step_abs * 100.0) if step_abs > 0 else 0.0
    peak_idx = int(np.argmax(deviation))
    peak_time_s = float(post_time[peak_idx]) if n > 0 else 0.0

    # Undershoot (deviation in opposite direction)
    undershoot_dev = float(np.min(deviation))
    undershoot_pct = max(0.0, -undershoot_dev / step_abs * 100.0) if step_abs > 0 else 0.0

    # Rise time: 10 % → 90 % of step_magnitude
    threshold_10 = initial_val + sign * 0.10 * step_magnitude
    threshold_90 = initial_val + sign * 0.90 * step_magnitude
    idx_10 = _first_crossing(post_resp, threshold_10, sign)
    idx_90 = _first_crossing(post_resp, threshold_90, sign)
    if idx_10 is not None and idx_90 is not None and idx_90 > idx_10:
        rise_time_s = float(post_time[idx_90] - post_time[idx_10])
    else:
        rise_time_s = float(post_time[-1])  # Did not reach 90 %

    # Settling time: last time response leaves ±band around final
    band = settle_band * step_abs
    outside = np.abs(post_resp - final_val) > band
    if not np.any(outside):
        settling_time_s = 0.0
    else:
        last_outside = int(np.max(np.where(outside)[0]))
        settling_time_s = float(post_time[min(last_outside + 1, n - 1)])

    # Error integrals (error = setpoint - response ≈ final_val - response)
    error = final_val - post_resp  # signed error relative to final value
    iae = float(trapezoid(np.abs(error), post_time))
    ise = float(trapezoid(error ** 2, post_time))
    itae = float(trapezoid(post_time * np.abs(error), post_time))

    return {
        "overshoot_pct": overshoot_pct,
        "undershoot_pct": undershoot_pct,
        "settling_time_s": settling_time_s,
        "rise_time_s": rise_time_s,
        "peak_time_s": peak_time_s,
        "iae": iae,
        "ise": ise,
        "itae": itae,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_variable(obs: np.ndarray, env: Any, variable: str) -> float:
    """Extract a named variable from the latest observation window."""
    obs_vars = getattr(env, "_obs_vars", [])
    history_steps = getattr(env, "_history_steps", 1)
    n_obs = len(obs_vars)

    # Latest window starts at offset (history_steps - 1) * n_obs
    offset = max(history_steps - 1, 0) * n_obs

    if variable in obs_vars:
        idx = obs_vars.index(variable)
        full_idx = offset + idx
        if full_idx < len(obs):
            return float(obs[full_idx])

    # Fallback: search all obs windows
    for i, v in enumerate(obs_vars):
        if v == variable or variable.lower() in v.lower():
            full_idx = offset + i
            if full_idx < len(obs):
                return float(obs[full_idx])

    return 0.0


def _first_crossing(arr: np.ndarray, threshold: float, sign: float) -> int | None:
    """Return index of first element crossing threshold in direction ``sign``."""
    if sign > 0:
        indices = np.where(arr >= threshold)[0]
    else:
        indices = np.where(arr <= threshold)[0]
    return int(indices[0]) if len(indices) > 0 else None
