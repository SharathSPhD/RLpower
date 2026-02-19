#!/usr/bin/env python3
"""PID gain tuning via open-loop step-response characterization.

For each actuator channel, applies a step input while holding all others at
their nominal values, measures the open-loop step response from the FMU, and
computes Ziegler-Nichols PID gains from the response characteristics (static
gain K, time constant T, dead time L).

ZN tuning rules used (closed-loop, continuous):
    kp = 1.2 * T / (K * L)
    ki = kp / (2 * L)         (= Ti = 2L)
    kd = kp * 0.5 * L         (= Td = 0.5L)

Output: artifacts/pid_tuning/pid_gains.json

Usage inside Docker:
    python scripts/tune_pid_baseline.py \
        --fmu-path artifacts/fmu_build/SCO2RecuperatedCycle.fmu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# Nominal operating point (all channels held here while one is stepped)
# ---------------------------------------------------------------------------
NOMINAL_NORMALIZED = {
    "regulator.T_init":    0.0,   # mid of 800–1200 K → 1000 K
    "regulator.m_flow_init": 0.0, # mid of 60–130 kg/s → 95 kg/s
    "turbine.p_out":       0.0,   # mid of 7–9 MPa → 8 MPa
    "precooler.T_output":  0.0,   # mid of 305.65–315 K → 310 K
}

STEP_AMPLITUDE = 0.3   # normalized step size (30 % of range)
STEP_AT_STEP   = 5     # apply step at this simulation step
OBSERVE_STEPS  = 60    # total steps in each characterization episode


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PID gain tuning via step response")
    p.add_argument("--fmu-path", required=True)
    p.add_argument("--output", default="artifacts/pid_tuning/pid_gains.json")
    p.add_argument("--step-amplitude", type=float, default=STEP_AMPLITUDE)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _build_env(fmu_path: str):
    """Build a single SCO2FMUEnv for characterization."""
    import yaml
    from sco2rl.simulation.fmu.fmpy_adapter import FMPyAdapter
    from sco2rl.environment.sco2_env import SCO2FMUEnv

    env_cfg_raw = yaml.safe_load(
        (PROJECT_ROOT / "configs/environment/env.yaml").read_text()
    )
    safety_cfg = yaml.safe_load(
        (PROJECT_ROOT / "configs/safety/constraints.yaml").read_text()
    )

    obs_vars_raw = env_cfg_raw["observation"]["variables"]
    fmu_obs_vars = [v for v in obs_vars_raw if v.get("fmu_var") is not None]
    obs_vars = [v["fmu_var"] for v in fmu_obs_vars]
    obs_bounds = {v["fmu_var"]: (v["min"], v["max"]) for v in fmu_obs_vars}

    act_vars_raw = env_cfg_raw["action"]["variables"]
    action_vars = [v["fmu_var"] for v in act_vars_raw]
    action_config = {
        v["fmu_var"]: {
            "min": v["physical_min"],
            "max": v["physical_max"],
            "rate": v.get("rate_limit_per_step", 1.0),
        }
        for v in act_vars_raw
    }

    hard = safety_cfg.get("hard_constraints", {})
    env_config = {
        "obs_vars": obs_vars,
        "obs_bounds": obs_bounds,
        "action_vars": action_vars,
        "action_config": action_config,
        "history_steps": env_cfg_raw["observation"].get("history_steps", 5),
        "step_size": 5.0,
        "episode_max_steps": OBSERVE_STEPS + 10,
        "reward": env_cfg_raw["reward"],
        "safety": {
            "T_compressor_inlet_min": hard.get("compressor_inlet_temp_min_c", 32.2),
            "surge_margin_min": hard.get("surge_margin_main_min", 0.05),
        },
        "setpoint": {"W_net": 10.0},
    }

    adapter = FMPyAdapter(
        fmu_path=fmu_path,
        obs_vars=obs_vars,
        action_vars=action_vars,
        instance_name="pid_tune",
        scale_offset=FMPyAdapter.default_scale_offset(),
    )
    adapter.initialize(
        start_time=0.0,
        stop_time=(OBSERVE_STEPS + 10) * 5.0,
        step_size=5.0,
    )
    return SCO2FMUEnv(fmu=adapter, config=env_config), obs_vars, action_vars


def _channel_names(obs_vars: list[str], action_vars: list[str]) -> dict[str, dict]:
    """Map action variables to their measurement obs variable and index."""
    def _idx(candidates: list[str]) -> int:
        for c in candidates:
            if c in obs_vars:
                return obs_vars.index(c)
        return 0

    return {
        action_vars[0]: {
            "label": "bypass_valve",
            "meas_var": obs_vars[_idx(["turbine.T_inlet_rt"])],
            "meas_idx": _idx(["turbine.T_inlet_rt"]),
            "meas_unit": "°C",
        },
        action_vars[1]: {
            "label": "igv",
            "meas_var": obs_vars[_idx(["main_compressor.T_inlet_rt"])],
            "meas_idx": _idx(["main_compressor.T_inlet_rt"]),
            "meas_unit": "°C",
        },
        action_vars[2]: {
            "label": "inventory_valve",
            "meas_var": obs_vars[_idx(["main_compressor.p_outlet"])],
            "meas_idx": _idx(["main_compressor.p_outlet"]),
            "meas_unit": "MPa",
        },
        action_vars[3]: {
            "label": "cooling_flow",
            "meas_var": obs_vars[_idx(["precooler.T_outlet_rt"])],
            "meas_idx": _idx(["precooler.T_outlet_rt"]),
            "meas_unit": "°C",
        },
    }


def _step_response(
    env,
    action_vars: list[str],
    act_idx: int,
    step_amp: float,
    channels: dict,
) -> dict:
    """Run a single open-loop step-response episode for act_idx."""
    act_var = action_vars[act_idx]
    info = channels[act_var]
    obs_idx = info["meas_idx"]

    obs, _ = env.reset(seed=42)
    n_obs_raw = obs.shape[0] // 5  # history_steps=5

    readings: list[float] = []

    for step in range(OBSERVE_STEPS):
        action = np.zeros(len(action_vars), dtype=np.float32)
        # Apply step at STEP_AT_STEP
        if step >= STEP_AT_STEP:
            action[act_idx] = step_amp

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

        # Read measurement from latest history slice
        raw_obs = obs[-n_obs_raw:]
        readings.append(float(raw_obs[obs_idx]))

    return {
        "action_var": act_var,
        "label": info["label"],
        "meas_var": info["meas_var"],
        "meas_unit": info["meas_unit"],
        "step_amplitude": step_amp,
        "step_at_step": STEP_AT_STEP,
        "readings": readings,
    }


def _zn_gains(result: dict, dt: float = 5.0) -> dict:
    """Compute ZN-based PID gains from step-response data.

    Uses the graphical method:
    1. Estimate steady-state gain K = Δy / Δu
    2. Find inflection point → dead time L and time constant T via tangent line
    3. Apply ZN formulas

    Falls back to conservative defaults if response is too noisy.
    """
    readings = np.array(result["readings"])
    step_at = result["step_at_step"]
    step_amp = result["step_amplitude"]

    if len(readings) < step_at + 5:
        return {"kp": 0.02, "ki": 0.002, "kd": 0.004, "note": "insufficient_data"}

    baseline = np.mean(readings[:step_at]) if step_at > 0 else readings[0]
    response = readings[step_at:]

    delta_y = float(np.nanmean(response[-5:])) - baseline
    if abs(delta_y) < 1e-6 or abs(step_amp) < 1e-6:
        return {"kp": 0.02, "ki": 0.002, "kd": 0.004, "note": "no_response"}

    K = delta_y / step_amp  # static gain: output change / input change

    # Find inflection point (max derivative in response)
    diff = np.diff(response)
    infl_idx = int(np.argmax(np.abs(diff)))
    slope = diff[infl_idx] / dt                     # °C/s (or MPa/s)
    tangent_x0 = infl_idx * dt                      # time at inflection
    tangent_y0 = response[infl_idx]

    # Dead time L: intersection of tangent line with baseline level
    if abs(slope) < 1e-9:
        L = dt  # minimum 1 step dead time
    else:
        L = max(dt, tangent_x0 - (tangent_y0 - baseline) / slope)

    # Time constant T: tangent reaches final value
    T = max(dt, abs(delta_y) / abs(slope))

    # ZN closed-loop PID rules
    kp = 1.2 * T / (abs(K) * L + 1e-9)
    ki = kp / (2.0 * L)
    kd = kp * 0.5 * L

    # Clip to reasonable bounds for normalized [-1, 1] action space
    kp = float(np.clip(kp, 0.001, 1.0))
    ki = float(np.clip(ki, 0.0001, 0.2))
    kd = float(np.clip(kd, 0.0001, 0.5))

    return {
        "kp": round(kp, 6),
        "ki": round(ki, 6),
        "kd": round(kd, 6),
        "K": round(K, 4),
        "L_s": round(L, 2),
        "T_s": round(T, 2),
        "delta_y": round(delta_y, 4),
        "derivative_filter_tau": 0.15,
        "note": "ziegler_nichols",
    }


def main() -> int:
    args = _parse_args()
    fmu_path = str(PROJECT_ROOT / args.fmu_path) if not Path(args.fmu_path).is_absolute() else args.fmu_path
    if not Path(fmu_path).exists():
        print(f"[tune_pid] FMU not found: {fmu_path}")
        return 1

    print(f"[tune_pid] Building env from {fmu_path}...")
    env, obs_vars, action_vars = _build_env(fmu_path)
    channels = _channel_names(obs_vars, action_vars)

    results = {}
    for act_idx, act_var in enumerate(action_vars):
        label = channels[act_var]["label"]
        print(f"[tune_pid] Step-response characterization: {label} ({act_var})")
        sr = _step_response(env, action_vars, act_idx, args.step_amplitude, channels)
        gains = _zn_gains(sr)
        results[act_var] = {
            "label": label,
            "meas_var": channels[act_var]["meas_var"],
            "step_response_summary": {
                "n_readings": len(sr["readings"]),
                "baseline": float(np.mean(sr["readings"][:STEP_AT_STEP])) if STEP_AT_STEP > 0 else sr["readings"][0],
                "steady_state": float(np.mean(sr["readings"][-5:])) if len(sr["readings"]) >= 5 else None,
            },
            "gains": gains,
        }
        print(f"  → kp={gains['kp']:.5f}  ki={gains['ki']:.5f}  kd={gains['kd']:.5f}  note={gains.get('note','')}")

    env.close()

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[tune_pid] Gains written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
