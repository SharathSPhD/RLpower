#!/usr/bin/env python3
"""Evaluate PID baseline policy on the FMU environment."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PID baseline evaluation")
    parser.add_argument(
        "--fmu-path",
        type=str,
        default="artifacts/fmu_build/SCO2RecuperatedCycle.fmu",
        help="Path to compiled FMU",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Episode count")
    parser.add_argument("--max-steps", type=int, default=120, help="Max steps per episode")
    return parser.parse_args()


def _resolve(project_root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root / p


def _load_env_config(project_root: Path, max_steps: int) -> dict:
    env_cfg = yaml.safe_load((project_root / "configs/environment/env.yaml").read_text())
    safety_cfg = yaml.safe_load((project_root / "configs/safety/constraints.yaml").read_text())
    obs_vars_raw = env_cfg["observation"]["variables"]
    fmu_obs_vars = [v for v in obs_vars_raw if v.get("fmu_var") is not None]
    obs_vars = [v["fmu_var"] for v in fmu_obs_vars]
    obs_bounds = {v["fmu_var"]: (v["min"], v["max"]) for v in fmu_obs_vars}

    act_vars_raw = env_cfg["action"]["variables"]
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
    return {
        "obs_vars": obs_vars,
        "obs_bounds": obs_bounds,
        "action_vars": action_vars,
        "action_config": action_config,
        "history_steps": env_cfg["observation"].get("history_steps", 5),
        "step_size": 5.0,
        "episode_max_steps": max_steps,
        "reward": env_cfg["reward"],
        "safety": {
            "T_compressor_inlet_min": hard.get("compressor_inlet_temp_min_c", 32.2),
            "surge_margin_min": hard.get("surge_margin_main_min", 0.05),
        },
        "setpoint": {"W_net": 10.0},
    }


def _find_obs_index(obs_vars: list[str], candidates: list[str], default: int = 0) -> int:
    for name in candidates:
        if name in obs_vars:
            return obs_vars.index(name)
    return default


def _build_pid_config(env_config: dict) -> dict:
    obs_vars = env_config["obs_vars"]
    action_vars = env_config["action_vars"]
    return {
        "obs_vars": obs_vars,
        "action_vars": action_vars,
        "n_obs": len(obs_vars),
        "history_steps": env_config.get("history_steps", 1),
        "gains": {
            action_vars[0]: {"kp": 0.02, "ki": 0.001},
            action_vars[1]: {"kp": 0.02, "ki": 0.001},
            action_vars[2]: {"kp": 0.02, "ki": 0.001},
            action_vars[3]: {"kp": 0.02, "ki": 0.001},
        },
        "setpoints": {
            "T_turbine_inlet": 750.0,
            "T_compressor_inlet": 33.0,
            "P_high": 20.0,
            "W_net": 10.0,
        },
        "measurement_indices": {
            action_vars[0]: _find_obs_index(obs_vars, ["turbine.T_inlet_rt", "T_turbine_inlet"], 0),
            action_vars[1]: _find_obs_index(obs_vars, ["main_compressor.T_inlet_rt", "T_compressor_inlet"], 0),
            action_vars[2]: _find_obs_index(obs_vars, ["main_compressor.p_outlet", "P_high"], 0),
            action_vars[3]: _find_obs_index(obs_vars, ["precooler.T_outlet_rt", "T_precooler_outlet"], 0),
        },
        "dt": env_config.get("step_size", 5.0),
    }


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "src"))
    fmu_path = _resolve(project_root, args.fmu_path)
    if not fmu_path.exists():
        print(f"FMU not found: {fmu_path}")
        return 1

    env_cfg = _load_env_config(project_root, max_steps=args.max_steps)

    from sco2rl.environment.sco2_env import SCO2FMUEnv
    from sco2rl.simulation.fmu.fmpy_adapter import FMPyAdapter
    from sco2rl.deployment.inference.pid_baseline import PIDBaseline

    adapter = FMPyAdapter(
        fmu_path=str(fmu_path),
        obs_vars=env_cfg["obs_vars"],
        action_vars=env_cfg["action_vars"],
        instance_name="pid_baseline_eval",
        scale_offset=FMPyAdapter.default_scale_offset(),
    )
    env = SCO2FMUEnv(fmu=adapter, config=env_cfg)
    pid = PIDBaseline(_build_pid_config(env_cfg))

    ep_returns: list[float] = []
    try:
        for seed in range(args.episodes):
            obs, _ = env.reset(seed=seed)
            pid.reset()
            ep_return = 0.0
            done = False
            while not done:
                action, _ = pid.predict(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += float(reward)
                done = terminated or truncated
            ep_returns.append(ep_return)
    finally:
        env.close()

    print("[evaluate_pid_baseline] Done")
    print(f"  episodes: {len(ep_returns)}")
    print(f"  mean_return: {float(np.mean(ep_returns)):.4f}")
    print(f"  min_return:  {float(np.min(ep_returns)):.4f}")
    print(f"  max_return:  {float(np.max(ep_returns)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
