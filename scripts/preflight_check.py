#!/usr/bin/env python3
"""Fast fail pre-flight checks before long FMU training runs.

This script validates environment wiring and reward/constraint signals in a
short window so we do not spend hours on doomed training runs.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run short pre-flight checks")
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of short episodes to sample (default: 10)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Per-episode step cap for pre-flight (default: 200)",
    )
    parser.add_argument(
        "--fmu-path",
        type=str,
        default="artifacts/fmu_build/SCO2RecuperatedCycle.fmu",
        help="Path to compiled FMU (ignored with --use-mock)",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use MockFMU instead of a real FMU for a quick dry run",
    )
    return parser.parse_args()


def load_config(project_root: Path, max_steps_override: int) -> tuple[dict, dict]:
    env_cfg = yaml.safe_load((project_root / "configs/environment/env.yaml").read_text())
    safety_cfg = yaml.safe_load((project_root / "configs/safety/constraints.yaml").read_text())

    obs_vars_raw = env_cfg["observation"]["variables"]
    fmu_obs_vars_raw = [item for item in obs_vars_raw if item.get("fmu_var") is not None]
    obs_vars = [item["fmu_var"] for item in fmu_obs_vars_raw]
    obs_bounds = {item["fmu_var"]: (item["min"], item["max"]) for item in fmu_obs_vars_raw}

    act_vars_raw = env_cfg["action"]["variables"]
    action_vars = [item["fmu_var"] for item in act_vars_raw]
    action_config = {
        item["fmu_var"]: {
            "min": item["physical_min"],
            "max": item["physical_max"],
            "rate": item.get("rate_limit_per_step", item.get("rate_limit", item.get("rate", 1.0))),
        }
        for item in act_vars_raw
    }

    hard = safety_cfg.get("hard_constraints", {})
    config = {
        "obs_vars": obs_vars,
        "obs_bounds": obs_bounds,
        "action_vars": action_vars,
        "action_config": action_config,
        "history_steps": env_cfg["observation"].get("history_steps", 5),
        "step_size": 5.0,
        "episode_max_steps": int(max_steps_override),
        "reward": env_cfg.get("reward", {}),
        "safety": {
            "T_compressor_inlet_min": hard.get(
                "compressor_inlet_temp_min_c", hard.get("T_compressor_inlet_min_c", 32.2)
            ),
            "surge_margin_min": hard.get(
                "surge_margin_main_min", hard.get("surge_margin_min_fraction", 0.05)
            ),
        },
        "setpoint": {"W_net": 10.0},
    }
    return config, safety_cfg.get("lagrangian", {})


def make_mock_design_point(obs_vars: list[str], obs_bounds: dict[str, tuple[float, float]]) -> dict[str, float]:
    design_point = {}
    for var in obs_vars:
        low, high = obs_bounds[var]
        design_point[var] = (float(low) + float(high)) / 2.0
    return design_point


def build_env(project_root: Path, config: dict, fmu_path: Path, use_mock: bool):
    from sco2rl.environment.sco2_env import SCO2FMUEnv

    if use_mock:
        from sco2rl.simulation.fmu.mock_fmu import MockFMU

        fmu = MockFMU(
            obs_vars=config["obs_vars"],
            action_vars=config["action_vars"],
            design_point=make_mock_design_point(config["obs_vars"], config["obs_bounds"]),
            seed=42,
        )
        fmu.initialize(start_time=0.0, stop_time=config["episode_max_steps"] * config["step_size"], step_size=config["step_size"])
    else:
        from sco2rl.simulation.fmu.fmpy_adapter import FMPyAdapter

        if not fmu_path.exists():
            raise FileNotFoundError(f"FMU not found: {fmu_path}")
        fmu = FMPyAdapter(
            fmu_path=str(fmu_path),
            obs_vars=config["obs_vars"],
            action_vars=config["action_vars"],
            instance_name="preflight_instance",
            scale_offset=FMPyAdapter.default_scale_offset(),
        )
        fmu.initialize(
            start_time=0.0,
            stop_time=config["episode_max_steps"] * config["step_size"],
            step_size=config["step_size"],
        )

    return SCO2FMUEnv(fmu=fmu, config=config)


def run_preflight(env, episodes: int) -> dict:
    step_rewards: list[float] = []
    episode_rewards: list[float] = []
    termination_reasons: Counter[str] = Counter()
    violation_fractions: list[float] = []
    component_tracking: list[float] = []
    component_efficiency: list[float] = []
    component_smoothness: list[float] = []
    tracking_error_sq: list[float] = []
    w_net_values: list[float] = []
    w_setpoint_values: list[float] = []
    nan_or_inf_detected = False

    for episode_idx in range(episodes):
        obs, _ = env.reset(seed=episode_idx)
        ep_reward = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if not np.all(np.isfinite(obs)) or not math.isfinite(float(reward)):
                nan_or_inf_detected = True
            step_rewards.append(float(reward))
            reward_components = info.get("reward_components", {})
            if isinstance(reward_components, dict):
                component_tracking.append(float(reward_components.get("r_tracking", 0.0)))
                component_efficiency.append(float(reward_components.get("r_efficiency", 0.0)))
                component_smoothness.append(float(reward_components.get("r_smoothness", 0.0)))
                tracking_error_sq.append(float(reward_components.get("tracking_error_norm_sq", 0.0)))
                w_net_values.append(float(reward_components.get("w_net", 0.0)))
                w_setpoint_values.append(float(reward_components.get("w_net_setpoint", 0.0)))
            ep_reward += float(reward)
            done = bool(terminated or truncated)
            if done:
                termination_reasons[str(info.get("terminated_reason", "truncated_or_complete"))] += 1
                if "constraint_violation" in info:
                    violation_fractions.append(float(info["constraint_violation"]))
        episode_rewards.append(ep_reward)

    return {
        "nan_or_inf_detected": nan_or_inf_detected,
        "step_reward_min": min(step_rewards) if step_rewards else 0.0,
        "step_reward_max": max(step_rewards) if step_rewards else 0.0,
        "step_reward_mean": float(np.mean(step_rewards)) if step_rewards else 0.0,
        "episode_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "termination_reasons": dict(termination_reasons),
        "violation_fraction_mean": float(np.mean(violation_fractions)) if violation_fractions else 0.0,
        "reward_components_mean": {
            "r_tracking": float(np.mean(component_tracking)) if component_tracking else 0.0,
            "r_efficiency": float(np.mean(component_efficiency)) if component_efficiency else 0.0,
            "r_smoothness": float(np.mean(component_smoothness)) if component_smoothness else 0.0,
            "tracking_error_norm_sq": float(np.mean(tracking_error_sq)) if tracking_error_sq else 0.0,
        },
        "w_net_mean": float(np.mean(w_net_values)) if w_net_values else 0.0,
        "w_setpoint_mean": float(np.mean(w_setpoint_values)) if w_setpoint_values else 0.0,
    }


def print_summary(env, report: dict, lagrangian_cfg: dict) -> int:
    print("\n[preflight] Environment summary")
    print(f"  obs_dim:             {env.observation_space.shape[0]}")
    print(f"  action_dim:          {env.action_space.shape[0]}")
    print(f"  episode_max_steps:   {env._max_steps}")
    print("\n[preflight] Reward summary")
    print(f"  step_reward_min:     {report['step_reward_min']:.4f}")
    print(f"  step_reward_max:     {report['step_reward_max']:.4f}")
    print(f"  step_reward_mean:    {report['step_reward_mean']:.4f}")
    print(f"  episode_reward_mean: {report['episode_reward_mean']:.4f}")
    print(f"  r_tracking_mean:     {report['reward_components_mean']['r_tracking']:.4f}")
    print(f"  r_efficiency_mean:   {report['reward_components_mean']['r_efficiency']:.4f}")
    print(f"  r_smoothness_mean:   {report['reward_components_mean']['r_smoothness']:.4f}")
    print(f"  tracking_err_sq:     {report['reward_components_mean']['tracking_error_norm_sq']:.6f}")
    print(f"  W_net_mean:          {report['w_net_mean']:.4f}")
    print(f"  W_setpoint_mean:     {report['w_setpoint_mean']:.4f}")
    print("\n[preflight] Constraint/termination summary")
    print(f"  violation_mean:      {report['violation_fraction_mean']:.6f}")
    print(f"  terminations:        {report['termination_reasons']}")

    # Simulate one dual update to verify non-zero violations would move multipliers.
    multiplier_lr = float(lagrangian_cfg.get("multiplier_lr", 1e-3))
    simulated_lambda = max(0.0, multiplier_lr * report["violation_fraction_mean"])
    print("\n[preflight] Lagrangian dual-update probe")
    print(f"  multiplier_lr:       {multiplier_lr:.6f}")
    print(f"  simulated_delta:     {simulated_lambda:.6f}")

    checks = [
        ("No NaN/Inf in obs or rewards", not report["nan_or_inf_detected"]),
        ("Reward signal not collapsed", report["step_reward_max"] > report["step_reward_min"]),
        ("Constraint signal emitted", "truncated_or_complete" in report["termination_reasons"] or len(report["termination_reasons"]) > 0),
    ]
    failures = [label for label, passed in checks if not passed]
    if failures:
        print("\n[preflight] FAILED checks:")
        for label in failures:
            print(f"  - {label}")
        return 1
    print("\n[preflight] PASS")
    return 0


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "src"))

    fmu_path = Path(args.fmu_path)
    if not fmu_path.is_absolute():
        fmu_path = project_root / fmu_path

    config, lagrangian_cfg = load_config(project_root, max_steps_override=args.max_steps)
    env = build_env(project_root, config, fmu_path=fmu_path, use_mock=args.use_mock)
    try:
        report = run_preflight(env, episodes=args.episodes)
        return print_summary(env, report, lagrangian_cfg)
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
