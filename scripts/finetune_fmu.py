#!/usr/bin/env python3
"""Stage 4: Fine-tune a selected checkpoint on the real FMU."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune PPO checkpoint on FMU")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to RULE-C4 checkpoint JSON metadata",
    )
    parser.add_argument(
        "--fmu-path",
        type=str,
        default="artifacts/fmu_build/SCO2RecuperatedCycle.fmu",
        help="Path to compiled FMU",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/finetune_fmu.yaml",
        help="Fine-tuning YAML config path",
    )
    parser.add_argument(
        "--finetune-steps",
        type=int,
        default=None,
        help="Optional override for fine_tuning.finetune_steps",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=None,
        help="Optional override for fine_tuning.finetune_lr",
    )
    return parser.parse_args()


def _resolve(project_root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root / p


def _load_env_config(project_root: Path) -> dict:
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
        "episode_max_steps": env_cfg["episode"].get("max_steps", 720),
        "reward": env_cfg["reward"],
        "safety": {
            "T_compressor_inlet_min": hard.get("compressor_inlet_temp_min_c", 32.2),
            "surge_margin_min": hard.get("surge_margin_main_min", 0.05),
        },
        "setpoint": {"W_net": 10.0},
    }


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "src"))

    checkpoint_path = _resolve(project_root, args.checkpoint)
    fmu_path = _resolve(project_root, args.fmu_path)
    config_path = _resolve(project_root, args.config)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1
    if not fmu_path.exists():
        print(f"FMU not found: {fmu_path}", file=sys.stderr)
        return 1
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    cfg_raw = yaml.safe_load(config_path.read_text())
    fine_cfg = dict(cfg_raw.get("fine_tuning", {}))
    if args.finetune_steps is not None:
        fine_cfg["finetune_steps"] = int(args.finetune_steps)
    if args.finetune_lr is not None:
        fine_cfg["finetune_lr"] = float(args.finetune_lr)

    from sco2rl.environment.sco2_env import SCO2FMUEnv
    from sco2rl.simulation.fmu.fmpy_adapter import FMPyAdapter
    from sco2rl.training.fine_tuner import FineTuner

    env_cfg = _load_env_config(project_root)

    def env_factory() -> SCO2FMUEnv:
        adapter = FMPyAdapter(
            fmu_path=str(fmu_path),
            obs_vars=env_cfg["obs_vars"],
            action_vars=env_cfg["action_vars"],
            instance_name="finetune_instance",
            scale_offset=FMPyAdapter.default_scale_offset(),
        )
        adapter.initialize(
            start_time=0.0,
            stop_time=env_cfg["episode_max_steps"] * env_cfg["step_size"],
            step_size=env_cfg["step_size"],
        )
        return SCO2FMUEnv(fmu=adapter, config=env_cfg)

    fine_tuner = FineTuner(env_factory=env_factory, config=fine_cfg)
    result = fine_tuner.finetune(str(checkpoint_path))
    print("[finetune_fmu] Done")
    print(f"  total_timesteps:   {result['total_timesteps']}")
    print(f"  final_mean_reward: {result['final_mean_reward']:.6f}")
    print(f"  checkpoint_path:   {result['checkpoint_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
