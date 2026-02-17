"""Stage 2: SB3 PPO training on real FMU with curriculum.

Usage (inside Docker):
    PYTHONPATH=src python scripts/train_fmu.py \
        --n-envs 8 \
        --total-timesteps 5000000 \
        --fmu-path artifacts/fmu_build/SCO2RecuperatedCycle.fmu

    # Mini validation run (no real FMU needed):
    PYTHONPATH=src python scripts/train_fmu.py \
        --n-envs 1 \
        --total-timesteps 10000 \
        --fmu-path artifacts/fmu_build/SCO2RecuperatedCycle.fmu \
        --verbose 1

Config files loaded:
    configs/environment/env.yaml
    configs/training/ppo_fmu.yaml
    configs/curriculum/curriculum.yaml
    configs/safety/constraints.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: FMU-based PPO training")
    p.add_argument("--n-envs", type=int, default=8,
                   help="Number of parallel FMU environments (default: 8)")
    p.add_argument("--total-timesteps", type=int, default=5_000_000,
                   help="Total training timesteps (default: 5_000_000)")
    p.add_argument("--fmu-path", type=str,
                   default="artifacts/fmu_build/SCO2RecuperatedCycle.fmu",
                   help="Path to compiled .fmu file")
    p.add_argument("--checkpoint-dir", type=str,
                   default="artifacts/checkpoints/fmu_direct",
                   help="Directory for RULE-C4 checkpoints")
    p.add_argument("--run-name", type=str, default="fmu_ppo",
                   help="Run identifier for checkpoint filenames")
    p.add_argument("--verbose", type=int, default=0,
                   help="Verbosity: 0=silent, 1=progress")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


def load_configs(project_root: Path) -> dict:
    """Merge environment, training, curriculum, and safety YAML configs.

    Produces a flat config dict matching FMUTrainer expectations.
    """
    import yaml

    env_cfg = yaml.safe_load(
        (project_root / "configs/environment/env.yaml").read_text()
    )
    ppo_cfg = yaml.safe_load(
        (project_root / "configs/training/ppo_fmu.yaml").read_text()
    )
    curriculum_cfg = yaml.safe_load(
        (project_root / "configs/curriculum/curriculum.yaml").read_text()
    )
    safety_cfg = yaml.safe_load(
        (project_root / "configs/safety/constraints.yaml").read_text()
    )

    # Parse obs_vars from env.yaml observation.variables
    # Filter out derived variables (fmu_var: null) — computed by env, not read from FMU
    obs_section = env_cfg["observation"]
    obs_vars_raw = obs_section["variables"]
    fmu_obs_vars_raw = [v for v in obs_vars_raw if v.get("fmu_var") is not None]
    obs_vars = [v["fmu_var"] for v in fmu_obs_vars_raw]
    obs_bounds = {v["fmu_var"]: (v["min"], v["max"]) for v in fmu_obs_vars_raw}

    # Parse action_vars from env.yaml action.variables
    act_section = env_cfg["action"]
    act_vars_raw = act_section["variables"]
    action_vars = [v["fmu_var"] for v in act_vars_raw]
    action_config = {
        v["fmu_var"]: {
            "min": v["physical_min"],
            "max": v["physical_max"],
            "rate": v.get("rate_limit_per_step", v.get("rate_limit", v.get("rate", 1.0))),
        }
        for v in act_vars_raw
    }

    # Safety bounds from constraints.yaml
    hard = safety_cfg.get("hard_constraints", {})
    safety = {
        "T_compressor_inlet_min": hard.get("T_compressor_inlet_min_c", 32.2),
        "surge_margin_min": hard.get("surge_margin_min_fraction", 0.05),
    }

    # Episode + normalization from env.yaml
    episode = env_cfg.get("episode", {})
    norm = env_cfg.get("normalization", {})

    # PPO from ppo_fmu.yaml
    ppo = ppo_cfg.get("ppo", {})
    network = ppo_cfg.get("network", {})
    training = ppo_cfg.get("training", {})

    # Lagrangian multiplier config
    lagrangian_cfg = safety_cfg.get("lagrangian", {})
    constraint_names = ["T_comp_min", "surge_margin_main"]

    config = {
        # Env
        "obs_vars": obs_vars,
        "obs_bounds": obs_bounds,
        "action_vars": action_vars,
        "action_config": action_config,
        "history_steps": obs_section.get("history_steps", 5),
        "step_size": 5.0,
        "episode_max_steps": episode.get("max_steps", 720),
        "reward": env_cfg.get("reward", {}),
        "safety": safety,
        "setpoint": {"W_net": 10.0},
        # Normalization
        "normalization": norm,
        # PPO
        "ppo": ppo,
        "network": network,
        # Constraints
        "constraint_names": constraint_names,
        "multiplier_lr": lagrangian_cfg.get("multiplier_lr", 1e-3),
        # Curriculum
        "curriculum": curriculum_cfg,
        # Checkpoint
        "checkpoint_freq": training.get("checkpoint_freq", 100_000),
    }
    return config


def main():
    args = parse_args()

    # Resolve project root (script may run from any CWD)
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / "src"))

    fmu_path = Path(args.fmu_path)
    if not fmu_path.is_absolute():
        fmu_path = project_root / fmu_path

    if not fmu_path.exists():
        print(f"ERROR: FMU not found at {fmu_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[train_fmu] Loading configs from {project_root}/configs/")
    config = load_configs(project_root)
    config["checkpoint_dir"] = args.checkpoint_dir
    config["run_name"] = args.run_name
    config["verbose"] = args.verbose

    print(f"[train_fmu] FMU path: {fmu_path}")
    print(f"[train_fmu] n_envs={args.n_envs}, total_timesteps={args.total_timesteps:,}")

    # Import here to keep startup fast
    from sco2rl.simulation.fmu.fmpy_adapter import FMPyAdapter
    from sco2rl.training.fmu_trainer import FMUTrainer

    # obs_vars and action_vars are already flat strings from load_configs()
    obs_vars = config["obs_vars"]
    action_vars = config["action_vars"]

    # FMU factory — called once per VecEnv worker
    def fmu_factory() -> FMPyAdapter:
        adapter = FMPyAdapter(
            fmu_path=str(fmu_path),
            obs_vars=obs_vars,
            action_vars=action_vars,
            instance_name="training_instance",
            scale_offset=FMPyAdapter.default_scale_offset(),
        )
        adapter.initialize(
            start_time=0.0,
            stop_time=config.get("episode_max_steps", 720) * config.get("step_size", 5.0),
            step_size=config.get("step_size", 5.0),
        )
        return adapter

    # Set up trainer
    trainer = FMUTrainer(config=config)
    trainer.setup(fmu_factory=fmu_factory, n_envs=args.n_envs)

    # Resume from checkpoint if requested
    if args.resume:
        from sco2rl.curriculum.phase import CurriculumPhase
        from sco2rl.training.lagrangian_ppo import LagrangianPPO
        print(f"[train_fmu] Resuming from checkpoint: {args.resume}")
        data = trainer._checkpoint_mgr.load(args.resume)
        # Restore policy weights + Lagrange multipliers (LagrangianPPO.load() reads
        # both the .zip and the companion _multipliers.pkl via CheckpointManager.save())
        trainer._policy = LagrangianPPO.load(data["model_path"], env=trainer._env)
        # Restore curriculum phase so scheduler picks up where training left off
        phase = CurriculumPhase(int(data["curriculum_phase"]))
        trainer._curriculum_callback.scheduler._phase = phase
        # VecNormalize stats were saved as null placeholder; fresh stats is acceptable
        # (policy weights are what matter for warm-starting)
        print(f"[train_fmu] Resumed from phase {phase.name} @ step {data['total_timesteps']:,}")

    print("[train_fmu] Starting training...")
    trainer.train(total_timesteps=args.total_timesteps)

    print("[train_fmu] Training complete. Evaluating...")
    results = trainer.evaluate(n_episodes=10)
    print("[train_fmu] Evaluation results:")
    print(f"  mean_reward:    {results['mean_reward']:.4f}")
    print(f"  violation_rate: {results['violation_rate']:.4f}")
    print(f"[train_fmu] Done. Checkpoints in: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
