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
    """Merge environment, training, curriculum, and safety YAML configs."""
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

    # Flatten into single config dict (FMUTrainer expects flat config)
    config = {}
    config.update(env_cfg.get("environment", env_cfg))   # support nested or flat
    config.update(ppo_cfg.get("training", ppo_cfg))
    config["curriculum"] = curriculum_cfg.get("curriculum", curriculum_cfg)
    config["safety"].update(safety_cfg.get("safety", safety_cfg))
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

    # Build obs_vars and action_vars from config
    obs_vars = [v["fmu_var"] for v in config["obs_vars"]] if isinstance(
        config["obs_vars"][0], dict
    ) else config["obs_vars"]
    action_vars = [v["fmu_var"] for v in config["action_vars"]] if isinstance(
        config["action_vars"][0], dict
    ) else config["action_vars"]

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
        print(f"[train_fmu] Resuming from checkpoint: {args.resume}")
        trainer._checkpoint_mgr.load(args.resume, trainer._policy, trainer._env)

    print(f"[train_fmu] Starting training...")
    policy = trainer.train(total_timesteps=args.total_timesteps)

    print(f"[train_fmu] Training complete. Evaluating...")
    results = trainer.evaluate(n_episodes=10)
    print(f"[train_fmu] Evaluation results:")
    print(f"  mean_reward:    {results['mean_reward']:.4f}")
    print(f"  violation_rate: {results['violation_rate']:.4f}")
    print(f"[train_fmu] Done. Checkpoints in: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
