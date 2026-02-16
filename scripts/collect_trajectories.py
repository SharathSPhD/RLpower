"""Stage 3: Collect FMU trajectories via Latin Hypercube Sampling for FNO training.

Usage (inside Docker):
    PYTHONPATH=src python scripts/collect_trajectories.py \
        --n-samples 75000 \
        --output artifacts/trajectories/lhs_75k.h5 \
        --fmu-path artifacts/fmu_build/SCO2RecuperatedCycle.fmu

    # Small test run:
    PYTHONPATH=src python scripts/collect_trajectories.py \
        --n-samples 500 \
        --output /tmp/lhs_test.h5 \
        --fmu-path artifacts/fmu_build/SCO2RecuperatedCycle.fmu \
        --verbose 1

Produces an HDF5 file at --output with datasets:
    observations: (N, T, obs_dim) float32
    actions:      (N, T, action_dim) float32
    rewards:      (N, T) float32
    terminals:    (N, T) bool
where N = n_samples, T = max episode length.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Stage 3: LHS trajectory collection")
    p.add_argument("--n-samples", type=int, default=75_000,
                   help="Number of trajectories to collect (default: 75_000)")
    p.add_argument("--output", type=str,
                   default="artifacts/trajectories/lhs_75k.h5",
                   help="Output HDF5 file path")
    p.add_argument("--fmu-path", type=str,
                   default="artifacts/fmu_build/SCO2RecuperatedCycle.fmu",
                   help="Path to compiled .fmu file")
    p.add_argument("--n-envs", type=int, default=4,
                   help="Parallel FMU instances for collection (default: 4)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for LHS sampling (default: 42)")
    p.add_argument("--verbose", type=int, default=0,
                   help="Verbosity: 0=silent, 1=progress")
    return p.parse_args()


def load_env_config(project_root: Path) -> dict:
    """Load environment and surrogate configs for trajectory collection."""
    import yaml

    env_cfg = yaml.safe_load(
        (project_root / "configs/environment/env.yaml").read_text()
    )
    surrogate_cfg = yaml.safe_load(
        (project_root / "configs/surrogate/surrogate.yaml").read_text()
    )
    config = {}
    config.update(env_cfg.get("environment", env_cfg))
    config["surrogate"] = surrogate_cfg.get("surrogate", surrogate_cfg)
    return config


def main():
    args = parse_args()

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / "src"))

    fmu_path = Path(args.fmu_path)
    if not fmu_path.is_absolute():
        fmu_path = project_root / fmu_path

    if not fmu_path.exists():
        print(f"ERROR: FMU not found at {fmu_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[collect_trajectories] Loading configs...")
    config = load_env_config(project_root)

    print(f"[collect_trajectories] FMU: {fmu_path}")
    print(f"[collect_trajectories] Collecting {args.n_samples:,} trajectories → {output_path}")

    from sco2rl.simulation.fmu.fmpy_adapter import FMPyAdapter
    from sco2rl.surrogate.lhs_sampler import LatinHypercubeSampler
    from sco2rl.surrogate.trajectory_collector import TrajectoryCollector
    from sco2rl.surrogate.trajectory_dataset import TrajectoryDataset

    # Build obs_vars and action_vars from config
    obs_vars = [v["fmu_var"] for v in config["obs_vars"]] if isinstance(
        config["obs_vars"][0], dict
    ) else config["obs_vars"]
    action_vars = [v["fmu_var"] for v in config["action_vars"]] if isinstance(
        config["action_vars"][0], dict
    ) else config["action_vars"]

    # Action bounds for LHS sampling
    action_config = config.get("action_config", {})
    action_bounds = {
        v: (action_config[v]["min"], action_config[v]["max"])
        for v in action_vars
        if v in action_config
    }

    # FMU factory for trajectory collection
    def fmu_factory() -> FMPyAdapter:
        adapter = FMPyAdapter(
            fmu_path=str(fmu_path),
            obs_vars=obs_vars,
            action_vars=action_vars,
            instance_name="collector_instance",
            scale_offset=FMPyAdapter.default_scale_offset(),
        )
        adapter.initialize(
            start_time=0.0,
            stop_time=config.get("episode_max_steps", 720) * config.get("step_size", 5.0),
            step_size=config.get("step_size", 5.0),
        )
        return adapter

    # LHS sampler generates action sequences
    sampler = LatinHypercubeSampler(
        action_bounds=action_bounds,
        action_vars=action_vars,
        seed=args.seed,
    )

    # Trajectory collector
    collector = TrajectoryCollector(
        fmu_factory=fmu_factory,
        obs_vars=obs_vars,
        action_vars=action_vars,
        step_size=float(config.get("step_size", 5.0)),
        episode_steps=int(config.get("episode_max_steps", 720)),
    )

    # Dataset in HDF5 append mode
    dataset = TrajectoryDataset(
        path=str(output_path),
        obs_dim=len(obs_vars),
        action_dim=len(action_vars),
        episode_steps=int(config.get("episode_max_steps", 720)),
    )

    print(f"[collect_trajectories] Starting collection with {args.n_envs} FMU instances...")
    collected = 0
    report_interval = max(1, args.n_samples // 20)  # report every 5%

    samples = sampler.sample(args.n_samples)
    for i, action_sequence in enumerate(samples):
        trajectory = collector.collect_trajectory(action_sequence)
        dataset.append(trajectory)
        collected += 1

        if args.verbose >= 1 and collected % report_interval == 0:
            pct = 100.0 * collected / args.n_samples
            print(f"[collect_trajectories]   {collected:6d}/{args.n_samples} ({pct:.1f}%)")

    print(f"[collect_trajectories] Done. {collected} trajectories saved to {output_path}")
    print(f"[collect_trajectories] File size: {output_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
