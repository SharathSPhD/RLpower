"""Stage 3: Collect FMU trajectories via Latin Hypercube Sampling for FNO training.

Usage (inside Docker):
    PYTHONPATH=src python scripts/collect_trajectories.py \
        --n-samples 75000 \
        --output artifacts/trajectories/lhs_75k.h5 \
        --fmu-path artifacts/fmu_build/SCO2RecuperatedCycle.fmu

    # Small test run:
    PYTHONPATH=src python scripts/collect_trajectories.py \
        --n-samples 100 \
        --output /tmp/lhs_test.h5 \
        --fmu-path artifacts/fmu_build/SCO2RecuperatedCycle.fmu \
        --verbose 1

Produces an HDF5 file at --output with datasets:
    states:   (N, T, obs_dim)    float32  -- flattened obs history per step
    actions:  (N, T-1, act_dim)  float32  -- normalized action per step
    metadata: (N, 3)             float32  -- [T_exhaust_K, mdot_exhaust_kgs, W_setpoint_MW]
where N = n_samples, T = trajectory_length_steps (= episode_max_steps).

LHS parameters label WHR operating conditions; trajectory diversity comes from
random-walk action perturbation applied per step.
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
    p.add_argument("--batch-size", type=int, default=500,
                   help="Trajectories per write batch (default: 500)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for LHS sampling (default: 42)")
    p.add_argument("--verbose", type=int, default=0,
                   help="Verbosity: 0=silent, 1=progress")
    return p.parse_args()


def load_configs(project_root: Path) -> dict:
    """Load and parse environment + safety YAML configs.

    Reuses the same parsing logic as train_fmu.py.
    """
    import yaml

    env_cfg = yaml.safe_load(
        (project_root / "configs/environment/env.yaml").read_text()
    )
    safety_cfg = yaml.safe_load(
        (project_root / "configs/safety/constraints.yaml").read_text()
    )

    # Observation vars — filter derived vars with fmu_var: null
    obs_section = env_cfg["observation"]
    obs_vars_raw = obs_section["variables"]
    fmu_obs_vars_raw = [v for v in obs_vars_raw if v.get("fmu_var") is not None]
    obs_vars = [v["fmu_var"] for v in fmu_obs_vars_raw]
    obs_bounds = {v["fmu_var"]: (v["min"], v["max"]) for v in fmu_obs_vars_raw}

    # Action vars
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

    # Safety constraints
    hard = safety_cfg.get("hard_constraints", {})
    safety = {
        "T_compressor_inlet_min": hard.get("T_compressor_inlet_min_c", 32.2),
        "surge_margin_min": hard.get("surge_margin_min_fraction", 0.05),
    }

    episode = env_cfg.get("episode", {})
    return {
        "obs_vars": obs_vars,
        "obs_bounds": obs_bounds,
        "obs_section": obs_section,
        "action_vars": action_vars,
        "action_config": action_config,
        "safety": safety,
        "episode": episode,
        "reward": env_cfg.get("reward", {}),
    }


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

    print(f"[collect_trajectories] Loading configs from {project_root}/configs/")
    cfg = load_configs(project_root)

    obs_vars = cfg["obs_vars"]
    obs_bounds = cfg["obs_bounds"]
    action_vars = cfg["action_vars"]
    action_config = cfg["action_config"]
    episode_max_steps = cfg["episode"].get("max_steps", 720)
    history_steps = cfg["obs_section"].get("history_steps", 5)

    print(f"[collect_trajectories] FMU: {fmu_path}")
    print(f"[collect_trajectories] obs_dim={len(obs_vars)}, action_dim={len(action_vars)}, "
          f"episode_steps={episode_max_steps}")
    print(f"[collect_trajectories] Collecting {args.n_samples:,} trajectories → {output_path}")

    from sco2rl.simulation.fmu.fmpy_adapter import FMPyAdapter
    from sco2rl.environment.sco2_env import SCO2FMUEnv
    from sco2rl.surrogate.lhs_sampler import LatinHypercubeSampler
    from sco2rl.surrogate.trajectory_collector import TrajectoryCollector
    from sco2rl.surrogate.trajectory_dataset import TrajectoryDataset

    # Build SCO2FMUEnv wrapping a single FMPyAdapter instance
    env_config = {
        "obs_vars": obs_vars,
        "obs_bounds": obs_bounds,
        "action_vars": action_vars,
        "action_config": action_config,
        "history_steps": history_steps,
        "step_size": 5.0,
        "episode_max_steps": episode_max_steps,
        "reward": cfg["reward"],
        "safety": cfg["safety"],
        "setpoint": {"W_net": 10.0},
    }

    fmu = FMPyAdapter(
        fmu_path=str(fmu_path),
        obs_vars=obs_vars,
        action_vars=action_vars,
        instance_name="collector_instance",
        scale_offset=FMPyAdapter.default_scale_offset(),
    )
    fmu.initialize(
        start_time=0.0,
        stop_time=episode_max_steps * 5.0,
        step_size=5.0,
    )
    env = SCO2FMUEnv(fmu=fmu, config=env_config)

    # TrajectoryCollector: random-walk action perturbation over full episode length
    collector_config = {
        "trajectory_length_steps": episode_max_steps,
        "action_perturbation": {
            "type": "random_walk",
            "step_std": 0.05,   # std of per-step action perturbation in [-1,1] space
            "clip": 0.2,        # max single-step change in normalized action
        },
    }
    collector = TrajectoryCollector(env=env, config=collector_config, seed=args.seed)

    # LHS sampler: 3 WHR operating condition parameters stored as metadata per trajectory.
    # These label the operating point; trajectory diversity comes from random-walk actions.
    lhs_config = {
        "parameter_ranges": {
            "T_exhaust_K": {"min": 473.0, "max": 1473.0},    # 200–1200°C (K)
            "mdot_exhaust_kgs": {"min": 10.0, "max": 50.0},  # kg/s
            "W_setpoint_MW": {"min": 7.0, "max": 12.0},      # MW
        }
    }
    sampler = LatinHypercubeSampler(config=lhs_config, seed=args.seed)
    all_samples = sampler.sample(n=args.n_samples)  # (n_samples, 3)

    print(f"[collect_trajectories] LHS sampling complete. Starting episode collection...")

    # Collect and write in batches to avoid accumulating large in-memory arrays
    batch_size = args.batch_size
    n_batches = (args.n_samples + batch_size - 1) // batch_size
    collected = 0
    report_interval = max(1, args.n_samples // 20)  # report every 5%

    with TrajectoryDataset(filepath=str(output_path), mode="w") as dataset:
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, args.n_samples)
            batch_samples = all_samples[start:end]  # (<=batch_size, 3)

            trajectories = collector.collect_batch(batch_samples)
            dataset.write_batch(trajectories)
            collected += len(trajectories)

            if args.verbose >= 1 and (collected % report_interval < batch_size
                                      or collected == args.n_samples):
                pct = 100.0 * collected / args.n_samples
                print(f"[collect_trajectories]   {collected:6d}/{args.n_samples} ({pct:.1f}%)")

    env.close()

    print(f"[collect_trajectories] Done. {collected} trajectories saved to {output_path}")
    print(f"[collect_trajectories] File size: {output_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
