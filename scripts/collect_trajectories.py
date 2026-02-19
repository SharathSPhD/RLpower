"""Stage 3: Collect FMU trajectories via Latin Hypercube Sampling for FNO training.

Usage (inside Docker):
    PYTHONPATH=src python scripts/collect_trajectories.py \
        --n-samples 75000 \
        --output artifacts/trajectories/lhs_75k.h5 \
        --fmu-path artifacts/fmu_build/SCO2RecuperatedCycle.fmu \
        --n-workers 8

    # Small test run:
    PYTHONPATH=src python scripts/collect_trajectories.py \
        --n-samples 100 --output /tmp/lhs_test.h5 \
        --fmu-path artifacts/fmu_build/SCO2RecuperatedCycle.fmu \
        --n-workers 4 --verbose 1

Produces an HDF5 file at --output with datasets:
    states:   (N, T, obs_dim)    float32
    actions:  (N, T-1, act_dim)  float32
    metadata: (N, 3)             float32  [T_exhaust_K, mdot_exhaust_kgs, W_setpoint_MW]

Parallelism: n_workers independent FMU instances collect batches concurrently
via multiprocessing.Pool with spawn start method (CUDA-safe).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path


# Worker at module level — required for spawn-based multiprocessing serialization
def _collect_batch_worker(args: tuple) -> list[dict]:
    """Spawned worker: creates its own FMU + SCO2FMUEnv + TrajectoryCollector.

    Fully self-contained. Returns list of trajectory dicts.
    args: (src_path, fmu_path_str, obs_vars, action_vars,
           env_config, collector_config, batch_samples, seed)
    """
    (src_path, fmu_path_str, obs_vars, action_vars,
     env_config, collector_config, batch_samples, seed) = args

    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from sco2rl.simulation.fmu.fmpy_adapter import FMPyAdapter
    from sco2rl.environment.sco2_env import SCO2FMUEnv
    from sco2rl.surrogate.trajectory_collector import TrajectoryCollector

    episode_max_steps = env_config["episode_max_steps"]
    fmu = FMPyAdapter(
        fmu_path=fmu_path_str,
        obs_vars=obs_vars,
        action_vars=action_vars,
        instance_name=f"worker_{seed}",
        scale_offset=FMPyAdapter.default_scale_offset(),
    )
    fmu.initialize(
        start_time=0.0,
        stop_time=episode_max_steps * 5.0,
        step_size=5.0,
    )
    env = SCO2FMUEnv(fmu=fmu, config=env_config)
    collector = TrajectoryCollector(
        env=env,
        config=collector_config,
        seed=seed,
        raw_obs_dim=len(obs_vars),
    )
    try:
        return collector.collect_batch(batch_samples)
    finally:
        env.close()


def parse_args():
    p = argparse.ArgumentParser(description="Stage 3: Parallel LHS trajectory collection")
    p.add_argument("--n-samples", type=int, default=75_000)
    p.add_argument("--output", type=str, default="artifacts/trajectories/lhs_75k.h5")
    p.add_argument("--fmu-path", type=str,
                   default="artifacts/fmu_build/SCO2RecuperatedCycle.fmu")
    p.add_argument("--n-workers", type=int, default=8,
                   help="Parallel FMU worker processes (default: 8)")
    p.add_argument("--batch-size", type=int, default=100,
                   help="Trajectories per worker task (default: 100)")
    p.add_argument(
        "--episode-max-steps",
        type=int,
        default=None,
        help="Override episode length for data collection (default: env.yaml value)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", type=int, default=0)
    return p.parse_args()


def load_configs(project_root: Path) -> dict:
    """Load and parse environment + safety YAML configs (mirrors train_fmu.py)."""
    import yaml

    env_cfg = yaml.safe_load(
        (project_root / "configs/environment/env.yaml").read_text()
    )
    safety_cfg = yaml.safe_load(
        (project_root / "configs/safety/constraints.yaml").read_text()
    )

    obs_section = env_cfg["observation"]
    obs_vars_raw = obs_section["variables"]
    fmu_obs_vars_raw = [v for v in obs_vars_raw if v.get("fmu_var") is not None]
    obs_vars = [v["fmu_var"] for v in fmu_obs_vars_raw]
    obs_bounds = {v["fmu_var"]: (v["min"], v["max"]) for v in fmu_obs_vars_raw}

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

    hard = safety_cfg.get("hard_constraints", {})
    safety = {
        "T_compressor_inlet_min": hard.get(
            "compressor_inlet_temp_min_c",
            hard.get("T_compressor_inlet_min_c", 32.2),
        ),
        "surge_margin_min": hard.get(
            "surge_margin_main_min",
            hard.get("surge_margin_min_fraction", 0.05),
        ),
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
    src_path = str(project_root / "src")
    sys.path.insert(0, src_path)

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
    cfg = load_configs(project_root)

    obs_vars = cfg["obs_vars"]
    obs_bounds = cfg["obs_bounds"]
    action_vars = cfg["action_vars"]
    action_config = cfg["action_config"]
    episode_max_steps = int(
        args.episode_max_steps
        if args.episode_max_steps is not None
        else cfg["episode"].get("max_steps", 720)
    )
    history_steps = cfg["obs_section"].get("history_steps", 5)

    print(f"[collect_trajectories] FMU: {fmu_path}")
    print(f"[collect_trajectories] obs={len(obs_vars)}x{history_steps}, "
          f"act={len(action_vars)}, T={episode_max_steps}")
    print(f"[collect_trajectories] {args.n_samples:,} trajectories, "
          f"{args.n_workers} workers, batch={args.batch_size} → {output_path}")

    from sco2rl.surrogate.lhs_sampler import LatinHypercubeSampler
    from sco2rl.surrogate.trajectory_dataset import TrajectoryDataset

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

    collector_config = {
        "trajectory_length_steps": episode_max_steps,
        "action_perturbation": {
            "type": "random_walk",
            "step_std": 0.05,
            "clip": 0.2,
        },
    }

    lhs_config = {
        "parameter_ranges": {
            "T_exhaust_K":       {"min": 473.0,  "max": 1473.0},
            "mdot_exhaust_kgs":  {"min": 10.0,   "max": 50.0},
            "W_setpoint_MW":     {"min": 7.0,    "max": 12.0},
        }
    }
    sampler = LatinHypercubeSampler(config=lhs_config, seed=args.seed)
    all_samples = sampler.sample(n=args.n_samples)

    # Split into batches — each becomes one worker task
    bs = args.batch_size
    batches = [all_samples[i: i + bs] for i in range(0, args.n_samples, bs)]
    n_batches = len(batches)

    worker_args = [
        (src_path, str(fmu_path), obs_vars, action_vars,
         env_config, collector_config, batches[i], args.seed + i)
        for i in range(n_batches)
    ]

    print(f"[collect_trajectories] Dispatching {n_batches} tasks to {args.n_workers} workers...")

    report_interval = max(1, args.n_samples // 20)
    collected = 0

    ctx = mp.get_context("spawn")
    with TrajectoryDataset(filepath=str(output_path), mode="w") as dataset:
        with ctx.Pool(processes=args.n_workers) as pool:
            for trajectories in pool.imap_unordered(_collect_batch_worker, worker_args):
                dataset.write_batch(trajectories)
                prev = collected
                collected += len(trajectories)
                if args.verbose >= 1:
                    if prev // report_interval < collected // report_interval or collected >= args.n_samples:
                        pct = 100.0 * collected / args.n_samples
                        print(f"[collect_trajectories]   {collected:6d}/{args.n_samples} ({pct:.1f}%)")

    print(f"[collect_trajectories] Done. {collected} trajectories → {output_path}")
    print(f"[collect_trajectories] File size: {output_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
