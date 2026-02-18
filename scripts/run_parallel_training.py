#!/usr/bin/env python3
"""Launch CPU FMU and GPU surrogate training in parallel."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPU+GPU training paths in parallel")
    parser.add_argument("--fmu-path", type=str, required=True, help="Path to compiled FMU")
    parser.add_argument("--cpu-timesteps", type=int, default=200000, help="CPU FMU PPO timesteps")
    parser.add_argument("--gpu-timesteps", type=int, default=500000, help="GPU surrogate PPO timesteps")
    parser.add_argument("--fno-epochs", type=int, default=50, help="FNO epochs before surrogate PPO")
    parser.add_argument(
        "--allow-gpu-fidelity-fail",
        action="store_true",
        help="Allow GPU path to continue past fidelity gate failures (smoke/debug runs)",
    )
    parser.add_argument("--dataset", type=str, default="artifacts/trajectories/fno_training.h5")
    parser.add_argument("--n-envs-cpu", type=int, default=8)
    return parser.parse_args()


def _resolve(project_root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root / p


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    fmu_path = _resolve(project_root, args.fmu_path)
    dataset_path = _resolve(project_root, args.dataset)
    if not fmu_path.exists():
        print(f"FMU not found: {fmu_path}")
        return 1
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 1

    cpu_cmd = [
        sys.executable,
        "scripts/train_fmu.py",
        "--n-envs",
        str(args.n_envs_cpu),
        "--total-timesteps",
        str(args.cpu_timesteps),
        "--fmu-path",
        str(fmu_path),
        "--checkpoint-dir",
        "artifacts/checkpoints/fmu_parallel",
        "--run-name",
        "cpu_parallel",
    ]
    gpu_cmd = [
        sys.executable,
        "scripts/train_surrogate.py",
        "--dataset",
        str(dataset_path),
        "--rl-timesteps",
        str(args.gpu_timesteps),
        "--fno-epochs",
        str(args.fno_epochs),
        "--device",
        "cuda",
    ]
    if args.allow_gpu_fidelity_fail:
        gpu_cmd.append("--allow-fidelity-fail")

    print(f"[run_parallel_training] CPU cmd: {' '.join(cpu_cmd)}")
    print(f"[run_parallel_training] GPU cmd: {' '.join(gpu_cmd)}")

    cpu_proc = subprocess.Popen(cpu_cmd, cwd=project_root)
    gpu_proc = subprocess.Popen(gpu_cmd, cwd=project_root)
    cpu_rc = cpu_proc.wait()
    gpu_rc = gpu_proc.wait()

    print(f"[run_parallel_training] cpu_exit={cpu_rc}, gpu_exit={gpu_rc}")
    return 0 if (cpu_rc == 0 and gpu_rc == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
