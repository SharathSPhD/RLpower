#!/usr/bin/env python3
"""Fail-fast surrogate pre-flight: small dataset + short FNO run.

This script guards against expensive 75K trajectory collection when basic
surrogate assumptions are broken.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run surrogate pre-flight checks")
    parser.add_argument("--fmu-path", type=str, required=True, help="Path to compiled FMU")
    parser.add_argument(
        "--dataset",
        type=str,
        default="artifacts/trajectories/preflight_100.h5",
        help="Temporary dataset output path",
    )
    parser.add_argument("--n-samples", type=int, default=100, help="Pre-flight trajectories")
    parser.add_argument("--n-workers", type=int, default=4, help="Collector worker count")
    parser.add_argument("--batch-size", type=int, default=10, help="Collector batch size")
    parser.add_argument(
        "--episode-max-steps",
        type=int,
        default=120,
        help="Short episode length for pre-flight trajectory collection",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Short FNO training epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    return parser.parse_args()


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print(f"[preflight_surrogate] $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    fmu_path = Path(args.fmu_path)
    if not fmu_path.is_absolute():
        fmu_path = project_root / fmu_path
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    collect_cmd = [
        sys.executable,
        "scripts/collect_trajectories.py",
        "--n-samples",
        str(args.n_samples),
        "--output",
        str(dataset_path),
        "--fmu-path",
        str(fmu_path),
        "--n-workers",
        str(args.n_workers),
        "--batch-size",
        str(args.batch_size),
        "--episode-max-steps",
        str(args.episode_max_steps),
        "--verbose",
        "1",
    ]
    train_cmd = [
        sys.executable,
        "scripts/train_surrogate.py",
        "--dataset",
        str(dataset_path),
        "--fno-epochs",
        str(args.epochs),
        "--device",
        args.device,
        "--skip-rl",
        "--verbose",
        "1",
    ]

    try:
        run_cmd(collect_cmd, cwd=project_root)
        run_cmd(train_cmd, cwd=project_root)
    except RuntimeError as exc:
        print(f"[preflight_surrogate] FAILED: {exc}")
        return 1

    print("[preflight_surrogate] PASS")
    print(f"[preflight_surrogate] Dataset: {dataset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
