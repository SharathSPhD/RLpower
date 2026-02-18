#!/usr/bin/env python3
"""Export a controller-ready FMU and report FMI I/O causality counts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export controlled FMU with controller interface")
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model/base_cycle.yaml",
        help="Model topology YAML",
    )
    parser.add_argument(
        "--fmu-config",
        type=str,
        default="configs/fmu/fmu_export.yaml",
        help="FMU export YAML",
    )
    parser.add_argument(
        "--controller-mode",
        type=str,
        default="rl_external",
        choices=["none", "rl_external", "pid_baseline"],
        help="Controller integration mode for generated model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/fmu_controlled",
        help="Output directory for generated .mo and .fmu",
    )
    return parser.parse_args()


def _resolve(project_root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "src"))

    model_cfg_path = _resolve(project_root, args.model_config)
    fmu_cfg_path = _resolve(project_root, args.fmu_config)
    output_dir = _resolve(project_root, args.output_dir)

    model_cfg = yaml.safe_load(model_cfg_path.read_text())
    fmu_cfg = yaml.safe_load(fmu_cfg_path.read_text())

    model_cfg["controller"] = {
        "mode": args.controller_mode,
        "n_commands": 4,
        "n_measurements": 14,
    }

    from sco2rl.physics.metamodel.builder import SCO2CycleBuilder
    from sco2rl.physics.compiler.fmu_exporter import FMUExporter
    from sco2rl.physics.compiler.omc_session import OMCSessionWrapper

    cycle = SCO2CycleBuilder.from_config(model_cfg).build()
    with OMCSessionWrapper(load_thermopower=False) as omc:
        exporter = FMUExporter(config=fmu_cfg, output_dir=output_dir, omc=omc)
        fmu_path = exporter.export(cycle)

    from fmpy import read_model_description

    md = read_model_description(str(fmu_path))
    n_inputs = 0
    n_outputs = 0
    for variable in md.modelVariables:
        if variable.causality == "input":
            n_inputs += 1
        elif variable.causality == "output":
            n_outputs += 1

    print("[export_controlled_fmu] Done")
    print(f"  FMU:      {fmu_path}")
    print(f"  inputs:   {n_inputs}")
    print(f"  outputs:  {n_outputs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
