#!/usr/bin/env python3
"""Run complete control system analysis across curriculum phases.

Executes step-response and frequency-response experiments for both PID
and RL controllers, then saves results to JSON for notebook consumption.

Usage
-----
# MockFMU only (Colab-compatible, fast):
python scripts/run_control_analysis.py --use-mock

# With a real FMU and pre-trained RL checkpoint:
python scripts/run_control_analysis.py \\
    --fmu-path artifacts/fmu/SCO2_WHR.fmu \\
    --checkpoint artifacts/checkpoints/run01/final \\
    --phases 0 1 2 3 \\
    --n-seeds 5 \\
    --output-dir data/

# Phases only (skip frequency response for speed):
python scripts/run_control_analysis.py --use-mock --no-frequency
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make src importable when run directly (project root / src)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Control system response analysis")
    p.add_argument(
        "--use-mock", action="store_true",
        help="Use MockFMU (no real FMU required; Colab-compatible)",
    )
    p.add_argument(
        "--fmu-path", type=str,
        default="artifacts/fmu_build/SCO2RecuperatedCycle.fmu",
        help="Path to compiled FMU (ignored with --use-mock)",
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to RL checkpoint for RL vs PID comparison",
    )
    p.add_argument(
        "--phases", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6],
        help="Curriculum phases to evaluate (space-separated)",
    )
    p.add_argument(
        "--n-seeds", type=int, default=3,
        help="Number of seeds per scenario for averaging",
    )
    p.add_argument(
        "--output-dir", type=str, default="data",
        help="Directory for output JSON files",
    )
    p.add_argument(
        "--no-frequency", action="store_true",
        help="Skip frequency response analysis (faster)",
    )
    p.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print progress messages",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = _PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    from sco2rl.analysis.scenario_runner import (
        ScenarioRunner,
        ControlScenario,
        build_mock_env,
        build_mock_pid,
    )

    # ── Build PID controller ──────────────────────────────────────────────────
    pid = build_mock_pid()
    print(f"PID controller: {pid.name}")

    # ── Build RL controller (optional) ───────────────────────────────────────
    rl = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            from sco2rl.control.rl_controller import RLController
            try:
                rl = RLController.from_checkpoint(ckpt_path, controller_name="RL")
                print(f"RL checkpoint loaded: {ckpt_path}")
            except Exception as exc:
                print(f"Warning: could not load RL checkpoint: {exc}")
        else:
            print(f"Warning: checkpoint not found: {ckpt_path}")

    # ── Build env factory ─────────────────────────────────────────────────────
    if args.use_mock:
        env_factory = build_mock_env
        print("Using MockFMU (--use-mock)")
    else:
        fmu_path = Path(args.fmu_path)
        if not fmu_path.is_absolute():
            fmu_path = _PROJECT_ROOT / args.fmu_path
        if not fmu_path.exists():
            print(f"Error: FMU not found: {fmu_path}")
            print("Use --use-mock for MockFMU-based analysis.")
            return 1
        env_factory = _make_fmu_factory(fmu_path)
        print(f"Using real FMU: {fmu_path}")

    # ── Run analysis ──────────────────────────────────────────────────────────
    runner = ScenarioRunner(
        n_seeds=args.n_seeds,
        dt=5.0,
        verbose=args.verbose,
    )

    scenarios = [
        ControlScenario.STEP_LOAD_UP_20,
        ControlScenario.STEP_LOAD_DOWN_20,
        ControlScenario.LOAD_REJECTION_50,
    ]

    print(f"\nRunning control analysis: {len(args.phases)} phases × {len(scenarios)} scenarios")
    t0 = time.time()

    results = runner.run_all(
        env_factory=env_factory,
        pid_policy=pid,
        rl_policy=rl,
        phases=args.phases,
        scenarios=scenarios,
        run_frequency=not args.no_frequency,
    )

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s  ({len(results)} result objects)")

    # ── Save results ──────────────────────────────────────────────────────────
    all_phases_path = output_dir / "control_analysis_all_phases.json"
    ScenarioRunner.save(results, all_phases_path)

    # Also save Phase 0 separately for quick notebook loading
    phase0_results = [r for r in results if r.phase == 0]
    if phase0_results:
        phase0_path = output_dir / "control_analysis_phase0.json"
        ScenarioRunner.save(phase0_results, phase0_path)

    # ── Print summary table ───────────────────────────────────────────────────
    _print_summary(results)

    return 0


def _make_fmu_factory(fmu_path: Path):
    """Build an env factory using the real FMU."""
    import yaml

    project_root = fmu_path.resolve().parent.parent.parent  # heuristic

    def _factory():
        from sco2rl.environment.sco2_env import SCO2FMUEnv
        from sco2rl.simulation.fmu.fmpy_adapter import FMPyAdapter

        env_yaml = project_root / "configs" / "environment" / "env.yaml"
        safety_yaml = project_root / "configs" / "safety" / "constraints.yaml"
        env_cfg_raw = yaml.safe_load(env_yaml.read_text())
        safety_cfg_raw = yaml.safe_load(safety_yaml.read_text())

        obs_vars_raw = [
            v for v in env_cfg_raw["observation"]["variables"]
            if v.get("fmu_var") is not None
        ]
        obs_vars = [v["fmu_var"] for v in obs_vars_raw]
        obs_bounds = {v["fmu_var"]: (v["min"], v["max"]) for v in obs_vars_raw}
        act_vars_raw = env_cfg_raw["action"]["variables"]
        action_vars = [v["fmu_var"] for v in act_vars_raw]
        action_config = {
            v["fmu_var"]: {
                "min": v["physical_min"],
                "max": v["physical_max"],
                "rate": v.get("rate_limit_per_step", 1.0),
            }
            for v in act_vars_raw
        }
        hard = safety_cfg_raw.get("hard_constraints", {})
        env_config = {
            "obs_vars": obs_vars,
            "obs_bounds": obs_bounds,
            "action_vars": action_vars,
            "action_config": action_config,
            "history_steps": env_cfg_raw["observation"].get("history_steps", 5),
            "step_size": 5.0,
            "episode_max_steps": 300,
            "reward": env_cfg_raw["reward"],
            "safety": {
                "T_compressor_inlet_min": hard.get("compressor_inlet_temp_min_c", 32.2),
                "surge_margin_min": hard.get("surge_margin_main_min", 0.05),
            },
            "setpoint": {"W_net": 10.0},
        }
        adapter = FMPyAdapter(
            fmu_path=str(fmu_path),
            obs_vars=obs_vars,
            action_vars=action_vars,
            scale_offset=FMPyAdapter.default_scale_offset(),
        )
        return SCO2FMUEnv(fmu=adapter, config=env_config)

    return _factory


def _print_summary(results) -> None:
    """Print a concise summary table of step-response metrics."""
    print("\n─── Step Response Summary ─────────────────────────────────────────")
    print(f"{'Phase':>5}  {'Scenario':<24}  {'Controller':<14}  "
          f"{'Settle(s)':>9}  {'Overshoot%':>10}  {'IAE':>8}")
    print("─" * 80)

    for r in results:
        for step_res, label in [(r.pid_step, "PID"), (r.rl_step, "RL")]:
            if step_res is None:
                continue
            print(
                f"{r.phase:>5}  {r.scenario:<24}  {label:<14}  "
                f"{step_res.settling_time_s:>9.1f}  "
                f"{step_res.overshoot_pct:>10.1f}  "
                f"{step_res.iae:>8.2f}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
