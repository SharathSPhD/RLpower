#!/usr/bin/env python3
"""Cross-validate RL vs PID and export ONNX/TRT artifacts."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-validation and deployment export")
    parser.add_argument("--rl-checkpoint", type=str, required=True, help="Path to RULE-C4 checkpoint json")
    parser.add_argument("--fmu-path", type=str, required=True, help="Path to FMU")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--report-out", type=str, default="artifacts/reports/cross_validation_report.json")
    parser.add_argument("--onnx-out", type=str, default="artifacts/export/policy.onnx")
    parser.add_argument("--trt-out", type=str, default="artifacts/export/policy.plan")
    parser.add_argument("--skip-trt", action="store_true")
    parser.add_argument(
        "--all-phases",
        action="store_true",
        help="Evaluate RL vs PID across all curriculum phases",
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


def _pid_config(env_config: dict) -> dict:
    obs_vars = env_config["obs_vars"]
    action_vars = env_config["action_vars"]
    measurement_indices = {
        action_vars[0]: _find_obs_index(obs_vars, ["turbine.T_inlet_rt", "T_turbine_inlet"], 0),
        action_vars[1]: _find_obs_index(obs_vars, ["main_compressor.T_inlet_rt", "T_compressor_inlet"], 0),
        action_vars[2]: _find_obs_index(obs_vars, ["main_compressor.p_outlet", "P_high"], 0),
        action_vars[3]: _find_obs_index(obs_vars, ["precooler.T_outlet_rt", "T_precooler_outlet"], 0),
    }
    gains = {name: {"kp": 0.02, "ki": 0.001} for name in action_vars}
    return {
        "obs_vars": obs_vars,
        "action_vars": action_vars,
        "n_obs": len(obs_vars),
        "history_steps": env_config.get("history_steps", 1),
        "gains": gains,
        "setpoints": {"W_net": 10.0, "T_compressor_inlet": 33.0},
        "measurement_indices": measurement_indices,
        "dt": env_config.get("step_size", 5.0),
    }


def _find_obs_index(obs_vars: list[str], candidates: list[str], default: int = 0) -> int:
    for name in candidates:
        if name in obs_vars:
            return obs_vars.index(name)
    return default


def _load_curriculum_phases(project_root: Path) -> list[dict]:
    curriculum = yaml.safe_load((project_root / "configs/curriculum/curriculum.yaml").read_text())
    return list(curriculum.get("phases", []))


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "src"))

    ckpt_path = _resolve(project_root, args.rl_checkpoint)
    fmu_path = _resolve(project_root, args.fmu_path)
    report_out = _resolve(project_root, args.report_out)
    onnx_out = _resolve(project_root, args.onnx_out)
    trt_out = _resolve(project_root, args.trt_out)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return 1
    if not fmu_path.exists():
        print(f"FMU not found: {fmu_path}")
        return 1

    from sco2rl.environment.sco2_env import SCO2FMUEnv
    from sco2rl.simulation.fmu.fmpy_adapter import FMPyAdapter
    from sco2rl.training.lagrangian_ppo import LagrangianPPO
    from sco2rl.training.policy_evaluator import PolicyEvaluator
    from sco2rl.deployment.inference.pid_baseline import PIDBaseline
    from sco2rl.deployment.export.onnx_exporter import ONNXExporter
    from sco2rl.deployment.export.trt_exporter import TensorRTExporter

    ckpt = json.loads(ckpt_path.read_text())
    model_path = _resolve(project_root, ckpt["model_path"])
    env_cfg = _load_env_config(project_root)
    curriculum_phases = _load_curriculum_phases(project_root)
    adapter = FMPyAdapter(
        fmu_path=str(fmu_path),
        obs_vars=env_cfg["obs_vars"],
        action_vars=env_cfg["action_vars"],
        instance_name="cross_validate_eval",
        scale_offset=FMPyAdapter.default_scale_offset(),
    )
    env = SCO2FMUEnv(fmu=adapter, config=env_cfg)
    try:
        rl_model = LagrangianPPO.load(str(model_path), env=env)
        pid_model = PIDBaseline(_pid_config(env_cfg))
        evaluator = PolicyEvaluator(env, {"n_eval_episodes": args.episodes, "deterministic": True})
        phases_to_run: list[tuple[int, int]] = [(0, int(env_cfg["episode_max_steps"]))]
        if args.all_phases and curriculum_phases:
            phases_to_run = [
                (int(p.get("id", idx)), int(p.get("episode_length_steps", env_cfg["episode_max_steps"])))
                for idx, p in enumerate(curriculum_phases)
            ]

        per_phase: list[dict] = []
        for phase_id, phase_steps in phases_to_run:
            env.set_curriculum_phase(phase_id, episode_max_steps=phase_steps)
            rl_metrics = evaluator.evaluate(rl_model, phase=phase_id)
            pid_metrics = evaluator.evaluate(pid_model, phase=phase_id)
            improvement_pct = (
                (rl_metrics.mean_reward - pid_metrics.mean_reward)
                / (abs(pid_metrics.mean_reward) + 1e-9)
                * 100.0
            )
            per_phase.append(
                {
                    "phase": phase_id,
                    "episode_length_steps": phase_steps,
                    "rl_mean_reward": rl_metrics.mean_reward,
                    "pid_mean_reward": pid_metrics.mean_reward,
                    "reward_improvement_pct": improvement_pct,
                    "rl_violation_rate": rl_metrics.violation_rate,
                    "pid_violation_rate": pid_metrics.violation_rate,
                }
            )
    finally:
        env.close()

    phase0 = next((item for item in per_phase if item["phase"] == 0), per_phase[0])
    report = {
        "rl_mean_reward": phase0["rl_mean_reward"],
        "pid_mean_reward": phase0["pid_mean_reward"],
        "reward_improvement_pct": phase0["reward_improvement_pct"],
        "rl_violation_rate": phase0["rl_violation_rate"],
        "pid_violation_rate": phase0["pid_violation_rate"],
        "per_phase": per_phase,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2))
    print(f"[cross_validate_and_export] Report: {report_out}")

    verify_enabled = importlib.util.find_spec("onnxruntime") is not None
    exporter = ONNXExporter(
        {
            "opset_version": 17,
            "verify": {
                "enabled": verify_enabled,
                "n_test_samples": 3,
                "tolerance_abs": 1e-3,
            },
        }
    )
    onnx_result = exporter.export(
        model=rl_model,
        obs_dim=len(env_cfg["obs_vars"]) * env_cfg["history_steps"],
        output_path=str(onnx_out),
    )
    print(f"[cross_validate_and_export] ONNX: {onnx_result.onnx_path}")

    if not args.skip_trt:
        try:
            trt_exporter = TensorRTExporter({"precision": "fp16", "workspace_gb": 1})
            trt_result = trt_exporter.build_engine(
                onnx_path=str(onnx_out),
                trt_path=str(trt_out),
                obs_dim=onnx_result.obs_dim,
                act_dim=onnx_result.act_dim,
            )
            print(f"[cross_validate_and_export] TRT: {trt_result.trt_engine_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[cross_validate_and_export] TRT export skipped/failing: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
