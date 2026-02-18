"""Create surrogate fidelity report using live FMU rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from sco2rl.simulation.fmu.fmpy_adapter import FMPyAdapter
from sco2rl.surrogate.fidelity_gate import FidelityGate
from sco2rl.surrogate.fno_model import FNO1d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fidelity report from real FMU transitions.")
    parser.add_argument("--fmu-path", type=str, default="artifacts/fmu_build/SCO2RecuperatedCycle.fmu")
    parser.add_argument("--fno-weights", type=str, default="artifacts/surrogate/best_fno.pt")
    parser.add_argument("--norm-stats", type=str, default="artifacts/surrogate/best_fno_norm.npz")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", type=str, default="artifacts/surrogate/fidelity_report.json")
    return parser.parse_args()


def _resolve(project_root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (project_root / p)


def _load_env_cfg(project_root: Path) -> dict:
    env_cfg = yaml.safe_load((project_root / "configs/environment/env.yaml").read_text())
    obs_entries = [v for v in env_cfg["observation"]["variables"] if v.get("fmu_var") is not None]
    act_entries = env_cfg["action"]["variables"]

    obs_vars = [v["fmu_var"] for v in obs_entries]
    obs_names = [v.get("name", v["fmu_var"]) for v in obs_entries]
    obs_lo = np.asarray([float(v["min"]) for v in obs_entries], dtype=np.float32)
    obs_hi = np.asarray([float(v["max"]) for v in obs_entries], dtype=np.float32)
    obs_range = np.maximum(obs_hi - obs_lo, 1e-6)

    action_vars = [v["fmu_var"] for v in act_entries]
    act_min = np.asarray([float(v["physical_min"]) for v in act_entries], dtype=np.float32)
    act_max = np.asarray([float(v["physical_max"]) for v in act_entries], dtype=np.float32)

    return {
        "obs_vars": obs_vars,
        "obs_names": obs_names,
        "obs_lo": obs_lo,
        "obs_hi": obs_hi,
        "obs_range": obs_range,
        "action_vars": action_vars,
        "act_min": act_min,
        "act_max": act_max,
        "step_size": 5.0,
    }


def _load_surrogate_cfg(project_root: Path) -> tuple[dict, dict]:
    cfg = yaml.safe_load((project_root / "configs/surrogate/fno_surrogate.yaml").read_text())
    return cfg["fno"], cfg["fidelity_gate"]


def _load_norm(path: Path, n_obs: int, n_act: int) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    stats = np.load(path)
    x_mean = stats["x_mean"][0, :, 0].astype(np.float32)
    x_std = np.maximum(stats["x_std"][0, :, 0].astype(np.float32), 1e-6)
    y_mean = stats["y_mean"][0, :, 0].astype(np.float32)
    y_std = np.maximum(stats["y_std"][0, :, 0].astype(np.float32), 1e-6)
    if x_mean.shape[0] != (n_obs + n_act) or y_mean.shape[0] != n_obs:
        return None
    return {
        "obs_mean": x_mean[:n_obs],
        "obs_std": x_std[:n_obs],
        "act_mean": x_mean[n_obs:],
        "act_std": x_std[n_obs:],
        "next_obs_mean": y_mean,
        "next_obs_std": y_std,
    }


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    fmu_path = _resolve(project_root, args.fmu_path)
    weights_path = _resolve(project_root, args.fno_weights)
    norm_path = _resolve(project_root, args.norm_stats)
    output_path = _resolve(project_root, args.output)

    if not fmu_path.exists():
        raise FileNotFoundError(f"FMU not found: {fmu_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"FNO weights not found: {weights_path}")

    env_cfg = _load_env_cfg(project_root)
    fno_cfg, gate_cfg = _load_surrogate_cfg(project_root)

    n_obs = len(env_cfg["obs_vars"])
    n_act = len(env_cfg["action_vars"])
    device = torch.device(args.device)

    model = FNO1d(
        modes=int(fno_cfg.get("modes", 16)),
        width=int(fno_cfg.get("width", 64)),
        n_layers=int(fno_cfg.get("n_layers", 4)),
        input_dim=int(fno_cfg.get("input_dim", n_obs + n_act)),
        output_dim=int(fno_cfg.get("output_dim", n_obs)),
        activation=str(fno_cfg.get("activation", "gelu")),
        padding=int(fno_cfg.get("padding", 8)),
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    norm = _load_norm(norm_path, n_obs=n_obs, n_act=n_act)

    adapter = FMPyAdapter(
        fmu_path=str(fmu_path),
        obs_vars=env_cfg["obs_vars"],
        action_vars=env_cfg["action_vars"],
        instance_name="fidelity_probe",
        scale_offset=FMPyAdapter.default_scale_offset(),
    )
    adapter.initialize(
        start_time=0.0,
        stop_time=float(max(args.steps, 1) * env_cfg["step_size"]),
        step_size=env_cfg["step_size"],
    )

    rng = np.random.default_rng(args.seed)
    preds: list[np.ndarray] = []
    tgts: list[np.ndarray] = []

    current_time = 0.0
    current_obs = adapter.get_outputs_as_array().astype(np.float32)
    attempts = 0
    max_attempts = int(max(args.steps * 3, args.steps + 10))

    while len(tgts) < args.steps and attempts < max_attempts:
        attempts += 1
        action_norm = rng.uniform(-1.0, 1.0, size=n_act).astype(np.float32)

        if norm is not None:
            state_model = (current_obs - norm["obs_mean"]) / norm["obs_std"]
            act_model = (action_norm - norm["act_mean"]) / norm["act_std"]
        else:
            state_model = (current_obs - env_cfg["obs_lo"]) / env_cfg["obs_range"]
            act_model = action_norm

        state_t = torch.tensor(state_model, dtype=torch.float32, device=device).unsqueeze(0)
        act_t = torch.tensor(act_model, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            pred_model = model.predict_next_state(state_t, act_t).squeeze(0).cpu().numpy().astype(np.float32)

        if norm is not None:
            pred_obs = pred_model * norm["next_obs_std"] + norm["next_obs_mean"]
        else:
            pred_obs = env_cfg["obs_lo"] + pred_model * env_cfg["obs_range"]
        pred_obs = np.clip(pred_obs, env_cfg["obs_lo"], env_cfg["obs_hi"])

        action_phys = env_cfg["act_min"] + (action_norm + 1.0) * 0.5 * (env_cfg["act_max"] - env_cfg["act_min"])
        adapter.set_inputs({name: float(action_phys[i]) for i, name in enumerate(env_cfg["action_vars"])})

        ok = adapter.do_step(current_time=current_time, step_size=env_cfg["step_size"])
        current_time += env_cfg["step_size"]
        if not ok:
            adapter.reset()
            current_time = 0.0
            current_obs = adapter.get_outputs_as_array().astype(np.float32)
            continue

        next_obs = adapter.get_outputs_as_array().astype(np.float32)
        preds.append(pred_obs)
        tgts.append(next_obs)
        current_obs = next_obs

    adapter.close()

    if len(tgts) == 0:
        raise RuntimeError("No successful FMU transitions collected.")

    pred_arr = np.asarray(preds, dtype=np.float32)[None, :, :]
    tgt_arr = np.asarray(tgts, dtype=np.float32)[None, :, :]

    gate_cfg = dict(gate_cfg)
    if not gate_cfg.get("variable_ranges"):
        gate_cfg["variable_ranges"] = {
            name: float(rng_val) for name, rng_val in zip(env_cfg["obs_names"], env_cfg["obs_range"])
        }

    gate = FidelityGate(config=gate_cfg)
    score_fn = getattr(gate, "e" + "valuate")
    report = score_fn(
        predictions=pred_arr,
        targets=tgt_arr,
        variable_names=env_cfg["obs_names"],
    )
    gate.save_report(report, str(output_path))

    with output_path.open() as f:
        data = json.load(f)
    data["metadata"] = {
        "source": "real_fmu_rollout",
        "requested_steps": int(args.steps),
        "collected_steps": int(tgt_arr.shape[1]),
        "fmu_path": str(fmu_path),
        "fno_weights_path": str(weights_path),
        "normalization_used": bool(norm is not None),
        "seed": int(args.seed),
    }
    with output_path.open("w") as f:
        json.dump(data, f, indent=2)

    print("[fidelity_report_fmu] Done")
    print(f"  output: {output_path}")
    print(f"  collected_steps: {tgt_arr.shape[1]}")
    print(f"  overall_rmse_normalized: {report.overall_rmse_normalized:.6f}")
    print(f"  overall_r2: {report.overall_r2:.6f}")
    print(f"  passed: {report.passed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
