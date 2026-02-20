#!/usr/bin/env python3
"""Auto-tune PID gains for SCO2FMUEnv using open-loop step tests.

Estimates process gain and time constant for each action channel via an
open-loop step test, then applies IMC (Internal Model Control) tuning
to compute Kp and Ki.  Saves recommended gains to
``configs/control/pid_gains_tuned.yaml``.

Usage
-----
python scripts/tune_pid.py --use-mock           # MockFMU estimate
python scripts/tune_pid.py --fmu-path ...       # Real FMU (requires FMU artifact)

IMC tuning formulas
-------------------
Given process gain K, time constant τ, desired closed-loop bandwidth λ:

    Kp = τ / (|K| * λ)
    Ki = Kp / τ = 1 / (|K| * λ)
    Kd = τ_d * Kp  (τ_d = τ / 5 as heuristic)

Target: phase margin ≥ 45°, gain margin ≥ 6 dB.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PID auto-tuning via open-loop step tests")
    p.add_argument("--use-mock", action="store_true",
                   help="Use MockFMU instead of real FMU")
    p.add_argument("--fmu-path", type=str,
                   default="artifacts/fmu_build/SCO2RecuperatedCycle.fmu")
    p.add_argument("--step-amplitude", type=float, default=0.1,
                   help="Step amplitude in normalised action units [0, 1]")
    p.add_argument("--warmup-steps", type=int, default=40,
                   help="Steps at nominal before step injection")
    p.add_argument("--record-steps", type=int, default=120,
                   help="Steps recorded after step injection")
    p.add_argument("--lambda-factor", type=float, default=2.0,
                   help="IMC λ = λ_factor * τ  (larger = more conservative)")
    p.add_argument("--output", type=str,
                   default="configs/control/pid_gains_tuned.yaml")
    return p.parse_args()


def step_test(env, channel_idx: int, step_amp: float,
              warmup: int, record: int, dt: float) -> dict[str, float]:
    """Run open-loop step test on one action channel.

    Returns
    -------
    dict with keys: K (gain), tau (time constant), output_var
    """
    obs_vars = getattr(env, "_obs_vars", [])
    action_vars = getattr(env, "_action_vars", [])

    output_mapping = {
        0: "W_net",
        1: "T_turbine_inlet",
        2: "P_high",
        3: "T_compressor_inlet",
    }
    output_var = output_mapping.get(channel_idx, "W_net")

    obs, _ = env.reset(seed=42)

    # Warm up at mid-range action
    baseline_action = np.zeros(len(action_vars), dtype=np.float32)
    y_before: list[float] = []

    for _ in range(warmup):
        obs, _, terminated, truncated, info = env.step(baseline_action)
        raw = info.get("raw_obs", {})
        y_val = raw.get(output_var, 0.0)
        y_before.append(y_val)
        if terminated or truncated:
            break

    y_initial = float(np.mean(y_before[-10:])) if len(y_before) >= 10 else float(y_before[-1]) if y_before else 0.0

    # Apply step
    step_action = baseline_action.copy()
    step_action[channel_idx] = float(step_amp)
    y_after: list[float] = []

    for _ in range(record):
        obs, _, terminated, truncated, info = env.step(step_action)
        raw = info.get("raw_obs", {})
        y_val = raw.get(output_var, 0.0)
        y_after.append(y_val)
        if terminated or truncated:
            break

    y_arr = np.array(y_after)

    # Final value (mean of last 20%)
    n_tail = max(1, int(0.2 * len(y_arr)))
    y_final = float(np.mean(y_arr[-n_tail:]))

    # Process gain K = Δy / Δu
    K = (y_final - y_initial) / max(abs(step_amp), 1e-9)

    # Time constant: time to reach 63.2% of step
    step_delta = y_final - y_initial
    threshold_632 = y_initial + 0.632 * step_delta
    if abs(step_delta) > 1e-9:
        above = (y_arr - threshold_632) * np.sign(step_delta) >= 0
        if np.any(above):
            tau = float(np.argmax(above)) * dt
        else:
            tau = float(len(y_arr)) * dt
    else:
        tau = 20.0  # Default if no response

    tau = max(tau, dt)  # At least one step

    print(f"  Channel {channel_idx} ({output_var}):  K={K:.4f}  τ={tau:.1f}s  "
          f"Δy={step_delta:.4f}")

    return {"K": K, "tau": tau, "output_var": output_var, "channel_idx": channel_idx}


def imc_gains(K: float, tau: float, lambda_factor: float, dt: float) -> dict[str, float]:
    """Compute IMC-tuned PID gains.

    Kp = τ / (|K| * λ)
    Ki = 1 / (|K| * λ)   ≡ Kp / τ
    Kd = (τ/5) * Kp       (conservative D-term)
    λ = lambda_factor * τ
    """
    lam = lambda_factor * max(tau, dt)
    abs_K = max(abs(K), 1e-9)
    kp = tau / (abs_K * lam)
    ki = kp / tau
    kd = (tau / 5.0) * kp
    # Saturate to reasonable bounds (avoid overly aggressive gains)
    kp = float(np.clip(kp, 0.001, 2.0))
    ki = float(np.clip(ki, 1e-5, 0.1))
    kd = float(np.clip(kd, 0.0, 5.0))
    return {"kp": kp, "ki": ki, "kd": kd}


def main() -> int:
    args = parse_args()

    from sco2rl.analysis.scenario_runner import (
        _MOCK_OBS_VARS, _MOCK_ACTION_VARS, _MOCK_DESIGN_POINT, _MOCK_ENV_CONFIG,
        build_mock_env,
    )

    if args.use_mock:
        print("Using MockFMU for PID tuning")
        env_factory = build_mock_env
    else:
        fmu_path = Path(args.fmu_path)
        if not fmu_path.is_absolute():
            fmu_path = _PROJECT_ROOT / args.fmu_path
        if not fmu_path.exists():
            print(f"Error: FMU not found: {fmu_path}. Use --use-mock.")
            return 1
        print(f"Using real FMU: {fmu_path}")
        import yaml as _yaml
        from scripts.run_control_analysis import _make_fmu_factory
        env_factory = _make_fmu_factory(fmu_path)

    dt = 5.0
    n_channels = len(_MOCK_ACTION_VARS)

    print(f"\nRunning open-loop step tests (amplitude={args.step_amplitude})")
    print(f"  warmup={args.warmup_steps} steps, record={args.record_steps} steps")
    print(f"  IMC λ_factor={args.lambda_factor}")

    channel_results: list[dict] = []
    for ch_idx in range(n_channels):
        env = env_factory()
        try:
            result = step_test(
                env=env,
                channel_idx=ch_idx,
                step_amp=args.step_amplitude,
                warmup=args.warmup_steps,
                record=args.record_steps,
                dt=dt,
            )
        finally:
            env.close()

        gains = imc_gains(result["K"], result["tau"], args.lambda_factor, dt)
        result.update(gains)
        channel_results.append(result)

    # ── Build YAML output ────────────────────────────────────────────────────
    action_vars = _MOCK_ACTION_VARS
    output_doc: dict = {
        "# IMC-tuned PID gains from open-loop step tests": None,
        "# Generated by scripts/tune_pid.py": None,
        "mock_channels": {},
    }

    print("\n─── Tuned Gains ──────────────────────────────────────────")
    print(f"{'Channel':<30}  {'Kp':>7}  {'Ki':>8}  {'Kd':>7}")
    print("─" * 60)

    channels_yaml: dict = {}
    for r in channel_results:
        act_name = action_vars[r["channel_idx"]]
        channels_yaml[act_name] = {
            "kp": round(r["kp"], 5),
            "ki": round(r["ki"], 6),
            "kd": round(r["kd"], 4),
            "anti_windup_gain": 0.10,
            "derivative_filter_tau": round(r["tau"] / 2.0, 1),
            "measurement_obs": r["output_var"],
            "process_gain_K": round(r["K"], 5),
            "time_constant_tau_s": round(r["tau"], 1),
            "imc_lambda_s": round(args.lambda_factor * r["tau"], 1),
        }
        print(f"{act_name:<30}  {r['kp']:>7.5f}  {r['ki']:>8.6f}  {r['kd']:>7.4f}")

    output_doc_clean = {
        "# Generated by scripts/tune_pid.py": None,
        "mock_channels": channels_yaml,
    }

    output_path = _PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build YAML with comment header manually
    yaml_lines = [
        "# IMC-tuned PID gains from open-loop step tests",
        "# Generated by: python scripts/tune_pid.py",
        "# See configs/control/pid_gains.yaml for human-tuned reference",
        "",
        "mock_channels:",
    ]
    for act_name, gains in channels_yaml.items():
        yaml_lines.append(f"  {act_name}:")
        for k, v in gains.items():
            if k.startswith("#"):
                continue
            yaml_lines.append(f"    {k}: {v}")
        yaml_lines.append("")

    output_path.write_text("\n".join(yaml_lines) + "\n")
    print(f"\nTuned gains saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
