"""Generate paper figures from training/evaluation artifacts.

Generates 6 figures:
  phase_rewards.png        — RL vs PID mean reward per phase (dual bars: 5M + interleaved)
  phase_improvement.png    — % improvement over PID per phase (5M and interleaved)
  latency_summary.png      — TensorRT inference latency percentiles
  fidelity_rmse.png        — Surrogate RMSE per variable

New figures (training curve + thermo trajectories) are in separate scripts:
  plot_training_curve.py   — reward vs timestep with phase annotations
  plot_thermo_trajectories.py — T/P/W_net trajectories from preflight_100.h5
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
DATA_DIR = ROOT / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
}
plt.rcParams.update(STYLE)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _try_load_json(*paths: Path) -> dict | None:
    """Return the first existing JSON file, or None."""
    for p in paths:
        if p.exists():
            return load_json(p)
    return None


# ── Phase rewards: dual bars (5M baseline + interleaved) ──────────────────────

def plot_phase_rewards(report_5m: dict, report_il: dict | None = None) -> None:
    per_phase_5m = report_5m.get("per_phase", [])
    phases = [p["phase"] for p in per_phase_5m]

    rl_5m  = np.array([p["rl_mean_reward"]  for p in per_phase_5m])
    pid    = np.array([p["pid_mean_reward"]  for p in per_phase_5m])

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(phases))
    width = 0.25

    ax.bar(x - width, pid,   width, label="PID baseline", color="#7F8C8D", alpha=0.85)
    ax.bar(x,         rl_5m, width, label="RL (5M steps)", color="#2980B9", alpha=0.85)

    if report_il is not None:
        per_phase_il = report_il.get("per_phase", [])
        rl_il = np.array([p["rl_mean_reward"] for p in per_phase_il])
        ax.bar(x + width, rl_il, width,
               label="RL + Interleaved replay", color="#27AE60", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Ph {ph}" for ph in phases])
    ax.set_xlabel("Curriculum Phase")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("RL vs PID Mean Reward across Curriculum Phases")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)

    # Annotate catastrophic forgetting region on 5M-only run
    if report_il is None or True:
        ax.axvspan(2.5, 6.5, alpha=0.05, color="red",
                   label="_nolegend_")
        ax.text(4.5, ax.get_ylim()[1] * 0.93,
                "Catastrophic\nforgetting region\n(5M-only run)",
                ha="center", fontsize=8, color="#C0392B",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FADBD8", alpha=0.7))

    plt.tight_layout()
    plt.savefig(FIG_DIR / "phase_rewards.png", dpi=200)
    plt.close()


def plot_phase_improvement(report_5m: dict, report_il: dict | None = None) -> None:
    per_phase_5m = report_5m.get("per_phase", [])
    phases = [p["phase"] for p in per_phase_5m]
    impr_5m = np.array([p["reward_improvement_pct"] for p in per_phase_5m])

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(phases))
    width = 0.35

    colors_5m = ["#27AE60" if v >= 0 else "#E74C3C" for v in impr_5m]

    if report_il is not None:
        per_phase_il = report_il.get("per_phase", [])
        impr_il = np.array([p["reward_improvement_pct"] for p in per_phase_il])
        colors_il = ["#2ECC71" if v >= 0 else "#C0392B" for v in impr_il]
        ax.bar(x - width / 2, impr_5m, width, color=colors_5m,
               label="RL 5M steps", alpha=0.85)
        ax.bar(x + width / 2, impr_il, width, color=colors_il,
               label="RL + Interleaved", alpha=0.85)
    else:
        ax.bar(x, impr_5m, width * 1.4, color=colors_5m, label="RL 5M steps", alpha=0.85)

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Ph {ph}" for ph in phases])
    ax.set_xlabel("Curriculum Phase")
    ax.set_ylabel("Improvement over PID [%]")
    ax.set_title("RL Improvement over PID Baseline")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "phase_improvement.png", dpi=200)
    plt.close()


def plot_latency_hist(latency: dict) -> None:
    metrics = latency["host_latency_ms"]
    labels = ["p90", "p95", "p99", "max"]
    vals = [metrics[k] for k in labels]

    plt.figure(figsize=(6.8, 4.2))
    bars = plt.bar(labels, vals, color=["#3498DB", "#2980B9", "#2471A3", "#1A5276"])
    plt.axhline(1.0, color="tab:red", linestyle="--", linewidth=1.5, label="1.0 ms SLA")
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f} ms", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Latency [ms]")
    plt.title("TensorRT FP16 Inference Latency (plant-edge deployment)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "latency_summary.png", dpi=200)
    plt.close()


def plot_fidelity_rmse(fidelity: dict) -> None:
    per_var = fidelity.get("per_variable", {})
    names = list(per_var.keys())
    rmse = np.array([per_var[k]["rmse"] for k in names])

    order = np.argsort(rmse)
    names_sorted = [names[i] for i in order]
    rmse_sorted = rmse[order]
    colors = ["#E74C3C" if v > 0.05 else "#27AE60" for v in rmse_sorted]

    plt.figure(figsize=(8.5, 5.5))
    bars = plt.barh(names_sorted, rmse_sorted, color=colors)
    plt.axvline(0.05, color="tab:red", linestyle="--", linewidth=1.5,
                label="Gate threshold (5% NRMSE)")
    plt.xlabel("Normalized RMSE")
    plt.title("Surrogate Model Fidelity: FNO vs FMU (per state variable)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fidelity_rmse.png", dpi=200)
    plt.close()


def main() -> None:
    # Load 5M baseline cross-validation (always present)
    report_5m = _try_load_json(
        DATA_DIR / "cross_validation_final_5M.json",
        ROOT / "artifacts/policies/evaluation_report.json",
        DATA_DIR / "evaluation_report.json",
    )
    if report_5m is None:
        print("WARNING: no 5M cross-validation data found — using evaluation_report.json")
        report_5m = load_json(DATA_DIR / "evaluation_report.json")

    # Load interleaved cross-validation (may not exist yet)
    report_il = _try_load_json(DATA_DIR / "cross_validation_interleaved.json")
    if report_il is None:
        print("NOTE: interleaved results not yet available — showing 5M-only baseline")

    latency  = _try_load_json(ROOT / "artifacts/policies/latency_benchmark_phase3.json")
    fidelity = _try_load_json(
        ROOT / "artifacts/surrogate/fidelity_report.json",
        DATA_DIR / "surrogate_fidelity_report.json",
    )

    plot_phase_rewards(report_5m, report_il)
    plot_phase_improvement(report_5m, report_il)
    if latency:
        plot_latency_hist(latency)
    else:
        print("WARNING: latency benchmark file not found — skipping latency_summary.png")
    if fidelity:
        plot_fidelity_rmse(fidelity)
    else:
        print("WARNING: fidelity report not found — skipping fidelity_rmse.png")

    print("Generated figures in", FIG_DIR)


if __name__ == "__main__":
    main()
