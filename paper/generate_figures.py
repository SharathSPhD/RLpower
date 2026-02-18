"""Generate paper figures from training/evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def plot_phase_rewards(report: dict) -> None:
    per_phase = report.get("per_phase", [])
    phases = [p["phase"] for p in per_phase]
    rl = np.asarray([p["rl_mean_reward"] for p in per_phase], dtype=np.float64)
    pid = np.asarray([p["pid_mean_reward"] for p in per_phase], dtype=np.float64)

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(phases, rl, marker="o", linewidth=2.0, label="RL")
    plt.plot(phases, pid, marker="s", linewidth=2.0, label="PID")
    plt.xlabel("Curriculum Phase")
    plt.ylabel("Mean Episode Reward")
    plt.title("RL vs PID across 7 phases")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "phase_rewards.png", dpi=200)
    plt.close()


def plot_phase_improvement(report: dict) -> None:
    per_phase = report.get("per_phase", [])
    phases = [p["phase"] for p in per_phase]
    impr = np.asarray([p["reward_improvement_pct"] for p in per_phase], dtype=np.float64)
    colors = ["tab:green" if x >= 0 else "tab:red" for x in impr]

    plt.figure(figsize=(7.5, 4.5))
    plt.bar(phases, impr, color=colors)
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xlabel("Curriculum Phase")
    plt.ylabel("Improvement over PID [%]")
    plt.title("Policy improvement by phase")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "phase_improvement.png", dpi=200)
    plt.close()


def plot_latency_hist(latency: dict) -> None:
    metrics = latency["host_latency_ms"]
    labels = ["p90", "p95", "p99", "max"]
    vals = [metrics[k] for k in labels]

    plt.figure(figsize=(6.8, 4.2))
    plt.bar(labels, vals, color="tab:blue")
    plt.axhline(1.0, color="tab:red", linestyle="--", label="1.0 ms SLA")
    plt.ylabel("Latency [ms]")
    plt.title("TensorRT host latency summary")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "latency_summary.png", dpi=200)
    plt.close()


def plot_fidelity_rmse(fidelity: dict) -> None:
    per_var = fidelity.get("per_variable", {})
    names = list(per_var.keys())
    rmse = np.asarray([per_var[k]["rmse"] for k in names], dtype=np.float64)

    order = np.argsort(rmse)
    names_sorted = [names[i] for i in order]
    rmse_sorted = rmse[order]

    plt.figure(figsize=(8.5, 5.5))
    plt.barh(names_sorted, rmse_sorted, color="tab:purple")
    plt.axvline(0.05, color="tab:red", linestyle="--", label="Gate threshold (overall)")
    plt.xlabel("Normalized RMSE")
    plt.title("Surrogate fidelity: per-variable normalized RMSE")
    plt.legend()
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fidelity_rmse.png", dpi=200)
    plt.close()


def main() -> None:
    eval_report = load_json(ROOT / "artifacts/policies/evaluation_report.json")
    latency = load_json(ROOT / "artifacts/policies/latency_benchmark_phase3.json")
    fidelity = load_json(ROOT / "artifacts/surrogate/fidelity_report.json")

    plot_phase_rewards(eval_report)
    plot_phase_improvement(eval_report)
    plot_latency_hist(latency)
    plot_fidelity_rmse(fidelity)

    print("Generated figures in", FIG_DIR)


if __name__ == "__main__":
    main()
