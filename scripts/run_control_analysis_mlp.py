#!/usr/bin/env python3
"""Run control analysis using MLP surrogate instead of MockFMU."""
from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def main() -> int:
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sco2rl.analysis.scenario_runner import (
        ScenarioRunner,
        ControlScenario,
        build_mlp_env,
        build_mlp_pid,
    )
    from sco2rl.analysis.metrics import ControlMetricsSummary

    mlp_path = _PROJECT_ROOT / "artifacts/surrogate/mlp_step.pt"
    norm_path = _PROJECT_ROOT / "artifacts/surrogate/mlp_step_norm.npz"
    policy_path = _PROJECT_ROOT / "artifacts/rl/ppo_mlp/best_policy.pt"
    data_dir = _PROJECT_ROOT / "data"
    fig_dir = _PROJECT_ROOT / "paper/figures"

    if not mlp_path.exists():
        print(f"Error: MLP model not found: {mlp_path}")
        return 1
    if not norm_path.exists():
        print(f"Error: Norm file not found: {norm_path}")
        return 1

    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    norm = dict(np.load(norm_path))
    n_s = int(norm["s_mean"].shape[0])
    n_a = int(norm["a_mean"].shape[0])
    obs_lo = np.array([31.5, 70, 300, 200, 200, 70, 100, 31.5, 5, 0.5, 0.85, 0.9, 5, 15])[:n_s]
    obs_hi = np.array([42, 120, 1100, 1100, 1100, 120, 175, 42, 25, 8, 0.92, 0.95, 120, 21])[:n_s]
    obs_range = obs_hi - obs_lo

    class PolicyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(n_s, 256), torch.nn.Tanh(),
                torch.nn.Linear(256, 256), torch.nn.Tanh(),
                torch.nn.Linear(256, 128), torch.nn.Tanh(),
                torch.nn.Linear(128, n_a),
            )
            self.log_std = torch.nn.Parameter(-0.5 * torch.ones(n_a))

        def forward(self, obs):
            mean = self.net(obs)
            std = self.log_std.exp().expand_as(mean)
            return mean, std

    class MLPRLPolicy:
        def __init__(self, path):
            self._policy = PolicyNet()
            self._policy.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
            self._policy.eval()
            self.name = "PPO-MLP"

        def _norm_obs(self, obs):
            o = np.asarray(obs, dtype=np.float32).flatten()
            if len(o) == 15:
                o = np.concatenate([o[:7], o[8:15]])
            o = np.clip(o[:n_s], obs_lo, obs_hi)
            return torch.from_numpy(2.0 * (o - obs_lo) / obs_range - 1.0).float().unsqueeze(0)

        def predict(self, obs, deterministic=True):
            with torch.no_grad():
                mean, _ = self._policy(self._norm_obs(obs))
                action = torch.tanh(mean).squeeze(0).numpy()
            return np.array(action, dtype=np.float32), None

        def reset(self):
            pass

    def env_factory():
        return build_mlp_env(mlp_model_path=str(mlp_path), norm_path=str(norm_path), seed=42)

    pid = build_mlp_pid()
    rl = MLPRLPolicy(policy_path) if policy_path.exists() else None
    if rl is None:
        print(f"Warning: RL policy not found at {policy_path}, running PID only")

    print("Using MLP surrogate environment")
    print(f"PID: {pid.name}")
    if rl:
        print(f"RL: {rl.name}")

    runner = ScenarioRunner(n_seeds=2, dt=5.0, verbose=True)
    phases = list(range(7))
    scenarios = [
        ControlScenario.STEP_LOAD_UP_20,
        ControlScenario.STEP_LOAD_DOWN_20,
        ControlScenario.LOAD_REJECTION_50,
    ]

    print("\nRunning step response analysis...")
    t0 = time.time()
    results = runner.run_all(
        env_factory=env_factory,
        pid_policy=pid,
        rl_policy=rl,
        phases=phases,
        scenarios=scenarios,
        run_frequency=True,
        freq_env_factory=env_factory,
    )
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s ({len(results)} result objects)")

    out_path = data_dir / "control_analysis_mlp_phases.json"
    ScenarioRunner.save(results, out_path, env_type="MLP")
    print(f"Saved to {out_path}")

    PHASE_NAMES = {0: "Phase 0 — Steady-State", 1: "Phase 1 — Gradual Load",
        2: "Phase 2 — Ambient Disturbance", 3: "Phase 3 — EAF Transients",
        4: "Phase 4 — Load Rejection", 5: "Phase 5 — Cold Startup",
        6: "Phase 6 — Emergency Trip"}

    def find_result(res, phase, substr):
        for r in res:
            if r.phase == phase and substr in r.scenario:
                return r
        return None

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 9, "lines.linewidth": 1.5, "axes.grid": True, "grid.alpha": 0.3})

    r0 = find_result(results, 0, "+20")
    if r0 and (r0.pid_step or r0.rl_step):
        fig, ax = plt.subplots(figsize=(9, 4))
        if r0.pid_step and r0.pid_step.time_s:
            ps = r0.pid_step
            ax.plot(ps.time_s, ps.setpoint, "k--", lw=1.2, alpha=0.7, label="Setpoint")
            ax.plot(ps.time_s, ps.response, color="#e07b39", lw=1.8, label="PID")
        if r0.rl_step and r0.rl_step.time_s:
            ax.plot(r0.rl_step.time_s, r0.rl_step.response, color="steelblue", lw=1.8, label="PPO-MLP")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("$W_{net}$ (MW)")
        ax.set_title(f"Step Response — {PHASE_NAMES[0]}, +20% Load Step (MLP)")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(fig_dir / "step_response_phase0.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved step_response_phase0.png")

    r2 = find_result(results, 2, "+20")
    if r2 and (r2.pid_step or r2.rl_step):
        fig, ax = plt.subplots(figsize=(9, 4))
        if r2.pid_step and r2.pid_step.time_s:
            ax.plot(r2.pid_step.time_s, r2.pid_step.setpoint, "k--", lw=1.2, alpha=0.7, label="Setpoint")
            ax.plot(r2.pid_step.time_s, r2.pid_step.response, color="#e07b39", lw=1.8, label="PID")
        if r2.rl_step and r2.rl_step.time_s:
            ax.plot(r2.rl_step.time_s, r2.rl_step.response, color="steelblue", lw=1.8, label="PPO-MLP")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("$W_{net}$ (MW)")
        ax.set_title(f"Step Response — {PHASE_NAMES[2]}, +20% Load Step (MLP)")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(fig_dir / "step_response_phase2.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved step_response_phase2.png")

    r_freq = find_result(results, 0, "freq")
    if r_freq and r_freq.pid_freq and r_freq.pid_freq.frequencies_hz:
        pf = r_freq.pid_freq
        freqs = np.array(pf.frequencies_hz)
        mag_db = np.array(pf.magnitude_db)
        ph_deg = np.array(pf.phase_deg)
        fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
        axes[0].semilogx(freqs, mag_db, color="#e07b39", label="PID")
        axes[0].axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        axes[0].set_ylabel("Magnitude (dB)")
        axes[0].set_title("Bode Plot — Phase 0 (MLP)")
        axes[0].legend(fontsize=8)
        axes[1].semilogx(freqs, ph_deg, color="#e07b39", label="PID")
        axes[1].axhline(-180, color="k", lw=0.8, ls="--", alpha=0.5)
        axes[1].set_ylabel("Phase (deg)")
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "bode_plot_phase0.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved bode_plot_phase0.png")

    scenarios_list = ["step_load_+20pct", "step_load_-20pct", "load_rejection_-50pct"]
    sc_labels = ["+20%", "-20%", "-50%"]
    iae_mat = np.full((7, 3), np.nan)
    settle_mat = np.full((7, 3), np.nan)
    for r in results:
        if r.pid_step and r.phase < 7:
            for j, s in enumerate(scenarios_list):
                if s == r.scenario:
                    iae_mat[r.phase, j] = r.pid_step.iae
                    settle_mat[r.phase, j] = r.pid_step.settling_time_s
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, mat, title in [
        (axes[0], iae_mat, "IAE — PID (MLP)"),
        (axes[1], settle_mat, "Settling Time (s) — PID (MLP)"),
    ]:
        plot_mat = np.where(np.isnan(mat), 0, mat)
        im = ax.imshow(plot_mat, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(3))
        ax.set_xticklabels(sc_labels)
        ax.set_yticks(range(7))
        ax.set_yticklabels([f"Ph {p}" for p in range(7)])
        ax.set_title(title)
        ax.set_xlabel("Scenario")
        plt.colorbar(im, ax=ax, shrink=0.8)
        mx = plot_mat.max() if plot_mat.max() > 0 else 1
        for i in range(7):
            for j in range(3):
                if not np.isnan(mat[i, j]):
                    c = "white" if plot_mat[i, j] > 0.6 * mx else "black"
                    ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7, color=c)
    fig.suptitle("PID Control Metrics (MLP Surrogate)", fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_dir / "control_metrics_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved control_metrics_heatmap.png")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
