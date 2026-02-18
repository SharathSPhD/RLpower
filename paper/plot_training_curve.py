#!/usr/bin/env python3
"""Plot training reward curve with phase transition annotations.

Parses artifacts/monitoring/fmu_monitor.log for rollout log lines:
  [CurriculumCallback] step=NNN phase=P episodes=E mean_reward=R.RR ...
  [CurriculumCallback] *** ADVANCED to phase P at step=NNN ***

Outputs: paper/figures/training_curve.png
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ROOT / "artifacts/monitoring/fmu_monitor.log"

# Monitor daemon log format: step=NNN[,NNN] phase=P ... mean_r=R.RR ...
_MONITOR_RE = re.compile(
    r"step=([\d,]+)\s+phase=(\d+).*?mean_r=([-\d\.]+)"
)
# CurriculumCallback training log format: step=NNN phase=P ... mean_reward=R.RR
_ROLLOUT_RE = re.compile(
    r"step=(\d+).*?phase=(\d+).*?mean_reward=([-\d\.]+)"
)
_ADVANCE_RE = re.compile(r"ADVANCED to phase (\d+) at step=(\d+)")

PHASE_COLORS = [
    "#3498DB", "#2ECC71", "#E67E22", "#E74C3C",
    "#9B59B6", "#F39C12", "#1ABC9C",
]


def parse_log(log_path: Path) -> tuple[list, list, list, list[tuple[int, int]]]:
    """Returns (steps, phases, rewards, advances).

    Accepts both the monitor daemon log format (step=NNN,NNN phase=P mean_r=R)
    and the CurriculumCallback training log format (step=NNN phase=P mean_reward=R).
    """
    steps, phases, rewards = [], [], []
    advances = []
    seen: set[tuple[int, int]] = set()  # deduplicate monitor polling duplicates

    text = log_path.read_text(errors="replace")
    for line in text.splitlines():
        # Try monitor log format first (most common in fmu_monitor.log)
        m = _MONITOR_RE.search(line)
        if m:
            step = int(m.group(1).replace(",", ""))
            phase = int(m.group(2))
            reward = float(m.group(3))
            key = (step, phase)
            if key not in seen:
                seen.add(key)
                steps.append(step)
                phases.append(phase)
                rewards.append(reward)
            continue
        # Fallback to training log format
        m2 = _ROLLOUT_RE.search(line)
        if m2:
            step = int(m2.group(1))
            phase = int(m2.group(2))
            reward = float(m2.group(3))
            key = (step, phase)
            if key not in seen:
                seen.add(key)
                steps.append(step)
                phases.append(phase)
                rewards.append(reward)
        a = _ADVANCE_RE.search(line)
        if a:
            advances.append((int(a.group(2)), int(a.group(1))))  # (step, new_phase)

    # Sort by step for correct time ordering
    order = sorted(range(len(steps)), key=lambda i: steps[i])
    steps  = [steps[i]   for i in order]
    phases = [phases[i]  for i in order]
    rewards = [rewards[i] for i in order]
    return steps, phases, rewards, advances


def smooth(values: list[float], window: int = 20) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def main(log_path: Path = LOG_FILE) -> None:
    if not log_path.exists():
        print(f"Log not found: {log_path}")
        sys.exit(1)

    steps, phases, rewards, advances = parse_log(log_path)
    if not steps:
        print("No rollout data found in log.")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    steps_arr = np.array(steps)
    rewards_raw = np.array(rewards)
    rewards_smooth = smooth(rewards, window=min(30, len(rewards) // 4 + 1))

    # Color scatter points by phase
    for ph in sorted(set(phases)):
        mask = np.array(phases) == ph
        ax.scatter(steps_arr[mask] / 1e6, rewards_raw[mask],
                   s=4, alpha=0.25, color=PHASE_COLORS[ph % len(PHASE_COLORS)],
                   label=f"_ph{ph}_raw")

    ax.plot(steps_arr / 1e6, rewards_smooth, linewidth=2.0,
            color="#2C3E50", alpha=0.9, label="Smoothed reward")

    # Phase transition vertical lines
    for step, ph in advances:
        color = PHASE_COLORS[ph % len(PHASE_COLORS)]
        ax.axvline(step / 1e6, color=color, linestyle="--",
                   linewidth=1.2, alpha=0.7)
        ax.text(step / 1e6 + 0.02, ax.get_ylim()[1] * 0.92,
                f"Ph{ph}", fontsize=7.5, color=color, rotation=90,
                va="top")

    # Phase legend patches
    phase_set = sorted(set(phases))
    patches = [
        plt.matplotlib.patches.Patch(
            color=PHASE_COLORS[ph % len(PHASE_COLORS)],
            label=f"Phase {ph}",
        )
        for ph in phase_set
    ]
    ax.legend(handles=patches + [
        plt.Line2D([0], [0], color="#2C3E50", linewidth=2, label="Smoothed")
    ], fontsize=8.5, loc="upper left")

    ax.set_xlabel("Training steps [×10⁶]", fontsize=11)
    ax.set_ylabel("Episode reward", fontsize=11)
    ax.set_title(
        "PPO Training Curve — sCO₂ Waste Heat Recovery\n"
        "Curriculum phases 0–6; dashed lines = phase advances",
        fontsize=11, fontweight="bold",
    )
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "training_curve.png", dpi=200)
    plt.close()
    print(f"Saved: {FIG_DIR / 'training_curve.png'}")
    print(f"  steps={len(steps):,}  phases seen={sorted(set(phases))}  advances={len(advances)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else LOG_FILE)
