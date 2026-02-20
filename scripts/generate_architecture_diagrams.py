#!/usr/bin/env python3
"""Generate publication-quality architecture and training pipeline diagrams.

Outputs:
    paper/figures/system_architecture.png   — 3-layer system architecture
    paper/figures/training_pipeline.png     — Left-to-right training flowchart

Usage:
    python scripts/generate_architecture_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
DPI = 150


def _add_box(ax, xy, w, h, text, color, fontsize=9):
    """Add a rounded box with text."""
    box = mpatches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                                  facecolor=color, edgecolor="#333", linewidth=0.8)
    ax.add_patch(box)
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha="center", va="center",
            fontsize=fontsize, wrap=True)


def _add_arrow(ax, start, end, label="", color="#333"):
    """Add arrow with optional label."""
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2))
    if label:
        mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(mid[0], mid[1], label, fontsize=7, ha="center", va="bottom")


def generate_system_architecture() -> Path:
    """Figure 1: 3-layer system architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    # Layer 1 (top): Physics Simulation
    layer1_y = 8.5
    ax.add_patch(mpatches.FancyBboxPatch((1, layer1_y - 0.4), 8, 0.8,
                 boxstyle="round,pad=0.02", facecolor="#E8F4FD", edgecolor="#2980B9"))
    ax.text(5, layer1_y, "Layer 1: Physics Simulation", fontsize=11, fontweight="bold", ha="center")
    ax.text(2.5, layer1_y - 0.15, "OpenModelica FMU\n(FMI 2.0 Co-Simulation)", fontsize=9, ha="center")
    ax.text(5, layer1_y - 0.15, "ThermoPower +\nCoolProp (Span-Wagner CO₂ EOS)", fontsize=9, ha="center")
    ax.text(7.5, layer1_y - 0.15, "CVODE solver\nembedded", fontsize=9, ha="center")

    # Layer 2 (middle): Gymnasium Interface
    layer2_y = 5.5
    box_w, box_h = 1.8, 0.7
    _add_box(ax, (2, layer2_y - box_h/2), box_w, box_h, "SCO2FMUEnv\n(FMU path)", "#D5F5E3")
    _add_box(ax, (5.1, layer2_y - box_h/2), box_w, box_h, "MLPSurrogateEnv\n(GPU-vectorized)", "#D5F5E3")
    _add_box(ax, (7.5, layer2_y - 0.5), 1.5, 0.5, "FNO (R²=1.000)\nvalidated, not used for RL", "#E8DAEF")

    # Arrows Layer 1 → Layer 2
    _add_arrow(ax, (5, layer1_y - 0.5), (5, layer2_y + 0.4), "obs, actions")
    _add_arrow(ax, (3.5, layer1_y - 0.5), (2.9, layer2_y + 0.4))
    _add_arrow(ax, (6.5, layer1_y - 0.5), (6, layer2_y + 0.4))

    # Layer 3 (bottom): PPO agents
    layer3_y = 2.5
    _add_box(ax, (1.5, layer3_y - 0.35), 1.5, 0.7, "PPO Agent\n(FMU path)", "#FDEBD0")
    _add_box(ax, (5.2, layer3_y - 0.35), 1.5, 0.7, "PPO Agent\n(MLP path, 1024 envs)", "#FDEBD0")
    _add_box(ax, (3.5, layer3_y - 1.2), 2.2, 0.5, "Lagrangian Safety Layer", "#FADBD8")
    _add_box(ax, (3.5, layer3_y - 2.0), 2.5, 0.5,
             "4 actuators: bypass, IGV, inventory, cooling", "#D6EAF8")

    # Arrows Layer 2 → Layer 3
    _add_arrow(ax, (2.9, layer2_y - 0.5), (2.25, layer3_y + 0.4))
    _add_arrow(ax, (6, layer2_y - 0.5), (5.95, layer3_y + 0.4))
    _add_arrow(ax, (4.6, layer3_y - 0.5), (4.6, layer3_y - 1.0))
    _add_arrow(ax, (4.6, layer3_y - 1.5), (4.6, layer3_y - 1.85))

    # Key numbers
    ax.text(8.5, 7.2, "76,600 LHS\ntrajectories", fontsize=8, ha="center",
            bbox=dict(boxstyle="round", facecolor="#EBF5FB"))
    ax.text(8.5, 6.5, "55M transitions", fontsize=8, ha="center",
            bbox=dict(boxstyle="round", facecolor="#EBF5FB"))
    ax.text(8.5, 4.2, "5M steps\n250k fps", fontsize=8, ha="center",
            bbox=dict(boxstyle="round", facecolor="#D5F5E3"))

    ax.set_title("sCO₂ RL System Architecture — 3-Layer Design", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = FIG_DIR / "system_architecture.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_training_pipeline() -> Path:
    """Figure 2: Left-to-right training pipeline flowchart."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    # Color scheme: Blue=data, Green=training, Orange=eval/deploy, Purple=surrogate
    colors = {"data": "#D6EAF8", "train": "#D5F5E3", "eval": "#FDEBD0", "surrogate": "#E8DAEF"}

    boxes = [
        (1.0, 3.0, 1.8, 1.2, "FMU Simulation\nLHS Sampling", colors["data"]),
        (3.2, 3.0, 2.0, 1.2, "76,600 trajectories\n55M (s,a,s') pairs", colors["data"]),
        (5.6, 3.0, 2.2, 1.2, "MLP Training\n55M pairs, 20 epochs\nval_loss=5e-6, 8.5 min", colors["train"]),
        (8.2, 3.0, 1.4, 1.2, "mlp_step.pt", colors["surrogate"]),
        (10.0, 3.0, 2.2, 1.2, "PPO Training\n5M steps, 1024 envs\n250k fps, 23 min", colors["train"]),
        (12.6, 3.0, 1.2, 1.2, "best_policy.pt", colors["surrogate"]),
        (10.0, 1.2, 2.4, 0.9, "Evaluation vs ZN-PID\n18.5× tracking improvement", colors["eval"]),
        (12.8, 1.2, 1.0, 0.9, "TensorRT FP16\np99=0.046ms", colors["eval"]),
    ]

    for x, y, w, h, text, color in boxes:
        _add_box(ax, (x, y - h/2), w, h, text, color, fontsize=8)

    # FNO branch
    _add_box(ax, (3.2, 4.8), 1.8, 0.6, "trajectories", colors["data"])
    _add_box(ax, (5.2, 4.8), 1.6, 0.6, "FNO training\nR²=1.000", colors["surrogate"])
    _add_arrow(ax, (4.1, 3.6), (4.1, 4.5))
    _add_arrow(ax, (5.0, 4.5), (5.0, 3.6))

    # Main flow arrows
    arrows = [
        ((2.8, 3), (3.1, 3)),
        ((5.2, 3), (5.5, 3)),
        ((7.4, 3), (7.9, 3)),
        ((9.6, 3), (9.9, 3)),
        ((11.4, 3), (12.5, 3)),
        ((11.2, 2.4), (11.2, 1.65)),
        ((13.0, 1.65), (13.3, 1.65)),
    ]
    for (sx, sy), (ex, ey) in arrows:
        _add_arrow(ax, (sx, sy), (ex, ey))

    ax.set_title("sCO₂ RL Training Pipeline — FMU → MLP Surrogate → PPO → Deployment",
                fontsize=11, fontweight="bold")
    plt.tight_layout()
    out_path = FIG_DIR / "training_pipeline.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    print("Generating architecture diagrams...")
    p1 = generate_system_architecture()
    print(f"  -> {p1}")
    p2 = generate_training_pipeline()
    print(f"  -> {p2}")
    print("Done.")


if __name__ == "__main__":
    main()
