#!/usr/bin/env python3
"""Plot thermodynamic state trajectories from preflight_100.h5.

The HDF5 file has:
  states   (100, 120, 14)  — 100 trajectories × 120 steps × 14 state vars
  metadata (100, 3)        — [T_exhaust_init, mdot_exhaust, W_net_setpoint]

Variable order (from env.yaml observation section, first 14 channels):
  0  T_compressor_inlet  [K]
  1  T_turbine_inlet     [K]
  2  P_high              [Pa]  (high pressure side)
  3  P_low               [Pa]  (low pressure side)
  4  W_net               [W]
  5  eta_thermal         [-]
  6  m_flow              [kg/s]
  7  T_recuperator_hot_in [K]
  8  T_recuperator_hot_out [K]
  9  T_recuperator_cold_in [K]
  10 T_recuperator_cold_out [K]
  11 T_precooler_out     [K]
  12 mdot_bypass         [kg/s]
  13 surge_margin        [-]

Outputs: paper/figures/thermo_trajectories.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

H5_PATH = ROOT / "artifacts/trajectories/preflight_100.h5"

# Variable indices in the state array
IDX = {
    "T_comp_in": 0,   # K
    "T_turb_in": 1,   # K
    "P_high": 2,       # Pa
    "W_net": 4,        # W
    "eta_thermal": 5,  # -
    "T_recup_hot_out": 8,  # K
}


def to_degC(arr: np.ndarray) -> np.ndarray:
    return arr - 273.15


def to_MW(arr: np.ndarray) -> np.ndarray:
    return arr / 1e6


def categorize_by_exhaust(metadata: np.ndarray) -> dict[str, np.ndarray]:
    """Split trajectory indices by initial exhaust temperature."""
    T_ex = metadata[:, 0]
    low  = np.where(T_ex < 400)[0]
    mid  = np.where((T_ex >= 400) & (T_ex < 800))[0]
    high = np.where(T_ex >= 800)[0]
    return {"Low (<400°C)": low, "Mid (400–800°C)": mid, "High (≥800°C)": high}


def main(h5_path: Path = H5_PATH) -> None:
    if not h5_path.exists():
        print(f"H5 not found: {h5_path}")
        sys.exit(1)

    with h5py.File(h5_path, "r") as f:
        states   = f["states"][:]    # (100, 120, 14)
        metadata = f["metadata"][:]  # (100, 3)

    n_traj, n_steps, n_vars = states.shape
    steps_sec = np.arange(n_steps) * 5  # 5-second simulation steps
    steps_min = steps_sec / 60

    cats = categorize_by_exhaust(metadata)
    cat_colors = {"Low (<400°C)": "#2980B9", "Mid (400–800°C)": "#E67E22", "High (≥800°C)": "#E74C3C"}

    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor("#F8F9FA")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.35)

    plots = [
        ("T_comp_in",  gs[0, 0], "Compressor inlet T [°C]",  to_degC,
         "Must stay > 33°C (CO₂ critical region constraint)", 33.0),
        ("T_turb_in",  gs[0, 1], "Turbine inlet T [°C]",     to_degC, None, None),
        ("P_high",     gs[1, 0], "High-pressure side [MPa]",  lambda x: x / 1e6, None, None),
        ("W_net",      gs[1, 1], "Net power output [MW]",     to_MW,  None, None),
        ("eta_thermal", gs[2, 0], "Thermal efficiency [%]",   lambda x: x * 100, None, None),
        ("T_recup_hot_out", gs[2, 1], "Recuperator hot outlet T [°C]", to_degC, None, None),
    ]

    for var, subplot_spec, ylabel, transform, annotation, threshold in plots:
        ax = fig.add_subplot(subplot_spec)
        ax.set_facecolor("#F9F9F9")
        idx = IDX[var]
        raw = states[:, :, idx]
        vals = transform(raw)

        for cat_name, indices in cats.items():
            if len(indices) == 0:
                continue
            color = cat_colors[cat_name]
            subset = vals[indices]  # (n_cat, n_steps)
            mean_traj = subset.mean(axis=0)
            std_traj  = subset.std(axis=0)
            ax.plot(steps_min, mean_traj, color=color, linewidth=1.8,
                    label=cat_name, alpha=0.9)
            ax.fill_between(steps_min, mean_traj - std_traj, mean_traj + std_traj,
                            color=color, alpha=0.12)

        if threshold is not None:
            ax.axhline(threshold, color="#C0392B", linestyle="--",
                       linewidth=1.0, alpha=0.8)
            if annotation:
                ax.text(steps_min[-1] * 0.02, threshold + 0.5, annotation,
                        fontsize=7, color="#C0392B")

        ax.set_xlabel("Time [min]", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel.split("[")[0].strip(), fontsize=9.5, fontweight="bold")
        ax.grid(True, alpha=0.2)
        if var == "T_comp_in":
            ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle(
        "Thermodynamic State Trajectories — Preflight Evaluation (n=100)\n"
        "Grouped by initial exhaust gas temperature (EAF/BOF source)",
        fontsize=11, fontweight="bold", y=0.98,
    )

    out_path = FIG_DIR / "thermo_trajectories.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    print(f"  trajectories={n_traj}, steps={n_steps}, vars={n_vars}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else H5_PATH)
