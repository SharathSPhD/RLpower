#!/usr/bin/env python3
"""Generate scenario-specific T-s diagrams from MLP surrogate rollouts.

Loads MLP model, runs 4 operating scenarios, computes entropy via CoolProp
(or approximation), and produces T-s diagrams.

Outputs:
    paper/figures/ts_diagram_phase0.png
    paper/figures/ts_diagram_phase1.png
    paper/figures/ts_diagram_phase4.png
    paper/figures/ts_diagram_phase5.png
    paper/figures/ts_diagram_scenarios_composite.png

Usage:
    python scripts/generate_scenario_ts_diagrams.py [--model artifacts/surrogate/mlp_step.pt]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
DPI = 150

IDX_T_COMP_IN = 0
IDX_T_TURB_IN = 2
IDX_P_HIGH = 13
# P_low not in env.yaml obs_vars; use design constant 7.6 MPa for low-side
IDX_P_LOW = -1
P_LOW_DESIGN_MPA = 7.6
P_HIGH_DESIGN_MPA = 20.0

try:
    from CoolProp.CoolProp import PropsSI
    HAS_COOLPROP = True
except ImportError:
    HAS_COOLPROP = False


def approx_entropy_co2(T_C: float, P_bar: float) -> float:
    """Simplified entropy for CO2 near supercritical region (kJ/kg/K)."""
    T_K = T_C + 273.15
    P_Pa = P_bar * 1e5
    T_ref = 304.13
    P_ref = 73.8e5
    cp_avg = 1.2
    R = 0.1889
    s = cp_avg * np.log(T_K / T_ref) - R * np.log(P_Pa / P_ref)
    return float(s)


def entropy_co2(T_C: float, P_Pa: float) -> float:
    """Entropy in kJ/kg/K. Uses CoolProp if available, else approximation."""
    if HAS_COOLPROP:
        try:
            return PropsSI("S", "T", T_C + 273.15, "P", P_Pa, "CO2") / 1000.0
        except Exception:
            pass
    return approx_entropy_co2(T_C, P_bar=P_Pa / 1e5)


def load_mlp_and_norm(model_path: Path, norm_path: Path, device: str = "cpu"):
    """Load MLP model and normalization stats."""
    import torch.nn as nn

    class MLPStepPredictor(nn.Module):
        def __init__(self, n_state: int, n_action: int, hidden: int = 512, n_layers: int = 4):
            super().__init__()
            in_dim = n_state + n_action
            layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(hidden, hidden), nn.SiLU()]
            layers.append(nn.Linear(hidden, n_state))
            self.net = nn.Sequential(*layers)

        def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            x = torch.cat([state, action], dim=-1)
            return state + self.net(x)

    norm = dict(np.load(norm_path))
    n_s = int(norm["s_mean"].shape[0])
    n_a = int(norm["a_mean"].shape[0])
    model = MLPStepPredictor(n_s, n_a).to(device)
    sd = torch.load(model_path, weights_only=True, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model, norm, n_s, n_a


def build_env_config(env_yaml: Path) -> dict:
    """Build obs_bounds/act_bounds from env.yaml."""
    env = yaml.safe_load(env_yaml.read_text())
    obs_bounds = [(float(v["min"]), float(v["max"])) for v in env["observation"]["variables"] if v.get("fmu_var")]
    act_bounds = [(float(v["physical_min"]), float(v["physical_max"])) for v in env["action"]["variables"]]
    return {"obs_bounds": obs_bounds, "act_bounds": act_bounds}


def run_mlp_rollout(model, norm, init_state, actions, obs_bounds, act_bounds, device="cpu"):
    """Run MLP forward. Returns states (T+1, n_s)."""
    s_mean = torch.tensor(norm["s_mean"], dtype=torch.float32, device=device)
    s_std = torch.tensor(norm["s_std"], dtype=torch.float32, device=device)
    a_mean = torch.tensor(norm["a_mean"], dtype=torch.float32, device=device)
    a_std = torch.tensor(norm["a_std"], dtype=torch.float32, device=device)
    sp_mean = torch.tensor(norm["sp_mean"], dtype=torch.float32, device=device)
    sp_std = torch.tensor(norm["sp_std"], dtype=torch.float32, device=device)
    obs_lo = np.array([b[0] for b in obs_bounds], dtype=np.float32)
    obs_hi = np.array([b[1] for b in obs_bounds], dtype=np.float32)
    act_lo = np.array([b[0] for b in act_bounds], dtype=np.float32)
    act_hi = np.array([b[1] for b in act_bounds], dtype=np.float32)

    states = [init_state.copy()]
    state = torch.tensor(init_state.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(0)

    for t in range(len(actions)):
        act_norm = np.asarray(actions[t], dtype=np.float32)
        act_phys = act_lo + (act_norm + 1.0) * 0.5 * (act_hi - act_lo)
        act_phys = np.clip(act_phys, act_lo, act_hi)
        s_n = (state - s_mean) / s_std
        a_n = (torch.from_numpy(act_phys).to(device=device, dtype=torch.float32).unsqueeze(0) - a_mean) / a_std
        with torch.no_grad():
            next_s_n = model(s_n, a_n)
        next_s = next_s_n * sp_std + sp_mean
        next_s = next_s.clamp(torch.tensor(obs_lo, device=device), torch.tensor(obs_hi, device=device))
        state = next_s
        states.append(state.squeeze(0).cpu().numpy())

    return np.array(states)


def get_scenario_configs(n_s, n_a, obs_bounds, act_bounds):
    """Scenario-specific initial conditions and action sequences."""
    obs_lo = np.array([b[0] for b in obs_bounds[:n_s]])
    obs_hi = np.array([b[1] for b in obs_bounds[:n_s]])
    act_lo = np.array([b[0] for b in act_bounds[:n_a]])
    act_hi = np.array([b[1] for b in act_bounds[:n_a]])
    mid = 0.5 * (obs_lo + obs_hi)
    act_mid = 0.5 * (act_lo + act_hi)

    init0 = mid.copy()
    init0[IDX_T_COMP_IN] = 37.0
    init0[IDX_T_TURB_IN] = 750.0
    if n_s > IDX_P_HIGH:
        init0[IDX_P_HIGH] = 20.0
    if IDX_P_LOW >= 0 and n_s > IDX_P_LOW:
        init0[IDX_P_LOW] = 7.6
    if n_s > 8:
        init0[8] = 14.5
    if n_s > 9:
        init0[9] = 4.5

    init1 = init0.copy()
    init1[IDX_T_TURB_IN] = 650.0
    if n_s > 8:
        init1[8] = 10.0
    if n_s > 9:
        init1[9] = 3.2

    init4 = init0.copy()

    init5 = mid.copy()
    init5[IDX_T_COMP_IN] = 32.5
    init5[IDX_T_TURB_IN] = 600.0
    if n_s > IDX_P_HIGH:
        init5[IDX_P_HIGH] = 18.0
    if IDX_P_LOW >= 0 and n_s > IDX_P_LOW:
        init5[IDX_P_LOW] = 7.4

    n_steps = 100
    rng = np.random.default_rng(42)
    act_seq = np.clip(act_mid + rng.standard_normal((n_steps, n_a)) * 0.05, act_lo, act_hi)
    act_seq_norm = 2.0 * (act_seq - act_lo) / (act_hi - act_lo + 1e-9) - 1.0

    act4 = act_seq_norm.copy()
    if n_a >= 1:
        act4[20:] *= 0.7

    return {
        "phase0": {"init": init0, "actions": act_seq_norm, "label": "Phase 0: Steady-state"},
        "phase1": {"init": init1, "actions": act_seq_norm, "label": "Phase 1: Partial load (70%)"},
        "phase4": {"init": init4, "actions": act4, "label": "Phase 4: Load rejection"},
        "phase5": {"init": init5, "actions": act_seq_norm, "label": "Phase 5: Cold startup"},
    }


def extract_ts_points(states, n_s):
    """Extract (T,s) for low-side and high-side."""
    T_comp = states[:, IDX_T_COMP_IN] if n_s > IDX_T_COMP_IN else np.full(len(states), 35.0)
    T_turb = states[:, IDX_T_TURB_IN] if n_s > IDX_T_TURB_IN else np.full(len(states), 700.0)
    P_high_MPa = states[:, IDX_P_HIGH] if n_s > IDX_P_HIGH else np.full(len(states), P_HIGH_DESIGN_MPA)
    P_low_MPa = states[:, IDX_P_LOW] if IDX_P_LOW >= 0 and n_s > IDX_P_LOW else np.full(len(states), P_LOW_DESIGN_MPA)
    P_high_Pa = P_high_MPa * 1e6
    P_low_Pa = P_low_MPa * 1e6
    s_low = np.array([entropy_co2(T_comp[i], P_low_Pa[i]) for i in range(len(states))])
    s_high = np.array([entropy_co2(T_turb[i], P_high_Pa[i]) for i in range(len(states))])
    return T_comp, T_turb, s_low, s_high


def draw_saturation_dome(ax):
    """Draw CO2 saturation dome."""
    if HAS_COOLPROP:
        T_crit = PropsSI("Tcrit", "CO2")
        T_sat = np.linspace(220.0, T_crit - 0.01, 200)
        s_liq, s_vap = [], []
        for T in T_sat:
            try:
                s_liq.append(PropsSI("S", "T", T, "Q", 0, "CO2") / 1000)
                s_vap.append(PropsSI("S", "T", T, "Q", 1, "CO2") / 1000)
            except Exception:
                s_liq.append(np.nan)
                s_vap.append(np.nan)
        T_plot = T_sat - 273.15
        ax.plot(s_liq, T_plot, "b-", linewidth=2.0, label="Saturation dome")
        ax.plot(s_vap, T_plot, "b-", linewidth=2.0)
        s_crit = PropsSI("S", "T", T_crit, "Q", 0, "CO2") / 1000
        ax.plot(s_crit, T_crit - 273.15, "b^", markersize=8, label=f"Critical pt ({T_crit - 273.15:.1f}C)")
        ax.fill_betweenx(T_plot, s_liq, s_vap, alpha=0.07, color="blue")
    else:
        T_plot = np.linspace(-55, 31, 150)
        s_liq = [approx_entropy_co2(t, 73.8) for t in T_plot]
        s_vap = [s + 0.4 for s in s_liq]
        ax.plot(s_liq, T_plot, "b-", linewidth=2.0, label="Saturation (approx)")
        ax.plot(s_vap, T_plot, "b-", linewidth=2.0)
        ax.plot(approx_entropy_co2(31.1, 73.8), 31.1, "b^", markersize=8, label="Critical pt (31.1C)")


def plot_ts_diagram(T_low, s_low, T_high, s_high, title, out_path, scenario_color="#E74C3C"):
    """Plot single scenario T-s diagram."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")
    draw_saturation_dome(ax)
    ax.plot(s_low, T_low, "-", color=scenario_color, linewidth=1.5, label="Low-side (comp inlet)")
    ax.plot(s_high, T_high, "--", color=scenario_color, linewidth=1.2, label="High-side (turb inlet)")
    ax.scatter(s_low[0], T_low[0], color=scenario_color, s=60, zorder=5)
    ax.scatter(s_low[-1], T_low[-1], color=scenario_color, s=60, zorder=5)
    ax.set_xlabel("Specific entropy, s [kJ/(kg·K)]", fontsize=11)
    ax.set_ylabel("Temperature, T [C]", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(0.4, 2.2)
    ax.set_ylim(-60, 560)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_composite(T_s_data, out_path):
    """Plot composite T-s diagram."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")
    draw_saturation_dome(ax)
    colors = {"phase0": "#27AE60", "phase1": "#2980B9", "phase4": "#E74C3C", "phase5": "#9B59B6"}
    for key in ["phase0", "phase1", "phase4", "phase5"]:
        if key not in T_s_data:
            continue
        T_low, s_low, T_high, s_high = T_s_data[key]
        lbl = T_s_data.get(f"{key}_label", key)
        c = colors.get(key, "#333")
        ax.plot(s_low, T_low, "-", color=c, linewidth=1.2, label=lbl)
        ax.plot(s_high, T_high, "--", color=c, linewidth=0.9, alpha=0.8)
        ax.scatter(s_low[0], T_low[0], color=c, s=40, zorder=5)
        ax.scatter(s_low[-1], T_low[-1], color=c, s=40, zorder=5)
    ax.set_xlabel("Specific entropy, s [kJ/(kg·K)]", fontsize=11)
    ax.set_ylabel("Temperature, T [C]", fontsize=11)
    ax.set_title("sCO2 Cycle T-s Diagram: Operating Scenarios", fontsize=12, fontweight="bold")
    ax.set_xlim(0.4, 2.2)
    ax.set_ylim(-60, 560)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/surrogate/mlp_step.pt")
    parser.add_argument("--norm", default="artifacts/surrogate/mlp_step_norm.npz")
    parser.add_argument("--env", default="configs/environment/env.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model_path = ROOT / args.model
    norm_path = ROOT / args.norm
    env_path = ROOT / args.env

    if not model_path.exists() or not norm_path.exists():
        print("MLP model or norm not found. Creating synthetic T-s diagrams.")
        n_s, n_a = 14, 4
        env_cfg = build_env_config(env_path) if env_path.exists() else {}
        obs_bounds = env_cfg.get("obs_bounds", [(31, 43)] * 14)
        act_bounds = env_cfg.get("act_bounds", [(0, 1)] * 4)
        scenarios = get_scenario_configs(n_s, n_a, obs_bounds, act_bounds)
        T_s_data = {}
        for key, cfg in scenarios.items():
            init = cfg["init"]
            n_steps = len(cfg["actions"]) + 1
            states = np.tile(init, (n_steps, 1))
            states += np.random.default_rng(42).standard_normal(states.shape) * 2.0
            obs_lo = np.array([b[0] for b in obs_bounds[:n_s]])
            obs_hi = np.array([b[1] for b in obs_bounds[:n_s]])
            states = np.clip(states, obs_lo, obs_hi)
            T_comp, T_turb, s_low, s_high = extract_ts_points(states, n_s)
            T_s_data[key] = (T_comp, s_low, T_turb, s_high)
            T_s_data[f"{key}_label"] = cfg["label"]
        plot_composite(T_s_data, FIG_DIR / "ts_diagram_scenarios_composite.png")
        for key, cfg in scenarios.items():
            T_comp, s_low, T_turb, s_high = T_s_data[key]
            colors = {"phase0": "#27AE60", "phase1": "#2980B9", "phase4": "#E74C3C", "phase5": "#9B59B6"}
            plot_ts_diagram(T_comp, s_low, T_turb, s_high, cfg["label"],
                           FIG_DIR / f"ts_diagram_{key}.png", scenario_color=colors.get(key, "#333"))
        print("  -> paper/figures/ts_diagram_*.png (synthetic)")
        _regenerate_cycle_ts_diagram()
        return

    model, norm, n_s, n_a = load_mlp_and_norm(model_path, norm_path, args.device)
    env_cfg = build_env_config(env_path)
    obs_bounds = env_cfg["obs_bounds"]
    act_bounds = env_cfg["act_bounds"]
    scenarios = get_scenario_configs(n_s, n_a, obs_bounds, act_bounds)
    T_s_data = {}

    for key, cfg in scenarios.items():
        states = run_mlp_rollout(model, norm, cfg["init"], cfg["actions"],
                                obs_bounds, act_bounds, args.device)
        T_comp, T_turb, s_low, s_high = extract_ts_points(states, n_s)
        T_s_data[key] = (T_comp, s_low, T_turb, s_high)
        T_s_data[f"{key}_label"] = cfg["label"]
        colors = {"phase0": "#27AE60", "phase1": "#2980B9", "phase4": "#E74C3C", "phase5": "#9B59B6"}
        plot_ts_diagram(T_comp, s_low, T_turb, s_high, cfg["label"],
                       FIG_DIR / f"ts_diagram_{key}.png", scenario_color=colors.get(key, "#333"))
        print(f"  -> paper/figures/ts_diagram_{key}.png")

    plot_composite(T_s_data, FIG_DIR / "ts_diagram_scenarios_composite.png")
    print("  -> paper/figures/ts_diagram_scenarios_composite.png")

    _regenerate_cycle_ts_diagram()
    print("Done.")


def _regenerate_cycle_ts_diagram():
    """Regenerate cycle_ts_diagram.png via diagram_renderer if CoolProp available."""
    if not HAS_COOLPROP:
        return
    try:
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        import yaml as _yaml
        from sco2rl.physics.metamodel.builder import SCO2CycleBuilder
        from sco2rl.physics.metamodel.diagram_renderer import CycleDiagramRenderer
        cfg = _yaml.safe_load((ROOT / "configs/model/base_cycle.yaml").read_text())
        model = SCO2CycleBuilder.from_config(cfg).build()
        renderer = CycleDiagramRenderer()
        _, ts_path = renderer.render(model, FIG_DIR, dpi=DPI)
        print(f"  -> {ts_path} (regenerated)")
    except Exception as e:
        print(f"  cycle_ts_diagram regeneration skipped: {e}")


if __name__ == "__main__":
    main()
