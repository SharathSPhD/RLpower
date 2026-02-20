"""Generate thermodynamic state tables for each curriculum scenario."""
from __future__ import annotations
import json, sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# MLP state variable definitions
STATE_NAMES = [
    "T_comp_in", "P_high", "T_turb_in", "T_hot_in", "T_hot_out",
    "P_low", "T_regen", "T_comp_out", "W_turb", "W_comp",
    "eta", "p_outlet", "Q_in", "v14"
]
OBS_LO = np.array([31.5, 70., 300., 200., 200., 70., 100., 31.5, 5., 0.5, 0.85, 0.90, 5., 15.], dtype=np.float32)
OBS_HI = np.array([42., 120., 1100., 1100., 1100., 120., 175., 42., 25., 8., 0.92, 0.95, 120., 21.], dtype=np.float32)

SCENARIOS = {
    "phase0_steady": {
        "label": "Phase 0 – Steady-State (Design Point)",
        "initial_norm": [0.55, 0.5, 0.45, 0.45, 0.45, 0.5, 0.5, 0.55, 0.55, 0.5, 0.5, 0.5, 0.45, 0.5],
    },
    "phase1_partial": {
        "label": "Phase 1 – Partial Load (70% Rated)",
        "initial_norm": [0.55, 0.35, 0.35, 0.35, 0.35, 0.45, 0.45, 0.55, 0.35, 0.35, 0.45, 0.5, 0.35, 0.5],
    },
    "phase4_rejection": {
        "label": "Phase 4 – Load Rejection (−50%)",
        "initial_norm": [0.55, 0.5, 0.45, 0.45, 0.45, 0.5, 0.5, 0.55, 0.55, 0.5, 0.5, 0.5, 0.45, 0.5],
    },
    "phase5_startup": {
        "label": "Phase 5 – Cold Startup (Near-Critical)",
        "initial_norm": [0.1, 0.25, 0.15, 0.15, 0.15, 0.25, 0.15, 0.1, 0.1, 0.15, 0.3, 0.3, 0.1, 0.3],
    },
}

def load_mlp():
    """Load MLP model and normalisation stats."""
    import torch
    model_path = ROOT / "artifacts" / "surrogate" / "mlp_step.pt"
    norm_path = ROOT / "artifacts" / "surrogate" / "mlp_step_norm.npz"
    
    class MLPStepPredictor(torch.nn.Module):
        def __init__(self, n_state, n_action, hidden=512, n_layers=4):
            super().__init__()
            in_dim = n_state + n_action
            layers = [torch.nn.Linear(in_dim, hidden), torch.nn.SiLU()]
            for _ in range(n_layers - 1):
                layers += [torch.nn.Linear(hidden, hidden), torch.nn.SiLU()]
            layers.append(torch.nn.Linear(hidden, n_state))
            self.net = torch.nn.Sequential(*layers)
        def forward(self, s, a):
            x = torch.cat([s, a], dim=-1)
            return s + self.net(x)
    
    model = MLPStepPredictor(14, 4)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    
    norms = np.load(norm_path)
    return model, norms

def run_scenario(model, norms, initial_norm, n_steps=50):
    """Run MLP scenario and return trajectory."""
    import torch
    s_mean = torch.tensor(norms["s_mean"], dtype=torch.float32)
    s_std  = torch.tensor(norms["s_std"],  dtype=torch.float32)
    a_mean = torch.tensor(norms["a_mean"], dtype=torch.float32)
    a_std  = torch.tensor(norms["a_std"],  dtype=torch.float32)
    sp_mean = torch.tensor(norms.get("sp_mean", norms["s_mean"]), dtype=torch.float32)
    sp_std  = torch.tensor(norms.get("sp_std",  norms["s_std"]),  dtype=torch.float32)
    
    # Convert norm [0,1] to physical state
    obs_lo = torch.tensor(OBS_LO, dtype=torch.float32)
    obs_hi = torch.tensor(OBS_HI, dtype=torch.float32)
    state = obs_lo + torch.tensor(initial_norm, dtype=torch.float32) * (obs_hi - obs_lo)
    
    trajectory = [state.numpy()]
    for _ in range(n_steps):
        # Fixed "maintain" action (midpoint)
        a_phys = torch.ones(4, dtype=torch.float32) * 0.5
        s_n = (state - s_mean) / (s_std + 1e-8)
        a_n = (a_phys - a_mean) / (a_std + 1e-8)
        with torch.no_grad():
            sp_n = model(s_n.unsqueeze(0), a_n.unsqueeze(0)).squeeze(0)
        state = sp_n * sp_std + sp_mean
        state = state.clamp(obs_lo, obs_hi)
        trajectory.append(state.numpy())
    return np.array(trajectory)

def compute_entropy_approx(T_C, P_bar):
    """Approximate CO2 entropy relative to critical point (kJ/kg/K)."""
    T_K = T_C + 273.15
    P_Pa = P_bar * 1e5
    T_ref = 304.13  # critical point K
    P_ref = 73.8e5  # critical point Pa
    cp_avg = 1.2    # kJ/kg/K for sCO2
    R = 0.1889      # kJ/kg/K
    s = cp_avg * np.log(T_K / T_ref) - R * np.log(P_Pa / P_ref)
    return s

def main():
    print("Loading MLP model...")
    try:
        model, norms = load_mlp()
    except Exception as e:
        print(f"Could not load MLP: {e}")
        return
    
    tables = {}
    for phase_key, cfg in SCENARIOS.items():
        print(f"  Running {cfg['label']}...")
        traj = run_scenario(model, norms, cfg["initial_norm"], n_steps=50)
        
        # Initial state (t=0)
        s0 = traj[0]
        # Steady-state (last 10 steps average)
        ss = traj[-10:].mean(axis=0)
        
        tables[phase_key] = {
            "label": cfg["label"],
            "initial": {
                "T_comp_in_C":  float(s0[0]),
                "P_high_bar":   float(s0[1]),
                "T_turb_in_C":  float(s0[2]),
                "T_hot_in_C":   float(s0[3]),
                "P_low_bar":    float(s0[5]),
                "T_regen_C":    float(s0[6]),
                "W_turb_MW":    float(s0[8]),
                "W_comp_MW":    float(s0[9]),
                "W_net_MW":     float(s0[8] - s0[9]),
                "eta":          float(s0[10]),
                "Q_in_MW":      float(s0[12]),
                "s_low_kJkgK":  float(compute_entropy_approx(float(s0[0]), float(s0[5]))),
                "s_high_kJkgK": float(compute_entropy_approx(float(s0[2]), float(s0[1]))),
            },
            "steady_state": {
                "T_comp_in_C":  float(ss[0]),
                "P_high_bar":   float(ss[1]),
                "T_turb_in_C":  float(ss[2]),
                "T_hot_in_C":   float(ss[3]),
                "P_low_bar":    float(ss[5]),
                "T_regen_C":    float(ss[6]),
                "W_turb_MW":    float(ss[8]),
                "W_comp_MW":    float(ss[9]),
                "W_net_MW":     float(ss[8] - ss[9]),
                "eta":          float(ss[10]),
                "Q_in_MW":      float(ss[12]),
                "s_low_kJkgK":  float(compute_entropy_approx(float(ss[0]), float(ss[5]))),
                "s_high_kJkgK": float(compute_entropy_approx(float(ss[2]), float(ss[1]))),
            }
        }
    
    out_path = ROOT / "data" / "thermo_state_tables.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(tables, f, indent=2)
    print(f"Saved to {out_path}")
    
    # Print LaTeX table
    print("\n=== LATEX TABLE ===")
    print(r"""\begin{table}[h]
\centering
\caption{Thermodynamic state summary for key curriculum scenarios (MLP surrogate steady-state values).
         Entropy computed relative to CO$_2$ critical point (304.13~K, 73.8~bar).}
\label{tab:thermo_states}
\begin{tabular}{lcccccc}
\toprule
Scenario & $T_{\text{comp,in}}$ (\textdegree{}C) & $P_{\text{high}}$ (bar) & $T_{\text{turb,in}}$ (\textdegree{}C) & $W_{\text{net}}$ (MW) & $Q_{\text{in}}$ (MW) & $\eta$ \\
\midrule""")
    for ph, d in tables.items():
        ss = d["steady_state"]
        print(f"{d['label'].split('–')[0].strip()} & {ss['T_comp_in_C']:.1f} & {ss['P_high_bar']:.1f} & {ss['T_turb_in_C']:.1f} & {ss['W_net_MW']:.2f} & {ss['Q_in_MW']:.1f} & {ss['eta']:.3f} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")

if __name__ == "__main__":
    main()
