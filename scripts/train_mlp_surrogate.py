"""Train a simple MLP step predictor for sCO2 RL.

The FNO is trained on full 719-step trajectories (non-causal, sequence-to-sequence),
making it unsuitable for one-step RL prediction. This script extracts all (s,a,s')
tuples from the trajectory dataset and trains a simple MLP residual predictor
which is directly suited for RL.

Usage:
    python scripts/train_mlp_surrogate.py [--data artifacts/trajectories/lhs_100k.h5]
                                           [--epochs 60] [--batch 8192]
                                           [--out artifacts/surrogate/mlp_step.pt]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLPStepPredictor(nn.Module):
    """Residual MLP: predicts (next_state - state) from (state, action).

    Residual formulation stabilises training because most state variables
    change slowly between timesteps.
    """

    def __init__(self, n_state: int, n_action: int, hidden: int = 512, n_layers: int = 4):
        super().__init__()
        in_dim = n_state + n_action
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, n_state))
        self.net = nn.Sequential(*layers)

        # Near-zero init on last layer so residuals start small
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        delta = self.net(x)
        return state + delta


def load_transitions(data_path: str, max_traj: int | None = None):
    print(f"Loading trajectories from {data_path} ...")
    t0 = time.time()
    with h5py.File(data_path, "r") as f:
        states = np.array(f["states"])    # (N, T, n_s) or (N, T+1, n_s)
        actions = np.array(f["actions"])  # (N, T, n_a)

    if max_traj is not None:
        states = states[:max_traj]
        actions = actions[:max_traj]

    N, T_s, n_s = states.shape
    _N, T_a, n_a = actions.shape
    T = min(T_s, T_a + 1)

    s  = states[:, :T-1, :].reshape(-1, n_s)   # (N*(T-1), n_s)
    a  = actions[:, :T-1, :].reshape(-1, n_a)
    sp = states[:, 1:T, :].reshape(-1, n_s)

    print(f"  {N} trajectories -> {len(s)} transitions in {time.time()-t0:.1f}s")
    print(f"  n_state={n_s}, n_action={n_a}")
    return s, a, sp, n_s, n_a


def compute_normalisation(s: np.ndarray, a: np.ndarray, sp: np.ndarray):
    s_mean  = s.mean(0).astype(np.float32)
    s_std   = s.std(0).astype(np.float32)
    s_std   = np.where(s_std < 1e-6, 1.0, s_std)
    a_mean  = a.mean(0).astype(np.float32)
    a_std   = a.std(0).astype(np.float32)
    a_std   = np.where(a_std < 1e-6, 1.0, a_std)
    sp_mean = sp.mean(0).astype(np.float32)
    sp_std  = sp.std(0).astype(np.float32)
    sp_std  = np.where(sp_std < 1e-6, 1.0, sp_std)
    return dict(s_mean=s_mean, s_std=s_std, a_mean=a_mean, a_std=a_std,
                sp_mean=sp_mean, sp_std=sp_std)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/trajectories/lhs_100k.h5")
    parser.add_argument("--max_traj", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--out", default="artifacts/surrogate/mlp_step.pt")
    parser.add_argument("--norm_out", default="artifacts/surrogate/mlp_step_norm.npz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    s_np, a_np, sp_np, n_s, n_a = load_transitions(args.data, max_traj=args.max_traj)
    norm = compute_normalisation(s_np, a_np, sp_np)

    s_n  = (s_np  - norm["s_mean"])  / norm["s_std"]
    a_n  = (a_np  - norm["a_mean"])  / norm["a_std"]
    sp_n = (sp_np - norm["sp_mean"]) / norm["sp_std"]

    S  = torch.tensor(s_n,  dtype=torch.float32)
    A  = torch.tensor(a_n,  dtype=torch.float32)
    SP = torch.tensor(sp_n, dtype=torch.float32)

    N_total = len(S)
    N_val   = int(N_total * args.val_frac)
    idx = torch.randperm(N_total)
    val_idx, tr_idx = idx[:N_val], idx[N_val:]

    tr_ds  = TensorDataset(S[tr_idx], A[tr_idx], SP[tr_idx])
    val_ds = TensorDataset(S[val_idx], A[val_idx], SP[val_idx])
    tr_loader  = DataLoader(tr_ds,  batch_size=args.batch, shuffle=True,
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch*4, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"Train: {len(tr_ds):,} | Val: {len(val_ds):,}")

    model = MLPStepPredictor(n_s, n_a, hidden=args.hidden, n_layers=args.n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MLP parameters: {n_params:,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    best_val  = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for s_b, a_b, sp_b in tr_loader:
            s_b, a_b, sp_b = s_b.to(device), a_b.to(device), sp_b.to(device)
            pred = model(s_b, a_b)
            loss = nn.functional.mse_loss(pred, sp_b)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(s_b)
        sched.step()
        tr_loss /= len(tr_ds)

        model.train(False)
        val_loss = 0.0
        with torch.no_grad():
            for s_b, a_b, sp_b in val_loader:
                s_b, a_b, sp_b = s_b.to(device), a_b.to(device), sp_b.to(device)
                pred = model(s_b, a_b)
                val_loss += nn.functional.mse_loss(pred, sp_b).item() * len(s_b)
        val_loss /= len(val_ds)

        if val_loss < best_val:
            best_val  = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs}: tr={tr_loss:.6f} val={val_loss:.6f} "
                  f"lr={sched.get_last_lr()[0]:.1e} best={best_val:.6f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_path)
    np.savez(args.norm_out, **norm)
    print(f"\nBest val_loss={best_val:.6f}")
    print(f"Saved model -> {out_path}")
    print(f"Saved norms -> {args.norm_out}")


if __name__ == "__main__":
    main()
