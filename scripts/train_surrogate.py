"""Stage 3: FNO supervised training + SKRL PPO on GPU surrogate.

Pipeline:
  1. Load HDF5 trajectory dataset → PyTorch tensors
  2. Train FNO1d dynamics surrogate (MSE loss, Adam, cosine LR)
  3. Run fidelity gate (RMSE + R² on held-out test rollouts)
  4. Launch SurrogateTrainer SKRL PPO on GPU (vectorized)

Usage (inside Docker):
    PYTHONPATH=src python scripts/train_surrogate.py \\
        --dataset artifacts/trajectories/lhs_75k.h5 \\
        --device cuda --verbose 1

    # Skip SKRL PPO (train FNO only):
    PYTHONPATH=src python scripts/train_surrogate.py \\
        --dataset artifacts/trajectories/lhs_75k.h5 \\
        --skip-rl

    # Load pre-trained FNO, run SKRL PPO only:
    PYTHONPATH=src python scripts/train_surrogate.py \\
        --dataset artifacts/trajectories/lhs_75k.h5 \\
        --fno-checkpoint artifacts/checkpoints/fno/best_fno.pt \\
        --skip-rl=false

Config files loaded:
    configs/surrogate/fno_surrogate.yaml
    configs/environment/env.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, random_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_surrogate")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stage 3: FNO training + SKRL PPO")
    p.add_argument(
        "--dataset",
        type=str,
        default="artifacts/trajectories/lhs_75k.h5",
        help="Path to HDF5 trajectory file (states, actions, metadata)",
    )
    p.add_argument(
        "--fno-checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained FNO .pt weights (skips FNO training if given)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device: 'cuda' (default) or 'cpu'",
    )
    p.add_argument(
        "--fno-epochs",
        type=int,
        default=None,
        help="Override FNO training epochs (default: from fno_surrogate.yaml)",
    )
    p.add_argument(
        "--skip-rl",
        action="store_true",
        help="Train FNO only; skip SKRL PPO step",
    )
    p.add_argument(
        "--allow-fidelity-fail",
        action="store_true",
        help="Continue to SKRL PPO even if fidelity gate fails (for smoke/debug runs)",
    )
    p.add_argument(
        "--rl-timesteps",
        type=int,
        default=None,
        help="Override SKRL PPO total_timesteps from config",
    )
    p.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity: 0=silent, 1=progress, 2=debug",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

def load_configs(project_root: Path) -> tuple[dict, dict]:
    """Load fno_surrogate.yaml and env.yaml, return (fno_cfg, env_cfg)."""
    import yaml
    fno_cfg = yaml.safe_load(
        (project_root / "configs/surrogate/fno_surrogate.yaml").read_text()
    )
    env_cfg = yaml.safe_load(
        (project_root / "configs/environment/env.yaml").read_text()
    )
    return fno_cfg, env_cfg


def build_env_config(env_cfg: dict) -> dict:
    """Extract SurrogateEnv-compatible config from env.yaml."""
    obs_section = env_cfg["observation"]
    obs_vars_raw = obs_section["variables"]
    # Raw obs_vars (without stacking) — SurrogateEnv handles history internally
    fmu_obs_vars = [v for v in obs_vars_raw if v.get("fmu_var") is not None]
    obs_vars = [v["fmu_var"] for v in fmu_obs_vars]
    obs_names = [v.get("name", v["fmu_var"]) for v in fmu_obs_vars]
    obs_bounds = {v["fmu_var"]: (v["min"], v["max"]) for v in fmu_obs_vars}

    act_section = env_cfg["action"]
    act_vars_raw = act_section["variables"]
    action_vars = [v["fmu_var"] for v in act_vars_raw]
    action_config = {
        v["fmu_var"]: {
            "phys_min": v["physical_min"],
            "phys_max": v["physical_max"],
            "rate_limit": v.get("rate_limit_per_step", v.get("rate_limit", v.get("rate", 1.0))),
        }
        for v in act_vars_raw
    }

    episode = env_cfg.get("episode", {})
    return {
        "obs_vars": obs_vars,
        "obs_names": obs_names,
        "obs_bounds": obs_bounds,
        "action_vars": action_vars,
        "action_config": action_config,
        "history_steps": obs_section.get("history_steps", 5),
        "step_size": 5.0,
        "episode_max_steps": episode.get("max_steps", 720),
        "reward": env_cfg.get("reward", {}),
        "safety": {},
        "setpoint": {"W_net": 10.0},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(dataset_path: str, device: str) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Load HDF5 trajectory data and build FNO input/target tensors.

    HDF5 schema: /states (N, T, obs_dim), /actions (N, T-1, act_dim)

    FNO1d expects channels-first: (B, C, T)
    - x: (N, obs_dim + act_dim, T-1)  — concat(obs_t, action_t) permuted
    - y: (N, obs_dim, T-1)            — obs_{t+1} permuted

    Returns (x, y, obs_dim, act_dim).
    """
    import h5py

    logger.info("Loading dataset from %s ...", dataset_path)
    with h5py.File(dataset_path, "r") as f:
        states_np = f["states"][:]   # (N, T, obs_dim)
        actions_np = f["actions"][:] # (N, T-1, act_dim)

    n_traj, t_len, obs_dim = states_np.shape
    _, t_minus_1, act_dim = actions_np.shape
    assert t_minus_1 == t_len - 1, f"Expected T-1={t_len-1} action steps, got {t_minus_1}"

    logger.info(
        "Dataset: N=%d trajectories, T=%d steps, obs_dim=%d, act_dim=%d",
        n_traj, t_len, obs_dim, act_dim,
    )

    states = torch.from_numpy(states_np).float()    # (N, T, obs_dim)
    actions = torch.from_numpy(actions_np).float()  # (N, T-1, act_dim)

    # Build (input, target) pairs for next-state prediction
    obs_t = states[:, :-1, :]   # (N, T-1, obs_dim) — current state
    obs_t1 = states[:, 1:, :]   # (N, T-1, obs_dim) — next state (target)

    # Concatenate along feature axis then permute to channels-first (FNO convention)
    x = torch.cat([obs_t, actions], dim=-1).permute(0, 2, 1)  # (N, obs_dim+act_dim, T-1)
    y = obs_t1.permute(0, 2, 1)                               # (N, obs_dim, T-1)

    logger.info("FNO tensors: x=%s, y=%s", tuple(x.shape), tuple(y.shape))
    return x, y, obs_dim, act_dim


# ──────────────────────────────────────────────────────────────────────────────
# FNO Training
# ──────────────────────────────────────────────────────────────────────────────

def make_splits(
    x: torch.Tensor,
    y: torch.Tensor,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """80/10/10 train/val/test split (deterministic)."""
    total = len(x)
    n_val = int(val_frac * total)
    n_test = int(test_frac * total)
    n_train = total - n_val - n_test
    ds = TensorDataset(x, y)
    generator = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val, n_test], generator=generator)


def _subset_tensors(dataset: TensorDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Return x/y tensors for a TensorDataset or Subset[TensorDataset]."""
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base = dataset.dataset
        idx = torch.as_tensor(dataset.indices, dtype=torch.long)
        x_all, y_all = base.tensors
        return x_all[idx], y_all[idx]
    return dataset.tensors


def compute_normalization_stats(train_ds: TensorDataset) -> dict[str, torch.Tensor]:
    """Compute channel-wise z-score stats on training split only."""
    x_train, y_train = _subset_tensors(train_ds)
    x_mean = x_train.mean(dim=(0, 2), keepdim=True)
    x_std = x_train.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
    y_mean = y_train.mean(dim=(0, 2), keepdim=True)
    y_std = y_train.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def apply_normalization(
    dataset: TensorDataset,
    stats: dict[str, torch.Tensor],
) -> TensorDataset:
    """Apply z-score normalization to x/y channels."""
    x_raw, y_raw = _subset_tensors(dataset)
    x_norm = (x_raw - stats["x_mean"]) / stats["x_std"]
    y_norm = (y_raw - stats["y_mean"]) / stats["y_std"]
    return TensorDataset(x_norm, y_norm)


def save_normalization_stats(path: str, stats: dict[str, torch.Tensor]) -> str:
    """Persist normalization tensors as numpy arrays."""
    out_path = str(Path(path).with_name("best_fno_norm.npz"))
    np.savez(
        out_path,
        x_mean=stats["x_mean"].cpu().numpy(),
        x_std=stats["x_std"].cpu().numpy(),
        y_mean=stats["y_mean"].cpu().numpy(),
        y_std=stats["y_std"].cpu().numpy(),
    )
    return out_path


def load_normalization_stats(path: str) -> dict[str, torch.Tensor] | None:
    """Load normalization stats if present next to model weights."""
    norm_path = Path(path).with_name("best_fno_norm.npz")
    if not norm_path.exists():
        return None
    data = np.load(norm_path)
    return {
        "x_mean": torch.from_numpy(data["x_mean"]).float(),
        "x_std": torch.from_numpy(data["x_std"]).float(),
        "y_mean": torch.from_numpy(data["y_mean"]).float(),
        "y_std": torch.from_numpy(data["y_std"]).float(),
    }


def val_loss(model: nn.Module, dataset: TensorDataset, device: str, batch_size: int = 512) -> float:
    """Compute MSE loss on a validation or test split."""
    model.eval()
    criterion = nn.MSELoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            total_loss += criterion(pred, yb).item() * len(xb)
    return total_loss / len(dataset)


def train_fno(
    model: nn.Module,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    train_cfg: dict,
    device: str,
    fno_checkpoint_out: str,
    epochs_override: int | None = None,
) -> float:
    """Train FNO1d with MSE loss + cosine LR + early stopping.

    Saves best weights to fno_checkpoint_out.
    Returns best validation loss.
    """
    epochs = epochs_override if epochs_override is not None else train_cfg["epochs"]
    lr = float(train_cfg["lr"])
    lr_min = float(train_cfg.get("lr_min", 1e-5))
    batch_size = int(train_cfg.get("batch_size", 256))
    patience_limit = int(train_cfg.get("early_stopping_patience", 20))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
    criterion = nn.MSELoss()

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    os.makedirs(os.path.dirname(os.path.abspath(fno_checkpoint_out)), exist_ok=True)

    best_val = float("inf")
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        scheduler.step()
        epoch_loss /= len(train_ds)

        v_loss = val_loss(model, val_ds, device, batch_size)

        if v_loss < best_val:
            best_val = v_loss
            patience = 0
            torch.save(model.state_dict(), fno_checkpoint_out)
        else:
            patience += 1

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d/%d  train_loss=%.6f  val_loss=%.6f  best=%.6f  patience=%d",
                epoch, epochs, epoch_loss, v_loss, best_val, patience,
            )

        if patience >= patience_limit:
            logger.info("Early stopping at epoch %d (patience=%d).", epoch, patience)
            break

    logger.info("FNO training complete. Best val_loss=%.6f", best_val)
    return best_val


# ──────────────────────────────────────────────────────────────────────────────
# Fidelity gate
# ──────────────────────────────────────────────────────────────────────────────

def run_fidelity_gate(
    model: nn.Module,
    test_ds: TensorDataset,
    gate_cfg: dict,
    variable_names: list[str],
    device: str,
    norm_stats: dict[str, torch.Tensor] | None = None,
) -> bool:
    """Evaluate fidelity gate on test split.

    FidelityGate.evaluate() expects (N, T, n_vars) numpy arrays.
    test_ds tensors are (N, C, T) → permute to (N, T, C) before passing.

    Returns True if gate passed.
    """
    from sco2rl.surrogate.fidelity_gate import FidelityGate

    model.eval()

    y_mean = y_std = None
    if norm_stats is not None:
        y_mean = norm_stats["y_mean"].to(device)
        y_std = norm_stats["y_std"].to(device)

    # Collect predictions and targets from test set
    preds_list, targets_list = [], []
    with torch.no_grad():
        for xb, yb in DataLoader(test_ds, batch_size=256, shuffle=False):
            xb = xb.to(device)
            pred = model(xb)  # (B, obs_dim, T)
            if y_mean is not None and y_std is not None:
                pred = pred * y_std + y_mean
                yb = yb.to(device) * y_std + y_mean
            else:
                yb = yb.to(device)
            # Permute to (B, T, obs_dim) for FidelityGate
            preds_list.append(pred.cpu().permute(0, 2, 1).numpy())
            targets_list.append(yb.cpu().permute(0, 2, 1).numpy())

    predictions = np.concatenate(preds_list, axis=0)  # (N, T, obs_dim)
    targets = np.concatenate(targets_list, axis=0)    # (N, T, obs_dim)

    if predictions.shape[-1] != len(variable_names):
        variable_names = [f"obs_{i}" for i in range(predictions.shape[-1])]

    # Populate variable ranges from env bounds when omitted.
    ranges_cfg = dict(gate_cfg.get("variable_ranges", {}))
    if not ranges_cfg:
        ranges_cfg = {
            name: 1.0
            for name in variable_names
        }

    gate_config = {
        "max_rmse_normalized": gate_cfg.get("max_rmse_normalized", 0.05),
        "min_r2": gate_cfg.get("min_r2", 0.97),
        "critical_variables": gate_cfg.get("critical_variables", []),
        "variable_ranges": ranges_cfg,
    }

    gate = FidelityGate(config=gate_config)
    report = gate.evaluate(predictions, targets, variable_names)

    logger.info(
        "Fidelity gate: passed=%s  overall_rmse=%.4f  overall_r2=%.4f",
        report.passed,
        report.overall_rmse_normalized,
        report.overall_r2,
    )
    if not report.passed:
        logger.warning("FIDELITY GATE FAILED: overall_rmse=%.4f > threshold=%.4f or r2=%.4f < threshold=%.4f",
                       report.overall_rmse_normalized, gate_cfg.get("max_rmse_normalized", 0.05),
                       report.overall_r2, gate_cfg.get("min_r2", 0.97))
    else:
        logger.info("FIDELITY GATE PASSED")

    return report.passed


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / "src"))

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device = "cpu"
    logger.info("Device: %s", device)

    # ── Load configs ──────────────────────────────────────────────────────────
    logger.info("Loading configs from %s/configs/", project_root)
    fno_cfg, env_cfg = load_configs(project_root)
    fno_arch_cfg = fno_cfg["fno"]
    train_cfg = fno_cfg["training"]
    gate_cfg = fno_cfg["fidelity_gate"]
    skrl_cfg = fno_cfg["skrl_ppo"]

    # Checkpoint output path for best FNO weights
    fno_ckpt_dir = project_root / train_cfg.get("checkpoint_dir", "artifacts/surrogate")
    fno_ckpt_out = str(fno_ckpt_dir / "best_fno.pt")

    # ── Load dataset (read actual dims, not YAML) ─────────────────────────────
    dataset_path = args.dataset
    if not Path(dataset_path).is_absolute():
        dataset_path = str(project_root / dataset_path)

    if not Path(dataset_path).exists():
        logger.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    x, y, obs_dim, act_dim = load_dataset(dataset_path, device)
    input_dim = obs_dim + act_dim  # actual dims from data (e.g. 74 = 70 + 4)
    output_dim = obs_dim           # predict next observation (same dim)

    logger.info("FNO dims: input_dim=%d, output_dim=%d (from data)", input_dim, output_dim)

    # 80/10/10 split
    train_ds_raw, val_ds_raw, test_ds_raw = make_splits(
        x, y,
        val_frac=float(train_cfg.get("validation_split", 0.10)),
        test_frac=float(train_cfg.get("test_split", 0.10)),
        seed=int(train_cfg.get("seed", 42)),
    )
    logger.info(
        "Splits: train=%d, val=%d, test=%d",
        len(train_ds_raw),
        len(val_ds_raw),
        len(test_ds_raw),
    )

    normalize_cfg = train_cfg.get("loss", {})
    use_normalization = bool(normalize_cfg.get("normalize_per_variable", True))
    norm_stats: dict[str, torch.Tensor] | None = None
    if use_normalization:
        norm_stats = compute_normalization_stats(train_ds_raw)
        train_ds = apply_normalization(train_ds_raw, norm_stats)
        val_ds = apply_normalization(val_ds_raw, norm_stats)
        test_ds = apply_normalization(test_ds_raw, norm_stats)
        del x, y, train_ds_raw, val_ds_raw, test_ds_raw
        logger.info("Enabled per-variable z-score normalization for FNO training.")
    else:
        train_ds, val_ds, test_ds = train_ds_raw, val_ds_raw, test_ds_raw

    # ── Build SCO2SurrogateFNO (PhysicsNeMo-backed) ────────────────────────
    from sco2rl.surrogate.fno_model import SCO2SurrogateFNO

    fno_config = {
        "input_dim":  input_dim,    # from actual data (obs_dim + act_dim)
        "output_dim": output_dim,   # from actual data (obs_dim)
        "modes":    int(fno_arch_cfg.get("modes",   16)),
        "width":    int(fno_arch_cfg.get("width",   64)),
        "n_layers": int(fno_arch_cfg.get("n_layers", 4)),
        "padding":  int(fno_arch_cfg.get("padding",  8)),
    }
    model = SCO2SurrogateFNO(config=fno_config).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("SCO2SurrogateFNO (PhysicsNeMo FNO) parameters: %s", f"{n_params:,}")

    # ── FNO training (or load pre-trained weights) ────────────────────────────
    if args.fno_checkpoint:
        fno_ckpt_in = args.fno_checkpoint
        if not Path(fno_ckpt_in).is_absolute():
            fno_ckpt_in = str(project_root / fno_ckpt_in)
        logger.info("Loading pre-trained FNO weights from %s", fno_ckpt_in)
        state = torch.load(fno_ckpt_in, map_location=device)
        model.load_state_dict(state)
        if use_normalization:
            norm_stats = load_normalization_stats(fno_ckpt_in)
            if norm_stats is None:
                logger.warning(
                    "Normalization stats file not found next to checkpoint. "
                    "Fidelity gate will use unnormalized outputs."
                )
    else:
        logger.info("Starting FNO supervised training ...")
        train_fno(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            train_cfg=train_cfg,
            device=device,
            fno_checkpoint_out=fno_ckpt_out,
            epochs_override=args.fno_epochs,
        )
        # Reload best weights
        logger.info("Reloading best FNO weights from %s", fno_ckpt_out)
        model.load_state_dict(torch.load(fno_ckpt_out, map_location=device))
        if use_normalization and norm_stats is not None:
            norm_path = save_normalization_stats(fno_ckpt_out, norm_stats)
            logger.info("Saved normalization stats to %s", norm_path)

    # ── Fidelity gate ─────────────────────────────────────────────────────────
    logger.info("Running fidelity gate on test split ...")
    obs_variables = [
        v for v in env_cfg["observation"]["variables"] if v.get("fmu_var") is not None
    ]
    variable_names = [v.get("name", v["fmu_var"]) for v in obs_variables]
    variable_ranges = {
        v.get("name", v["fmu_var"]): max(float(v["max"]) - float(v["min"]), 1e-6)
        for v in obs_variables
    }
    gate_cfg = dict(gate_cfg)
    if not gate_cfg.get("variable_ranges"):
        gate_cfg["variable_ranges"] = variable_ranges
    gate_passed = run_fidelity_gate(
        model=model,
        test_ds=test_ds,
        gate_cfg=gate_cfg,
        variable_names=variable_names,
        device=device,
        norm_stats=norm_stats if use_normalization else None,
    )

    if not gate_passed and not args.allow_fidelity_fail:
        logger.error(
            "Fidelity gate FAILED. "
            "Collect more data or tune FNO hyperparameters before launching SKRL PPO."
        )
        sys.exit(1)
    if not gate_passed and args.allow_fidelity_fail:
        logger.warning(
            "Fidelity gate failed but --allow-fidelity-fail is enabled; "
            "continuing to SKRL PPO for fail-fast orchestration validation."
        )

    if args.skip_rl:
        logger.info("--skip-rl: FNO training complete. Exiting before SKRL PPO.")
        return

    # ── SKRL PPO on GPU surrogate ─────────────────────────────────────────────
    logger.info("Launching SKRL PPO on GPU-vectorized SurrogateEnv ...")

    # Inject env config (obs/action vars) into skrl config
    env_config = build_env_config(env_cfg)
    if use_normalization and norm_stats is not None:
        env_config["normalization"] = {
            "obs_mean": norm_stats["x_mean"][0, :obs_dim, 0].cpu().numpy().tolist(),
            "obs_std": norm_stats["x_std"][0, :obs_dim, 0].cpu().numpy().tolist(),
            "act_mean": norm_stats["x_mean"][0, obs_dim:, 0].cpu().numpy().tolist(),
            "act_std": norm_stats["x_std"][0, obs_dim:, 0].cpu().numpy().tolist(),
            "next_obs_mean": norm_stats["y_mean"][0, :, 0].cpu().numpy().tolist(),
            "next_obs_std": norm_stats["y_std"][0, :, 0].cpu().numpy().tolist(),
        }
    skrl_cfg["env_config"] = env_config

    from sco2rl.surrogate.surrogate_trainer import SurrogateTrainer

    surrogate_trainer = SurrogateTrainer(
        surrogate_model=model,
        config=skrl_cfg,
        device=device,
    )
    surrogate_trainer.build_envs()
    surrogate_trainer.build_agent()

    if args.rl_timesteps is not None:
        skrl_cfg["total_timesteps"] = int(args.rl_timesteps)
    total_timesteps = int(skrl_cfg.get("total_timesteps", 5_000_000))
    logger.info("Training SKRL PPO: timesteps=%d ...", total_timesteps)
    stats = surrogate_trainer.train(timesteps=total_timesteps)

    logger.info(
        "SKRL PPO complete: mean_reward=%.3f, total_timesteps=%d",
        stats["mean_reward"],
        stats["total_timesteps"],
    )

    # Save checkpoint (RULE-C4)
    ckpt_dir = project_root / skrl_cfg.get("checkpoint_dir", "artifacts/checkpoints/surrogate_ppo")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(ckpt_dir / "final_checkpoint.json")
    surrogate_trainer.save_checkpoint(ckpt_path)
    logger.info("Checkpoint saved to %s", ckpt_path)

    print("\n[train_surrogate] Done.")
    print(f"  FNO weights:  {fno_ckpt_out}")
    print(f"  SKRL ckpt:    {ckpt_path}")
    print(f"  mean_reward:  {stats['mean_reward']:.3f}")


if __name__ == "__main__":
    main()
