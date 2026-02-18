"""Checkpoint bridge utilities (SKRL -> SB3)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from stable_baselines3 import PPO


def convert_skrl_to_sb3_checkpoint(
    skrl_checkpoint_path: str,
    output_dir: str,
    env: Any,
    model_stem: str = "skrl_bridge_model",
) -> str:
    """Convert a SKRL checkpoint manifest into an SB3-compatible RULE-C4 checkpoint.

    Parameters
    ----------
    skrl_checkpoint_path:
        Path to SKRL manifest JSON created by ``SurrogateTrainer.save_checkpoint()``.
    output_dir:
        Directory where converted model/checkpoint files are written.
    env:
        VecEnv passed to SB3 PPO constructor.
    model_stem:
        Base filename for converted SB3 model.

    Returns
    -------
    str
        Path to converted RULE-C4 JSON checkpoint metadata.
    """
    skrl_ckpt = Path(skrl_checkpoint_path)
    meta = json.loads(skrl_ckpt.read_text())
    weights_ref = meta.get("model_weights")
    if not weights_ref:
        raise ValueError(
            f"SKRL checkpoint {skrl_checkpoint_path!r} missing 'model_weights'."
        )
    weights_path = _resolve_relative(skrl_ckpt.parent, str(weights_ref))
    if not weights_path.exists():
        raise FileNotFoundError(f"SKRL weights not found: {weights_path}")

    state = torch.load(str(weights_path), map_location="cpu")
    policy_state = state.get("policy")
    value_state = state.get("value")
    if not isinstance(policy_state, dict) or not isinstance(value_state, dict):
        raise ValueError("SKRL weights file must contain 'policy' and 'value' state dicts.")

    pi_hidden = _extract_hidden_layers(policy_state, prefix="net")
    vf_hidden = _extract_hidden_layers(value_state, prefix="net")
    policy_kwargs = {"net_arch": {"pi": pi_hidden, "vf": vf_hidden}}

    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        n_steps=16,
        batch_size=8,
        n_epochs=1,
        verbose=0,
    )
    _load_skrl_weights_into_sb3(model, policy_state=policy_state, value_state=value_state)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / model_stem
    model.save(str(model_path))

    total_timesteps = int(meta.get("total_timesteps", 0))
    converted_meta = {
        "model_path": str(model_path),
        "vecnorm_stats": None,
        "curriculum_phase": int(meta.get("curriculum_phase", 0)),
        "lagrange_multipliers": dict(meta.get("lagrange_multipliers", {})),
        "total_timesteps": total_timesteps,
        "step": total_timesteps,
        "source_skrl_checkpoint": str(skrl_ckpt),
    }
    converted_ckpt = out_dir / f"{model_stem}_checkpoint.json"
    converted_ckpt.write_text(json.dumps(converted_meta, indent=2))
    return str(converted_ckpt)


def _resolve_relative(base_dir: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base_dir / p


def _extract_hidden_layers(state_dict: dict[str, torch.Tensor], prefix: str) -> list[int]:
    """Infer hidden layer widths from SKRL sequential net weights."""
    hidden: list[int] = []
    layer_idx = 0
    while True:
        key = f"{prefix}.{layer_idx}.weight"
        if key not in state_dict:
            break
        out_features = int(state_dict[key].shape[0])
        next_key = f"{prefix}.{layer_idx + 2}.weight"
        if next_key in state_dict:
            hidden.append(out_features)
        layer_idx += 2
    if not hidden:
        raise ValueError("Could not infer hidden architecture from SKRL state dict.")
    return hidden


def _copy_weight(
    src: dict[str, torch.Tensor],
    dst: dict[str, torch.Tensor],
    src_key: str,
    dst_key: str,
) -> None:
    tensor = src[src_key].detach().cpu()
    if dst_key not in dst:
        raise KeyError(f"SB3 key {dst_key!r} not found during conversion.")
    if tuple(dst[dst_key].shape) != tuple(tensor.shape):
        raise ValueError(
            f"Shape mismatch for {dst_key}: expected {tuple(dst[dst_key].shape)}, "
            f"got {tuple(tensor.shape)} from {src_key}"
        )
    dst[dst_key] = tensor


def _load_skrl_weights_into_sb3(
    model: PPO,
    policy_state: dict[str, torch.Tensor],
    value_state: dict[str, torch.Tensor],
) -> None:
    """Map SKRL actor/critic MLP weights into SB3 MlpPolicy tensors."""
    sb3_state = model.policy.state_dict()

    # Actor hidden layers + action head.
    _copy_weight(policy_state, sb3_state, "net.0.weight", "mlp_extractor.policy_net.0.weight")
    _copy_weight(policy_state, sb3_state, "net.0.bias", "mlp_extractor.policy_net.0.bias")
    _copy_weight(policy_state, sb3_state, "net.2.weight", "mlp_extractor.policy_net.2.weight")
    _copy_weight(policy_state, sb3_state, "net.2.bias", "mlp_extractor.policy_net.2.bias")
    _copy_weight(policy_state, sb3_state, "net.4.weight", "mlp_extractor.policy_net.4.weight")
    _copy_weight(policy_state, sb3_state, "net.4.bias", "mlp_extractor.policy_net.4.bias")
    _copy_weight(policy_state, sb3_state, "net.6.weight", "action_net.weight")
    _copy_weight(policy_state, sb3_state, "net.6.bias", "action_net.bias")
    if "log_std" in policy_state and "log_std" in sb3_state:
        _copy_weight(policy_state, sb3_state, "log_std", "log_std")

    # Critic hidden layers + value head.
    _copy_weight(value_state, sb3_state, "net.0.weight", "mlp_extractor.value_net.0.weight")
    _copy_weight(value_state, sb3_state, "net.0.bias", "mlp_extractor.value_net.0.bias")
    _copy_weight(value_state, sb3_state, "net.2.weight", "mlp_extractor.value_net.2.weight")
    _copy_weight(value_state, sb3_state, "net.2.bias", "mlp_extractor.value_net.2.bias")
    _copy_weight(value_state, sb3_state, "net.4.weight", "mlp_extractor.value_net.4.weight")
    _copy_weight(value_state, sb3_state, "net.4.bias", "mlp_extractor.value_net.4.bias")
    _copy_weight(value_state, sb3_state, "net.6.weight", "value_net.weight")
    _copy_weight(value_state, sb3_state, "net.6.bias", "value_net.bias")

    model.policy.load_state_dict(sb3_state, strict=True)
