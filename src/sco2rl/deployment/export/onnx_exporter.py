"""ONNX export for SB3 PPO actor network."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import warnings

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ONNXExportResult:
    onnx_path: str
    opset_version: int
    obs_dim: int
    act_dim: int
    max_abs_error: float
    passed_tolerance: bool


class _ActorWrapper(nn.Module):
    """Thin wrapper that exposes only the deterministic mean-action path."""

    def __init__(self, policy: Any) -> None:
        super().__init__()
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent_pi, _ = self.mlp_extractor(obs)
        return self.action_net(latent_pi)


class ONNXExporter:
    """Exports a trained SB3 PPO model to ONNX format.

    Config keys:
        opset_version (int): ONNX opset (default 17).
        do_constant_folding (bool): Graph optimization flag.
        dynamic_axes (dict): Batch-dimension axes spec.
        verify (dict): post-export numerical verification settings.
            enabled (bool), tolerance_abs (float), n_test_samples (int), seed (int).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = config
        self._verify_cfg = config.get("verify", {})
        self._last_model: Any = None
        self._last_onnx_path: str = ""

    def export(self, model: Any, obs_dim: int, output_path: str) -> ONNXExportResult:
        """Export PPO actor to ONNX.

        Args:
            model: Trained SB3 PPO model.
            obs_dim: Observation space dimensionality.
            output_path: Destination .onnx file path.

        Returns:
            ONNXExportResult with metadata.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        policy = model.policy
        policy.eval()
        # Move to CPU for ONNX export — TRT will handle GPU inference
        wrapper = _ActorWrapper(policy).cpu()
        wrapper.eval()

        dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
        with torch.no_grad():
            sample_out = wrapper(dummy)
        act_dim = sample_out.shape[-1]

        opset = self._cfg.get("opset_version", 17)
        dynamic_axes = self._cfg.get(
            "dynamic_axes",
            {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                wrapper,
                dummy,
                output_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=opset,
                do_constant_folding=self._cfg.get("do_constant_folding", True),
                dynamic_axes=dynamic_axes,
                dynamo=False,
            )

        self._last_model = model
        self._last_onnx_path = output_path

        max_err = 0.0
        passed = True
        if self._verify_cfg.get("enabled", True):
            tol = self._verify_cfg.get("tolerance_abs", 1e-4)
            n = self._verify_cfg.get("n_test_samples", 5)
            seed = self._verify_cfg.get("seed", 42)
            max_err = self.verify(model, output_path, obs_dim, n_samples=n, seed=seed, tolerance=tol)
            passed = bool(max_err <= tol)

        return ONNXExportResult(
            onnx_path=output_path,
            opset_version=opset,
            obs_dim=obs_dim,
            act_dim=act_dim,
            max_abs_error=max_err,
            passed_tolerance=passed,
        )

    def verify(
        self,
        model: Any,
        onnx_path: str,
        obs_dim: int,
        *,
        n_samples: int = 10,
        seed: int = 42,
        tolerance: float = 1e-3,
    ) -> float:
        """Compare ONNX runtime outputs against PyTorch policy outputs.

        Returns:
            Maximum absolute error across all samples.
        """
        import onnxruntime as ort  # local import — optional dep

        rng = np.random.default_rng(seed)
        observations = rng.standard_normal((n_samples, obs_dim)).astype(np.float32)

        policy = model.policy
        policy.eval()
        wrapper = _ActorWrapper(policy).cpu()
        wrapper.eval()

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        max_err = 0.0
        for obs in observations:
            obs_t = torch.tensor(obs[None], dtype=torch.float32)
            with torch.no_grad():
                pt_out = wrapper(obs_t).numpy()
            ort_out = sess.run(None, {"input": obs[None]})[0]
            err = float(np.max(np.abs(pt_out - ort_out)))
            if err > max_err:
                max_err = err
        return max_err
