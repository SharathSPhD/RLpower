"""TensorRT FP16 engine builder for the exported ONNX actor policy."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TRTExportResult:
    trt_engine_path: str
    precision: str
    onnx_path: str
    obs_dim: int
    act_dim: int
    max_abs_error_vs_onnx: float
    passed_tolerance: bool


class TensorRTExporter:
    """Builds a serialized TensorRT engine from an ONNX file.

    Config keys:
        precision (str): 'fp16' or 'fp32'.
        workspace_gb (int): Builder workspace size in GB.
        min_batch / opt_batch / max_batch (int): Optimization profile batch sizes.
        builder_optimization_level (int): TRT builder optimization level (0–5).
        verify (dict): post-build verification settings.
            tolerance_abs (float), n_test_samples (int).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = config

    def build_engine(
        self,
        onnx_path: str,
        trt_path: str,
        obs_dim: int,
        act_dim: int,
    ) -> TRTExportResult:
        """Build and serialize a TensorRT engine.

        Args:
            onnx_path: Path to the source ONNX file.
            trt_path: Destination path for the serialized TRT engine.
            obs_dim: Observation dimensionality (used for optimization profile).
            act_dim: Action dimensionality (output size).

        Returns:
            TRTExportResult with metadata.
        """
        import tensorrt as trt  # type: ignore[import]

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            onnx_bytes = f.read()
        if not parser.parse(onnx_bytes):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parsing failed: {errors}")

        cfg = builder.create_builder_config()
        workspace_bytes = self._cfg.get("workspace_gb", 1) << 30
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

        precision = self._cfg.get("precision", "fp16")
        if precision == "fp16" and builder.platform_has_fast_fp16:
            cfg.set_flag(trt.BuilderFlag.FP16)

        opt_level = self._cfg.get("builder_optimization_level", 3)
        cfg.builder_optimization_level = opt_level

        profile = builder.create_optimization_profile()
        min_b = self._cfg.get("min_batch", 1)
        opt_b = self._cfg.get("opt_batch", 1)
        max_b = self._cfg.get("max_batch", 4)
        input_name = network.get_input(0).name
        profile.set_shape(input_name, (min_b, obs_dim), (opt_b, obs_dim), (max_b, obs_dim))
        cfg.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, cfg)
        if serialized_engine is None:
            raise RuntimeError("TensorRT engine build failed — build_serialized_network returned None")

        Path(trt_path).parent.mkdir(parents=True, exist_ok=True)
        with open(trt_path, "wb") as f:
            f.write(serialized_engine)

        max_err, passed = self._verify_engine(
            trt_path, onnx_path, obs_dim, act_dim, logger
        )

        return TRTExportResult(
            trt_engine_path=trt_path,
            precision=precision,
            onnx_path=onnx_path,
            obs_dim=obs_dim,
            act_dim=act_dim,
            max_abs_error_vs_onnx=max_err,
            passed_tolerance=passed,
        )

    def _verify_engine(
        self,
        trt_path: str,
        onnx_path: str,
        obs_dim: int,
        act_dim: int,
        logger: Any,
    ) -> tuple[float, bool]:
        """Compare TRT engine output against ONNX runtime output."""
        import tensorrt as trt  # type: ignore[import]
        import onnxruntime as ort

        verify_cfg = self._cfg.get("verify", {})
        tol = verify_cfg.get("tolerance_abs", 0.5)
        n = verify_cfg.get("n_test_samples", 5)

        with open(trt_path, "rb") as f:
            engine_bytes = f.read()

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        context = engine.create_execution_context()

        import pycuda.driver as cuda  # type: ignore[import]
        import pycuda.autoinit  # type: ignore[import]  # noqa: F401

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        rng = np.random.default_rng(42)
        samples = rng.standard_normal((n, obs_dim)).astype(np.float32)

        max_err = 0.0
        for obs in samples:
            inp = obs[None]  # (1, obs_dim)
            ort_out = sess.run(None, {"input": inp})[0]

            inp_t = cuda.mem_alloc(inp.nbytes)
            out_arr = np.empty((1, act_dim), dtype=np.float32)
            out_t = cuda.mem_alloc(out_arr.nbytes)
            cuda.memcpy_htod(inp_t, inp)
            context.execute_v2([int(inp_t), int(out_t)])
            cuda.memcpy_dtoh(out_arr, out_t)

            err = float(np.max(np.abs(out_arr - ort_out)))
            if err > max_err:
                max_err = err

        passed = max_err <= tol
        return max_err, passed
