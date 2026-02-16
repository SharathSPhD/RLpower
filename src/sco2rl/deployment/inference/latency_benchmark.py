"""Latency benchmarking for TensorRT inference engine."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class LatencyReport:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    n_iterations: int
    batch_size: int
    passed_sla: bool


class LatencyBenchmark:
    """Measures inference latency of a TensorRT engine.

    Config keys:
        n_warmup (int): Warmup iterations before measurement.
        n_iterations (int): Measurement iterations.
        batch_size (int): Batch size per inference call.
        max_latency_ms (float): SLA threshold for p99 (ms).
        output_file (str): Optional path to write JSON report.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = config

    def benchmark(self, trt_path: str, obs_dim: int) -> LatencyReport:
        """Run latency benchmark on a serialized TRT engine.

        Args:
            trt_path: Path to the .trt serialized engine file.
            obs_dim: Observation dimensionality.

        Returns:
            LatencyReport with percentile latencies and SLA result.
        """
        import tensorrt as trt  # type: ignore[import]
        import pycuda.driver as cuda  # type: ignore[import]
        import pycuda.autoinit  # type: ignore[import]  # noqa: F401
        import numpy as np

        logger = trt.Logger(trt.Logger.WARNING)
        with open(trt_path, "rb") as f:
            engine_bytes = f.read()
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        context = engine.create_execution_context()

        batch = self._cfg.get("batch_size", 1)

        # TRT 10.x API: num_io_tensors + get_tensor_name/mode
        n_tensors = engine.num_io_tensors
        input_name = output_name = None
        act_dim = 4  # fallback
        for i in range(n_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_name = name
            else:
                output_name = name
                shape = engine.get_tensor_shape(name)
                act_dim = int(shape[-1])

        inp = np.zeros((batch, obs_dim), dtype=np.float32)
        out = np.empty((batch, act_dim), dtype=np.float32)
        inp_buf = cuda.mem_alloc(inp.nbytes)
        out_buf = cuda.mem_alloc(out.nbytes)
        cuda.memcpy_htod(inp_buf, inp)

        # Set dynamic input shape before executing
        context.set_input_shape(input_name, (batch, obs_dim))
        context.set_tensor_address(input_name, int(inp_buf))
        context.set_tensor_address(output_name, int(out_buf))

        n_warmup = self._cfg.get("n_warmup", 10)
        n_iter = self._cfg.get("n_iterations", 100)

        stream = cuda.Stream()
        for _ in range(n_warmup):
            context.execute_async_v3(stream.handle)
        stream.synchronize()

        latencies_ms: list[float] = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            context.execute_async_v3(stream.handle)
            stream.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        arr = np.array(latencies_ms)
        sla = self._cfg.get("max_latency_ms", 1.0)
        report = LatencyReport(
            mean_ms=float(arr.mean()),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            min_ms=float(arr.min()),
            max_ms=float(arr.max()),
            n_iterations=n_iter,
            batch_size=batch,
            passed_sla=bool(np.percentile(arr, 99) <= sla),
        )

        out_file = self._cfg.get("output_file")
        if out_file:
            self.save_report(report, out_file)

        return report

    def save_report(self, report: LatencyReport, path: str) -> None:
        """Persist a LatencyReport to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(report), f, indent=2)

    @staticmethod
    def load_report(path: str) -> LatencyReport:
        """Load a LatencyReport from JSON."""
        with open(path) as f:
            data = json.load(f)
        return LatencyReport(**data)
