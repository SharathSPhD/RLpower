# Stage 5 Memory — TRT FP16 Deployment + Evaluation Report

**Status**: COMPLETE (infrastructure) — merged to main @ 5010a55
**Stage goal**: TRT FP16 < 1ms p99; full RL vs PID evaluation report.

## Infrastructure Built (25 tests, all passing)

### Export Pipeline (11 tests)
- `ONNXExporter` — `_ActorWrapper(nn.Module)` wraps SB3 PPO actor (CPU export);
  `torch.onnx.export(dynamo=False)` with `warnings.simplefilter("ignore")`
- `TensorRTExporter` — TRT 10.x API: `build_serialized_network()`, FP16 flag,
  optimization profile for dynamic batch; `_verify_engine()` uses pycuda
- `LatencyBenchmark` — TRT 10.x: `set_input_shape()` + `execute_async_v3(stream.handle)`;
  p50/p95/p99 with SLA gate

### Evaluation Pipeline (14 tests)
- `PIDController` — single-channel PI with integral reset and output clipping
- `PIDBaseline` — multi-channel PI policy; `predict()` returns `(action, None)`
- `EvaluationReporter(env, config, evaluator_factory=None)` — injected factory for testability;
  Gate 5: passed_sla AND zero violations
- `EvaluationReport` — per-phase RL vs PID comparison; JSON save/load

## Key Design Decisions

1. **CPU ONNX export**: Move wrapper to `.cpu()` before `torch.onnx.export()`.
2. **TRT 10.x API** (breaking changes from TRT 8):
   - `engine.num_bindings` → `engine.num_io_tensors`
   - `binding_is_input(i)` → `get_tensor_mode(name) == trt.TensorIOMode.INPUT`
   - `execute_v2(bindings)` → `set_input_shape() + execute_async_v3(stream.handle)`
3. **Warning suppression**: `torch.onnx.export(dynamo=False)` triggers `DeprecationWarning`;
   pyproject.toml filter `ignore::DeprecationWarning:torch.onnx` does NOT work (pytest intercepts first);
   use `warnings.simplefilter("ignore")` inside `ONNXExporter.export()`.
4. **pycuda**: Not in base Docker image — install with `pip install pycuda`.
5. **Merge conflict**: Both branches wrote `latency_benchmark.py`; resolved by keeping full implementation.

## Bugs Encountered

1. **Device mismatch**: SB3 PPO on GPU, ONNX export needs CPU. Fix: `_ActorWrapper(policy).cpu()`.
2. **TRT 10.x API**: `num_bindings` AttributeError. Rewrote with `num_io_tensors + get_tensor_name/mode`.
3. **Dynamic shape**: `executeV2` fails "Not all shapes specified". Fix: `set_input_shape()` + `execute_async_v3`.
4. **conftest collision**: Run pytest from `-w /workspace/.worktrees/<name>` to avoid module name clash.

## Gate 5 Verification Results (Infrastructure)

- `pytest tests/unit/ -q` (409 tests): ✅ PASS @ 5010a55
- ONNXExporter: ✅ 5 tests
- TensorRTExporter: ✅ 3 tests
- LatencyBenchmark: ✅ 3 tests
- PIDBaseline: ✅ 6 tests
- EvaluationReporter: ✅ 8 tests

## Gate 5 Full Results (pending real deployment)

- TRT p99 < 1ms on DGX Spark Blackwell: [ ] YES / [ ] NO (value: ___)
- ONNX max_abs_error vs PyTorch < 1e-4: [ ] YES / [ ] NO (value: ___)
- PID baseline comparison per phase documented: [ ] YES / [ ] NO
- Phase 6 (emergency trip) demonstrated with TRT engine: [ ] YES / [ ] NO
