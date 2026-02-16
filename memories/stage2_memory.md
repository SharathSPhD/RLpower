# Stage 2 Memory — Curriculum + FMU Integration

Status: COMPLETE — merged to main (2026-02-16)

## Gate Criteria Results

| Criterion | Result |
|-----------|--------|
| FMPyAdapter drives real FMU without crash | PASS |
| All 4 action mappings produce physical responses | PASS |
| Unit conversions (K→°C, W→MW, Pa→MPa) verified | PASS |
| 100-step env loop with real FMU completes | PASS |
| All 410 unit tests still pass | PASS |
| Integration smoke test 13/13 passed | PASS |

## Architecture Decisions

### ADR-S2-1: setReal() on SCOPE Parameter Variables (2026-02-16)
**Decision**: Actions map to SCOPE component *parameter* variables via `fmpy setReal()` between CVODE co-simulation steps.

**Context**: The compiled SCO2RecuperatedCycle.fmu has NO `input` causality variables in its model description. All 86 variables are `local` or `parameter` causality. Standard FMI 2.0 input/output ports are absent.

**Verified mappings (via experiment 2026-02-16)**:
| Action | FMU Variable | Units | Effect Verified |
|--------|-------------|-------|-----------------|
| bypass_valve | `regulator.T_init` | K (800–1200) | +100K → dW_turbine = +1.454 MW |
| igv | `regulator.m_flow_init` | kg/s (60–130) | +25 kg/s → dW_turbine = +3.532 MW |
| inventory_valve | `turbine.p_out` | Pa (7–9 MPa) | +1 MPa → dW_turbine = -1.784 MW |
| cooling_flow | `precooler.T_output` | K (305.65–315) | +4.35K → dT_comp_inlet = +4.35°C |

**Rejected**: `main_compressor.eta` as IGV proxy — SCOPE Pump model with fixed outlet pressure renders eta changes ineffective (dW_comp = 0.000 MW).

**Consequence**: No FMU recompilation required. setReal() on parameter variables persists across CVODE steps.

### ADR-S2-2: Unit Conversion via FMPyAdapter.scale_offset (2026-02-16)
**Decision**: All unit conversions (K→°C, W→MW, Pa→MPa) happen inside FMPyAdapter using a `scale_offset` dict applied as `result = raw * scale + offset`.

**Context**: FMU returns SI units (K, W, Pa); environment/safety configs use engineering units (°C, MW, MPa).

**Implementation**: `_SIMPLE_RECUPERATED_SCALE_OFFSET` module-level constant + `FMPyAdapter.default_scale_offset()` static method. MockFMU unaffected.

### ADR-S2-3: env.yaml Rewritten for simple_recuperated (2026-02-16)
**Decision**: env.yaml was rewritten from recompression cycle (20 obs vars, 5 actions, wrong variable names) to simple_recuperated (16 obs vars, 4 actions, verified FMU variable names).

**Changes**:
- obs_dim: 100 → 80 (16 vars × 5 history steps)
- action_dim: 5 → 4 (removed split_ratio which has no SCOPE analog)
- All `fmu_var` fields updated to actual FMU model description names
- design_efficiency: 0.47 → 0.40 (simple recuperated vs recompression)

### ADR-S2-4: Integration Test Gate Mechanism
**Decision**: Integration tests in `tests/integration/` are gated by BOTH:
1. `--run-integration` CLI flag (checked in conftest.py `pytest_collection_modifyitems`)
2. `RUN_INTEGRATION=1` env var (checked in test file `pytestmark`)

pytest auto-derives `"integration"` keyword from directory path — conftest catches all tests in `tests/integration/`.

**Correct invocation**:
```bash
docker exec -e RUN_INTEGRATION=1 sco2rl-dev bash -c \
  "cd /workspace && PYTHONPATH=src pytest tests/integration/ -v --run-integration \
   --no-cov --override-ini='addopts=' -p no:cacheprovider"
```

## FMU Variable Discovery

All 86 model variables are `local` or `parameter` causality:
- Temperature vars all returned in Kelvin (verified: T_comp_inlet = 305.7K = 32.55°C)
- Power vars in Watts (W_turbine ≈ 13.42e6 W = 13.42 MW at design point)
- Pressure vars in Pa (p_outlet ≈ 18.0e6 Pa = 18.0 MPa at design point)

Design point confirmed:
- W_net ≈ 10.78 MW (W_turbine 13.42 - W_comp 2.64)
- T_comp_inlet = 32.5°C (305.65K) — 1.4°C above critical point ✓ (RULE-P1)
- T_turbine_inlet = 700°C (973.15K)
- η_recuperator ≈ 0.86 (86% effectiveness)
- Q_recuperator ≈ 50.6 MW

## Files Modified in Stage 2

| File | Change |
|------|--------|
| `configs/environment/env.yaml` | Rewritten for simple_recuperated cycle |
| `src/sco2rl/simulation/fmu/fmpy_adapter.py` | Added scale_offset unit conversion |
| `tests/unit/test_config_loader.py` | obs_dim 100→80, action_dim 5→4, var_count 20→16 |
| `tests/integration/test_fmpy_adapter_real_fmu.py` | Created: 13 integration smoke tests |
| `memories/stage2_memory.md` | This file |

## Open Questions for Stage 3

- LHS sampler range: use action physical ranges from env.yaml directly?
- FNO input: include history (ring buffer) or just current state?
- Fidelity gate threshold: RMSE < 5% of full operating range or absolute?
