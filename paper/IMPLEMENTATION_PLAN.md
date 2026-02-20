# RLpower — Control Systems Enhancement: Design & Implementation Document

**Version**: 3.0
**Date**: 2026-02-20
**Branch**: `claude/control-systems-cloud-enhancements-bUVTD`
**Scope**: Adds classical control metrics, reusable controller library, interactive notebook, and control-theoretic paper section to the RLpower sCO₂ Brayton cycle RL project.

---

## Table of Contents

1. [Project Gap Analysis](#1-project-gap-analysis)
2. [Architecture Overview](#2-architecture-overview)
3. [Control Library API](#3-control-library-api-sco2rlcontrol)
4. [Analysis Pipeline](#4-analysis-pipeline-sco2rlanalysis)
5. [PID Tuning Details](#5-pid-tuning-details)
6. [PRBS Frequency Estimation](#6-prbs-based-frequency-response-estimation)
7. [Bug Log](#7-bug-log)
8. [File Manifest](#8-file-manifest)
9. [Acceptance Criteria](#9-acceptance-criteria)
10. [Usage Examples](#10-usage-examples)

---

## 1. Project Gap Analysis

Prior to this enhancement, the RLpower project reported results **only as cumulative episode reward** — a proxy metric with no classical control-theoretic interpretation. Six gaps were identified:

### GAP-1 — No control system response characterisation
Zero classical control metrics existed in any analysis code or notebook:
- No step response (overshoot %, settling time, rise time, steady-state error)
- No frequency-domain analysis (Bode plots, gain margin, phase margin, bandwidth)
- No error integrals (IAE, ISE, ITAE)
- No disturbance rejection characterisation
- No cross-channel coupling analysis

### GAP-2 — PID baseline was inadequate
The original `PIDBaseline` used uniform flat gains (`Kp = 0.02`, `Ki = 0.001`) for all 4 channels with no derivative action and no anti-windup. This is an under-tuned heuristic — comparing RL against it is misleading for a control-systems audience.

### GAP-3 — Notebooks were static dashboards
Notebooks 03 and 04 visualise reward curves and thermodynamic state distributions but provide no interactive scenario selection, no time-series trajectory plots, and no control-theoretic visualisations (Bode, step response).

### GAP-4 — Paper lacked control-theoretic substance
The paper was strong as an MLOps contribution but thin for a control systems audience: no settling times, no overshoot figures, no stability margin tables, and no discussion of what the RL policy learns compared to classical control.

### GAP-5 — Controller blocks were not reusable
No public `Controller` interface existed. The PID was tightly coupled to observation indexing with no `python-control` or `scipy.signal` integration.

### GAP-6 — No W_net trajectory plots across curriculum
The paper showed reward curves only. Actual controlled-variable trajectories (W_net(t), T_comp(t)) — the fundamental evidence of control performance — were never plotted.

---

## 2. Architecture Overview

```
src/sco2rl/
├── control/                    ← NEW: Reusable controller library
│   ├── __init__.py             ← Public API
│   ├── interfaces.py           ← Abstract Controller base class
│   ├── pid.py                  ← Full PID with anti-windup + derivative filter
│   ├── multi_loop_pid.py       ← 4-channel IMC-tuned MultiLoopPID
│   └── rl_controller.py        ← RLController wrapper (SB3-compatible)
│
├── analysis/                   ← NEW: Control metrics computation
│   ├── __init__.py             ← Public API
│   ├── metrics.py              ← StepResponseResult, FrequencyResponseResult, ControlMetricsSummary
│   ├── step_response.py        ← run_step_scenario(), compute_step_metrics()
│   ├── frequency_analysis.py   ← PRBS generation, ETFE, margin computation
│   ├── scenario_runner.py      ← ScenarioRunner orchestrator + build_mock_env/pid
│   └── _dynamic_mock.py        ← DynamicMockFMU (first-order lag dynamics)

scripts/
├── run_control_analysis.py     ← CLI: runs PID + RL through all scenarios
└── tune_pid.py                 ← CLI: open-loop step tests → IMC gains → YAML

configs/control/
└── pid_gains.yaml              ← IMC-tuned gains for all 4 channels

notebooks/
└── 05_control_analysis.ipynb   ← Interactive: scenario selector + control plots

paper/
├── sec_control_analysis.tex    ← NEW section: Control-Theoretic Performance Analysis
├── sec_results.tex             ← UPDATED: added W_net tracking subsection
├── sec_conclusion.tex          ← UPDATED: added control-theoretic findings paragraph
├── main.tex                    ← UPDATED: \input{sec_control_analysis} inserted
└── figures/                    ← PNG figures (step response, Bode, heatmap, W_net)

data/
├── control_analysis_all_phases.json   ← Pre-computed PID results, all 7 phases
└── control_analysis_phase0.json       ← Phase 0 detail
```

---

## 3. Control Library API (`sco2rl.control`)

All controllers implement the abstract `Controller` interface, which is SB3-compatible (same `predict()` signature as SB3 policies).

### 3.1 `Controller` (abstract base)

```python
from sco2rl.control import Controller   # abstract

class Controller(ABC):
    @abstractmethod
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        """Return (action, state) — same interface as SB3 policies."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (integral accumulator, history, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable controller name (used in result metadata)."""
```

### 3.2 `PIDController`

Single-channel PID with:
- **Back-calculation anti-windup**: prevents integral from accumulating past output saturation
- **Derivative-on-measurement**: avoids derivative kick on setpoint changes
- **First-order derivative filter**: configurable time constant `derivative_filter_tau`

```python
from sco2rl.control import PIDController

pid = PIDController(
    kp=0.25, ki=0.010, kd=0.50,
    setpoint=10.0,                 # W_net setpoint in MW
    output_limits=(-1.0, 1.0),    # normalised action bounds
    anti_windup_gain=0.1,
    derivative_filter_tau=10.0,   # seconds (> 0 enables filter)
    dt=5.0,                       # simulation step size in seconds
)
action = pid.compute(measurement=9.8)
```

**IMC Tuning formula** (see §5 for details):
```
Kp = τ / (|K| · λ)
Ki = Kp / τ
Kd = Kp · τ_d    (optional)
```

### 3.3 `MultiLoopPID`

4-channel decentralised PID for the sCO₂ cycle. Implements `Controller.predict()` so it is a drop-in for any SB3-style evaluation loop.

| Channel | Actuator | Controlled variable | Kp | Ki | Kd |
|---------|----------|--------------------|----|----|----|
| 0 | `bypass_valve_opening` | `W_net` | 0.25 | 0.010 | 0.50 |
| 1 | `igv_angle_normalized` | `T_turbine_inlet` | 0.010 | 0.0002 | 0.05 |
| 2 | `inventory_valve_opening` | `P_high` | 0.30 | 0.010 | 0.50 |
| 3 | `cooling_flow_normalized` | `T_compressor_inlet` | 0.20 | 0.008 | 0.40 |

```python
from sco2rl.control import MultiLoopPID

pid = MultiLoopPID()   # IMC-tuned defaults
action, _ = pid.predict(obs, deterministic=True)
```

### 3.4 `RLController`

Wraps any SB3-compatible policy (LagrangianPPO, PPO, etc.) as a `Controller`:

```python
from sco2rl.control import RLController

# Load from checkpoint
rl = RLController.from_checkpoint("artifacts/checkpoints/run01/final")

# Or wrap an existing policy object
rl = RLController(policy=my_policy, controller_name="RL-5M")

action, _ = rl.predict(obs, deterministic=True)
```

`from_checkpoint()` tries `LagrangianPPO.load()` first, falls back to `PPO.load()` gracefully.

---

## 4. Analysis Pipeline (`sco2rl.analysis`)

### 4.1 Core Dataclasses (`metrics.py`)

```python
@dataclass
class StepResponseResult:
    variable: str        # "W_net", "T_turbine_inlet", etc.
    controller: str      # "MultiLoopPID", "RL", ...
    phase: int           # 0–6 curriculum phase
    scenario: str        # "step_load_up", "step_load_down", ...
    seed: int
    time_s: list[float]       # simulation time axis
    setpoint: list[float]     # setpoint trajectory
    response: list[float]     # actual output trajectory
    step_onset_s: float       # time at which step was applied
    # Computed metrics
    overshoot_pct: float      # % above final value
    undershoot_pct: float     # % below initial value (for down-steps)
    settling_time_s: float    # time to stay within ±2% of final
    rise_time_s: float        # 10%→90% of step magnitude
    peak_time_s: float        # time to first peak
    steady_state_error: float # asymptotic error
    iae: float                # ∫|e(t)|dt
    ise: float                # ∫e²(t)dt
    itae: float               # ∫t·|e(t)|dt

@dataclass
class FrequencyResponseResult:
    output_variable: str
    controller: str
    phase: int
    channel_idx: int
    frequencies_hz: list[float]
    magnitude_db: list[float]
    phase_deg: list[float]
    gain_margin_db: float       # dB (inf if no phase crossover)
    phase_margin_deg: float     # degrees
    bandwidth_hz: float         # −3 dB frequency
    gain_crossover_hz: float
    phase_crossover_hz: float

@dataclass
class ControlMetricsSummary:
    phase: int
    scenario: str
    pid_step: StepResponseResult | None
    rl_step: StepResponseResult | None
    pid_freq: FrequencyResponseResult | None
    rl_freq: FrequencyResponseResult | None
```

### 4.2 Step Response (`step_response.py`)

```python
from sco2rl.analysis import run_step_scenario

result = run_step_scenario(
    env=env,                  # gym.Env backed by MockFMU or real FMU
    policy=pid,               # any Controller (PID or RL)
    step_magnitude=2.0,       # MW change in W_net setpoint
    step_at_step=20,          # apply step at simulation step 20
    n_steps=200,
    dt=5.0,                   # seconds per step
    variable="W_net",
    phase=0,
    scenario="step_load_up",
)
print(f"Overshoot: {result.overshoot_pct:.1f}%")
print(f"Settling:  {result.settling_time_s:.0f}s")
print(f"IAE:       {result.iae:.1f}")
```

**Algorithm**:
1. Reset environment → warm-up for `step_at_step` steps using policy
2. Record initial steady-state value of `variable`
3. Apply setpoint step of `step_magnitude`
4. Continue for `n_steps - step_at_step` steps, recording response
5. Compute metrics: overshoot, settling time, rise time, IAE/ISE/ITAE via `scipy.integrate.trapezoid`

**Settling criterion**: response stays within ±2% of final value for ≥20 consecutive steps.
**Rise time criterion**: 10%→90% of step magnitude.

### 4.3 Frequency Response (`frequency_analysis.py`)

```python
from sco2rl.analysis import estimate_frequency_response

result = estimate_frequency_response(
    env=env,              # should use DynamicMockFMU for meaningful Bode plots
    policy=pid,
    channel_idx=0,        # which actuator to perturb (bypass valve)
    output_variable="W_net",
    prbs_amplitude=0.05,  # ±5% of action range
    n_bits=7,             # PRBS length = 2^7 - 1 = 127 samples
    n_periods=3,
    warmup_steps=20,
    phase=0,
)
print(f"Gain margin:   {result.gain_margin_db:.1f} dB")
print(f"Phase margin:  {result.phase_margin_deg:.1f}°")
print(f"Bandwidth:     {result.bandwidth_hz*1000:.2f} mHz")
```

See §6 for the PRBS/ETFE algorithm.

### 4.4 ScenarioRunner (`scenario_runner.py`)

Orchestrates batch analysis across phases, scenarios, and controllers:

```python
from sco2rl.analysis import ScenarioRunner, ControlScenario

runner = ScenarioRunner()
results = runner.run_all(
    env_factory=lambda: build_mock_env(dynamic=True),
    pid_policy=MultiLoopPID(),
    rl_policy=None,           # set to RLController.from_checkpoint(...) if available
    phases=[0, 1, 2],
    scenarios=[ControlScenario.STEP_LOAD_UP, ControlScenario.STEP_LOAD_DOWN],
    run_frequency=True,
)
runner.save("data/control_analysis_all_phases.json", results)
```

Available scenarios:

| Enum value | Description |
|------------|-------------|
| `STEP_LOAD_UP` | +20% W_net setpoint step |
| `STEP_LOAD_DOWN` | −20% W_net setpoint step |
| `LOAD_REJECTION` | −50% step (Phase 4 equivalent) |
| `FREQUENCY_RESPONSE` | PRBS excitation for Bode estimation |

---

## 5. PID Tuning Details

### IMC Tuning Formula

Internal Model Control (IMC) tuning is used because it has a single tuning knob (λ, the desired closed-loop time constant) and provides guaranteed gain/phase margins:

```
Process: G(s) = K · exp(-θs) / (τs + 1)
IMC:     Kp = τ / (K · λ)
         Ki = Kp / τ = 1 / (K · λ)
         Kd = (optional) Kp · τ_d
```

**Design target**: phase margin ≥ 45°, gain margin ≥ 6 dB.

### Channel Pairings and Design Point

The MockFMU 12×4 linearised sensitivity matrix was used to identify channel pairings at the design point (W_net = 10 MW):

| Channel | Actuator | Controlled var | K (sens) | τ (s) | λ (s) | Kp | Ki | Kd |
|---------|----------|----------------|----------|--------|-------|-----|-----|-----|
| 0 | bypass_valve | W_net | −2.5 MW/unit | 25 | 10 | 0.25 | 0.010 | 0.50 |
| 1 | igv_angle | T_turbine_inlet | −0.5 °C/unit | 60 | 12 | 0.010 | 0.0002 | 0.05 |
| 2 | inventory_valve | P_high | −3.0 MPa/unit | 30 | 10 | 0.30 | 0.010 | 0.50 |
| 3 | cooling_flow | T_comp_inlet | −3.0 °C/unit | 30 | 12 | 0.20 | 0.008 | 0.40 |

Gains are stored in `configs/control/pid_gains.yaml` and used as defaults in `MultiLoopPID`.

---

## 6. PRBS-Based Frequency Response Estimation

### Why PRBS?

Sinusoidal sweep to cover 0.001–0.05 Hz at 5 s/step would require ~10 000 simulation steps (≈14 hours of simulated time). PRBS with `n_bits=7` covers the same range in 381 steps, 26× faster.

### Algorithm (Empirical Transfer Function Estimation, ETFE)

```
1. Warm-up: run env for warmup_steps using policy to reach operating point

2. PRBS generation:
   - Generate n_periods × (2^n_bits − 1) binary sequence using shift register
   - Scale to ±prbs_amplitude (default ±0.05 of action range)

3. Injection:
   For each PRBS sample k:
     - Get nominal action from policy.predict(obs)
     - Add PRBS[k] to action[channel_idx], clip to [−1, 1]
     - Step environment, record (action_perturbation[k], output[k])

4. ETFE via cross-spectral density:
   H(f) = S_yu(f) / S_uu(f)
   where:
     S_yu(f) = scipy.signal.csd(output, input, fs=1/dt)
     S_uu(f) = scipy.signal.welch(input, fs=1/dt)

5. Margin computation (_compute_margins):
   - Phase crossover: first frequency where phase ≤ −180°
   - Gain margin = −magnitude_db at phase crossover
   - Gain crossover: first frequency where magnitude_db ≤ 0 dB
   - Phase margin = 180° + phase_deg at gain crossover
   - Bandwidth: highest frequency where magnitude_db ≥ peak_db − 3
```

### DynamicMockFMU

For realistic Bode plots without a real FMU, `_dynamic_mock.py` wraps `MockFMU` with first-order lag dynamics:

```
y[k] = α · y[k−1] + (1−α) · y_ss[k]
α = exp(−Δt / τ)
```

Default time constants: W_net τ = 25 s, T_turbine_inlet τ = 60 s, T_compressor_inlet τ = 30 s, P_high τ = 30 s.

---

## 7. Bug Log

### BUG-1 — IndexError in `step_response.py` (Fixed)

**File**: `src/sco2rl/analysis/step_response.py`
**Symptom**: `IndexError: index 14 is out of bounds for axis 0 with size 3` when episode terminates before `step_at_step` steps are recorded.
**Root cause**: `resp_arr[step_at_step - 1]` was indexed unconditionally; if the episode terminated early (due to BUG-2), `resp_arr` had fewer elements.
**Fix**: Clamped index to valid range:
```python
# Old:
initial_value = float(resp_arr[step_at_step - 1]) if step_at_step > 0 else float(resp_arr[0])

# Fixed:
initial_value = (
    float(resp_arr[min(max(step_at_step - 1, 0), len(resp_arr) - 1)])
    if len(resp_arr) > 0 else 0.0
)
```

### BUG-2 — Immediate episode termination from wrong action bounds (Fixed)

**File**: `src/sco2rl/analysis/scenario_runner.py`
**Symptom**: Episode terminates after ≤ 3 steps; all environment-based tests fail.
**Root cause**: `_MOCK_ENV_CONFIG["action_config"]` had `min: 0.0, max: 1.0`. `SCO2FMUEnv` maps `action_norm = 0` (PID output at design point) to:
```
physical = (0 + 1) / 2 × (1 − 0) + 0 = 0.5
```
MockFMU cooling-flow sensitivity: `T_comp_inlet += −3.0 × 0.5 = −1.5°C → 31.5°C < 32.2°C` (minimum constraint) → constraint violation on first step → termination.
**Fix**: Changed all four action channels to `min: −1.0, max: 1.0`:
```
physical = action_norm    (identity mapping when bounds are [−1, 1])
action_norm = 0  →  physical = 0  →  design point stable
```

---

## 8. File Manifest

### New files added by this enhancement

```
src/sco2rl/control/__init__.py
src/sco2rl/control/interfaces.py
src/sco2rl/control/pid.py
src/sco2rl/control/multi_loop_pid.py
src/sco2rl/control/rl_controller.py
src/sco2rl/analysis/__init__.py
src/sco2rl/analysis/metrics.py
src/sco2rl/analysis/step_response.py
src/sco2rl/analysis/frequency_analysis.py
src/sco2rl/analysis/scenario_runner.py
src/sco2rl/analysis/_dynamic_mock.py
configs/control/pid_gains.yaml
scripts/run_control_analysis.py
scripts/tune_pid.py
notebooks/05_control_analysis.ipynb
paper/sec_control_analysis.tex
paper/IMPLEMENTATION_PLAN.md                ← this file
paper/figures/step_response_phase0.png
paper/figures/step_response_phase2.png
paper/figures/bode_plot_phase0.png
paper/figures/control_metrics_heatmap.png
paper/figures/wnet_tracking_phase0.png
data/control_analysis_all_phases.json
data/control_analysis_phase0.json
tests/unit/control/__init__.py
tests/unit/control/test_pid.py              (11 tests)
tests/unit/control/test_rl_controller.py   (9 tests)
tests/unit/analysis/__init__.py
tests/unit/analysis/test_step_response.py  (13 PID + 3 RL = 16 tests)
tests/unit/analysis/test_frequency_analysis.py  (11 PID + 2 RL = 13 tests)
```

### Modified files

```
src/sco2rl/__init__.py              Added public API exports for control + analysis
pyproject.toml                      Added [control] optional extras, entry points
paper/sec_results.tex               Added "Control System Time-Series Response" subsection
paper/sec_conclusion.tex            Added control-theoretic findings paragraph
paper/main.tex                      Added \input{sec_control_analysis}
```

---

## 9. Acceptance Criteria

### Control Library
- [x] `from sco2rl.control import Controller, PIDController, MultiLoopPID, RLController` works
- [x] `PIDController` anti-windup prevents integral from exceeding saturation bounds
- [x] `MultiLoopPID` outperforms legacy flat-gains PID (lower IAE) on Phase 0 step test
- [x] `RLController.from_checkpoint()` gracefully handles missing file (raises `FileNotFoundError`)

### Analysis Module
- [x] `StepResponseResult` has all 9 metrics: overshoot, undershoot, settling, rise, peak, SSE, IAE, ISE, ITAE
- [x] `compute_step_metrics()` verified against first-order analytical result (rise ≈ 2.197τ, settling ≈ 3.91τ)
- [x] `generate_prbs()` produces ±amplitude binary sequence of length n_periods × (2^n_bits − 1)
- [x] `estimate_frequency_response()` returns finite gain/phase margins for DynamicMockFMU
- [x] Both `MultiLoopPID` and `MockRLPolicy` work as drop-in policy arguments

### Tests
- [x] 9 `test_rl_controller.py` tests pass
- [x] 11 `test_pid.py` tests pass
- [x] 16 `test_step_response.py` tests pass (13 PID + 3 RL)
- [x] 13 `test_frequency_analysis.py` tests pass (11 PID + 2 RL)

### Notebook
- [x] `notebooks/05_control_analysis.ipynb` renders without FMU (loads pre-computed JSON)
- [x] ipywidgets scenario selector (Phase, Controller, Scenario, Variable)
- [x] Step response plot with annotated settling time, overshoot, rise time
- [x] Bode plot with gain margin, phase margin annotations

### Paper
- [x] `paper/sec_control_analysis.tex` compiles without LaTeX errors
- [x] Tables include overshoot, settling, IAE, ITAE, gain margin, phase margin
- [x] 2+ step response figures and 1 Bode figure in `paper/figures/`

### SCOPE Library Integration
- [x] `pip install sco2rl[control]` installs `python-control>=0.10.0` and `ipywidgets>=8.0`
- [x] `sco2-analyze-control --use-mock` CLI runs end-to-end
- [x] `sco2-tune-pid --use-mock` CLI outputs YAML gain recommendations

---

## 10. Usage Examples

### Run control analysis with MockFMU (no real FMU needed)

```bash
# Full analysis — all 7 phases, PID only (no RL checkpoint available)
sco2-analyze-control --use-mock --phases 0 1 2 3 4 5 6 --output-dir data/

# With RL checkpoint
sco2-analyze-control --use-mock \
    --checkpoint artifacts/checkpoints/run01/final \
    --phases 0 1 2 \
    --output-dir data/

# Frequency response only
sco2-analyze-control --use-mock --phases 0 --no-step
```

### Auto-tune PID gains

```bash
sco2-tune-pid --use-mock
# → writes configs/control/pid_gains.yaml with IMC-derived gains
```

### Use the Python API directly

```python
from sco2rl.control import MultiLoopPID, RLController
from sco2rl.analysis import ScenarioRunner, ControlScenario, build_mock_env

pid = MultiLoopPID()
# rl = RLController.from_checkpoint("artifacts/checkpoints/run01/final")

runner = ScenarioRunner()
results = runner.run_all(
    env_factory=lambda: build_mock_env(dynamic=True),
    pid_policy=pid,
    rl_policy=None,                                 # or rl
    phases=[0, 1, 2],
    scenarios=[ControlScenario.STEP_LOAD_UP],
)

runner.save("data/my_results.json", results)

# Inspect results
r = results[0]
print(f"PID settling: {r.pid_step.settling_time_s:.0f} s")
print(f"PID IAE:      {r.pid_step.iae:.1f}")
print(f"PID ITAE:     {r.pid_step.itae:.1f}")
if r.pid_freq:
    print(f"PID gain margin:  {r.pid_freq.gain_margin_db:.1f} dB")
    print(f"PID phase margin: {r.pid_freq.phase_margin_deg:.1f}°")
```

### Open the interactive notebook

```bash
jupyter notebook notebooks/05_control_analysis.ipynb
# OR on Google Colab: all data is pre-computed in data/control_analysis_*.json
```

The notebook auto-detects whether it is running in Colab (loads JSON) or locally (can re-run live simulation with MockFMU).

---

*Document maintained alongside the codebase. For questions, see the inline docstrings in `src/sco2rl/control/` and `src/sco2rl/analysis/`.*
