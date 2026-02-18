# sCO2RL — Deep RL for Supercritical CO₂ Brayton Cycle Control

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/RLpower/blob/main/notebooks/04_policy_evaluation.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

Deep reinforcement learning for autonomous control of a **supercritical CO₂ (sCO₂) simple recuperated Brayton cycle** recovering waste heat from steel industry electric arc furnace (EAF) and basic oxygen furnace (BOF) exhaust (200–1,200°C).

## Key Results

| Metric | Value |
|--------|-------|
| RL vs PID improvement (Phases 0–2) | **+24–29%** (steady-state, gradual load, ambient disturbance) |
| Curriculum phases traversed | **7/7** (Phase 0 → 6) within 229,376 steps |
| Phase 6 mean reward (emergency turbine trip) | **412.7** |
| Constraint violations (all 70 eval episodes) | **0.000** (Lagrangian safety mechanism) |
| TensorRT p99 inference latency | **0.046 ms** (22× under 1 ms SLA) |
| Interleaved replay run | **In progress** (3M steps; prevents catastrophic forgetting) |

## What this repository provides

- **Physics simulation**: OpenModelica FMU (FMI 2.0 Co-Simulation) via FMPy with CoolProp Span-Wagner CO₂ EOS
- **Gymnasium environment**: 100-dim observation (20 vars × 5 history), 5-dim continuous action space, Lagrangian safety constraints
- **7-phase curriculum**: steady-state → load following → ambient disturbance → EAF transients → load rejection → cold startup → emergency trip
- **Dual training paths**: FMU-direct PPO (SB3, 8 CPU workers) and FNO surrogate GPU path (SKRL)
- **Deployment**: PyTorch → ONNX → TensorRT FP16, p99 = 0.046 ms
- **Practitioner lessons**: Five non-trivial infrastructure bugs documented with diagnosis, fix, and detection strategy

## Interactive Notebooks (run on Google Colab — no setup required)

| Notebook | Description | Colab |
|----------|-------------|-------|
| [01_cycle_analysis](notebooks/01_cycle_analysis.ipynb) | Open-loop FMU thermodynamic traces | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/RLpower/blob/main/notebooks/01_cycle_analysis.ipynb) |
| [02_reward_shaping](notebooks/02_reward_shaping.ipynb) | Reward component diagnostics with MockFMU | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/RLpower/blob/main/notebooks/02_reward_shaping.ipynb) |
| [03_surrogate_validation](notebooks/03_surrogate_validation.ipynb) | FNO surrogate fidelity metrics | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/RLpower/blob/main/notebooks/03_surrogate_validation.ipynb) |
| [04_policy_evaluation](notebooks/04_policy_evaluation.ipynb) | RL vs PID evaluation across all phases | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/RLpower/blob/main/notebooks/04_policy_evaluation.ipynb) |

All notebooks auto-detect Google Colab, clone the repo, and install requirements. No Google Drive connection needed.

## Repository layout

```
src/sco2rl/          Core library (environment, training, surrogate, deployment)
configs/             YAML configs (environment, curriculum, surrogate, training)
scripts/             CLI scripts (train, evaluate, export, monitor)
notebooks/           Interactive analysis notebooks (Colab-ready)
paper/               LaTeX manuscript and figure generation
data/                Small pre-computed report JSON files (tracked for Colab)
tests/               Unit tests
```

## Quickstart (Docker, NVIDIA DGX Spark)

```bash
# Build image (compiles OpenModelica + CoolProp + ExternalMedia for ARM64)
docker build -t sco2-rl-automation:latest .

# Launch with GPU access
docker run --rm -it --gpus all -v $(pwd):/workspace --shm-size=64g sco2-rl-automation:latest

# Inside container — run FMU training
cd /workspace
PYTHONPATH=src python scripts/train_fmu.py \
  --n-envs 8 \
  --total-timesteps 5000000 \
  --fmu-path artifacts/fmu_build/SCO2RecuperatedCycle.fmu \
  --checkpoint-dir artifacts/checkpoints/run01 \
  --run-name run01 \
  --verbose 1

# Monitor training progress
PYTHONPATH=src python scripts/monitor_fmu_training.py \
  --checkpoint-dir artifacts/checkpoints/run01/run01 \
  --training-log /tmp/train.log \
  --poll-seconds 120
```

## Key CLI workflows

```bash
# Surrogate fidelity report
PYTHONPATH=src python scripts/fidelity_report_fmu.py \
  --steps 1000 \
  --output artifacts/surrogate/fidelity_report.json

# RL vs PID cross-validation
PYTHONPATH=src python scripts/cross_validate_and_export.py \
  --checkpoint artifacts/checkpoints/run01/run01/step_XXXXXX_phase_6_checkpoint.json

# Generate paper figures
PYTHONPATH=src python paper/generate_figures.py
```

## Practitioner Bugs Documented

Five non-trivial infrastructure defects discovered and fixed during development, each documented with root cause, fix, and a test strategy to prevent recurrence:

1. **VecNormalize persistence failure** — training stalled in Phase 0 for 2.8M steps; fix: always save/load `vecnorm.pkl` alongside policy weights
2. **Episode boundary misalignment** — `CurriculumCallback._on_rollout_end` missed 99% of episode completions; fix: moved recording to `_on_step`
3. **Reward unit double-scaling** — `FMPyAdapter` W→MW + `env.yaml` 1e-6 factor collapsed `r_tracking` to ~0; fix: `w_net_unit_scale: 1.0`
4. **Stale disturbance profile** — `set_curriculum_phase()` didn't rebuild profile, causing `KeyError` on phase transition; fix: rebuild atomically
5. **Zero-violation advancement gate** — stochastic exploration blocked curriculum by requiring 0.0 violation rate; fix: allow up to 10% during training

See `paper/main.tex` Section 6 for full diagnosis and lessons.

## Publication

Manuscript: `paper/main.tex` — *Deep Reinforcement Learning for Autonomous Control of Supercritical CO₂ Brayton Cycles in Steel Industry Waste Heat Recovery*

## Citation

```bibtex
@misc{sco2rl2026,
  title  = {Deep Reinforcement Learning for Autonomous Control of
            Supercritical CO2 Brayton Cycles in Steel Industry Waste Heat Recovery},
  author = {sCO2RL Project Team},
  year   = {2026},
  url    = {https://github.com/SharathSPhD/RLpower}
}
```

## License

MIT — see `pyproject.toml` and repository metadata.
