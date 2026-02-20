# sCO2RL — Deep RL for Supercritical CO₂ Brayton Cycle Control

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/RLpower/blob/main/notebooks/04_policy_evaluation.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

**Author:** Sharath Sathish, University of York, UK

Deep reinforcement learning for autonomous control of a **supercritical CO₂ (sCO₂) recuperated Brayton cycle** recovering waste heat from steel industry electric arc furnace (EAF) and basic oxygen furnace (BOF) exhaust (200–1,200°C). Trained on a physics-faithful OpenModelica FMU via the FMPy interface on an NVIDIA DGX Spark (GB10 Grace Blackwell, 128 GB unified memory).

## Key Results

| Metric | Value |
|--------|-------|
| RL vs ZN-PID: Phase 0 (steady-state) | **+30.3%** cumulative reward |
| RL vs ZN-PID: Phase 1 (±30% load following) | **+30.4%** |
| RL vs ZN-PID: Phase 2 (ambient disturbance) | **+39.0%** |
| Phases 3–6 (severe transients) | PID wins — curriculum imbalance (<5% training steps each) |
| Constraint violations (140 eval episodes) | **0** (RL and PID) |
| MLP surrogate: PPO vs PID tracking error | **18.5× lower** (0.122 MW vs 2.259 MW) |
| MLP surrogate: GPU training throughput | **250,000 steps/s** (470× faster than FMU) |
| FNO surrogate (PhysicsNeMo) | R² = 1.000, RMSE = 0.0010 (76,600 LHS trajectories) |
| TensorRT FP16 deployment | p99 = **0.046 ms** (22× under 1 ms SLA) |
| Training hardware | NVIDIA DGX Spark, GB10 Grace Blackwell, 128 GB |

## What this repository provides

- **Physics simulation**: OpenModelica FMU (FMI 2.0 Co-Simulation) via FMPy with CoolProp Span-Wagner CO₂ EOS
- **Gymnasium environment**: 14-variable observation, 4-dim continuous action space, Lagrangian safety constraints
- **7-phase curriculum**: steady-state → load following → ambient disturbance → EAF transients → load rejection → cold startup → emergency trip
- **Dual training paths**: FMU-direct PPO (SB3, 8 CPU workers, 530 steps/s) and MLP surrogate GPU path (1,024 parallel envs, 250K steps/s)
- **Surrogate models**: MLP step predictor (val_loss = 5×10⁻⁶) + NVIDIA PhysicsNeMo FNO (R² = 1.000)
- **Deployment**: PyTorch → ONNX → TensorRT FP16, p99 = 0.046 ms
- **Paper**: Full LaTeX manuscript (41 pages, 4 appendices) ready for arXiv submission
- **Practitioner lessons**: Five non-trivial infrastructure bugs documented with diagnosis, fix, and detection strategy

## Interactive Notebooks (run on Google Colab — no setup required)

| Notebook | Description | Colab |
|----------|-------------|-------|
| [01_cycle_analysis](notebooks/01_cycle_analysis.ipynb) | Open-loop FMU thermodynamic traces | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/RLpower/blob/main/notebooks/01_cycle_analysis.ipynb) |
| [02_reward_shaping](notebooks/02_reward_shaping.ipynb) | Reward component diagnostics | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/RLpower/blob/main/notebooks/02_reward_shaping.ipynb) |
| [03_surrogate_validation](notebooks/03_surrogate_validation.ipynb) | FNO surrogate V1 vs V2 fidelity analysis | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/RLpower/blob/main/notebooks/03_surrogate_validation.ipynb) |
| [04_policy_evaluation](notebooks/04_policy_evaluation.ipynb) | RL vs PID evaluation across all phases | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/RLpower/blob/main/notebooks/04_policy_evaluation.ipynb) |
| [05_control_analysis](notebooks/05_control_analysis.ipynb) | Step response, Bode plots, control metrics | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/RLpower/blob/main/notebooks/05_control_analysis.ipynb) |

All notebooks auto-detect Google Colab, clone the repo, and install requirements. No Google Drive connection needed.

## Repository layout

```
src/sco2rl/          Core library (environment, training, surrogate, deployment)
configs/             YAML configs (environment, curriculum, surrogate, training)
scripts/             CLI scripts (train, evaluate, export, figure generation)
notebooks/           Interactive analysis notebooks (Colab-ready, with inline outputs)
paper/               LaTeX manuscript (arxiv-compatible split .tex + .bib)
data/                Pre-computed report JSON files (tracked for Colab)
tests/               Unit tests
```

## Quickstart (Docker, NVIDIA DGX Spark)

```bash
# Build image (compiles OpenModelica + CoolProp + ExternalMedia for ARM64)
docker build -t sco2-rl-automation:latest .

# Launch with GPU access
docker run --rm -it --gpus all -v $(pwd):/workspace --shm-size=64g sco2-rl-automation:latest

# Inside container: collect trajectories, train surrogate, train RL
python scripts/collect_trajectories.py --n-trajectories 100000
python scripts/train_mlp_surrogate.py
python scripts/train_ppo_mlp.py
python scripts/cross_validate_and_export.py
```

## Paper

The paper is in `paper/` with split section files for arXiv compatibility:

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Pre-compiled PDF: `paper/main.pdf` (41 pages)

## Citation

```bibtex
@article{sathish2026sco2rl,
  title={Deep Reinforcement Learning for Autonomous Control of Supercritical CO$_2$ Brayton Cycles in Steel Industry Waste Heat Recovery},
  author={Sathish, Sharath},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT
