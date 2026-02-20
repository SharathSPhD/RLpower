# DEPRECATED: skrl_ppo_results.json

This file contains results from the failed SKRL/FNO training path where
mean_reward=-100 (immediate termination). The FNO surrogate was found 
architecturally incompatible with step-by-step RL (non-causal sequence-to-sequence
architecture causes spectral aliasing when queried step-by-step).

**Use `mlp_ppo_results.json` instead** — this contains the successful MLP 
surrogate PPO results (mean_reward=+27.1, 18.5x better tracking than PID).
