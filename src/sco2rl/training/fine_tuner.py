"""FineTuner -- fine-tune a selected checkpoint on the real FMU environment.

Loads a RULE-C4 checkpoint, sets a lower learning rate, runs for
finetune_steps environment steps, and saves the updated checkpoint.

This is Stage 4 of the sco2rl pipeline: after cross-validation selects the
best policy (FMU-direct or surrogate-trained), the FineTuner corrects any
surrogate bias by continuing PPO training on the actual FMU.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from sco2rl.environment.sco2_env import SCO2FMUEnv


class FineTuner:
    """Fine-tune the selected checkpoint on the real FMU environment.

    Parameters
    ----------
    env_factory:
        Callable that returns a fresh SCO2FMUEnv each time it is called.
        Called once to build the training environment.
    config:
        Dict with keys:
          - finetune_steps (int): total environment steps to train
          - finetune_lr (float): learning rate (lower than training lr)
          - checkpoint_dir (str): directory where fine-tuned checkpoint is saved
          - eval_freq (int): evaluate every N steps
          - eval_episodes (int): number of episodes per evaluation
          - seed (int): RNG seed
    """

    def __init__(
        self,
        env_factory: Callable[[], SCO2FMUEnv],
        config: dict,
    ) -> None:
        self._env_factory = env_factory
        self._config = config
        self._model: PPO | None = None
        self._curriculum_phase: int = 0

    # ── Public interface ──────────────────────────────────────────────────────

    def finetune(self, checkpoint_path: str) -> dict:
        """Load checkpoint, fine-tune, return metrics dict.

        Parameters
        ----------
        checkpoint_path:
            Path to RULE-C4 JSON metadata file.

        Returns
        -------
        dict with keys:
            - total_timesteps (int)
            - final_mean_reward (float)
            - checkpoint_path (str): path to saved fine-tuned checkpoint JSON
        """
        cfg = self._config
        finetune_steps: int = int(cfg["finetune_steps"])
        finetune_lr: float = float(cfg["finetune_lr"])
        checkpoint_dir: str = str(cfg["checkpoint_dir"])
        eval_freq: int = int(cfg.get("eval_freq", finetune_steps))
        eval_episodes: int = int(cfg.get("eval_episodes", 5))
        seed: int = int(cfg.get("seed", 42))

        os.makedirs(checkpoint_dir, exist_ok=True)

        # ── 1. Load checkpoint metadata ────────────────────────────────────
        with open(checkpoint_path, "r") as f:
            meta = json.load(f)

        original_timesteps: int = int(meta.get("total_timesteps", 0))
        self._curriculum_phase = int(meta.get("curriculum_phase", 0))

        # ── 2. Build training environment ─────────────────────────────────
        env = self._env_factory()
        vec_env = DummyVecEnv([lambda: env])

        # Support both SB3 checkpoints (model_path) and SKRL checkpoints
        # (model_weights). SKRL manifests are bridged into an SB3 RULE-C4
        # checkpoint before fine-tuning.
        if "model_path" not in meta:
            if "model_weights" not in meta:
                raise ValueError(
                    "Checkpoint must contain either 'model_path' (SB3) "
                    "or 'model_weights' (SKRL)."
                )
            from sco2rl.training.checkpoint_bridge import convert_skrl_to_sb3_checkpoint

            bridge_dir = os.path.join(checkpoint_dir, "bridge")
            bridged_checkpoint = convert_skrl_to_sb3_checkpoint(
                skrl_checkpoint_path=checkpoint_path,
                output_dir=bridge_dir,
                env=vec_env,
                model_stem="skrl_to_sb3_bridge",
            )
            with open(bridged_checkpoint, "r") as f:
                meta = json.load(f)
            original_timesteps = int(meta.get("total_timesteps", original_timesteps))
            self._curriculum_phase = int(meta.get("curriculum_phase", self._curriculum_phase))

        weights_path: str = meta["model_path"]
        # SB3 saves with .zip; handle both with and without extension
        if not weights_path.endswith(".zip"):
            weights_path_zip = weights_path + ".zip"
        else:
            weights_path_zip = weights_path

        # ── 3. Load SB3 PPO weights and set fine-tuning LR ────────────────
        # set_env allows reuse of weights on a new env
        model = PPO.load(
            weights_path_zip,
            env=vec_env,
            verbose=0,
            seed=seed,
        )
        # Override learning rate
        model.learning_rate = finetune_lr
        model.policy.optimizer.param_groups[0]["lr"] = finetune_lr

        # ── 4. Fine-tune ───────────────────────────────────────────────────
        model.learn(
            total_timesteps=finetune_steps,
            reset_num_timesteps=False,  # continue from loaded num_timesteps
        )
        self._model = model

        # ── 5. Evaluate final policy ───────────────────────────────────────
        eval_env = self._env_factory()
        eval_vec_env = DummyVecEnv([lambda: eval_env])
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_vec_env,
            n_eval_episodes=eval_episodes,
            deterministic=True,
        )
        eval_vec_env.close()

        # ── 6. Save fine-tuned checkpoint (simplified, no LagrangianPPO) ──
        total_ts = original_timesteps + finetune_steps
        run_prefix = f"finetune_step_{total_ts:08d}"
        model_save_path = os.path.join(checkpoint_dir, f"{run_prefix}_model")
        model.save(model_save_path)

        ft_meta = {
            "model_path": model_save_path,
            "vecnorm_stats": None,
            "curriculum_phase": self._curriculum_phase,
            "lagrange_multipliers": {},
            "total_timesteps": total_ts,
            "step": total_ts,
            "finetune_lr": finetune_lr,
            "final_mean_reward": float(mean_reward),
        }
        ft_meta_path = os.path.join(checkpoint_dir, f"{run_prefix}_checkpoint.json")
        with open(ft_meta_path, "w") as f:
            json.dump(ft_meta, f, indent=2)

        vec_env.close()

        return {
            "total_timesteps": total_ts,
            "final_mean_reward": float(mean_reward),
            "checkpoint_path": ft_meta_path,
        }

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def model(self) -> PPO | None:
        """The SB3 PPO model after fine-tuning (None before finetune() call)."""
        return self._model

    @property
    def curriculum(self) -> int:
        """Curriculum phase at the time the checkpoint was loaded."""
        return self._curriculum_phase
