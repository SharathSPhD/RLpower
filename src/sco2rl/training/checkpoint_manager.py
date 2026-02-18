"""CheckpointManager -- saves and loads full training state per RULE-C4.

RULE-C4: checkpoint MUST include ALL of:
  1. model_weights  (path to SB3 .zip)
  2. vecnorm_stats  (VecNormalize running mean/var dict)
  3. curriculum_phase  (int)
  4. lagrange_multipliers  (dict[str, float])
  5. total_timesteps  (int)

Raises ValueError on load if any field is missing.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sco2rl.training.lagrangian_ppo import LagrangianPPO

# Required checkpoint fields (RULE-C4)
_REQUIRED_FIELDS = {
    "model_path",
    "vecnorm_stats",
    "curriculum_phase",
    "lagrange_multipliers",
    "total_timesteps",
}


class CheckpointManager:
    """Saves and loads full training state.

    File layout::

        <checkpoint_dir>/<run_name>/
            step_<step>_phase_<phase>_checkpoint.json   <- metadata + stats
            step_<step>_phase_<phase>_model.zip         <- SB3 model weights
            step_<step>_phase_<phase>_model_multipliers.pkl

    Parameters
    ----------
    checkpoint_dir:
        Root directory where checkpoint subdirectories are created.
    run_name:
        Unique name for this training run (used as subdirectory).
    """

    def __init__(self, checkpoint_dir: str, run_name: str) -> None:
        self._run_dir = Path(checkpoint_dir) / run_name
        self._run_dir.mkdir(parents=True, exist_ok=True)

    # -- Saving ---------------------------------------------------------------

    def save(
        self,
        model: "LagrangianPPO",
        vecnorm_stats: dict,
        curriculum_phase: int,
        lagrange_multipliers: dict[str, float],
        total_timesteps: int,
        step: int,
        vecnorm: object | None = None,
    ) -> str:
        """Save checkpoint, return path to the JSON metadata file.

        Parameters
        ----------
        model:
            LagrangianPPO instance to save.
        vecnorm_stats:
            Legacy field kept for RULE-C4 schema compat (value stored in JSON).
        curriculum_phase:
            Current curriculum phase (int).
        lagrange_multipliers:
            Current lambda values dict.
        total_timesteps:
            Total environment steps trained so far.
        step:
            Training step counter (used in filename for ordering).
        vecnorm:
            Optional VecNormalize instance. When provided, its running
            mean/variance stats are serialised via VecNormalize.save() so
            they can be restored on resume.  This is the correct way to
            persist observation normalization state across interruptions.
        """
        prefix = f"step_{step:08d}_phase_{curriculum_phase}"
        model_path_stem = str(self._run_dir / f"{prefix}_model")

        # Save model weights (creates <stem>.zip and <stem>_multipliers.pkl)
        model.save(model_path_stem)

        # Persist VecNormalize running stats when the instance is supplied.
        vecnorm_path: str | None = None
        if vecnorm is not None:
            vecnorm_path = str(self._run_dir / f"{prefix}_vecnorm.pkl")
            vecnorm.save(vecnorm_path)

        # Build metadata dict (RULE-C4 all 5 fields + optional vecnorm_path)
        checkpoint_data = {
            "model_path": model_path_stem,                 # field 1
            "vecnorm_stats": vecnorm_stats,                # field 2 (legacy)
            "vecnorm_path": vecnorm_path,                  # real stats file path
            "curriculum_phase": curriculum_phase,          # field 3
            "lagrange_multipliers": lagrange_multipliers,  # field 4
            "total_timesteps": total_timesteps,            # field 5
            "step": step,
        }

        json_path = str(self._run_dir / f"{prefix}_checkpoint.json")
        with open(json_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        return json_path

    # -- Loading --------------------------------------------------------------

    def load(self, path: str) -> dict:
        """Load checkpoint dict.

        Returns dict with keys: model_path, vecnorm_stats, curriculum_phase,
        lagrange_multipliers, total_timesteps, step.

        Raises
        ------
        ValueError
            If any RULE-C4 required field is missing from the checkpoint.
        FileNotFoundError
            If path does not exist.
        """
        with open(path, "r") as f:
            data = json.load(f)

        missing = _REQUIRED_FIELDS - set(data.keys())
        if missing:
            raise ValueError(
                f"Checkpoint at '{path}' is missing RULE-C4 required fields: {missing}"
            )

        return data

    # -- Discovery ------------------------------------------------------------

    def get_latest(self) -> str | None:
        """Return path to the most recent checkpoint JSON, or None."""
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None

    def list_checkpoints(self) -> list[str]:
        """Return all checkpoint JSON paths sorted oldest -> newest by step."""
        pattern = re.compile(r"step_(\d+)_phase_\d+_checkpoint\.json$")
        results: list[tuple[int, str]] = []
        for p in self._run_dir.iterdir():
            m = pattern.match(p.name)
            if m:
                step_num = int(m.group(1))
                results.append((step_num, str(p)))
        results.sort(key=lambda x: x[0])
        return [path for _, path in results]
