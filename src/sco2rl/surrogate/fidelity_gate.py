"""FidelityGate: evaluate surrogate model fidelity against reference FMU data.

Computes per-variable RMSE and R2, checks against configured thresholds,
and produces a FidelityReport that drives the Stage 3 gate decision.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FidelityReport:
    """Results from FidelityGate.evaluate().

    Attributes
    ----------
    overall_rmse_normalized : float
        Mean normalized RMSE across all output variables.
    overall_r2 : float
        Mean R2 across all output variables.
    per_variable : dict
        Per-variable metrics: {name: {rmse, r2}}.
    critical_variables : dict
        Critical-variable metrics: {name: {rmse, r2, passed}}.
    passed : bool
        True iff ALL configured thresholds are satisfied.
    timestamp : str
        ISO 8601 UTC timestamp of evaluation.
    """

    overall_rmse_normalized: float
    overall_r2: float
    per_variable: dict[str, dict[str, Any]] = field(default_factory=dict)
    critical_variables: dict[str, dict[str, Any]] = field(default_factory=dict)
    passed: bool = False
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


class FidelityGate:
    """Evaluate surrogate model fidelity against reference (FMU) trajectories.

    Parameters
    ----------
    config : dict
        Must contain at least:
        - max_rmse_normalized (float)
        - min_r2 (float)
        - critical_variables (list of dicts with name + thresholds)
        - variable_ranges (dict mapping variable name -> physical range for normalization)
    """

    def __init__(self, config: dict) -> None:
        self._max_rmse: float = config["max_rmse_normalized"]
        self._min_r2: float = config["min_r2"]
        self._critical_cfg: list[dict] = config.get("critical_variables", [])
        self._var_ranges: dict[str, float] = config.get("variable_ranges", {})

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        variable_names: list[str],
    ) -> FidelityReport:
        """Compute RMSE and R2 for each variable and check gate thresholds.

        Parameters
        ----------
        predictions : np.ndarray
            Shape (N, T, n_vars) -- model rollout outputs.
        targets : np.ndarray
            Shape (N, T, n_vars) -- ground-truth FMU outputs.
        variable_names : list[str]
            Names corresponding to the n_vars dimension.

        Returns
        -------
        FidelityReport
        """
        n_vars = len(variable_names)
        assert predictions.shape[-1] == n_vars, (
            f"predictions last dim {predictions.shape[-1]} != len(variable_names) {n_vars}"
        )
        assert targets.shape[-1] == n_vars, (
            f"targets last dim {targets.shape[-1]} != len(variable_names) {n_vars}"
        )

        per_variable: dict[str, dict[str, Any]] = {}
        rmse_list: list[float] = []
        r2_list: list[float] = []

        for i, name in enumerate(variable_names):
            pred_i = predictions[..., i].ravel()
            tgt_i = targets[..., i].ravel()

            # Raw RMSE
            raw_rmse = float(np.sqrt(np.mean((pred_i - tgt_i) ** 2)))

            # Normalize by variable range
            var_range = self._var_ranges.get(name, 1.0)
            norm_rmse = raw_rmse / max(var_range, 1e-12)

            # R2
            ss_res = np.sum((pred_i - tgt_i) ** 2)
            ss_tot = np.sum((tgt_i - tgt_i.mean()) ** 2)
            r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))

            per_variable[name] = {"rmse": norm_rmse, "r2": r2}
            rmse_list.append(norm_rmse)
            r2_list.append(r2)

        overall_rmse = float(np.mean(rmse_list))
        overall_r2 = float(np.mean(r2_list))

        # ── Critical variable checks ──────────────────────────────────────────
        critical_variables: dict[str, dict[str, Any]] = {}
        crit_all_pass = True

        for crit in self._critical_cfg:
            name = crit["name"]
            if name not in variable_names:
                # Skip gracefully if this variable is not in the current evaluation set
                continue

            metrics = per_variable[name]
            norm_rmse = metrics["rmse"]
            r2 = metrics["r2"]

            # Determine per-critical RMSE threshold
            # Use max_rmse_c / variable_range if provided, else max_rmse directly
            if "max_rmse_c" in crit:
                var_range = self._var_ranges.get(name, 1.0)
                threshold_rmse = crit["max_rmse_c"] / max(var_range, 1e-12)
            elif "max_rmse" in crit:
                var_range = self._var_ranges.get(name, 1.0)
                threshold_rmse = crit["max_rmse"] / max(var_range, 1e-12)
            else:
                threshold_rmse = self._max_rmse

            threshold_r2 = crit.get("min_r2", self._min_r2)
            passed = bool(norm_rmse <= threshold_rmse and r2 >= threshold_r2)
            if not passed:
                crit_all_pass = False

            critical_variables[name] = {
                "rmse": norm_rmse,
                "r2": r2,
                "passed": passed,
                "threshold_rmse": threshold_rmse,
                "threshold_r2": threshold_r2,
            }

        # ── Overall gate decision ─────────────────────────────────────────────
        gate_passed = (
            overall_rmse <= self._max_rmse
            and overall_r2 >= self._min_r2
            and crit_all_pass
        )

        return FidelityReport(
            overall_rmse_normalized=overall_rmse,
            overall_r2=overall_r2,
            per_variable=per_variable,
            critical_variables=critical_variables,
            passed=gate_passed,
        )

    def save_report(self, report: FidelityReport, path: str) -> None:
        """Save FidelityReport as JSON to `path`, creating parent dirs if needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump(asdict(report), f, indent=2)

    @classmethod
    def load_report(cls, path: str) -> FidelityReport:
        """Load a previously saved FidelityReport from JSON."""
        with open(path) as f:
            data = json.load(f)
        return FidelityReport(**data)
