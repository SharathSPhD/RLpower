"""CrossValidator -- compare two trained policies on the same FMU environment.

Selects the better policy based on a configurable metric
(mean_reward: higher is better; violation_rate: lower is better).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from sco2rl.training.policy_evaluator import EvaluationMetrics, PolicyEvaluator


@dataclass
class CrossValidationReport:
    """Full report from a two-path cross-validation run."""

    path_a_metrics: EvaluationMetrics
    path_b_metrics: EvaluationMetrics
    selected_path: str          # "path_a" or "path_b"
    selection_reason: str       # human-readable explanation
    timestamp: str              # ISO 8601 format


def _metrics_to_dict(m: EvaluationMetrics) -> dict:
    return {
        "mean_reward": m.mean_reward,
        "std_reward": m.std_reward,
        "mean_episode_length": m.mean_episode_length,
        "violation_rate": m.violation_rate,
        "T_comp_inlet_min": m.T_comp_inlet_min,
        "T_comp_inlet_mean": m.T_comp_inlet_mean,
        "n_episodes": m.n_episodes,
        "phase": m.phase,
        "per_episode_rewards": m.per_episode_rewards,
    }


def _metrics_from_dict(d: dict) -> EvaluationMetrics:
    return EvaluationMetrics(
        mean_reward=float(d["mean_reward"]),
        std_reward=float(d["std_reward"]),
        mean_episode_length=float(d["mean_episode_length"]),
        violation_rate=float(d["violation_rate"]),
        T_comp_inlet_min=float(d["T_comp_inlet_min"]),
        T_comp_inlet_mean=float(d["T_comp_inlet_mean"]),
        n_episodes=int(d["n_episodes"]),
        phase=int(d["phase"]),
        per_episode_rewards=list(d.get("per_episode_rewards", [])),
    )


class CrossValidator:
    """Compare two policies and select the better one.

    Parameters
    ----------
    env:
        Gymnasium-compatible environment used for evaluation.
    config:
        Dict with keys:
          - n_eval_episodes (int)
          - T_comp_inlet_var (str)
          - deterministic (bool)
          - selection_metric (str): "mean_reward" (higher=better) or
            "violation_rate" (lower=better).
    evaluator:
        Optional PolicyEvaluator to inject (for testing). If None, one is
        created internally from env + config.
    """

    def __init__(self, env, config: dict, evaluator=None) -> None:
        self._env = env
        self._config = config
        self._selection_metric: str = config.get("selection_metric", "mean_reward")

        # Injected or internally constructed evaluator
        self._evaluator = evaluator if evaluator is not None else PolicyEvaluator(
            env=env, config=config
        )

    def compare(self, model_a, model_b) -> CrossValidationReport:
        """Evaluate both models and return a CrossValidationReport.

        The evaluator's ``evaluate()`` is called twice: once for model_a,
        once for model_b.
        """
        metrics_a = self._evaluator.evaluate(model_a)
        metrics_b = self._evaluator.evaluate(model_b)

        metric = self._selection_metric
        val_a = getattr(metrics_a, metric)
        val_b = getattr(metrics_b, metric)

        # For mean_reward: higher is better.  For violation_rate: lower is better.
        if metric == "violation_rate":
            path_a_wins = val_a <= val_b
        else:
            path_a_wins = val_a >= val_b

        selected_path = "path_a" if path_a_wins else "path_b"
        winning_val = val_a if path_a_wins else val_b
        losing_val = val_b if path_a_wins else val_a

        if metric == "violation_rate":
            reason = (
                f"Selected {selected_path} by {metric}: "
                f"{winning_val:.4f} vs {losing_val:.4f} (lower is better)."
            )
        else:
            reason = (
                f"Selected {selected_path} by {metric}: "
                f"{winning_val:.4f} vs {losing_val:.4f} (higher is better)."
            )

        return CrossValidationReport(
            path_a_metrics=metrics_a,
            path_b_metrics=metrics_b,
            selected_path=selected_path,
            selection_reason=reason,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )

    def save_report(self, report: CrossValidationReport, path: str) -> None:
        """Serialize CrossValidationReport to JSON."""
        import os

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        data = {
            "path_a_metrics": _metrics_to_dict(report.path_a_metrics),
            "path_b_metrics": _metrics_to_dict(report.path_b_metrics),
            "selected_path": report.selected_path,
            "selection_reason": report.selection_reason,
            "timestamp": report.timestamp,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_report(cls, path: str) -> CrossValidationReport:
        """Deserialize CrossValidationReport from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        return CrossValidationReport(
            path_a_metrics=_metrics_from_dict(data["path_a_metrics"]),
            path_b_metrics=_metrics_from_dict(data["path_b_metrics"]),
            selected_path=data["selected_path"],
            selection_reason=data["selection_reason"],
            timestamp=data["timestamp"],
        )
