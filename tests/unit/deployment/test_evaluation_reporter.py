"""Tests for EvaluationReporter."""
import numpy as np
import pytest

from sco2rl.training.policy_evaluator import EvaluationMetrics

def make_metrics(mean_reward: float, violation_rate: float = 0.0, phase: int = 0):
    return EvaluationMetrics(
        mean_reward=mean_reward, std_reward=0.1, mean_episode_length=20.0,
        violation_rate=violation_rate, T_comp_inlet_min=32.5, T_comp_inlet_mean=33.0,
        n_episodes=3, phase=phase, per_episode_rewards=[mean_reward] * 3,
    )

class _MockEvaluator:
    """Returns alternating RL/PID metrics."""
    def __init__(self, rl_metrics, pid_metrics):
        self._metrics = iter(rl_metrics + pid_metrics)
        self._rl_rewards = [m.mean_reward for m in rl_metrics]
        self._pid_rewards = [m.mean_reward for m in pid_metrics]
        self._calls = 0
    def evaluate(self, model):
        self._calls += 1
        try:
            return next(self._metrics)
        except StopIteration:
            return make_metrics(1.0)

RL_METRICS = [make_metrics(2.0, phase=i) for i in range(3)]
PID_METRICS = [make_metrics(1.0, phase=i) for i in range(3)]

REPORTER_CONFIG = {
    "n_episodes_per_phase": 3,
    "curriculum_phases": [0, 1, 2],
    "baseline_comparison": "pid",
    "output_file": "/tmp/eval_report.json",
    "metrics": ["mean_episode_reward", "constraint_violation_rate"],
}

@pytest.fixture
def reporter(tmp_path):
    from sco2rl.deployment.inference.evaluation_reporter import EvaluationReporter
    config = {**REPORTER_CONFIG, "output_file": str(tmp_path / "report.json")}
    mock_eval = _MockEvaluator(RL_METRICS, PID_METRICS)
    return EvaluationReporter(
        env_factory=lambda: None,
        config=config,
        evaluator_factory=lambda env, cfg: mock_eval,
    )

class _ConstantModel:
    pass

def test_evaluate_returns_report(reporter):
    from sco2rl.deployment.inference.evaluation_reporter import EvaluationReport
    rl = _ConstantModel()
    pid_model = _ConstantModel()
    report = reporter.evaluate(rl, pid_model)
    assert isinstance(report, EvaluationReport)

def test_per_phase_count_matches_config(reporter):
    rl, pid_model = _ConstantModel(), _ConstantModel()
    report = reporter.evaluate(rl, pid_model)
    assert len(report.per_phase) == len(REPORTER_CONFIG["curriculum_phases"])

def test_improvement_computed(reporter):
    rl, pid_model = _ConstantModel(), _ConstantModel()
    report = reporter.evaluate(rl, pid_model)
    assert report.overall_improvement_pct == pytest.approx(100.0)

def test_save_and_load_report(reporter, tmp_path):
    from sco2rl.deployment.inference.evaluation_reporter import EvaluationReporter
    rl, pid_model = _ConstantModel(), _ConstantModel()
    report = reporter.evaluate(rl, pid_model)
    path = str(tmp_path / "saved_report.json")
    reporter.save_report(report, path)
    loaded = EvaluationReporter.load_report(path)
    assert loaded.gate5_passed == report.gate5_passed

def test_T_comp_min_is_minimum_across_phases(reporter):
    rl, pid_model = _ConstantModel(), _ConstantModel()
    report = reporter.evaluate(rl, pid_model)
    assert report.T_comp_min_across_all_phases == pytest.approx(32.5)

def test_gate5_passed_false_without_latency(reporter):
    rl, pid_model = _ConstantModel(), _ConstantModel()
    report = reporter.evaluate(rl, pid_model, latency_report=None)
    assert report.gate5_passed is False

def test_gate5_passed_true_with_good_latency(reporter):
    from sco2rl.deployment.inference.latency_benchmark import LatencyReport
    good_latency = LatencyReport(
        mean_ms=0.3, p50_ms=0.28, p95_ms=0.45, p99_ms=0.55,
        min_ms=0.1, max_ms=0.9, n_iterations=1000, batch_size=1, passed_sla=True,
    )
    rl, pid_model = _ConstantModel(), _ConstantModel()
    report = reporter.evaluate(rl, pid_model, latency_report=good_latency)
    assert report.gate5_passed is True

def test_overall_improvement_pct(reporter):
    rl, pid_model = _ConstantModel(), _ConstantModel()
    report = reporter.evaluate(rl, pid_model)
    expected = (2.0 - 1.0) / abs(1.0) * 100.0
    assert report.overall_improvement_pct == pytest.approx(expected)
