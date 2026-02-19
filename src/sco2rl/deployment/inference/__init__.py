"""Sub-millisecond plant-edge inference via TensorRT runtime.

Heavy imports (EvaluationReporter, LatencyBenchmark) use TYPE_CHECKING or lazy
imports to avoid pulling stable-baselines3/torch into lightweight module loads.
"""
from sco2rl.deployment.inference.pid_baseline import PIDBaseline, PIDController

# Lazy imports: pulled in only when explicitly requested to avoid pulling
# stable-baselines3 / torch into environments that only need PID baseline.
def __getattr__(name: str):
    if name in ("EvaluationReporter", "EvaluationReport"):
        from sco2rl.deployment.inference.evaluation_reporter import (
            EvaluationReporter, EvaluationReport,
        )
        globals()["EvaluationReporter"] = EvaluationReporter
        globals()["EvaluationReport"] = EvaluationReport
        return globals()[name]
    if name in ("LatencyBenchmark", "LatencyReport"):
        from sco2rl.deployment.inference.latency_benchmark import (
            LatencyBenchmark, LatencyReport,
        )
        globals()["LatencyBenchmark"] = LatencyBenchmark
        globals()["LatencyReport"] = LatencyReport
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LatencyBenchmark", "LatencyReport",
    "PIDBaseline", "PIDController",
    "EvaluationReporter", "EvaluationReport",
]
