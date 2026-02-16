"""Latency benchmark -- full implementation in stage5-export branch.
Stub here for EvaluationReporter import compatibility."""
from dataclasses import dataclass


@dataclass
class LatencyReport:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    n_iterations: int
    batch_size: int
    passed_sla: bool
