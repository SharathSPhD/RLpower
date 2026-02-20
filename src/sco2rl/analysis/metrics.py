"""Control metrics dataclasses for the sCO₂ RL control analysis framework.

All results are serialisable to JSON via ``dataclasses.asdict()`` so they can
be persisted to ``data/`` for offline notebook consumption.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StepResponseResult:
    """Transient response metrics for a single step-input experiment.

    All time values are in seconds (relative to the step onset).
    Energy-integral metrics (IAE, ISE, ITAE) are computed over the full
    post-step recording window.
    """

    # ── Identification ────────────────────────────────────────────────────────
    variable: str           # Output variable tracked (e.g. "W_net")
    controller: str         # "PID", "MultiLoopPID", "RL", etc.
    phase: int              # Curriculum phase (0–6)
    scenario: str           # e.g. "step_load_+20pct"
    seed: int = 0

    # ── Time series ───────────────────────────────────────────────────────────
    time_s: list[float] = field(default_factory=list)
    setpoint: list[float] = field(default_factory=list)
    response: list[float] = field(default_factory=list)

    # ── Step parameters ───────────────────────────────────────────────────────
    step_onset_s: float = 0.0    # Time at which the step was applied
    initial_value: float = 0.0   # Output value just before step
    final_value: float = 0.0     # Asymptotic (mean of last 10 %) value
    step_magnitude: float = 0.0  # Requested step size (signed)

    # ── Transient response metrics ────────────────────────────────────────────
    overshoot_pct: float = 0.0     # (max - final) / |step| * 100
    undershoot_pct: float = 0.0    # (min - final) / |step| * 100  (if negative)
    settling_time_s: float = 0.0   # First time response stays within ±2 % of final
    rise_time_s: float = 0.0       # 10 %→90 % of step magnitude
    peak_time_s: float = 0.0       # Time from onset to first peak
    steady_state_error: float = 0.0  # Signed offset from setpoint at end

    # ── Error integral metrics ────────────────────────────────────────────────
    iae: float = 0.0    # ∫|e(t)|  dt  (Integral Absolute Error)
    ise: float = 0.0    # ∫ e(t)²  dt  (Integral Squared Error)
    itae: float = 0.0   # ∫ t|e(t)| dt  (Integral Time-Absolute Error)


@dataclass
class FrequencyResponseResult:
    """Frequency-domain characterisation for a single controller–channel pair.

    Computed via PRBS excitation + empirical transfer function estimation
    (scipy.signal.csd / scipy.signal.welch).
    """

    # ── Identification ────────────────────────────────────────────────────────
    output_variable: str     # Observed output (e.g. "W_net")
    input_channel: int       # Action channel index perturbed (0–3)
    controller: str
    phase: int

    # ── Frequency response data ───────────────────────────────────────────────
    frequencies_hz: list[float] = field(default_factory=list)
    magnitude_db: list[float] = field(default_factory=list)
    phase_deg: list[float] = field(default_factory=list)

    # ── Stability margins ─────────────────────────────────────────────────────
    gain_margin_db: float = 0.0        # dB at phase crossover (target > 6 dB)
    phase_margin_deg: float = 0.0      # degrees at gain crossover (target > 45°)
    gain_crossover_hz: float = 0.0     # Frequency where |H| = 0 dB
    phase_crossover_hz: float = 0.0    # Frequency where ∠H = −180°
    bandwidth_hz: float = 0.0          # −3 dB closed-loop bandwidth


@dataclass
class ControlMetricsSummary:
    """All metrics for one (phase, scenario) combination.

    Aggregates step response and frequency response for both PID and RL
    so notebooks can display side-by-side comparisons with a single object.
    """

    phase: int
    scenario: str

    # Step response (None if not computed for this combination)
    pid_step: StepResponseResult | None = None
    rl_step: StepResponseResult | None = None

    # Frequency response (None if not computed)
    pid_freq: FrequencyResponseResult | None = None
    rl_freq: FrequencyResponseResult | None = None

    def improvement_settling(self) -> float | None:
        """RL settling time improvement vs PID (negative = RL faster)."""
        if self.pid_step and self.rl_step and self.pid_step.settling_time_s > 0:
            return (
                (self.rl_step.settling_time_s - self.pid_step.settling_time_s)
                / self.pid_step.settling_time_s
                * 100.0
            )
        return None

    def improvement_iae(self) -> float | None:
        """RL IAE improvement vs PID (negative = RL lower error integral)."""
        if self.pid_step and self.rl_step and self.pid_step.iae > 0:
            return (
                (self.rl_step.iae - self.pid_step.iae) / self.pid_step.iae * 100.0
            )
        return None
