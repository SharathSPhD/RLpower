"""CyclePhysicsValidator — validates physics constraints on cycle operating state.

Operates on a dict of measured/simulated values; no FMU dependency.
Raises PhysicsViolation on the first failed constraint, matching fail-fast
semantics for safety-critical checks (RULE-P1, RULE-P5).

Usage:
    validator = CyclePhysicsValidator()
    report = validator.validate({
        "T_compressor_inlet_C": 35.0,
        "Q_heat_source_kW": 10_000.0,
        "W_net_kW": 1_000.0,
        "Q_reject_kW": 9_000.0,
        "surge_margin": 0.10,
    })
    # report["energy_balance"] == {"passed": True, "value": 0.0, "threshold": 0.02}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class PhysicsViolation(Exception):
    """Raised when a physics constraint is violated.

    The message includes the constraint name, actual value, and threshold.
    """


@dataclass(frozen=True)
class CheckResult:
    """Result of a single physics check."""
    passed: bool
    value: float
    threshold: float

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "value": self.value, "threshold": self.threshold}


class CyclePhysicsValidator:
    """Validates physics constraints for the sCO₂ simple recuperated cycle.

    Constraint thresholds per RULES.md:
        RULE-P1: T_compressor_inlet ≥ 32.2°C (1.1°C above CO₂ critical point)
        RULE-P5: surge_margin ≥ 5% for main compressor
        Stage 0 gate: energy balance |Q_in - W_net - Q_reject| / Q_in < 2%
    """

    # Physics thresholds — change only with ADR justification
    COMPRESSOR_INLET_MIN_C: float = 32.2   # RULE-P1
    SURGE_MARGIN_MIN: float = 0.05          # RULE-P5 (5%)
    ENERGY_BALANCE_TOL: float = 0.02        # Stage 0 gate criterion (2%)

    def validate(self, state: dict[str, float]) -> dict[str, dict[str, Any]]:
        """Run all physics checks against the provided operating state.

        Checks are run in order of severity. Raises PhysicsViolation on the
        first failed check (fail-fast for safety).

        Args:
            state: Dict with keys:
                - T_compressor_inlet_C (float): Compressor inlet temperature [°C]
                - Q_heat_source_kW (float): Heat input [kW]
                - W_net_kW (float): Net electrical output [kW]
                - Q_reject_kW (float): Heat rejection [kW]
                - surge_margin (float): Surge margin [fraction, 0–1]

        Returns:
            Dict mapping check name → CheckResult.as_dict() for all checks.

        Raises:
            PhysicsViolation: On first failed constraint, with descriptive message.
        """
        report: dict[str, dict[str, Any]] = {}

        # ── 1. Compressor inlet temperature (RULE-P1) ─────────────────────────
        t_inlet = float(state["T_compressor_inlet_C"])
        t_check = CheckResult(
            passed=t_inlet >= self.COMPRESSOR_INLET_MIN_C,
            value=t_inlet,
            threshold=self.COMPRESSOR_INLET_MIN_C,
        )
        report["compressor_inlet_temp"] = t_check.as_dict()
        if not t_check.passed:
            raise PhysicsViolation(
                f"compressor_inlet_temp violated: "
                f"T_inlet={t_inlet:.2f}°C < {self.COMPRESSOR_INLET_MIN_C}°C minimum. "
                f"CO₂ approaches critical point (31.1°C) — risk of two-phase flow. "
                f"See RULE-P1."
            )

        # ── 2. Surge margin (RULE-P5) ─────────────────────────────────────────
        surge = float(state["surge_margin"])
        surge_check = CheckResult(
            passed=surge >= self.SURGE_MARGIN_MIN,
            value=surge,
            threshold=self.SURGE_MARGIN_MIN,
        )
        report["surge_margin"] = surge_check.as_dict()
        if not surge_check.passed:
            raise PhysicsViolation(
                f"surge_margin violated: "
                f"margin={surge:.3f} ({surge*100:.1f}%) < {self.SURGE_MARGIN_MIN*100:.0f}% minimum. "
                f"Compressor surge risk — immediate action required. "
                f"See RULE-P5."
            )

        # ── 3. Energy balance (Stage 0 gate: ≤2%) ────────────────────────────
        q_in = float(state["Q_heat_source_kW"])
        w_net = float(state["W_net_kW"])
        q_reject = float(state["Q_reject_kW"])

        if q_in == 0.0:
            imbalance = 0.0
        else:
            imbalance = abs(q_in - w_net - q_reject) / q_in

        balance_check = CheckResult(
            passed=imbalance <= self.ENERGY_BALANCE_TOL,
            value=imbalance,
            threshold=self.ENERGY_BALANCE_TOL,
        )
        report["energy_balance"] = balance_check.as_dict()
        if not balance_check.passed:
            raise PhysicsViolation(
                f"energy_balance violated: "
                f"imbalance={imbalance*100:.2f}% > {self.ENERGY_BALANCE_TOL*100:.0f}% tolerance. "
                f"|Q_in({q_in:.0f}) - W_net({w_net:.0f}) - Q_reject({q_reject:.0f})| / Q_in. "
                f"Check FMU model fidelity (RULE-P2 CVODE tolerance 1e-4 for training)."
            )

        return report
