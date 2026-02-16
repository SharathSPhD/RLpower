"""Tests for CyclePhysicsValidator.

No FMU, no OMPython — pure Python physics constraint checks.
Tests must pass with: PYTHONPATH=src pytest tests/unit/physics/test_physics_validator.py
"""
from __future__ import annotations

import pytest

from sco2rl.physics.compiler.physics_validator import (
    CyclePhysicsValidator,
    PhysicsViolation,
)

# ---------------------------------------------------------------------------
# Shared fixture: a valid operating state
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_state():
    """Physics state that passes all constraints."""
    return {
        "T_compressor_inlet_C": 35.0,    # > 32.2°C (RULE-P1)
        "Q_heat_source_kW": 10_000.0,    # 10 MW
        "W_net_kW": 1_000.0,             # 1 MW net output
        "Q_reject_kW": 9_000.0,          # 9 MW rejected → balance: |10000-1000-9000|/10000 = 0%
        "surge_margin": 0.10,             # 10% > 5% (RULE-P5)
    }


# ---------------------------------------------------------------------------
# Compressor inlet temperature (RULE-P1)
# ---------------------------------------------------------------------------

class TestCompressorInletTemp:
    def test_pass_at_minimum_valid(self, valid_state):
        """Exactly at 32.2°C should pass."""
        valid_state["T_compressor_inlet_C"] = 32.2
        validator = CyclePhysicsValidator()
        report = validator.validate(valid_state)
        assert report["compressor_inlet_temp"]["passed"] is True

    def test_pass_above_minimum(self, valid_state):
        valid_state["T_compressor_inlet_C"] = 40.0
        validator = CyclePhysicsValidator()
        report = validator.validate(valid_state)
        assert report["compressor_inlet_temp"]["passed"] is True

    def test_fail_below_minimum(self, valid_state):
        valid_state["T_compressor_inlet_C"] = 30.0   # < 32.2°C
        validator = CyclePhysicsValidator()
        with pytest.raises(PhysicsViolation, match="compressor_inlet_temp|32.2"):
            validator.validate(valid_state)

    def test_fail_at_critical_point(self, valid_state):
        """31.1°C = CO₂ critical point → must reject (< 32.2°C)."""
        valid_state["T_compressor_inlet_C"] = 31.1
        validator = CyclePhysicsValidator()
        with pytest.raises(PhysicsViolation):
            validator.validate(valid_state)


# ---------------------------------------------------------------------------
# Surge margin (RULE-P5)
# ---------------------------------------------------------------------------

class TestSurgeMargin:
    def test_pass_at_minimum_valid(self, valid_state):
        """Exactly 5% should pass."""
        valid_state["surge_margin"] = 0.05
        validator = CyclePhysicsValidator()
        report = validator.validate(valid_state)
        assert report["surge_margin"]["passed"] is True

    def test_pass_above_minimum(self, valid_state):
        valid_state["surge_margin"] = 0.20
        validator = CyclePhysicsValidator()
        report = validator.validate(valid_state)
        assert report["surge_margin"]["passed"] is True

    def test_fail_below_minimum(self, valid_state):
        valid_state["surge_margin"] = 0.03   # < 5%
        validator = CyclePhysicsValidator()
        with pytest.raises(PhysicsViolation, match="surge_margin|0.05"):
            validator.validate(valid_state)

    def test_fail_at_zero(self, valid_state):
        valid_state["surge_margin"] = 0.0
        validator = CyclePhysicsValidator()
        with pytest.raises(PhysicsViolation):
            validator.validate(valid_state)


# ---------------------------------------------------------------------------
# Energy balance (Stage 0 gate: ≤2%)
# ---------------------------------------------------------------------------

class TestEnergyBalance:
    def test_pass_perfect_balance(self, valid_state):
        """Exact balance: 10000 = 1000 + 9000 → 0% error."""
        validator = CyclePhysicsValidator()
        report = validator.validate(valid_state)
        assert report["energy_balance"]["passed"] is True

    def test_pass_within_tolerance(self, valid_state):
        """1% imbalance → under 2% tolerance."""
        valid_state["Q_reject_kW"] = 8_900.0  # imbalance = |10000-1000-8900|/10000 = 1%
        validator = CyclePhysicsValidator()
        report = validator.validate(valid_state)
        assert report["energy_balance"]["passed"] is True

    def test_fail_above_tolerance(self, valid_state):
        """5% imbalance → over 2% tolerance."""
        valid_state["Q_reject_kW"] = 8_500.0  # imbalance = |10000-1000-8500|/10000 = 5%
        validator = CyclePhysicsValidator()
        with pytest.raises(PhysicsViolation, match="energy_balance|0.02"):
            validator.validate(valid_state)

    def test_fail_at_exactly_2_percent_plus_epsilon(self, valid_state):
        """Exactly 2.01% → should fail (> 2%)."""
        valid_state["Q_reject_kW"] = 10_000.0 - 1_000.0 - 10_000.0 * 0.0201  # Q_in - W_net - 2.01%*Q_in
        validator = CyclePhysicsValidator()
        with pytest.raises(PhysicsViolation):
            validator.validate(valid_state)


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------

class TestValidatorReport:
    def test_returns_dict(self, valid_state):
        validator = CyclePhysicsValidator()
        report = validator.validate(valid_state)
        assert isinstance(report, dict)

    def test_report_has_all_check_keys(self, valid_state):
        validator = CyclePhysicsValidator()
        report = validator.validate(valid_state)
        assert "compressor_inlet_temp" in report
        assert "surge_margin" in report
        assert "energy_balance" in report

    def test_report_entry_has_passed_field(self, valid_state):
        validator = CyclePhysicsValidator()
        report = validator.validate(valid_state)
        for key, entry in report.items():
            assert "passed" in entry, f"Entry {key!r} missing 'passed' field"

    def test_report_entry_has_value_field(self, valid_state):
        validator = CyclePhysicsValidator()
        report = validator.validate(valid_state)
        for key, entry in report.items():
            assert "value" in entry, f"Entry {key!r} missing 'value' field"

    def test_report_entry_has_threshold_field(self, valid_state):
        validator = CyclePhysicsValidator()
        report = validator.validate(valid_state)
        for key, entry in report.items():
            assert "threshold" in entry, f"Entry {key!r} missing 'threshold' field"

    def test_all_pass_no_exception(self, valid_state):
        """No exception raised when all checks pass."""
        validator = CyclePhysicsValidator()
        report = validator.validate(valid_state)
        assert all(entry["passed"] for entry in report.values())
