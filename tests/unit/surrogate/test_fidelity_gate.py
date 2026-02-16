"""Unit tests for FidelityGate -- TDD RED phase."""

import json
import numpy as np
import pytest

from sco2rl.surrogate.fidelity_gate import FidelityGate, FidelityReport


GATE_CONFIG = {
    "max_rmse_normalized": 0.05,
    "min_r2": 0.97,
    "critical_variables": [
        {"name": "T_compressor_inlet", "max_rmse_c": 0.5, "min_r2": 0.98},
        {"name": "surge_margin_main", "max_rmse": 0.01, "min_r2": 0.90},
    ],
    "variable_ranges": {
        "T_compressor_inlet": 15.0,
        "surge_margin_main": 0.60,
        "T_turbine_inlet": 200.0,
        "P_high": 5.0,
        "P_low": 3.0,
    },
}

VAR_NAMES = ["T_compressor_inlet", "T_turbine_inlet", "P_high", "P_low", "surge_margin_main"]
N, T, V = 10, 20, len(VAR_NAMES)


@pytest.fixture
def gate():
    return FidelityGate(config=GATE_CONFIG)


@pytest.fixture
def perfect_data():
    rng = np.random.default_rng(42)
    targets = rng.standard_normal((N, T, V))
    predictions = targets.copy()  # perfect match
    return predictions, targets


@pytest.fixture
def bad_data():
    rng = np.random.default_rng(42)
    targets = rng.standard_normal((N, T, V))
    predictions = rng.standard_normal((N, T, V)) * 10  # large error
    return predictions, targets


# ─── Return type ─────────────────────────────────────────────────────────────

def test_evaluate_returns_fidelity_report(gate, perfect_data):
    preds, targets = perfect_data
    report = gate.evaluate(preds, targets, VAR_NAMES)
    assert isinstance(report, FidelityReport), f"Expected FidelityReport, got {type(report)}"


# ─── Pass/fail logic ─────────────────────────────────────────────────────────

def test_perfect_prediction_passes_gate(gate, perfect_data):
    preds, targets = perfect_data
    report = gate.evaluate(preds, targets, VAR_NAMES)
    assert report.passed, f"Perfect predictions should pass. Report: {report}"


def test_bad_prediction_fails_gate(gate, bad_data):
    preds, targets = bad_data
    report = gate.evaluate(preds, targets, VAR_NAMES)
    assert not report.passed, "Large random errors should fail gate"


# ─── RMSE correctness ────────────────────────────────────────────────────────

def test_rmse_calculation_correct(gate):
    """Known input: RMSE should equal expected value."""
    # 1D case: predictions differ by known constant
    targets = np.zeros((1, 1, 1))
    predictions = np.full((1, 1, 1), 2.0)  # error = 2.0 per element
    # normalized RMSE = 2.0 / variable_ranges["T_compressor_inlet"] = 2.0 / 15.0
    report = gate.evaluate(predictions, targets, ["T_compressor_inlet"])
    expected_rmse = 2.0 / 15.0
    actual_rmse = report.per_variable["T_compressor_inlet"]["rmse"]
    assert abs(actual_rmse - expected_rmse) < 1e-6, (
        f"Expected RMSE {expected_rmse:.6f}, got {actual_rmse:.6f}"
    )


# ─── R2 correctness ──────────────────────────────────────────────────────────

def test_r2_calculation_correct(gate):
    """Perfect prediction -> R2 = 1.0."""
    rng = np.random.default_rng(7)
    targets = rng.standard_normal((5, 10, 1))
    predictions = targets.copy()
    report = gate.evaluate(predictions, targets, ["T_compressor_inlet"])
    r2 = report.per_variable["T_compressor_inlet"]["r2"]
    assert abs(r2 - 1.0) < 1e-6, f"Expected R2=1.0, got {r2}"


# ─── per_variable keys ────────────────────────────────────────────────────────

def test_per_variable_keys_match_variable_names(gate, perfect_data):
    preds, targets = perfect_data
    report = gate.evaluate(preds, targets, VAR_NAMES)
    assert set(report.per_variable.keys()) == set(VAR_NAMES), (
        f"per_variable keys mismatch: {set(report.per_variable.keys())} vs {set(VAR_NAMES)}"
    )


# ─── Critical variable checks ────────────────────────────────────────────────

def test_critical_variable_checked_if_present(gate, perfect_data):
    """If T_compressor_inlet is in variable_names, it must appear in critical_variables."""
    preds, targets = perfect_data
    report = gate.evaluate(preds, targets, VAR_NAMES)
    assert "T_compressor_inlet" in report.critical_variables, (
        "T_compressor_inlet should be in critical_variables"
    )


def test_critical_variable_not_checked_if_absent(gate):
    """If critical variable not in variable_names, skip gracefully (no KeyError)."""
    rng = np.random.default_rng(0)
    # Only include non-critical variables
    names = ["T_turbine_inlet", "P_high"]
    targets = rng.standard_normal((3, 5, len(names)))
    preds = targets.copy()
    report = gate.evaluate(preds, targets, names)  # must not raise
    assert "T_compressor_inlet" not in report.critical_variables


# ─── Save / load roundtrip ────────────────────────────────────────────────────

def test_save_and_load_report_roundtrip(gate, perfect_data, tmp_path):
    preds, targets = perfect_data
    report = gate.evaluate(preds, targets, VAR_NAMES)
    save_path = str(tmp_path / "fidelity_report.json")
    gate.save_report(report, save_path)

    loaded = FidelityGate.load_report(save_path)
    assert loaded.passed == report.passed
    assert abs(loaded.overall_rmse_normalized - report.overall_rmse_normalized) < 1e-9
    assert abs(loaded.overall_r2 - report.overall_r2) < 1e-9
    assert set(loaded.per_variable.keys()) == set(report.per_variable.keys())


# ─── Threshold edge case ─────────────────────────────────────────────────────

def test_passed_false_if_rmse_exceeds_threshold(gate):
    """RMSE slightly over max_rmse_normalized -> passed=False."""
    max_rmse = GATE_CONFIG["max_rmse_normalized"]  # 0.05
    # Use T_turbine_inlet range = 200.0. Want normalized RMSE = 0.06 (> 0.05).
    # normalized RMSE = raw_rmse / 200.0 = 0.06 -> raw_rmse = 12.0
    targets = np.zeros((1, 1, 1))
    predictions = np.full((1, 1, 1), 12.0)
    report = gate.evaluate(predictions, targets, ["T_turbine_inlet"])
    assert not report.passed, (
        f"RMSE {report.overall_rmse_normalized:.4f} should exceed threshold {max_rmse}"
    )
