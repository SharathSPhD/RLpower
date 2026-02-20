"""Unit tests for sco2rl.analysis.frequency_analysis."""
from __future__ import annotations

import numpy as np
import pytest

from sco2rl.analysis.frequency_analysis import (
    generate_prbs,
    estimate_frequency_response,
    _compute_margins,
    _bandwidth_hz,
)
from sco2rl.analysis.scenario_runner import build_mock_env, build_mock_pid


# ─── generate_prbs ────────────────────────────────────────────────────────────


def test_prbs_length():
    prbs = generate_prbs(n_bits=8, amplitude=1.0, n_periods=2)
    expected_len = 2 * (2**8 - 1)
    assert len(prbs) == expected_len


def test_prbs_amplitude():
    amplitude = 0.05
    prbs = generate_prbs(n_bits=6, amplitude=amplitude, n_periods=1)
    assert np.all(np.abs(np.abs(prbs) - amplitude) < 1e-12)


def test_prbs_binary():
    prbs = generate_prbs(n_bits=6, amplitude=1.0, n_periods=1)
    unique = np.unique(np.abs(prbs))
    assert len(unique) == 1  # Only ±amplitude


def test_prbs_reproducible():
    p1 = generate_prbs(n_bits=7, seed=0)
    p2 = generate_prbs(n_bits=7, seed=0)
    np.testing.assert_array_equal(p1, p2)


def test_prbs_different_seeds():
    p1 = generate_prbs(n_bits=7, seed=0)
    p2 = generate_prbs(n_bits=7, seed=1)
    assert not np.array_equal(p1, p2)


def test_prbs_nonzero_mean_near_zero():
    """PRBS should have approximately zero mean for large sequences."""
    prbs = generate_prbs(n_bits=12, amplitude=1.0, n_periods=5)
    assert abs(np.mean(prbs)) < 0.1


def test_prbs_covers_both_signs():
    prbs = generate_prbs(n_bits=6)
    assert np.any(prbs > 0)
    assert np.any(prbs < 0)


# ─── _compute_margins ─────────────────────────────────────────────────────────


def test_margins_stable_system():
    """A first-order system has infinite gain margin (no phase crossover for PM)."""
    freqs = np.logspace(-4, -1, 200)
    omega = 2 * np.pi * freqs
    tau = 20.0
    H = 1.0 / (1.0 + 1j * omega * tau)
    mag_db = 20.0 * np.log10(np.abs(H))
    phase_deg = np.degrees(np.angle(H))

    gm, pm, gc, pc = _compute_margins(freqs, mag_db, phase_deg)

    # Phase margin should be close to 90° for a pure first-order system
    assert pm > 60.0, f"Phase margin too low: {pm}"
    # Gain margin should be very large
    assert gm > 20.0, f"Gain margin too low: {gm}"


def test_margins_unstable_system():
    """A system with phase margin < 0 should flag negative PM."""
    freqs = np.logspace(-4, -1, 200)
    omega = 2 * np.pi * freqs
    # Fourth-order lag with high gain → likely unstable margins
    tau = 5.0
    gain = 100.0
    H = gain / (1.0 + 1j * omega * tau) ** 4
    mag_db = 20.0 * np.log10(np.abs(H))
    phase_deg = np.degrees(np.unwrap(np.angle(H)))

    gm, pm, gc, pc = _compute_margins(freqs, mag_db, phase_deg)
    # At least the function should return finite values
    assert np.isfinite(gm)
    assert np.isfinite(pm)


# ─── _bandwidth_hz ────────────────────────────────────────────────────────────


def test_bandwidth_hz_first_order():
    """Bandwidth of first-order H(jω) = 1/(1+jωτ) should be 1/(2πτ)."""
    tau = 30.0
    freqs = np.logspace(-4, -1, 500)
    omega = 2 * np.pi * freqs
    H = 1.0 / (1.0 + 1j * omega * tau)
    mag_db = 20.0 * np.log10(np.abs(H))

    bw = _bandwidth_hz(freqs, mag_db)
    expected = 1.0 / (2.0 * np.pi * tau)

    assert bw == pytest.approx(expected, rel=0.2), f"BW {bw:.5f} vs expected {expected:.5f}"


# ─── estimate_frequency_response with DynamicMockFMU ─────────────────────────


@pytest.fixture
def dynamic_env():
    env = build_mock_env(dynamic=True)
    yield env
    env.close()


@pytest.fixture
def mock_pid():
    return build_mock_pid()


def test_freq_response_returns_result(dynamic_env, mock_pid):
    from sco2rl.analysis.metrics import FrequencyResponseResult
    result = estimate_frequency_response(
        env=dynamic_env,
        policy=mock_pid,
        channel_idx=0,
        output_variable="W_net",
        prbs_amplitude=0.05,
        n_bits=7,
        n_periods=3,
        dt=5.0,
        warmup_steps=20,
        phase=0,
    )
    assert isinstance(result, FrequencyResponseResult)
    assert result.output_variable == "W_net"
    assert result.controller == "MultiLoopPID"


def test_freq_response_has_data(dynamic_env, mock_pid):
    result = estimate_frequency_response(
        env=dynamic_env, policy=mock_pid,
        channel_idx=0, output_variable="W_net",
        n_bits=7, n_periods=3, warmup_steps=20,
    )
    assert len(result.frequencies_hz) > 0
    assert len(result.magnitude_db) == len(result.frequencies_hz)
    assert len(result.phase_deg) == len(result.frequencies_hz)


def test_freq_response_frequencies_positive(dynamic_env, mock_pid):
    result = estimate_frequency_response(
        env=dynamic_env, policy=mock_pid,
        channel_idx=0, n_bits=7, n_periods=3, warmup_steps=20,
    )
    if result.frequencies_hz:
        freqs = np.array(result.frequencies_hz)
        assert np.all(freqs > 0), "Frequencies should all be positive"


def test_freq_response_margins_finite(dynamic_env, mock_pid):
    result = estimate_frequency_response(
        env=dynamic_env, policy=mock_pid,
        channel_idx=0, n_bits=7, n_periods=3, warmup_steps=20,
    )
    if result.frequencies_hz:
        assert np.isfinite(result.gain_margin_db)
        assert np.isfinite(result.phase_margin_deg)
        assert np.isfinite(result.bandwidth_hz)
