"""Frequency-domain analysis for SCO2FMUEnv controllers.

Uses Pseudo-Random Binary Sequence (PRBS) excitation to estimate the
empirical transfer function of the closed-loop system:

    H(f) = S_yu(f) / S_uu(f)

where S_yu is the cross-spectral density of output and PRBS input, and
S_uu is the power spectral density of the PRBS input.

This gives realistic Bode plots and allows computation of gain/phase margins
without requiring an explicit analytical model.

Reference:
    Ljung, "System Identification: Theory for the User", 2nd ed., Sec. 6.3.
    Pintelon & Schoukens, "System Identification: A Frequency Domain Approach",
    2nd ed., Ch. 2.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy import signal as sp_signal

from sco2rl.analysis.metrics import FrequencyResponseResult


# ---------------------------------------------------------------------------
# PRBS generation
# ---------------------------------------------------------------------------


def generate_prbs(
    n_bits: int = 10,
    amplitude: float = 0.05,
    n_periods: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """Generate a Pseudo-Random Binary Sequence (PRBS).

    Parameters
    ----------
    n_bits:
        Sequence length = 2^n_bits - 1 samples per period.
    amplitude:
        Signal amplitude (±amplitude).
    n_periods:
        Number of periods to generate.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (n_periods * (2^n_bits - 1),)
        PRBS signal in {-amplitude, +amplitude}.
    """
    rng = np.random.default_rng(seed)
    period = (2 ** n_bits) - 1
    # Generate one period as ±1 binary
    prbs_period = rng.choice([-1.0, 1.0], size=period)
    prbs = np.tile(prbs_period, n_periods)
    return (prbs * amplitude).astype(np.float64)


# ---------------------------------------------------------------------------
# Empirical transfer function estimation
# ---------------------------------------------------------------------------


def estimate_frequency_response(
    env: Any,
    policy: Any,
    channel_idx: int = 0,
    output_variable: str = "W_net",
    prbs_amplitude: float = 0.05,
    n_bits: int = 9,
    n_periods: int = 4,
    dt: float = 5.0,
    warmup_steps: int = 80,
    phase: int = 0,
    seed: int = 0,
) -> FrequencyResponseResult:
    """Estimate closed-loop frequency response via PRBS excitation.

    Protocol:
    1. Run env to steady state under policy for ``warmup_steps``.
    2. Superimpose PRBS perturbation on action channel ``channel_idx``.
    3. Record input (PRBS) and output (``output_variable``) time series.
    4. Estimate H(f) = S_yu / S_uu via scipy.signal.csd / welch.
    5. Compute stability margins from the resulting Bode data.

    Parameters
    ----------
    env:
        SCO2FMUEnv (not VecNormalized).
    policy:
        Controller with ``predict(obs, deterministic) → (action, None)``.
    channel_idx:
        Action channel to perturb (0=bypass_valve, 1=igv, 2=inventory,
        3=cooling).
    output_variable:
        Name of the output variable to record (must be in env._obs_vars).
    prbs_amplitude:
        PRBS peak amplitude in normalised action units [-1, 1].
    n_bits:
        PRBS sequence length = 2^n_bits - 1 steps per period.
    n_periods:
        Number of PRBS periods for averaging.
    dt:
        Simulation step size in seconds.
    warmup_steps:
        Steps run before PRBS injection to reach operating point.
    phase:
        Curriculum phase for labelling.
    seed:
        Episode seed.

    Returns
    -------
    FrequencyResponseResult with Bode data and stability margins.
    """
    controller_name = getattr(policy, "name", type(policy).__name__)

    prbs = generate_prbs(n_bits=n_bits, amplitude=prbs_amplitude,
                         n_periods=n_periods, seed=seed)
    n_excitation = len(prbs)
    n_total = warmup_steps + n_excitation

    env.set_curriculum_phase(phase, episode_max_steps=n_total)
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()

    # ── Warm-up ───────────────────────────────────────────────────────────────
    for _ in range(warmup_steps):
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    # ── PRBS excitation ───────────────────────────────────────────────────────
    u_recorded: list[float] = []
    y_recorded: list[float] = []

    for k, prbs_val in enumerate(prbs):
        action, _ = policy.predict(obs, deterministic=True)

        # Inject PRBS on selected channel
        action_perturbed = action.copy()
        action_perturbed[channel_idx] = float(
            np.clip(action[channel_idx] + prbs_val, -1.0, 1.0)
        )

        u_recorded.append(float(prbs_val))
        y_val = _extract_variable_arr(obs, env, output_variable)
        y_recorded.append(y_val)

        obs, _, terminated, truncated, _ = env.step(action_perturbed)
        if terminated or truncated:
            break

    u = np.array(u_recorded, dtype=np.float64)
    y = np.array(y_recorded, dtype=np.float64)

    if len(u) < 32:
        # Not enough data for spectral estimation
        return FrequencyResponseResult(
            output_variable=output_variable,
            input_channel=channel_idx,
            controller=controller_name,
            phase=phase,
        )

    # ── Empirical Transfer Function Estimation ─────────────────────────────────
    fs = 1.0 / dt  # Sampling frequency in Hz
    nperseg = min(len(u) // 2, 128)

    # Cross-spectral density S_yu and input PSD S_uu
    freqs, S_yu = sp_signal.csd(u, y, fs=fs, nperseg=nperseg, scaling="density")
    _, S_uu = sp_signal.welch(u, fs=fs, nperseg=nperseg, scaling="density")

    # Avoid division by near-zero
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        H = np.where(np.abs(S_uu) > 1e-20, S_yu / S_uu, 0.0 + 0.0j)

    # Remove DC and Nyquist
    valid = (freqs > 0) & (freqs < fs / 2)
    freqs = freqs[valid]
    H = H[valid]

    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))
    phase_deg = np.degrees(np.angle(H))
    # Unwrap phase for cleaner display
    phase_deg = np.degrees(np.unwrap(np.radians(phase_deg)))

    # ── Stability margins ─────────────────────────────────────────────────────
    gm_db, pm_deg, gc_hz, pc_hz = _compute_margins(freqs, magnitude_db, phase_deg)
    bw_hz = _bandwidth_hz(freqs, magnitude_db)

    return FrequencyResponseResult(
        output_variable=output_variable,
        input_channel=channel_idx,
        controller=controller_name,
        phase=phase,
        frequencies_hz=freqs.tolist(),
        magnitude_db=magnitude_db.tolist(),
        phase_deg=phase_deg.tolist(),
        gain_margin_db=gm_db,
        phase_margin_deg=pm_deg,
        gain_crossover_hz=gc_hz,
        phase_crossover_hz=pc_hz,
        bandwidth_hz=bw_hz,
    )


# ---------------------------------------------------------------------------
# Stability margin helpers
# ---------------------------------------------------------------------------


def _compute_margins(
    freqs: np.ndarray,
    mag_db: np.ndarray,
    phase_deg: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute gain and phase margins from Bode data.

    Returns
    -------
    (gain_margin_db, phase_margin_deg, gain_crossover_hz, phase_crossover_hz)
    """
    # Phase crossover: where phase = -180°
    sign_changes_pc = np.where(np.diff(np.sign(phase_deg + 180.0)))[0]
    if len(sign_changes_pc) > 0:
        k = sign_changes_pc[-1]
        pc_hz = float(np.interp(0.0, [phase_deg[k] + 180.0, phase_deg[k + 1] + 180.0],
                                 [freqs[k], freqs[k + 1]]))
        mag_at_pc = float(np.interp(pc_hz, freqs, mag_db))
        gm_db = -mag_at_pc
    else:
        pc_hz = float(freqs[-1])
        gm_db = 40.0  # Very stable — no phase crossover found

    # Gain crossover: where magnitude = 0 dB
    sign_changes_gc = np.where(np.diff(np.sign(mag_db)))[0]
    if len(sign_changes_gc) > 0:
        k = sign_changes_gc[-1]
        gc_hz = float(np.interp(0.0, [mag_db[k], mag_db[k + 1]],
                                 [freqs[k], freqs[k + 1]]))
        phase_at_gc = float(np.interp(gc_hz, freqs, phase_deg))
        pm_deg = phase_at_gc + 180.0
    else:
        gc_hz = float(freqs[0])
        pm_deg = 90.0  # No gain crossover — very stable

    return float(gm_db), float(pm_deg), float(gc_hz), float(pc_hz)


def _bandwidth_hz(freqs: np.ndarray, mag_db: np.ndarray) -> float:
    """Find the −3 dB bandwidth."""
    dc_mag = float(mag_db[0]) if len(mag_db) > 0 else 0.0
    threshold = dc_mag - 3.0
    crossings = np.where(mag_db < threshold)[0]
    if len(crossings) == 0:
        return float(freqs[-1])
    return float(freqs[crossings[0]])


def _extract_variable_arr(obs: np.ndarray, env: Any, variable: str) -> float:
    """Extract a named variable from the latest observation window."""
    obs_vars = getattr(env, "_obs_vars", [])
    history_steps = getattr(env, "_history_steps", 1)
    n_obs = len(obs_vars)
    offset = max(history_steps - 1, 0) * n_obs

    for i, v in enumerate(obs_vars):
        if v == variable or variable.lower() in v.lower():
            full_idx = offset + i
            if full_idx < len(obs):
                return float(obs[full_idx])
    return 0.0
