"""Full PID controller with anti-windup, derivative filter, and output saturation.

Implements the ISA standard PID form with:
- Proportional on error
- Integral with back-calculation anti-windup (prevents windup at output limits)
- Derivative on measurement (avoids derivative kick on setpoint steps)
- First-order derivative filter (configurable time constant Tf)
- Bumpless setpoint changes via measurement-based derivative

Reference:
    Astrom & Wittenmark, "Computer-Controlled Systems", 3rd ed., Sec. 3.5–3.7.
"""
from __future__ import annotations

import numpy as np


class PIDController:
    """Single-loop PID with back-calculation anti-windup and derivative filter.

    Control law (continuous-time equivalent)::

        u(t) = Kp * e(t) + Ki * ∫e dt + Kd * s / (Tf*s + 1) * (-y)

    Discretised with Euler integration (step size ``dt`` seconds).
    Derivative is applied to the *measurement* (not the error) to avoid
    derivative kick when the setpoint changes.

    Anti-windup uses back-calculation: when the output is saturated, a
    correction term ``(u_sat - u_unsat) / Tt`` is fed back into the integrator,
    where ``Tt = 1 / anti_windup_gain``.

    Parameters
    ----------
    kp:
        Proportional gain (dimensionless; maps error in physical units → output
        in physical units before saturation).
    ki:
        Integral gain (1/s).
    kd:
        Derivative gain (s). Set to 0.0 to disable derivative action.
    setpoint:
        Initial setpoint in physical units.
    output_limits:
        ``(lo, hi)`` output saturation bounds.  Defaults to ``(-1.0, 1.0)``
        matching the normalised action space of ``SCO2FMUEnv``.
    anti_windup_gain:
        Back-calculation gain ``1/Tt``.  Higher values give faster windup
        recovery.  Typical range: ``0.05 – 0.5``.
    derivative_filter_tau:
        First-order filter time constant in seconds for derivative term.
        Set to 0.0 for no filtering (pure discrete derivative).
    dt:
        Simulation step size in seconds.  Must match the ``step_size``
        configured in ``SCO2FMUEnv``.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float = 0.0,
        setpoint: float = 0.0,
        output_limits: tuple[float, float] = (-1.0, 1.0),
        anti_windup_gain: float = 0.1,
        derivative_filter_tau: float = 0.0,
        dt: float = 5.0,
    ) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self._kp = float(kp)
        self._ki = float(ki)
        self._kd = float(kd)
        self._setpoint = float(setpoint)
        self._lo, self._hi = float(output_limits[0]), float(output_limits[1])
        self._aw_gain = float(anti_windup_gain)
        self._df_tau = float(derivative_filter_tau)
        self._dt = float(dt)

        # Mutable state (reset on reset())
        self._integral: float = 0.0
        self._prev_measurement: float = float("nan")  # NaN flags first call
        self._d_filtered: float = 0.0  # filtered derivative state

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute(self, measurement: float) -> float:
        """Compute the control output for the given measurement.

        Parameters
        ----------
        measurement:
            Current process variable in physical units.

        Returns
        -------
        float
            Saturated output in ``[output_limits[0], output_limits[1]]``.
        """
        measurement = float(measurement)
        error = self._setpoint - measurement

        # ── Proportional ──────────────────────────────────────────────────────
        p_term = self._kp * error

        # ── Integral ──────────────────────────────────────────────────────────
        self._integral += error * self._dt
        i_term = self._ki * self._integral

        # ── Derivative (on measurement, with optional filter) ─────────────────
        d_term = 0.0
        if self._kd > 0.0 and not np.isnan(self._prev_measurement):
            # Negative sign: derivative on measurement, not error
            dm_dt = (measurement - self._prev_measurement) / self._dt
            if self._df_tau > 0.0:
                # First-order low-pass filter: α = τ / (τ + dt)
                alpha = self._df_tau / (self._df_tau + self._dt)
                self._d_filtered = alpha * self._d_filtered + (1.0 - alpha) * (-dm_dt)
            else:
                self._d_filtered = -dm_dt
            d_term = self._kd * self._d_filtered

        # ── Unsaturated output ────────────────────────────────────────────────
        output_unsat = p_term + i_term + d_term

        # ── Saturate ─────────────────────────────────────────────────────────
        output = float(np.clip(output_unsat, self._lo, self._hi))

        # ── Back-calculation anti-windup ───────────────────────────────────────
        # When output is saturated, reduce integral for next step
        saturation_error = output - output_unsat
        if abs(saturation_error) > 1e-12:
            self._integral += self._aw_gain * saturation_error * self._dt

        self._prev_measurement = measurement
        return output

    def reset(self) -> None:
        """Reset integral accumulator and derivative state."""
        self._integral = 0.0
        self._prev_measurement = float("nan")
        self._d_filtered = 0.0

    def set_setpoint(self, setpoint: float) -> None:
        """Update the setpoint without resetting integral state."""
        self._setpoint = float(setpoint)

    @property
    def setpoint(self) -> float:
        return self._setpoint

    @property
    def integral(self) -> float:
        """Current integral accumulator value (useful for diagnostics)."""
        return self._integral
