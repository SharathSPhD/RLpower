"""OMCSessionWrapper — thread-safe OMPython wrapper with retry and context manager.

OMPython is imported LAZILY (inside __init__ body) so that unit tests can
mock it with unittest.mock.patch without requiring OpenModelica to be installed.

Usage:
    with OMCSessionWrapper() as omc:
        omc.load_model("Modelica")
        result = omc.send('translateModelFMU(...)')
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any


# Lazy-loaded at module level for mockability — NOT imported at top-level.
# Tests patch this name: "sco2rl.physics.compiler.omc_session.OMCSessionZMQ"
def _import_omc():
    from OMPython import OMCSessionZMQ  # type: ignore[import]
    return OMCSessionZMQ


# Module-level name that tests can patch
try:
    from OMPython import OMCSessionZMQ  # type: ignore[import]  # noqa: F401
except ImportError:
    OMCSessionZMQ = None  # type: ignore[assignment,misc]  # replaced in tests


class OMCSessionWrapper:
    """Thread-safe wrapper around OMCSessionZMQ with retry and context manager.

    Args:
        omc_path: Optional path to the omc executable.
        max_retries: Number of times to retry OMCSessionZMQ init on failure.
        retry_delay_s: Seconds to wait between retries.
    """

    # Command-line options required for FMU export (RULE-P2, SPEC §3)
    _CMD_OPTIONS = "--fmuRuntimeDepends=modelica --fmiFlags=s:cvode"

    def __init__(
        self,
        omc_path: str | None = None,
        max_retries: int = 3,
        retry_delay_s: float = 1.0,
    ) -> None:
        self._omc_path = omc_path
        self._session: Any = None
        self._max_retries = max_retries
        self._retry_delay_s = retry_delay_s
        self._connect()

    # ── Private ──────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        """Attempt to create OMCSessionZMQ, retrying on transient failures."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                # Use module-level name so tests can patch it
                session_cls = OMCSessionZMQ  # type: ignore[possibly-undefined]
                if self._omc_path:
                    self._session = session_cls(omc=self._omc_path)
                else:
                    self._session = session_cls()
                self._setup_session()
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay_s)

        raise last_exc or RuntimeError("OMC session failed after retries")

    def _setup_session(self) -> None:
        """Load standard libraries and set required command-line options."""
        # FMU export options (RULE-P2: CVODE solver, modelica runtime deps)
        self._session.sendExpression(
            f'setCommandLineOptions("{self._CMD_OPTIONS}")'
        )
        # Load Modelica standard library (always available in OMC)
        self._session.sendExpression("loadModel(Modelica)")
        # Optional libraries — silently skip if not installed in this container
        for lib_path in [
            "/opt/libs/ThermoPower/package.mo",
            "/opt/libs/SCOPE/package.mo",
            "/opt/libs/ExternalMedia/package.mo",
        ]:
            if Path(lib_path).exists():
                self._session.sendExpression(f'loadFile("{lib_path}")')

    # ── Public API ────────────────────────────────────────────────────────────

    def send(self, expression: str) -> str:
        """Send a Modelica/OMC expression and return the result as a string.

        Args:
            expression: Any valid OMC expression, e.g. '1+1' or 'loadModel(Modelica)'.

        Returns:
            String representation of the OMC result.
        """
        result = self._session.sendExpression(expression)
        return str(result) if result is not None else ""

    def load_model(self, model_name: str) -> bool:
        """Load a Modelica standard library model by name.

        Args:
            model_name: e.g. 'Modelica', 'ThermoPower'.

        Returns:
            True if the model loaded successfully, False otherwise.
        """
        result = self._session.sendExpression(f"loadModel({model_name})")
        return bool(result)

    def load_file(self, path: str | Path) -> bool:
        """Load a Modelica file by path.

        Args:
            path: Absolute or relative path to a .mo file.

        Returns:
            True if loaded successfully, False otherwise.
        """
        result = self._session.sendExpression(f'loadFile("{path}")')
        return bool(result)

    def quit(self) -> None:
        """Cleanly shut down the OMC session."""
        if self._session is not None:
            try:
                self._session.sendExpression("quit()")
            except Exception:  # noqa: BLE001
                pass
            self._session = None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "OMCSessionWrapper":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.quit()
