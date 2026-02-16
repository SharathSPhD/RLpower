"""Structured JSON logging for the sco2rl project.

All log records are emitted as JSON lines, making them easy to ingest into
monitoring pipelines (Tensorboard, Grafana, etc.) running on DGX Spark.

Tests run inside Docker on DGX Spark (ARM64). See RULE-D1, RULE-D2.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


# Standard LogRecord attributes that are part of every record — NOT user "extra" fields.
# Kept at module level for performance (not recreated on each format() call).
_STANDARD_RECORD_ATTRS: frozenset[str] = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "id", "levelname", "levelno", "lineno", "module",
    "msecs", "message", "msg", "name", "pathname", "process",
    "processName", "relativeCreated", "stack_info", "thread",
    "threadName", "taskName",
})


class _JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        # Base fields always present
        log_obj: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        # Include user "extra" fields (any key not in the standard set)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_"):
                log_obj[key] = value

        # Exception info
        if record.exc_info:
            log_obj["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, default=str)


class StructuredLogger:
    """Thin wrapper around stdlib Logger that emits JSON-formatted records.

    Usage::

        logger = get_logger("sco2rl.training")
        logger.info("Episode complete", extra={"episode": 42, "reward": 9.7})

    The ``extra`` dict is merged into the JSON output alongside the standard
    timestamp / level / name / message fields.
    """

    def __init__(self, name: str, level: int = logging.DEBUG) -> None:
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(_JSONFormatter())
            self._logger.addHandler(handler)
            self._logger.propagate = False
        self._logger.setLevel(level)

    # Convenience pass-through methods ------------------------------------------

    def debug(self, msg: str, **extra: Any) -> None:
        self._logger.debug(msg, extra=extra or None)

    def info(self, msg: str, **extra: Any) -> None:
        self._logger.info(msg, extra=extra or None)

    def warning(self, msg: str, **extra: Any) -> None:
        self._logger.warning(msg, extra=extra or None)

    def error(self, msg: str, **extra: Any) -> None:
        self._logger.error(msg, extra=extra or None)

    def critical(self, msg: str, **extra: Any) -> None:
        self._logger.critical(msg, extra=extra or None)

    def exception(self, msg: str, **extra: Any) -> None:
        self._logger.exception(msg, extra=extra or None)

    @property
    def name(self) -> str:
        return self._logger.name


# Module-level cache: reuse StructuredLogger instances by name
_loggers: dict[str, StructuredLogger] = {}


def get_logger(name: str, level: int = logging.DEBUG) -> StructuredLogger:
    """Return a named StructuredLogger, creating it if it does not yet exist.

    Args:
        name: Logger name, typically the module's ``__name__`` or a dotted path.
        level: Logging level (default DEBUG so nothing is suppressed during
               development; set to INFO/WARNING in production via env var or
               training config).

    Returns:
        A StructuredLogger instance.
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, level=level)
    return _loggers[name]
