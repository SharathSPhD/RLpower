"""Unit tests for StructuredLogger / get_logger.

Tests run inside Docker on DGX Spark (ARM64). See RULE-D1, RULE-D2.
"""

from __future__ import annotations

import io
import json
import logging

import pytest

from sco2rl.utils.logging import StructuredLogger, get_logger, _loggers


class TestStructuredLogger:
    def test_get_logger_returns_structured_logger(self) -> None:
        """get_logger returns a StructuredLogger instance."""
        logger = get_logger("test.structured_logger.basic")
        assert isinstance(logger, StructuredLogger)

    def test_get_logger_same_name_returns_same_instance(self) -> None:
        """Calling get_logger with the same name returns the cached instance."""
        name = "test.structured_logger.cache"
        a = get_logger(name)
        b = get_logger(name)
        assert a is b

    def test_logger_name(self) -> None:
        """Logger name matches the name passed to get_logger."""
        name = "test.structured_logger.name_check"
        logger = get_logger(name)
        assert logger.name == name

    def test_output_is_valid_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Each log record emitted is valid JSON."""
        logger = get_logger("test.json_output_unique_1")
        logger.info("hello from unit test")
        captured = capsys.readouterr()
        # At least one line should be parseable JSON
        lines = [l for l in captured.out.strip().splitlines() if l]
        assert len(lines) >= 1
        for line in lines:
            obj = json.loads(line)  # raises if not valid JSON
            assert "timestamp" in obj
            assert "level" in obj
            assert "name" in obj
            assert "message" in obj

    def test_output_contains_message(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The message field in JSON output matches the logged message."""
        logger = get_logger("test.json_output_unique_2")
        logger.info("specific test message 42")
        captured = capsys.readouterr()
        lines = [l for l in captured.out.strip().splitlines() if l]
        # Find the line containing our message
        found = any(
            json.loads(l).get("message") == "specific test message 42"
            for l in lines
        )
        assert found, f"Expected message not found in output: {captured.out}"

    def test_warning_level_in_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """logger.warning emits a record with level='WARNING'."""
        logger = get_logger("test.warning_level_unique_3")
        logger.warning("this is a warning")
        captured = capsys.readouterr()
        lines = [l for l in captured.out.strip().splitlines() if l]
        found = any(
            json.loads(l).get("level") == "WARNING"
            for l in lines
        )
        assert found

    def test_extra_fields_in_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Extra keyword arguments appear in the JSON output."""
        logger = get_logger("test.extra_fields_unique_4")
        logger.info("episode done", episode=99, reward=7.5)
        captured = capsys.readouterr()
        lines = [l for l in captured.out.strip().splitlines() if l]
        found = any(
            json.loads(l).get("episode") == 99 and json.loads(l).get("reward") == 7.5
            for l in lines
        )
        assert found, f"Extra fields not found in output: {captured.out}"
