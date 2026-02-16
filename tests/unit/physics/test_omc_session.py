"""Tests for OMCSessionWrapper.

All OMPython calls are mocked — OpenModelica is NOT required to run these tests.
Tests must pass with: PYTHONPATH=src pytest tests/unit/physics/test_omc_session.py
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_omc_mock(send_return="2"):
    """Return a mock OMCSessionZMQ instance."""
    m = MagicMock()
    m.sendExpression.return_value = send_return
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOMCSessionWrapperContextManager:
    """OMCSessionWrapper must implement __enter__ / __exit__."""

    def test_wrapper_is_context_manager(self):
        with patch("sco2rl.physics.compiler.omc_session.OMCSessionZMQ", return_value=_make_omc_mock()):
            from sco2rl.physics.compiler.omc_session import OMCSessionWrapper
            wrapper = OMCSessionWrapper()
            result = wrapper.__enter__()
            assert result is wrapper

    def test_wrapper_exit_calls_quit(self):
        mock_omc = _make_omc_mock()
        with patch("sco2rl.physics.compiler.omc_session.OMCSessionZMQ", return_value=mock_omc):
            from sco2rl.physics.compiler.omc_session import OMCSessionWrapper
            wrapper = OMCSessionWrapper()
            wrapper.__enter__()
            wrapper.__exit__(None, None, None)
        mock_omc.sendExpression.assert_any_call("quit()")

    def test_wrapper_close_on_exception(self):
        """__exit__ calls quit() even when an exception is passed."""
        mock_omc = _make_omc_mock()
        with patch("sco2rl.physics.compiler.omc_session.OMCSessionZMQ", return_value=mock_omc):
            from sco2rl.physics.compiler.omc_session import OMCSessionWrapper
            wrapper = OMCSessionWrapper()
            wrapper.__enter__()
            wrapper.__exit__(RuntimeError, RuntimeError("boom"), None)
        mock_omc.sendExpression.assert_any_call("quit()")

    def test_wrapper_context_manager_protocol(self):
        """Using 'with' statement calls __enter__ and __exit__."""
        mock_omc = _make_omc_mock()
        with patch("sco2rl.physics.compiler.omc_session.OMCSessionZMQ", return_value=mock_omc):
            from sco2rl.physics.compiler.omc_session import OMCSessionWrapper
            with OMCSessionWrapper() as w:
                assert w is not None


class TestOMCSessionWrapperSend:
    """send() and load_model() surface OMC results correctly."""

    def test_send_expression_returns_string(self):
        mock_omc = _make_omc_mock(send_return="2")
        with patch("sco2rl.physics.compiler.omc_session.OMCSessionZMQ", return_value=mock_omc):
            from sco2rl.physics.compiler.omc_session import OMCSessionWrapper
            wrapper = OMCSessionWrapper()
            result = wrapper.send("1+1")
        assert result == "2"

    def test_load_model_returns_true_on_success(self):
        mock_omc = _make_omc_mock(send_return=True)
        with patch("sco2rl.physics.compiler.omc_session.OMCSessionZMQ", return_value=mock_omc):
            from sco2rl.physics.compiler.omc_session import OMCSessionWrapper
            wrapper = OMCSessionWrapper()
            result = wrapper.load_model("Modelica")
        assert result is True

    def test_load_model_returns_false_on_failure(self):
        mock_omc = _make_omc_mock(send_return=False)
        with patch("sco2rl.physics.compiler.omc_session.OMCSessionZMQ", return_value=mock_omc):
            from sco2rl.physics.compiler.omc_session import OMCSessionWrapper
            wrapper = OMCSessionWrapper()
            result = wrapper.load_model("NonExistentLib")
        assert result is False

    def test_load_file_passes_path_to_omc(self, tmp_path):
        mo_file = tmp_path / "test.mo"
        mo_file.write_text("model Test end Test;")
        mock_omc = _make_omc_mock(send_return=True)
        with patch("sco2rl.physics.compiler.omc_session.OMCSessionZMQ", return_value=mock_omc):
            from sco2rl.physics.compiler.omc_session import OMCSessionWrapper
            wrapper = OMCSessionWrapper()
            wrapper.load_file(mo_file)
        # Should have called sendExpression with loadFile(...)
        calls_str = str(mock_omc.sendExpression.call_args_list)
        assert "loadFile" in calls_str


class TestOMCSessionWrapperSetup:
    """_setup_session() must set command-line options and load libraries."""

    def test_sets_fmu_runtime_depends_option(self):
        mock_omc = _make_omc_mock(send_return=True)
        with patch("sco2rl.physics.compiler.omc_session.OMCSessionZMQ", return_value=mock_omc):
            from sco2rl.physics.compiler.omc_session import OMCSessionWrapper
            OMCSessionWrapper()
        calls_str = str(mock_omc.sendExpression.call_args_list)
        assert "--fmuRuntimeDepends=modelica" in calls_str

    def test_sets_cvode_flag(self):
        mock_omc = _make_omc_mock(send_return=True)
        with patch("sco2rl.physics.compiler.omc_session.OMCSessionZMQ", return_value=mock_omc):
            from sco2rl.physics.compiler.omc_session import OMCSessionWrapper
            OMCSessionWrapper()
        calls_str = str(mock_omc.sendExpression.call_args_list)
        assert "--fmiFlags=s:cvode" in calls_str


class TestOMCSessionWrapperRetry:
    """Wrapper retries OMCSessionZMQ init up to max_retries times."""

    def test_retry_on_init_failure(self):
        """Raises after max_retries failed attempts."""
        with patch(
            "sco2rl.physics.compiler.omc_session.OMCSessionZMQ",
            side_effect=RuntimeError("ZMQ connection failed"),
        ):
            from sco2rl.physics.compiler.omc_session import OMCSessionWrapper
            with pytest.raises(RuntimeError, match="ZMQ connection failed|OMC session failed"):
                OMCSessionWrapper(max_retries=2)

    def test_succeeds_after_one_retry(self):
        """Succeeds if second attempt works."""
        mock_omc = _make_omc_mock()
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient failure")
            return mock_omc

        with patch("sco2rl.physics.compiler.omc_session.OMCSessionZMQ", side_effect=side_effect):
            from sco2rl.physics.compiler.omc_session import OMCSessionWrapper
            wrapper = OMCSessionWrapper(max_retries=3)
        assert call_count == 2
        assert wrapper is not None
