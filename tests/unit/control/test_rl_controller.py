"""Unit tests for sco2rl.control.rl_controller.RLController."""
from __future__ import annotations

import numpy as np
import pytest

from sco2rl.control.rl_controller import RLController
from sco2rl.control.interfaces import Controller


class _FakePolicy:
    """Minimal mock policy that returns constant actions."""

    def __init__(self, action: np.ndarray):
        self._action = action

    def predict(self, obs, deterministic=True):
        return self._action.copy(), None


class _CountingPolicy:
    """Policy that counts how many times predict() was called."""

    def __init__(self, n_actions: int = 4):
        self.call_count = 0
        self._n = n_actions

    def predict(self, obs, deterministic=True):
        self.call_count += 1
        return np.zeros(self._n, dtype=np.float32), None


# ─── Tests ────────────────────────────────────────────────────────────────────


def test_is_controller_subclass():
    policy = _FakePolicy(np.zeros(4))
    ctrl = RLController(policy)
    assert isinstance(ctrl, Controller)


def test_predict_returns_correct_shape():
    action = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    ctrl = RLController(_FakePolicy(action))
    out, state = ctrl.predict(np.zeros(10))
    assert out.shape == (4,)
    assert state is None


def test_predict_delegates_to_policy():
    action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    ctrl = RLController(_FakePolicy(action))
    out, _ = ctrl.predict(np.zeros(10))
    np.testing.assert_array_almost_equal(out, action)


def test_predict_float32_output():
    ctrl = RLController(_FakePolicy(np.array([1.0, 0.0], dtype=np.float64)))
    out, _ = ctrl.predict(np.zeros(5))
    assert out.dtype == np.float32


def test_reset_is_no_op():
    """reset() should not raise and should not affect subsequent predictions."""
    counting = _CountingPolicy(n_actions=4)
    ctrl = RLController(counting)
    ctrl.reset()  # Should not raise
    ctrl.predict(np.zeros(10))
    assert counting.call_count == 1


def test_name_default():
    ctrl = RLController(_FakePolicy(np.zeros(4)))
    assert ctrl.name == "RL"


def test_name_custom():
    ctrl = RLController(_FakePolicy(np.zeros(4)), controller_name="PPO-5M")
    assert ctrl.name == "PPO-5M"


def test_from_checkpoint_raises_if_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        RLController.from_checkpoint(tmp_path / "nonexistent.zip")


def test_predict_deterministic_forwarded():
    """deterministic flag should be forwarded to the underlying policy."""

    class _CheckDetPolicy:
        def __init__(self):
            self.last_deterministic = None

        def predict(self, obs, deterministic=True):
            self.last_deterministic = deterministic
            return np.zeros(4, dtype=np.float32), None

    inner = _CheckDetPolicy()
    ctrl = RLController(inner)

    ctrl.predict(np.zeros(5), deterministic=False)
    assert inner.last_deterministic is False

    ctrl.predict(np.zeros(5), deterministic=True)
    assert inner.last_deterministic is True
