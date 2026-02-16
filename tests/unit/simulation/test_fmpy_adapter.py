"""Unit tests for FMPyAdapter.

TDD strategy: fmpy.fmi2.FMU2Slave is patched with unittest.mock.patch.
No real FMU files are loaded (RULE-C1).

Tests verify:
1. FMPyAdapter is a subclass of FMUInterface.
2. do_step() returns False when fmpy returns fmi2Error / fmi2Fatal.
3. get_outputs_as_array() returns float32 ndarray of correct shape.
4. reset() calls freeInstance and re-instantiates (NOT fmi2SetFMUState).
5. set_inputs() raises KeyError for unknown variable names.
6. initialize() builds vr_map from model description.
7. close() calls freeInstance.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

from sco2rl.simulation.fmu.interface import FMUInterface


# ---- Constants ----------------------------------------------------------------

OBS_VARS = ["T_turbine_inlet", "T_compressor_inlet", "P_high", "W_net"]
ACTION_VARS = ["bypass_valve_opening", "cooling_flow_normalized"]
FMU_PATH = "/fake/path/SCO2.fmu"

FMI2_OK = 0
FMI2_ERROR = 3
FMI2_FATAL = 4


# ---- Fixtures -----------------------------------------------------------------

def _make_mock_model_desc(obs_vars, action_vars):
    """Build a minimal mock model description."""
    all_vars = obs_vars + action_vars
    model_vars = []
    for i, name in enumerate(all_vars):
        mv = MagicMock()
        mv.name = name
        mv.valueReference = i
        model_vars.append(mv)

    md = MagicMock()
    md.modelVariables = model_vars
    md.guid = "test-guid-1234"
    md.coSimulation.modelIdentifier = "SCO2Model"
    return md


@pytest.fixture
def mock_fmpy_env():
    """Patch fmpy and FMU2Slave for isolated unit testing."""
    mock_slave = MagicMock()
    mock_slave.component = MagicMock()  # Simulates fmi2Component
    # Default doStep returns fmi2OK (0)
    mock_slave.fmi2DoStep.return_value = FMI2_OK
    # Default getReal returns zeros for all obs vars
    mock_slave.getReal.return_value = [0.0] * len(OBS_VARS)

    mock_fmu2slave_cls = MagicMock(return_value=mock_slave)
    mock_model_desc = _make_mock_model_desc(OBS_VARS, ACTION_VARS)

    with patch("fmpy.extract", return_value="/tmp/fake_extract") as mock_extract, \
         patch("fmpy.read_model_description", return_value=mock_model_desc) as mock_rmd, \
         patch("fmpy.fmi2.FMU2Slave", mock_fmu2slave_cls), \
         patch("fmpy.fmi2.fmi2Error", FMI2_ERROR), \
         patch("fmpy.fmi2.fmi2Fatal", FMI2_FATAL):
        yield {
            "slave": mock_slave,
            "slave_cls": mock_fmu2slave_cls,
            "model_desc": mock_model_desc,
            "extract": mock_extract,
            "read_md": mock_rmd,
        }


@pytest.fixture
def adapter(mock_fmpy_env):
    """FMPyAdapter with fmpy mocked."""
    from sco2rl.simulation.fmu.fmpy_adapter import FMPyAdapter
    a = FMPyAdapter(
        fmu_path=FMU_PATH,
        obs_vars=OBS_VARS,
        action_vars=ACTION_VARS,
        instance_name="test_instance",
    )
    a.initialize(start_time=0.0, stop_time=3600.0, step_size=5.0)
    return a


# ---- Test classes -------------------------------------------------------------

class TestFMPyAdapterContract:
    def test_is_fmu_interface_subclass(self):
        from sco2rl.simulation.fmu.fmpy_adapter import FMPyAdapter
        assert issubclass(FMPyAdapter, FMUInterface)

    def test_adapter_instance_is_fmu_interface(self, adapter):
        assert isinstance(adapter, FMUInterface)

    def test_implements_all_abstract_methods(self):
        from sco2rl.simulation.fmu.fmpy_adapter import FMPyAdapter
        for method_name in (
            "initialize", "set_inputs", "do_step",
            "get_outputs", "get_outputs_as_array", "reset", "close",
        ):
            assert hasattr(FMPyAdapter, method_name), f"Missing: {method_name}"


class TestFMPyAdapterInitialize:
    def test_initialize_calls_extract(self, mock_fmpy_env, adapter):
        mock_fmpy_env["extract"].assert_called_once_with(FMU_PATH)

    def test_initialize_calls_read_model_description(self, mock_fmpy_env, adapter):
        mock_fmpy_env["read_md"].assert_called_once()

    def test_initialize_calls_instantiate(self, mock_fmpy_env, adapter):
        slave = mock_fmpy_env["slave"]
        slave.instantiate.assert_called()

    def test_initialize_calls_enter_exit_initialization_mode(self, mock_fmpy_env, adapter):
        slave = mock_fmpy_env["slave"]
        slave.enterInitializationMode.assert_called()
        slave.exitInitializationMode.assert_called()

    def test_vr_map_built_correctly(self, adapter):
        # OBS_VARS: indices 0-3, ACTION_VARS: indices 4-5
        assert adapter._vr_map["T_turbine_inlet"] == 0
        assert adapter._vr_map["bypass_valve_opening"] == 4


class TestFMPyAdapterDoStep:
    def test_do_step_returns_true_on_fmi2ok(self, mock_fmpy_env, adapter):
        mock_fmpy_env["slave"].fmi2DoStep.return_value = FMI2_OK
        result = adapter.do_step(current_time=0.0, step_size=5.0)
        assert result is True

    def test_do_step_returns_false_on_fmi2_error(self, mock_fmpy_env, adapter):
        mock_fmpy_env["slave"].fmi2DoStep.return_value = FMI2_ERROR
        result = adapter.do_step(current_time=0.0, step_size=5.0)
        assert result is False

    def test_do_step_returns_false_on_fmi2_fatal(self, mock_fmpy_env, adapter):
        mock_fmpy_env["slave"].fmi2DoStep.return_value = FMI2_FATAL
        result = adapter.do_step(current_time=0.0, step_size=5.0)
        assert result is False

    def test_do_step_returns_false_on_exception(self, mock_fmpy_env, adapter):
        mock_fmpy_env["slave"].fmi2DoStep.side_effect = RuntimeError("solver crash")
        result = adapter.do_step(current_time=0.0, step_size=5.0)
        assert result is False


class TestFMPyAdapterOutputs:
    def test_get_outputs_as_array_returns_float32_ndarray(self, mock_fmpy_env, adapter):
        mock_fmpy_env["slave"].getReal.return_value = [750.0, 33.0, 20.0, 10.0]
        arr = adapter.get_outputs_as_array()
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32

    def test_get_outputs_as_array_correct_shape(self, mock_fmpy_env, adapter):
        mock_fmpy_env["slave"].getReal.return_value = [750.0, 33.0, 20.0, 10.0]
        arr = adapter.get_outputs_as_array()
        assert arr.shape == (len(OBS_VARS),)

    def test_get_outputs_returns_dict(self, mock_fmpy_env, adapter):
        mock_fmpy_env["slave"].getReal.return_value = [750.0, 33.0, 20.0, 10.0]
        out = adapter.get_outputs()
        assert isinstance(out, dict)
        for var in OBS_VARS:
            assert var in out

    def test_get_outputs_values_match_getreal(self, mock_fmpy_env, adapter):
        expected = [750.0, 33.0, 20.0, 10.0]
        mock_fmpy_env["slave"].getReal.return_value = expected
        out = adapter.get_outputs()
        for var, val in zip(OBS_VARS, expected):
            assert out[var] == pytest.approx(val)


class TestFMPyAdapterSetInputs:
    def test_set_inputs_calls_setreal(self, mock_fmpy_env, adapter):
        adapter.set_inputs({"bypass_valve_opening": 0.5})
        mock_fmpy_env["slave"].setReal.assert_called()

    def test_set_inputs_raises_keyerror_for_unknown_var(self, mock_fmpy_env, adapter):
        with pytest.raises(KeyError, match="nonexistent_var"):
            adapter.set_inputs({"nonexistent_var": 0.5})

    def test_set_inputs_known_action_var_no_error(self, mock_fmpy_env, adapter):
        # Should not raise
        adapter.set_inputs({"bypass_valve_opening": 0.3, "cooling_flow_normalized": 0.7})


class TestFMPyAdapterReset:
    def test_reset_calls_free_instance(self, mock_fmpy_env, adapter):
        slave = mock_fmpy_env["slave"]
        slave.freeInstance.reset_mock()
        adapter.reset()
        slave.freeInstance.assert_called()

    def test_reset_does_not_call_fmi2setfmustate(self, mock_fmpy_env, adapter):
        """ADR: reset MUST NOT use fmi2SetFMUState (non-deterministic with CoolProp)."""
        slave = mock_fmpy_env["slave"]
        adapter.reset()
        slave.fmi2SetFMUstate.assert_not_called()

    def test_reset_re_instantiates(self, mock_fmpy_env, adapter):
        cls = mock_fmpy_env["slave_cls"]
        initial_call_count = cls.call_count
        adapter.reset()
        assert cls.call_count > initial_call_count


class TestFMPyAdapterClose:
    def test_close_calls_free_instance(self, mock_fmpy_env, adapter):
        slave = mock_fmpy_env["slave"]
        slave.freeInstance.reset_mock()
        adapter.close()
        slave.freeInstance.assert_called()

    def test_close_sets_fmu_to_none(self, mock_fmpy_env, adapter):
        adapter.close()
        assert adapter._fmu is None
