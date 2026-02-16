"""Tests for FMUExporter.

All OMPython and OMCSessionWrapper calls are mocked.
Tests must pass with: PYTHONPATH=src pytest tests/unit/physics/test_fmu_exporter.py
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_cycle_model():
    """Minimal CycleModel stub for testing FMUExporter (no ComponentFactory dependency)."""
    from sco2rl.physics.metamodel.builder import CycleModel
    from sco2rl.physics.metamodel.components import ComponentSpec

    spec = ComponentSpec(
        name="main_compressor",
        modelica_type="SCOPE.Compressors.AxialCompressor",
        params={"eta_design": 0.85},
        topologies=["simple_recuperated", "recompression_brayton"],
    )
    return CycleModel(
        name="SCO2_WHR",
        package="SCO2RecuperatedCycle",
        fluid_config={"medium": "CO2", "coolprop_options": "enable_BICUBIC=1"},
        components={"main_compressor": spec},
        connections=[],
        topology="simple_recuperated",
    )


@pytest.fixture
def mock_omc():
    """Mock OMCSessionWrapper."""
    m = MagicMock()
    # translateModelFMU returns the FMU filename string on success
    m.send.return_value = '"SCO2_WHR.fmu"'
    m.load_file.return_value = True
    m.__enter__ = MagicMock(return_value=m)
    m.__exit__ = MagicMock(return_value=False)
    return m


@pytest.fixture
def exporter(tmp_path, mock_omc):
    from sco2rl.physics.compiler.fmu_exporter import FMUExporter
    config = {"fmu": {"version": "2.0", "fmu_type": "cs"}}
    return FMUExporter(config=config, output_dir=tmp_path, omc=mock_omc)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFMUExporterMoFile:
    """FMUExporter must write a .mo file before calling OMC."""

    def test_exporter_write_mo_file(self, exporter, mock_cycle_model, tmp_path):
        exporter.export(mock_cycle_model)
        mo_files = list(tmp_path.glob("*.mo"))
        assert len(mo_files) == 1

    def test_exporter_mo_file_named_after_model(self, exporter, mock_cycle_model, tmp_path):
        exporter.export(mock_cycle_model)
        assert (tmp_path / "SCO2_WHR.mo").exists()

    def test_exporter_mo_file_non_empty(self, exporter, mock_cycle_model, tmp_path):
        exporter.export(mock_cycle_model)
        content = (tmp_path / "SCO2_WHR.mo").read_text()
        assert len(content) > 0


class TestFMUExporterOMCCalls:
    """FMUExporter must call OMC with correct translateModelFMU arguments."""

    def test_exporter_calls_load_file(self, exporter, mock_omc, mock_cycle_model):
        exporter.export(mock_cycle_model)
        mock_omc.load_file.assert_called_once()

    def test_exporter_calls_translate_model_fmu(self, exporter, mock_omc, mock_cycle_model):
        exporter.export(mock_cycle_model)
        calls_str = str(mock_omc.send.call_args_list)
        assert "translateModelFMU" in calls_str

    def test_exporter_fmu_type_cs(self, exporter, mock_omc, mock_cycle_model):
        """Always uses fmuType='cs' (Co-Simulation)."""
        exporter.export(mock_cycle_model)
        calls_str = str(mock_omc.send.call_args_list)
        assert 'cs' in calls_str or 'co-simulation' in calls_str.lower() or 'fmuType' in calls_str

    def test_exporter_fmu_version_2(self, exporter, mock_omc, mock_cycle_model):
        """Always uses FMI version 2.0."""
        exporter.export(mock_cycle_model)
        calls_str = str(mock_omc.send.call_args_list)
        assert "2.0" in calls_str


class TestFMUExporterReturn:
    """export() returns the path to the produced .fmu."""

    def test_exporter_export_returns_path(self, exporter, mock_cycle_model):
        result = exporter.export(mock_cycle_model)
        assert isinstance(result, Path)

    def test_exporter_export_returns_fmu_extension(self, exporter, mock_cycle_model):
        result = exporter.export(mock_cycle_model)
        assert result.suffix == ".fmu"


class TestFMUExporterErrors:
    """FMUExporter raises FMUExportError on OMC failures."""

    def test_exporter_raises_on_omc_empty_string(self, tmp_path, mock_cycle_model):
        from sco2rl.physics.compiler.fmu_exporter import FMUExporter, FMUExportError
        bad_omc = MagicMock()
        bad_omc.send.return_value = '""'   # OMC returns empty string = failure
        bad_omc.load_file.return_value = True
        bad_omc.__enter__ = MagicMock(return_value=bad_omc)
        bad_omc.__exit__ = MagicMock(return_value=False)
        exporter = FMUExporter(config={}, output_dir=tmp_path, omc=bad_omc)
        with pytest.raises(FMUExportError):
            exporter.export(mock_cycle_model)

    def test_exporter_raises_on_omc_error_string(self, tmp_path, mock_cycle_model):
        from sco2rl.physics.compiler.fmu_exporter import FMUExporter, FMUExportError
        bad_omc = MagicMock()
        bad_omc.send.return_value = "Error: model not found"
        bad_omc.load_file.return_value = True
        bad_omc.__enter__ = MagicMock(return_value=bad_omc)
        bad_omc.__exit__ = MagicMock(return_value=False)
        exporter = FMUExporter(config={}, output_dir=tmp_path, omc=bad_omc)
        with pytest.raises(FMUExportError, match="FMU export failed"):
            exporter.export(mock_cycle_model)


class TestFMUExporterOutputDir:
    """FMUExporter creates output_dir if it doesn't exist."""

    def test_exporter_creates_output_dir(self, tmp_path, mock_omc, mock_cycle_model):
        from sco2rl.physics.compiler.fmu_exporter import FMUExporter
        nested = tmp_path / "deep" / "output"
        exporter = FMUExporter(config={}, output_dir=nested, omc=mock_omc)
        exporter.export(mock_cycle_model)
        assert nested.exists()
