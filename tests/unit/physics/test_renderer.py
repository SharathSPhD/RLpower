"""Tests for MoFileRenderer (Modelica file generation).

TDD: These tests are written BEFORE the implementation.
Tests cover render(), render_to_file(), structural validity, and content checks.
"""
import pytest
import yaml
from pathlib import Path

from sco2rl.physics.metamodel.builder import CycleModel, SCO2CycleBuilder
from sco2rl.physics.metamodel.renderer import MoFileRenderer


# ─── Config fixture ────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "model" / "base_cycle.yaml"


@pytest.fixture
def model_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def simple_cycle(model_config) -> CycleModel:
    """Build a simple_recuperated CycleModel from the real config."""
    builder = SCO2CycleBuilder.from_config(model_config)
    return builder.build()


@pytest.fixture
def recompression_cycle(model_config) -> CycleModel:
    """Build a recompression_brayton CycleModel from the real config."""
    model_config["topology"]["type"] = "recompression_brayton"
    builder = SCO2CycleBuilder.from_config(model_config)
    return builder.build()


@pytest.fixture
def controlled_cycle(model_config) -> CycleModel:
    """Build a simple cycle with external controller interface enabled."""
    model_config["controller"] = {
        "mode": "rl_external",
        "n_commands": 4,
        "n_measurements": 14,
    }
    builder = SCO2CycleBuilder.from_config(model_config)
    return builder.build()


@pytest.fixture
def renderer() -> MoFileRenderer:
    return MoFileRenderer()


# ─── Basic render() ───────────────────────────────────────────────────────────

class TestRenderBasics:
    def test_render_produces_string(self, renderer, simple_cycle):
        """render(cycle_model) returns a non-empty string."""
        result = renderer.render(simple_cycle)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_contains_model_name(self, renderer, simple_cycle):
        """Output contains the cycle package name (e.g., 'SCO2RecuperatedCycle')."""
        result = renderer.render(simple_cycle)
        assert simple_cycle.package in result

    def test_render_valid_modelica_structure_starts(self, renderer, simple_cycle):
        """Output starts with 'within Steps.Cycle;' (SCOPE package declaration)."""
        result = renderer.render(simple_cycle)
        assert result.strip().startswith("within Steps.Cycle;")

    def test_render_valid_modelica_structure_ends(self, renderer, simple_cycle):
        """Output ends with 'end ' + package name + ';'."""
        result = renderer.render(simple_cycle)
        stripped = result.strip()
        expected_end = f"end {simple_cycle.package};"
        assert stripped.endswith(expected_end), (
            f"Expected output to end with {expected_end!r}, got: {stripped[-80:]!r}"
        )

    def test_render_contains_equation_section(self, renderer, simple_cycle):
        """Output contains the 'equation' keyword."""
        result = renderer.render(simple_cycle)
        assert "equation" in result

    def test_render_is_deterministic(self, renderer, simple_cycle):
        """render() is deterministic given the same CycleModel input."""
        result1 = renderer.render(simple_cycle)
        result2 = renderer.render(simple_cycle)
        assert result1 == result2


# ─── Fluid medium & BICUBIC ───────────────────────────────────────────────────

class TestFluidContent:
    def test_render_contains_scope_coolprop_note(self, renderer, simple_cycle):
        """Output contains SCOPE CoolProp backend note (not ExternalMedia)."""
        result = renderer.render(simple_cycle)
        # SCOPE uses Steps.Utilities.CoolProp (libMyProps.so), not ExternalMedia
        assert "Steps.Utilities.CoolProp" in result

    def test_render_contains_bicubic_flag(self, renderer, simple_cycle):
        """Output contains 'enable_BICUBIC=1' (RULE-P3) as a comment."""
        result = renderer.render(simple_cycle)
        assert "enable_BICUBIC=1" in result

    def test_render_contains_bicubic_in_recompression(self, renderer, recompression_cycle):
        """RULE-P3: enable_BICUBIC=1 appears for all topologies."""
        result = renderer.render(recompression_cycle)
        assert "enable_BICUBIC=1" in result


# ─── Component declarations ────────────────────────────────────────────────────

class TestComponentDeclarations:
    def test_render_contains_compressor_type(self, renderer, simple_cycle):
        """Output contains Steps.Components.Pump (SCOPE compressor)."""
        result = renderer.render(simple_cycle)
        assert "Steps.Components.Pump" in result

    def test_render_contains_turbine_type(self, renderer, simple_cycle):
        """Output contains Steps.Components.Turbine (SCOPE turbine)."""
        result = renderer.render(simple_cycle)
        assert "Steps.Components.Turbine" in result

    def test_render_contains_recuperator_type(self, renderer, simple_cycle):
        """Output contains Steps.Components.Recuperator."""
        result = renderer.render(simple_cycle)
        assert "Steps.Components.Recuperator" in result

    def test_render_contains_precooler_type(self, renderer, simple_cycle):
        """Output contains Steps.Components.FanCooler."""
        result = renderer.render(simple_cycle)
        assert "Steps.Components.FanCooler" in result

    def test_render_simple_cycle_no_valve_type(self, renderer, simple_cycle):
        """simple_recuperated output does NOT contain Steps.Components.Valve.

        bypass_valve and inventory_valve are declared in the topology config but
        have no connections in the simple_recuperated flow path.  build() filters
        unconnected components, so no Valve appears in the rendered model.
        """
        result = renderer.render(simple_cycle)
        assert "Steps.Components.Valve" not in result

    def test_render_contains_regulator_type(self, renderer, simple_cycle):
        """Output contains Steps.Components.Regulator (inlet boundary condition)."""
        result = renderer.render(simple_cycle)
        assert "Steps.Components.Regulator" in result

    def test_render_contains_all_component_names(self, renderer, simple_cycle):
        """Output contains each component's instance name."""
        result = renderer.render(simple_cycle)
        for name in simple_cycle.components:
            assert name in result, f"Component name {name!r} not found in rendered output"


# ─── Topology-specific content ─────────────────────────────────────────────────

class TestTopologySpecificContent:
    def test_render_simple_recuperated_has_one_recuperator_instance(self, renderer, simple_cycle):
        """For simple_recuperated topology, 'recuperator' appears as instance name."""
        result = renderer.render(simple_cycle)
        # Should have 'recuperator' as an instance
        assert "recuperator" in result
        # Should NOT have the recompression-only instances
        assert "recuperator_high_temp" not in result
        assert "recuperator_low_temp" not in result
        assert "recompressor" not in result

    def test_render_recompression_has_two_recuperators(self, renderer, recompression_cycle):
        """For recompression topology, two HE instance names appear."""
        result = renderer.render(recompression_cycle)
        assert "recuperator_high_temp" in result
        assert "recuperator_low_temp" in result

    def test_render_recompression_has_split_valve(self, renderer, recompression_cycle):
        """Recompression cycle output contains split_valve instance."""
        result = renderer.render(recompression_cycle)
        assert "split_valve" in result
        assert "Steps.Components.Splitter" in result

    def test_render_simple_no_split_valve(self, renderer, simple_cycle):
        """Simple recuperated cycle output does NOT contain split_valve."""
        result = renderer.render(simple_cycle)
        assert "split_valve" not in result


# ─── connect() equations ─────────────────────────────────────────────────────

class TestConnectEquations:
    def test_render_contains_connect_statements(self, renderer, simple_cycle):
        """Output contains 'connect(' calls in the equation section."""
        result = renderer.render(simple_cycle)
        assert "connect(" in result

    def test_render_connect_uses_component_names(self, renderer, simple_cycle):
        """connect() statements reference real component names."""
        result = renderer.render(simple_cycle)
        # At least one connect statement referencing main_compressor
        assert "main_compressor." in result


class TestControllerInterfaceRendering:
    def test_render_includes_controller_wrappers(self, renderer, controlled_cycle):
        result = renderer.render(controlled_cycle)
        assert "model ControlledRegulator" in result
        assert "model ControlledTurbine" in result
        assert "model ControlledFanCooler" in result

    def test_render_includes_command_and_measurement_ports(self, renderer, controlled_cycle):
        result = renderer.render(controlled_cycle)
        assert "RealInput commands[4]" in result
        assert "RealOutput measurements[14]" in result

    def test_render_connects_commands_to_actuator_wrappers(self, renderer, controlled_cycle):
        result = renderer.render(controlled_cycle)
        assert "connect(commands[1], regulator.T_set);" in result
        assert "connect(commands[3], turbine.p_out_set);" in result


# ─── render_to_file() ─────────────────────────────────────────────────────────

class TestRenderToFile:
    def test_render_to_file_creates_file(self, renderer, simple_cycle, tmp_path):
        """render_to_file() writes a .mo file to disk."""
        out_path = tmp_path / "SCO2RecuperatedCycle.mo"
        returned = renderer.render_to_file(simple_cycle, out_path)
        assert out_path.exists()

    def test_render_to_file_returns_path(self, renderer, simple_cycle, tmp_path):
        """render_to_file() returns the output path."""
        out_path = tmp_path / "SCO2RecuperatedCycle.mo"
        returned = renderer.render_to_file(simple_cycle, out_path)
        assert returned == out_path

    def test_render_to_file_content_matches_render(self, renderer, simple_cycle, tmp_path):
        """Content of written file matches render() output."""
        out_path = tmp_path / "SCO2RecuperatedCycle.mo"
        renderer.render_to_file(simple_cycle, out_path)
        expected = renderer.render(simple_cycle)
        actual = out_path.read_text()
        assert actual == expected

    def test_render_to_file_creates_parent_dirs(self, renderer, simple_cycle, tmp_path):
        """render_to_file() creates parent directories if they don't exist."""
        out_path = tmp_path / "nested" / "dir" / "cycle.mo"
        renderer.render_to_file(simple_cycle, out_path)
        assert out_path.exists()

    def test_render_to_file_mo_extension(self, renderer, simple_cycle, tmp_path):
        """Written file has content consistent with Modelica (.mo) syntax."""
        out_path = tmp_path / "test.mo"
        renderer.render_to_file(simple_cycle, out_path)
        content = out_path.read_text()
        assert content.strip().startswith("within Steps.Cycle;")
        assert "equation" in content
