"""Tests for SCO2CycleBuilder fluent API and CycleModel.

TDD: These tests are written BEFORE the implementation.
Tests cover add_component(), connect(), build(), from_config(), and validation.
"""
import pytest
import yaml
from pathlib import Path

from sco2rl.physics.metamodel.components import ComponentFactory, ComponentSpec
from sco2rl.physics.metamodel.builder import (
    BuildError,
    CycleModel,
    SCO2CycleBuilder,
)
from sco2rl.physics.metamodel.connections import ConnectionSpec


# ─── Config fixture ────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "model" / "base_cycle.yaml"


@pytest.fixture
def model_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def simple_spec():
    """A minimal ComponentSpec for testing."""
    return ComponentSpec(
        name="main_compressor",
        modelica_type="SCOPE.Compressors.CentrifugalCompressor",
        params={"eta_design": 0.88},
        topologies=["simple_recuperated", "recompression_brayton"],
    )


@pytest.fixture
def turbine_spec():
    return ComponentSpec(
        name="turbine",
        modelica_type="SCOPE.Turbines.AxialTurbine",
        params={"eta_design": 0.92},
        topologies=["simple_recuperated", "recompression_brayton"],
    )


@pytest.fixture
def recuperator_spec():
    return ComponentSpec(
        name="recuperator",
        modelica_type="ThermoPower.Gas.HE",
        params={"UA_design": 750.0},
        topologies=["simple_recuperated"],
    )


@pytest.fixture
def precooler_spec():
    return ComponentSpec(
        name="precooler",
        modelica_type="ThermoPower.Gas.HE",
        params={"UA_design": 380.0},
        topologies=["simple_recuperated", "recompression_brayton"],
    )


@pytest.fixture
def heat_source_spec():
    return ComponentSpec(
        name="heat_source",
        modelica_type="SCOPE.HeatSources.ExhaustHeatSource",
        params={"T_exhaust_max_c": 1200.0},
        topologies=["simple_recuperated", "recompression_brayton"],
    )


@pytest.fixture
def minimal_simple_builder(simple_spec, turbine_spec, recuperator_spec, precooler_spec, heat_source_spec):
    """Builder with the 5 required components for simple_recuperated."""
    builder = SCO2CycleBuilder()
    builder._topology = "simple_recuperated"
    builder.add_component("main_compressor", simple_spec)
    builder.add_component("turbine", turbine_spec)
    builder.add_component("recuperator", recuperator_spec)
    builder.add_component("precooler", precooler_spec)
    builder.add_component("heat_source", heat_source_spec)
    return builder


# ─── Fluent API ───────────────────────────────────────────────────────────────

class TestFluentAPI:
    def test_builder_add_component_returns_self(self, simple_spec):
        """add_component() returns self for fluent chaining."""
        builder = SCO2CycleBuilder()
        result = builder.add_component("main_compressor", simple_spec)
        assert result is builder

    def test_builder_connect_returns_self(self, simple_spec, turbine_spec):
        """connect() returns self for fluent chaining."""
        builder = SCO2CycleBuilder()
        builder.add_component("main_compressor", simple_spec)
        builder.add_component("turbine", turbine_spec)
        result = builder.connect("main_compressor.outlet", "turbine.inlet")
        assert result is builder

    def test_builder_connect_stores_connection(self, simple_spec, turbine_spec):
        """connect() stores the ConnectionSpec correctly."""
        builder = SCO2CycleBuilder()
        builder.add_component("main_compressor", simple_spec)
        builder.add_component("turbine", turbine_spec)
        builder.connect("main_compressor.outlet", "turbine.inlet")
        assert len(builder._connections) == 1
        conn = builder._connections[0]
        assert conn.from_port == "main_compressor.outlet"
        assert conn.to_port == "turbine.inlet"

    def test_builder_validates_connection_endpoints(self, simple_spec):
        """connect() raises BuildError if component name not in added components."""
        builder = SCO2CycleBuilder()
        builder.add_component("main_compressor", simple_spec)
        with pytest.raises(BuildError, match="not_a_component"):
            builder.connect("not_a_component.outlet", "main_compressor.inlet")

    def test_builder_validates_to_endpoint(self, simple_spec):
        """connect() raises BuildError if to-port component not added."""
        builder = SCO2CycleBuilder()
        builder.add_component("main_compressor", simple_spec)
        with pytest.raises(BuildError, match="missing_component"):
            builder.connect("main_compressor.outlet", "missing_component.inlet")


# ─── build() ─────────────────────────────────────────────────────────────────

class TestBuild:
    def test_build_returns_cycle_model(self, minimal_simple_builder):
        """build() returns a CycleModel instance."""
        cycle = minimal_simple_builder.build()
        assert isinstance(cycle, CycleModel)

    def test_builder_build_simple_recuperated(self, model_config):
        """build() returns CycleModel with 9 components for simple_recuperated."""
        builder = SCO2CycleBuilder.from_config(model_config)
        cycle = builder.build()
        assert len(cycle.components) == 9

    def test_builder_build_recompression(self, model_config):
        """build() returns CycleModel with 12+ components for recompression_brayton."""
        # Override topology
        model_config["topology"]["type"] = "recompression_brayton"
        builder = SCO2CycleBuilder.from_config(model_config)
        cycle = builder.build()
        assert len(cycle.components) >= 12

    def test_builder_missing_required_component_raises(self, simple_spec):
        """build() raises BuildError if mandatory component missing."""
        builder = SCO2CycleBuilder()
        builder._topology = "simple_recuperated"
        # Only add one component — missing turbine, recuperator, precooler, heat_source
        builder.add_component("main_compressor", simple_spec)
        with pytest.raises(BuildError, match="turbine"):
            builder.build()

    def test_cycle_model_has_all_connections(self, model_config):
        """Built CycleModel.connections list is non-empty."""
        builder = SCO2CycleBuilder.from_config(model_config)
        cycle = builder.build()
        assert len(cycle.connections) > 0

    def test_cycle_model_component_count_simple(self, model_config):
        """CycleModel.components dict has 9 components for simple_recuperated."""
        builder = SCO2CycleBuilder.from_config(model_config)
        cycle = builder.build()
        assert len(cycle.components) == 9

    def test_cycle_model_component_count_recompression(self, model_config):
        """CycleModel.components dict has 12 components for recompression_brayton."""
        model_config["topology"]["type"] = "recompression_brayton"
        builder = SCO2CycleBuilder.from_config(model_config)
        cycle = builder.build()
        assert len(cycle.components) == 12

    def test_cycle_model_has_topology(self, model_config):
        """CycleModel carries the topology string."""
        builder = SCO2CycleBuilder.from_config(model_config)
        cycle = builder.build()
        assert cycle.topology == "simple_recuperated"

    def test_cycle_model_has_name(self, model_config):
        """CycleModel carries the cycle name from config."""
        builder = SCO2CycleBuilder.from_config(model_config)
        cycle = builder.build()
        assert cycle.name == "SCO2_WHR"

    def test_cycle_model_has_package(self, model_config):
        """CycleModel carries the package name from config."""
        builder = SCO2CycleBuilder.from_config(model_config)
        cycle = builder.build()
        assert cycle.package == "SCO2RecuperatedCycle"

    def test_cycle_model_has_fluid_config(self, model_config):
        """CycleModel carries the fluid config dict."""
        builder = SCO2CycleBuilder.from_config(model_config)
        cycle = builder.build()
        assert "medium" in cycle.fluid_config
        assert "CoolPropMedium" in cycle.fluid_config["medium"]


# ─── from_config() ────────────────────────────────────────────────────────────

class TestFromConfig:
    def test_builder_from_config_creates_builder(self, model_config):
        """from_config() returns a SCO2CycleBuilder instance."""
        builder = SCO2CycleBuilder.from_config(model_config)
        assert isinstance(builder, SCO2CycleBuilder)

    def test_builder_from_config_reads_topology(self, model_config):
        """from_config() reads topology.type from the config."""
        builder = SCO2CycleBuilder.from_config(model_config)
        assert builder._topology == "simple_recuperated"

    def test_builder_from_config_reads_topology_recompression(self, model_config):
        """from_config() correctly reads recompression_brayton topology."""
        model_config["topology"]["type"] = "recompression_brayton"
        builder = SCO2CycleBuilder.from_config(model_config)
        assert builder._topology == "recompression_brayton"

    def test_builder_from_config_adds_correct_components(self, model_config):
        """from_config() adds only components for the active topology."""
        builder = SCO2CycleBuilder.from_config(model_config)
        # simple_recuperated should have recuperator, not recompressor
        assert "recuperator" in builder._components
        assert "recompressor" not in builder._components

    def test_builder_from_config_adds_connections(self, model_config):
        """from_config() populates connections from flow_paths config."""
        builder = SCO2CycleBuilder.from_config(model_config)
        assert len(builder._connections) > 0
