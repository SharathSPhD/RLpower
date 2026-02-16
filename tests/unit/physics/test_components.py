"""Tests for ComponentFactory and component dataclasses.

Tests cover ComponentFactory.create(), topology filtering, and error handling.
All Modelica types use the real SCOPE library namespace (Steps.Components.*).
"""
import pytest

from sco2rl.physics.metamodel.components import (
    ComponentFactory,
    ComponentSpec,
    ComponentTypeError,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

COMPRESSOR_CONFIG = {
    "type": "Steps.Components.Pump",
    "topologies": ["simple_recuperated", "recompression_brayton"],
    "params": {
        "p_outlet": 18000000.0,
        "eta": 0.88,
    },
}

TURBINE_CONFIG = {
    "type": "Steps.Components.Turbine",
    "topologies": ["simple_recuperated", "recompression_brayton"],
    "params": {
        "p_out": 7500000.0,
        "eta": 0.92,
    },
}

RECUPERATOR_CONFIG = {
    "type": "Steps.Components.Recuperator",
    "topologies": ["simple_recuperated"],
    "params": {
        "eta": 0.92,
    },
}

BYPASS_VALVE_CONFIG = {
    "type": "Steps.Components.Valve",
    "topologies": ["simple_recuperated", "recompression_brayton"],
    "params": {
        "p_outlet": 7500000.0,
    },
}

REGULATOR_CONFIG = {
    "type": "Steps.Components.Regulator",
    "topologies": ["simple_recuperated", "recompression_brayton"],
    "params": {
        "p_init": 18000000.0,
        "T_init": 973.15,
        "m_flow_init": 95.0,
    },
}

# Full set of components matching base_cycle.yaml
ALL_COMPONENTS_CONFIG = {
    "regulator": {
        "type": "Steps.Components.Regulator",
        "topologies": ["simple_recuperated", "recompression_brayton"],
        "params": {"p_init": 18000000.0, "T_init": 973.15, "m_flow_init": 95.0},
    },
    "main_compressor": {
        "type": "Steps.Components.Pump",
        "topologies": ["simple_recuperated", "recompression_brayton"],
        "params": {"p_outlet": 18000000.0, "eta": 0.88},
    },
    "turbine": {
        "type": "Steps.Components.Turbine",
        "topologies": ["simple_recuperated", "recompression_brayton"],
        "params": {"p_out": 7500000.0, "eta": 0.92},
    },
    "recuperator": {
        "type": "Steps.Components.Recuperator",
        "topologies": ["simple_recuperated"],
        "params": {"eta": 0.92},
    },
    "recuperator_high_temp": {
        "type": "Steps.Components.Recuperator",
        "topologies": ["recompression_brayton"],
        "params": {"eta": 0.95},
    },
    "recuperator_low_temp": {
        "type": "Steps.Components.Recuperator",
        "topologies": ["recompression_brayton"],
        "params": {"eta": 0.93},
    },
    "precooler": {
        "type": "Steps.Components.FanCooler",
        "topologies": ["simple_recuperated", "recompression_brayton"],
        "params": {"T_output": 305.65},
    },
    "bypass_valve": {
        "type": "Steps.Components.Valve",
        "topologies": ["simple_recuperated", "recompression_brayton"],
        "params": {"p_outlet": 7500000.0},
    },
    "inventory_valve": {
        "type": "Steps.Components.Valve",
        "topologies": ["simple_recuperated", "recompression_brayton"],
        "params": {"p_outlet": 7500000.0},
    },
    "recompressor": {
        "type": "Steps.Components.Pump",
        "topologies": ["recompression_brayton"],
        "params": {"p_outlet": 18000000.0, "eta": 0.87},
    },
    "split_valve": {
        "type": "Steps.Components.Splitter",
        "topologies": ["recompression_brayton"],
        "params": {"split_ratio": 0.35},
    },
}


# ─── ComponentFactory.create() ────────────────────────────────────────────────

class TestComponentFactoryCreate:
    def test_create_compressor_from_yaml(self):
        spec = ComponentFactory.create("main_compressor", COMPRESSOR_CONFIG)
        assert isinstance(spec, ComponentSpec)
        assert spec.name == "main_compressor"
        assert spec.modelica_type == "Steps.Components.Pump"
        assert spec.params["eta"] == pytest.approx(0.88)
        assert spec.params["p_outlet"] == pytest.approx(18000000.0)

    def test_create_turbine_from_yaml(self):
        spec = ComponentFactory.create("turbine", TURBINE_CONFIG)
        assert isinstance(spec, ComponentSpec)
        assert spec.name == "turbine"
        assert spec.modelica_type == "Steps.Components.Turbine"
        assert spec.params["eta"] == pytest.approx(0.92)
        assert spec.params["p_out"] == pytest.approx(7500000.0)

    def test_create_recuperator_from_yaml(self):
        spec = ComponentFactory.create("recuperator", RECUPERATOR_CONFIG)
        assert isinstance(spec, ComponentSpec)
        assert spec.name == "recuperator"
        assert spec.modelica_type == "Steps.Components.Recuperator"
        assert spec.params["eta"] == pytest.approx(0.92)

    def test_create_valve_from_yaml(self):
        spec = ComponentFactory.create("bypass_valve", BYPASS_VALVE_CONFIG)
        assert isinstance(spec, ComponentSpec)
        assert spec.name == "bypass_valve"
        assert spec.modelica_type == "Steps.Components.Valve"
        assert spec.params["p_outlet"] == pytest.approx(7500000.0)

    def test_create_regulator_from_yaml(self):
        spec = ComponentFactory.create("regulator", REGULATOR_CONFIG)
        assert isinstance(spec, ComponentSpec)
        assert spec.name == "regulator"
        assert spec.modelica_type == "Steps.Components.Regulator"
        assert spec.params["p_init"] == pytest.approx(18000000.0)
        assert spec.params["T_init"] == pytest.approx(973.15)
        assert spec.params["m_flow_init"] == pytest.approx(95.0)

    def test_unknown_component_type_raises(self):
        bad_config = {
            "type": "UNKNOWN.Widget.DoesNotExist",
            "topologies": ["simple_recuperated"],
            "params": {},
        }
        with pytest.raises(ComponentTypeError):
            ComponentFactory.create("bad_widget", bad_config)

    def test_component_modelica_type_string(self):
        """Each spec's modelica_type property must match the YAML 'type' field."""
        spec = ComponentFactory.create("main_compressor", COMPRESSOR_CONFIG)
        assert spec.modelica_type == COMPRESSOR_CONFIG["type"]

    def test_component_params_preserved(self):
        """All params from YAML are accessible on the spec object."""
        spec = ComponentFactory.create("main_compressor", COMPRESSOR_CONFIG)
        for key, value in COMPRESSOR_CONFIG["params"].items():
            assert key in spec.params
            assert spec.params[key] == pytest.approx(value)

    def test_component_topologies_preserved(self):
        """Topologies list from YAML is stored on the spec."""
        spec = ComponentFactory.create("recuperator", RECUPERATOR_CONFIG)
        assert spec.topologies == ["simple_recuperated"]

    def test_compressor_topologies_both(self):
        """Components in both topologies have both listed."""
        spec = ComponentFactory.create("main_compressor", COMPRESSOR_CONFIG)
        assert "simple_recuperated" in spec.topologies
        assert "recompression_brayton" in spec.topologies


# ─── Topology Filtering ───────────────────────────────────────────────────────

class TestTopologyFiltering:
    @pytest.fixture
    def all_specs(self):
        return {
            name: ComponentFactory.create(name, cfg)
            for name, cfg in ALL_COMPONENTS_CONFIG.items()
        }

    def test_component_for_topology_filter_simple(self, all_specs):
        """simple_recuperated returns correct subset."""
        filtered = ComponentFactory.get_components_for_topology(all_specs, "simple_recuperated")
        expected_included = {
            "regulator", "main_compressor", "turbine", "recuperator",
            "precooler", "bypass_valve", "inventory_valve",
        }
        for name in expected_included:
            assert name in filtered, f"Expected {name!r} in simple_recuperated components"

    def test_component_for_topology_excludes_recompression_only(self, all_specs):
        """simple_recuperated excludes recompression-only components."""
        filtered = ComponentFactory.get_components_for_topology(all_specs, "simple_recuperated")
        excluded = {"recuperator_high_temp", "recuperator_low_temp", "recompressor", "split_valve"}
        for name in excluded:
            assert name not in filtered, f"Expected {name!r} to be excluded from simple_recuperated"

    def test_component_for_recompression_filter(self, all_specs):
        """recompression_brayton includes recompressor and split_valve."""
        filtered = ComponentFactory.get_components_for_topology(all_specs, "recompression_brayton")
        assert "recompressor" in filtered
        assert "split_valve" in filtered
        assert "recuperator_high_temp" in filtered
        assert "recuperator_low_temp" in filtered

    def test_component_for_recompression_excludes_simple_only(self, all_specs):
        """recompression_brayton excludes the single recuperator."""
        filtered = ComponentFactory.get_components_for_topology(all_specs, "recompression_brayton")
        assert "recuperator" not in filtered

    def test_simple_recuperated_count(self, all_specs):
        """simple_recuperated topology yields exactly 7 components."""
        filtered = ComponentFactory.get_components_for_topology(all_specs, "simple_recuperated")
        # regulator, main_compressor, turbine, recuperator, precooler, bypass_valve, inventory_valve
        assert len(filtered) == 7

    def test_recompression_brayton_count(self, all_specs):
        """recompression_brayton topology yields exactly 10 components."""
        filtered = ComponentFactory.get_components_for_topology(all_specs, "recompression_brayton")
        # regulator, main_compressor, turbine, recuperator_ht, recuperator_lt, precooler,
        # bypass_valve, inventory_valve, recompressor, split_valve
        assert len(filtered) == 10
