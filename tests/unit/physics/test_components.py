"""Tests for ComponentFactory and component dataclasses.

TDD: These tests are written BEFORE the implementation.
Tests cover ComponentFactory.create(), topology filtering, and error handling.
"""
import pytest

from sco2rl.physics.metamodel.components import (
    ComponentFactory,
    ComponentSpec,
    ComponentTypeError,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

COMPRESSOR_CONFIG = {
    "type": "SCOPE.Compressors.CentrifugalCompressor",
    "topologies": ["simple_recuperated", "recompression_brayton"],
    "params": {
        "eta_design": 0.88,
        "N_design_rpm": 3600,
        "beta_ratio_design": 2.4,
        "inlet_volume_flow_m3_s": 0.70,
    },
}

TURBINE_CONFIG = {
    "type": "SCOPE.Turbines.AxialTurbine",
    "topologies": ["simple_recuperated", "recompression_brayton"],
    "params": {
        "eta_design": 0.92,
        "N_design_rpm": 3600,
        "expansion_ratio_design": 2.4,
    },
}

RECUPERATOR_CONFIG = {
    "type": "ThermoPower.Gas.HE",
    "topologies": ["simple_recuperated"],
    "params": {
        "UA_design": 750.0,
        "effectiveness_design": 0.92,
        "pressure_drop_hot_kpa": 45.0,
        "pressure_drop_cold_kpa": 70.0,
    },
}

BYPASS_VALVE_CONFIG = {
    "type": "ThermoPower.Water.ValveLin",
    "topologies": ["simple_recuperated", "recompression_brayton"],
    "params": {
        "Cv_design": 100.0,
        "opening_min": 0.0,
        "opening_max": 1.0,
    },
}

# Full set of components matching base_cycle.yaml
ALL_COMPONENTS_CONFIG = {
    "main_compressor": {
        "type": "SCOPE.Compressors.CentrifugalCompressor",
        "topologies": ["simple_recuperated", "recompression_brayton"],
        "params": {"eta_design": 0.88, "N_design_rpm": 3600},
    },
    "turbine": {
        "type": "SCOPE.Turbines.AxialTurbine",
        "topologies": ["simple_recuperated", "recompression_brayton"],
        "params": {"eta_design": 0.92, "N_design_rpm": 3600},
    },
    "recuperator": {
        "type": "ThermoPower.Gas.HE",
        "topologies": ["simple_recuperated"],
        "params": {"UA_design": 750.0},
    },
    "recuperator_high_temp": {
        "type": "ThermoPower.Gas.HE",
        "topologies": ["recompression_brayton"],
        "params": {"UA_design": 850.0},
    },
    "recuperator_low_temp": {
        "type": "ThermoPower.Gas.HE",
        "topologies": ["recompression_brayton"],
        "params": {"UA_design": 650.0},
    },
    "precooler": {
        "type": "ThermoPower.Gas.HE",
        "topologies": ["simple_recuperated", "recompression_brayton"],
        "params": {"UA_design": 380.0},
    },
    "heat_source": {
        "type": "SCOPE.HeatSources.ExhaustHeatSource",
        "topologies": ["simple_recuperated", "recompression_brayton"],
        "params": {"T_exhaust_min_c": 200.0},
    },
    "bypass_valve": {
        "type": "ThermoPower.Water.ValveLin",
        "topologies": ["simple_recuperated", "recompression_brayton"],
        "params": {"Cv_design": 100.0},
    },
    "igv": {
        "type": "SCOPE.Actuators.InletGuideVane",
        "topologies": ["simple_recuperated", "recompression_brayton"],
        "params": {"angle_min_deg": -30.0},
    },
    "inventory_valve": {
        "type": "ThermoPower.Water.ValveLin",
        "topologies": ["simple_recuperated", "recompression_brayton"],
        "params": {"Cv_design": 40.0},
    },
    "cooling_valve": {
        "type": "ThermoPower.Water.ValveLin",
        "topologies": ["simple_recuperated", "recompression_brayton"],
        "params": {"Cv_design": 85.0},
    },
    "recompressor": {
        "type": "SCOPE.Compressors.CentrifugalCompressor",
        "topologies": ["recompression_brayton"],
        "params": {"eta_design": 0.87, "N_design_rpm": 3600},
    },
    "split_valve": {
        "type": "ThermoPower.Water.SplitValve",
        "topologies": ["recompression_brayton"],
        "params": {"split_ratio_min": 0.15},
    },
}


# ─── ComponentFactory.create() ────────────────────────────────────────────────

class TestComponentFactoryCreate:
    def test_create_compressor_from_yaml(self):
        spec = ComponentFactory.create("main_compressor", COMPRESSOR_CONFIG)
        assert isinstance(spec, ComponentSpec)
        assert spec.name == "main_compressor"
        assert spec.modelica_type == "SCOPE.Compressors.CentrifugalCompressor"
        assert spec.params["eta_design"] == pytest.approx(0.88)
        assert spec.params["N_design_rpm"] == 3600

    def test_create_turbine_from_yaml(self):
        spec = ComponentFactory.create("turbine", TURBINE_CONFIG)
        assert isinstance(spec, ComponentSpec)
        assert spec.name == "turbine"
        assert spec.modelica_type == "SCOPE.Turbines.AxialTurbine"
        assert spec.params["eta_design"] == pytest.approx(0.92)
        assert spec.params["expansion_ratio_design"] == pytest.approx(2.4)

    def test_create_heat_exchanger_from_yaml(self):
        spec = ComponentFactory.create("recuperator", RECUPERATOR_CONFIG)
        assert isinstance(spec, ComponentSpec)
        assert spec.name == "recuperator"
        assert spec.modelica_type == "ThermoPower.Gas.HE"
        assert spec.params["UA_design"] == pytest.approx(750.0)
        assert spec.params["effectiveness_design"] == pytest.approx(0.92)

    def test_create_valve_from_yaml(self):
        spec = ComponentFactory.create("bypass_valve", BYPASS_VALVE_CONFIG)
        assert isinstance(spec, ComponentSpec)
        assert spec.name == "bypass_valve"
        assert spec.modelica_type == "ThermoPower.Water.ValveLin"
        assert spec.params["Cv_design"] == pytest.approx(100.0)

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
        # Must include these
        expected_included = {
            "main_compressor", "turbine", "recuperator", "precooler",
            "heat_source", "bypass_valve", "igv", "inventory_valve", "cooling_valve",
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
        """simple_recuperated topology yields exactly 9 components."""
        filtered = ComponentFactory.get_components_for_topology(all_specs, "simple_recuperated")
        assert len(filtered) == 9

    def test_recompression_brayton_count(self, all_specs):
        """recompression_brayton topology yields exactly 12 components."""
        filtered = ComponentFactory.get_components_for_topology(all_specs, "recompression_brayton")
        assert len(filtered) == 12
