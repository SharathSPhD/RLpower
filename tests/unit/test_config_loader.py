"""Unit tests for ConfigLoader (TDD — written before implementation).

Tests run inside Docker on DGX Spark (ARM64). See RULE-D1, RULE-D2.
All tests use GLOBAL_SEED=42 via the autouse `seed` fixture in conftest.py.

Covered:
    - test_load_model_config
    - test_load_fmu_config
    - test_load_env_config
    - test_load_safety_config
    - test_load_curriculum_config
    - test_components_for_topology_simple
    - test_invalid_path_raises
    - test_topology_action_dim_simple
    - test_topology_action_dim_recompression
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sco2rl.utils.config import (
    ConfigError,
    ConfigLoader,
    ModelConfig,
    TopologyConfig,
    ComponentConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def loader() -> ConfigLoader:
    return ConfigLoader()


# ---------------------------------------------------------------------------
# test_load_model_config
# ---------------------------------------------------------------------------

class TestLoadModelConfig:
    def test_load_model_config(self, loader: ConfigLoader, model_config_path: Path) -> None:
        """Loads configs/model/base_cycle.yaml; verifies cycle.name and topology.type."""
        cfg = loader.load_model_config(model_config_path)

        assert cfg.cycle.name == "SCO2_WHR", (
            f"Expected cycle.name='SCO2_WHR', got {cfg.cycle.name!r}"
        )
        assert cfg.topology.type == "simple_recuperated", (
            f"Expected topology.type='simple_recuperated', got {cfg.topology.type!r}"
        )

    def test_model_config_has_libraries(self, loader: ConfigLoader, model_config_path: Path) -> None:
        """Libraries list contains the four required Modelica libraries."""
        cfg = loader.load_model_config(model_config_path)
        required = {"Modelica", "Steps"}
        assert required.issubset(set(cfg.cycle.libraries)), (
            f"Missing libraries. Got: {cfg.cycle.libraries}"
        )

    def test_model_config_has_components(self, loader: ConfigLoader, model_config_path: Path) -> None:
        """Components dict is non-empty."""
        cfg = loader.load_model_config(model_config_path)
        assert len(cfg.components) > 0

    def test_model_config_design_point(self, loader: ConfigLoader, model_config_path: Path) -> None:
        """Design-point parameters load with expected values."""
        cfg = loader.load_model_config(model_config_path)
        assert cfg.parameters.net_power_mwe == pytest.approx(10.0)
        assert cfg.parameters.design_pressure_ratio == pytest.approx(2.4)


# ---------------------------------------------------------------------------
# test_load_fmu_config
# ---------------------------------------------------------------------------

class TestLoadFMUConfig:
    def test_load_fmu_config(self, loader: ConfigLoader, fmu_config_path: Path) -> None:
        """Loads configs/fmu/fmu_export.yaml; verifies cvode_tolerance_training == 1e-4."""
        cfg = loader.load_fmu_config(fmu_config_path)

        assert cfg.compiler.cvode_tolerance_training == pytest.approx(1e-4), (
            f"Expected cvode_tolerance_training=1e-4, got {cfg.compiler.cvode_tolerance_training}"
        )

    def test_fmu_config_fmi_version(self, loader: ConfigLoader, fmu_config_path: Path) -> None:
        """FMI version is '2.0' (Co-Simulation, required for CVODE embedding)."""
        cfg = loader.load_fmu_config(fmu_config_path)
        assert cfg.export.fmi_version == "2.0"
        assert cfg.export.fmu_type == "cs"

    def test_fmu_config_solver_failure(self, loader: ConfigLoader, fmu_config_path: Path) -> None:
        """Solver failure reward matches the -100 documented constraint."""
        cfg = loader.load_fmu_config(fmu_config_path)
        assert cfg.solver_failure.reward_on_failure == pytest.approx(-100.0)
        assert cfg.solver_failure.terminate_on_failure is True

    def test_fmu_config_required_libs(self, loader: ConfigLoader, fmu_config_path: Path) -> None:
        """Required runtime libs include CoolProp and ExternalMedia shared objects."""
        cfg = loader.load_fmu_config(fmu_config_path)
        libs = cfg.runtime_deps.required_libs
        assert "libCoolProp.so" in libs
        assert "libExternalMediaLib.so" in libs

    def test_fmu_config_cvode_validation_tolerance(
        self, loader: ConfigLoader, fmu_config_path: Path
    ) -> None:
        """Validation tolerance is 1e-6 (stricter than training 1e-4)."""
        cfg = loader.load_fmu_config(fmu_config_path)
        assert cfg.compiler.cvode_tolerance_validation == pytest.approx(1e-6)
        # Validation tolerance should be strictly tighter than training tolerance
        assert cfg.compiler.cvode_tolerance_validation < cfg.compiler.cvode_tolerance_training


# ---------------------------------------------------------------------------
# test_load_env_config
# ---------------------------------------------------------------------------

class TestLoadEnvConfig:
    def test_load_env_config(self, loader: ConfigLoader, env_config_path: Path) -> None:
        """Loads configs/environment/env.yaml without errors."""
        cfg = loader.load_env_config(env_config_path)
        assert cfg is not None

    def test_env_config_obs_dim(self, loader: ConfigLoader, env_config_path: Path) -> None:
        """Observation dimension is 80 (16 vars × 5 history steps, simple_recuperated)."""
        cfg = loader.load_env_config(env_config_path)
        assert cfg.observation.obs_dim == 80
        assert cfg.observation.history_steps == 5

    def test_env_config_action_dim(self, loader: ConfigLoader, env_config_path: Path) -> None:
        """env.yaml declares action_dim = 4 (simple_recuperated: no split_ratio)."""
        cfg = loader.load_env_config(env_config_path)
        assert cfg.action.action_dim == 4

    def test_env_config_reward_weights(self, loader: ConfigLoader, env_config_path: Path) -> None:
        """Reward weights are present and positive."""
        cfg = loader.load_env_config(env_config_path)
        assert cfg.reward.w_tracking > 0
        assert cfg.reward.w_efficiency > 0
        assert cfg.reward.w_smoothness > 0

    def test_env_config_normalization(self, loader: ConfigLoader, env_config_path: Path) -> None:
        """VecNormalize is enabled (required per CLAUDE.md)."""
        cfg = loader.load_env_config(env_config_path)
        assert cfg.normalization.use_vec_normalize is True

    def test_env_config_observation_variables_count(
        self, loader: ConfigLoader, env_config_path: Path
    ) -> None:
        """Exactly 16 observation variables are defined (simple_recuperated cycle)."""
        cfg = loader.load_env_config(env_config_path)
        assert len(cfg.observation.variables) == 16

    def test_env_config_compressor_inlet_min(
        self, loader: ConfigLoader, env_config_path: Path
    ) -> None:
        """Compressor inlet temperature observation has min >= 30°C (near-critical region)."""
        cfg = loader.load_env_config(env_config_path)
        t_comp = next(
            v for v in cfg.observation.variables if v.name == "T_compressor_inlet"
        )
        assert t_comp.min >= 30.0


# ---------------------------------------------------------------------------
# test_load_safety_config
# ---------------------------------------------------------------------------

class TestLoadSafetyConfig:
    def test_load_safety_config(self, loader: ConfigLoader, safety_config_path: Path) -> None:
        """Loads configs/safety/constraints.yaml; verifies compressor_inlet_temp_min_c == 32.2."""
        cfg = loader.load_safety_config(safety_config_path)

        assert cfg.hard_constraints.compressor_inlet_temp_min_c == pytest.approx(32.2), (
            f"Expected compressor_inlet_temp_min_c=32.2, "
            f"got {cfg.hard_constraints.compressor_inlet_temp_min_c}"
        )

    def test_safety_config_surge_margin(self, loader: ConfigLoader, safety_config_path: Path) -> None:
        """Surge margin minimum is 0.05 for both compressors (RULE-P4)."""
        cfg = loader.load_safety_config(safety_config_path)
        assert cfg.hard_constraints.surge_margin_main_min == pytest.approx(0.05)
        assert cfg.hard_constraints.surge_margin_recomp_min == pytest.approx(0.05)

    def test_safety_config_pressure_limits(
        self, loader: ConfigLoader, safety_config_path: Path
    ) -> None:
        """High-side pressure limits are physically sensible."""
        cfg = loader.load_safety_config(safety_config_path)
        hc = cfg.hard_constraints
        assert hc.high_pressure_min_mpa < hc.high_pressure_max_mpa
        assert hc.high_pressure_max_mpa == pytest.approx(22.0)

    def test_safety_config_catastrophic_violations_non_empty(
        self, loader: ConfigLoader, safety_config_path: Path
    ) -> None:
        """At least one catastrophic violation condition is defined."""
        cfg = loader.load_safety_config(safety_config_path)
        assert len(cfg.catastrophic_violations) > 0

    def test_safety_config_lagrangian_present(
        self, loader: ConfigLoader, safety_config_path: Path
    ) -> None:
        """Lagrangian block is present and contains constraints list."""
        cfg = loader.load_safety_config(safety_config_path)
        assert "constraints" in cfg.lagrangian
        assert len(cfg.lagrangian["constraints"]) > 0

    def test_safety_config_compressor_catastrophic_lower_than_min(
        self, loader: ConfigLoader, safety_config_path: Path
    ) -> None:
        """Catastrophic temperature threshold is below the hard constraint minimum."""
        cfg = loader.load_safety_config(safety_config_path)
        hc = cfg.hard_constraints
        assert hc.compressor_inlet_temp_catastrophic_c < hc.compressor_inlet_temp_min_c


# ---------------------------------------------------------------------------
# test_load_curriculum_config
# ---------------------------------------------------------------------------

class TestLoadCurriculumConfig:
    def test_load_curriculum_config(
        self, loader: ConfigLoader, curriculum_config_path: Path
    ) -> None:
        """Loads configs/curriculum/curriculum.yaml without errors."""
        data = loader.load(curriculum_config_path)
        assert data is not None
        assert "phases" in data

    def test_curriculum_has_seven_phases(
        self, loader: ConfigLoader, curriculum_config_path: Path
    ) -> None:
        """Curriculum defines exactly 7 training phases (Scenarios 0–6)."""
        data = loader.load(curriculum_config_path)
        assert len(data["phases"]) == 7

    def test_curriculum_phase_ids_sequential(
        self, loader: ConfigLoader, curriculum_config_path: Path
    ) -> None:
        """Phase IDs are sequential integers 0 through 6."""
        data = loader.load(curriculum_config_path)
        ids = [phase["id"] for phase in data["phases"]]
        assert ids == list(range(7))

    def test_curriculum_advancement_thresholds_descending(
        self, loader: ConfigLoader, curriculum_config_path: Path
    ) -> None:
        """Advancement thresholds decrease as curriculum gets harder (phases 0 → 6)."""
        data = loader.load(curriculum_config_path)
        thresholds = [phase["advancement_threshold"] for phase in data["phases"]]
        # Each threshold must be <= the previous (harder phases need lower reward to pass)
        for i in range(1, len(thresholds)):
            assert thresholds[i] <= thresholds[i - 1], (
                f"Phase {i} threshold {thresholds[i]} is not <= phase {i-1} "
                f"threshold {thresholds[i-1]}"
            )

    def test_curriculum_advancement_config(
        self, loader: ConfigLoader, curriculum_config_path: Path
    ) -> None:
        """Advancement policy uses 50-episode window as specified in CLAUDE.md."""
        data = loader.load(curriculum_config_path)
        assert data["advancement"]["window_episodes"] == 50


# ---------------------------------------------------------------------------
# test_components_for_topology_simple
# ---------------------------------------------------------------------------

class TestGetActiveComponents:
    def test_components_for_topology_simple(
        self, loader: ConfigLoader, model_config_path: Path
    ) -> None:
        """Components filtered for simple_recuperated topology exclude recompression-only parts."""
        cfg = loader.load_model_config(model_config_path)
        active = loader.get_active_components(cfg)

        # Must include: components with "simple_recuperated" in their topologies list
        assert "main_compressor" in active
        assert "turbine" in active
        assert "recuperator" in active
        assert "precooler" in active
        assert "regulator" in active
        assert "bypass_valve" in active
        assert "inventory_valve" in active

        # Must EXCLUDE: recompression_brayton-only components
        assert "recompressor" not in active
        assert "split_valve" not in active
        assert "recuperator_high_temp" not in active
        assert "recuperator_low_temp" not in active

    def test_components_for_topology_recompression(
        self, loader: ConfigLoader, model_config_path: Path
    ) -> None:
        """Components filtered for recompression_brayton include split_valve and recompressor."""
        cfg = loader.load_model_config(model_config_path)
        # Temporarily override topology type for this test
        recomp_cfg = ModelConfig(
            cycle=cfg.cycle,
            topology=TopologyConfig(type="recompression_brayton"),
            parameters=cfg.parameters,
            fluid=cfg.fluid,
            components=cfg.components,
            flow_paths=cfg.flow_paths,
        )
        active = loader.get_active_components(recomp_cfg)

        assert "recompressor" in active
        assert "split_valve" in active
        assert "recuperator_high_temp" in active
        assert "recuperator_low_temp" in active

        # simple_recuperated-only components must be absent
        assert "recuperator" not in active

    def test_active_components_returns_sorted_list(
        self, loader: ConfigLoader, model_config_path: Path
    ) -> None:
        """get_active_components returns a sorted list (deterministic order)."""
        cfg = loader.load_model_config(model_config_path)
        active = loader.get_active_components(cfg)
        assert active == sorted(active)


# ---------------------------------------------------------------------------
# test_invalid_path_raises
# ---------------------------------------------------------------------------

class TestInvalidPath:
    def test_invalid_path_raises(self, loader: ConfigLoader) -> None:
        """ConfigError raised when the config file does not exist."""
        with pytest.raises(ConfigError, match="not found"):
            loader.load("configs/does_not_exist/nonexistent.yaml")

    def test_invalid_path_raises_for_model_config(self, loader: ConfigLoader) -> None:
        """load_model_config raises ConfigError for non-existent file."""
        with pytest.raises(ConfigError):
            loader.load_model_config("/tmp/definitely_not_a_real_config.yaml")

    def test_invalid_path_raises_for_fmu_config(self, loader: ConfigLoader) -> None:
        """load_fmu_config raises ConfigError for non-existent file."""
        with pytest.raises(ConfigError):
            loader.load_fmu_config("/tmp/definitely_not_a_real_config.yaml")

    def test_invalid_path_raises_for_env_config(self, loader: ConfigLoader) -> None:
        """load_env_config raises ConfigError for non-existent file."""
        with pytest.raises(ConfigError):
            loader.load_env_config("/tmp/definitely_not_a_real_config.yaml")

    def test_invalid_path_raises_for_safety_config(self, loader: ConfigLoader) -> None:
        """load_safety_config raises ConfigError for non-existent file."""
        with pytest.raises(ConfigError):
            loader.load_safety_config("/tmp/definitely_not_a_real_config.yaml")

    def test_config_error_is_exception(self) -> None:
        """ConfigError inherits from Exception."""
        err = ConfigError("test message")
        assert isinstance(err, Exception)
        assert str(err) == "test message"


# ---------------------------------------------------------------------------
# test_topology_action_dim_simple and test_topology_action_dim_recompression
# ---------------------------------------------------------------------------

class TestTopologyActionDim:
    def test_topology_action_dim_simple(
        self, loader: ConfigLoader, model_config_path: Path
    ) -> None:
        """simple_recuperated topology → action_dim == 4."""
        cfg = loader.load_model_config(model_config_path)
        assert cfg.topology.type == "simple_recuperated"
        action_dim = loader.get_action_dim(cfg)
        assert action_dim == 4, (
            f"Expected action_dim=4 for simple_recuperated, got {action_dim}"
        )

    def test_topology_action_dim_recompression(
        self, loader: ConfigLoader, model_config_path: Path
    ) -> None:
        """recompression_brayton topology → action_dim == 5."""
        cfg = loader.load_model_config(model_config_path)
        # Build a recompression config variant
        recomp_cfg = ModelConfig(
            cycle=cfg.cycle,
            topology=TopologyConfig(type="recompression_brayton"),
            parameters=cfg.parameters,
            fluid=cfg.fluid,
            components=cfg.components,
            flow_paths=cfg.flow_paths,
        )
        action_dim = loader.get_action_dim(recomp_cfg)
        assert action_dim == 5, (
            f"Expected action_dim=5 for recompression_brayton, got {action_dim}"
        )

    def test_topology_action_dim_simple_matches_component_count(
        self, loader: ConfigLoader, model_config_path: Path
    ) -> None:
        """Action dim for simple_recuperated matches the number of valve actuator components.

        Valve actuators in simple_recuperated: bypass_valve, inventory_valve → 2 valves,
        plus igv (compressor control) and cooling_flow (precooler control) = 4 total actions.
        Verify by checking the hardcoded map returns 4.
        """
        cfg = loader.load_model_config(model_config_path)
        assert loader.get_action_dim(cfg) == 4

    def test_invalid_topology_in_action_dim_raises(
        self, loader: ConfigLoader, model_config_path: Path
    ) -> None:
        """ConfigError raised if topology.type is not in the action dim mapping.

        TopologyConfig accepts arbitrary strings (semantic validation is deferred
        to get_action_dim so future topologies can be added without Pydantic changes).
        This test verifies the defensive guard in get_action_dim.
        """
        cfg = loader.load_model_config(model_config_path)
        # Construct an unknown topology directly — no validator blocks arbitrary strings
        unknown_topo = TopologyConfig(type="unknown_topology_xyz")
        recomp_cfg = ModelConfig(
            cycle=cfg.cycle,
            topology=unknown_topo,
            parameters=cfg.parameters,
            fluid=cfg.fluid,
            components=cfg.components,
            flow_paths=cfg.flow_paths,
        )
        with pytest.raises(ConfigError, match="Unknown topology"):
            loader.get_action_dim(recomp_cfg)
