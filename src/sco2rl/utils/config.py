"""ConfigLoader — typed YAML configuration loading with Pydantic v2 validation.

All paths are resolved relative to PROJECT_ROOT (repo root), never hardcoded
absolute paths. See RULE-C3.

Tests run inside Docker (DGX Spark / ARM64). See RULE-D1, RULE-D2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

# PROJECT_ROOT: three parents up from this file:
#   src/sco2rl/utils/config.py → src/sco2rl/utils → src/sco2rl → src → PROJECT_ROOT
PROJECT_ROOT = Path(__file__).parents[3]


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised for missing files, invalid YAML, or Pydantic validation failures."""


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CycleConfig(BaseModel):
    """Top-level cycle identity from base_cycle.yaml."""

    name: str
    package: str
    libraries: list[str]


class TopologyConfig(BaseModel):
    """Topology selection block.

    The type field accepts any string at parse time; semantic validation
    (allowed topology names) happens in ConfigLoader.get_action_dim() and
    ConfigLoader.get_active_components() so that future topology types can be
    added without changing the Pydantic model.
    """

    type: str  # "simple_recuperated" or "recompression_brayton"


class ComponentConfig(BaseModel):
    """Single component definition."""

    type: str
    topologies: list[str]
    params: dict[str, Any] = Field(default_factory=dict)


class ParametersConfig(BaseModel):
    """Design-point parameters."""

    net_power_mwe: float
    design_turbine_inlet_temp_c: float
    design_turbine_inlet_pressure_mpa: float
    design_compressor_inlet_temp_c: float
    design_compressor_inlet_pressure_mpa: float
    design_mass_flow_kg_s: float
    design_thermal_efficiency: float
    design_pressure_ratio: float


class FluidConfig(BaseModel):
    """CO₂ fluid configuration."""

    medium: str
    coolprop_name: str
    coolprop_options: str
    critical_temp_c: float
    critical_pressure_mpa: float


class ModelConfig(BaseModel):
    """Full model configuration (base_cycle.yaml)."""

    cycle: CycleConfig
    topology: TopologyConfig
    parameters: ParametersConfig
    fluid: FluidConfig
    components: dict[str, ComponentConfig]
    # flow_paths is documentation-only; load as raw dict, no strict schema
    flow_paths: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------


class CompilerConfig(BaseModel):
    """FMU compiler settings."""

    omc_flags: list[str]
    cvode_tolerance_training: float
    cvode_tolerance_validation: float
    jacobian: str
    max_integration_order: int

    @field_validator("cvode_tolerance_training", "cvode_tolerance_validation")
    @classmethod
    def tolerances_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("CVODE tolerance must be positive")
        return v


class RuntimeDepsConfig(BaseModel):
    required_libs: list[str]
    inject_target: str
    lib_source_dir: str


class SolverFailureConfig(BaseModel):
    reward_on_failure: float
    terminate_on_failure: bool
    max_consecutive_failures: int
    log_warnings: bool


class ExportConfig(BaseModel):
    fmi_version: str
    fmu_type: str
    output_dir: str
    fmu_name: str
    modelica_source_dir: str


class FMUConfig(BaseModel):
    """Full FMU export/runtime configuration (fmu_export.yaml)."""

    export: ExportConfig
    compiler: CompilerConfig
    runtime_deps: RuntimeDepsConfig
    communication_step_s: float
    max_steps_per_episode: int
    stop_time_s: float
    warmup_seconds: float
    warmup_actions: dict[str, float]
    solver_failure: SolverFailureConfig
    validation: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------


class ObsVariableConfig(BaseModel):
    name: str
    fmu_var: str | None
    index: int
    min: float
    max: float
    unit: str
    description: str = ""


class ActionVariableConfig(BaseModel):
    name: str
    fmu_var: str
    index: int
    physical_min: float
    physical_max: float
    rate_limit_per_step: float
    description: str = ""


class ObservationConfig(BaseModel):
    history_steps: int
    obs_dim: int
    variables: list[ObsVariableConfig]


class ActionConfig(BaseModel):
    action_dim: int
    variables: list[ActionVariableConfig]


class RewardConfig(BaseModel):
    w_tracking: float
    w_efficiency: float
    w_smoothness: float
    rated_power_mw: float
    design_efficiency: float
    terminal_failure_reward: float


class NormalizationConfig(BaseModel):
    use_vec_normalize: bool
    norm_obs: bool
    norm_reward: bool
    clip_obs: float
    clip_reward: float
    gamma: float
    update_freq: int


class EpisodeConfig(BaseModel):
    max_steps: int
    reset_mode: str


class EnvConfig(BaseModel):
    """Full Gymnasium environment configuration (env.yaml)."""

    observation: ObservationConfig
    action: ActionConfig
    reward: RewardConfig
    normalization: NormalizationConfig
    episode: EpisodeConfig


# ---------------------------------------------------------------------------


class HardConstraintsConfig(BaseModel):
    compressor_inlet_temp_min_c: float
    compressor_inlet_temp_max_c: float
    compressor_inlet_temp_catastrophic_c: float
    turbine_inlet_temp_min_c: float
    turbine_inlet_temp_max_c: float
    high_pressure_min_mpa: float
    high_pressure_max_mpa: float
    surge_margin_main_min: float
    surge_margin_recomp_min: float
    net_power_min_mw: float
    turbine_temp_rate_max_c_per_min: float


class SafetyConfig(BaseModel):
    """Full safety constraints configuration (constraints.yaml)."""

    hard_constraints: HardConstraintsConfig
    catastrophic_violations: list[dict[str, Any]]
    lagrangian: dict[str, Any]
    deployment: dict[str, Any]


# ---------------------------------------------------------------------------
# Topology → action dimension mapping
# ---------------------------------------------------------------------------

_TOPOLOGY_ACTION_DIM: dict[str, int] = {
    "simple_recuperated": 4,   # bypass_valve, igv, inventory_valve, cooling_flow
    "recompression_brayton": 5,  # above + split_ratio
}


# ---------------------------------------------------------------------------
# ConfigLoader
# ---------------------------------------------------------------------------


class ConfigLoader:
    """Loads and validates YAML configuration files for the sCO₂ RL project.

    All paths are resolved relative to PROJECT_ROOT (repo root). No hardcoded
    absolute paths (RULE-C3).

    Usage (inside Docker on DGX Spark — RULE-D1, RULE-D2)::

        loader = ConfigLoader()
        model_cfg = loader.load_model_config("configs/model/base_cycle.yaml")
        action_dim = loader.get_action_dim(model_cfg)
    """

    def load(self, path: str | Path) -> dict[str, Any]:
        """Load a YAML file and return it as a plain dict.

        Resolves relative paths against PROJECT_ROOT.

        Raises:
            ConfigError: if the file does not exist, is not valid YAML, or is empty.
        """
        resolved = self._resolve(path)
        if not resolved.exists():
            raise ConfigError(f"Config file not found: {resolved}")
        try:
            with resolved.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigError(f"YAML parse error in {resolved}: {exc}") from exc
        if data is None:
            raise ConfigError(f"Config file is empty: {resolved}")
        if not isinstance(data, dict):
            raise ConfigError(
                f"Expected a YAML mapping at top level in {resolved}, got {type(data).__name__}"
            )
        return data

    def load_model_config(self, path: str | Path) -> ModelConfig:
        """Load and validate base_cycle.yaml → ModelConfig.

        Raises:
            ConfigError: on file/parse/validation failure.
        """
        data = self.load(path)
        return self._parse(ModelConfig, data, path)

    def load_env_config(self, path: str | Path) -> EnvConfig:
        """Load and validate env.yaml → EnvConfig.

        Raises:
            ConfigError: on file/parse/validation failure.
        """
        data = self.load(path)
        return self._parse(EnvConfig, data, path)

    def load_fmu_config(self, path: str | Path) -> FMUConfig:
        """Load and validate fmu_export.yaml → FMUConfig.

        Raises:
            ConfigError: on file/parse/validation failure.
        """
        data = self.load(path)
        return self._parse(FMUConfig, data, path)

    def load_safety_config(self, path: str | Path) -> SafetyConfig:
        """Load and validate constraints.yaml → SafetyConfig.

        Raises:
            ConfigError: on file/parse/validation failure.
        """
        data = self.load(path)
        return self._parse(SafetyConfig, data, path)

    def get_active_components(self, model_config: ModelConfig) -> list[str]:
        """Return component names whose topologies list includes the active topology.

        Args:
            model_config: A validated ModelConfig instance.

        Returns:
            Sorted list of component names compatible with the active topology.
        """
        active_topology = model_config.topology.type
        return sorted(
            name
            for name, comp in model_config.components.items()
            if active_topology in comp.topologies
        )

    def get_action_dim(self, model_config: ModelConfig) -> int:
        """Return action space dimensionality for the active topology.

        Args:
            model_config: A validated ModelConfig instance.

        Returns:
            4 for simple_recuperated, 5 for recompression_brayton.

        Raises:
            ConfigError: if topology.type is not recognized.
        """
        topo = model_config.topology.type
        if topo not in _TOPOLOGY_ACTION_DIM:
            raise ConfigError(
                f"Unknown topology {topo!r}; no action_dim mapping defined. "
                f"Known topologies: {list(_TOPOLOGY_ACTION_DIM)}"
            )
        return _TOPOLOGY_ACTION_DIM[topo]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(path: str | Path) -> Path:
        """Resolve path: absolute paths used as-is; relative paths → PROJECT_ROOT."""
        p = Path(path)
        if p.is_absolute():
            return p
        return PROJECT_ROOT / p

    @staticmethod
    def _parse(model_cls: type[BaseModel], data: dict[str, Any], path: str | Path) -> Any:
        """Instantiate a Pydantic model, wrapping ValidationError as ConfigError."""
        from pydantic import ValidationError

        try:
            return model_cls.model_validate(data)
        except ValidationError as exc:
            raise ConfigError(
                f"Validation failed for {path}:\n{exc}"
            ) from exc
