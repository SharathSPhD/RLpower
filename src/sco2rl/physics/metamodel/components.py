"""Component specifications for sCO₂ cycle topology.

ComponentFactory creates typed component specs from YAML config dicts.
All specs are topology-tagged so SCO2CycleBuilder can filter by active topology.

Supported Modelica type strings (must match YAML 'type' field exactly):
    SCOPE.Compressors.CentrifugalCompressor
    SCOPE.Turbines.AxialTurbine
    ThermoPower.Gas.HE
    ThermoPower.Water.ValveLin
    ThermoPower.Water.SplitValve
    SCOPE.Actuators.InletGuideVane
    SCOPE.HeatSources.ExhaustHeatSource
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ─── Known Modelica type strings ──────────────────────────────────────────────
# These must match the 'type' values in base_cycle.yaml exactly.

_KNOWN_MODELICA_TYPES: frozenset[str] = frozenset({
    "SCOPE.Compressors.CentrifugalCompressor",
    "SCOPE.Turbines.AxialTurbine",
    "ThermoPower.Gas.HE",
    "ThermoPower.Water.ValveLin",
    "ThermoPower.Water.SplitValve",
    "SCOPE.Actuators.InletGuideVane",
    "SCOPE.HeatSources.ExhaustHeatSource",
})


# ─── Exceptions ───────────────────────────────────────────────────────────────

class ComponentTypeError(Exception):
    """Raised when an unknown Modelica component type string is encountered."""


# ─── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class ComponentSpec:
    """Typed specification for a single sCO₂ cycle component.

    Attributes:
        name: Instance name in Modelica (e.g. 'main_compressor').
        modelica_type: Fully-qualified Modelica type string
            (e.g. 'SCOPE.Compressors.CentrifugalCompressor').
        params: Dict of parameter names and values from YAML config.
        topologies: List of topology names that include this component
            (e.g. ['simple_recuperated', 'recompression_brayton']).
    """

    name: str
    modelica_type: str
    params: dict[str, Any]
    topologies: list[str] = field(default_factory=list)


# ─── Factory ─────────────────────────────────────────────────────────────────

class ComponentFactory:
    """Creates ComponentSpec instances from YAML config dicts.

    All methods are static — no instance needed.
    """

    @staticmethod
    def create(name: str, config: dict[str, Any]) -> ComponentSpec:
        """Create a ComponentSpec from a YAML component config dict.

        Args:
            name: Instance name (key from the 'components:' block in YAML).
            config: Dict with keys 'type', 'topologies', 'params'.

        Returns:
            A validated ComponentSpec.

        Raises:
            ComponentTypeError: If config['type'] is not a recognised Modelica
                type string.
        """
        modelica_type = config["type"]
        if modelica_type not in _KNOWN_MODELICA_TYPES:
            raise ComponentTypeError(
                f"Unknown Modelica component type {modelica_type!r}. "
                f"Known types: {sorted(_KNOWN_MODELICA_TYPES)}"
            )

        topologies: list[str] = list(config.get("topologies", []))
        params: dict[str, Any] = dict(config.get("params", {}))

        return ComponentSpec(
            name=name,
            modelica_type=modelica_type,
            params=params,
            topologies=topologies,
        )

    @staticmethod
    def get_components_for_topology(
        all_components: dict[str, ComponentSpec],
        topology: str,
    ) -> dict[str, ComponentSpec]:
        """Filter the component dict to only those belonging to a given topology.

        Args:
            all_components: Full dict of name → ComponentSpec (all topologies).
            topology: Active topology name, e.g. 'simple_recuperated'.

        Returns:
            Filtered dict containing only components whose topologies list
            includes the requested topology.
        """
        return {
            name: spec
            for name, spec in all_components.items()
            if topology in spec.topologies
        }
