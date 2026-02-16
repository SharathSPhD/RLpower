"""Fluent builder for sCO₂ cycle Modelica models.

Usage (manual):
    builder = SCO2CycleBuilder()
    builder._topology = "simple_recuperated"
    builder.add_component("main_compressor", spec).add_component("turbine", spec)
    builder.connect("main_compressor.outlet", "recuperator.cold_in")
    cycle_model = builder.build()

Usage (from YAML config):
    builder = SCO2CycleBuilder.from_config(model_config_dict)
    cycle_model = builder.build()
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .components import ComponentFactory, ComponentSpec
from .connections import ConnectionSpec


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class CycleModel:
    """Assembled sCO₂ cycle model ready for Modelica rendering.

    Attributes:
        name: Cycle identifier from config (e.g. 'SCO2_WHR').
        package: Modelica package/model name (e.g. 'SCO2RecuperatedCycle').
        fluid_config: Dict of fluid medium settings (medium, coolprop_options, …).
        components: Dict of instance name → ComponentSpec for active topology.
        connections: List of ConnectionSpec (connect() equations).
        topology: Active topology name (e.g. 'simple_recuperated').
    """

    name: str
    package: str
    fluid_config: dict[str, Any]
    components: dict[str, ComponentSpec]
    connections: list[ConnectionSpec]
    topology: str


# ─── Exception ────────────────────────────────────────────────────────────────

class BuildError(Exception):
    """Raised when the builder cannot produce a valid CycleModel."""


# ─── Builder ─────────────────────────────────────────────────────────────────

class SCO2CycleBuilder:
    """Fluent builder that assembles a CycleModel from components and connections.

    Topology is driven by ``_topology`` (RULE-D3).  ``from_config()`` sets this
    from the YAML ``topology.type`` key — no hardcoding needed.
    """

    # Required component names per topology. build() enforces these.
    REQUIRED_COMPONENTS: dict[str, list[str]] = {
        "simple_recuperated": [
            "regulator",        # inlet boundary condition (turbine inlet T, P, m_flow)
            "main_compressor",
            "turbine",
            "recuperator",
            "precooler",
        ],
        "recompression_brayton": [
            "regulator",        # inlet boundary condition
            "main_compressor",
            "turbine",
            "recuperator_high_temp",
            "recuperator_low_temp",
            "precooler",
            "recompressor",
        ],
    }

    def __init__(self) -> None:
        self._components: dict[str, ComponentSpec] = {}
        self._connections: list[ConnectionSpec] = []
        self._topology: str = "simple_recuperated"
        self._name: str = "SCO2_WHR"
        self._package: str = "SCO2RecuperatedCycle"
        self._fluid_config: dict[str, Any] = {}

    # ── Fluent methods ────────────────────────────────────────────────────────

    def add_component(self, name: str, spec: ComponentSpec) -> "SCO2CycleBuilder":
        """Add a component to the builder.

        Args:
            name: Instance name for the component.
            spec: Populated ComponentSpec.

        Returns:
            self (for method chaining).
        """
        self._components[name] = spec
        return self

    def connect(self, from_port: str, to_port: str) -> "SCO2CycleBuilder":
        """Add a connection between two component ports.

        Args:
            from_port: Source port in "component.port" format.
            to_port: Destination port in "component.port" format.

        Returns:
            self (for method chaining).

        Raises:
            BuildError: If the component referenced in from_port or to_port
                has not been added to the builder yet.
        """
        from_component = from_port.split(".")[0]
        to_component = to_port.split(".")[0]

        if from_component not in self._components:
            raise BuildError(
                f"Cannot connect: component {from_component!r} not added to builder. "
                f"Add it with add_component() first."
            )
        if to_component not in self._components:
            raise BuildError(
                f"Cannot connect: component {to_component!r} not added to builder. "
                f"Add it with add_component() first."
            )

        self._connections.append(ConnectionSpec(from_port=from_port, to_port=to_port))
        return self

    def build(self) -> CycleModel:
        """Validate the builder state and return a CycleModel.

        Only components that appear in at least one connection are included in
        the output model.  Declared-but-unconnected components (e.g. actuator
        valves not yet wired into the flow path) are silently excluded, which
        prevents Modelica from reporting them as floating subsystems with
        imbalanced equations.

        Raises:
            BuildError: If a required component for the active topology is
                missing from the builder.

        Returns:
            A fully-assembled CycleModel.
        """
        required = self.REQUIRED_COMPONENTS.get(self._topology, [])
        for req in required:
            if req not in self._components:
                raise BuildError(
                    f"Required component {req!r} is missing for topology "
                    f"{self._topology!r}. Add it with add_component()."
                )

        # Only include components that appear in at least one connection.
        # Unconnected actuators (bypass_valve, inventory_valve) are omitted so
        # that the generated .mo file has no floating, equation-unbalanced blocks.
        connected: set[str] = set()
        for conn in self._connections:
            connected.add(conn.from_port.split(".")[0])
            connected.add(conn.to_port.split(".")[0])

        active_components = {
            name: spec
            for name, spec in self._components.items()
            if name in connected
        }

        return CycleModel(
            name=self._name,
            package=self._package,
            fluid_config=dict(self._fluid_config),
            components=active_components,
            connections=list(self._connections),
            topology=self._topology,
        )

    # ── Class method constructor ──────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SCO2CycleBuilder":
        """Construct a fully-populated builder from a YAML config dict.

        Reads topology.type (RULE-D3), creates all components via
        ComponentFactory, filters by topology, and populates connections
        from flow_paths.

        Args:
            config: Parsed content of base_cycle.yaml (or equivalent).

        Returns:
            A SCO2CycleBuilder ready to call .build() on.
        """
        builder = cls()

        # ── Cycle metadata ────────────────────────────────────────────────
        cycle_cfg = config.get("cycle", {})
        builder._name = cycle_cfg.get("name", "SCO2_WHR")
        builder._package = cycle_cfg.get("package", "SCO2RecuperatedCycle")

        # ── Fluid config ──────────────────────────────────────────────────
        builder._fluid_config = dict(config.get("fluid", {}))

        # ── Topology (RULE-D3) ────────────────────────────────────────────
        topology_cfg = config.get("topology", {})
        builder._topology = topology_cfg.get("type", "simple_recuperated")

        # ── Build all ComponentSpecs, then filter for active topology ─────
        all_component_cfgs: dict[str, Any] = config.get("components", {})
        all_specs: dict[str, ComponentSpec] = {}
        for comp_name, comp_cfg in all_component_cfgs.items():
            spec = ComponentFactory.create(comp_name, comp_cfg)
            all_specs[comp_name] = spec

        active_specs = ComponentFactory.get_components_for_topology(
            all_specs, builder._topology
        )
        for name, spec in active_specs.items():
            builder._components[name] = spec

        # ── Connections from flow_paths ────────────────────────────────────
        flow_paths: dict[str, list[str]] = config.get("flow_paths", {})
        topology_paths: list[str] = flow_paths.get(builder._topology, [])

        for path_entry in topology_paths:
            conn = _parse_flow_path_entry(path_entry, builder._components)
            if conn is not None:
                builder._connections.append(conn)

        return builder


# ─── Helpers ─────────────────────────────────────────────────────────────────

# Matches "a.b -> c.d" (standard flow path entry)
_ARROW_RE = re.compile(
    r"(?P<from_port>\w+\.\w+)\s*->\s*(?P<to_port>\w+\.\w+)"
)

# Matches "valve_name connects: a.b -> c.d" (bypass / auxiliary path entry)
_CONNECTS_RE = re.compile(
    r"\w+\s+connects:\s*(?P<from_port>\w+\.\w+)\s*->\s*(?P<to_port>\w+\.\w+)"
)


def _parse_flow_path_entry(
    entry: str,
    active_components: dict[str, ComponentSpec],
) -> ConnectionSpec | None:
    """Parse one flow_paths list entry into a ConnectionSpec.

    Supported formats:
        "main_compressor.outlet -> recuperator.cold_in"
        "bypass_valve connects: turbine.outlet -> precooler.hot_in"

    Returns None if the entry cannot be parsed or references a component not
    present in the active topology (silently skipped so cross-topology entries
    in the YAML do not cause build failures).

    Args:
        entry: Raw string from the flow_paths YAML list.
        active_components: Components currently in the builder (active topology).

    Returns:
        ConnectionSpec, or None if the entry should be skipped.
    """
    # YAML may parse "key: value" entries as dicts (e.g. "bypass_valve connects: a.b -> c.d"
    # becomes {"bypass_valve connects": "a.b -> c.d"}).  Reconstruct as a string.
    if isinstance(entry, dict):
        parts = []
        for k, v in entry.items():
            parts.append(f"{k}: {v}")
        entry = " ".join(parts)

    # Strip inline comments
    entry = str(entry).split("#")[0].strip()

    for pattern in (_ARROW_RE, _CONNECTS_RE):
        m = pattern.search(entry)
        if m:
            from_port = m.group("from_port")
            to_port = m.group("to_port")
            from_comp = from_port.split(".")[0]
            to_comp = to_port.split(".")[0]
            # Only add if both components are in the active topology
            if from_comp in active_components and to_comp in active_components:
                return ConnectionSpec(from_port=from_port, to_port=to_port)
            return None  # cross-topology reference, skip

    return None  # unparseable entry, skip gracefully
