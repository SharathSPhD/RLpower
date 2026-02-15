"""Connection specifications for sCO₂ cycle Modelica models.

A ConnectionSpec represents a single Modelica connect() equation:
    connect(from_port, to_port);

Port format: "component_name.port_name"
    e.g. "main_compressor.outlet", "recuperator.cold_in"
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConnectionSpec:
    """Immutable specification for a Modelica connect() equation.

    Attributes:
        from_port: Source port in "component.port" format.
        to_port: Destination port in "component.port" format.

    Raises:
        ValueError: If either port string does not contain at least one dot,
            indicating a malformed "component.port" expression.
    """

    from_port: str
    to_port: str

    def __post_init__(self) -> None:
        for port in (self.from_port, self.to_port):
            if port.count(".") < 1:
                raise ValueError(
                    f"Invalid port spec: {port!r}. "
                    f"Expected format 'component_name.port_name' (at least one dot)."
                )

    @property
    def from_component(self) -> str:
        """Extract the component name from from_port."""
        return self.from_port.split(".")[0]

    @property
    def to_component(self) -> str:
        """Extract the component name from to_port."""
        return self.to_port.split(".")[0]
