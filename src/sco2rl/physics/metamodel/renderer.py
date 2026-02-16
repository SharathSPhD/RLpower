"""Renders a CycleModel to a Modelica (.mo) source file.

Output is deterministic given the same CycleModel input (component iteration
order is preserved via dict insertion order, which Python 3.7+ guarantees).

RULE-P3: Every rendered .mo file MUST contain 'enable_BICUBIC=1' in the fluid
medium annotation.  This is critical for stable CoolProp interpolation near the
CO₂ critical point (31.1°C, 7.38 MPa).

Example output structure:

    model SCO2RecuperatedCycle
      import ExternalMedia.Media.CoolPropMedium;

      package Medium = ExternalMedia.Media.CoolPropMedium(
        mediumName="CarbonDioxide",
        libraryName="CoolProp",
        substanceNames={"CarbonDioxide|enable_BICUBIC=1"},
        ThermoStates=Modelica.Media.Interfaces.Choices.IndependentVariables.ph
      );

      // --- Component declarations ---
      SCOPE.Compressors.CentrifugalCompressor main_compressor(
        eta_design=0.88,
        N_design_rpm=3600
      );

    equation
      connect(main_compressor.outlet, recuperator.cold_in);

    end SCO2RecuperatedCycle;
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .builder import CycleModel
from .components import ComponentSpec
from .connections import ConnectionSpec


class MoFileRenderer:
    """Renders a CycleModel to valid Modelica source code.

    All rendering is pure string manipulation — no Jinja2 or templating engine
    is needed for this structural output.
    """

    # Indent used in the Modelica body
    _INDENT = "  "

    def render(self, model: CycleModel) -> str:
        """Return the complete Modelica model source as a string.

        Args:
            model: Fully-assembled CycleModel from SCO2CycleBuilder.build().

        Returns:
            A deterministic, well-formed Modelica model string.
        """
        sections: list[str] = [
            self._render_model_header(model),
            self._render_imports(model),
            self._render_fluid_medium(model),
            self._render_component_declarations(model),
            self._render_equation_section(model),
            self._render_connect_equations(model),
            self._render_model_footer(model),
        ]
        return "\n".join(sections)

    def render_to_file(self, model: CycleModel, output_path: Path) -> Path:
        """Write the rendered .mo source to disk.

        Creates parent directories if they do not exist.

        Args:
            model: Fully-assembled CycleModel.
            output_path: Destination path (should end in .mo).

        Returns:
            The output_path passed in.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = self.render(model)
        output_path.write_text(content, encoding="utf-8")
        return output_path

    # ── Private render helpers ────────────────────────────────────────────────

    def _render_model_header(self, model: CycleModel) -> str:
        """Render the opening 'model Name' line."""
        return f"model {model.package}"

    def _render_imports(self, model: CycleModel) -> str:
        """Render the import statements for external libraries."""
        i = self._INDENT
        lines = [
            f"{i}// --- Imports ---",
            f"{i}import ExternalMedia.Media.CoolPropMedium;",
        ]
        return "\n".join(lines)

    def _render_fluid_medium(self, model: CycleModel) -> str:
        """Render the fluid medium package declaration with enable_BICUBIC=1.

        RULE-P3: 'enable_BICUBIC=1' is injected into the substanceNames
        annotation here, ensuring it appears in every rendered .mo file.
        """
        i = self._INDENT
        fluid = model.fluid_config
        medium_class = fluid.get("medium", "ExternalMedia.Media.CoolPropMedium")
        coolprop_name = fluid.get("coolprop_name", "CarbonDioxide")
        # RULE-P3: always include enable_BICUBIC=1
        coolprop_options = fluid.get("coolprop_options", "enable_BICUBIC=1")
        substance_str = f"{coolprop_name}|{coolprop_options}"

        lines = [
            "",
            f"{i}// --- Fluid medium (RULE-P3: enable_BICUBIC=1 mandatory near CO\u2082 critical point) ---",
            f"{i}package Medium = {medium_class}(",
            f"{i}{i}mediumName=\"{coolprop_name}\",",
            f"{i}{i}libraryName=\"CoolProp\",",
            f"{i}{i}substanceNames={{\" {substance_str}\"}},",
            f"{i}{i}ThermoStates=Modelica.Media.Interfaces.Choices.IndependentVariables.ph",
            f"{i});",
            "",
        ]
        return "\n".join(lines)

    def _render_component_declarations(self, model: CycleModel) -> str:
        """Render Modelica component declarations for all active components."""
        i = self._INDENT
        lines = [f"{i}// --- Component declarations ---"]
        for name, spec in model.components.items():
            lines.append(self._render_single_component(name, spec, i))
        return "\n".join(lines)

    def _render_single_component(
        self,
        name: str,
        spec: ComponentSpec,
        indent: str,
    ) -> str:
        """Render a single Modelica component declaration with parameters."""
        if not spec.params:
            return f"{indent}{spec.modelica_type} {name};"

        param_lines = []
        param_items = list(spec.params.items())
        for idx, (param_name, param_value) in enumerate(param_items):
            comma = "," if idx < len(param_items) - 1 else ""
            param_lines.append(
                f"{indent}{indent}{param_name}={_format_param_value(param_value)}{comma}"
            )

        decl_lines = [f"{indent}{spec.modelica_type} {name}("]
        decl_lines.extend(param_lines)
        decl_lines.append(f"{indent});")
        return "\n".join(decl_lines)

    def _render_equation_section(self, model: CycleModel) -> str:
        """Render the 'equation' keyword that opens the equation section."""
        return "\nequation"

    def _render_connect_equations(self, model: CycleModel) -> str:
        """Render all connect() equations for the model's connections."""
        i = self._INDENT
        if not model.connections:
            return f"{i}// No connections defined"

        lines = [f"{i}// --- Connections ---"]
        for conn in model.connections:
            lines.append(f"{i}connect({conn.from_port}, {conn.to_port});")
        return "\n".join(lines)

    def _render_model_footer(self, model: CycleModel) -> str:
        """Render the closing 'end Name;' line."""
        return f"\nend {model.package};"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _format_param_value(value: Any) -> str:
    """Format a Python parameter value for Modelica syntax.

    - Floats are written with enough precision to avoid rounding loss.
    - Integers are written as integers.
    - Strings are quoted.
    - Booleans use Modelica 'true'/'false'.

    Args:
        value: Python value from the params dict.

    Returns:
        String representation suitable for Modelica parameter assignment.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # Use repr for round-trip fidelity; strip trailing zeros after decimal
        formatted = f"{value:g}"
        return formatted
    if isinstance(value, str):
        return f'"{value}"'
    # Fallback: str representation
    return str(value)
