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
            self._render_controller_wrapper_models(model),
            self._render_controller_interface_declarations(model),
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
        """Render the 'within' declaration and opening 'model Name' line.

        SCOPE components live in the Steps.* package hierarchy; the generated
        model must declare itself 'within Steps.Cycle' so OMPython can resolve
        all Steps.Components.* references without extra import statements.
        """
        return f"within Steps.Cycle;\n\nmodel {model.package}"

    def _render_imports(self, model: CycleModel) -> str:
        """Render the import statements for external libraries.

        SCOPE components use Steps.* namespace; all SI types are resolved
        within the SCOPE package itself. No Modelica.SIunits imports are
        needed in the wrapper model (avoids Modelica 3.x/4.x compatibility issues).
        """
        i = self._INDENT
        lines = [
            f"{i}// --- Steps library (SCOPE) ---",
            f"{i}// Components: Steps.Components.*   Fluid: Steps.Media.SCO2",
        ]
        return "\n".join(lines)

    def _render_fluid_medium(self, model: CycleModel) -> str:
        """Render the CoolProp medium note.

        RULE-P3: enable_BICUBIC=1 is required near the CO₂ critical point.
        In the SCOPE library, CoolProp is accessed via Steps.Utilities.CoolProp
        (backed by libMyProps.so — the BICUBIC flag is compiled into that library).
        The ExternalMedia package is NOT used by SCOPE components; they inherit
        PBMedia = Steps.Media.SCO2 from TwoPorts.
        """
        i = self._INDENT
        lines = [
            "",
            f"{i}// --- Fluid medium ---",
            f"{i}// CoolProp backend: Steps.Utilities.CoolProp (libMyProps.so)",
            f"{i}// RULE-P3: enable_BICUBIC=1 is enforced inside libMyProps.so",
            f"{i}// for stable interpolation near CO2 critical point (31.1\u00b0C, 7.38 MPa).",
            f"{i}// PBMedia = Steps.Media.SCO2 is inherited by all TwoPorts components.",
            "",
        ]
        return "\n".join(lines)

    def _render_component_declarations(self, model: CycleModel) -> str:
        """Render Modelica component declarations for all active components."""
        i = self._INDENT
        lines = [f"{i}// --- Component declarations ---"]
        for name, spec in model.components.items():
            lines.append(self._render_single_component(name, spec, i, model))
        return "\n".join(lines)

    def _render_single_component(
        self,
        name: str,
        spec: ComponentSpec,
        indent: str,
        model: CycleModel,
    ) -> str:
        """Render a single Modelica component declaration with parameters."""
        modelica_type = self._component_type_for_render(name, spec, model)
        if not spec.params:
            return f"{indent}{modelica_type} {name};"

        param_lines = []
        param_items = list(spec.params.items())
        for idx, (param_name, param_value) in enumerate(param_items):
            comma = "," if idx < len(param_items) - 1 else ""
            param_lines.append(
                f"{indent}{indent}{param_name}={_format_param_value(param_value)}{comma}"
            )

        decl_lines = [f"{indent}{modelica_type} {name}("]
        decl_lines.extend(param_lines)
        decl_lines.append(f"{indent});")
        return "\n".join(decl_lines)

    def _render_equation_section(self, model: CycleModel) -> str:
        """Render the 'equation' keyword that opens the equation section."""
        return "\nequation"

    def _render_connect_equations(self, model: CycleModel) -> str:
        """Render all connect() equations for the model's connections."""
        i = self._INDENT
        lines = [f"{i}// --- Connections ---"]
        for conn in model.connections:
            lines.append(f"{i}connect({conn.from_port}, {conn.to_port});")
        control_lines = self._render_control_connections(model)
        lines.extend(control_lines)
        if len(lines) == 1:
            lines.append(f"{i}// No connections defined")
        return "\n".join(lines)

    def _render_controller_wrapper_models(self, model: CycleModel) -> str:
        """Render lightweight controller-aware wrapper component definitions."""
        mode = model.controller_config.get("mode", "none")
        if mode not in {"rl_external", "pid_baseline"}:
            return ""

        i = self._INDENT
        return "\n".join([
            "",
            f"{i}// --- Controller/actuator wrappers ---",
            f"{i}model ControlledRegulator",
            f"{i}{i}extends Steps.Components.Regulator;",
            f"{i}{i}Modelica.Blocks.Interfaces.RealInput T_set;",
            f"{i}{i}Modelica.Blocks.Interfaces.RealInput mdot_set;",
            f"{i}{i}equation",
            f"{i}{i}{i}T_init = T_set;",
            f"{i}{i}{i}m_flow_init = mdot_set;",
            f"{i}end ControlledRegulator;",
            "",
            f"{i}model ControlledTurbine",
            f"{i}{i}extends Steps.Components.Turbine;",
            f"{i}{i}Modelica.Blocks.Interfaces.RealInput p_out_set;",
            f"{i}{i}equation",
            f"{i}{i}{i}p_out = p_out_set;",
            f"{i}end ControlledTurbine;",
            "",
            f"{i}model ControlledFanCooler",
            f"{i}{i}extends Steps.Components.FanCooler;",
            f"{i}{i}Modelica.Blocks.Interfaces.RealInput T_out_set;",
            f"{i}{i}equation",
            f"{i}{i}{i}T_output = T_out_set;",
            f"{i}end ControlledFanCooler;",
            "",
        ])

    def _render_controller_interface_declarations(self, model: CycleModel) -> str:
        """Declare controller I/O interfaces for swappable control policies."""
        mode = model.controller_config.get("mode", "none")
        if mode not in {"rl_external", "pid_baseline"}:
            return ""

        i = self._INDENT
        n_commands = int(model.controller_config.get("n_commands", 4))
        n_measurements = int(model.controller_config.get("n_measurements", 14))
        return "\n".join([
            f"{i}// --- Controller interface ({mode}) ---",
            f"{i}Modelica.Blocks.Interfaces.RealInput commands[{n_commands}];",
            f"{i}Modelica.Blocks.Interfaces.RealOutput measurements[{n_measurements}];",
            "",
        ])

    def _component_type_for_render(
        self,
        name: str,
        spec: ComponentSpec,
        model: CycleModel,
    ) -> str:
        """Swap core components for controlled wrappers when enabled."""
        mode = model.controller_config.get("mode", "none")
        if mode not in {"rl_external", "pid_baseline"}:
            return spec.modelica_type

        wrapper_map = {
            ("regulator", "Steps.Components.Regulator"): "ControlledRegulator",
            ("turbine", "Steps.Components.Turbine"): "ControlledTurbine",
            ("precooler", "Steps.Components.FanCooler"): "ControlledFanCooler",
        }
        return wrapper_map.get((name, spec.modelica_type), spec.modelica_type)

    def _render_control_connections(self, model: CycleModel) -> list[str]:
        """Connect top-level controller commands to actuator wrapper inputs."""
        mode = model.controller_config.get("mode", "none")
        if mode not in {"rl_external", "pid_baseline"}:
            return []

        i = self._INDENT
        lines: list[str] = [f"{i}// --- Control links ---"]
        if "regulator" in model.components:
            lines.append(f"{i}connect(commands[1], regulator.T_set);")
            lines.append(f"{i}connect(commands[2], regulator.mdot_set);")
        if "turbine" in model.components:
            lines.append(f"{i}connect(commands[3], turbine.p_out_set);")
        if "precooler" in model.components:
            lines.append(f"{i}connect(commands[4], precooler.T_out_set);")

        n_measurements = int(model.controller_config.get("n_measurements", 14))
        if n_measurements > 0:
            lines.append(f"{i}// Measurement wiring (simple_recuperated default order)")
            measurement_exprs = self._measurement_expressions(model)
            for idx in range(1, n_measurements + 1):
                expr = measurement_exprs[idx - 1] if idx - 1 < len(measurement_exprs) else "0.0"
                lines.append(f"{i}measurements[{idx}] = {expr};")
        return lines

    @staticmethod
    def _measurement_expressions(model: CycleModel) -> list[str]:
        """Return top-level measurement expressions for controller output vector."""
        has = set(model.components.keys())

        def _maybe(component: str, expr: str, fallback: str = "0.0") -> str:
            return expr if component in has else fallback

        return [
            _maybe("main_compressor", "main_compressor.T_inlet_rt"),
            _maybe("main_compressor", "main_compressor.T_outlet_rt"),
            _maybe("turbine", "turbine.T_inlet_rt"),
            _maybe("turbine", "turbine.T_outlet_rt"),
            _maybe("recuperator", "recuperator.T_hot_in"),
            _maybe("recuperator", "recuperator.T_cold_in"),
            _maybe("precooler", "precooler.T_inlet_rt"),
            _maybe("precooler", "precooler.T_outlet_rt"),
            _maybe("turbine", "turbine.W_turbine"),
            _maybe("main_compressor", "main_compressor.W_comp"),
            _maybe("main_compressor", "main_compressor.eta"),
            _maybe("recuperator", "recuperator.eta"),
            _maybe("recuperator", "recuperator.Q_actual"),
            _maybe("main_compressor", "main_compressor.p_outlet"),
        ]

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
