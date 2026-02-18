"""CycleDiagramRenderer — matplotlib-based sCO₂ cycle schematic and T-s diagram.

Produces two publication-quality figures:
  1. Engineering schematic: component boxes with flow-direction arrows and
     state-point labels, laid out as a clockwise thermodynamic loop.
  2. CO₂ T-s diagram: saturation dome (via CoolProp) with six reference
     state points and connecting isobar segments.

No OMPython/OMEdit required: matplotlib + CoolProp (already pinned in the
project) are sufficient.  The layout is hardcoded for the simple_recuperated
topology because a deliberate loop shape communicates the cycle better than
force-directed auto-layout.

Usage
-----
    from sco2rl.physics.metamodel.builder import SCO2CycleBuilder
    from sco2rl.physics.metamodel.diagram_renderer import CycleDiagramRenderer
    import yaml
    from pathlib import Path

    cfg  = yaml.safe_load(Path("configs/model/base_cycle.yaml").read_text())
    model = SCO2CycleBuilder.from_config(cfg).build()
    renderer = CycleDiagramRenderer()
    schematic_path, ts_path = renderer.render(model, Path("paper/figures/"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np


# ── Layout coordinates (normalised axes, 0–1) ─────────────────────────────────
# Clockwise loop: heat source (regulator/PCMHeater) at top, turbine right,
# recuperator bottom-right, precooler bottom-left, compressor left.
_LAYOUT: dict[str, tuple[float, float]] = {
    "regulator":       (0.50, 0.82),
    "turbine":         (0.82, 0.50),
    "recuperator":     (0.58, 0.18),
    "precooler":       (0.28, 0.18),
    "main_compressor": (0.12, 0.50),
    # Fallback for any unlisted component
    "_default":        (0.50, 0.50),
}

# ── Human-readable labels ─────────────────────────────────────────────────────
_LABEL: dict[str, str] = {
    "Steps.Components.Regulator":  "Heat Source\n(EAF/BOF)",
    "Steps.Components.PCMHeater":  "Heat Source\n(EAF/BOF)",
    "Steps.Components.Turbine":    "Turbine",
    "Steps.Components.Recuperator": "Recuperator",
    "Steps.Components.FanCooler":  "Pre-Cooler",
    "Steps.Components.Pump":       "Main\nCompressor",
    "Steps.Components.Valve":      "Valve",
    "Steps.Components.Splitter":   "Splitter",
}

# ── Box fill colours ──────────────────────────────────────────────────────────
_COLOR: dict[str, str] = {
    "Steps.Components.Regulator":   "#FF8C42",   # orange – heat source
    "Steps.Components.PCMHeater":   "#FF8C42",
    "Steps.Components.Turbine":     "#E74C3C",   # red – hot high-enthalpy
    "Steps.Components.Recuperator": "#9B59B6",   # purple – heat exchanger
    "Steps.Components.FanCooler":   "#2980B9",   # blue – cold side
    "Steps.Components.Pump":        "#27AE60",   # green – work input
    "Steps.Components.Valve":       "#95A5A6",
    "Steps.Components.Splitter":    "#BDC3C7",
}
_DEFAULT_COLOR = "#BDC3C7"

# ── State point labels (between components, anticlockwise numbering) ──────────
# Placed at the midpoint of each flow arrow
_STATE_POINTS: list[tuple[str, str, str]] = [
    # (from_component, to_component, label)
    ("regulator",       "turbine",         "[3] T3~770K\nP3~20MPa"),
    ("turbine",         "recuperator",     "[4] T4~590K\nP4~7.6MPa"),
    ("recuperator",     "precooler",       "[5] T5~330K\nP5~7.6MPa"),
    ("precooler",       "main_compressor", "[1] T1~306K\nP1~7.6MPa"),
    ("main_compressor", "recuperator",     "[2] T2~320K\nP2~20MPa"),
    ("recuperator",     "regulator",       "[2r] T2r~680K\nP2r~20MPa"),
]

# ── RL actuator annotation positions ─────────────────────────────────────────
_ACTUATOR_ANNOTATIONS: list[tuple[float, float, str]] = [
    (0.50, 0.65, "[A1] bypass valve"),
    (0.82, 0.75, "[A2] IGV angle"),
    (0.50, 0.95, "[A3] inventory valve"),
    (0.20, 0.35, "[A4] cooling flow"),
    (0.35, 0.50, "[A5] split ratio"),
]


class CycleDiagramRenderer:
    """Renders the sCO₂ recuperated cycle as an engineering schematic and T-s diagram."""

    # ── Public API ─────────────────────────────────────────────────────────────

    def render(
        self,
        model: Any,
        output_dir: Path,
        state_points: list[dict] | None = None,
        dpi: int = 300,
    ) -> tuple[Path, Path]:
        """Render both figures.  Returns (schematic_path, ts_path)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sch_path = self.render_schematic(model, output_dir / "cycle_diagram.png", dpi=dpi)
        ts_path = self.render_ts_diagram(output_dir / "cycle_ts_diagram.png",
                                         state_points=state_points, dpi=dpi)
        return sch_path, ts_path

    def render_schematic(self, model: Any, path: Path, dpi: int = 300) -> Path:
        """Draw the clockwise engineering schematic."""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor("#F8F9FA")
        fig.patch.set_facecolor("#F8F9FA")

        components = model.components
        connections = model.connections

        # Draw flow arrows first (behind boxes)
        self._draw_flow_arrows(ax, components, connections)

        # Draw component boxes
        box_dims: dict[str, tuple[float, float, float, float]] = {}
        for name, spec in components.items():
            pos = _LAYOUT.get(name, _LAYOUT["_default"])
            color = _COLOR.get(spec.modelica_type, _DEFAULT_COLOR)
            label = _LABEL.get(spec.modelica_type, name)
            bx, by, bw, bh = self._draw_component_box(ax, pos, label, color)
            box_dims[name] = (bx, by, bw, bh)

        # Draw state point labels on arrows
        self._draw_state_point_labels(ax, components)

        # RL actuator legend
        self._draw_actuator_legend(ax)

        # Title and source annotation
        ax.set_title(
            "sCO₂ Simple Recuperated Brayton Cycle — WHR System Architecture\n"
            "Heat source: EAF/BOF exhaust (200–1200°C); Net output: 10 MWe",
            fontsize=11, fontweight="bold", pad=14,
        )

        # Critical point annotation (compressor inlet constraint)
        ax.annotate(
            "Critical constraint:\nT_in > 33°C (306 K)\n1–2°C above CO₂ critical point",
            xy=_LAYOUT["precooler"],
            xytext=(0.02, 0.03),
            fontsize=7.5,
            color="#C0392B",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FADBD8", alpha=0.8),
        )

        plt.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return Path(path)

    def render_ts_diagram(
        self,
        path: Path,
        state_points: list[dict] | None = None,
        dpi: int = 300,
    ) -> Path:
        """Draw CO₂ T-s diagram with saturation dome and reference state points."""
        try:
            from CoolProp.CoolProp import PropsSI
        except ImportError:
            raise RuntimeError(
                "CoolProp is required for T-s diagram. "
                "Install with: pip install CoolProp"
            )

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_facecolor("#F8F9FA")
        fig.patch.set_facecolor("#F8F9FA")

        # ── Saturation dome ──────────────────────────────────────────────────
        T_crit = PropsSI("Tcrit", "CO2")   # 304.13 K
        P_crit = PropsSI("Pcrit", "CO2")   # 7.377e6 Pa

        T_sat = np.linspace(220.0, T_crit - 0.01, 200)
        s_liq, s_vap = [], []
        for T in T_sat:
            try:
                s_liq.append(PropsSI("S", "T", T, "Q", 0, "CO2") / 1000)
                s_vap.append(PropsSI("S", "T", T, "Q", 1, "CO2") / 1000)
            except Exception:
                s_liq.append(np.nan)
                s_vap.append(np.nan)

        s_crit = PropsSI("S", "T", T_crit, "Q", 0, "CO2") / 1000
        T_plot = T_sat - 273.15  # K → °C

        ax.plot(s_liq, T_plot, "b-", linewidth=2.0, label="Saturation dome (CO₂)")
        ax.plot(s_vap, T_plot, "b-", linewidth=2.0)
        ax.plot(s_crit, T_crit - 273.15, "b^", markersize=8, label=f"Critical point ({T_crit - 273.15:.1f}°C, {P_crit/1e6:.2f} MPa)")
        ax.fill_betweenx(T_plot, s_liq, s_vap, alpha=0.07, color="blue")

        # ── Reference state points ───────────────────────────────────────────
        if state_points is None:
            state_points = self._default_state_points()

        T_sp = np.array([sp["T_C"] for sp in state_points])
        s_sp = np.array([sp["s_kJ_kgK"] for sp in state_points])
        # Use plain ASCII labels for the T-s diagram
        labels_sp = [
            "[3] Turbine in",
            "[4] Turbine out",
            "[5] Recup hot out",
            "[1] Comp in",
            "[2] Comp out",
            "[2r] Recup cold out",
        ]
        colors_sp = ["#E74C3C", "#E67E22", "#9B59B6", "#2980B9", "#27AE60", "#F39C12"]

        # Connect state points with isobar-approximate lines (cycle path)
        cycle_order = [0, 1, 2, 3, 4, 5, 0]  # wrap back to start
        for i in range(len(cycle_order) - 1):
            a, b = cycle_order[i], cycle_order[i + 1]
            ax.plot([s_sp[a], s_sp[b]], [T_sp[a], T_sp[b]],
                    "-", color="#666666", linewidth=1.0, alpha=0.6, zorder=1)

        for i, (s, T, lbl, color) in enumerate(zip(s_sp, T_sp, labels_sp, colors_sp)):
            ax.scatter(s, T, color=color, s=80, zorder=5, edgecolors="white", linewidths=0.8)
            ax.annotate(
                lbl,
                xy=(s, T),
                xytext=(s + 0.04, T + 6),
                fontsize=8,
                color=color,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.5),
            )

        # Isobar lines at P_high=20MPa and P_low=7.6MPa
        for P_iso, lstyle, P_label in [
            (20e6, "--", "P_high = 20 MPa"),
            (7.6e6, ":", "P_low = 7.6 MPa"),
        ]:
            T_iso = np.linspace(280, 830, 300)
            s_iso = []
            for T in T_iso:
                try:
                    s_iso.append(PropsSI("S", "T", T, "P", P_iso, "CO2") / 1000)
                except Exception:
                    s_iso.append(np.nan)
            ax.plot(s_iso, T_iso - 273.15, lstyle, color="#999999",
                    linewidth=0.8, alpha=0.7, label=P_label)

        # Critical temperature line
        ax.axhline(T_crit - 273.15, color="#2980B9", linestyle="-.",
                   linewidth=0.8, alpha=0.5, label=f"T_crit = {T_crit - 273.15:.1f}°C")

        # Compressor inlet constraint
        T_min_inlet = 33.0  # °C, must stay above this
        ax.axhline(T_min_inlet, color="#E74C3C", linestyle=":", linewidth=1.2, alpha=0.7)
        ax.text(0.52, T_min_inlet + 2, "Min. compressor inlet T = 33°C",
                fontsize=7.5, color="#E74C3C")

        ax.set_xlabel("Specific entropy, s [kJ/(kg·K)]", fontsize=11)
        ax.set_ylabel("Temperature, T [°C]", fontsize=11)
        ax.set_title(
            "CO₂ Temperature-Entropy (T-s) Diagram\n"
            "Simple Recuperated Brayton Cycle — Reference Operating Points",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlim(0.4, 2.2)
        ax.set_ylim(-60, 560)
        ax.legend(loc="upper left", fontsize=8.5)
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return Path(path)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _draw_component_box(
        self,
        ax: plt.Axes,
        pos: tuple[float, float],
        label: str,
        color: str,
        w: float = 0.13,
        h: float = 0.10,
    ) -> tuple[float, float, float, float]:
        """Draw a FancyBboxPatch box and return (x, y, w, h) of bounding box."""
        cx, cy = pos
        x, y = cx - w / 2, cy - h / 2
        box = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.01",
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
            alpha=0.88,
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(
            cx, cy, label,
            ha="center", va="center",
            fontsize=8.5, fontweight="bold",
            color="white",
            zorder=4,
            multialignment="center",
            path_effects=[pe.withStroke(linewidth=0.5, foreground="black")],
        )
        return x, y, w, h

    def _draw_flow_arrows(self, ax: plt.Axes, components: dict, connections: list) -> None:
        """Draw arrows between component positions."""
        comp_set = set(components.keys())
        for conn in connections:
            from_name = conn.from_port.split(".")[0]
            to_name = conn.to_port.split(".")[0]
            if from_name not in comp_set or to_name not in comp_set:
                continue
            fx, fy = _LAYOUT.get(from_name, _LAYOUT["_default"])
            tx, ty = _LAYOUT.get(to_name, _LAYOUT["_default"])
            # Offset start/end points to edge of box (approx 0.065 from centre)
            dx, dy = tx - fx, ty - fy
            dist = max((dx**2 + dy**2) ** 0.5, 1e-6)
            offset = 0.07
            ax.annotate(
                "",
                xy=(tx - dx / dist * offset, ty - dy / dist * offset),
                xytext=(fx + dx / dist * offset, fy + dy / dist * offset),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#444444",
                    lw=1.4,
                    mutation_scale=14,
                ),
                zorder=2,
            )

    def _draw_state_point_labels(self, ax: plt.Axes, components: dict) -> None:
        """Place state-point labels at arrow midpoints."""
        for from_name, to_name, lbl in _STATE_POINTS:
            if from_name not in components or to_name not in components:
                continue
            fx, fy = _LAYOUT.get(from_name, _LAYOUT["_default"])
            tx, ty = _LAYOUT.get(to_name, _LAYOUT["_default"])
            mx, my = (fx + tx) / 2, (fy + ty) / 2
            ax.text(
                mx, my, lbl,
                ha="center", va="center",
                fontsize=6.8,
                color="#2C3E50",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#BDC3C7", alpha=0.85),
                zorder=5,
            )

    def _draw_actuator_legend(self, ax: plt.Axes) -> None:
        """Add RL actuator legend box in top-right corner."""
        legend_text = (
            "RL Actuators (action space, dim=5)\n"
            "[A1] Bypass valve position [0-1]\n"
            "[A2] IGV angle [deg]\n"
            "[A3] Inventory valve [0-1]\n"
            "[A4] Cooling flow rate [0-1]\n"
            "[A5] Recompression split ratio [0-1]"
        )
        ax.text(
            0.98, 0.98,
            legend_text,
            transform=ax.transAxes,
            fontsize=7,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF5FB",
                      edgecolor="#2980B9", alpha=0.9),
            family="monospace",
        )

    @staticmethod
    def _default_state_points() -> list[dict]:
        """Reference state points for simple recuperated cycle at design point."""
        try:
            from CoolProp.CoolProp import PropsSI

            def s_kj(T_C: float, P_Pa: float) -> float:
                try:
                    return PropsSI("S", "T", T_C + 273.15, "P", P_Pa, "CO2") / 1000
                except Exception:
                    return float("nan")

            P_hi = 20.0e6  # Pa
            P_lo = 7.60e6  # Pa
            return [
                {"label": "[3]", "T_C": 497.0, "s_kJ_kgK": s_kj(497.0, P_hi)},
                {"label": "[4]", "T_C": 317.0, "s_kJ_kgK": s_kj(317.0, P_lo)},
                {"label": "[5]", "T_C":  57.0, "s_kJ_kgK": s_kj(57.0, P_lo)},
                {"label": "[1]", "T_C":  33.0, "s_kJ_kgK": s_kj(33.0, P_lo)},
                {"label": "[2]", "T_C":  47.0, "s_kJ_kgK": s_kj(47.0, P_hi)},
                {"label": "[2r]", "T_C": 407.0, "s_kJ_kgK": s_kj(407.0, P_hi)},
            ]
        except ImportError:
            return [
                {"label": "[3]", "T_C": 497.0, "s_kJ_kgK": 1.62},
                {"label": "[4]", "T_C": 317.0, "s_kJ_kgK": 1.75},
                {"label": "[5]", "T_C":  57.0, "s_kJ_kgK": 1.40},
                {"label": "[1]", "T_C":  33.0, "s_kJ_kgK": 1.17},
                {"label": "[2]", "T_C":  47.0, "s_kJ_kgK": 1.18},
                {"label": "[2r]", "T_C": 407.0, "s_kJ_kgK": 1.60},
            ]
