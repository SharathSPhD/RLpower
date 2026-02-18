#!/usr/bin/env python3
"""Generate sCO₂ cycle engineering schematic and T-s diagram.

Outputs:
    paper/figures/cycle_diagram.png     — clockwise engineering schematic
    paper/figures/cycle_ts_diagram.png  — CO₂ T-s diagram with saturation dome

Usage:
    python scripts/generate_cycle_diagram.py [CONFIG_YAML]

If CONFIG_YAML is not supplied, defaults to configs/model/base_cycle.yaml.
"""

import sys
from pathlib import Path

import yaml


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/model/base_cycle.yaml")

    cfg = yaml.safe_load(config_path.read_text())

    # Lazy import so that the script fails loudly if PYTHONPATH is wrong
    from sco2rl.physics.metamodel.builder import SCO2CycleBuilder
    from sco2rl.physics.metamodel.diagram_renderer import CycleDiagramRenderer

    model = SCO2CycleBuilder.from_config(cfg).build()

    output_dir = Path("paper/figures")
    renderer = CycleDiagramRenderer()
    sch_path, ts_path = renderer.render(model, output_dir, dpi=300)

    print(f"Schematic:  {sch_path}")
    print(f"T-s diagram: {ts_path}")


if __name__ == "__main__":
    main()
