"""FMUExporter — converts a CycleModel to a compiled FMU via OMPython.

Pipeline:
    CycleModel
        → MoFileRenderer.render()     (Modelica source)
        → OMCSessionWrapper.load_file()
        → translateModelFMU(...)       (OMC)
        → .fmu file on disk

Usage:
    with OMCSessionWrapper() as omc:
        exporter = FMUExporter(config=cfg, output_dir=Path("artifacts/fmu"), omc=omc)
        fmu_path = exporter.export(cycle_model)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from sco2rl.physics.metamodel.builder import CycleModel
from sco2rl.physics.metamodel.renderer import MoFileRenderer
from sco2rl.physics.compiler.omc_session import OMCSessionWrapper


class FMUExportError(Exception):
    """Raised when FMU compilation fails or OMC returns an error."""


class FMUExporter:
    """Converts a CycleModel to a compiled Co-Simulation FMU (FMI 2.0).

    Args:
        config: Parsed FMU export config (from configs/fmu/fmu_export.yaml).
        output_dir: Directory where .mo and .fmu files are written.
        omc: An already-initialized OMCSessionWrapper.
    """

    # FMI constants — never change these without updating RULES.md
    _FMI_VERSION = "2.0"
    _FMU_TYPE = "cs"   # Co-Simulation (not Model Exchange)

    def __init__(
        self,
        config: dict[str, Any],
        output_dir: Path,
        omc: OMCSessionWrapper,
    ) -> None:
        self._config = config
        self._output_dir = Path(output_dir)
        self._omc = omc
        self._renderer = MoFileRenderer()

    def export(self, cycle_model: CycleModel) -> Path:
        """Render .mo source, load it into OMC, and compile to .fmu.

        Args:
            cycle_model: Fully assembled CycleModel from SCO2CycleBuilder.

        Returns:
            Path to the produced .fmu file.

        Raises:
            FMUExportError: If OMC returns an error or empty result.
        """
        # ── Step 1: Create output directory ──────────────────────────────────
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 2: Render .mo to disk ────────────────────────────────────────
        mo_path = self._output_dir / f"{cycle_model.name}.mo"
        mo_content = self._renderer.render(cycle_model)
        mo_path.write_text(mo_content, encoding="utf-8")

        # ── Step 3: Load .mo into OMC ─────────────────────────────────────────
        self._omc.load_file(mo_path)

        # ── Step 4: Translate to FMU ─────────────────────────────────────────
        expr = (
            f'translateModelFMU({cycle_model.name}, '
            f'version="{self._FMI_VERSION}", '
            f'fmuType="{self._FMU_TYPE}")'
        )
        result = self._omc.send(expr)

        # ── Step 5: Validate result ───────────────────────────────────────────
        # OMC returns the FMU filename as a quoted string on success, e.g. '"SCO2_WHR.fmu"'
        # or an empty string / error message on failure.
        fmu_name = result.strip().strip('"')
        if not fmu_name or fmu_name.startswith("Error") or not fmu_name.endswith(".fmu"):
            raise FMUExportError(
                f"FMU export failed for model '{cycle_model.name}'. "
                f"OMC returned: {result!r}. "
                f"Check the Modelica source at {mo_path}."
            )

        # OMC writes the FMU to the working directory; move it to output_dir
        fmu_path = self._output_dir / fmu_name
        # If OMC put it in the CWD instead, move it
        cwd_fmu = Path(fmu_name)
        if cwd_fmu.exists() and not fmu_path.exists():
            cwd_fmu.rename(fmu_path)

        return fmu_path
