"""FMPyAdapter -- wraps fmpy FMU2Slave to implement FMUInterface.

Production adapter: drives real .fmu files via fmpy.
RULE-C1: This class MUST NOT be used in unit tests -- use MockFMU instead.

Key design decisions:
- reset() calls freeInstance() + re-instantiate (NOT fmi2SetFMUState -- ADR:
  non-deterministic with CoolProp near CO2 critical point).
- set_inputs() raises KeyError on unknown variable names (fail-fast).
- do_step() returns False on fmi2Error / fmi2Fatal (CVODE solver failure).
- Variable name -> value reference mapping built once at initialize() time.
"""
from __future__ import annotations

import numpy as np

from sco2rl.simulation.fmu.interface import FMUInterface


class FMPyAdapter(FMUInterface):
    """fmpy-based FMU adapter for production training.

    Parameters
    ----------
    fmu_path:
        Path to the .fmu file.
    obs_vars:
        Variable names to read as observations (in order).
    action_vars:
        Variable names to write as inputs.
    instance_name:
        FMU instance name string.
    """

    def __init__(
        self,
        fmu_path: str,
        obs_vars: list[str],
        action_vars: list[str],
        instance_name: str = "sco2_instance",
    ) -> None:
        self._fmu_path = fmu_path
        self._obs_vars: list[str] = list(obs_vars)
        self._action_vars: list[str] = list(action_vars)
        self._instance_name = instance_name

        # Populated during initialize()
        self._fmu = None
        self._unzip_dir: str | None = None
        self._model_description = None
        # name -> valueReference mapping
        self._vr_map: dict[str, int] = {}
        self._obs_vrs: list[int] = []
        self._action_vrs: list[int] = []

        # Saved for re-instantiation on reset()
        self._start_time: float = 0.0
        self._stop_time: float = 3600.0
        self._step_size: float = 5.0
        self._initialized: bool = False

    # -- FMUInterface implementation --------------------------------------------

    def initialize(self, start_time: float, stop_time: float, step_size: float) -> None:
        """Extract FMU, read model description, instantiate and enter init mode."""
        import fmpy
        from fmpy.fmi2 import FMU2Slave

        self._start_time = start_time
        self._stop_time = stop_time
        self._step_size = step_size

        # Extract FMU archive (fmpy extracts to a temp dir)
        if self._unzip_dir is None:
            self._unzip_dir = fmpy.extract(self._fmu_path)

        # Read model description once; build value-reference map
        if self._model_description is None:
            self._model_description = fmpy.read_model_description(
                self._fmu_path, validate=False
            )
            self._vr_map = {
                v.name: v.valueReference
                for v in self._model_description.modelVariables
            }
            # Validate all requested variables exist in the model
            for name in self._obs_vars + self._action_vars:
                if name not in self._vr_map:
                    raise KeyError(
                        f"Variable '{name}' not found in FMU model description."
                    )
            self._obs_vrs = [self._vr_map[n] for n in self._obs_vars]
            self._action_vrs = [self._vr_map[n] for n in self._action_vars]

        # Instantiate a new FMU Co-Simulation slave
        self._fmu = FMU2Slave(
            instanceName=self._instance_name,
            guid=self._model_description.guid,
            modelIdentifier=self._model_description.coSimulation.modelIdentifier,
            unzipDirectory=self._unzip_dir,
        )
        self._fmu.instantiate()
        self._fmu.setupExperiment(startTime=start_time, stopTime=stop_time)
        self._fmu.enterInitializationMode()
        self._fmu.exitInitializationMode()
        self._initialized = True

    def set_inputs(self, inputs: dict[str, float]) -> None:
        """Write action values into the FMU.

        Raises
        ------
        KeyError
            If any key in inputs is not in the FMU variable catalogue.
        """
        for name in inputs:
            if name not in self._vr_map:
                raise KeyError(
                    f"Unknown variable '{name}'. "
                    f"Valid action vars: {self._action_vars}"
                )
        for name, value in inputs.items():
            vr = self._vr_map[name]
            self._fmu.setReal([vr], [float(value)])

    def do_step(self, current_time: float, step_size: float) -> bool:
        """Advance simulation by step_size seconds.

        Returns True on success, False on fmi2Error or fmi2Fatal.
        """
        from fmpy.fmi2 import fmi2Error, fmi2Fatal
        try:
            status = self._fmu.fmi2DoStep(
                self._fmu.component,
                current_time,
                step_size,
                True,  # noSetFMUStatePriorToCurrentPoint
            )
            return status not in (fmi2Error, fmi2Fatal)
        except Exception:
            return False

    def get_outputs(self) -> dict[str, float]:
        """Return current output variables as name -> value mapping."""
        values = self._fmu.getReal(self._obs_vrs)
        return {name: float(val) for name, val in zip(self._obs_vars, values)}

    def get_outputs_as_array(self) -> np.ndarray:
        """Return outputs in obs_vars order as a float32 numpy array."""
        values = self._fmu.getReal(self._obs_vrs)
        return np.array(values, dtype=np.float32)

    def reset(self) -> None:
        """Re-initialise FMU by freeInstance() + re-instantiate.

        ADR: fmi2SetFMUState is NOT used -- non-deterministic with CoolProp
        near the CO2 critical point.
        """
        if self._fmu is not None:
            try:
                self._fmu.terminate()
            except Exception:
                pass
            try:
                self._fmu.freeInstance()
            except Exception:
                pass
            self._fmu = None

        self._initialized = False
        # Re-instantiate using saved parameters
        self.initialize(self._start_time, self._stop_time, self._step_size)

    def close(self) -> None:
        """Release FMU resources (terminate + freeInstance)."""
        if self._fmu is not None:
            try:
                self._fmu.terminate()
            except Exception:
                pass
            try:
                self._fmu.freeInstance()
            except Exception:
                pass
            self._fmu = None
        self._initialized = False
