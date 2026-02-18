from __future__ import annotations

import json
from pathlib import Path

import pytest

from sco2rl.simulation.fmu.mock_fmu import MockFMU


GOLDEN_PATH = Path(__file__).parent / "golden" / "mock_fmu_step.json"


def test_mock_fmu_single_step_matches_golden_outputs() -> None:
    golden = json.loads(GOLDEN_PATH.read_text())
    obs_vars = ["T_compressor_inlet", "W_turbine", "W_main_compressor", "W_net"]
    action_vars = list(golden["inputs"].keys())
    fmu = MockFMU(
        obs_vars=obs_vars,
        action_vars=action_vars,
        design_point={
            "T_compressor_inlet": 33.0,
            "W_turbine": 14.5,
            "W_main_compressor": 4.0,
            "W_net": 10.0,
        },
        seed=42,
    )
    fmu.initialize(start_time=0.0, stop_time=10.0, step_size=5.0)
    fmu.set_inputs({k: float(v) for k, v in golden["inputs"].items()})
    assert fmu.do_step(current_time=0.0, step_size=5.0) is True
    outputs = fmu.get_outputs()
    for key, expected in golden["expected_outputs"].items():
        assert outputs[key] == pytest.approx(expected, abs=1e-6)
