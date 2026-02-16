"""Shared fixtures and pytest configuration for all sco2rl tests.

All tests run inside Docker on DGX Spark (ARM64). See RULE-D1, RULE-D2.
SeedManager.set_all(42) is called in the autouse `seed` fixture to guarantee
deterministic behaviour for every unit test.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from sco2rl.utils.seeds import SeedManager, GLOBAL_SEED

PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


# ---------------------------------------------------------------------------
# Autouse seed fixture — ensures GLOBAL_SEED=42 for every unit test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def seed() -> None:
    """Set all random seeds to GLOBAL_SEED (42) before each test."""
    SeedManager.set_all(GLOBAL_SEED)


# ---------------------------------------------------------------------------
# Path fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def configs_dir() -> Path:
    return CONFIGS_DIR


@pytest.fixture
def model_config_path() -> Path:
    return CONFIGS_DIR / "model" / "base_cycle.yaml"


@pytest.fixture
def env_config_path() -> Path:
    return CONFIGS_DIR / "environment" / "env.yaml"


@pytest.fixture
def safety_config_path() -> Path:
    return CONFIGS_DIR / "safety" / "constraints.yaml"


@pytest.fixture
def fmu_config_path() -> Path:
    return CONFIGS_DIR / "fmu" / "fmu_export.yaml"


@pytest.fixture
def curriculum_config_path() -> Path:
    return CONFIGS_DIR / "curriculum" / "curriculum.yaml"


# ---------------------------------------------------------------------------
# Pytest mark registration and integration-test gating
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "integration: requires --run-integration and compiled FMU"
    )
    config.addinivalue_line("markers", "slow: tests that take > 60 seconds")
    config.addinivalue_line("markers", "gpu: tests that require CUDA GPU")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not config.getoption("--run-integration", default=False):
        skip_integration = pytest.mark.skip(reason="need --run-integration flag")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests (requires compiled FMU at artifacts/fmu/SCO2_WHR.fmu)",
    )
