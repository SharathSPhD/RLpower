"""Helper to execute repository CLI scripts from package entrypoints."""

from __future__ import annotations

import runpy
from pathlib import Path


def run_repo_script(script_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    runpy.run_path(str(script_path), run_name="__main__")
