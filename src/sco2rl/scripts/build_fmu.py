from __future__ import annotations

from sco2rl.scripts._runner import run_repo_script


def main() -> None:
    # The repository currently exposes FMU export through this script.
    run_repo_script("export_controlled_fmu.py")
