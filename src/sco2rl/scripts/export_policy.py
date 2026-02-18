from __future__ import annotations

from sco2rl.scripts._runner import run_repo_script


def main() -> None:
    # The repository currently bundles export + validation in this script.
    run_repo_script("cross_validate_and_export.py")
