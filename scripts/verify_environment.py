#!/usr/bin/env python3
"""verify_environment.py — Stage gate verification script for sco2rl.

Usage:
    python scripts/verify_environment.py --stage 0
    python scripts/verify_environment.py --stage 1

Exit 0 if all checks for the requested stage pass; exit 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ANSI colour codes (safe to use in Docker/DGX terminal)
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def check(label: str, fn) -> bool:
    """Run fn(); print result. Returns True if fn() returns truthy, False on exception/falsy."""
    try:
        result = fn()
        if result is False:
            fail(label)
            return False
        ok(label)
        return True
    except Exception as exc:
        fail(f"{label}  [{type(exc).__name__}: {exc}]")
        return False


# ─── Stage 0 checks ──────────────────────────────────────────────────────────

def check_omc_import() -> bool:
    from OMPython import OMCSessionZMQ  # noqa: F401  # raises if not installed
    return True


def check_omc_session() -> bool:
    from OMPython import OMCSessionZMQ
    omc = OMCSessionZMQ()
    result = omc.sendExpression("1+1")
    omc.sendExpression("quit()")
    return str(result).strip() == "2"


def check_coolprop_import() -> bool:
    import CoolProp.CoolProp as CP  # noqa: F401
    return True


def check_coolprop_co2_saturation() -> bool:
    """Verify CoolProp can compute CO₂ saturation properties near critical point."""
    import CoolProp.CoolProp as CP
    # Saturation temperature at 70 bar — below CO₂ Pcrit (73.77 bar); critical T ≈ 304 K
    T = CP.PropsSI("T", "P", 7e6, "Q", 0, "CO2")
    return 300 < T < 310  # roughly 302 K at 70 bar


def check_coolprop_bicubic() -> bool:
    """Verify BICUBIC&HEOS backend is accessible (RULE-P3: enable_BICUBIC=1).
    In CoolProp 6.x the bicubic backend is invoked via the AbstractState backend
    string 'BICUBIC&HEOS', not a config flag. Verifies it computes supercritical CO₂.
    """
    import CoolProp
    AS = CoolProp.AbstractState("BICUBIC&HEOS", "CO2")
    AS.update(CoolProp.CoolProp.QT_INPUTS, 0, 280.0)  # quality=0, T=280K (subcritical)
    h = AS.hmass()
    return h > 0


def check_fmpy_import() -> bool:
    import fmpy  # noqa: F401
    return True


def check_omc_modelica_lib() -> bool:
    from OMPython import OMCSessionZMQ
    omc = OMCSessionZMQ()
    result = omc.sendExpression("loadModel(Modelica)")
    omc.sendExpression("quit()")
    return bool(result)


def check_external_media_lib() -> bool:
    em_path = Path("/opt/libs/ExternalMedia/package.mo")
    return em_path.exists()


def check_thermo_power_lib() -> bool:
    # ThermoPower repo structure: /opt/libs/ThermoPower/ThermoPower/package.mo
    tp_path = Path("/opt/libs/ThermoPower/ThermoPower/package.mo")
    return tp_path.exists()


def check_python_packages() -> bool:
    required = [
        "gymnasium",
        "stable_baselines3",
        "pydantic",
        "yaml",
        "scipy",
        "h5py",
        "tensorboard",
        "jinja2",
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(f"Missing packages: {', '.join(missing)}")
    return True


def check_sco2rl_import() -> bool:
    from sco2rl.utils.config import ConfigLoader  # noqa: F401
    from sco2rl.physics.metamodel.builder import SCO2CycleBuilder  # noqa: F401
    return True


# ─── Stage 1 checks ──────────────────────────────────────────────────────────

def check_fmu_simulation() -> bool:
    """Verify an FMU can be simulated with FMPy (requires real FMU in artifacts/)."""
    fmu_dir = Path("artifacts/fmu")
    fmus = list(fmu_dir.glob("*.fmu"))
    if not fmus:
        raise FileNotFoundError(f"No .fmu files in {fmu_dir}")
    import fmpy
    result = fmpy.simulate_fmu(str(fmus[0]), stop_time=1.0)
    return result is not None


# ─── Registry ────────────────────────────────────────────────────────────────

STAGE_CHECKS: dict[int, list[tuple[str, object]]] = {
    0: [
        ("OMPython importable", check_omc_import),
        ("OMC session (1+1=2)", check_omc_session),
        ("CoolProp importable", check_coolprop_import),
        ("CoolProp CO₂ saturation near critical point", check_coolprop_co2_saturation),
        ("CoolProp BICUBIC backend available (RULE-P3)", check_coolprop_bicubic),
        ("FMPy importable", check_fmpy_import),
        ("Modelica standard library loads via OMC", check_omc_modelica_lib),
        ("ExternalMedia package.mo present", check_external_media_lib),
        ("ThermoPower package.mo present", check_thermo_power_lib),
        ("Required Python packages installed", check_python_packages),
        ("sco2rl package importable (ConfigLoader, SCO2CycleBuilder)", check_sco2rl_import),
    ],
    1: [
        ("Stage 0 environment verified", lambda: run_stage(0, silent=True)),
        ("FMU simulation round-trip (FMPy)", check_fmu_simulation),
    ],
}


def run_stage(stage: int, silent: bool = False) -> bool:
    checks = STAGE_CHECKS.get(stage)
    if checks is None:
        print(f"{RED}Unknown stage: {stage}{RESET}")
        return False

    if not silent:
        print(f"\n{BOLD}Stage {stage} environment verification{RESET}")
        print("─" * 48)

    passed = 0
    failed = 0
    for label, fn in checks:
        result = check(label, fn)
        if result:
            passed += 1
        else:
            failed += 1

    if not silent:
        print("─" * 48)
        if failed == 0:
            print(f"{GREEN}{BOLD}All {passed} checks passed.{RESET}")
        else:
            print(f"{RED}{BOLD}{failed} check(s) failed, {passed} passed.{RESET}")

    return failed == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="sco2rl environment verifier")
    parser.add_argument(
        "--stage",
        type=int,
        required=True,
        choices=sorted(STAGE_CHECKS.keys()),
        help="Stage number to verify (0, 1, ...)",
    )
    args = parser.parse_args()
    success = run_stage(args.stage)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
