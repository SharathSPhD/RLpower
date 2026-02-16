# Stage 0 Memory — Physics Infrastructure

**Status**: COMPLETE ✓ — FMU compiles, loads, runs, energy balance verified (2026-02-16)
**Stage goal**: Python-generated `.mo` → FMU compilation pipeline, fully tested.

## Key Decisions Made This Stage

### ADR-S0-1: YAML dict handling in flow_paths
- **Problem**: YAML parses `"bypass_valve connects: a.b -> c.d"` as `{"bypass_valve connects": "a.b -> c.d"}` (colon triggers mapping)
- **Fix**: `_parse_flow_path_entry()` checks `isinstance(entry, dict)` and reconstructs string as `"key: value"`. Always quote complex flow_path entries in YAML going forward.

### ADR-S0-2: pycache dirs in worktrees owned by Docker root
- **Problem**: Docker container runs as root; pycache files in worktrees can't be removed by `sharaths` user
- **Fix**: Add `**/__pycache__/` to `.gitignore`. Plan: clean with `docker run --rm -v ... alpine sh -c "rm -rf ..."`

### ADR-S0-3: pyproject.toml hatchling build backend missing in Docker base
- **Problem**: `pip install -e .[dev]` fails in pytorch base image because `hatchling` is not pre-installed
- **Fix**: Use `PYTHONPATH=src` for unit tests in CI until full `sco2-rl:latest` image is built

### ADR-S0-4: Temporary Docker image for unit tests
- **Constraint**: `sco2-rl:latest` doesn't exist yet (Dockerfile is Stage 0 compiler deliverable)
- **Workaround**: Use `nvcr.io/nvidia/pytorch:25.11-py3` + `PYTHONPATH=src` for metamodel tests
- **Fix**: Build proper Dockerfile in stage/0-compiler branch

## Bugs / Issues Encountered

1. `_parse_flow_path_entry` received `dict` instead of `str` → fixed with isinstance check
2. `pytest --no-cov` conflicted with `addopts` in pyproject.toml → fixed with `--override-ini='addopts='`
3. `ComponentFactory.create()` rejects unknown modelica_type strings — test fixtures must use spec from `_KNOWN_MODELICA_TYPES` or construct `ComponentSpec` directly
4. `.dockerignore` was accidentally in `.gitignore` — removed; `.dockerignore` must be committed for docker build context
5. pycache in worktrees owned by Docker (root) — cannot rm without sudo; add `**/__pycache__/` to .gitignore

## Numerical Findings
<!-- e.g., "CVODE with 1e-4 tolerance: X s/episode. With 1e-6: Y s/episode." -->

## Configuration Changes Since Baseline
<!-- e.g., "configs/fmu/fmu_export.yaml: cvode_tolerance_training: 1e-4 → 1e-3, reason: ..." -->

## OMPython Quirks
<!-- Document any non-obvious OMPython API behaviors discovered -->

## ExternalMedia / CoolProp ARM64 Notes
### Docker Build Fixes (chronological)
1. **OMC GPG key URL**: Changed from `omc/bootstrapkey.asc` → `apt/openmodelica.asc`; `arch=arm64` → `arch=$(dpkg --print-architecture)`
2. **CoolProp `-m64` flag**: ARM64 doesn't have `-m64`. Fix: `-DFORCE_BITNESS_NATIVE=ON` in cmake
3. **CoolProp submodules**: `--depth 1` needs `--recurse-submodules --shallow-submodules` for fmtlib/msgpack-c
4. **ExternalMedia cmake path**: CMakeLists.txt is at `Projects/` (not root). Library installs to `Modelica/ExternalMedia/Resources/Library/linux64/`
5. **ExternalMedia CoolProp network failure**: Reuse `/build/CoolProp/src` via `mv` to `externals/CoolProp.git` to avoid second clone
6. **ExternalMedia fmtlib include**: CoolProp 6.6.0 has `fmtlib/include/fmt/format.h` but ExternalMedia 4.0.0 cmake adds only `fmtlib/` (no `/include`). Fix: `-DCMAKE_CXX_FLAGS="-I.../fmtlib/include"`
7. **Stage 2 OMC paths**: Don't copy from builder; re-install `openmodelica` from apt in Stage 2 (correct deps handling)
8. **CoolProp Python not in Stage 2**: Add `CoolProp==${COOLPROP_VERSION}` to Stage 2 pip install
9. **OMC root restriction**: OMC ZMQ server refuses to start as root. Fix: Add non-root user `sco2rl` (uid=1001)
10. **GID collision**: GID 1000 exists in pytorch base. Fix: Use GID 1001
11. **Modelica MSL not installed**: `openmodelica` apt package doesn't include MSL. Fix: Pre-install via `installPackage(Modelica)` as sco2rl user during Docker build

### Final Image Layout
- OMC: `/usr/bin/omc`, `/usr/lib/omc/`, `/usr/lib/aarch64-linux-gnu/omc/`, `/usr/share/omc/`
- Modelica MSL 4.1.0: `/home/sco2rl/.openmodelica/libraries/Modelica 4.1.0+maint.om/`
- libCoolProp: `/opt/libs/CoolProp/libCoolProp.so`
- libExternalMediaLib: `/opt/libs/ExternalMedia/Resources/Library/linux64/libExternalMediaLib.so`
- ThermoPower: `/opt/libs/ThermoPower/ThermoPower/package.mo`
- SCOPE: `/opt/libs/SCOPE/src/Modelica/...`

### CoolProp API Notes for verify_environment.py
- `PropsSI("T", "P", 8e6, "Q", 0, "CO2")` FAILS: 8 MPa > Pcrit (7.377 MPa). Use P=7e6
- `CP.ENABLE_BICUBIC_FIX` does NOT exist in CoolProp 6.6.0. BICUBIC backend is via `AbstractState("BICUBIC&HEOS", "CO2")`
- ThermoPower path: `/opt/libs/ThermoPower/ThermoPower/package.mo` (not `/opt/libs/ThermoPower/package.mo`)

## SCOPE Modelica 4.x Compatibility Fixes (ARM64, OMC 1.26.2 + MSL 4.1.0)

All patches are now in Dockerfile and persist across container rebuilds.

### ADR-S0-5: DataRecord.R renamed to R_s in MSL 4.x
- **Problem**: `Modified element R not found in class DataRecord` — MSL 4.x renamed `DataRecord.R` to `DataRecord.R_s`
- **Files patched**: `Steps/Media/SCO2.mo`, `Steps/Media/CO2.mo`
- **Fix**: `sed 'R=188.92...' → 'R_s=188.92...'` and `data.R` → `data.R_s`

### ADR-S0-6: Modelica.SIunits → Modelica.Units.SI (MSL 4.x namespace)
- **Problem**: `Class Modelica.SIunits.Temperature not found` — package was reorganized
- **Fix**: Bulk `sed` on all SCOPE `.mo` files: `Modelica.SIunits.Conversions` → `Modelica.Units.Conversions`, then `Modelica.SIunits` → `Modelica.Units.SI`
- **Note**: The `within Modelica; package SIunits = Modelica.Units.SI` shim does NOT work — member access through package aliases fails in OMC 1.26.2

### ADR-S0-7: Valve.mo undeclared medium_in/out.state variables
- **Problem**: Valve.mo used `medium_in.state` and `medium_out.state` — never declared in TwoPorts parent
- **Fix**: Rewrote Valve.mo as pure isenthalpic throttle using only fluid port equations (like Pump.mo pattern)

### ADR-S0-8: ExternalMedia C library not available for ARM64
- **Problem**: `External function 'TwoPhaseMedium_getMolarMass_C_impl' could not be found` — precompiled `ExternalMediaLib.so` is x86-64 only
- **Impact**: `Recuperator.mo` and `BaseExchanger.mo` originally used `PBMedia.BaseProperties.setState_phX` which requires ExternalMedia C wrapper
- **Fix**: Rewrote `BaseExchanger.mo` (port-only, no BaseProperties) and `Recuperator.mo` (uses `Steps.Utilities.CoolProp.PropsSI` directly)
- **ARM64 CoolProp**: Available at `/opt/libs/CoolProp/libCoolProp.so` (6.6.0, ARM64-native)

### ADR-S0-9: libMyProps.so not available for ARM64
- **Problem**: `MyPropsSI` external function (called by `Steps.Utilities.CoolProp.PropsSI`) requires `libMyProps.so` — SCOPE only ships x86-64/Windows binaries
- **Fix**: Wrote minimal C stub `scripts/myprops_stub.c` that wraps CoolProp's public C API (`PropsSI`, `MyPropsSI_pH`, `MyPropsSI_pT`). Built as `libMyProps.so` using `-lCoolProp`
- **Key insight**: SCOPE's `MyPropsLib.cpp` uses CoolProp internal C++ headers which need `fmtlib`; stub avoids all internal headers, uses only `CoolPropLib.h` (public C API)

### ADR-S0-10: CMake build of FMU — CMAKE_SHARED_LIBRARY_PREFIX="" breaks -l flag generation
- **Problem**: OMC-generated `CMakeLists.txt` sets `CMAKE_SHARED_LIBRARY_PREFIX ""` (empty). This causes CMake to generate `-llibMyProps` (wrong) instead of `-lMyProps` when linking a full-path shared library
- **Root cause**: With empty prefix, CMake doesn't strip `lib` from `libMyProps.so` → `-l` + `libMyProps` = `-llibMyProps`
- **Fix**: Directly edited `link.txt` to replace `-llibMyProps` with full path `/path/to/libMyProps.so`. In Dockerfile: build step runs after cmake generates link.txt

### ADR-S0-11: CycleBuilder must filter to connected components only
- **Problem**: `bypass_valve` and `inventory_valve` are declared in simple_recuperated topology but have no flow connections. OMC rejects floating components: "imbalanced equations (5) and variables (6)"
- **Fix**: `SCO2CycleBuilder.build()` collects all component names referenced in connections, includes ONLY those in the CycleModel
- **Tests updated**: simple_recuperated → 5 components (not 7), recompression → 8 components (not 10)

## FMU Compilation Result (2026-02-16)

- **Model**: `Steps.Cycle.SCO2RecuperatedCycle`, FMI 2.0 Co-Simulation
- **Variables**: 86 model variables
- **FMU size**: 3.7 MB
- **Energy balance at design point**:
  - W_turbine = 13.4 MW, W_comp = 2.6 MW, W_net = 10.8 MW ✓
  - Q_recuperator = 50.6 MW ✓
  - T_turbine_outlet = 581°C (physically plausible for 700°C inlet) ✓
- **FMU contents**: `binaries/linux64/SCO2RecuperatedCycle.so` + `libMyProps.so` + `libCoolProp.so.6`

## Gate 0 Verification Results
- `pytest tests/unit/physics/ -v`: [x] PASS — Docker 2026-02-16
- `pytest tests/unit/ -v` (all): [x] PASS (410/410) — Docker 2026-02-16
- `scripts/verify_environment.py --stage 0`: [x] PASS (11/11 checks) — Docker 2026-02-16
- Energy balance check at design point: [x] PASS — W_net=10.8 MW, Q_rec=50.6 MW
- `libMyProps.so` + `libCoolProp.so.6` bundled in FMU: [x] PASS
- FMU loads with FMPy: [x] PASS — `read_model_description()` + `simulate_fmu()` both succeed

## Open Questions for Stage 1
- How does FMPy handle the FMU's CVODE initialization? Does it use the embedded solver?
- Are the FMU's initial conditions (p_init=18 MPa, T_init=973 K, m_flow_init=95 kg/s) from the Regulator sufficient, or do we need to pass them as start_values?
- The FMU has no `output` causality variables. Stage 1's MockFMU and SCO2FMUEnv need to read `local` variables — verify FMPy can read these.
- `FMPy/logging/linux64/logging.so` missing for ARM64 — benign but could affect debug builds
