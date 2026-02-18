# Contributing to sCO2RL

Thanks for your interest in improving sCO2RL.

## Development principles

- Preserve physical safety constraints and units discipline.
- Prefer small, testable changes over broad refactors.
- Keep FMU-facing behavior deterministic where possible.
- Do not bypass safety checks in production-facing code paths.

## Local setup

Recommended path is Docker-based development.

```bash
docker build -t sco2-rl-automation:latest .
docker run --rm -it --gpus all -v $(pwd):/workspace --shm-size=64g sco2-rl-automation:latest
```

Inside container:

```bash
cd /workspace
PYTHONPATH=src python -m pip install -e .[dev]
```

## Testing

Run targeted tests during development:

```bash
PYTHONPATH=src pytest tests/unit/environment/test_sco2_env.py -q --no-cov --override-ini='addopts=' -p no:cacheprovider
```

Run full unit suite before submitting:

```bash
PYTHONPATH=src pytest tests/unit/ -q --no-cov --override-ini='addopts=' -p no:cacheprovider
```

## Coding guidelines

- Keep config-driven behavior in YAML where possible.
- Use clear naming for physical quantities and units.
- Add/adjust tests for every behavior change.
- Avoid introducing hidden global state.

## Pull request checklist

- [ ] Tests pass in container
- [ ] New/changed behavior documented
- [ ] No regression to safety constraint handling
- [ ] Artifacts are not committed unless intentionally required

## Release process

After arXiv submission, tag the release:

```bash
git tag -a v0.1.0 -m "sCO2RL v0.1.0 (post-arXiv submission)"
git push origin v0.1.0
```

Do not tag pre-submission unless explicitly requested by maintainers.
