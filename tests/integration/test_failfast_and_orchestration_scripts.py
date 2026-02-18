from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(script_name: str):
    script_path = PROJECT_ROOT / "scripts" / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module: {script_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_check_mock_path_runs_end_to_end() -> None:
    module = _load_script_module("preflight_check")
    config, lagrangian_cfg = module.load_config(PROJECT_ROOT, max_steps_override=8)
    env = module.build_env(
        project_root=PROJECT_ROOT,
        config=config,
        fmu_path=PROJECT_ROOT / "artifacts/fmu_build/does_not_matter.fmu",
        use_mock=True,
    )
    try:
        report = module.run_preflight(env, episodes=2)
        rc = module.print_summary(env, report, lagrangian_cfg)
    finally:
        env.close()

    assert rc == 0
    assert report["nan_or_inf_detected"] is False
    assert isinstance(report["termination_reasons"], dict)


def test_preflight_surrogate_dispatches_collect_then_train(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module("preflight_surrogate")
    fake_fmu = tmp_path / "model.fmu"
    fake_fmu.write_text("placeholder")
    fake_dataset = tmp_path / "data" / "preflight.h5"

    executed: list[list[str]] = []

    def _fake_run_cmd(cmd: list[str], cwd: Path) -> None:
        assert cwd == PROJECT_ROOT
        executed.append(cmd)

    monkeypatch.setattr(module, "run_cmd", _fake_run_cmd)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            fmu_path=str(fake_fmu),
            dataset=str(fake_dataset),
            n_samples=12,
            n_workers=2,
            batch_size=3,
            episode_max_steps=30,
            epochs=4,
            device="cpu",
        ),
    )

    rc = module.main()
    assert rc == 0
    assert len(executed) == 2
    assert "collect_trajectories.py" in executed[0][1]
    assert "train_surrogate.py" in executed[1][1]
    assert "--episode-max-steps" in executed[0]


def test_run_parallel_training_launches_two_processes(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module("run_parallel_training")
    fake_fmu = tmp_path / "plant.fmu"
    fake_fmu.write_text("placeholder")
    fake_dataset = tmp_path / "fno_training.h5"
    fake_dataset.write_text("placeholder")

    launched: list[list[str]] = []

    class _FakeProc:
        def __init__(self, cmd: list[str]) -> None:
            self.cmd = cmd

        def wait(self) -> int:
            return 0

    def _fake_popen(cmd: list[str], cwd: Path):
        assert cwd == PROJECT_ROOT
        launched.append(cmd)
        return _FakeProc(cmd)

    monkeypatch.setattr(module.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            fmu_path=str(fake_fmu),
            cpu_timesteps=1000,
            gpu_timesteps=2000,
            fno_epochs=5,
            allow_gpu_fidelity_fail=True,
            dataset=str(fake_dataset),
            n_envs_cpu=2,
        ),
    )

    rc = module.main()
    assert rc == 0
    assert len(launched) == 2
    assert "train_fmu.py" in launched[0][1]
    assert "train_surrogate.py" in launched[1][1]
    assert "--rl-timesteps" in launched[1]
    assert "--allow-fidelity-fail" in launched[1]


def test_cross_validate_returns_error_when_checkpoint_missing(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module("cross_validate_and_export")
    fake_fmu = tmp_path / "plant.fmu"
    fake_fmu.write_text("placeholder")

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            rl_checkpoint=str(tmp_path / "missing_checkpoint.json"),
            fmu_path=str(fake_fmu),
            episodes=2,
            report_out=str(tmp_path / "report.json"),
            onnx_out=str(tmp_path / "policy.onnx"),
            trt_out=str(tmp_path / "policy.plan"),
            skip_trt=True,
        ),
    )

    rc = module.main()
    assert rc == 1
