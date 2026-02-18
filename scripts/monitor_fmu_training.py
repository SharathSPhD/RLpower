#!/usr/bin/env python3
"""Monitor FMU PPO training progress and raise stall alarms.

This script tracks checkpoint progression in a run directory and reports:
- latest step / phase / throughput / ETA
- mean_reward / violation_rate parsed from training log (CurriculumCallback output)
- alarm conditions (stalled checkpoints, low throughput, no phase advance,
  broken reward signal r_tracking≈0)
- training process liveness

Guardrails (optional --auto-restart):
  If the checkpoint is stale AND the training PID is dead, automatically
  relaunch the training command so unattended runs self-heal.

Use --once for a one-shot status report, or run continuously in daemon mode.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


_CKPT_RE = re.compile(r"step_(\d+)_phase_(\d+)_checkpoint\.json$")
# Matches: [CurriculumCallback] rollout_end step=N phase=P episodes=E mean_r=R viol_rate=V
_LOG_ROLLOUT_RE = re.compile(
    r"rollout_end step=(\d+) phase=(\d+) episodes=(\d+) mean_r=([\-\d\.]+) viol_rate=([\d\.]+)"
)
# Matches: [CurriculumCallback] *** ADVANCED to phase N at step=M ***
_LOG_ADVANCE_RE = re.compile(r"ADVANCED to phase (\d+) at step=(\d+)")


@dataclass
class Snapshot:
    timestamp_utc: str
    checkpoint_count: int
    latest_step: int
    latest_phase: int
    latest_checkpoint: str
    latest_checkpoint_mtime_epoch: float
    avg_recent_steps_per_sec: float
    median_recent_steps_per_sec: float
    eta_seconds: float
    training_pid_alive: bool
    # Parsed from CurriculumCallback log output
    log_mean_reward: float
    log_violation_rate: float
    log_phase_advances: int
    alarms: list[str]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _pid_alive(pid_file: Path | None) -> bool:
    if pid_file is None or not pid_file.exists():
        return True  # assume alive if no PID file
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, OSError):
        return False


def _collect_checkpoints(run_dir: Path) -> list[tuple[int, int, float, Path]]:
    rows: list[tuple[int, int, float, Path]] = []
    if not run_dir.exists():
        return rows
    for p in run_dir.glob("*_checkpoint.json"):
        m = _CKPT_RE.match(p.name)
        if not m:
            continue
        step = int(m.group(1))
        phase = int(m.group(2))
        rows.append((step, phase, p.stat().st_mtime, p))
    rows.sort(key=lambda x: x[0])
    return rows


def _compute_rates(rows: list[tuple[int, int, float, Path]], recent_n: int) -> tuple[float, float]:
    if len(rows) < 2:
        return 0.0, 0.0
    rates: list[float] = []
    for (s0, _p0, t0, _f0), (s1, _p1, t1, _f1) in zip(rows[:-1], rows[1:]):
        dt = max(1e-9, t1 - t0)
        rates.append((s1 - s0) / dt)
    recent = rates[-recent_n:] if len(rates) > recent_n else rates
    avg = sum(recent) / len(recent)
    sorted_r = sorted(recent)
    n = len(sorted_r)
    med = sorted_r[n // 2] if n % 2 == 1 else (sorted_r[n // 2 - 1] + sorted_r[n // 2]) / 2.0
    return avg, med


def _parse_training_log(log_path: Path | None) -> tuple[float, float, int]:
    """Parse CurriculumCallback log output from training log file.

    Returns (latest_mean_reward, latest_violation_rate, n_phase_advances).
    """
    if log_path is None or not log_path.exists():
        return 0.0, 0.0, 0
    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        return 0.0, 0.0, 0

    mean_r = 0.0
    viol_r = 0.0
    for m in _LOG_ROLLOUT_RE.finditer(text):
        mean_r = float(m.group(4))
        viol_r = float(m.group(5))

    n_advances = len(_LOG_ADVANCE_RE.findall(text))
    return mean_r, viol_r, n_advances


def _build_snapshot(
    rows: list[tuple[int, int, float, Path]],
    target_steps: int,
    recent_n: int,
    min_steps_per_sec: float,
    max_stall_seconds: float,
    phase0_alarm_step: int,
    pid_file: Path | None,
    training_log: Path | None = None,
    min_reward_threshold: float = 50.0,
) -> Snapshot:
    alarms: list[str] = []
    pid_alive = _pid_alive(pid_file)
    log_mean_r, log_viol_r, n_advances = _parse_training_log(training_log)

    if not rows:
        alarms.append("ALARM:NO_CHECKPOINTS_FOUND")
        if not pid_alive:
            alarms.append("ALARM:TRAINING_PROCESS_DEAD")
        return Snapshot(
            timestamp_utc=_now_utc(),
            checkpoint_count=0,
            latest_step=0,
            latest_phase=0,
            latest_checkpoint="",
            latest_checkpoint_mtime_epoch=0.0,
            avg_recent_steps_per_sec=0.0,
            median_recent_steps_per_sec=0.0,
            eta_seconds=float("inf"),
            training_pid_alive=pid_alive,
            log_mean_reward=log_mean_r,
            log_violation_rate=log_viol_r,
            log_phase_advances=n_advances,
            alarms=alarms,
        )

    latest_step, latest_phase, latest_mtime, latest_path = rows[-1]
    avg_sps, med_sps = _compute_rates(rows, recent_n=recent_n)

    now = time.time()
    age_sec = now - latest_mtime

    if age_sec > max_stall_seconds:
        age_min = age_sec / 60.0
        alarms.append(f"ALARM:CHECKPOINT_STALE age_minutes={age_min:.1f}")

    if avg_sps > 0.0 and avg_sps < min_steps_per_sec:
        alarms.append(
            f"ALARM:THROUGHPUT_LOW avg_steps_per_sec={avg_sps:.1f} threshold={min_steps_per_sec:.1f}"
        )

    if latest_phase == 0 and latest_step >= phase0_alarm_step:
        alarms.append(
            f"ALARM:PHASE0_STALL step={latest_step} threshold_step={phase0_alarm_step}"
        )

    if not pid_alive:
        alarms.append("ALARM:TRAINING_PROCESS_DEAD")

    # Critical combined alarm: stale checkpoint AND dead PID suggests a crash
    if age_sec > max_stall_seconds and not pid_alive:
        alarms.append("ALARM:TRAINING_CRASHED_SELF_HEAL_NEEDED")

    # Reward health check: if log has data and mean_reward is very low, reward may be broken
    if log_mean_r != 0.0 and log_mean_r < min_reward_threshold and n_advances == 0:
        alarms.append(
            f"ALARM:REWARD_TOO_LOW mean_r={log_mean_r:.2f} threshold={min_reward_threshold:.1f} "
            f"(possible double-scaling or unit mismatch)"
        )

    rem = max(0, target_steps - latest_step)
    eta = rem / avg_sps if avg_sps > 0 else float("inf")
    return Snapshot(
        timestamp_utc=_now_utc(),
        checkpoint_count=len(rows),
        latest_step=latest_step,
        latest_phase=latest_phase,
        latest_checkpoint=str(latest_path),
        latest_checkpoint_mtime_epoch=latest_mtime,
        avg_recent_steps_per_sec=avg_sps,
        median_recent_steps_per_sec=med_sps,
        eta_seconds=eta,
        training_pid_alive=pid_alive,
        log_mean_reward=log_mean_r,
        log_violation_rate=log_viol_r,
        log_phase_advances=n_advances,
        alarms=alarms,
    )


def _fmt_snapshot(s: Snapshot) -> str:
    eta = "inf" if s.eta_seconds == float("inf") else f"{s.eta_seconds / 60.0:.1f} min"
    pid_str = "alive" if s.training_pid_alive else "DEAD"
    base = (
        f"[{s.timestamp_utc}] step={s.latest_step:,} phase={s.latest_phase} "
        f"ckpts={s.checkpoint_count} sps(avg/med)={s.avg_recent_steps_per_sec:.1f}/{s.median_recent_steps_per_sec:.1f} "
        f"eta={eta} pid={pid_str} "
        f"mean_r={s.log_mean_reward:.2f} viol={s.log_violation_rate:.3f} advances={s.log_phase_advances}"
    )
    if not s.alarms:
        return f"{base} status=OK"
    return f"{base} status=ALARM alarms={' | '.join(s.alarms)}"


def _auto_restart(restart_cmd: str, pid_file: Path | None, log_file: Path) -> None:
    """Relaunch training command and update PID file."""
    msg = f"[{_now_utc()}] AUTO-RESTART: relaunching training: {restart_cmd}"
    print(msg, flush=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")

    proc = subprocess.Popen(
        restart_cmd,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    if pid_file is not None:
        pid_file.write_text(str(proc.pid))
    msg2 = f"[{_now_utc()}] AUTO-RESTART: new PID={proc.pid}"
    print(msg2, flush=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(msg2 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor FMU training checkpoints and alarms")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="artifacts/checkpoints/final_fixed/final_fixed_run",
        help="Checkpoint run directory to monitor",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=5_000_000,
        help="Target total timesteps for ETA computation",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=120,
        help="Polling interval for daemon mode",
    )
    parser.add_argument(
        "--recent-intervals",
        type=int,
        default=6,
        help="How many recent checkpoint intervals to average for throughput",
    )
    parser.add_argument(
        "--stall-minutes",
        type=float,
        default=20.0,
        help="Raise alarm if checkpoint file age exceeds this threshold",
    )
    parser.add_argument(
        "--min-steps-per-sec",
        type=float,
        default=350.0,
        help="Raise alarm if recent average throughput falls below this",
    )
    parser.add_argument(
        "--phase0-alarm-step",
        type=int,
        default=100_000,
        help="Raise alarm if still in phase 0 after this many steps (use 0 to disable)",
    )
    parser.add_argument(
        "--training-log",
        type=str,
        default=None,
        help="Path to training stdout log file for parsing mean_reward and violation_rate",
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=50.0,
        help="Raise ALARM:REWARD_TOO_LOW if log mean_reward < this (and no phase advances yet)",
    )
    parser.add_argument(
        "--pid-file",
        type=str,
        default="artifacts/monitoring/training.pid",
        help="Path to file containing training process PID for liveness checks",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default="artifacts/monitoring/fmu_monitor_state.json",
        help="Path to write latest machine-readable snapshot",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="artifacts/monitoring/fmu_monitor.log",
        help="Path to append human-readable monitor lines",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one status sample and exit",
    )
    parser.add_argument(
        "--auto-restart",
        type=str,
        default=None,
        metavar="COMMAND",
        help=(
            "Shell command to relaunch training if crash is detected "
            "(ALARM:TRAINING_CRASHED_SELF_HEAL_NEEDED). Only fires once per guard cycle."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.checkpoint_dir)
    state_file = Path(args.state_file)
    log_file = Path(args.log_file)
    pid_file = Path(args.pid_file) if args.pid_file else None
    state_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    _ = _load_json(state_file)  # warm for future comparative alarms
    auto_restarted = False  # guard against restart loop

    while True:
        rows = _collect_checkpoints(run_dir)
        training_log = Path(args.training_log) if args.training_log else None
        snap = _build_snapshot(
            rows=rows,
            target_steps=args.target_steps,
            recent_n=args.recent_intervals,
            min_steps_per_sec=args.min_steps_per_sec,
            max_stall_seconds=args.stall_minutes * 60.0,
            phase0_alarm_step=args.phase0_alarm_step,
            pid_file=pid_file,
            training_log=training_log,
            min_reward_threshold=args.min_reward,
        )
        line = _fmt_snapshot(snap)
        print(line, flush=True)

        state_file.write_text(json.dumps(asdict(snap), indent=2))
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

        # Auto-restart guardrail
        if (
            args.auto_restart
            and not auto_restarted
            and "ALARM:TRAINING_CRASHED_SELF_HEAL_NEEDED" in snap.alarms
        ):
            _auto_restart(args.auto_restart, pid_file, log_file)
            auto_restarted = True  # only restart once per monitor session
        elif auto_restarted and snap.training_pid_alive:
            auto_restarted = False  # reset after successful restart

        if args.once:
            return 2 if snap.alarms else 0

        time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
