"""Machine-wide guard for parallel pytest runs (tests/_run_guard.py + conftest)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import _run_guard as guard

LA = ZoneInfo("America/Los_Angeles")
TESTS_DIR = Path(__file__).resolve().parent


# --- time window --------------------------------------------------------------


@pytest.mark.parametrize(
    ("stamp", "expected"),
    [
        (datetime(2026, 9, 23, 5, 59, tzinfo=LA), False),  # Wed before window
        (datetime(2026, 9, 23, 6, 0, tzinfo=LA), True),  # opens at 06:00
        (datetime(2026, 9, 23, 7, 47, tzinfo=LA), True),  # the incident
        (datetime(2026, 9, 23, 13, 29, tzinfo=LA), True),
        (datetime(2026, 9, 23, 13, 30, tzinfo=LA), False),  # closes at 13:30
        (datetime(2026, 9, 26, 9, 0, tzinfo=LA), False),  # Saturday
        (datetime(2026, 9, 27, 9, 0, tzinfo=LA), False),  # Sunday
    ],
)
def test_desk_could_be_live_window(stamp: datetime, expected: bool) -> None:
    assert guard.desk_could_be_live(stamp) is expected


def test_desk_window_converts_other_timezones() -> None:
    # 14:47 UTC on Wed 2026-09-23 is 07:47 PT; 21:00 UTC is 14:00 PT.
    assert guard.desk_could_be_live(datetime(2026, 9, 23, 14, 47, tzinfo=ZoneInfo("UTC")))
    assert not guard.desk_could_be_live(datetime(2026, 9, 23, 21, 0, tzinfo=ZoneInfo("UTC")))


def test_desk_window_rejects_naive_time() -> None:
    with pytest.raises(ValueError):
        guard.desk_could_be_live(datetime(2026, 9, 23, 9, 0))


# --- cap math -------------------------------------------------------------------


def test_capped_workers_math() -> None:
    assert guard.capped_workers(8, live=True) == 4
    assert guard.capped_workers(4, live=True) == 4
    assert guard.capped_workers(2, live=True) == 2
    assert guard.capped_workers(8, live=False) == 8
    assert guard.capped_workers(None, live=True) is None
    assert guard.capped_workers(16, live=True, cap=6) == 6


def test_cap_mode_env() -> None:
    assert guard.cap_mode({}) == "auto"
    assert guard.cap_mode({guard.ENV_WORKERS_CAP: "off"}) == "off"
    assert guard.cap_mode({guard.ENV_WORKERS_CAP: "OFF"}) == "off"
    assert guard.cap_mode({guard.ENV_WORKERS_CAP: "on"}) == "on"


def test_only_parallel_controller_is_guarded() -> None:
    assert guard.is_parallel_controller(8, is_worker=False)
    assert guard.is_parallel_controller(2, is_worker=False)
    assert not guard.is_parallel_controller(1, is_worker=False)
    assert not guard.is_parallel_controller(0, is_worker=False)
    assert not guard.is_parallel_controller(None, is_worker=False)
    assert not guard.is_parallel_controller(8, is_worker=True)


def test_lock_env_switches() -> None:
    assert guard.lock_enabled({})
    assert not guard.lock_enabled({guard.ENV_LOCK: "0"})
    assert guard.lock_timeout({}) == 30 * 60
    assert guard.lock_timeout({guard.ENV_LOCK_TIMEOUT: "1.5"}) == 1.5


def test_default_lock_dir_never_under_live_store(tmp_path: Path) -> None:
    assert guard.default_lock_dir({}, str(tmp_path)) == tmp_path / "TradingBotV3-tests"
    assert guard.default_lock_dir({guard.ENV_LOCK_DIR: str(tmp_path / "x")}, None) == tmp_path / "x"
    assert "TradingBotData" not in str(guard.default_lock_dir({}, None))


def _config(numprocesses, *, worker: bool = False) -> SimpleNamespace:
    tx = ["popen"] * numprocesses if numprocesses else []
    config = SimpleNamespace(option=SimpleNamespace(numprocesses=numprocesses, tx=tx))
    if worker:
        config.workerinput = {"workerid": "gw0"}
    return config


def test_configure_caps_and_locks_controller_in_live_window(tmp_path: Path) -> None:
    config = _config(8)
    env = {guard.ENV_LOCK_DIR: str(tmp_path)}
    guard.configure(config, real_localappdata=None, env=env, now=datetime(2026, 9, 23, 7, 47, tzinfo=LA))
    try:
        assert config.option.numprocesses == 4
        assert config.option.tx == ["popen"] * 4  # what xdist actually spawns from
        assert getattr(config, "_tbv3_run_lock").fd >= 0
    finally:
        guard.unconfigure(config)


def test_configure_leaves_workers_after_hours_and_when_off(tmp_path: Path) -> None:
    env = {guard.ENV_LOCK: "0"}
    after = _config(8)
    guard.configure(after, real_localappdata=None, env=env, now=datetime(2026, 9, 23, 14, 0, tzinfo=LA))
    assert after.option.numprocesses == 8
    assert len(after.option.tx) == 8
    off = _config(8)
    guard.configure(
        off,
        real_localappdata=None,
        env={**env, guard.ENV_WORKERS_CAP: "off"},
        now=datetime(2026, 9, 23, 9, 0, tzinfo=LA),
    )
    assert off.option.numprocesses == 8


def test_configure_skips_workers_and_single_process(tmp_path: Path) -> None:
    env = {guard.ENV_LOCK_DIR: str(tmp_path)}
    now = datetime(2026, 9, 23, 9, 0, tzinfo=LA)
    worker = _config(8, worker=True)
    guard.configure(worker, real_localappdata=None, env=env, now=now)
    assert worker.option.numprocesses == 8
    assert getattr(worker, "_tbv3_run_lock", None) is None
    single = _config(None)
    guard.configure(single, real_localappdata=None, env=env, now=now)
    assert getattr(single, "_tbv3_run_lock", None) is None


# --- the lock -------------------------------------------------------------------


def test_second_lock_waits_then_times_out_naming_holder(tmp_path: Path) -> None:
    held = guard.acquire_run_lock(tmp_path, timeout=0)
    lines: list[str] = []
    ticks = iter([0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    try:
        with pytest.raises(guard.RunLockTimeout) as err:
            guard.acquire_run_lock(
                tmp_path, timeout=1.0, poll=0.5, echo=lines.append, clock=lambda: next(ticks), sleep=lambda _s: None
            )
    finally:
        held.release()
    assert f"pid {os.getpid()}" in str(err.value)
    assert len(lines) == 1 and f"pid {os.getpid()}" in lines[0]
    # Released: the next caller gets it straight away.
    guard.acquire_run_lock(tmp_path, timeout=0).release()


_HOLD_SCRIPT = textwrap.dedent(
    """
    import os, sys
    sys.path.insert(0, sys.argv[1])
    import _run_guard as guard
    from pathlib import Path
    guard.acquire_run_lock(Path(sys.argv[2]), timeout=0)
    print("HELD", os.getpid(), flush=True)
    sys.stdin.readline()
    os._exit(0)  # die without releasing: the OS must drop the lock
    """
)


def test_lock_is_not_stale_after_holder_process_dies(tmp_path: Path) -> None:
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLD_SCRIPT, str(TESTS_DIR), str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        tag, holder_pid = proc.stdout.readline().split()
        assert tag == "HELD"
        with pytest.raises(guard.RunLockTimeout) as err:
            guard.acquire_run_lock(tmp_path, timeout=0)
        assert f"pid {holder_pid}" in str(err.value)
        proc.stdin.write("\n")
        proc.stdin.flush()
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    guard.acquire_run_lock(tmp_path, timeout=0).release()


def test_second_parallel_pytest_run_fails_fast_while_locked(tmp_path: Path) -> None:
    """End to end through conftest: a held lock makes an ``-n 2`` run give up loudly."""
    held = guard.acquire_run_lock(tmp_path, timeout=0)
    env = {
        **os.environ,
        guard.ENV_LOCK_DIR: str(tmp_path),
        guard.ENV_LOCK_TIMEOUT: "1",
        guard.ENV_WORKERS_CAP: "off",
        "QT_QPA_PLATFORM": "offscreen",
    }
    env.pop("PYTEST_XDIST_WORKER", None)
    env.pop(guard.ENV_LOCK, None)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-n",
                "2",
                "-q",
                "-p",
                "no:cacheprovider",
                f"{Path(__file__).name}::test_capped_workers_math",
            ],
            cwd=TESTS_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
    finally:
        held.release()
    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert "test-run-guard" in output
    assert f"pid {os.getpid()}" in output
