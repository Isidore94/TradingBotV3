"""Machine-wide guard for parallel (xdist) pytest runs.

Two parallel full-suite runs next to the live desk exhausted RAM + pagefile and
bluescreened the desk machine. This module keeps it to one ``-n`` > 1 run at a
time (an OS file lock the OS drops when the process dies) and caps workers at
``MARKET_HOURS_WORKER_CAP`` while the desk could be live, and at
``NIGHT_AI_WORKER_CAP`` while the night AI's model holds most of the RAM.
``tests/conftest.py``
wires it in; everything here is stdlib only.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

#: Desk-could-be-live window: market hours plus margin, desk-local time.
DESK_TIMEZONE = "America/Los_Angeles"
DESK_LIVE_START = dtime(6, 0)
DESK_LIVE_END = dtime(13, 30)
MARKET_HOURS_WORKER_CAP = 4

#: Night-AI window, desk-local time, every day: the local model takes ~20 GB of RAM.
NIGHT_AI_START = dtime(22, 0)
NIGHT_AI_END = dtime(2, 0)
NIGHT_AI_WORKER_CAP = 2

#: Default wait for another parallel run to finish before giving up.
DEFAULT_LOCK_TIMEOUT_SECONDS = 30 * 60
LOCK_POLL_SECONDS = 2.0

ENV_LOCK = "TBV3_TEST_LOCK"  # "0" bypasses the lock
ENV_LOCK_TIMEOUT = "TBV3_TEST_LOCK_TIMEOUT"  # seconds
ENV_LOCK_DIR = "TBV3_TEST_LOCK_DIR"  # override lock folder (tests)
ENV_WORKERS_CAP = "TBV3_TEST_WORKERS_CAP"  # "off" disables, "on" forces

LOCK_FILE_NAME = "pytest-parallel.lock"
HOLDER_FILE_NAME = "pytest-parallel.holder.json"


class RunLockTimeout(RuntimeError):
    """Another parallel run held the lock for longer than the timeout."""


def desk_could_be_live(now: datetime) -> bool:
    """True on a weekday between DESK_LIVE_START and DESK_LIVE_END desk time."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    local = now.astimezone(ZoneInfo(DESK_TIMEZONE))
    if local.weekday() >= 5:
        return False
    return DESK_LIVE_START <= local.time() < DESK_LIVE_END


def night_ai_could_be_running(now: datetime) -> bool:
    """True between NIGHT_AI_START and NIGHT_AI_END desk time, any day."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    local = now.astimezone(ZoneInfo(DESK_TIMEZONE)).time()
    return local >= NIGHT_AI_START or local < NIGHT_AI_END


def window_cap(now: datetime) -> int | None:
    """The worker cap for this moment, or None when no window applies."""
    if night_ai_could_be_running(now):
        return NIGHT_AI_WORKER_CAP
    if desk_could_be_live(now):
        return MARKET_HOURS_WORKER_CAP
    return None


def cap_mode(env: Mapping[str, str]) -> str:
    """'off', 'on' (always cap) or 'auto' (cap only in the live window)."""
    raw = str(env.get(ENV_WORKERS_CAP, "")).strip().lower()
    if raw in {"off", "0", "false", "no"}:
        return "off"
    if raw in {"on", "1", "true", "yes", "force"}:
        return "on"
    return "auto"


def capped_workers(requested: Any, *, live: bool, cap: int = MARKET_HOURS_WORKER_CAP) -> Any:
    """Return the worker count to use; only an int above ``cap`` while live changes."""
    if not live or not isinstance(requested, int) or isinstance(requested, bool):
        return requested
    return min(requested, cap)


def is_parallel_controller(numprocesses: Any, *, is_worker: bool) -> bool:
    """True for the xdist controller of a run with more than one worker."""
    if is_worker:
        return False
    return isinstance(numprocesses, int) and not isinstance(numprocesses, bool) and numprocesses > 1


def lock_enabled(env: Mapping[str, str]) -> bool:
    return str(env.get(ENV_LOCK, "")).strip() != "0"


def lock_timeout(env: Mapping[str, str]) -> float:
    raw = str(env.get(ENV_LOCK_TIMEOUT, "")).strip()
    if not raw:
        return float(DEFAULT_LOCK_TIMEOUT_SECONDS)
    try:
        return max(0.0, float(raw))
    except ValueError:
        return float(DEFAULT_LOCK_TIMEOUT_SECONDS)


def default_lock_dir(env: Mapping[str, str], real_localappdata: str | None) -> Path:
    """Lock folder: env override, else real LOCALAPPDATA, else the temp dir. Never a live store."""
    override = str(env.get(ENV_LOCK_DIR, "")).strip()
    if override:
        return Path(override)
    base = real_localappdata or tempfile.gettempdir()
    return Path(base) / "TradingBotV3-tests"


# ---------------------------------------------------------------------------
# OS-level exclusive lock (released by the OS when the process dies)
# ---------------------------------------------------------------------------


def _try_lock(fd: int) -> bool:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


@dataclass
class RunLock:
    fd: int
    lock_dir: Path

    def release(self) -> None:
        if self.fd < 0:
            return
        try:
            _unlock(self.fd)
        except OSError:
            pass
        try:
            os.close(self.fd)
        finally:
            self.fd = -1


def read_holder(lock_dir: Path) -> str:
    """One-line description of the current holder, or 'unknown holder'."""
    try:
        data = json.loads((lock_dir / HOLDER_FILE_NAME).read_text(encoding="utf-8"))
        return f"pid {data.get('pid', '?')} started {data.get('started', '?')}"
    except (OSError, ValueError, AttributeError):
        return "unknown holder"


def _write_holder(lock_dir: Path) -> None:
    info = {
        "pid": os.getpid(),
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
        "argv": sys.argv[1:],
    }
    try:
        (lock_dir / HOLDER_FILE_NAME).write_text(json.dumps(info), encoding="utf-8")
    except OSError:
        pass  # holder info is a courtesy; the lock itself is what guards


def acquire_run_lock(
    lock_dir: Path,
    *,
    timeout: float,
    poll: float = LOCK_POLL_SECONDS,
    echo: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> RunLock:
    """Take the machine-wide lock, waiting up to ``timeout`` seconds.

    Prints one line naming the holder when it has to wait; raises RunLockTimeout
    when the wait runs out.
    """
    lock_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_dir / LOCK_FILE_NAME), os.O_RDWR | os.O_CREAT, 0o666)
    try:
        deadline = clock() + timeout
        announced = False
        while not _try_lock(fd):
            holder = read_holder(lock_dir)
            if clock() >= deadline:
                raise RunLockTimeout(
                    f"another parallel pytest run holds the machine test lock ({holder}); "
                    f"gave up after {timeout:g}s. Wait for it, or run without -n. "
                    f"Lock: {lock_dir / LOCK_FILE_NAME}"
                )
            if not announced and echo is not None:
                echo(
                    f"[test-run-guard] waiting for another parallel pytest run ({holder}); "
                    f"timeout {timeout:g}s"
                )
                announced = True
            sleep(max(0.0, min(poll, deadline - clock())))
    except BaseException:
        os.close(fd)
        raise
    _write_holder(lock_dir)
    return RunLock(fd=fd, lock_dir=lock_dir)


# ---------------------------------------------------------------------------
# pytest wiring (called from tests/conftest.py)
# ---------------------------------------------------------------------------

_CONFIG_LOCK_ATTR = "_tbv3_run_lock"


def _echo(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def configure(
    config: Any,
    *,
    real_localappdata: str | None,
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> None:
    """Cap workers in the live window, then take the lock for a parallel run."""
    env = os.environ if env is None else env
    is_worker = hasattr(config, "workerinput") or bool(env.get("PYTEST_XDIST_WORKER"))
    requested = getattr(config.option, "numprocesses", None)
    if not is_parallel_controller(requested, is_worker=is_worker):
        return

    mode = cap_mode(env)
    if mode != "off":
        cap = window_cap(now or datetime.now().astimezone())
        if mode == "on" and cap is None:
            cap = MARKET_HOURS_WORKER_CAP
        workers = capped_workers(requested, live=cap is not None, cap=cap or MARKET_HOURS_WORKER_CAP)
        if workers != requested:
            config.option.numprocesses = workers
            # xdist turned -n into a "popen" tx list in its pytest_cmdline_main;
            # that list is what spawns workers, so trim it too.
            tx = getattr(config.option, "tx", None)
            if isinstance(tx, list) and len(tx) > workers:
                config.option.tx = tx[:workers]
            _echo(
                f"[test-run-guard] desk or night-AI hours: capping -n {requested} to {workers} workers "
                f"({ENV_WORKERS_CAP}=off disables)"
            )

    if not lock_enabled(env):
        return
    lock_dir = default_lock_dir(env, real_localappdata)
    try:
        lock = acquire_run_lock(lock_dir, timeout=lock_timeout(env), echo=_echo)
    except RunLockTimeout as exc:
        import pytest

        raise pytest.UsageError(f"[test-run-guard] {exc}") from None
    setattr(config, _CONFIG_LOCK_ATTR, lock)


def unconfigure(config: Any) -> None:
    lock = getattr(config, _CONFIG_LOCK_ATTR, None)
    if lock is not None:
        lock.release()
        setattr(config, _CONFIG_LOCK_ATTR, None)
