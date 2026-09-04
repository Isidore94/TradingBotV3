from __future__ import annotations

"""Per-thread CPU gauge: names the thread that is starving the desk.

The stall watchdog (``ui/stall_watchdog.py``) samples the GUI thread's own
stack, so it can only name a stall the GUI thread CAUSED. When the cause is
another thread holding the interpreter lock, the GUI thread's stack is the
event loop itself and the watchdog attributes hundreds of seconds to
``app.exec`` - true and useless. That is what happened on 2026-08-31 (the
in-process warehouse build), on the morning of 2026-09-03 (same), and again
on the evening of 2026-09-03, when the research M5 tee thread held the GIL in
91% of samples and nothing on the desk said so until someone ran py-spy.

This gauge closes that gap with the cheapest possible measurement: once a
minute, the CPU time of every Python thread in the process, read from the OS
(``GetThreadTimes`` on Windows, ``/proc/self/task`` on Linux). The delta over
the interval is the fraction of one core each thread used. One JSONL record
per tick goes to the diagnostics dir, and any thread other than the GUI thread
that used more than :data:`HOT_THREAD_FRACTION` of a core is logged as a
warning with its name - a one-line answer to "who is holding the lock?".

Always on: one syscall per thread per minute costs nothing, and the point is
that it is already running the day the next hot thread appears. It measures
and reports; it never stops, renices or interrupts anything.

Read the log back with::

    .venv\\Scripts\\python.exe scripts/ui/thread_cpu_gauge.py
"""

import ctypes
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

#: A thread above this share of one core over one interval is "hot".
HOT_THREAD_FRACTION = 0.5
#: Sampling interval. A minute is coarse enough to be free and fine enough
#: that a thread burning a core is named within a minute of starting.
INTERVAL_SECONDS = 60.0
#: Threads written per record: the busiest few, never all fifty.
TOP_THREADS = 6
#: A runaway loop must not fill the disk; one record a minute is 1,440 a day.
MAX_RECORDS_PER_HOUR = 120
LOG_NAME = "thread_cpu.jsonl"

_REPO_ROOT = Path(__file__).resolve().parents[2]

_THREAD_QUERY_INFORMATION = 0x0040
_THREAD_QUERY_LIMITED_INFORMATION = 0x0800


def log_path() -> Path:
    from project_paths import get_diagnostics_dir

    return Path(get_diagnostics_dir()) / LOG_NAME


# ---------------------------------------------------------------------
# Reading thread CPU time from the OS
# ---------------------------------------------------------------------
class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]

    def seconds(self) -> float:
        return ((self.dwHighDateTime << 32) | self.dwLowDateTime) / 1e7


def _windows_thread_seconds(native_id: int) -> float | None:
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenThread(_THREAD_QUERY_LIMITED_INFORMATION, False, int(native_id))
    if not handle:
        handle = kernel32.OpenThread(_THREAD_QUERY_INFORMATION, False, int(native_id))
    if not handle:
        return None
    try:
        creation, exit_, kernel, user = _FILETIME(), _FILETIME(), _FILETIME(), _FILETIME()
        ok = kernel32.GetThreadTimes(
            handle, ctypes.byref(creation), ctypes.byref(exit_), ctypes.byref(kernel), ctypes.byref(user)
        )
        if not ok:
            return None
        return kernel.seconds() + user.seconds()
    finally:
        kernel32.CloseHandle(handle)


def _linux_thread_seconds(native_id: int) -> float | None:
    try:
        text = Path(f"/proc/self/task/{int(native_id)}/stat").read_text(encoding="ascii")
    except OSError:
        return None
    # Fields after the parenthesised comm; utime and stime are the 14th and 15th.
    tail = text.rsplit(")", 1)[-1].split()
    try:
        ticks = int(tail[11]) + int(tail[12])
    except (IndexError, ValueError):
        return None
    return ticks / float(os.sysconf("SC_CLK_TCK"))


def thread_cpu_seconds(native_id: int) -> float | None:
    """Cumulative CPU seconds of one OS thread, or None where unsupported."""
    if native_id is None:
        return None
    try:
        if sys.platform.startswith("win"):
            return _windows_thread_seconds(native_id)
        if sys.platform.startswith("linux"):
            return _linux_thread_seconds(native_id)
    except Exception:
        return None
    return None


def supported() -> bool:
    return sys.platform.startswith("win") or sys.platform.startswith("linux")


def snapshot_threads() -> dict[int, tuple[str, float]]:
    """{native_id: (name, cpu_seconds)} for every live Python thread."""
    out: dict[int, tuple[str, float]] = {}
    for thread in threading.enumerate():
        native_id = getattr(thread, "native_id", None)
        if native_id is None:
            continue
        seconds = thread_cpu_seconds(native_id)
        if seconds is None:
            continue
        out[int(native_id)] = (thread.name, seconds)
    return out


# ---------------------------------------------------------------------
# The gauge
# ---------------------------------------------------------------------
class ThreadCpuGauge:
    """Daemon sampler: one thread-CPU record per interval, hot threads logged."""

    def __init__(
        self,
        *,
        interval_seconds: float = INTERVAL_SECONDS,
        hot_fraction: float = HOT_THREAD_FRACTION,
        log_path: Path | str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._interval = max(0.05, float(interval_seconds))
        self._hot = max(0.0, float(hot_fraction))
        self._log_path = Path(log_path) if log_path is not None else None
        self._logger = logger or logging.getLogger(__name__)
        self._main_native_id = threading.main_thread().native_id
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._records = 0
        self._hour_records = 0
        self._hour_key = ""
        self.last_record: dict | None = None
        self.hot_seen: list[dict] = []

    # -- lifecycle -----------------------------------------------------
    def start(self) -> None:
        if self._worker is not None or not supported():
            return
        self._worker = threading.Thread(target=self._loop, name="thread-cpu-gauge", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.join(timeout=1.0)

    @property
    def records_written(self) -> int:
        return self._records

    # -- sampling ------------------------------------------------------
    def _loop(self) -> None:
        before = snapshot_threads()
        started = time.perf_counter()
        while not self._stop.wait(self._interval):
            after = snapshot_threads()
            elapsed = max(1e-6, time.perf_counter() - started)
            self.tick(before, after, elapsed)
            before, started = after, time.perf_counter()

    def tick(self, before: dict, after: dict, elapsed: float) -> dict:
        """Fold two snapshots into one record. Pure enough to test directly."""
        rows = []
        for native_id, (name, seconds) in after.items():
            previous = before.get(native_id)
            if previous is None:
                continue  # a thread born inside the interval has no baseline
            delta = max(0.0, seconds - previous[1])
            rows.append(
                {
                    "thread": name,
                    "native_id": native_id,
                    "cpu_s": round(delta, 3),
                    "core_fraction": round(delta / elapsed, 3),
                    "gui": native_id == self._main_native_id,
                }
            )
        rows.sort(key=lambda row: row["cpu_s"], reverse=True)
        hot = [row for row in rows if not row["gui"] and row["core_fraction"] >= self._hot]
        record = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "interval_s": round(elapsed, 1),
            "threads": len(rows),
            "process_core_fraction": round(sum(row["cpu_s"] for row in rows) / elapsed, 3),
            "top": rows[:TOP_THREADS],
            "hot": [row["thread"] for row in hot],
        }
        self.last_record = record
        for row in hot:
            self.hot_seen.append(row)
            self._logger.warning(
                "Hot thread: %r used %.0f%% of one core over the last %.0f s "
                "(the GUI thread shares the interpreter lock with it).",
                row["thread"],
                row["core_fraction"] * 100.0,
                elapsed,
            )
        self._write(record)
        return record

    def _write(self, record: dict[str, Any]) -> None:
        hour = datetime.now().strftime("%Y-%m-%d %H")
        if hour != self._hour_key:
            self._hour_key = hour
            self._hour_records = 0
        if self._hour_records >= MAX_RECORDS_PER_HOUR:
            return
        try:
            path = self._log_path if self._log_path is not None else log_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except Exception:
            return  # diagnostics must never break the session
        self._records += 1
        self._hour_records += 1


def install() -> ThreadCpuGauge | None:
    """Start the gauge for this process. Never raises."""
    try:
        gauge = ThreadCpuGauge(log_path=log_path())
        gauge.start()
        return gauge
    except Exception:
        return None


# ---------------------------------------------------------------------
# Reading the log back
# ---------------------------------------------------------------------
def load_records(path: Path | str | None = None) -> list[dict]:
    target = Path(path) if path is not None else log_path()
    if not target.exists():
        return []
    records = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def summarize(path: Path | str | None = None, *, top: int = 10) -> list[dict]:
    """CPU seconds per thread name across the whole log, busiest first."""
    totals: dict[str, float] = {}
    hot_ticks: dict[str, int] = {}
    for record in load_records(path):
        for row in record.get("top") or []:
            name = str(row.get("thread") or "?")
            totals[name] = totals.get(name, 0.0) + float(row.get("cpu_s") or 0.0)
        for name in record.get("hot") or []:
            hot_ticks[name] = hot_ticks.get(name, 0) + 1
    rows = [
        {"thread": name, "cpu_s": round(seconds, 1), "hot_ticks": hot_ticks.get(name, 0)}
        for name, seconds in totals.items()
    ]
    rows.sort(key=lambda row: row["cpu_s"], reverse=True)
    return rows[:top]


def _main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else log_path()
    rows = summarize(target)
    if not rows:
        print(f"No thread-CPU records in {target}")
        return 0
    print(f"{'cpu s':>9}  {'hot':>4}  thread")
    for row in rows:
        print(f"{row['cpu_s']:9.1f}  {row['hot_ticks']:4d}  {row['thread']}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    raise SystemExit(_main())
