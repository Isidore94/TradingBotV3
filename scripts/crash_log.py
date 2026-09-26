"""The desk's native-crash log (``gui_crash.log``), written by faulthandler.

faulthandler writes the thread dump without a time. This module stamps the
log instead: a header with a zoned time each time it is armed, a stamp for
any unstamped crash left by the last run (the file's last-write time is the
crash time), and a timed line for each Qt fatal/critical message.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import TextIO

LOG_NAME = "gui_crash.log"
START_MARK = "=== GUI start "
STAMP_MARK = "=== crash above written "
CRASH_MARKS = ("Fatal Python error:", "Windows fatal exception:")
#: How much of the log's tail is read to find the last run's section.
_TAIL_BYTES = 256 * 1024

# faulthandler keeps a borrowed reference to this handle; a module global
# stops the file (and its fd) from being garbage-collected out from under it.
_HANDLE: TextIO | None = None


def _stamp(moment: datetime) -> str:
    return moment.astimezone().isoformat(timespec="seconds")


def unstamped_crash_line(path: Path) -> str | None:
    """A stamp line for a crash dump after the last header that has none yet."""
    try:
        size = path.stat().st_size
        mtime = path.stat().st_mtime
        with open(path, "rb") as handle:
            handle.seek(max(0, size - _TAIL_BYTES))
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    section = tail.rsplit(START_MARK, 1)[-1]
    if not any(mark in section for mark in CRASH_MARKS) or STAMP_MARK in section:
        return None
    # faulthandler marks the crashing thread "Current thread" only when it has Python state.
    where = "" if "Current thread " in section else "; no Current thread: a native (non-Python) thread crashed"
    return f"{STAMP_MARK}{_stamp(datetime.fromtimestamp(mtime))} (file last write{where}) ===\n"


def arm(log_dir: Path, *, enable) -> TextIO:
    """Open the log, stamp the last run's crash, write a header, then call ``enable(file)``."""
    global _HANDLE
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / LOG_NAME
    previous = unstamped_crash_line(path)
    handle = open(path, "a", buffering=1, encoding="utf-8")
    if previous:
        handle.write(previous)
    handle.write(f"\n{START_MARK}{_stamp(datetime.now())} pid={os.getpid()} ===\n")
    handle.flush()
    _HANDLE = handle
    enable(handle)
    return handle


def note(text: str) -> None:
    """Append one timed line to the armed log; never raises."""
    handle = _HANDLE
    if handle is None:
        return
    try:
        handle.write(f"[{_stamp(datetime.now())}] {text}\n")
        handle.flush()
    except Exception:
        return
