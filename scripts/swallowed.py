"""Log a deliberately swallowed exception with its reason, rate-limited per reason.

Use in place of a bare ``except ...: pass`` so a failure that the caller must
survive still leaves a trace. It never raises: a broken log must not turn a
swallow into a crash.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

_LOG = logging.getLogger("tradingbot.swallowed")
_LOCK = threading.Lock()
_LAST_LOGGED: dict[str, float] = {}
_COUNTS: dict[str, int] = {}

#: Seconds between two log lines for the same reason (hot loops stay quiet).
DEFAULT_INTERVAL_S = 300.0


def note_swallowed(
    reason: str,
    exc: BaseException | None = None,
    *,
    quiet: bool = False,
    level: int | None = None,
    interval_s: float = DEFAULT_INTERVAL_S,
    logger: logging.Logger | Any | None = None,
) -> None:
    """Count one swallowed failure and log ``reason`` at most once per ``interval_s``.

    ``quiet`` logs at DEBUG: for expected, harmless cases (temp-file cleanup,
    a widget already deleted, a value that simply does not parse).
    """
    try:
        now = time.monotonic()
        with _LOCK:
            count = _COUNTS.get(reason, 0) + 1
            _COUNTS[reason] = count
            last = _LAST_LOGGED.get(reason)
            if last is not None and now - last < max(0.0, float(interval_s)):
                return
            _LAST_LOGGED[reason] = now
        target = logger if logger is not None else _LOG
        detail = f": {type(exc).__name__}: {exc}" if exc is not None else ""
        suffix = f" (seen {count}x)" if count > 1 else ""
        if level is None:
            level = logging.DEBUG if quiet else logging.WARNING
        target.log(level, "swallowed %s%s%s", reason, detail, suffix)
    except Exception:  # logging must never break the caller
        return


def swallowed_counts() -> dict[str, int]:
    """Snapshot of how often each reason has been swallowed in this process."""
    with _LOCK:
        return dict(_COUNTS)


def reset_swallowed() -> None:
    """Forget counts and rate-limit state (tests only)."""
    with _LOCK:
        _COUNTS.clear()
        _LAST_LOGGED.clear()
