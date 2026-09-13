"""Is anyone at the desk? - WISHLIST 10J, the smallest primitive that answers it.

The Trade Mentor is the first thing on this desk that INTERRUPTS. Everything
else waits to be looked at; a prompt puts a card in front of the chart, and a
card in front of a chart nobody is watching is a card that expires unanswered
and leaves a hole in the record that reads exactly like "the trader had no view".
So presence is asked before a prompt is delivered, not after.

**One primitive, on purpose.** Windows' `GetLastInputInfo` reports how long the
whole session has been without keyboard or mouse input. That is enough for both
halves of the question this desk actually has:

* the trader walked away - idle climbs past the grace and the prompt is skipped;
* the workstation is LOCKED - input stops at the lock, so idle climbs the same
  way and the prompt is skipped inside the same grace window.

A `WTSRegisterSessionNotification` lock hook would answer the second half a few
minutes sooner. It is not built: it costs a message-only window and a native
event filter on the Qt thread for a distinction whose worst case is one skipped
prompt at the top of an hour, and the service still takes an injected
`session_locked` callable, so the hook can be added later without touching a
caller.

**Uncertainty is never absence.** Off Windows, or when the call fails, this
returns `None` and the caller treats the trader as present. Missing data is
uncertainty (plan.md sec 5), and reading "I cannot tell" as "nobody is there"
would silently switch the whole feature off on a machine where it works fine.
Nothing here writes, detects, scores, gates or alerts.
"""

from __future__ import annotations

import logging
import sys


def idle_seconds() -> float | None:
    """Seconds since the last keyboard or mouse input, or None if unmeasurable.

    Cheap: two ctypes calls into `user32`/`kernel32`, no allocation beyond one
    small struct. Safe to call from the Qt thread on a 60-second timer.
    """
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes

        class _LastInputInfo(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

        info = _LastInputInfo()
        info.cbSize = ctypes.sizeof(_LastInputInfo)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            return None
        now_ticks = ctypes.windll.kernel32.GetTickCount()
        # Both are 32-bit millisecond tick counts and both wrap after ~49.7
        # days of uptime. The mask makes the subtraction wrap with them, so an
        # uptime rollover cannot produce a negative - or a 49-day - idle time.
        elapsed_ms = (now_ticks - info.dwTime) & 0xFFFFFFFF
        return float(elapsed_ms) / 1000.0
    except Exception:  # noqa: BLE001 - never the reason a prompt is skipped
        logging.debug("Idle time could not be read.", exc_info=True)
        return None


def session_locked() -> bool:
    """Placeholder for a real lock signal: not modelled, so never asserted.

    Always False. The lock case is carried by `idle_seconds` (input stops at the
    lock), and claiming "locked" here without a session hook would be a fact
    nobody measured.
    """
    return False
