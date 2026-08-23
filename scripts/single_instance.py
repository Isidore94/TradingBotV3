"""One desk per machine (R10.A / Sol blocker 3, defence in depth).

`launch_gui_auto.ps1` already refuses to start a second desk - but only on that
path. A double-click, a shortcut, a terminal, a second scheduled task and the
frozen exe all reach `launch_gui.py` directly, and R10.0 proved concurrent desks
actually happened: pid 31848 overlapped three others on 2026-08-20, the worst by
**3.8 hours**.

**This is not the transaction.** The outcome finalizer fences itself with
`local_writer_lock` and re-reads the authoritative checkpoint inside that fence,
and it has to stay correct with two desks running, because a guard can always be
overridden and a second process can always be started another way. This guard
exists so the ordinary case never gets there.

**It fails OPEN when the primitive is missing.** If neither a named mutex nor a
byte-range file lock is available on this box, that is a reason to warn, not a
reason to refuse to start the trader's desk. It fails CLOSED only when it can
see that another desk holds the slot - which is the case it was written for.
"""

from __future__ import annotations

import logging
import os
import socket
from contextlib import contextmanager

#: One name for the desk slot on this machine. Not derived from a path: the
#: source launch and the frozen exe live in different directories and are still
#: the same desk.
DESK_LOCK_KEY = "tradingbotv3-desk"

#: Short on purpose. Waiting is pointless - the other desk holds this for its
#: whole session - so the only question is whether somebody has it right now.
ACQUIRE_TIMEOUT_SECONDS = 2.0

#: The trader's escape hatch, for the day two desks are genuinely wanted.
OVERRIDE_FLAG = "--allow-second-instance"


class AnotherDeskIsRunning(RuntimeError):
    """Another process on this machine holds the desk slot."""


@contextmanager
def desk_slot(*, allow_second: bool = False, key: str = DESK_LOCK_KEY):
    """Hold the machine's desk slot for the length of the session.

    Raises :class:`AnotherDeskIsRunning` when somebody else has it. Yields a
    short description of what protection is actually in force, so the caller can
    log the honest answer rather than assuming one.
    """
    if allow_second:
        yield "override: second instance allowed by flag"
        return

    try:
        from local_writer_lock import LocalLockUnavailable, local_writer_lock
    except Exception:
        logging.warning("Single-instance guard unavailable (import failed); continuing.")
        yield "guard unavailable: import failed"
        return

    try:
        manager = local_writer_lock(key, timeout_seconds=ACQUIRE_TIMEOUT_SECONDS)
        entered = manager.__enter__()
    except LocalLockUnavailable as exc:
        message = str(exc)
        if "no machine-local exclusion primitive" in message:
            # The box cannot exclude anything. Refusing to start the desk over
            # that would be a worse failure than the duplicate it prevents, and
            # the outcome transaction is fenced independently.
            logging.warning("Single-instance guard has no primitive on this machine: %s", message)
            yield "guard unavailable: no OS primitive"
            return
        raise AnotherDeskIsRunning(
            f"another TradingBotV3 desk is already running on {socket.gethostname()} "
            f"(pid {os.getpid()} tried to start a second one). Close the other window, "
            f"or pass {OVERRIDE_FLAG} if you really want two."
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("Single-instance guard failed (%s); continuing.", exc)
        yield "guard unavailable: unexpected error"
        return

    try:
        held = getattr(entered, "mutex", "?")
        file_lock = getattr(entered, "file_lock", "?")
        yield f"desk slot held (mutex: {held}, file lock: {file_lock})"
    finally:
        manager.__exit__(None, None, None)
