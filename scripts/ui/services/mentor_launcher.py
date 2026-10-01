"""The desk's one link to the Trade Mentor app: start it, or bring it to the front.

``launch_or_focus`` returns at once; the slot probe, the focus ping and the Popen run
on a daemon thread, never the desk's Qt thread. The desk never opens the tunnel and
never imports the app's window. In a frozen desk it does nothing: the app runs from
source only.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable

_DETACHED = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

#: Local setting: on = the Trade Mentor app owns the Mentor (card, slots file, journal
#: events) and the desk builds none of it. On by default since 2026-09-30 (P15b); an
#: explicit off (``false``/``0``/``no``/``off``) restores the desk's own Mentor popup.
MENTOR_APP_ENABLED_KEY = "mentor_app_enabled"
MENTOR_APP_DEFAULT = True
_OFF_WORDS = ("0", "false", "no", "off")
_ON_WORDS = ("1", "true", "yes", "on")
#: Passed only by the desk's launch: the app exits after the desk does.
FOLLOW_DESK_FLAG = "--follow-desk"


def mentor_app_enabled() -> bool:
    """On unless explicitly off: a bool is itself, ``false``/``0``/``no``/``off`` is off, unset or unreadable is on."""
    from project_paths import get_local_setting

    raw = get_local_setting(MENTOR_APP_ENABLED_KEY, MENTOR_APP_DEFAULT)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    word = str(raw if raw is not None else "").strip().lower()
    if word in _OFF_WORDS:
        return False
    if word in _ON_WORDS:
        return True
    return MENTOR_APP_DEFAULT


def _python() -> str:
    """pythonw beside the desk's interpreter when there is one (no console window)."""
    exe = Path(sys.executable)
    windowless = exe.with_name("pythonw.exe")
    return str(windowless if windowless.exists() else exe)


def _entry() -> Path:
    from project_paths import ROOT_DIR

    return Path(ROOT_DIR) / "launch_mentor.py"


def _probe() -> bool | None:
    from single_instance import MENTOR_LOCK_KEY, slot_is_free

    return slot_is_free(MENTOR_LOCK_KEY)


def _ping() -> bool:
    from mentor_app.focus_link import send_focus_ping

    return send_focus_ping(1000)


def _work(popen: Callable[..., Any], probe: Callable[[], bool | None], ping: Callable[[], bool]) -> str:
    try:
        if probe() is False:
            # Running already: bring it forward. A failed ping means it is still starting.
            return "focused" if ping() else "starting"
        entry = _entry()
        # Launched by the desk: the app closes itself once the desk has closed.
        popen([_python(), str(entry), FOLLOW_DESK_FLAG], cwd=str(entry.parent), close_fds=True, creationflags=_DETACHED)
        return "launched"
    except Exception as exc:  # noqa: BLE001 - a failed launch is logged, never raised into the desk
        logging.warning("Trade Mentor launch failed: %s", exc)
        return "failed"


def launch_or_focus(
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    probe: Callable[[], bool | None] | None = None,
    ping: Callable[[], bool] | None = None,
) -> threading.Thread | None:
    """Start or focus the app off the Qt thread; returns the worker thread (None when frozen)."""
    if getattr(sys, "frozen", False):
        logging.info("Trade Mentor runs from source; the frozen desk does not start it.")
        return None
    thread = threading.Thread(
        target=_work, args=(popen, probe or _probe, ping or _ping), name="mentor-launcher", daemon=True
    )
    thread.start()
    return thread
