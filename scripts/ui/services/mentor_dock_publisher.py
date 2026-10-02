"""The desk's half of "Dock the Mentor": publish where the Mentor tab's placeholder sits.

The desk is the one writer of ``MENTOR_DOCK_FILE``: its top-level window handle, the
placeholder's global rect (device-independent px) and device pixel ratio, whether the
Mentor tab is current and the desk is on screen, and a timestamp. The Mentor app (its
own process) reads it and floats a frameless window over that rect. Writes are
debounced and go through one background thread (tmp file + replace); a failed write is
logged at debug and never raises.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QEvent, QObject, QPoint, QTimer
from PySide6.QtWidgets import QWidget

#: Edits within this window are one write.
DEBOUNCE_MS = 200
#: The payload's schema version.
VERSION = 1

_WINDOW_EVENTS = {
    QEvent.Type.Move,
    QEvent.Type.Resize,
    QEvent.Type.Show,
    QEvent.Type.Hide,
    QEvent.Type.WindowStateChange,
}


def write_dock_file(path: Path, payload: dict) -> bool:
    """Atomic write (tmp + replace); False on any failure, never raises."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except Exception:  # noqa: BLE001 - a lost dock update never costs the desk
        logging.debug("Mentor dock file could not be written.", exc_info=True)
        return False


def read_dock_file(path: Path) -> dict | None:
    """The desk's last payload, or None when missing or unreadable."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - missing or half-written is "unknown"
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("rect"), dict):
        return None
    return payload


class MentorDockPublisher(QObject):
    """Watches the tab widget, the placeholder and the desk window; publishes on change."""

    def __init__(
        self,
        tabs,
        placeholder: QWidget,
        *,
        path: Path | None = None,
        writer: Callable[[Path, dict], Any] | None = None,
        clock: Callable[[], float] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent or placeholder)
        if path is None:
            from project_paths import MENTOR_DOCK_FILE

            path = MENTOR_DOCK_FILE
        self.path = Path(path)
        self._tabs = tabs
        self._placeholder = placeholder
        self._writer = writer
        self._clock = clock or time.time
        self._window: QWidget | None = None
        self._io: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self._last: dict | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self.publish_now)
        tabs.currentChanged.connect(lambda _index: self.request())
        placeholder.installEventFilter(self)

    # -- triggers -----------------------------------------------------------
    def request(self) -> None:
        """Publish within DEBOUNCE_MS (a burst of moves is one write)."""
        self._watch_window()
        self._timer.start()

    def _watch_window(self) -> None:
        window = self._placeholder.window()
        if window is not None and window is not self._window:
            if self._window is not None:
                self._window.removeEventFilter(self)
            self._window = window
            window.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override
        kind = event.type()
        if kind in _WINDOW_EVENTS or (watched is self._placeholder and kind == QEvent.Type.ParentChange):
            self.request()
        return False

    # -- the payload --------------------------------------------------------
    def payload(self) -> dict:
        """What the Mentor needs: built on the Qt thread from geometry only (cheap)."""
        placeholder = self._placeholder
        window = placeholder.window()
        origin = placeholder.mapToGlobal(QPoint(0, 0))
        # internalWinId: the native handle when it exists, 0 otherwise (never creates one).
        hwnd = int(window.internalWinId() or 0) if window is not None else 0
        desk_visible = bool(window is not None and window.isVisible() and not window.isMinimized())
        now = float(self._clock())
        return {
            "version": VERSION,
            "hwnd": hwnd,
            "pid": os.getpid(),
            "rect": {"x": origin.x(), "y": origin.y(), "w": placeholder.width(), "h": placeholder.height()},
            "dpr": float(placeholder.devicePixelRatioF()),
            # Current tab AND its desk page shown (another page like Journal hides the Mentor).
            "tab_current": self._tabs.currentWidget() is placeholder
            and (window is None or placeholder.isVisibleTo(window)),
            "desk_visible": desk_visible,
            "at": now,
            "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }

    def publish_now(self) -> None:
        """Build the payload here; write it on the one background thread."""
        self._watch_window()
        try:
            payload = self.payload()
        except Exception:  # noqa: BLE001 - a geometry read never costs the desk
            logging.debug("Mentor dock payload could not be built.", exc_info=True)
            return
        last = self._last
        if last is not None and not payload["tab_current"] and not last["tab_current"]:
            return  # the tab stays hidden: a desk move while on Setups writes nothing
        if last is not None and {k: v for k, v in payload.items() if k not in ("at", "at_utc")} == {
            k: v for k, v in last.items() if k not in ("at", "at_utc")
        }:
            return
        self._last = payload
        if self._writer is not None:
            self._writer(self.path, payload)
            return
        if self._io is None:
            self._io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mentor-dock-publish")
        self._io.submit(self._write_latest)

    def _write_latest(self) -> None:
        with self._lock:
            payload = self._last
            if payload is not None:
                write_dock_file(self.path, payload)

    def last_payload(self) -> dict | None:
        return self._last

    def shutdown(self) -> None:
        self._timer.stop()
        if self._io is not None:
            self._io.shutdown(wait=False)

