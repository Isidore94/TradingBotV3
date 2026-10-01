"""The Mentor's half of "Dock the Mentor": float frameless over the desk's Mentor tab.

The desk publishes ``MENTOR_DOCK_FILE`` (``ui.services.mentor_dock_publisher``). While
docked, a 250 ms timer stats it (a tiny local file; read only when its mtime moves),
puts the window on the placeholder's rect, and on Windows makes the desk window its
owner so it floats over the desk only and minimizes with it. Hidden while the tab is
not current or the desk is minimized. A missing file, or one >10 s old whose desk
window is gone, for 10 s: undock with a note. Every OS call is injectable.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QByteArray, QObject, QRect, Qt, QTimer, Signal

#: How often the desk's file is checked while docked.
POLL_MS = 250
#: A file this old whose desk window is gone (or no file) for this long: undock.
STALE_SECONDS = 10.0
#: The app's persisted on/off (MentorChatStore state).
STATE_KEY = "desk_dock"
DOCK_FLAGS = Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint


def _win_set_owner(child_hwnd: int, owner_hwnd: int) -> bool:
    """Windows: make ``owner_hwnd`` own ``child_hwnd`` (0 clears it). False elsewhere."""
    if sys.platform != "win32" or not child_hwnd:
        return False
    try:
        import ctypes

        user32 = ctypes.windll.user32
        setter = getattr(user32, "SetWindowLongPtrW", None) or user32.SetWindowLongW
        setter.restype = ctypes.c_void_p
        setter.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
        gwlp_hwndparent = -8
        setter(ctypes.c_void_p(child_hwnd), gwlp_hwndparent, ctypes.c_void_p(owner_hwnd or 0))
        return True
    except Exception:  # noqa: BLE001 - docking without an owner still works
        logging.debug("Mentor dock: could not set the owner window.", exc_info=True)
        return False


def _win_get_owner(child_hwnd: int) -> int | None:
    """Windows: the window that owns ``child_hwnd`` now (0 = none). None elsewhere (unknown)."""
    if sys.platform != "win32" or not child_hwnd:
        return None
    try:
        import ctypes

        get_window = ctypes.windll.user32.GetWindow
        get_window.restype = ctypes.c_void_p
        get_window.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        gw_owner = 4
        return int(get_window(ctypes.c_void_p(child_hwnd), gw_owner) or 0)
    except Exception:  # noqa: BLE001
        return None


def _win_is_window(hwnd: int) -> bool:
    """Windows: is the desk's window still alive? Elsewhere: assume yes."""
    if sys.platform != "win32":
        return True
    if not hwnd:
        return False
    try:
        import ctypes

        is_window = ctypes.windll.user32.IsWindow
        is_window.argtypes = (ctypes.c_void_p,)
        return bool(is_window(ctypes.c_void_p(hwnd)))
    except Exception:  # noqa: BLE001
        return True


def _default_path() -> Path:
    from project_paths import MENTOR_DOCK_FILE

    return MENTOR_DOCK_FILE


class DeskDock(QObject):
    """Docks one top-level window over the desk's Mentor tab; one owner of its poll timer."""

    #: (docked) - the state flipped (button text follows).
    changed = Signal(bool)
    #: (text) - a one-line note for the transcript (undocked on its own).
    note = Signal(str)

    def __init__(
        self,
        window,
        *,
        path: Path | None = None,
        reader: Callable[[Path], Any] | None = None,
        set_owner: Callable[[int, int], Any] | None = None,
        get_owner: Callable[[int], int | None] | None = None,
        desk_alive: Callable[[int], bool] | None = None,
        clock: Callable[[], float] | None = None,
        poll_ms: int = POLL_MS,
    ) -> None:
        super().__init__(window)
        self._window = window
        self.path = Path(path) if path is not None else _default_path()
        if reader is None:
            from ui.services.mentor_dock_publisher import read_dock_file

            reader = read_dock_file
        self._reader = reader
        self._set_owner = set_owner or _win_set_owner
        self._get_owner = get_owner or (_win_get_owner if set_owner is None else (lambda _own: None))
        self._desk_alive = desk_alive or _win_is_window
        self._clock = clock or time.time
        self.docked = False
        self._saved: tuple[Any, QByteArray] | None = None
        #: (our native handle, the owner we gave it); a flag change makes a new handle.
        self._owned = (0, 0)
        self._mtime: float | None = None
        self._payload: dict | None = None
        self._bad_since: float | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(int(poll_ms))
        self._timer.timeout.connect(self.poll)

    # -- on / off -------------------------------------------------------------
    def toggle(self) -> bool:
        if self.docked:
            self.undock()
        else:
            self.dock()
        return self.docked

    def dock(self) -> None:
        if self.docked:
            return
        win = self._window
        self._saved = (win.windowFlags(), win.saveGeometry())
        self.docked = True
        self._mtime = None
        self._payload = None
        self._bad_since = None
        win.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        win.setWindowFlags(DOCK_FLAGS)
        self._timer.start()
        self.changed.emit(True)
        self.poll()

    def undock(self, note: str = "") -> None:
        if not self.docked:
            return
        self._timer.stop()
        self.docked = False
        win = self._window
        self._apply_owner(0)
        flags, geometry = self._saved or (Qt.WindowType.Window, QByteArray())
        self._saved = None
        win.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, False)
        win.setWindowFlags(flags)
        if not geometry.isEmpty():
            win.restoreGeometry(geometry)
        win.show()
        self.changed.emit(False)
        if note:
            self.note.emit(note)

    def shutdown(self) -> None:
        self._timer.stop()

    # -- following the desk -----------------------------------------------------
    def poll(self) -> None:
        """Re-read the desk's file when it changed; place, hide or undock."""
        if not self.docked:
            return
        try:
            mtime = os.stat(self.path).st_mtime
        except OSError:
            mtime = None
        if mtime is not None and mtime != self._mtime:
            self._mtime = mtime
            self._payload = self._reader(self.path)
        elif mtime is None:
            self._payload = None
        payload = self._payload
        now = float(self._clock())
        hwnd = int((payload or {}).get("hwnd") or 0)
        fresh = payload is not None and now - float(payload.get("at") or 0.0) <= STALE_SECONDS
        # A known desk window answers "is the desk running"; without one, only a fresh file does.
        alive = payload is not None and (self._desk_alive(hwnd) if hwnd else fresh)
        if not alive:
            self._window.hide()
            if self._bad_since is None:
                self._bad_since = now
            elif now - self._bad_since >= STALE_SECONDS:
                self.undock("The desk is not showing its Mentor tab, so I undocked.")
            return
        self._bad_since = None
        self._apply(payload, hwnd)

    def _apply(self, payload: dict, hwnd: int) -> None:
        win = self._window
        if not (payload.get("tab_current") and payload.get("desk_visible")):
            win.hide()
            return
        rect = payload.get("rect") or {}
        try:
            target = QRect(int(rect["x"]), int(rect["y"]), max(1, int(rect["w"])), max(1, int(rect["h"])))
        except (KeyError, TypeError, ValueError):
            win.hide()
            return
        if win.geometry() != target:
            win.setGeometry(target)
        if not win.isVisible():
            win.show()
        self._apply_owner(hwnd)

    def _apply_owner(self, hwnd: int) -> None:
        own = int(self._window.internalWinId() or 0)
        if not own:
            return
        # Qt resets a Tool window's owner on every show, so trust the real owner over the cache.
        actual = self._get_owner(own)
        if actual is not None:
            if actual != hwnd:
                self._set_owner(own, hwnd)
            self._owned = (own, hwnd)
            return
        if (own, hwnd) == self._owned:
            return
        if self._owned[0] == own or hwnd:
            self._set_owner(own, hwnd)
        self._owned = (own, hwnd)
