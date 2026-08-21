from __future__ import annotations

import gc
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


def test_gui_thread_gc_waits_for_input_idle_and_defers_full_sweep():
    from ui.app import UiActivityMonitor, install_gui_thread_gc

    swept: list[int] = []
    clock = [100.0]
    activity = UiActivityMonitor(_app, clock=lambda: clock[0])

    timer = install_gui_thread_gc(
        _app,
        interval_ms=50,
        activity_monitor=activity,
        collector=lambda generation=2: swept.append(generation),
    )
    try:
        assert not gc.isenabled()
        assert timer.isActive()
        assert timer.interval() == 50

        # A click immediately before a tick protects the interaction.
        activity.mark_input()
        timer.timeout.emit()
        assert swept == []

        clock[0] += 0.3
        timer.timeout.emit()
        assert swept == [0]

        # Make a full sweep due while input remains too recent for gen2.
        for _ in range(28):
            activity.mark_input()
            timer.timeout.emit()
        assert swept == [0]

        clock[0] += 0.3
        timer.timeout.emit()
        assert swept == [0, 0]

        clock[0] += 2.0
        timer.timeout.emit()
        assert swept == [0, 0, 2]

    finally:
        timer.stop()
        gc.enable()
