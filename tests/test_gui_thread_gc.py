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


def test_gui_thread_gc_disables_automatic_collection_and_sweeps_from_timer(monkeypatch):
    from ui.app import install_gui_thread_gc

    swept: list[int] = []
    monkeypatch.setattr(gc, "collect", lambda generation=2: swept.append(generation))

    timer = install_gui_thread_gc(_app, interval_ms=50)
    try:
        assert not gc.isenabled()
        assert timer.isActive()
        assert timer.interval() == 50

        for _ in range(30):
            timer.timeout.emit()

        # Young generation every tick, the full heap on the 30th.
        assert swept == [0] * 29 + [2]
    finally:
        timer.stop()
        gc.enable()
