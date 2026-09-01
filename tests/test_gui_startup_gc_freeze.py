"""Desk snappiness packet 2, item 2: freeze the startup heap after one sweep.

All cyclic collection runs on the GUI thread by design - `gc.disable()` is
process-wide, so that timer is the only collector, and that design is not what
this changes. What changes is how much heap each of its sweeps has to walk.

On 2026-08-31 gen-0 sweeps averaged ~300 ms and full sweeps ~770 ms, and 6.5
minutes of that day's ~78 minutes of GUI freeze was the collector itself. Most
of the graph it walks is the startup graph - the widget tree, the theme, every
imported module - which lives for the whole process and can never be garbage.
`gc.freeze()` moves it into a permanent generation the collector does not scan.

Order matters and is what these pin: **collect, then freeze.** Freezing first
would make every piece of startup garbage permanent instead of collecting it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt

pytest.importorskip("PySide6")


class _RecordingGc:
    """Stands in for the `gc` module and remembers the order of the calls."""

    def __init__(self):
        self.calls: list[tuple] = []

    def collect(self, generation=None):
        self.calls.append(("collect", generation))
        return 0

    def freeze(self):
        self.calls.append(("freeze", None))

    def disable(self):
        self.calls.append(("disable", None))

    def __getattr__(self, name):  # anything else falls through to the real gc
        import gc as real_gc

        return getattr(real_gc, name)


def _run_main(monkeypatch, recorder):
    """Drive `main()` far enough to pass the startup-GC step, no further."""
    from ui import app as app_mod

    monkeypatch.setattr(app_mod, "gc", recorder)

    class _Window:
        def __init__(self, state):
            self.state = state

        def show(self):
            pass

    class _App:
        def __init__(self, *args):
            self.ui_activity_monitor = None

        def setApplicationName(self, name):
            pass

        def setOrganizationName(self, name):
            pass

        def installEventFilter(self, obj):
            pass

        def exec(self):
            return 0

        @staticmethod
        def setAttribute(*args):
            pass

    monkeypatch.setattr(app_mod, "MainWindow", _Window)
    monkeypatch.setattr(app_mod, "QApplication", _App)
    monkeypatch.setattr(app_mod, "UiActivityMonitor", lambda app: object())
    monkeypatch.setattr(app_mod, "install_qt_message_rate_limit", lambda: None)
    monkeypatch.setattr(app_mod, "_print_qt_message_tally", lambda: None)
    monkeypatch.setattr(app_mod, "apply_theme", lambda *a, **k: None)
    monkeypatch.setattr(app_mod, "install_gui_thread_gc", lambda *a, **k: None)
    monkeypatch.setattr(app_mod, "_available_screen_size", lambda: (1920, 1080))

    import ui.stall_watchdog as watchdog_mod

    monkeypatch.setattr(watchdog_mod, "install", lambda window: None)

    class _State:
        theme_name = "dark"
        compact_density = False
        ui_scale = "auto"
        workspace_mode = "workspace"

        def save(self):
            pass

    monkeypatch.setattr(app_mod.UiState, "load", staticmethod(lambda: _State()))
    return app_mod.main([])


def test_startup_collects_then_freezes(monkeypatch):
    recorder = _RecordingGc()

    _run_main(monkeypatch, recorder)

    kinds = [name for name, _arg in recorder.calls]
    assert "collect" in kinds, "the startup garbage is swept before anything is frozen"
    assert "freeze" in kinds
    assert kinds.index("collect") < kinds.index("freeze"), (
        "freezing first would make startup GARBAGE permanent instead of collecting it"
    )


def test_the_startup_sweep_is_a_full_one(monkeypatch):
    """Gen 2, not gen 0: the point is to leave nothing collectable behind
    before the survivors become permanent."""
    recorder = _RecordingGc()

    _run_main(monkeypatch, recorder)

    collects = [arg for name, arg in recorder.calls if name == "collect"]
    assert 2 in collects


def test_the_collector_design_is_untouched(monkeypatch):
    """The GUI-thread collector, its cadence and its bounded waits are what
    keep Qt destructors on the owning thread (2026-07-29 crash). This packet
    only shrinks what they scan."""
    import inspect

    from ui import app as app_mod

    source = inspect.getsource(app_mod.install_gui_thread_gc)
    assert "gc.disable()" in source, "automatic collection stays off, process-wide"
    assert "freeze" not in source, "the freeze belongs to startup, not to the controller"

    controller = inspect.getsource(app_mod._GuiGcController)
    assert "freeze" not in controller
