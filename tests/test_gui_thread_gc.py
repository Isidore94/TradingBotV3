"""All cyclic GC happens on the GUI thread, and it actually happens.

Two properties, and the second one is the expensive lesson. Automatic
collection is disabled process-wide so Qt wrapper destructors stay on the
owning thread, which makes this timer the ONLY collector in the process. A
"wait until the trader stops clicking" with no upper bound is therefore
indistinguishable from "never collect": on 2026-08-21 the desk reached 8 GB in
ninety minutes of continuous use and then froze for 298 seconds in the sweep
that finally ran.

So: activity may DELAY a sweep, never CANCEL one.
"""

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


def _harness(**kwargs):
    """(timer, activity, clock, swept) with collection recorded, not performed."""
    from ui.app import UiActivityMonitor, install_gui_thread_gc

    swept: list[int] = []
    clock = [100.0]
    activity = UiActivityMonitor(_app, clock=lambda: clock[0])
    timer = install_gui_thread_gc(
        _app,
        interval_ms=50,
        activity_monitor=activity,
        collector=lambda generation=2: swept.append(generation),
        **kwargs,
    )
    return timer, activity, clock, swept


def test_gui_thread_gc_waits_for_input_idle_and_defers_full_sweep():
    timer, activity, clock, swept = _harness()
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

        # Make a full sweep due while input remains too recent for gen2. The
        # young deadline is deliberately raised out of the way here so this
        # test still measures the IDLENESS rule and nothing else.
        timer._gc_controller.young_deadline_ticks = 10_000
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


def test_continuous_input_cannot_starve_the_young_sweep():
    """The failing case: the trader never stops, so nothing is ever collected."""
    timer, activity, clock, swept = _harness()
    try:
        controller = timer._gc_controller
        for _ in range(controller.young_deadline_ticks):
            activity.mark_input()
            timer.timeout.emit()
        assert swept == [], "a sweep landed on top of live interaction"

        # One tick past the deadline: activity has had its say, and the sweep
        # runs anyway.
        activity.mark_input()
        timer.timeout.emit()
        assert swept == [0]
    finally:
        timer.stop()
        gc.enable()


def test_continuous_input_cannot_starve_the_full_sweep():
    timer, activity, clock, swept = _harness()
    try:
        controller = timer._gc_controller
        deadline = controller.full_deadline_ticks
        for _ in range(controller.full_every_ticks + deadline):
            activity.mark_input()
            timer.timeout.emit()
        assert 2 in swept, "the full sweep never ran under continuous input"
    finally:
        timer.stop()
        gc.enable()


def test_the_full_sweep_deadline_is_measured_from_when_it_came_due():
    """Not from process start, and not reset by the ticks that skip it."""
    timer, activity, clock, swept = _harness(full_every_ticks=4, full_deadline_ticks=3)
    try:
        for _ in range(3):
            activity.mark_input()
            timer.timeout.emit()
        assert 2 not in swept
        # Tick 4 makes it due; ticks 5 and 6 are still inside the deadline.
        for _ in range(3):
            activity.mark_input()
            timer.timeout.emit()
        assert 2 not in swept
        activity.mark_input()
        timer.timeout.emit()
        assert 2 in swept
    finally:
        timer.stop()
        gc.enable()


def test_an_idle_desk_still_prefers_quiet_over_the_deadline():
    """The deadline is a floor on collection, not a new schedule: when the
    trader IS idle, the sweep happens on idleness as before."""
    timer, activity, clock, swept = _harness()
    try:
        clock[0] += 5.0
        timer.timeout.emit()
        assert swept == [0]
    finally:
        timer.stop()
        gc.enable()


def test_a_deadline_of_zero_means_never_defer():
    """Belt and braces: the knob's degenerate value collects every tick."""
    timer, activity, clock, swept = _harness(young_deadline_ticks=0)
    try:
        activity.mark_input()
        timer.timeout.emit()
        assert swept == [0]
    finally:
        timer.stop()
        gc.enable()
