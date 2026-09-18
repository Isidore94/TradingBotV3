"""TJ-1 blocker 2 - the SPY bar accessor is a Qt-THREAD call, and only that.

`alert_center.journal_chart_bars` looks like a cache read and is not one: it
mutates the Alert Center's `_m5_bar_dicts` and it arms `QTimer.singleShot`. A
`singleShot` armed from a thread with no event loop NEVER FIRES, so calling it
from the Day Review read worker latched `_d1_prefetch_flush_armed` True and
killed D1 prefetch for the rest of the session (reviewer, 2026-09-17, reproduced).

So the rule is: the Qt-thread slot that starts a read calls it - once, only when
the session on screen has not closed - and hands the bars into the worker's
inputs. `DayReviewService` holds no bars reader at all; `read_day` takes
`spy_m5_bars` as an input and cannot go looking for them.

This file pins the thread, not the shape: it records `threading.get_ident()`
inside the reader and compares it with the id of the thread that asked.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Day Review page is a Qt panel")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

SESSION = "2026-09-10"
#: Friday morning, mid-session: 2026-09-10 is the last COMPLETED session and
#: 2026-09-11 is today, the one session whose bars the desk holds.
NOW = datetime(2026, 9, 11, 7, 30)
TODAY = "2026-09-11"

BARS = [
    {
        "dt": datetime(2026, 9, 11, 6, 30) + timedelta(minutes=5 * index),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000,
    }
    for index in range(6)
]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _Service:
    """Records what `read_day` was handed, and on which thread."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.threads: list[int] = []
        self.done = threading.Event()

    def read_day(self, session_date, **kwargs):
        self.threads.append(threading.get_ident())
        self.calls.append(dict(kwargs))
        self.done.set()
        return {"session_date": session_date, "spy_m5_bars": kwargs.get("spy_m5_bars") or []}


class _Reader:
    """A stand-in for `alert_center.journal_chart_bars`."""

    def __init__(self, answer=None) -> None:
        self.symbols: list[str] = []
        self.threads: list[int] = []
        self._answer = BARS if answer is None else answer

    def __call__(self, symbol):
        self.symbols.append(str(symbol))
        self.threads.append(threading.get_ident())
        if isinstance(self._answer, Exception):
            raise self._answer
        return self._answer


def _page(qapp, monkeypatch, *, service=None, reader=None, session=TODAY):
    from ui.panels.day_review_panel import DayReviewPanel

    widget = DayReviewPanel(service=service or _Service(), clock=lambda: NOW)
    if reader is not None:
        widget.set_bars_reader(reader)
    monkeypatch.setattr(widget, "_refresh_session_picker", lambda: None)
    index = widget.session_picker.findData(session)
    assert index >= 0, session
    widget.session_picker.setCurrentIndex(index)
    return widget


def _finish(qapp, widget, service, *, timeout: float = 5.0) -> None:
    assert service.done.wait(timeout), "the read never ran"
    worker = widget._worker
    if worker is not None:
        worker.wait(int(timeout * 1000))
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        qapp.processEvents()
        if not (widget._worker is not None and widget._worker.isRunning()):
            break
        time.sleep(0.01)
    qapp.processEvents()


def test_the_bars_are_read_on_the_thread_that_started_the_read(qapp, monkeypatch):
    service = _Service()
    reader = _Reader()
    widget = _page(qapp, monkeypatch, service=service, reader=reader)
    try:
        gui_thread = threading.get_ident()
        widget.reload()

        # Read BEFORE the worker was started, so by the time the worker exists
        # the answer is already in its inputs.
        assert reader.symbols == ["SPY"]
        assert reader.threads == [gui_thread], (
            "the SPY bar accessor was called off the Qt thread"
        )

        _finish(qapp, widget, service)
        assert service.threads and gui_thread not in service.threads, (
            "the read itself did not leave the Qt thread"
        )
        assert reader.threads == [gui_thread], "the worker called the accessor too"
        assert service.calls[0]["spy_m5_bars"] == BARS
    finally:
        widget.shutdown()
        widget.deleteLater()
        qapp.processEvents()


def test_the_service_holds_no_bars_reader_to_call(qapp):
    """The seam is gone, not merely unused: a service that could acquire one
    would be a worker that could call it."""
    from ui.services.day_review_service import DayReviewService

    service = DayReviewService()
    assert not hasattr(service, "set_bars_reader")
    assert not hasattr(service, "_bars_reader")
    assert not hasattr(service, "_spy_bars")


def test_the_service_only_passes_on_the_bars_it_is_handed(monkeypatch, tmp_path):
    import project_paths

    monkeypatch.setattr(project_paths, "RUNTIME_DATA_DIR", tmp_path, raising=False)
    from ui.services.day_review_service import DayReviewService

    payload = DayReviewService().read_day(SESSION, now=NOW, spy_m5_bars=BARS)
    assert payload["spy_m5_bars"] == BARS
    # ...and with none handed in, the section is empty and says so on the page.
    assert DayReviewService().read_day(SESSION, now=NOW)["spy_m5_bars"] == []


def test_a_completed_session_never_asks_for_bars_at_all(qapp, monkeypatch):
    """The accessor holds the RUNNING scanner's chart, which is today's. Asking
    it about a closed session would mutate that cache for nothing."""
    service = _Service()
    reader = _Reader()
    widget = _page(qapp, monkeypatch, service=service, reader=reader, session=SESSION)
    try:
        widget.reload()
        assert reader.symbols == [], "a closed session asked for today's bars"
        _finish(qapp, widget, service)
        assert service.calls[0]["spy_m5_bars"] == []
    finally:
        widget.shutdown()
        widget.deleteLater()
        qapp.processEvents()


def test_a_page_with_no_reader_reads_the_day_anyway(qapp, monkeypatch):
    service = _Service()
    widget = _page(qapp, monkeypatch, service=service)
    try:
        widget.reload()
        _finish(qapp, widget, service)
        assert service.calls[0]["spy_m5_bars"] == []
    finally:
        widget.shutdown()
        widget.deleteLater()
        qapp.processEvents()


def test_a_reader_that_raises_costs_the_page_nothing(qapp, monkeypatch):
    service = _Service()
    reader = _Reader(answer=RuntimeError("the alert center is not up"))
    widget = _page(qapp, monkeypatch, service=service, reader=reader)
    try:
        widget.reload()  # must not raise
        _finish(qapp, widget, service)
        assert service.calls[0]["spy_m5_bars"] == []
    finally:
        widget.shutdown()
        widget.deleteLater()
        qapp.processEvents()


def test_the_m5_half_of_the_alert_centers_answer_is_the_one_drawn(qapp, monkeypatch):
    """`journal_chart_bars` answers `(m5, d1)`; this page draws the M5."""
    service = _Service()
    reader = _Reader(answer=(BARS, [{"dt": datetime(2026, 9, 10), "close": 1.0}]))
    widget = _page(qapp, monkeypatch, service=service, reader=reader)
    try:
        widget.reload()
        _finish(qapp, widget, service)
        assert service.calls[0]["spy_m5_bars"] == BARS
    finally:
        widget.shutdown()
        widget.deleteLater()
        qapp.processEvents()


def test_the_desk_hands_the_reader_to_the_page_and_not_to_the_service():
    """Source-level, because wiring it to the service is what the reviewer
    found: `MainWindow` must hand the accessor to the PAGE."""
    source = (SCRIPTS_DIR / "ui" / "app.py").read_text(encoding="utf-8")

    assert "day_review_panel.set_bars_reader(" in source
    assert "day_review_panel.service.set_bars_reader(" not in source
    assert "journal_chart_bars" in source
