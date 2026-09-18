"""TJ-1 item 4, the missing half: who BUILDS today's index, and exactly once.

The tester pinned the index itself (`tests/test_tj1_day_review_index.py`) and the
page's two schedule functions (`tests/test_tj1_day_review_page.py`). What neither
pinned is the packet's sentence "the post-close tick builds today's index once",
and the lead's ruling of 2026-09-17 that it happens through ONE named seam:
`DayReviewService.build_index_for(session)`, called from the post-close branch of
`DayReviewPanel.poll_auto_read`.

It matters because the index is the whole speed claim. If nothing builds it after
the close, the first open of a completed session still streams the 476 MB log and
the trader still waits - and the page would look correct in every other test.

Three things are asserted here and nothing else:

* the post-close branch calls the seam, with that session;
* the NOON branch does not (the session is not closed yet, so an index built then
  would be pending and would be rebuilt on the next open anyway);
* the seam may fail without costing the read - a cache never costs the page.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
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
#: Friday morning: 2026-09-10 is the last COMPLETED session.
NOW = datetime(2026, 9, 11, 7, 30)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _Service:
    """A DayReviewService stand-in that records the seam call and reads nothing.

    It records the THREAD the build ran on, because where it runs is the point:
    the build streams the 476 MB intraday log and three other stores, and called
    inline from the 60-second timer slot it froze the desk for 22.8 seconds
    (reviewer, 2026-09-17).
    """

    def __init__(self, *, explode: bool = False, sleep_seconds: float = 0.0) -> None:
        self.built: list[tuple[str, dict]] = []
        self.threads: list[int] = []
        self.started = threading.Event()
        self._explode = explode
        self._sleep = float(sleep_seconds)

    def read_day(self, session_date, **kwargs):
        return {"session_date": session_date}

    def build_index_for(self, session_date, **kwargs):
        self.threads.append(threading.get_ident())
        self.started.set()
        if self._sleep:
            time.sleep(self._sleep)
        if self._explode:
            raise OSError("the day_review folder is read-only")
        self.built.append((str(session_date), dict(kwargs)))
        return {"schema": "day_review_index_v1", "session_date": str(session_date)}


def _settle(qapp, done, *, timeout: float = 5.0) -> bool:
    """Pump the event loop until `done()` or the deadline. Never sleeps blindly."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        qapp.processEvents()
        if done():
            return True
        time.sleep(0.01)
    qapp.processEvents()
    return done()


def _drain_build(qapp, panel, *, timeout: float = 5.0) -> None:
    """Let the index worker finish and deliver its signal."""
    worker = panel._index_worker
    if worker is not None:
        worker.wait(int(timeout * 1000))
    _settle(qapp, lambda: panel._index_worker is None, timeout=timeout)


@pytest.fixture()
def panel(qapp, monkeypatch):
    from ui.panels.day_review_panel import DayReviewPanel

    service = _Service()
    widget = DayReviewPanel(service=service, clock=lambda: NOW)
    monkeypatch.setattr(widget, "reload", lambda: None)
    monkeypatch.setattr(widget, "show_session", lambda _session: None)
    widget._service_stub = service
    yield widget
    try:
        widget.shutdown()
    except Exception:
        pass
    widget.deleteLater()


def _schedule(monkeypatch, *, noon=None, post_close=None):
    import daily_recap_schedule

    monkeypatch.setattr(daily_recap_schedule, "due_session", lambda *_a, **_k: noon)
    monkeypatch.setattr(
        daily_recap_schedule, "post_close_due_session", lambda *_a, **_k: post_close
    )


def test_the_post_close_tick_builds_that_sessions_index(panel, qapp, monkeypatch):
    _schedule(monkeypatch, noon=None, post_close=SESSION)

    assert panel.poll_auto_read() == SESSION
    _drain_build(qapp, panel)
    assert panel._service_stub.built, "nothing built the index after the close"
    session, kwargs = panel._service_stub.built[0]
    assert session == SESSION
    assert kwargs.get("lookback_sessions") == 3


def test_the_build_runs_on_a_worker_and_never_on_the_timer_slots_thread(
    panel, qapp, monkeypatch
):
    """Blocker 1 (reviewer, 2026-09-17): 22.8 s of frozen desk. The slot may
    START the build and must never BE it."""
    _schedule(monkeypatch, noon=None, post_close=SESSION)
    slot_thread = threading.get_ident()

    panel.poll_auto_read()
    _drain_build(qapp, panel)

    assert panel._service_stub.threads, "the build never ran"
    assert slot_thread not in panel._service_stub.threads, (
        "the index build ran on the thread that called the timer slot"
    )


def test_the_slot_returns_while_the_build_is_still_running(qapp, monkeypatch):
    """Measured rather than asserted by shape: with a build that takes 400 ms,
    the slot returns in milliseconds and the worker is still going."""
    from ui.panels.day_review_panel import DayReviewPanel

    service = _Service(sleep_seconds=0.4)
    widget = DayReviewPanel(service=service, clock=lambda: NOW)
    monkeypatch.setattr(widget, "reload", lambda: None)
    monkeypatch.setattr(widget, "show_session", lambda _session: None)
    _schedule(monkeypatch, noon=None, post_close=SESSION)
    try:
        start = time.perf_counter()
        assert widget.poll_auto_read() == SESSION
        elapsed = time.perf_counter() - start

        assert elapsed < 0.2, f"the timer slot blocked for {elapsed * 1000:.0f} ms"
        assert service.started.wait(2.0), "the build never started"
        assert widget._index_worker is not None
        assert widget._index_worker.isRunning(), "the build was not still running"
        # And the page says what it is doing rather than going quiet.
        assert SESSION in widget.status.text()
        assert "index" in widget.status.text().lower()

        _drain_build(qapp, widget)
        assert [session for session, _kwargs in service.built] == [SESSION]
    finally:
        widget.shutdown()
        widget.deleteLater()
        qapp.processEvents()


def test_a_second_tick_while_one_build_is_in_flight_starts_no_second_build(
    qapp, monkeypatch
):
    """Single-flight. Two builds of one session would stream the big stores
    twice for one answer."""
    from ui.panels.day_review_panel import DayReviewPanel

    service = _Service(sleep_seconds=0.3)
    widget = DayReviewPanel(service=service, clock=lambda: NOW)
    monkeypatch.setattr(widget, "reload", lambda: None)
    monkeypatch.setattr(widget, "show_session", lambda _session: None)
    try:
        widget._build_index_for(SESSION)
        assert service.started.wait(2.0)
        widget._build_index_for(SESSION)
        widget._build_index_for(SESSION)
        _drain_build(qapp, widget)
        assert len(service.threads) == 1, service.threads
    finally:
        widget.shutdown()
        widget.deleteLater()
        qapp.processEvents()


def test_it_builds_it_once_rather_than_on_every_tick(panel, qapp, monkeypatch):
    """`post_close_due_session` answers once per session per process; the page
    must not add a second build of its own on top of that."""
    calls = {"n": 0}

    import daily_recap_schedule

    def _post_close(_now, **_kwargs):
        calls["n"] += 1
        return SESSION if calls["n"] == 1 else None

    monkeypatch.setattr(daily_recap_schedule, "due_session", lambda *_a, **_k: None)
    monkeypatch.setattr(daily_recap_schedule, "post_close_due_session", _post_close)

    panel.poll_auto_read()
    _drain_build(qapp, panel)
    panel.poll_auto_read()
    panel.poll_auto_read()
    _drain_build(qapp, panel)
    assert len(panel._service_stub.built) == 1, panel._service_stub.built


def test_the_noon_read_builds_no_index(panel, qapp, monkeypatch):
    """At noon the session has not closed. An index built then is pending by
    definition and `is_stale` would have it rebuilt on the next open, so
    building it would be work with no answer at the end of it."""
    _schedule(monkeypatch, noon=SESSION, post_close=None)

    assert panel.poll_auto_read() == SESSION
    _drain_build(qapp, panel)
    assert panel._service_stub.built == []
    assert panel._service_stub.threads == []


def test_a_build_that_fails_never_costs_the_read(qapp, monkeypatch):
    """The index is derived and rebuildable: a folder that will not take it
    leaves a slow page, never a page that did not read."""
    from ui.panels.day_review_panel import DayReviewPanel

    widget = DayReviewPanel(service=_Service(explode=True), clock=lambda: NOW)
    monkeypatch.setattr(widget, "reload", lambda: None)
    shown: list[str] = []
    monkeypatch.setattr(widget, "show_session", lambda session: shown.append(session))
    _schedule(monkeypatch, noon=None, post_close=SESSION)
    try:
        assert widget.poll_auto_read() == SESSION
        assert shown == [SESSION], "the session was still read"
        _drain_build(qapp, widget)
        assert widget._index_worker is None, "a failed build left its worker behind"
    finally:
        widget.shutdown()
        widget.deleteLater()


def test_a_service_without_the_seam_is_not_a_broken_page(qapp, monkeypatch):
    """A host may hand this page a reader that only reads. The page asks for the
    seam and does without it, rather than raising into a timer slot."""
    from ui.panels.day_review_panel import DayReviewPanel

    class _ReaderOnly:
        def read_day(self, session_date, **_kwargs):
            return {"session_date": session_date}

    widget = DayReviewPanel(service=_ReaderOnly(), clock=lambda: NOW)
    monkeypatch.setattr(widget, "reload", lambda: None)
    monkeypatch.setattr(widget, "show_session", lambda _session: None)
    _schedule(monkeypatch, noon=None, post_close=SESSION)
    try:
        assert widget.poll_auto_read() == SESSION
    finally:
        widget.shutdown()
        widget.deleteLater()


def test_the_service_seam_writes_the_index_it_built(monkeypatch, tmp_path):
    """The named seam is not a wrapper around nothing: it BUILDS and it WRITES,
    and it is the same function the lazy path inside `read_day` uses."""
    import project_paths

    monkeypatch.setattr(project_paths, "RUNTIME_DATA_DIR", tmp_path, raising=False)

    import day_review_index
    from ui.services.day_review_service import DayReviewService

    written: list[object] = []
    monkeypatch.setattr(
        day_review_index, "write_index", lambda index, **_k: written.append(index)
    )
    index = DayReviewService().build_index_for(SESSION, now=NOW)
    assert index is not None
    assert index["session_date"] == SESSION
    assert index["schema"] == day_review_index.SCHEMA
    assert written == [index]


def test_the_seam_answers_none_rather_than_raising(monkeypatch, tmp_path):
    import project_paths

    monkeypatch.setattr(project_paths, "RUNTIME_DATA_DIR", tmp_path, raising=False)

    import day_review_index
    from ui.services.day_review_service import DayReviewService

    def _refuse(*_a, **_k):
        raise OSError("the outcome store is not mounted")

    monkeypatch.setattr(day_review_index, "build_index", _refuse)
    assert DayReviewService().build_index_for(SESSION, now=NOW) is None
