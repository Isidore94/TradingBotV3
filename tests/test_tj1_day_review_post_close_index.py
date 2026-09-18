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
    """A DayReviewService stand-in that records the seam call and reads nothing."""

    def __init__(self, *, explode: bool = False) -> None:
        self.built: list[tuple[str, dict]] = []
        self._explode = explode

    def read_day(self, session_date, **kwargs):
        return {"session_date": session_date}

    def build_index_for(self, session_date, **kwargs):
        if self._explode:
            raise OSError("the day_review folder is read-only")
        self.built.append((str(session_date), dict(kwargs)))
        return {"schema": "day_review_index_v1", "session_date": str(session_date)}


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


def test_the_post_close_tick_builds_that_sessions_index(panel, monkeypatch):
    _schedule(monkeypatch, noon=None, post_close=SESSION)

    assert panel.poll_auto_read() == SESSION
    assert panel._service_stub.built, "nothing built the index after the close"
    session, kwargs = panel._service_stub.built[0]
    assert session == SESSION
    assert kwargs.get("lookback_sessions") == 3


def test_it_builds_it_once_rather_than_on_every_tick(panel, monkeypatch):
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
    panel.poll_auto_read()
    panel.poll_auto_read()
    assert len(panel._service_stub.built) == 1, panel._service_stub.built


def test_the_noon_read_builds_no_index(panel, monkeypatch):
    """At noon the session has not closed. An index built then is pending by
    definition and `is_stale` would have it rebuilt on the next open, so
    building it would be work with no answer at the end of it."""
    _schedule(monkeypatch, noon=SESSION, post_close=None)

    assert panel.poll_auto_read() == SESSION
    assert panel._service_stub.built == []


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
