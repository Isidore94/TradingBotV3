"""Weekend Prep read its stores on the Qt thread. That was the 8.45 s freeze.

The 2026-08-25 fluidity capture recorded 264 stalls in ~45 minutes, and the
single worst - 8.45 s of a frozen desk - was Weekend Prep. Two pages are
responsible: `WeekReviewPage.reload` called `build_review_learning_state` plus
two RS log scans directly, and `FocusReviewPage.reload` ran five CSV/JSONL
reads and then built every table cell, all inside the click that selected the
page.

The `WalkawayPage` in the same file already owned a `QThread` for exactly this
reason, so the shape was not in question - only its application.

Two rules these tests pin, beyond "it is off the Qt thread":

- **last-good is never destroyed by a refresh.** A populated page that is
  refreshing keeps showing what it has. `WalkawayPage` gets this wrong today
  (it blanks its body to "Running walk-away..."), so the shape is reused but
  that line deliberately is not.
- **a failed read is STATED, never blank.** An unreadable store is a fact
  about the day; a page that silently empties itself asserts something false.
"""

from __future__ import annotations

import os
import sys
import threading
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.services.weekend_prep_service import WeekendPrepService  # noqa: E402

_app = QApplication.instance() or QApplication([])

MAIN_THREAD = threading.get_ident()


@pytest.fixture
def service(tmp_path):
    svc = WeekendPrepService(state_path=tmp_path / "state.json", now=datetime(2026, 8, 15, 10, 0))
    yield svc
    svc.shutdown()


def _settle(page, timeout_ms: int = 15000) -> None:
    """Wait for the page's worker, then let its queued signal be delivered."""
    worker = getattr(page, "_worker", None)
    if worker is not None:
        worker.wait(timeout_ms)
    for _ in range(20):
        _app.processEvents()


# ==========================================================================
# WeekReviewPage - the 8.45 s freeze itself
# ==========================================================================
def test_week_review_reads_review_learning_off_the_qt_thread(service, monkeypatch):
    import review_learning

    from ui.panels import weekend_prep_panel as panel_module

    idents: list[int] = []

    def recording_state(*args, **kwargs):
        idents.append(threading.get_ident())
        return {"takes": 1, "skips": 2}

    monkeypatch.setattr(review_learning, "build_review_learning_state", recording_state)

    page = panel_module.WeekReviewPage(service)
    page.reload()
    _settle(page)

    assert idents, "build_review_learning_state was never called"
    assert MAIN_THREAD not in idents, (
        "the review-learning state was built on the Qt thread - that is the "
        "8.45 s freeze"
    )


def test_week_review_scans_the_rs_logs_off_the_qt_thread(service, monkeypatch):
    from ui.panels import weekend_prep_panel as panel_module

    idents: list[int] = []

    def recording(bounds):
        idents.append(threading.get_ident())
        return []

    monkeypatch.setattr(panel_module, "_read_rrs_week", recording)
    monkeypatch.setattr(panel_module, "_read_rrs_group_week", recording)

    page = panel_module.WeekReviewPage(service)
    page.reload()
    _settle(page)

    assert idents, "the RS logs were never scanned"
    assert MAIN_THREAD not in idents, "the RS week logs were scanned on the Qt thread"


def test_week_review_keeps_last_good_text_while_refreshing(service, monkeypatch):
    """A populated page that refreshes must not flash empty."""
    from ui.panels import weekend_prep_panel as panel_module

    page = panel_module.WeekReviewPage(service)
    page.summary.setPlainText("LAST GOOD CONTENT")

    started = threading.Event()
    release = threading.Event()

    def slow_state(*args, **kwargs):
        started.set()
        release.wait(10)
        return {"takes": 1}

    import review_learning

    monkeypatch.setattr(review_learning, "build_review_learning_state", slow_state)

    page.reload()
    assert started.wait(10), "the worker never started"
    # Mid-refresh: the body still holds what it had.
    assert "LAST GOOD CONTENT" in page.summary.toPlainText()
    release.set()
    _settle(page)


def test_week_review_states_a_failed_read_instead_of_blanking(service, monkeypatch):
    from ui.panels import weekend_prep_panel as panel_module

    import review_learning

    def boom(*args, **kwargs):
        raise OSError("the review store is unreadable")

    monkeypatch.setattr(review_learning, "build_review_learning_state", boom)

    page = panel_module.WeekReviewPage(service)
    page.reload()
    _settle(page)

    text = page.summary.toPlainText()
    assert "unreadable" in text or "unavailable" in text, text


def test_week_review_refresh_is_single_flight(service, monkeypatch):
    """Three clicks must not start three readers over the same stores."""
    from ui.panels import weekend_prep_panel as panel_module

    import review_learning

    starts: list[int] = []
    release = threading.Event()

    def slow_state(*args, **kwargs):
        starts.append(1)
        release.wait(10)
        return {"takes": 1}

    monkeypatch.setattr(review_learning, "build_review_learning_state", slow_state)

    page = panel_module.WeekReviewPage(service)
    page.reload()
    page.reload()
    page.reload()
    release.set()
    _settle(page)

    assert len(starts) == 1, f"{len(starts)} concurrent readers were started"


# ==========================================================================
# FocusReviewPage - five reads and a full table rebuild on the click
# ==========================================================================
FOCUS_READS = (
    "_join_focus_week",
    "_read_veto_cohort",
    "_read_like_cohort",
    "_read_focus_performance",
    "_read_pick_feedback_week",
)


def test_focus_review_reads_every_store_off_the_qt_thread(service, monkeypatch):
    from ui.panels import weekend_prep_panel as panel_module

    idents: dict[str, int] = {}

    def make(name):
        def recording(*args, **kwargs):
            idents[name] = threading.get_ident()
            return []

        return recording

    for name in FOCUS_READS:
        monkeypatch.setattr(panel_module, name, make(name))

    page = panel_module.FocusReviewPage(service)
    page.reload()
    _settle(page)

    missing = [name for name in FOCUS_READS if name not in idents]
    assert not missing, f"never called: {missing}"
    on_qt = [name for name, ident in idents.items() if ident == MAIN_THREAD]
    assert not on_qt, f"read on the Qt thread: {on_qt}"


def test_focus_review_keeps_its_rows_when_a_refresh_fails(service, monkeypatch):
    """The graded cohort is not week-scoped. A bad week must not erase it."""
    from ui.panels import weekend_prep_panel as panel_module

    good = [{"cohort": "gap_fade", "side": "long", "n": "7",
             "win_rate": "0.43", "avg_return": "0.10", "profit_factor": "1.2",
             "horizon": "h3"}]
    monkeypatch.setattr(panel_module, "_read_veto_cohort", lambda: good)
    for name in ("_join_focus_week", "_read_like_cohort",
                 "_read_focus_performance", "_read_pick_feedback_week"):
        monkeypatch.setattr(panel_module, name, lambda *a, **k: [])

    page = panel_module.FocusReviewPage(service)
    page.reload()
    _settle(page)
    assert page.cohort_table.rowCount() == 1

    def boom(*args, **kwargs):
        raise OSError("the cohort CSV is unreadable")

    monkeypatch.setattr(panel_module, "_read_veto_cohort", boom)
    page.reload()
    _settle(page)

    assert page.cohort_table.rowCount() == 1, (
        "a failed refresh destroyed the last good cohort rows"
    )
    assert "unreadable" in page.note.text() or "unavailable" in page.note.text(), page.note.text()


# ==========================================================================
# a worker must not outlive the widget it was going to update
# ==========================================================================
def test_every_page_with_a_worker_is_joined_on_shutdown(service, monkeypatch, tmp_path):
    """The panel's shutdown named `walkaway` while it was the only threaded page.

    That is the kind of hand-maintained list that stops being complete the
    moment a second page grows a thread - which is what this change does.
    """
    from ui.panels.weekend_prep_panel import WeekendPrepPanel

    panel = WeekendPrepPanel(service=service)
    try:
        release = threading.Event()
        started = threading.Event()

        import review_learning

        def slow_state(*args, **kwargs):
            started.set()
            release.wait(10)
            return {"takes": 1}

        monkeypatch.setattr(review_learning, "build_review_learning_state", slow_state)

        page = panel.week_review
        page.reload()
        assert started.wait(10), "the worker never started"
        release.set()
        panel.shutdown()

        assert not page._worker.isRunning(), (
            "shutdown returned while the page's reader was still running"
        )
    finally:
        try:
            panel.close()
        except Exception:
            pass
