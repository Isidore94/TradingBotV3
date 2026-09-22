"""AI-R2: reach the one daily review, and say when its story did not arrive.

These tests drive the real Qt panel methods.  They use a recording Day Review
service only so the navigation and rendering tests cannot touch a broker,
provider, journal, or live store.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the daily route is a Qt panel")

from PySide6.QtWidgets import QApplication  # noqa: E402


MONDAY_MORNING = datetime(2026, 9, 21, 8, 0)
SATURDAY_MORNING = datetime(2026, 9, 19, 8, 0)
OLD_SESSION = "2026-09-16"


class _ReadOnlyDayService:
    """A service stand-in that records any accidental worker read."""

    def __init__(self) -> None:
        self.reads: list[str] = []

    def read_day(self, session_date, **_kwargs):
        self.reads.append(str(session_date))
        return {"session_date": str(session_date)}


class _BlockingDayService:
    """A worker-path stand-in: it blocks one old read until the test releases it."""

    def __init__(self) -> None:
        self.reads: list[str] = []
        self.started = threading.Event()
        self.release = threading.Event()

    def read_day(self, session_date, **_kwargs):
        session = str(session_date)
        self.reads.append(session)
        self.started.set()
        if len(self.reads) == 1:
            assert self.release.wait(2.0), "test did not release the historical read"
        return {
            **_payload(session=session),
            "walkaway_backfill_sessions": ("2026-09-15",),
        }


def _drain_until(qapp, predicate) -> bool:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def day_panel(qapp, monkeypatch):
    from ui.panels.day_review_panel import DayReviewPanel

    service = _ReadOnlyDayService()
    panel = DayReviewPanel(service=service, clock=lambda: MONDAY_MORNING)
    # Navigation chooses a date; it must not start a worker in this test.
    monkeypatch.setattr(panel, "reload", lambda: None)
    yield panel
    panel.shutdown()
    panel.deleteLater()
    qapp.processEvents()


def _card(*, night_status="read", failed=(), degraded=()):
    return {
        "lines": (
            {"key": "did_well", "text": "Measured facts stay here."},
            {
                "key": "how_fresh",
                "text": "How fresh: the overnight read was checked.",
                "night_status": night_status,
                "slots_failed": tuple(failed),
                "slots_degraded": tuple(degraded),
            },
        )
    }


def _payload(*, session="2026-09-18", day_story=None, report_card=None):
    return {
        "session_date": session,
        "provisional": False,
        "story": None,
        "day_story": day_story,
        "d1_view": None,
        "report_card": report_card or _card(),
        "theses": [],
        "entries": [],
        "rejected_that_worked": [],
        "walkaway": None,
        "trades": [],
        "forecast": {},
        "spy_m5_bars": [],
        "reads": [],
        "congruence": [],
        "ideas": [],
        "mood": {},
    }


def _prior_verified_story():
    return {
        "session_date": "2026-09-18",
        "narration": {
            "headline": "Friday had a measured story.",
            "what_happened": "The prior verified words remain useful.",
        },
    }


def test_ai_summary_daily_button_routes_to_the_one_latest_completed_day_review_without_work(
    qapp, monkeypatch
):
    """A real click emits the signal and the MainWindow title route selects Day Review.

    The recording host proves the route does not call a provider, importer,
    redo marker, or Day Review worker merely to open the completed session.
    """
    import ai_summary
    from ui import app
    from ui.panels import ai_summary_panel

    monkeypatch.setattr(ai_summary_panel.AiSummaryPanel, "refresh_gates", lambda self: None)
    monkeypatch.setattr(
        ai_summary, "request_ai_summary", lambda **_k: pytest.fail("daily navigation called a model")
    )
    monkeypatch.setattr(
        ai_summary, "build_evidence_package", lambda *_a, **_k: pytest.fail("daily navigation built evidence")
    )

    selected: list[str] = []
    shown: list[str] = []
    host = SimpleNamespace(
        _select_page_by_title=lambda title: selected.append(title) or True,
        day_review_panel=SimpleNamespace(
            show_latest_completed_session=lambda: shown.append("latest")
        ),
    )
    panel = ai_summary_panel.AiSummaryPanel()
    try:
        panel.resize(1000, 700)
        panel.show()
        qapp.processEvents()
        panel.dailyReviewRequested.connect(
            lambda: app.MainWindow.show_latest_completed_day_review(host)
        )
        panel.daily_review_button.click()

        assert selected == [app.DAY_REVIEW_PAGE_TITLE]
        assert shown == ["latest"]
        assert "Day Review" in panel.daily_review_button.text()
        assert "overnight" in panel.daily_review_button.toolTip().lower()
        assert panel.daily_review_button.y() < panel.generate_button.y()
    finally:
        panel.deleteLater()
        qapp.processEvents()


def test_latest_daily_route_defers_the_remembered_session_read_until_after_selection():
    """The title switch sees the guard before the panel starts its one latest-day read."""
    from ui import app

    events: list[object] = []
    host = SimpleNamespace()

    def select(title):
        events.append((title, host._opening_latest_day_review))
        return True

    host._select_page_by_title = select
    host.day_review_panel = SimpleNamespace(
        show_latest_completed_session=lambda: events.append("latest") or True
    )

    assert app.MainWindow.show_latest_completed_day_review(host) is True
    assert events == [(app.DAY_REVIEW_PAGE_TITLE, True), "latest"]
    assert host._opening_latest_day_review is False


@pytest.mark.parametrize(
    ("clock", "expected"),
    (
        (MONDAY_MORNING, "2026-09-18"),
        (SATURDAY_MORNING, "2026-09-18"),
    ),
)
def test_latest_completed_session_uses_the_exchange_calendar_and_replaces_old_selection(
    qapp, monkeypatch, clock, expected
):
    """Morning and weekend clicks use the panel's calendar owner, never its old picker row."""
    from ui.panels.day_review_panel import DayReviewPanel

    panel = DayReviewPanel(service=_ReadOnlyDayService(), clock=lambda: clock)
    monkeypatch.setattr(panel, "reload", lambda: None)
    try:
        panel.show_session(OLD_SESSION)
        assert panel.session_date() == OLD_SESSION

        panel.show_latest_completed_session()

        assert panel.session_date() == expected
    finally:
        panel.shutdown()
        panel.deleteLater()
        qapp.processEvents()


def test_latest_completed_session_fails_closed_when_the_calendar_is_uncertain(day_panel, monkeypatch):
    """A calendar error must leave the remembered history alone and say it is unknown."""
    import market_calendar

    shown: list[str] = []
    monkeypatch.setattr(day_panel, "show_session", lambda session: shown.append(str(session)))
    monkeypatch.setattr(
        market_calendar,
        "last_completed_session",
        lambda _now: (_ for _ in ()).throw(RuntimeError("calendar unavailable")),
    )

    assert day_panel.show_latest_completed_session() is False
    assert shown == []
    assert "uncertain" in day_panel.status.text().lower()


def test_latest_completed_session_reads_without_starting_a_missing_tape_backfill(qapp, monkeypatch):
    """The daily link reads existing facts only; a missing tape cannot start recovery work."""
    from ui.panels.day_review_panel import DayReviewPanel

    service = _ReadOnlyDayService()
    panel = DayReviewPanel(service=service, clock=lambda: MONDAY_MORNING)
    monkeypatch.setattr(
        panel,
        "_backfill_bars_for",
        lambda _session: pytest.fail("latest completed navigation started a tape backfill"),
    )
    try:
        assert panel.show_latest_completed_session() is True
        assert _drain_until(qapp, lambda: service.reads == ["2026-09-18"])
    finally:
        panel.shutdown()
        panel.deleteLater()
        qapp.processEvents()


def test_latest_completed_session_queues_behind_an_old_read_without_backfilling_or_rendering_it(
    qapp, monkeypatch
):
    """The explicit route is read-only and cannot render old facts under a new date."""
    from ui.panels.day_review_panel import DayReviewPanel

    service = _BlockingDayService()
    panel = DayReviewPanel(service=service, clock=lambda: MONDAY_MORNING)
    backfills: list[str] = []
    monkeypatch.setattr(panel, "_backfill_bars_for", lambda session: backfills.append(str(session)))
    try:
        panel.show_session(OLD_SESSION)
        assert service.started.wait(1.0)
        assert service.reads == [OLD_SESSION]
        assert backfills == [OLD_SESSION]

        assert panel.show_latest_completed_session() is True
        assert panel.session_date() == "2026-09-18"
        assert service.reads == [OLD_SESSION]
        assert backfills == [OLD_SESSION]

        service.release.set()
        assert _drain_until(qapp, lambda: service.reads == [OLD_SESSION, "2026-09-18"])
        assert _drain_until(qapp, lambda: panel._payload.get("session_date") == "2026-09-18")
        assert panel.session_date() == "2026-09-18"
        assert backfills == [OLD_SESSION]
    finally:
        service.release.set()
        panel.shutdown()
        panel.deleteLater()
        qapp.processEvents()


def test_missing_story_says_failed_checks_but_keeps_the_measured_report_card(day_panel):
    """An absent day narration with a failed or degraded narration slot is not "not yet"."""
    day_panel.render(
        _payload(
            report_card=_card(failed=("day_review_narration",), degraded=("day_review_narration",))
        )
    )

    assert day_panel.story_note.text() == (
        "The AI story failed its checks. Your measured results are still shown."
    )
    assert "Measured facts stay here." in day_panel._report_card_lines["did_well"].text()
    assert not day_panel.story_body.text().strip()


def test_prior_verified_exact_session_story_stays_visible_when_a_new_attempt_failed(day_panel):
    """A newer failure warns; it never erases a verified story for this same day."""
    day_panel.render(
        _payload(
            day_story=_prior_verified_story(),
            report_card=_card(failed=("day_review_narration",)),
        )
    )

    assert day_panel.story_note.text() == "Friday had a measured story."
    assert "prior verified words" in day_panel.story_body.text().lower()
    assert "new attempt failed" in day_panel.story_note.toolTip().lower()
    assert "new attempt failed" in day_panel.story_warning.text().lower()
    assert "Measured facts stay here." in day_panel._report_card_lines["did_well"].text()


def test_story_failure_warning_is_reset_for_another_sessions_missing_story(day_panel):
    """A prior day's failed-attempt warning must not stick to another day's empty state."""
    from ui.panels.day_review_panel import NO_STORY_YET

    day_panel.render(
        _payload(day_story=_prior_verified_story(), report_card=_card(failed=("day_review_narration",)))
    )
    assert day_panel.story_warning.text()

    day_panel.render(_payload(session="2026-09-17", report_card=_card(night_status="no_rows")))

    assert day_panel.story_warning.text() == ""
    assert day_panel.story_note.toolTip() == ""
    assert "not run yet" in day_panel.story_note.text().lower()
    assert "TJ-4" not in NO_STORY_YET


@pytest.mark.parametrize(
    ("night_status", "expected_words"),
    (("no_rows", "not run yet"), ("unknown", "unknown")),
)
def test_missing_story_names_not_run_and_unknown_nights_without_hiding_facts(
    day_panel, night_status, expected_words
):
    day_panel.render(_payload(report_card=_card(night_status=night_status)))

    assert expected_words in day_panel.story_note.text().lower()
    assert "Measured facts stay here." in day_panel._report_card_lines["did_well"].text()
