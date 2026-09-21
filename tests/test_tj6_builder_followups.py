r"""TJ-6 - the three decisions the BUILDER made, pinned. Offscreen.

The tester's nine files pin the packet. These three tests pin the readings the
builder chose where the packet left a choice, so a later change has to argue
with a test rather than with a comment:

1. **A second Keep never re-freezes the baseline.** The packet says the baseline
   is frozen at the keep and printed against the number now; it does not say
   what a second click does. The Week Review card shows kept ideas WITH a Keep
   button, so a second click is one mis-aim away at all times - and a keep that
   re-froze would silently make every kept idea read "the same as at the keep".
   So `keep_idea` is idempotent: it returns the stored record and writes
   nothing.
2. **Day Review shows the card OR the line, never both.** `NO_IDEAS_YET` is the
   line; the rows are the card; exactly one of them is on screen.
3. **Week Review renders the kept ideas through the same widget**, counting them
   correctly - which is what `checked_ideas` carrying ``status: kept`` is for.

**NO MODEL IS EVER CALLED HERE.**
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tj6_support as fx  # noqa: E402

TEXT = "Wait for the second test before sizing up."


@pytest.fixture
def night(tmp_path, monkeypatch):
    return fx.install_stores(monkeypatch, tmp_path, write_the_week=False)


def _reading(value: float, n: int) -> dict:
    return {
        "measurable": "report_card_did_well_rate",
        "value": float(value),
        "n": int(n),
        "measured": True,
        "window_sessions": fx.lately_sessions(),
    }


def test_two_readings_that_overlap_are_not_called_a_change(night):
    """Review 1 advisory 5, reproduced: 0.5000 (n 30) against 0.5001 (n 50,000)
    printed "higher than at the keep" - two samples that disagree about nothing.

    Hand-counted: both sides are over the floor (30), their Wilson intervals
    overlap, so the verdict is "no clear change". A difference is only SAID when
    the intervals do not overlap - 0.10 (n 400) against 0.60 (n 400) do not.
    """
    from ai_jobs import improvement_ideas as ideas

    assert ideas._verdict(_reading(0.5000, 30), _reading(0.5001, 50_000)) == (
        ideas.VERDICT_NO_CHANGE
    )
    assert ideas._verdict(_reading(0.10, 400), _reading(0.60, 400)) == ideas.VERDICT_HIGHER
    assert ideas._verdict(_reading(0.60, 400), _reading(0.10, 400)) == ideas.VERDICT_LOWER
    # Under the floor on either side it is still the exact words the packet
    # names, and an unreadable side is never compared at all.
    thin = fx.thin_reading()
    assert ideas._verdict(_reading(0.30, 400), _reading(thin["value"], thin["n"])) == (
        ideas.VERDICT_TOO_FEW
    )
    assert ideas._verdict(
        _reading(0.30, 400), {"measured": False, "value": None, "n": 0}
    ) == ideas.VERDICT_UNMEASURED
    # The interval is the desk's OWN Wilson, not a second one.
    import evidence_contrast

    banded = ideas.with_interval(_reading(0.25, 400))
    assert (banded["low"], banded["high"]) == (
        evidence_contrast.rate(100, 400)["low"],
        evidence_contrast.rate(100, 400)["high"],
    )


def test_a_second_keep_never_re_freezes_the_baseline(night, monkeypatch):
    """Hand-counted: keep at n 34, click again while the number reads n 41, and
    the stored baseline is still 34 - with the state file byte-identical."""
    from ai_jobs import improvement_ideas

    name = improvement_ideas.MEASURABLES[0].name
    row = fx.stored_idea_row(TEXT, session=fx.SESSION, measurable=name)
    fx.write_ideas(night["ideas"], [row])

    def _reading(body):
        return lambda measurable, **kwargs: {
            "measurable": measurable,
            "value": body["value"],
            "n": body["n"],
            "measured": True,
            "window_sessions": fx.lately_sessions(),
        }

    monkeypatch.setattr(improvement_ideas, "measure", _reading(fx.BASELINE))
    improvement_ideas.keep_idea(row["idea_id"], end_session=fx.SESSION)
    before = night["state"].read_bytes()

    monkeypatch.setattr(improvement_ideas, "measure", _reading(fx.AFTER))
    improvement_ideas.keep_idea(row["idea_id"], end_session=fx.SESSION)

    assert night["state"].read_bytes() == before, "a second keep rewrote the state"
    stored = improvement_ideas.read_state()[row["idea_id"]]
    assert stored["baseline"]["n"] == fx.BASELINE["n"]
    assert stored["baseline"]["value"] == fx.BASELINE["value"]


# ---------------------------------------------------------------------------
# the two pages
# ---------------------------------------------------------------------------
pytest.importorskip("PySide6", reason="both pages are PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

SESSION_ROWS = [
    {
        "idea_id": "idea:2026-09-18:000000000001",
        "kind": "process",
        "text": "Wait for the second test before sizing up.",
        "measurable": "report_card_did_well_rate",
        "evidence": ["2026-09-18/report_card:did_well"],
        "seen_count": 1,
        "status": "",
    }
]


@pytest.mark.qt
def test_the_day_page_shows_the_card_or_the_line_and_never_both():
    """Hand-counted: one idea -> one row on the card and no standing line; no
    ideas -> the line and no card."""
    from ui.panels.day_review_panel import NO_IDEAS_YET, DayReviewPanel

    panel = DayReviewPanel(service=_Stub(), clock=lambda: datetime(2026, 9, 19, 8, 0))
    try:
        # `isVisibleTo` rather than `isVisible`: an offscreen page nobody showed
        # has no visible widget at all, and the question here is which of the
        # two the section would draw.
        section = panel.ideas_section
        panel.render({"session_date": "2026-09-18", "ideas": SESSION_ROWS})
        assert len(panel.ideas_card.rows) == 1
        assert panel.ideas_card.isVisibleTo(section) is True
        assert panel.ideas_note.isVisibleTo(section) is False
        assert "1" in panel.ideas_card.summary_text()

        panel.render({"session_date": "2026-09-18", "ideas": []})
        assert panel.ideas_card.rows == ()
        assert panel.ideas_card.isVisibleTo(section) is False
        assert panel.ideas_note.isVisibleTo(section) is True
        assert NO_IDEAS_YET in panel.ideas_note.text()
    finally:
        panel.shutdown()
        panel.deleteLater()


@pytest.mark.qt
def test_a_row_that_arrives_mid_write_comes_in_disabled():
    """Review 1 advisory 1: a row added while a write is in flight had LIVE
    buttons, and its click was swallowed with nothing said.

    Hand-counted: one write in flight, a second render adding a row -> all four
    buttons disabled; after the write answers, all four live.
    """
    import threading
    import time

    from ui.widgets.ideas_card import IdeasCard

    gate = threading.Event()
    started = threading.Event()

    def _writer(idea_id, status):
        started.set()
        gate.wait(5.0)
        return {"idea_id": idea_id, "status": status}

    card = IdeasCard(writer=_writer)
    try:
        card.show_ideas(SESSION_ROWS)
        card.rows[0].keep_button.click()
        assert started.wait(5.0)
        card.show_ideas(SESSION_ROWS + [dict(SESSION_ROWS[0], idea_id="idea:2026-09-18:2", text="Two.")])
        assert len(card.rows) == 2
        assert [row.keep_button.isEnabled() for row in card.rows] == [False, False]
        assert [row.dismiss_button.isEnabled() for row in card.rows] == [False, False]
        gate.set()
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline and card.is_writing():
            _app.processEvents()
            time.sleep(0.01)
        _app.processEvents()
        assert [row.keep_button.isEnabled() for row in card.rows] == [True, True]
    finally:
        gate.set()
        card.shutdown()
        card.deleteLater()


@pytest.mark.qt
def test_the_week_page_counts_the_kept_ideas_it_was_handed(tmp_path):
    """Hand-counted: one kept idea -> one row, "Kept 1 of 1", and the before and
    after both printed with their own n."""
    from ui.panels.weekend_prep_panel import WeekReviewPage
    from ui.services import weekend_prep_service as prep
    from ui.services.weekend_prep_service import WeekendPrepService

    service = WeekendPrepService(
        state_path=tmp_path / "state.json", now=datetime(2026, 9, 19, 10, 0)
    )
    page = WeekReviewPage(service)
    try:
        payload = prep.empty_week_payload(fx.WEEK_ID)
        payload["ideas"] = [
            {
                "idea_id": "idea:2026-09-18:000000000001",
                "kind": "process",
                "text": TEXT,
                "measurable": "report_card_did_well_rate",
                "status": "kept",
                "kept_at": "2026-09-19T15:00:00+00:00",
                "before": {**fx.BASELINE, "measured": True},
                "after": {**fx.AFTER, "measured": True},
                "verdict": "higher than at the keep",
                "for_wishlist": False,
            }
        ]
        page._render(payload)
        assert len(page.ideas_card.rows) == 1
        assert "1 of 1" in page.ideas_card.summary_text()
        detail = page.ideas_card.rows[0].detail_label.text()
        assert f"n {fx.BASELINE['n']}" in detail and f"n {fx.AFTER['n']}" in detail
    finally:
        page.shutdown()
        page.deleteLater()
        service.shutdown()


class _Stub:
    """A `DayReviewService` stand-in: records, reads nothing, writes nothing."""

    def read_day(self, session_date, **kwargs):
        return {}

    def write_entry(self, **kwargs):
        return {"ok": True, "entry": {"entry_id": "mj-stub-0001"}}

    def import_daily_forecast(self, **kwargs):
        return {"ok": True, "entry": {"entry_id": "mj-stub-0002"}}
