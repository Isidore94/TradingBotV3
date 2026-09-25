r"""TJ-5 change 1 - the Week Review page itself. RED, offscreen.

`plan.md` §12.4 TJ-5 change 1: *"`WeekReviewPage` becomes: five Day Review cards
(headline · were-you-right tally · chased flag · said-vs-did line · small SPY
chart, built once on open and reused), the week story, the week's walk-away
totals (A-D summed with `n`), the week's kept ideas. Missing days are named,
never padded."*

**NO MODEL IS EVER CALLED HERE AND NOTHING IS FETCHED.** The service reader is
replaced by a recorder in every test that starts one, and the two tests that
prove the page calls no builder monkeypatch the builders to raise.

The contract these tests pin (the builder may ADD, never remove)
----------------------------------------------------------------

``scripts/ui/panels/weekend_prep_panel.py`` - `WeekReviewPage` only::

    page.day_cards   -> tuple of exactly `evidence_stats.WEEK_SESSIONS` widgets,
                        in week order, each with `.session`, `.headline`,
                        `.tally`, `.chased`, `.said_vs_did` and `.chart`
    page.facts_note  -> QLabel: "K of 5 sessions have facts"
    page.week_story  -> the widget the overnight narration is shown in
    page.strip       -> the deterministic week/month strip widget
    page.reload()    -> ONE call to `weekend_prep_service.read_week_review`,
                        on this page's worker, single-flight

``scripts/ui/services/weekend_prep_service.py``::

    WEEK_PAYLOAD_KEYS: tuple[str, ...]
    empty_week_payload(week_id="") -> dict        # every key present, nothing in it
    read_week_review(*, friday="", root=None, ledger_path=None, now=None) -> dict

The payload is ONE dict and the page RENDERS it. Every store this page opens is
opened inside `read_week_review`, on the worker - the rule TJ-1 and TJ-4 both
ship and the reason this page exists in its current shape at all (`reload`'s own
docstring: 8.45 s of frozen GUI, fluidity capture 2026-08-25).
"""

from __future__ import annotations

import os
import sys
import threading
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

import tj5_support as fx  # noqa: E402
from ui.services.weekend_prep_service import STEP_IDS, WeekendPrepService  # noqa: E402

_app = QApplication.instance() or QApplication([])

#: The Saturday the fixture week ends on, so `week_bounds` names 09-14..09-18.
WEEKEND_MOMENT = datetime(2026, 9, 19, 10, 0)


@pytest.fixture
def service(tmp_path):
    svc = WeekendPrepService(state_path=tmp_path / "state.json", now=WEEKEND_MOMENT)
    yield svc
    svc.shutdown()


@pytest.fixture
def page(service):
    from ui.panels.weekend_prep_panel import WeekReviewPage

    widget = WeekReviewPage(service)
    yield widget
    widget.shutdown()
    widget.deleteLater()


class _Reader:
    """A stand-in for `read_week_review` that records WHERE it was called.

    The count is read after an `Event` the reader sets, never after a sleep: a
    worker that has not started yet and a worker that never runs look the same
    to a race.
    """

    def __init__(self, payload):
        self.payload = payload
        self.calls: list[dict] = []
        self.threads: list[int] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.done = threading.Event()

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        self.threads.append(threading.get_ident())
        self.entered.set()
        self.release.wait(5.0)
        self.done.set()
        return self.payload


def _payload(*, with_facts=fx.PACKED_SESSIONS):
    """The ONE payload, hand-built in the shape the page renders.

    Three of five sessions carry facts - the live state on 2026-09-20, where
    `C:\\TradingBotData\\day_review\\sessions\\` holds three folders and ZERO
    `pack.json`.
    """
    from ui.services import weekend_prep_service as prep

    body = prep.empty_week_payload(fx.WEEK_ID)
    body.update(
        {
            "week_id": fx.WEEK_ID,
            "sessions": list(fx.WEEK),
            "sessions_with_facts": list(with_facts),
            "sessions_missing": list(fx.sessions_missing(with_facts)),
            "cards": [
                {
                    "session": session,
                    "has_facts": session in with_facts,
                    "headline": f"{session}: the tape held." if session in with_facts else "",
                    "tally": fx.tally_over([session]) if session in with_facts else {},
                    "chased": "no" if session in with_facts else "unmeasured",
                    "said_vs_did": f"{session}: you said up and you bought."
                    if session in with_facts
                    else "",
                    "spy_bars": [],
                }
                for session in fx.WEEK
            ],
        }
    )
    return body


def _drain(reader, page, timeout=5.0):
    assert reader.entered.wait(timeout), "the page never started its reader"
    assert reader.done.wait(timeout), "the reader never finished"
    worker = getattr(page, "_worker", None)
    if worker is not None:
        worker.wait(int(timeout * 1000))
    _app.processEvents()


# ---------------------------------------------------------------------------
# where it sits
# ---------------------------------------------------------------------------
def test_week_review_is_the_first_step_of_weekend_prep(service):
    """GREEN GUARD. It is already step 1 (`STEP_IDS[0]`), and TJ-5 is the packet
    that makes the first thing the trader sees on a Saturday worth opening - so
    a later packet must not reorder the rail out from under it."""
    from ui.panels.weekend_prep_panel import WeekendPrepPanel, WeekReviewPage

    assert STEP_IDS[0] == "week_review"
    panel = WeekendPrepPanel(service=service, focus_service=None)
    try:
        assert isinstance(panel.week_review, WeekReviewPage)
        assert panel.pages.widget(0) is panel.week_review
        assert panel.rail.item(0).text().endswith("Week in review")
    finally:
        panel.shutdown()
        panel.deleteLater()


# ---------------------------------------------------------------------------
# five cards, and the two days nobody packed
# ---------------------------------------------------------------------------
def test_the_page_shows_one_card_per_session_in_the_week(page, monkeypatch):
    """Five cards for five sessions - `evidence_stats.WEEK_SESSIONS`, in week
    order. A card is a DAY, so a week with three packs still has five of them."""
    import evidence_stats

    from ui.services import weekend_prep_service as prep

    reader = _Reader(_payload())
    monkeypatch.setattr(prep, "read_week_review", reader)
    page.reload()
    _drain(reader, page)

    assert len(page.day_cards) == evidence_stats.WEEK_SESSIONS == 5
    assert [card.session for card in page.day_cards] == list(fx.WEEK)


def test_a_day_with_no_pack_is_named_and_never_padded(page, monkeypatch):
    """Hand-counted: 2026-09-14 and 2026-09-15 have no pack.

    Their cards must SAY so. A blank card reads as a quiet day, and a zeroed
    tally reads as "you were right 0 of 0", which is a measurement nobody made.
    """
    from ui.services import weekend_prep_service as prep

    reader = _Reader(_payload())
    monkeypatch.setattr(prep, "read_week_review", reader)
    page.reload()
    _drain(reader, page)

    by_session = {card.session: card for card in page.day_cards}
    for missing in ("2026-09-14", "2026-09-15"):
        text = by_session[missing].headline.text().lower()
        assert "no facts" in text or "not packed" in text or "unmeasured" in text, text
        assert "0" not in by_session[missing].tally.text(), by_session[missing].tally.text()


def test_the_page_says_how_many_of_the_five_sessions_have_facts(page, monkeypatch):
    """The FIRST honest sentence on the page. The live store holds three session
    folders and zero packs, so this number is 0 or 3 on the trader's first
    Saturday - never a silent five."""
    from ui.services import weekend_prep_service as prep

    reader = _Reader(_payload())
    monkeypatch.setattr(prep, "read_week_review", reader)
    page.reload()
    _drain(reader, page)
    assert "3 of 5 sessions have facts" in page.facts_note.text()


def test_the_tally_on_a_card_is_that_days_measured_tally(page, monkeypatch):
    """Hand-counted from `tj5_support`'s table: 2026-09-17 is 0 right, 2 wrong,
    0 unresolved. A card that printed the WEEK's tally on every day would read
    the same five times."""
    from ui.services import weekend_prep_service as prep

    reader = _Reader(_payload())
    monkeypatch.setattr(prep, "read_week_review", reader)
    page.reload()
    _drain(reader, page)

    card = {card.session: card for card in page.day_cards}["2026-09-17"]
    text = card.tally.text()
    assert "0" in text and "2" in text, text
    assert fx.tally_over(["2026-09-17"]) == {"right": 0, "wrong": 2, "unresolved": 0, "n": 2}


# ---------------------------------------------------------------------------
# one payload, one worker, one click at a time
# ---------------------------------------------------------------------------
def test_the_page_reads_one_payload_on_one_worker(page, monkeypatch):
    """ONE call, and NOT on the Qt thread.

    `WeekReviewPage.reload`'s own docstring records why: this method used to BE
    the read, and it was the worst measured stall on the desk.
    """
    from ui.services import weekend_prep_service as prep

    reader = _Reader(_payload())
    monkeypatch.setattr(prep, "read_week_review", reader)
    page.reload()
    _drain(reader, page)

    assert len(reader.calls) == 1, reader.calls
    assert reader.threads[0] != threading.get_ident(), "the week was read on the Qt thread"


def test_a_second_click_starts_nothing_while_the_first_is_still_reading(page, monkeypatch):
    """TJ-4's round-3 blocker, restated: any click that starts work is ONE at a
    time, with its button disabled until every ending answers."""
    from ui.services import weekend_prep_service as prep

    reader = _Reader(_payload())
    reader.release.clear()
    monkeypatch.setattr(prep, "read_week_review", reader)

    page.reload()
    assert reader.entered.wait(5.0)
    assert page.refresh_button.isEnabled() is False
    page.reload()
    page.reload()
    reader.release.set()
    _drain(reader, page)

    assert len(reader.calls) == 1, reader.calls
    assert page.refresh_button.isEnabled() is True


def test_a_failed_read_re_enables_the_button_and_says_what_happened(page, monkeypatch):
    """Every ending answers. A reader that raises must not leave the page with a
    dead button and the word "Refreshing..." on it for the rest of the weekend."""
    from ui.services import weekend_prep_service as prep

    done = threading.Event()

    def _boom(**_kwargs):
        try:
            raise OSError("the pack folder is gone")
        finally:
            done.set()

    monkeypatch.setattr(prep, "read_week_review", _boom)
    page.reload()
    assert done.wait(5.0)
    worker = getattr(page, "_worker", None)
    if worker is not None:
        worker.wait(5000)
    _app.processEvents()

    assert page.refresh_button.isEnabled() is True
    assert "pack folder is gone" in page.refresh_note.text()


# ---------------------------------------------------------------------------
# the chart is built once
# ---------------------------------------------------------------------------
def test_the_small_spy_charts_are_built_once_and_reused(page, monkeypatch):
    """`plan.md` TJ-5 change 1: "a small SPY chart, built once on open and
    reused". Five chart widgets on five cards, and a second Refresh redraws them
    rather than replacing them - a rebuilt chart widget per refresh is a
    stylesheet pass and a relayout on the Qt thread, five times over."""
    from ui.services import weekend_prep_service as prep

    reader = _Reader(_payload())
    monkeypatch.setattr(prep, "read_week_review", reader)
    page.reload()
    _drain(reader, page)
    first = [card.chart for card in page.day_cards]
    assert len(first) == 5
    assert all(chart is not None for chart in first)

    reader.entered.clear()
    reader.done.clear()
    page.reload()
    _drain(reader, page)
    second = [card.chart for card in page.day_cards]
    assert all(a is b for a, b in zip(first, second, strict=False)), "the charts were rebuilt"


# ---------------------------------------------------------------------------
# the page computes nothing
# ---------------------------------------------------------------------------
def test_the_page_never_builds_a_pack_a_card_or_a_narration(page, monkeypatch):
    """One payload, on one worker, from ONE reader. The page may not reach past
    it into a builder - a 27B model load or a pack build behind a tab click is
    the whole reason the night window exists."""
    import ai_jobs.week_review_narration as week
    import day_report_card
    import day_review_pack

    from ui.services import weekend_prep_service as prep

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("the Week Review page called a builder on the Qt thread")

    monkeypatch.setattr(day_review_pack, "build_pack", _forbidden)
    monkeypatch.setattr(day_report_card, "build", _forbidden)
    monkeypatch.setattr(day_report_card, "week", _forbidden)
    monkeypatch.setattr(week, "run_week_review_narration", _forbidden)

    reader = _Reader(_payload())
    monkeypatch.setattr(prep, "read_week_review", reader)
    page.reload()
    _drain(reader, page)
    assert len(reader.calls) == 1


def test_the_empty_payload_has_every_key_and_nothing_in_it():
    """The shape of a first paint, a failed read and a quiet week are the same
    shape - the rule TJ-1's `empty_payload` ships and the reason the page has
    one render and no special cases."""
    from ui.services import weekend_prep_service as prep

    body = prep.empty_week_payload()
    assert set(body) == set(prep.WEEK_PAYLOAD_KEYS)
    assert not body["cards"]
    assert not body["sessions_with_facts"]
