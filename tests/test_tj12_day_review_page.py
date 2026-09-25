"""TJ-12 item 3 - the card heads Day Review, built on the ONE worker payload.

*"The page: the card heads Day Review above the story; a line click scrolls to
its target. One worker, one payload."* (packet TJ-12 item 3.)

TJ-1's rule is the one this must not break: `Day Review` reads ONE payload on
ONE worker. The card is another projection of that payload - not a second read,
and nothing about it happens on the Qt thread.

THE CONTRACT
------------
    ui/services/day_review_service.py
        `PAYLOAD_KEYS` gains `report_card`; `empty_payload` carries it PRESENT
        and empty; `read_day` fills it, and calls
        `prediction_ledger.your_reads` itself - on the worker - because that is
        a FILE read and the page may not do one.

    ui/panels/day_review_panel.py
        `REPORT_CARD_OBJECT_NAME` names the card widget.
        `panel.report_card_section` sits in the page's top-level column ABOVE
        `panel.columns` (which holds *What happened*).
        `panel.report_card_lines` - six widgets, in `LINE_KEYS` order, after a
        render.
        `panel.reveal_card_target(target)` -> the widget a line click opens.
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
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

import tj12_support as fx  # noqa: E402

SESSION = fx.SESSION
NOW = datetime(2026, 9, 19, 7, 30)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ---------------------------------------------------------------------------
# the worker payload
# ---------------------------------------------------------------------------
def test_the_payload_declares_the_card_present_and_empty():
    from ui.services.day_review_service import PAYLOAD_KEYS, empty_payload

    assert "report_card" in PAYLOAD_KEYS
    blank = empty_payload(SESSION)
    assert "report_card" in blank
    assert not blank["report_card"], "a first paint has an EMPTY card, not a fake one"


def _wire(monkeypatch):
    """`read_day` over plain dicts. No live store is opened."""
    import chart_snapshot
    import claimed_picks
    import daily_recap_reader
    import day_review_bars
    import journal_store
    from ui.services.day_review_service import DayReviewService

    class _Store:
        def __init__(self, rows):
            self.rows = list(rows)

    class _Journal:
        def entries_about(self, _session):
            return []

        def daily_story(self, _session):
            return None

        def theses_for(self, _session):
            return []

    monkeypatch.setattr(daily_recap_reader, "_read_jsonl", lambda name, *a, **k: _Store([]))
    monkeypatch.setattr(daily_recap_reader, "_read_csv", lambda name, *a, **k: _Store([]))
    monkeypatch.setattr(claimed_picks, "load_rows", lambda *a, **k: [])
    monkeypatch.setattr(
        journal_store, "JournalStore",
        lambda *a, **k: type("_J", (), {"list_trades": lambda self: []})(),
    )
    monkeypatch.setattr(day_review_bars, "read_session_bars", lambda *a, **k: {})
    monkeypatch.setattr(day_review_bars, "session_is_closed", lambda *a, **k: True)
    monkeypatch.setattr(day_review_bars, "session_is_backfillable", lambda *a, **k: False)
    monkeypatch.setattr(chart_snapshot, "load_d1_bars", lambda _symbol: [])

    service = DayReviewService(journal_service=_Journal())
    monkeypatch.setattr(service, "_read_recap", lambda *a, **k: object())
    monkeypatch.setattr(service, "_trades", lambda *a, **k: [])
    return service


def test_read_day_builds_the_card_inside_the_one_payload(monkeypatch):
    import day_report_card

    service = _wire(monkeypatch)
    payload = service.read_day(SESSION, now=NOW)

    card = payload["report_card"]
    assert card, payload.get("error")
    keys = tuple(line["key"] for line in (card["lines"] if isinstance(card, dict) else card.lines))
    assert keys == tuple(day_report_card.LINE_KEYS)


def test_the_worker_reads_the_ledger_through_the_owner_not_the_page(monkeypatch):
    """`prediction_ledger.your_reads` is a FILE read and belongs on the worker."""
    import prediction_ledger

    sentinel = {
        "text": "Your reads: 41 right of 57 - always up scored 13 of 19 on the same stamps",
        "session": SESSION, "n": 57, "right": 41, "wrong": 16, "flat": 0,
        "pending": 0, "empty": False,
        "baselines": {
            "always_up": {"right": 13, "n": 19, "rate": 13 / 19, "meets_floor": False},
            "same_as_the_last_hour": {"right": 0, "n": 0, "rate": None, "meets_floor": False},
            "with_the_d1_environment": {"right": 0, "n": 0, "rate": None, "meets_floor": False},
        },
    }
    called: list[str] = []

    def _your_reads(session, **_kwargs):
        called.append(session)
        return dict(sentinel)

    monkeypatch.setattr(prediction_ledger, "your_reads", _your_reads)

    service = _wire(monkeypatch)
    payload = service.read_day(SESSION, now=NOW)

    assert called == [SESSION], "the worker never asked the owner for the tally"
    card = payload["report_card"]
    lines = card["lines"] if isinstance(card, dict) else card.lines
    reads = next(line for line in lines if line["key"] == "your_reads")
    assert reads["n"] == 57


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------
class _StubService:
    """Reads nothing. The page is what is under test."""

    def __init__(self, payload=None) -> None:
        self.payload = payload or {}
        self.reads = 0

    def read_day(self, session_date, **_kwargs):
        self.reads += 1
        payload = dict(self.payload)
        payload.setdefault("session_date", session_date)
        return payload


@pytest.fixture()
def panel(qapp, monkeypatch):
    from ui.panels.day_review_panel import DayReviewPanel

    widget = DayReviewPanel(service=_StubService(), clock=lambda: NOW)
    monkeypatch.setattr(widget, "reload", lambda: None)
    yield widget
    try:
        widget.shutdown()
    except Exception:  # noqa: BLE001
        pass
    widget.deleteLater()
    qapp.processEvents()


def _payload(tmp_path):
    import day_report_card
    from ui.services.day_review_service import empty_payload

    fx.one_session_of_clicks(tmp_path)
    built = day_report_card.build(fx.day_inputs(tmp_path))
    payload = empty_payload(SESSION)
    payload["report_card"] = {
        "session": built.session,
        "lines": [dict(line) for line in built.lines],
    }
    return payload


def _index_in_page(panel, widget) -> int:
    body = panel.scroll.widget().layout()
    for index in range(body.count()):
        item = body.itemAt(index)
        if item.widget() is widget:
            return index
    raise AssertionError("the widget is not in the page's top-level column")


def test_the_card_heads_the_page_above_the_story(panel, tmp_path):
    from ui.panels.day_review_panel import REPORT_CARD_OBJECT_NAME

    panel.render(_payload(tmp_path))

    section = panel.report_card_section
    assert section.objectName() == REPORT_CARD_OBJECT_NAME
    assert _index_in_page(panel, section) < _index_in_page(panel, panel.columns), (
        "the report card must sit ABOVE the two columns that hold the story"
    )
    assert not panel.story_section.isAncestorOf(section)


def test_the_card_paints_its_six_lines_in_order(panel, tmp_path):
    import day_report_card

    panel.render(_payload(tmp_path))

    widgets = list(panel.report_card_lines)
    assert len(widgets) == 6
    texts = [w.text() for w in widgets]
    for key, text in zip(day_report_card.LINE_KEYS, texts, strict=False):
        assert text.strip(), key


def test_a_line_click_reveals_its_own_target(panel, tmp_path, monkeypatch):
    import day_report_card

    panel.render(_payload(tmp_path))

    revealed: list[str] = []
    monkeypatch.setattr(
        panel, "reveal_card_target", lambda target: revealed.append(target)
    )
    for widget in panel.report_card_lines:
        widget.click()

    assert revealed == [
        day_report_card.LINE_TARGETS[key] for key in day_report_card.LINE_KEYS
    ]


def test_every_target_resolves_to_a_widget_on_this_page(panel, tmp_path):
    import day_report_card

    panel.render(_payload(tmp_path))

    for key in day_report_card.LINE_KEYS:
        target = day_report_card.LINE_TARGETS[key]
        widget = panel.reveal_card_target(target)
        assert isinstance(widget, QWidget), f"{key} -> {target} opens nothing"
        assert panel.isAncestorOf(widget), f"{key} -> {target} is not on this page"


def test_the_page_never_builds_the_card_itself(panel, tmp_path, monkeypatch):
    """TJ-1: ONE payload, ONE worker. The page formats; it does not compute."""
    import day_report_card

    payload = _payload(tmp_path)
    monkeypatch.setattr(
        day_report_card, "build",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("built on the Qt thread")),
    )
    monkeypatch.setattr(
        day_report_card, "how_fresh",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("read on the Qt thread")),
    )
    panel.render(payload)
    assert len(list(panel.report_card_lines)) == 6


def test_an_empty_card_paints_without_inventing_a_number(panel):
    from ui.services.day_review_service import empty_payload

    panel.render(empty_payload(SESSION))
    for widget in panel.report_card_lines:
        assert "%" not in widget.text(), widget.text()
