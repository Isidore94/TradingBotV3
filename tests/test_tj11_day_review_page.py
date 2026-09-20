"""TJ-11 Part B - the Day Review page and its one read. RED before the build.

Packet `.claude/packets/TJ-11.md` items 8 and 9; `plan.md` §12.4 "TJ-11".

Part A (`tests/test_tj11_real_miss.py`, `tests/test_tj11_walkaway_v2.py`) is the
pure half and goes green first. These tests are about what reaches the trader's
screen, and about the ONE read that feeds it.

The contract these tests pin, so the builder has nothing to guess
-----------------------------------------------------------------

``scripts/ui/services/day_review_service.py``
    ``read_day`` still returns :data:`PAYLOAD_KEYS` and nothing more; the new
    populations ride inside ``payload["walkaway"]`` (``earlier_calls``,
    ``skill``, ``sentences``, ``money``). It forwards each decision's ``reason``
    (the veto vocabulary code) so the table's sentence can count reasons without
    a second read. Daily bars for ATR and the D1 horizons come from
    ``chart_snapshot.load_d1_bars`` - the symbol-level DAILY reader the desk
    already uses OFF the Qt thread (`scripts/chart_snapshot.py:321`; NOT
    `scripts/ui/journal_chart_bars.py`, which mutates the Alert Center cache and
    arms a `QTimer.singleShot`) - read at most once per symbol.

``scripts/ui/panels/day_review_panel.py``
    ``TJ2B_WALKAWAY_COLUMNS`` gains the new moves with FULL headers;
    ``walkaway_tables`` gains ``"earlier_calls"``; ``walkaway_skill`` is one
    label above the grid; ``walkaway_sentences`` is one label per population,
    above its table. Every table on the page keeps the TJ-1L width rule (last
    section stretched). ``render`` paints from the payload and starts no read.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Day Review page uses PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QTableWidget  # noqa: E402

EASTERN = ZoneInfo("America/New_York")
PACIFIC = ZoneInfo("America/Los_Angeles")

SESSION = "2026-09-18"
NOW = datetime(2026, 9, 19, 8, 0)

#: The moves TJ-11 adds. Spelled in full - the TJ-1L width rule exists because
#: the shared 260 px ceiling clipped "Against me first %" at both ends.
NEW_COLUMNS = (
    "Against you first %",
    "At the close %",
    "Ran after (ATR)",
    "Against you first (ATR)",
    "At the close (ATR)",
    "Real miss",
)

POPULATIONS = (
    "liked_not_traded", "rejected", "traded_left_early", "claimed_d1", "earlier_calls",
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


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


def _walkaway_day(**overrides):
    """A real `WalkawayDay`, built by the real builder, with one row per table."""
    from walkaway_day import WalkawayDay, WalkawayRow

    def _row(symbol: str, **kw):
        fields = dict(
            decision_id=(SESSION, symbol, "LONG", "chart_review", "veto", "D1", ""),
            time=datetime(2026, 9, 18, 13, 0, tzinfo=EASTERN),
            symbol=symbol,
            side="LONG",
            category="chart_review",
            what_you_did="veto",
            ran_after_pct=10.0,
            state="measured",
        )
        fields.update(kw)
        return WalkawayRow(**fields)

    day = WalkawayDay(
        liked_not_traded=(_row("AAA"),),
        rejected=(_row("BBB"),),
        traded_left_early=(_row("CCC"),),
        claimed_d1=(_row("DDD"),),
    )
    for name, value in overrides.items():
        object.__setattr__(day, name, value)
    return day


def _payload(**overrides):
    from ui.services.day_review_service import empty_payload

    payload = empty_payload(SESSION)
    payload["walkaway"] = _walkaway_day()
    payload.update(overrides)
    return payload


# -- the page ----------------------------------------------------------------


def test_the_page_shows_a_fifth_earlier_calls_table(panel):
    assert "earlier_calls" in panel.walkaway_tables
    assert isinstance(panel.walkaway_tables["earlier_calls"], QTableWidget)
    assert len(panel.walkaway_tables) == 5


def test_the_fifth_table_is_inside_the_walkaway_grid(panel):
    table = panel.walkaway_tables["earlier_calls"]
    owner = panel.walkaway_grid.parentWidget()

    parent = table.parentWidget()
    while parent is not None and parent is not owner:
        parent = parent.parentWidget()
    assert parent is owner, "the fifth population is not in the walk-away grid"


def test_every_walkaway_table_carries_the_new_columns_with_full_headers(panel):
    from ui.panels.day_review_panel import TJ2B_WALKAWAY_COLUMNS

    for column in NEW_COLUMNS:
        assert column in TJ2B_WALKAWAY_COLUMNS, column
    for name, table in panel.walkaway_tables.items():
        headers = [
            table.horizontalHeaderItem(index).text()
            for index in range(table.columnCount())
        ]
        for column in NEW_COLUMNS:
            assert column in headers, (name, column, headers)


def test_every_walkaway_table_keeps_the_tj1l_width_rule(panel):
    for name, table in panel.walkaway_tables.items():
        assert table.horizontalHeader().stretchLastSection() is True, name


def test_the_skill_line_sits_above_the_walkaway_grid(panel):
    assert isinstance(panel.walkaway_skill, QLabel)

    owner = panel.walkaway_grid.parentWidget()
    parent = panel.walkaway_skill.parentWidget()
    while parent is not None and parent is not owner:
        parent = parent.parentWidget()
    assert parent is not owner, "the skill line is a header, not a grid cell"


def test_every_population_gets_its_own_sentence_label(panel):
    assert set(panel.walkaway_sentences) == set(POPULATIONS)
    for name, label in panel.walkaway_sentences.items():
        assert isinstance(label, QLabel), name


def test_rendering_paints_the_sentence_and_the_skill_line_from_the_payload(panel):
    skill = {
        "session": {
            "window_sessions": 1,
            "cells": (
                {"population": "liked_or_claimed", "side": "LONG", "setup_family": "",
                 "n": 55, "measured": 55, "unmeasured": 0, "runs": 12, "rate": 0.21,
                 "low": 0.12, "high": 0.34, "reportable": True},
            ),
            "overlapping": (),
            "sentence": "Likes 21% (n 55), vetoes 8% (n 92), untouched 9% (n 310).",
        },
        "lately": {"window_sessions": 20, "cells": (), "overlapping": (), "sentence": ""},
    }
    day = _walkaway_day()
    object.__setattr__(day, "skill", skill)
    object.__setattr__(day, "sentences", {
        name: f"{name} sentence" for name in POPULATIONS
    })

    panel.render(_payload(walkaway=day))

    assert "n 55" in panel.walkaway_skill.text()
    assert panel.walkaway_sentences["rejected"].text() == "rejected sentence"


def test_an_unmeasured_move_paints_a_dash_and_never_a_zero(panel):
    from ui.panels.day_review_panel import UNMEASURED

    day = _walkaway_day()
    object.__setattr__(day, "sentences", dict.fromkeys(POPULATIONS, ""))
    object.__setattr__(day, "skill", None)
    panel.render(_payload(walkaway=day))

    table = panel.walkaway_tables["rejected"]
    headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
    for column in ("Against you first %", "Real miss", "At the close (ATR)"):
        text = table.item(0, headers.index(column)).text()
        assert text == UNMEASURED, (column, text)


def test_rendering_the_walkaway_grid_starts_no_second_read(panel):
    day = _walkaway_day()
    object.__setattr__(day, "sentences", dict.fromkeys(POPULATIONS, ""))
    object.__setattr__(day, "skill", None)
    before = panel._service.reads if hasattr(panel, "_service") else panel.service.reads

    panel.render(_payload(walkaway=day))
    panel.render(_payload(walkaway=day))

    after = panel._service.reads if hasattr(panel, "_service") else panel.service.reads
    assert after == before


# -- the one read ------------------------------------------------------------


class _Journal:
    def entries_about(self, _session):
        return []

    def daily_story(self, _session):
        return None

    def theses_for(self, _session):
        return []


def _wire(monkeypatch, *, annotations, trades=()):
    """Point `read_day` at plain dicts, and record every daily-bar read."""
    import claimed_picks
    import daily_recap_reader
    import day_review_bars
    import journal_store
    import chart_snapshot
    from ui.services.day_review_service import DayReviewService

    store = {"annotations": annotations, "pick_feedback": [], "swing_favorites": [],
             "review_events": []}

    class _Store:
        def __init__(self, rows):
            self.rows = list(rows)

    monkeypatch.setattr(daily_recap_reader, "_read_jsonl",
                        lambda name, *a, **k: _Store(store.get(name, [])))
    monkeypatch.setattr(daily_recap_reader, "_read_csv",
                        lambda name, *a, **k: _Store([]))
    monkeypatch.setattr(claimed_picks, "load_rows", lambda *a, **k: [])
    monkeypatch.setattr(journal_store, "JournalStore",
                        lambda *a, **k: type("_J", (), {"list_trades": lambda self: list(trades)})())
    monkeypatch.setattr(day_review_bars, "read_session_bars", lambda *a, **k: {})
    monkeypatch.setattr(day_review_bars, "session_is_closed", lambda *a, **k: True)
    monkeypatch.setattr(day_review_bars, "session_is_backfillable", lambda *a, **k: False)

    calls: list[str] = []

    def _load(symbol):
        calls.append(str(symbol).upper())
        from market_calendar import previous_session
        cursor = datetime(2026, 9, 21).date()
        days = [cursor.isoformat()]
        for _ in range(19):
            cursor = previous_session(cursor)
            days.append(cursor.isoformat())
        return [
            {"dt": datetime.fromisoformat(day), "open": 100.0, "high": 101.0,
             "low": 99.0, "close": 100.0, "volume": 1000.0}
            for day in reversed(days)
        ]

    monkeypatch.setattr(chart_snapshot, "load_d1_bars", _load)

    service = DayReviewService(journal_service=_Journal())
    monkeypatch.setattr(service, "_read_recap", lambda *a, **k: object())
    monkeypatch.setattr(service, "_trades", lambda *a, **k: [])
    return service, calls


def _annotation(symbol: str, *, session_date: str, created_at: str, code: str = "extended"):
    return {
        "schema_version": 1, "event_id": f"e-{symbol}", "event_type": "veto",
        "symbol": symbol, "side": "SHORT", "session_date": session_date,
        "timeframe": "D1", "created_at": created_at, "source": "chart_review",
        "reason_code": code, "vocab_version": 3,
    }


def test_read_day_hands_the_page_the_new_populations_in_one_payload(monkeypatch):
    from ui.services.day_review_service import PAYLOAD_KEYS

    rows = [_annotation("BBB", session_date=SESSION, created_at="2026-09-18T13:00:00-04:00")]
    service, _calls = _wire(monkeypatch, annotations=rows)

    payload = service.read_day(SESSION, now=NOW)

    assert set(payload) >= set(PAYLOAD_KEYS)
    day = payload["walkaway"]
    assert day is not None, payload.get("error")
    for name in ("earlier_calls", "skill", "sentences", "money"):
        assert hasattr(day, name), name
    assert day.skill is not None


def test_read_day_forwards_the_veto_reason_so_the_sentence_can_count_it(monkeypatch):
    rows = [
        _annotation("BBB", session_date=SESSION, created_at="2026-09-18T13:00:00-04:00",
                    code="extended"),
        _annotation("CCC", session_date=SESSION, created_at="2026-09-18T13:01:00-04:00",
                    code="extended"),
        _annotation("DDD", session_date=SESSION, created_at="2026-09-18T13:02:00-04:00",
                    code="sma_incoming"),
    ]
    service, _calls = _wire(monkeypatch, annotations=rows)

    day = service.read_day(SESSION, now=NOW)["walkaway"]

    sentence = day.sentences["rejected"]
    assert sentence.startswith("You vetoed 3."), sentence
    assert "extended" in sentence, sentence
    assert "2 share" in sentence, sentence


def test_read_day_reads_daily_bars_through_the_off_qt_reader_once_per_symbol(monkeypatch):
    rows = [
        _annotation("BBB", session_date=SESSION, created_at="2026-09-18T13:00:00-04:00"),
        _annotation("BBB", session_date=SESSION, created_at="2026-09-18T13:30:00-04:00"),
        _annotation("CCC", session_date=SESSION, created_at="2026-09-18T13:02:00-04:00"),
    ]
    service, calls = _wire(monkeypatch, annotations=rows)

    service.read_day(SESSION, now=NOW)

    assert sorted(set(calls)) == ["BBB", "CCC"]
    assert len(calls) == len(set(calls)), f"a symbol was read twice: {calls}"


def test_a_friday_evening_call_reaches_fridays_day_review(monkeypatch):
    """The live case: 18 D1 calls stamped 2026-09-19, a Saturday.

    UPDATED by TJ-11F on the trader's word (2026-09-19): *"a veto on friday
    night (after the market close) should not be considered monday since we
    have new information then."* This pinned Monday's page under TJ-11.
    """
    rows = [
        _annotation("EEE", session_date="2026-09-19",
                    created_at="2026-09-18T21:04:28.734007-07:00"),
        _annotation("FFF", session_date="2026-09-21",
                    created_at="2026-09-21T13:00:00-04:00"),
    ]
    service, _calls = _wire(monkeypatch, annotations=rows)

    friday = service.read_day(SESSION, now=datetime(2026, 9, 22, 8, 0))["walkaway"]
    monday = service.read_day("2026-09-21", now=datetime(2026, 9, 22, 8, 0))["walkaway"]

    assert [row.symbol for row in friday.rejected] == ["EEE"]
    assert [row.symbol for row in monday.rejected] == ["FFF"]


def test_a_saturday_stamped_call_is_not_shown_on_the_monday_after_it(monkeypatch):
    rows = [
        _annotation("EEE", session_date="2026-09-19",
                    created_at="2026-09-18T21:04:28.734007-07:00"),
    ]
    service, _calls = _wire(monkeypatch, annotations=rows)

    day = service.read_day("2026-09-21", now=datetime(2026, 9, 22, 8, 0))["walkaway"]

    assert [row.symbol for row in day.rejected] == []
