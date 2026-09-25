"""Red reproduction tests for Mentor context failure boundaries."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

PACIFIC = ZoneInfo("America/Los_Angeles")
NOW = datetime(2026, 9, 14, 10, tzinfo=PACIFIC)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _days(last: date, count: int = 21) -> list[date]:
    import market_calendar

    rows = []
    while len(rows) < count:
        if market_calendar.is_session(last):
            rows.append(last)
        last -= timedelta(days=1)
    return list(reversed(rows))


def _d1(last: date = date(2026, 9, 11)) -> list[dict]:
    return [
        {"dt": day.isoformat(), "open": 99 + i, "high": 101 + i,
         "low": 98 + i, "close": 100 + i, "volume": 1_000_000}
        for i, day in enumerate(_days(last))
    ]


def _m5() -> list[dict]:
    starts, stamp = [], datetime(2026, 9, 14, 6, 30, tzinfo=PACIFIC)
    while stamp <= datetime(2026, 9, 14, 9, 55, tzinfo=PACIFIC):
        starts.append(stamp)
        stamp += timedelta(minutes=5)
    return [
        {"dt": point, "open": close - .25, "high": close + .5,
         "low": close - .5, "close": close, "volume": 1_000}
        for i, point in enumerate(starts)
        for close in (100 + i - (len(starts) - 7) if i >= len(starts) - 7 else 90,)
    ]


def _all(factory):
    from trade_mentor_context import SYMBOLS

    return {symbol: factory() for symbol in SYMBOLS}


def _readings(context):
    return {row["symbol"]: row for row in context["readings"]}


def test_one_failed_timeframe_keeps_the_other_timeframes_real_measurements():
    """One Yahoo leg failing is uncertainty for that leg, never an abort-all."""
    from trade_mentor_context import SYMBOLS
    from ui.services.trade_mentor_context_service import TradeMentorContextService

    def first_cache(timeframe, *_args, **_kwargs):
        return _all(_d1) if timeframe == "d1" else {}

    calls = []
    def m5_fails(timeframe, names, **_kwargs):
        calls.append((timeframe, tuple(names)))
        if timeframe == "m5":
            raise OSError("M5 down")
        return {}

    retained_d1 = TradeMentorContextService(
        loader=m5_fails, cache_loader=first_cache, clock=lambda: NOW
    )._build(NOW)
    assert all(row["d1_status"] == "measured" for row in _readings(retained_d1).values())
    assert all(row["m5_status"] == "unavailable" for row in _readings(retained_d1).values())
    assert calls == [("m5", SYMBOLS)], "one M5 batch; D1 was already good"

    def second_cache(timeframe, *_args, **_kwargs):
        return _all(_m5) if timeframe == "m5" else {}

    calls.clear()
    def d1_fails(timeframe, names, **_kwargs):
        calls.append((timeframe, tuple(names)))
        if timeframe == "d1":
            raise OSError("D1 down")
        return {}

    retained_m5 = TradeMentorContextService(
        loader=d1_fails, cache_loader=second_cache, clock=lambda: NOW
    )._build(NOW)
    assert all(row["m5_status"] == "measured" for row in _readings(retained_m5).values())
    assert all(row["d1_status"] == "unavailable" for row in _readings(retained_m5).values())
    assert calls == [("d1", SYMBOLS)], "one D1 batch; M5 was already good"


def test_real_mentor_note_keeps_full_metadata_and_scalar_context_in_small_ai_package(tmp_path):
    from PySide6.QtCore import QObject, Signal
    from ai_summary import build_evidence_package
    from evidence_ledger import EvidenceLedger
    import market_journal
    from trade_mentor_context import SYMBOLS, build_context
    import trade_mentor_schedule as schedule
    from ui.services.market_journal_service import MarketJournalService
    from ui.widgets.trade_mentor_card import TradeMentorCard

    class _Context(QObject):
        contextReady = Signal(str, object)
        contextUnavailable = Signal(str, object)
        def request_context(self, *_args, **_kwargs): return True

    context = build_context(now=NOW, m5_bars=_all(_m5), d1_bars=_all(_d1))
    context["availability"], context["reason"] = "stale", "M5 outer cache was stale"
    context["readings"][0].update(m5_status="unavailable", m5_reason="stale", m5_change_30m_pct=None)
    journal = MarketJournalService()
    journal._ledger = EvidenceLedger(stream=market_journal.STREAM, schema=market_journal.SCHEMA_MARKET_JOURNAL_ENTRY, directory=tmp_path / "ledger")
    service = _Context()
    card = TradeMentorCard(journal=journal, clock=lambda: NOW, drafts_path=tmp_path / "drafts.json", context_service=service)
    slot = next(item for item in schedule.slots_for_session(NOW.date()) if item.scheduled_at.hour == 9)
    card.show_slot(slot)
    service.contextReady.emit(slot.slot_id, context)
    QtWidgets.QApplication.instance().processEvents()
    card.text_box.setPlainText("I think SPY is holding the open.")
    # TJ-14A: a Mentor answer carries a forced prediction click, and `submit()`
    # is gated on it. The click is ADDED here; every assertion below is the
    # tester's original.
    card.prediction_button("rest_of_day", "up").click()
    card.confidence_button("rest_of_day", "medium").click()
    assert card.submit()["ok"] is True
    stored = journal.entries_for("2026-09-14")[-1]
    assert stored["mentor"]["context"] == context

    # The budget moved with the payload, and the point of the test did not: a
    # SMALL source budget must still carry the real row rather than a banner.
    # TJ-14A item 4 widened the snapshot (eighteen symbols, four more facts
    # each, the derived block), so one compacted Mentor row measures ~2,900
    # characters here against ~1,600 before it. Production's journal scope is
    # nowhere near this tight (80,000 total; this asks for 6,000).
    package = build_evidence_package(["market_journal"], source_overrides={"journal.entries": next((tmp_path / "ledger").glob("*.jsonl"))}, now=NOW, session_date="2026-09-11", budget_chars=6_000)
    source = next(row for row in package["sources"] if row["source_id"] == "journal.entries")
    assert all(isinstance(row, dict) for row in source["content"]), (
        "the bounded package must retain the real journal row, not replace it with a banner"
    )
    ai_row = next(row for row in source["content"] if row.get("text") == stored["text"])
    compact = ai_row["mentor"]["context_compact"]
    common = compact["common"]
    rows = [dict(common, **dict(zip(compact["columns"], values, strict=False))) for values in compact["rows"]]
    assert ai_row["text"] == "I think SPY is holding the open."
    # TJ-14A item 4 widened the live builder to v2 (the day's change, the place
    # in the day's range, both prior-session sides and the derived block). A
    # STORED v1 row stays readable and is pinned elsewhere; this snapshot is
    # built live, so it is v2.
    assert common["schema"] == "trade_mentor_context_v2"
    # The derived block reaches the AI as a readable SENTENCE as well as a
    # structure: `_bounded` cuts six levels down, which is where a derived
    # line's own lists sit, so the structure alone would have handed the model
    # "[nested content omitted]" where the sector names should be.
    assert "Breadth (RSP-SPY)" in common["internals"]
    assert "Leaders" in common["internals"]
    assert common["availability"] == "stale" and common["reason"] == "M5 outer cache was stale"
    assert common["rules"] == context["rules"] and common["sources"] == context["sources"]
    assert [row["symbol"] for row in rows] == list(SYMBOLS)
    spy = rows[6]
    assert common["captured_at"] == NOW.isoformat()
    assert spy["m5_change_30m_pct"] == pytest.approx(6.0)
    assert spy["d1_change_5d_pct"] == pytest.approx((120 - 115) / 115 * 100)
    assert spy["m5_as_of"] == "2026-09-14T09:55:00-07:00" and spy["d1_as_of"] == "2026-09-11"
    assert rows[0]["m5_change_30m_pct"] is None


def test_daily_cache_survives_empty_next_hour_and_refreshes_once_next_session():
    from ui.services.trade_mentor_context_service import TradeMentorContextService

    calls, first = [], {"value": True}
    def cache(timeframe, *_args, **_kwargs):
        return _all(_d1) if timeframe == "d1" and first["value"] else {}
    def loader(timeframe, names, **_kwargs):
        calls.append((timeframe, tuple(names)))
        return _all(lambda: _d1(date(2026, 9, 14))) if timeframe == "d1" else {}

    service = TradeMentorContextService(loader=loader, cache_loader=cache, clock=lambda: NOW)
    assert all(row["d1_status"] == "measured" for row in _readings(service._build(NOW)).values())
    first["value"] = False
    hour_later = NOW + timedelta(hours=1)
    assert all(row["d1_status"] == "measured" for row in _readings(service._build(hour_later)).values())
    assert not [call for call in calls if call[0] == "d1"], "same completed session reuses good D1"
    next_session = datetime(2026, 9, 15, 10, tzinfo=PACIFIC)
    assert all(row["d1_status"] == "measured" for row in _readings(service._build(next_session)).values())
    assert len([call for call in calls if call[0] == "d1"]) == 1

    empty_calls = []
    def empty_loader(timeframe, names, **_kwargs):
        empty_calls.append((timeframe, tuple(names)))
        return {}

    empty = TradeMentorContextService(
        loader=empty_loader, cache_loader=lambda *_args, **_kwargs: {}, clock=lambda: NOW
    )
    empty._build(NOW)
    empty._build(hour_later)
    assert len([call for call in empty_calls if call[0] == "d1"]) == 1
    empty._build(next_session)
    assert len([call for call in empty_calls if call[0] == "d1"]) == 2


def test_missing_cache_m5_failure_still_fetches_useful_d1():
    from trade_mentor_context import SYMBOLS
    from ui.services.trade_mentor_context_service import TradeMentorContextService

    calls = []
    def loader(timeframe, names, **_kwargs):
        calls.append((timeframe, tuple(names)))
        if timeframe == 'm5':
            raise OSError('M5 unavailable')
        return _all(_d1)

    service = TradeMentorContextService(loader=loader, clock=lambda: NOW)
    context = service._build(NOW)
    assert calls == [('m5', SYMBOLS), ('d1', SYMBOLS)]
    assert all(row['d1_status'] == 'measured' for row in context['readings'])
    assert all(row['m5_status'] == 'unavailable' for row in context['readings'])


def test_partial_daily_fetch_survives_and_local_cache_fills_missing_symbol():
    from trade_mentor_context import SYMBOLS
    from ui.services.trade_mentor_context_service import TradeMentorContextService

    calls, local = [], {}
    def cache(timeframe, *_args, **_kwargs):
        return dict(local) if timeframe == 'd1' else {}
    def loader(timeframe, names, **_kwargs):
        calls.append((timeframe, tuple(names)))
        return {name: _d1() for name in SYMBOLS[:-1]} if timeframe == 'd1' else {}

    service = TradeMentorContextService(loader=loader, cache_loader=cache, clock=lambda: NOW)
    for hour in (0, 1):
        rows = _readings(service._build(NOW + timedelta(hours=hour)))
        assert all(rows[name]['d1_status'] == 'measured' for name in SYMBOLS[:-1])
        assert rows[SYMBOLS[-1]]['d1_status'] == 'unavailable'
    local[SYMBOLS[-1]] = _d1()
    # A stale local entry must not overwrite its valid session-cached peer.
    local[SYMBOLS[0]] = _d1(date(2026, 9, 10))
    rows = _readings(service._build(NOW + timedelta(hours=2)))
    assert all(row['d1_status'] == 'measured' for row in rows.values())
    assert len([call for call in calls if call[0] == 'd1']) == 1
