"""TJ-11F - the 18 live Friday-evening calls land on FRIDAY's Day Review.

Packet `.claude/packets/TJ-11F.md` item 3, end to end through the real read.
The trader, 2026-09-19: *"a veto on friday night (after the market close)
should not be considered monday since we have new information then."*

The live shape, measured read-only on a COPY of
``C:\\TradingBotData\\trader_annotations.jsonl`` (2026-09-19): 1,198 rows, of
which **18** carry `session_date` 2026-09-19 - a Saturday - because they were
filed 2026-09-18 21:04-21:07 Pacific, which is after midnight in New York.
Twelve are vetoes and six are claimed likes; all eighteen are D1; none of them
carries a `decision_session` key at all, because the desk's last write to that
file predates wave 1. Their symbols are reproduced here.

The contract these tests pin, so the builder has nothing to guess
-----------------------------------------------------------------

``DayReviewService.read_day("2026-09-18")`` returns a payload whose
``walkaway`` holds all eighteen: the twelve vetoes in ``rejected`` and the six
likes in ``liked_not_traded``. ``read_day("2026-09-21")`` holds none of them -
Monday's scan is new information. Still ONE payload, still one read, still
`PAYLOAD_KEYS` and nothing more.

And the disagreement TJ-11's review found goes away: for a call made after the
close on a session day, the row the store WRITES and the session the service
PLACES it on say the same date.

Nothing here touches a live store: every door is monkeypatched to plain dicts.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

EASTERN = ZoneInfo("America/New_York")
PACIFIC = ZoneInfo("America/Los_Angeles")

FRIDAY = "2026-09-18"
SATURDAY = "2026-09-19"
MONDAY = "2026-09-21"
NOW = datetime(2026, 9, 22, 8, 0)

#: The twelve vetoed and six claimed-like symbols the trader really filed at
#: 21:04-21:07 Pacific on Friday 2026-09-18.
VETOED = ("AVT", "ETN", "HLIT", "MKC", "OGE", "PDM", "QDEL", "SEI", "SRE", "TAP", "WEC", "WELL")
LIKED = ("EBAY", "MSTR", "RELX", "SMR", "SRAD", "VSAT")


class _Journal:
    def entries_about(self, _session):
        return []

    def daily_story(self, _session):
        return None

    def theses_for(self, _session):
        return []


def _wire(monkeypatch, annotations):
    """Point `read_day` at plain dicts. No live store is opened."""
    import chart_snapshot
    import claimed_picks
    import daily_recap_reader
    import day_review_bars
    import journal_store
    from ui.services.day_review_service import DayReviewService

    store = {"annotations": list(annotations), "pick_feedback": [],
             "swing_favorites": [], "review_events": []}

    class _Store:
        def __init__(self, rows):
            self.rows = list(rows)

    monkeypatch.setattr(daily_recap_reader, "_read_jsonl",
                        lambda name, *a, **k: _Store(store.get(name, [])))
    monkeypatch.setattr(daily_recap_reader, "_read_csv", lambda name, *a, **k: _Store([]))
    monkeypatch.setattr(claimed_picks, "load_rows", lambda *a, **k: [])
    monkeypatch.setattr(journal_store, "JournalStore",
                        lambda *a, **k: type("_J", (), {"list_trades": lambda self: []})())
    monkeypatch.setattr(day_review_bars, "read_session_bars", lambda *a, **k: {})
    monkeypatch.setattr(day_review_bars, "session_is_closed", lambda *a, **k: True)
    monkeypatch.setattr(day_review_bars, "session_is_backfillable", lambda *a, **k: False)

    def _load(symbol):
        from market_calendar import previous_session

        cursor = date.fromisoformat(MONDAY)
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
    return service


def _the_eighteen() -> list[dict]:
    """The live rows' shape: a Saturday `session_date`, a Friday-night stamp.

    The stamp is aware Pacific exactly as the desk wrote it, and no row carries
    `decision_session` - all 1,198 rows on the desk predate wave 1.
    """
    rows: list[dict] = []
    minute, second = 4, 28
    for index, symbol in enumerate(VETOED + LIKED):
        stamp = datetime(2026, 9, 18, 21, minute + index // 6, second + index, tzinfo=PACIFIC)
        row = {
            "schema_version": 1, "event_id": f"live-{symbol}", "symbol": symbol,
            "side": "LONG", "session_date": SATURDAY, "timeframe": "D1",
            "created_at": stamp.isoformat(), "source": "chart_review",
            "surface": "chart_review",
        }
        if symbol in VETOED:
            row |= {"event_type": "veto", "reason_code": "extended", "vocab_version": 3}
        else:
            row |= {"event_type": "like_claim", "like_mode": "claimed",
                    "claimed_setup_id": "trendline_break"}
        rows.append(row)
    return rows


def test_the_eighteen_friday_evening_calls_are_on_fridays_day_review(monkeypatch):
    from ui.services.day_review_service import PAYLOAD_KEYS

    service = _wire(monkeypatch, _the_eighteen())

    payload = service.read_day(FRIDAY, now=NOW)

    assert set(payload) >= set(PAYLOAD_KEYS)
    day = payload["walkaway"]
    assert day is not None, payload.get("error")
    assert sorted(row.symbol for row in day.rejected) == sorted(VETOED)
    assert sorted(row.symbol for row in day.liked_not_traded) == sorted(LIKED)


def test_the_eighteen_friday_evening_calls_are_not_on_mondays_day_review(monkeypatch):
    """Monday's scan is new information. The weekend's judgement is Friday's."""
    service = _wire(monkeypatch, _the_eighteen())

    day = service.read_day(MONDAY, now=NOW)["walkaway"]

    assert day is not None
    assert [row.symbol for row in day.rejected] == []
    assert [row.symbol for row in day.liked_not_traded] == []


def test_the_stored_session_and_the_page_agree_for_an_after_close_call(monkeypatch):
    """TJ-11's review found the row saying Monday and the page saying Friday.

    Written by the REAL store writer, read by the REAL service: a veto filed at
    16:30 Eastern on Friday is Friday's on both sides.
    """
    from ui.annotations import store as annotation_store
    from ui.annotations.vocabulary import load_veto_vocabulary

    written = annotation_store.build_annotation(
        annotation_store.EVENT_VETO,
        symbol="HLIT",
        side="SHORT",
        reason_code=list(load_veto_vocabulary().codes)[0],
        timeframe="D1",
        session_date=FRIDAY,
        created_at=datetime(2026, 9, 18, 16, 30, tzinfo=EASTERN),
    )
    service = _wire(monkeypatch, [written])

    day = service.read_day(FRIDAY, now=NOW)["walkaway"]

    assert written[annotation_store.DECISION_SESSION_FIELD] == FRIDAY
    assert [row.symbol for row in day.rejected] == ["HLIT"]
    assert [row.symbol for row in service.read_day(MONDAY, now=NOW)["walkaway"].rejected] == []
