"""TJ-3: a name's markers are placed on THAT NAME's tape, and claims draw.

Two things the packet's own files do not pin, added by the builder:

* **A marker index is an index into the bars it will be drawn on.** A name whose
  session tape is short (the bars file got 30 bars for it and 78 for SPY) places
  its decision on its OWN last completed bar at or before the stamp. The chart
  drops an index past the end of the tape it holds, so an index taken from
  another symbol's tape is a marker the trader never sees.
* **A claim gets a `claim` marker** when the payload already carries the claim
  rows (lead, 2026-09-19); the kind is otherwise unreachable, because no
  `daily_recap_reader` verdict is called `claim`.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

PACIFIC = ZoneInfo("America/Los_Angeles")
SESSION = "2026-09-18"
NOW = datetime(2026, 9, 19, 8, 0)


def _m5_bars(count: int = 78) -> list[dict]:
    first = datetime(2026, 9, 18, 6, 30, tzinfo=PACIFIC)
    return [
        {
            "dt": first + timedelta(minutes=5 * index),
            "open": 100.0, "high": 100.2, "low": 99.8, "close": 100.1, "volume": 1000,
        }
        for index in range(count)
    ]


def _decision(symbol: str, *, verdict: str, stamp: str):
    return {
        "session_date": SESSION, "symbol": symbol, "side": "LONG",
        "category": "chart_review", "verdict": verdict, "source": "annotations",
        "timeframe": "M5", "stamp": stamp, "capture_id": f"cap-{symbol}",
        "reason": "extended", "decision_session": SESSION,
    }


def test_a_decision_lands_on_the_last_bar_of_that_names_own_short_tape():
    """10:12 on a tape that ends at 08:55 is its LAST bar, never SPY's bar 44."""
    import day_review_markers

    short = _m5_bars(30)  # 06:30 through 08:55
    markers = day_review_markers.symbol_markers(
        "AAA", short,
        decisions=[_decision("AAA", verdict="veto", stamp="2026-09-18T17:12:00+00:00")],
    )

    assert [marker["index"] for marker in markers] == [29]
    assert markers[0]["index"] < len(short), "the marker names a bar this tape lacks"
    # The SAME decision on the full session tape lands on the 10:10 bar.
    assert day_review_markers.symbol_markers(
        "AAA", _m5_bars(),
        decisions=[_decision("AAA", verdict="veto", stamp="2026-09-18T17:12:00+00:00")],
    )[0]["index"] == 44


def test_name_charts_places_each_name_against_the_tape_it_hands_back():
    import day_review_markers

    charts = day_review_markers.name_charts(
        {"AAA": _m5_bars(30), "SPY": _m5_bars()},
        decisions=[_decision("AAA", verdict="veto", stamp="2026-09-18T17:12:00+00:00")],
    )

    assert set(charts) == {"AAA"}, "a name nobody decided on grew a chart"
    drawn = charts["AAA"]
    assert all(
        0 <= marker["index"] < len(drawn["bars"]) for marker in drawn["markers"]
    ), "a marker index points past the bars it is drawn on"


def test_a_traded_name_gets_a_chart_even_with_no_recorded_verdict():
    import day_review_markers

    charts = day_review_markers.name_charts(
        {"BBB": _m5_bars()},
        trades=[{
            "trade_id": "t-1", "symbol": "BBB", "direction": "LONG",
            "opened_at": "2026-09-18T18:00:00+00:00",
            "closed_at": "2026-09-18T18:47:00+00:00", "status": "closed",
        }],
    )

    assert set(charts) == {"BBB"}
    assert [marker["kind"] for marker in charts["BBB"]["markers"]] == [
        "trade_open", "trade_close"
    ]


def test_a_claim_row_draws_a_claim_marker_on_its_own_name():
    import day_review_markers

    claims = [
        {"schema": "claimed_pick_v1", "action": "claim", "symbol": "AAA",
         "side": "LONG", "horizon": "swing", "claimed_setup_id": "setup-7",
         "claim_at": "2026-09-18T10:12:00-07:00", "session_date": SESSION},
        {"schema": "claimed_pick_v1", "action": "claim", "symbol": "ZZZ",
         "side": "LONG", "horizon": "swing", "claimed_setup_id": "setup-9",
         "claim_at": "2026-09-18T10:12:00-07:00", "session_date": SESSION},
    ]

    markers = day_review_markers.symbol_markers("AAA", _m5_bars(), claims=claims)

    assert [(m["kind"], m["ref_id"], m["index"]) for m in markers] == [
        ("claim", "setup-7", 44)
    ]
    assert "claim" in day_review_markers.MARKER_KINDS


def test_read_day_places_a_name_marker_on_that_names_tape(monkeypatch):
    """The same rule through the worker: AAA's 30-bar tape, AAA's own index."""
    import chart_snapshot
    import claimed_picks
    import daily_recap_reader
    import day_review_bars
    import journal_store
    from ui.services.day_review_service import DayReviewService

    rows = {
        "annotations": [{
            "schema_version": 1, "event_id": "e-AAA", "event_type": "veto",
            "symbol": "AAA", "side": "LONG", "session_date": SESSION,
            "timeframe": "M5", "created_at": "2026-09-18T17:12:00+00:00",
            "source": "chart_review", "reason_code": "extended", "vocab_version": 3,
        }],
    }

    class _Store:
        def __init__(self, values):
            self.rows = list(values)

    monkeypatch.setattr(daily_recap_reader, "_read_jsonl",
                        lambda name, *a, **k: _Store(rows.get(name, [])))
    monkeypatch.setattr(daily_recap_reader, "_read_csv", lambda name, *a, **k: _Store([]))
    monkeypatch.setattr(claimed_picks, "load_rows", lambda *a, **k: [])
    monkeypatch.setattr(
        journal_store, "JournalStore",
        lambda *a, **k: type("_J", (), {"list_trades": lambda self: []})(),
    )
    monkeypatch.setattr(day_review_bars, "read_session_bars",
                        lambda *a, **k: {"SPY": _m5_bars(), "AAA": _m5_bars(30)})
    monkeypatch.setattr(day_review_bars, "session_is_closed", lambda *a, **k: True)
    monkeypatch.setattr(day_review_bars, "session_is_backfillable", lambda *a, **k: False)
    monkeypatch.setattr(chart_snapshot, "load_d1_bars", lambda _symbol: [])

    class _Journal:
        def entries_about(self, _session):
            return []

        def daily_story(self, _session):
            return None

        def theses_for(self, _session):
            return []

    service = DayReviewService(journal_service=_Journal())
    monkeypatch.setattr(service, "_read_recap", lambda *a, **k: object())
    monkeypatch.setattr(service, "_trades", lambda *a, **k: [])

    payload = service.read_day(SESSION, now=NOW)

    chart = payload["name_charts"]["AAA"]
    assert len(chart["bars"]) == 30
    assert [marker["index"] for marker in chart["markers"]] == [29]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
