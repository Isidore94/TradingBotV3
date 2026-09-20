"""TJ-3: a marker sits on a bar only when it happened during that bar.

Three things the packet's own files do not pin, added by the builder:

* **A marker index is an index into the bars it will be drawn on**, and a stamp
  the tape never covered has NO index. A name whose session tape is short (the
  bars file got 30 bars for it and 78 for SPY) does not have its 10:12 decision
  clamped onto its 08:55 candle: that marker is `after_tape`, kept so the page
  can say how many there are, and drawn nowhere. The reviewer measured the clamp
  on a copy of the live journal - 62 of 216 trade legs (29%) are filled after
  13:00 Pacific and were being drawn on the 12:55 candle.
* **A trade's two legs are separately ADDRESSABLE.** They share one `trade_id`,
  which is what the page SELECTS with (the tester's contract, unchanged), so the
  glyph carries a `marker_id` of `<trade_id>:in` / `:out` and
  `note_marker_position` can reach the exit.
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


def test_a_decision_after_a_short_tape_ends_is_not_clamped_onto_its_last_bar():
    """10:12 against a tape that ended at 08:55 is UNPLACED, not index 29."""
    import day_review_markers

    short = _m5_bars(30)  # 06:30 through 08:55
    markers = day_review_markers.symbol_markers(
        "AAA", short,
        decisions=[_decision("AAA", verdict="veto", stamp="2026-09-18T17:12:00+00:00")],
    )

    assert [marker["index"] for marker in markers] == [None]
    assert markers[0]["placement"] == "after_tape"
    # The SAME decision on the full session tape happened DURING the 10:10 bar.
    on_full = day_review_markers.symbol_markers(
        "AAA", _m5_bars(),
        decisions=[_decision("AAA", verdict="veto", stamp="2026-09-18T17:12:00+00:00")],
    )
    assert on_full[0]["index"] == 44
    assert on_full[0]["placement"] == "on_bar"


def test_a_stamp_eight_days_after_the_tape_is_unplaced():
    """The clamp reached across days, not just hours."""
    import day_review_markers

    assert day_review_markers.bar_index_for(
        _m5_bars(30), "2026-09-25T17:12:00+00:00"
    ) is None
    assert day_review_markers.placement_for(
        _m5_bars(30), "2026-09-25T17:12:00+00:00"
    ) == (None, "after_tape")


def test_a_fill_after_the_close_is_counted_rather_than_drawn_at_the_close():
    """The measured defect: 29% of the live journal's legs fill after 13:00 PT."""
    import day_review_markers

    tape = _m5_bars()  # 06:30 through 12:55, the regular session
    markers = day_review_markers.symbol_markers(
        "AAA", tape,
        trades=[{
            "trade_id": "t-late", "symbol": "AAA", "direction": "LONG",
            "opened_at": "2026-09-18T18:00:00+00:00",       # 11:00 PT, on a bar
            "closed_at": "2026-09-18T21:30:00+00:00",       # 14:30 PT, after it
            "status": "closed",
        }],
    )

    placed = {marker["kind"]: marker for marker in markers}
    assert placed["trade_open"]["index"] == 54
    assert placed["trade_close"]["index"] is None, "a 14:30 fill was drawn on a candle"
    assert placed["trade_close"]["placement"] == "after_tape"
    assert day_review_markers.placement_counts(markers) == {
        "on_bar": 1, "after_tape": 1, "between_bars": 0
    }
    # An unplaced mark never displaces a drawn one.
    assert [marker["kind"] for marker in markers] == ["trade_open", "trade_close"]


def test_the_last_bars_own_minutes_are_still_on_that_bar():
    """12:57 is INSIDE the 12:55 bar, and the tape's last candle carries it."""
    import day_review_markers

    tape = _m5_bars()
    assert day_review_markers.placement_for(tape, "2026-09-18T19:57:00+00:00") == (
        77, "on_bar"
    )
    assert day_review_markers.placement_for(tape, "2026-09-18T20:00:00+00:00") == (
        None, "after_tape"
    ), "13:00 is the first moment after the tape"


def test_a_daily_tape_still_matches_by_market_local_date():
    import day_review_markers

    daily = [
        {"dt": day, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5}
        for day in ("2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18")
    ]

    # 01:30 UTC on the 17th is 18:30 Pacific on the 16th.
    assert day_review_markers.placement_for(daily, "2026-09-17T01:30:00+00:00") == (
        2, "on_bar"
    )
    assert day_review_markers.placement_for(daily, "2026-09-11T20:00:00+00:00") == (
        None, "before_tape"
    )
    assert day_review_markers.placement_for(daily, "2026-09-25T20:00:00+00:00") == (
        None, "after_tape"
    )


def test_name_charts_places_each_name_against_the_tape_it_hands_back():
    import day_review_markers

    charts = day_review_markers.name_charts(
        {"AAA": _m5_bars(30), "SPY": _m5_bars()},
        decisions=[
            _decision("AAA", verdict="veto", stamp="2026-09-18T13:42:00+00:00"),
            _decision("AAA", verdict="pass", stamp="2026-09-18T17:12:00+00:00"),
        ],
    )

    assert set(charts) == {"AAA"}, "a name nobody decided on grew a chart"
    drawn = charts["AAA"]
    assert all(
        marker["index"] is None or 0 <= marker["index"] < len(drawn["bars"])
        for marker in drawn["markers"]
    ), "a marker index points past the bars it is drawn on"
    assert drawn["placements"] == {"on_bar": 1, "after_tape": 1, "between_bars": 0}


def test_a_stamp_in_a_hole_in_the_tape_is_neither_drawn_nor_lost():
    """A halt is a hole, and the candle to its left did not carry that decision."""
    import day_review_markers

    tape = _m5_bars(20)
    holed = tape[:5] + tape[15:]  # 07:00..08:10 never traded

    index, placement = day_review_markers.placement_for(
        holed, "2026-09-18T14:30:00+00:00"  # 07:30 PT, inside the hole
    )

    assert index is None
    assert placement == "between_bars"


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


def test_a_trades_two_legs_carry_one_selector_and_two_addresses():
    """`ref_id` selects (the tester's contract); `marker_id` addresses.

    Both legs are the same trade, so the page selects with one id - but two
    glyphs drawn at two different bars need two names, or nothing can ever ask
    where the EXIT was drawn.
    """
    import day_review_markers

    markers = day_review_markers.benchmark_markers(
        _m5_bars(),
        trades=[{
            "trade_id": "t-42", "symbol": "SPY", "direction": "LONG",
            "opened_at": "2026-09-18T18:00:00+00:00",
            "closed_at": "2026-09-18T18:47:00+00:00", "status": "closed",
        }],
    )

    assert {marker["ref_id"] for marker in markers} == {"t-42"}
    assert [marker["marker_id"] for marker in markers] == ["t-42:in", "t-42:out"]
    assert day_review_markers.TRADE_OPEN_SUFFIX == ":in"
    assert day_review_markers.TRADE_CLOSE_SUFFIX == ":out"


def test_read_day_never_reads_a_session_date_as_a_symbol(monkeypatch):
    """`stored` carries BOTH symbol -> bars and date -> {symbol: bars}.

    The walk-away read adds a LATER exit session's whole bars file under its
    DATE, so the mapping handed to `name_charts` has to be the session's symbol
    tapes and nothing else.
    """
    trades = [{
        "trade_id": "t-1", "symbol": "AAA", "direction": "LONG",
        "opened_at": "2026-09-18T18:00:00+00:00",
        "closed_at": "2026-09-22T18:47:00+00:00",
        "last_closing_leg_at": "2026-09-22T18:47:00+00:00",
        "status": "closed", "quantity": 100, "net_pnl": 42.0,
    }]
    service = _service(
        monkeypatch,
        session_bars={"SPY": _m5_bars(), "AAA": _m5_bars(30)},
        annotations=[_annotation("AAA", created_at="2026-09-18T13:42:00+00:00")],
        trades=trades,
    )

    payload = service.read_day(SESSION, now=NOW)

    assert set(payload["name_charts"]) == {"AAA"}, payload.get("error")
    assert "2026-09-22" not in payload["name_charts"]
    assert len(payload["name_charts"]["AAA"]["bars"]) == 30


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


def _annotation(symbol: str, *, created_at: str, verdict: str = "veto"):
    return {
        "schema_version": 1, "event_id": f"e-{symbol}", "event_type": verdict,
        "symbol": symbol, "side": "LONG", "session_date": SESSION,
        "timeframe": "M5", "created_at": created_at, "source": "chart_review",
        "reason_code": "extended", "vocab_version": 3,
    }


def _service(monkeypatch, *, session_bars, annotations=(), trades=()):
    """A `DayReviewService` whose every store is a plain dict. Reads nothing."""
    import chart_snapshot
    import claimed_picks
    import daily_recap_reader
    import day_review_bars
    import journal_store
    from ui.services.day_review_service import DayReviewService

    rows = {"annotations": list(annotations)}

    class _Store:
        def __init__(self, values):
            self.rows = list(values)

    monkeypatch.setattr(daily_recap_reader, "_read_jsonl",
                        lambda name, *a, **k: _Store(rows.get(name, [])))
    monkeypatch.setattr(daily_recap_reader, "_read_csv", lambda name, *a, **k: _Store([]))
    monkeypatch.setattr(claimed_picks, "load_rows", lambda *a, **k: [])
    monkeypatch.setattr(
        journal_store, "JournalStore",
        lambda *a, **k: type(
            "_J", (), {"list_trades": lambda self: [dict(row) for row in trades]}
        )(),
    )
    monkeypatch.setattr(day_review_bars, "read_session_bars",
                        lambda *a, **k: dict(session_bars))
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
    monkeypatch.setattr(service, "_trades", lambda *a, **k: [dict(row) for row in trades])
    return service


def test_read_day_places_a_name_marker_on_that_names_tape(monkeypatch):
    """The same rule through the worker: AAA's own 30-bar tape decides.

    07:42 PT happened during bar 14 of that tape; 10:12 PT never happened on it
    at all, and the payload says `after_tape` rather than naming a candle.
    """
    service = _service(
        monkeypatch,
        session_bars={"SPY": _m5_bars(), "AAA": _m5_bars(30)},
        annotations=[
            _annotation("AAA", created_at="2026-09-18T14:42:00+00:00"),
            _annotation("AAA", created_at="2026-09-18T17:12:00+00:00", verdict="pass"),
        ],
    )

    payload = service.read_day(SESSION, now=NOW)

    chart = payload["name_charts"]["AAA"]
    assert len(chart["bars"]) == 30
    assert [(m["kind"], m["index"], m["placement"]) for m in chart["markers"]] == [
        ("veto", 14, "on_bar"),
        ("pass", None, "after_tape"),
    ]
    assert chart["placements"] == {"on_bar": 1, "after_tape": 1, "between_bars": 0}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
