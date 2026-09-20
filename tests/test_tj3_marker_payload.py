"""TJ-3 item 2 - the PURE marker payload builder. RED before the build.

Packet `.claude/packets/TJ-3.md`; `plan.md` §12.4 "TJ-3" change 2. Trader,
2026-09-17: *"it also needs some sort of chart system to show me when I
commented on it so I can see exactly where I went wrong."*

The contract these tests pin, so the builder has nothing to guess
-----------------------------------------------------------------

``scripts/day_review_markers.py`` - pure: no Qt, no I/O, no clock read, no
store. Everything it needs is handed in.

``MARKER_KINDS: tuple[str, ...]``
    Every kind a marker may carry. `plan.md` names ten - ``note``, ``mentor``,
    ``forecast``, ``like``, ``pass``, ``veto``, ``click_away``, ``claim``,
    ``trade_open``, ``trade_close`` - and TJ-14A's split read (a description is
    not a prediction) needs one more, whose NAME is the builder's to choose.

``bar_index_for(bars, stamp) -> int | None``
    The index of the LAST bar at or before ``stamp``. ``None`` when the stamp
    sits before the first bar, carries no readable moment, or is absent - an
    unknown stamp yields NO marker and is never placed at an invented one.
    Stamps are compared with ``astimezone``, never by stripping a zone: the
    journal stores UTC and the tape is market-local. A daily bar whose ``dt`` is
    a date (the durable daily store, `market_story_rollups.load_index_bars`)
    compares by market-local DATE.

``benchmark_markers(bars, *, entries=(), trades=()) -> tuple[dict, ...]``
    The SPY chart: the trader's notes, the Mentor answers, the pasted forecast
    and the day's trades. A machine row (`market_journal.is_machine_entry`)
    never gets one.

``symbol_markers(symbol, bars, *, decisions=(), trades=()) -> tuple[dict, ...]``
    One name's chart: that name's decisions and that name's trades, nobody
    else's.

Each marker is ``{"stamp", "index", "kind", "label", "ref_id"}``; the tuple is
ordered by ``index`` ascending; ``ref_id`` is the id the page selects with
(a note's ``entry_id``, a trade's ``trade_id``) and is never empty.
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
SESSION = "2026-09-18"  # a Friday


def _m5_bars(count: int = 78) -> list[dict]:
    """The session tape as `day_review_bars` stores it: market-local, aware.

    06:30 Pacific through 12:55, five minutes apart - so bar 44 opens at 10:10
    and bar 54 at 11:00, and an index can be checked against a clock.
    """
    first = datetime(2026, 9, 18, 6, 30, tzinfo=PACIFIC)
    return [
        {
            "dt": first + timedelta(minutes=5 * index),
            "open": 100.0 + index * 0.01,
            "high": 100.2 + index * 0.01,
            "low": 99.8 + index * 0.01,
            "close": 100.1 + index * 0.01,
            "volume": 1000,
        }
        for index in range(count)
    ]


def _daily_bars() -> list[dict]:
    """Five daily bars as `market_story_rollups.load_index_bars` yields them.

    Its `dt` is an ISO **date string**, not a datetime - a reader that assumes
    a datetime cannot place a marker on the D1 toggle at all.
    """
    return [
        {"dt": day, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5}
        for day in ("2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18")
    ]


def _entry(text: str, created_at: str, *, origin: str = "", entry_id: str = "", **extra):
    """One stored Market Journal row.

    `mentor` is PRESENT AND EMPTY on every entry that no prompt asked for - the
    store has written it that way since Phase 0.31, and a reader that tells
    "no prompt" from "the key did not exist" by absence reads two different
    absences as one.
    """
    import market_journal

    row = {
        "event_type": "entry",
        "entry_id": entry_id or f"entry-{created_at}",
        "session_date": SESSION,
        "written_session_date": SESSION,
        "created_at": created_at,
        "created_local_date": SESSION,
        "written_after_the_session": False,
        "timeframe": "M5",
        "symbols": [],
        "origin": origin or market_journal.ORIGIN_JOURNAL_PAGE,
        "text": text,
        "supersedes": "",
        "mentor": {},
        "reaffirms": "",
    }
    row.update(extra)
    return row


def _mentor(hour_utc: str, *, kind: str = "m5_d1", **extra):
    """The `mentor` payload as `trade_mentor_card._mentor_payload` writes it.

    Its real keys TODAY are `slot_id`, `prompt_kind`, `scheduled_at`,
    `responded_at` and `context` - there is no `observation` and no
    `prediction` until TJ-14A splits the card. All three stamps are the SAME
    instant here (a reply typed on the hour), so no test below quietly pins
    WHICH of them a marker is placed at; that choice is the builder's.
    """
    payload = {
        "slot_id": f"{SESSION}-{hour_utc[11:13]}{hour_utc[14:16]}-{kind}",
        "prompt_kind": kind,
        "scheduled_at": hour_utc,
        "responded_at": hour_utc,
        "context": {"schema": "trade_mentor_context_v1", "readings": []},
    }
    payload.update(extra)
    return payload


def _decision(symbol: str, *, verdict: str, stamp: str, side: str = "LONG",
              capture_id: str | None = None, source: str = "annotations",
              category: str = "chart_review"):
    """A decision row exactly as `day_review_service` maps one forward.

    `capture_id` is PRESENT AND EMPTY on every row the pick-feedback and
    swing-favorite readers produce (`daily_recap_reader`), so a builder that
    uses it as the marker's id without a fallback loses those rows.
    """
    return {
        "session_date": SESSION,
        "symbol": symbol,
        "side": side,
        "category": category,
        "verdict": verdict,
        "source": source,
        "timeframe": "M5",
        "stamp": stamp,
        "capture_id": f"cap-{symbol}-{verdict}" if capture_id is None else capture_id,
        "reason": "extended",
        "decision_session": SESSION,
    }


def _trade(symbol: str, *, opened_at: str, closed_at: str = "", trade_id: str = "t-1"):
    return {
        "trade_id": trade_id,
        "symbol": symbol,
        "direction": "LONG",
        "opened_at": opened_at,
        "closed_at": closed_at,
        "status": "closed" if closed_at else "open",
        "quantity": 100,
        "net_pnl": 42.0,
    }


def _by_kind(markers):
    return {str(marker["kind"]) for marker in markers}


# -- where a stamp lands -----------------------------------------------------


def test_a_note_between_two_bars_lands_on_the_last_completed_bar_at_or_before_it():
    """10:12 belongs to the bar that OPENED at 10:10, never the one after it."""
    import day_review_markers

    bars = _m5_bars()
    stamp = datetime(2026, 9, 18, 10, 12, tzinfo=PACIFIC)

    assert day_review_markers.bar_index_for(bars, stamp) == 44
    assert bars[44]["dt"] == datetime(2026, 9, 18, 10, 10, tzinfo=PACIFIC)


def test_a_utc_note_is_converted_to_the_tapes_zone_and_never_stripped():
    """The journal stores UTC; the tape is market-local.

    17:12 UTC IS 10:12 Pacific. A reader that strips the zone and compares the
    wall clocks puts this note on the LAST bar of the day (index 77), which is
    the one place the trader was certainly not looking.
    """
    import day_review_markers

    bars = _m5_bars()

    index = day_review_markers.bar_index_for(bars, "2026-09-18T17:12:00+00:00")

    assert index == 44, "a UTC stamp was compared without converting it"


def test_a_stamp_before_the_first_bar_yields_no_marker():
    """A pre-market thought has no completed bar at or before it."""
    import day_review_markers

    bars = _m5_bars()

    assert day_review_markers.bar_index_for(bars, "2026-09-18T12:50:00+00:00") is None

    markers = day_review_markers.benchmark_markers(
        bars,
        entries=[_entry("before the bell", "2026-09-18T12:50:00+00:00", entry_id="e-pre")],
    )
    assert markers == ()


def test_an_unreadable_or_absent_stamp_yields_no_marker():
    import day_review_markers

    bars = _m5_bars()

    for stamp in ("", None, "not a date", "2026-13-45T99:99:99"):
        assert day_review_markers.bar_index_for(bars, stamp) is None, stamp

    markers = day_review_markers.benchmark_markers(
        bars,
        entries=[
            _entry("no stamp at all", "", entry_id="e-blank"),
            _entry("garbage stamp", "not a date", entry_id="e-bad"),
        ],
    )
    assert markers == ()


def test_no_bars_means_no_markers_rather_than_a_marker_at_zero():
    import day_review_markers

    assert day_review_markers.bar_index_for([], "2026-09-18T17:12:00+00:00") is None
    assert day_review_markers.benchmark_markers(
        [], entries=[_entry("a thought", "2026-09-18T17:12:00+00:00")]
    ) == ()


# -- the vocabulary ----------------------------------------------------------


def test_every_kind_the_plan_names_is_declared():
    """`plan.md` TJ-3 change 1 names ten. The eleventh is the prediction."""
    import day_review_markers

    assert set(day_review_markers.MARKER_KINDS) >= {
        "note", "mentor", "forecast", "like", "pass", "veto",
        "click_away", "claim", "trade_open", "trade_close",
    }


# -- the benchmark chart -----------------------------------------------------


def test_the_benchmark_chart_carries_notes_mentor_answers_the_forecast_and_trades():
    import day_review_markers
    import market_journal

    bars = _m5_bars()
    entries = [
        _entry("I like the base here", "2026-09-18T17:12:00+00:00", entry_id="e-note"),
        _entry(
            "08:00 read: choppy",
            "2026-09-18T15:00:00+00:00",
            origin=market_journal.ORIGIN_TRADE_MENTOR,
            entry_id="e-mentor",
            mentor=_mentor("2026-09-18T15:00:00+00:00"),
        ),
        _entry(
            "someone else's brief",
            "2026-09-18T14:03:00+00:00",
            origin=market_journal.ORIGIN_EXTERNAL_FORECAST,
            entry_id="e-forecast",
        ),
    ]
    trades = [_trade("SPY", opened_at="2026-09-18T18:00:00+00:00",
                     closed_at="2026-09-18T18:47:00+00:00", trade_id="t-spy")]

    markers = day_review_markers.benchmark_markers(bars, entries=entries, trades=trades)

    assert _by_kind(markers) == {"note", "mentor", "forecast", "trade_open", "trade_close"}
    placed = {marker["kind"]: marker["index"] for marker in markers}
    assert placed["note"] == 44
    assert placed["mentor"] == 18
    assert placed["forecast"] == 6
    assert placed["trade_open"] == 54
    assert placed["trade_close"] == 63
    for marker in markers:
        assert str(marker["label"]).strip(), marker
        assert str(marker["ref_id"]).strip(), marker
        assert marker["kind"] in day_review_markers.MARKER_KINDS, marker


def test_a_notes_ref_id_is_the_entry_id_the_page_selects_with():
    import day_review_markers

    markers = day_review_markers.benchmark_markers(
        _m5_bars(),
        entries=[_entry("the one", "2026-09-18T17:12:00+00:00", entry_id="e-the-one")],
    )

    assert [marker["ref_id"] for marker in markers] == ["e-the-one"]


def test_a_trades_ref_id_is_its_trade_id_on_both_legs():
    import day_review_markers

    markers = day_review_markers.benchmark_markers(
        _m5_bars(),
        trades=[_trade("SPY", opened_at="2026-09-18T18:00:00+00:00",
                       closed_at="2026-09-18T18:47:00+00:00", trade_id="t-42")],
    )

    assert {marker["ref_id"] for marker in markers} == {"t-42"}


def test_a_machine_row_never_gets_a_marker():
    """`is_machine_entry` is the ONE filter, and this page has no machine rows."""
    import day_review_markers
    import market_journal

    bars = _m5_bars()
    entries = [
        _entry("mine", "2026-09-18T17:12:00+00:00", entry_id="e-mine"),
        _entry(
            "Auto mode -> DESK",
            "2026-09-18T17:13:00+00:00",
            origin=market_journal.ORIGIN_AUTO_MODE_FLIP,
            entry_id="e-machine",
        ),
    ]

    markers = day_review_markers.benchmark_markers(bars, entries=entries)

    assert [marker["ref_id"] for marker in markers] == ["e-mine"]


def test_a_legacy_row_with_no_mentor_key_at_all_is_still_a_note():
    """Pre-Phase-0.31 rows predate `mentor`, and `entries_about` still returns them.

    "no prompt asked for this" and "this key did not exist yet" are two
    different absences; neither is a reason to lose the trader's words.
    """
    import day_review_markers

    legacy = _entry("an old thought", "2026-09-18T17:12:00+00:00", entry_id="e-old")
    legacy.pop("mentor")

    markers = day_review_markers.benchmark_markers(_m5_bars(), entries=[legacy])

    assert [(marker["ref_id"], marker["kind"]) for marker in markers] == [
        ("e-old", "note")
    ]


def test_an_observation_only_read_and_a_prediction_get_different_kinds():
    """A description is not a prediction (TJ-14 item 1), and the chart says so.

    The prediction kind's NAME is the builder's choice; that the two are not the
    same glyph is not. Three real shapes sit side by side here: the row the card
    writes TODAY (slot keys only, no `observation`, no `prediction`), the
    TJ-14A row that carries `observation` and nothing more, and the TJ-14A row
    where the trader actually clicked a direction. Only the third is a call.
    """
    import day_review_markers
    import market_journal

    bars = _m5_bars()
    today = _entry(
        "chop, no edge",
        "2026-09-18T15:00:00+00:00",
        origin=market_journal.ORIGIN_TRADE_MENTOR,
        entry_id="e-today",
        mentor=_mentor("2026-09-18T15:00:00+00:00"),
    )
    observed = _entry(
        "still chop",
        "2026-09-18T15:30:00+00:00",
        origin=market_journal.ORIGIN_TRADE_MENTOR,
        entry_id="e-observed",
        mentor=_mentor("2026-09-18T15:30:00+00:00", observation="still chop"),
    )
    predicted = _entry(
        "fading into the close",
        "2026-09-18T16:00:00+00:00",
        origin=market_journal.ORIGIN_TRADE_MENTOR,
        entry_id="e-predicted",
        mentor=_mentor(
            "2026-09-18T16:00:00+00:00",
            kind="m5",
            observation="",
            prediction={"direction": "down", "horizon": "rest_of_day",
                        "confidence": "medium", "because": "no bid"},
        ),
    )

    kinds = {
        marker["ref_id"]: marker["kind"]
        for marker in day_review_markers.benchmark_markers(
            bars, entries=[today, observed, predicted]
        )
    }

    assert kinds["e-today"] == "mentor"
    assert kinds["e-observed"] == "mentor", (
        "a described read was promoted to a call it never made"
    )
    assert kinds["e-predicted"] != kinds["e-observed"], (
        "a clicked prediction draws the same glyph as a description"
    )
    assert kinds["e-predicted"] in day_review_markers.MARKER_KINDS


def test_markers_come_back_in_bar_order():
    import day_review_markers

    bars = _m5_bars()
    entries = [
        _entry("third", "2026-09-18T19:00:00+00:00", entry_id="e-3"),
        _entry("first", "2026-09-18T14:03:00+00:00", entry_id="e-1"),
        _entry("second", "2026-09-18T17:12:00+00:00", entry_id="e-2"),
    ]

    markers = day_review_markers.benchmark_markers(bars, entries=entries)

    assert [marker["ref_id"] for marker in markers] == ["e-1", "e-2", "e-3"]
    assert [marker["index"] for marker in markers] == sorted(
        marker["index"] for marker in markers
    )


# -- a name's chart ----------------------------------------------------------


def test_a_names_chart_carries_only_that_names_decisions():
    import day_review_markers

    bars = _m5_bars()
    decisions = [
        _decision("AAA", verdict="veto", stamp="2026-09-18T17:12:00+00:00"),
        _decision("BBB", verdict="pass", stamp="2026-09-18T17:12:00+00:00"),
    ]

    markers = day_review_markers.symbol_markers("AAA", bars, decisions=decisions)

    assert len(markers) == 1
    assert markers[0]["kind"] == "veto"
    assert markers[0]["index"] == 44
    assert "veto" in str(markers[0]["label"]).lower()
    assert str(markers[0]["ref_id"]).strip()


def test_each_refusal_and_endorsement_draws_its_own_kind():
    import day_review_markers

    bars = _m5_bars()
    decisions = [
        _decision("AAA", verdict="veto", stamp="2026-09-18T17:12:00+00:00"),
        _decision("AAA", verdict="pass", stamp="2026-09-18T17:17:00+00:00"),
        _decision("AAA", verdict="m5_click_away", stamp="2026-09-18T17:22:00+00:00"),
        _decision("AAA", verdict="like", stamp="2026-09-18T17:27:00+00:00"),
    ]

    kinds = [marker["kind"] for marker in
             day_review_markers.symbol_markers("AAA", bars, decisions=decisions)]

    assert kinds == ["veto", "pass", "click_away", "like"]


def test_a_names_chart_carries_only_that_names_trades():
    import day_review_markers

    bars = _m5_bars()
    trades = [
        _trade("AAA", opened_at="2026-09-18T18:00:00+00:00",
               closed_at="2026-09-18T18:47:00+00:00", trade_id="t-aaa"),
        _trade("BBB", opened_at="2026-09-18T18:00:00+00:00",
               closed_at="2026-09-18T18:47:00+00:00", trade_id="t-bbb"),
    ]

    markers = day_review_markers.symbol_markers("AAA", bars, trades=trades)

    assert {marker["ref_id"] for marker in markers} == {"t-aaa"}
    assert _by_kind(markers) == {"trade_open", "trade_close"}


def test_a_decision_with_an_empty_capture_id_still_gets_a_usable_ref_id():
    """Every pick-feedback and swing-favorite row has `capture_id` EMPTY.

    `daily_recap_reader` passes `capture_id=""` for both sources, so a builder
    that hands the page an empty `ref_id` hands it a marker nothing can select
    and a page that cannot tell two of them apart.
    """
    import day_review_markers

    decisions = [
        _decision("AAA", verdict="not_today", stamp="2026-09-18T17:12:00+00:00",
                  capture_id="", source="pick_feedback", category="pick"),
        _decision("AAA", verdict="like", stamp="2026-09-18T17:32:00+00:00",
                  capture_id="", source="pick_feedback", category="pick"),
    ]

    markers = day_review_markers.symbol_markers("AAA", _m5_bars(), decisions=decisions)

    assert len(markers) == 2
    ids = [str(marker["ref_id"]) for marker in markers]
    assert all(ref.strip() for ref in ids), ids
    assert len(set(ids)) == 2, "two decisions share one id"


def test_a_verdict_the_plan_did_not_name_never_invents_a_kind():
    """`dislike`, `not_today` and `swing_favorite` are real and unlisted.

    They come back from `daily_recap_reader._decisions` on any real session.
    Whatever the builder decides to draw them as, the kind is one it declared -
    a chart that draws a glyph nobody can read is worse than one that does not.
    """
    import day_review_markers

    decisions = [
        _decision("AAA", verdict="dislike", stamp="2026-09-18T17:12:00+00:00",
                  capture_id="", source="pick_feedback", category="pick"),
        _decision("AAA", verdict="not_today", stamp="2026-09-18T17:17:00+00:00",
                  capture_id="", source="pick_feedback", category="pick"),
        _decision("AAA", verdict="swing_favorite", stamp="2026-09-18T17:22:00+00:00",
                  capture_id="", source="swing_favorites", category="swing"),
    ]

    for marker in day_review_markers.symbol_markers(
        "AAA", _m5_bars(), decisions=decisions
    ):
        assert marker["kind"] in day_review_markers.MARKER_KINDS, marker
        assert str(marker["label"]).strip(), marker
        assert str(marker["ref_id"]).strip(), marker


def test_a_name_is_matched_whatever_case_it_was_written_in():
    import day_review_markers

    bars = _m5_bars()
    decisions = [_decision("aaa", verdict="veto", stamp="2026-09-18T17:12:00+00:00")]

    assert len(day_review_markers.symbol_markers("AAA", bars, decisions=decisions)) == 1


def test_an_open_trade_draws_an_open_marker_and_no_close_marker():
    """An unclosed position is not given an invented exit."""
    import day_review_markers

    markers = day_review_markers.symbol_markers(
        "AAA",
        _m5_bars(),
        trades=[_trade("AAA", opened_at="2026-09-18T18:00:00+00:00", closed_at="",
                       trade_id="t-open")],
    )

    assert [marker["kind"] for marker in markers] == ["trade_open"]


# -- the D1 toggle -----------------------------------------------------------


def test_a_daily_bar_takes_the_marker_for_its_own_session():
    """The durable daily store's `dt` is a DATE STRING, and it still places."""
    import day_review_markers

    bars = _daily_bars()
    # 01:30 UTC on the 17th is 18:30 Pacific on the 16th - the evening of the
    # 16th session. A reader that compares UTC dates files it a day late.
    index = day_review_markers.bar_index_for(bars, "2026-09-17T01:30:00+00:00")

    assert index == 2
    assert bars[index]["dt"] == "2026-09-16"


def test_the_d1_toggle_keeps_every_marker_on_its_own_daily_bar():
    import day_review_markers

    bars = _daily_bars()
    entries = [
        _entry("monday", "2026-09-14T20:00:00+00:00", entry_id="e-mon"),
        _entry("wednesday", "2026-09-16T20:00:00+00:00", entry_id="e-wed"),
        _entry("friday", "2026-09-18T20:00:00+00:00", entry_id="e-fri"),
    ]

    placed = {
        marker["ref_id"]: marker["index"]
        for marker in day_review_markers.benchmark_markers(bars, entries=entries)
    }

    assert placed == {"e-mon": 0, "e-wed": 2, "e-fri": 4}


def test_a_note_older_than_the_first_daily_bar_yields_no_marker():
    import day_review_markers

    assert day_review_markers.bar_index_for(_daily_bars(), "2026-09-11T20:00:00+00:00") is None


# -- purity ------------------------------------------------------------------


def test_the_builder_is_pure_python_with_no_qt_and_no_store():
    """It runs on the worker and on the nightly slot; neither may import Qt."""
    import day_review_markers

    source = Path(day_review_markers.__file__).read_text(encoding="utf-8")
    for forbidden in ("PySide6", "pyqtgraph", "QtCore", "yfinance"):
        assert forbidden not in source, forbidden


def test_the_builder_never_mutates_the_bars_it_is_handed():
    import day_review_markers

    bars = _m5_bars()
    before = [dict(bar) for bar in bars]

    day_review_markers.benchmark_markers(
        bars, entries=[_entry("a thought", "2026-09-18T17:12:00+00:00")]
    )

    assert bars == before


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
