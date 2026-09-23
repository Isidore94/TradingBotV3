"""The "Working now" math - how today's M5 alerts have done since they fired.

Display only: nothing here reads or changes a detector, a score or an alert.
Every test below was run against a tree without `scripts/live_alert_results.py`
and seen to fail (ImportError) before the module was written.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import live_alert_results as lar  # noqa: E402

NY = ZoneInfo("America/New_York")


def _alert(symbol="NVDA", side="LONG", entry=100.0, stop=99.0, risk="", types="vwap"):
    feedback = {
        "symbol": symbol,
        "direction": side.lower(),
        "bounce_types": types,
        "entry_price": entry,
        "stop_price": stop,
        "risk_per_share": risk,
        "trade_date": "2026-09-22",
    }
    return SimpleNamespace(symbol=symbol, side=side, trigger=types, payload={"feedback": feedback})


def _at(hh, mm, ss=0):
    return datetime(2026, 9, 22, hh, mm, ss, tzinfo=NY)


def _bar(hh, mm, o, h, low, c, *, tz=None):
    dt = datetime(2026, 9, 22, hh, mm)
    if tz is not None:
        dt = dt.replace(tzinfo=NY).astimezone(tz)
    return {"dt": dt, "open": o, "high": h, "low": low, "close": c, "volume": 1000.0}


def _entry(**kwargs):
    received = kwargs.pop("received_at", _at(10, 35, 20))
    return lar.entry_from_alert(_alert(**kwargs), received)


# ------------------------------------------------------------------ entries
def test_entry_uses_risk_per_share_else_entry_stop_distance():
    entry = _entry(entry="100", stop="99.5", risk="0.4")
    assert entry["status"] == lar.OK
    assert entry["risk"] == 0.4
    entry = _entry(entry="100", stop="99.5", risk="")
    assert entry["risk"] == 0.5
    assert entry["side"] == "LONG" and entry["symbol"] == "NVDA"


def test_entry_missing_price_or_bad_risk_is_unknown():
    assert _entry(entry="", stop=99.0)["status"] == lar.UNKNOWN
    assert _entry(entry=100.0, stop=None)["status"] == lar.UNKNOWN
    assert _entry(entry=100.0, stop=100.0, risk="")["status"] == lar.UNKNOWN
    assert _entry(entry=100.0, stop=99.0, risk="-1")["status"] == lar.UNKNOWN


def test_non_trade_alert_is_not_an_entry():
    assert lar.entry_from_alert(_alert(side="WATCH"), _at(10, 35)) is None
    assert lar.entry_from_alert(SimpleNamespace(symbol="", side="LONG", payload={}), _at(10, 35)) is None


def test_received_at_must_be_aware_and_is_stamped_new_york():
    utc = datetime(2026, 9, 22, 14, 35, 20, tzinfo=timezone.utc)
    entry = lar.entry_from_alert(_alert(), utc)
    assert entry["received_at"].utcoffset() == timedelta(hours=-4)
    assert entry["received_at"].hour == 10


# ------------------------------------------------------------------ results
def test_long_winner_reports_r_now_best_and_worst():
    entry = _entry(entry=100.0, stop=99.0)
    bars = [
        _bar(10, 35, 100.0, 101.0, 99.7, 100.8),
        _bar(10, 40, 100.8, 102.1, 100.5, 101.4),
    ]
    result = lar.result_for(entry, bars, _at(10, 45, 1), naive_zone=NY)
    assert result["status"] == lar.OPEN
    assert result["r"] == 1.4
    assert result["best_r"] == 2.1
    assert result["worst_r"] == -0.3


def test_short_winner_is_mirrored():
    entry = _entry(side="SHORT", entry=50.0, stop=51.0)
    bars = [_bar(10, 35, 50.0, 50.4, 48.5, 49.0)]
    result = lar.result_for(entry, bars, _at(10, 40), naive_zone=NY)
    assert result["status"] == lar.OPEN
    assert result["r"] == 1.0
    assert result["best_r"] == 1.5
    assert result["worst_r"] == -0.4


def test_stop_first_touch_is_minus_one_and_later_bars_do_not_matter():
    entry = _entry(entry=100.0, stop=99.0)
    bars = [
        _bar(10, 35, 100.0, 100.5, 99.5, 100.2),
        _bar(10, 40, 100.2, 100.4, 98.9, 99.2),  # touches 99.0
        _bar(10, 45, 99.2, 104.0, 99.1, 103.9),  # a rip after the stop is not ours
    ]
    result = lar.result_for(entry, bars, _at(11, 0), naive_zone=NY)
    assert result["status"] == lar.STOPPED
    assert result["r"] == -1.0
    assert result["best_r"] == 0.5
    assert result["worst_r"] == -1.0


def test_short_stop_touch():
    entry = _entry(side="SHORT", entry=50.0, stop=51.0)
    bars = [_bar(10, 35, 50.0, 51.0, 49.8, 50.5)]
    result = lar.result_for(entry, bars, _at(10, 40), naive_zone=NY)
    assert result["status"] == lar.STOPPED and result["r"] == -1.0


def test_gap_through_the_stop_uses_the_open():
    entry = _entry(entry=100.0, stop=99.0)
    bars = [
        _bar(10, 35, 100.0, 100.3, 99.6, 99.8),
        _bar(10, 40, 98.5, 98.9, 98.0, 98.2),  # opens beyond the stop
    ]
    result = lar.result_for(entry, bars, _at(10, 45), naive_zone=NY)
    assert result["status"] == lar.STOPPED
    assert result["r"] == -1.5
    assert result["worst_r"] == -1.5


def test_incomplete_last_bar_is_ignored():
    entry = _entry(entry=100.0, stop=99.0)
    bars = [
        _bar(10, 35, 100.0, 100.5, 99.8, 100.3),
        _bar(10, 40, 100.3, 100.4, 98.0, 98.1),  # still printing at 10:44
    ]
    result = lar.result_for(entry, bars, _at(10, 44, 59), naive_zone=NY)
    assert result["status"] == lar.OPEN
    assert result["r"] == 0.3


def test_bars_before_the_alert_bucket_are_ignored():
    entry = _entry(entry=100.0, stop=99.0, received_at=_at(10, 37, 10))
    bars = [
        _bar(10, 30, 100.0, 100.1, 98.0, 99.9),  # before the alert: its stop touch is not ours
        _bar(10, 35, 100.0, 100.6, 99.6, 100.5),  # the bucket holding 10:37:10 counts
    ]
    result = lar.result_for(entry, bars, _at(10, 40), naive_zone=NY)
    assert result["status"] == lar.OPEN
    assert result["r"] == 0.5


def test_no_usable_bars_is_unknown_never_a_result():
    entry = _entry()
    assert lar.result_for(entry, [], _at(11, 0), naive_zone=NY)["status"] == lar.UNKNOWN
    assert lar.result_for(entry, None, _at(11, 0), naive_zone=NY)["status"] == lar.UNKNOWN
    only_before = [_bar(10, 30, 100.0, 101.0, 99.5, 100.5)]
    assert lar.result_for(entry, only_before, _at(11, 0), naive_zone=NY)["r"] is None
    broken = [{"dt": "junk", "open": 1}]
    assert lar.result_for(entry, broken, _at(11, 0), naive_zone=NY)["status"] == lar.UNKNOWN


def test_missing_prices_stay_unknown_even_with_bars():
    entry = _entry(entry="", stop=99.0)
    bars = [_bar(10, 35, 100.0, 101.0, 99.5, 100.5)]
    result = lar.result_for(entry, bars, _at(11, 0), naive_zone=NY)
    assert result["status"] == lar.UNKNOWN and result["r"] is None


def test_naive_bars_are_read_in_the_desk_zone_and_aware_bars_are_converted():
    """Pinned: the bot's cached bars carry NAIVE market-local stamps (IB
    formatDate=1, `_parse_ib_bar_datetime`). A naive stamp gets the desk's zone
    ATTACHED; an aware one is converted as the instant it is."""
    entry = _entry(entry=100.0, stop=99.0)
    now = _at(10, 45)
    naive = [_bar(10, 35, 100.0, 100.5, 99.8, 100.2), _bar(10, 40, 100.2, 100.6, 100.0, 100.4)]
    aware_utc = [
        _bar(10, 35, 100.0, 100.5, 99.8, 100.2, tz=timezone.utc),
        _bar(10, 40, 100.2, 100.6, 100.0, 100.4, tz=timezone.utc),
    ]
    assert lar.result_for(entry, naive, now, naive_zone=NY)["r"] == 0.4
    assert lar.result_for(entry, aware_utc, now, naive_zone=NY)["r"] == 0.4
    # a desk on Chicago time: the same naive wall clock is an hour later in NY,
    # so 10:35 and 10:40 Chicago have not printed yet at 10:45 New York.
    chicago = lar.result_for(entry, naive, now, naive_zone=ZoneInfo("America/Chicago"))
    assert chicago["status"] == lar.UNKNOWN
    # a naive `now` is read in the desk zone as well
    assert lar.result_for(entry, naive, datetime(2026, 9, 22, 10, 45), naive_zone=NY)["r"] == 0.4


# ------------------------------------------------------------------ book
def test_book_keeps_the_first_alert_per_symbol_side_per_day():
    book = lar.AlertBook()
    assert book.add(_alert(entry=100.0, stop=99.0), _at(10, 35)) is True
    assert book.add(_alert(entry=105.0, stop=104.0), _at(11, 0)) is False  # repeat
    assert book.add(_alert(side="SHORT", entry=100.0, stop=101.0), _at(11, 5)) is True
    assert book.add(_alert(symbol="AMD"), _at(11, 10)) is True
    assert book.add(_alert(side="WATCH"), _at(11, 15)) is False
    entries = book.entries()
    assert [(e["symbol"], e["side"]) for e in entries] == [
        ("NVDA", "LONG"),
        ("NVDA", "SHORT"),
        ("AMD", "LONG"),
    ]
    assert entries[0]["entry"] == 100.0
    # a new day starts a new book row for the same name
    assert book.add(_alert(), datetime(2026, 9, 23, 9, 45, tzinfo=NY)) is True
    book.clear()
    assert book.entries() == []


# ------------------------------------------------------------------ summary
def _result(grade, status, r):
    return {"grade": grade, "status": status, "r": r, "symbol": "X", "side": "LONG"}


def test_summary_groups_by_grade_best_first_and_keeps_unknowns_apart():
    results = [
        _result("C", lar.OPEN, -0.2),
        _result("A", lar.OPEN, 1.0),
        _result("New", lar.OPEN, 0.5),
        _result("A", lar.STOPPED, -1.0),
        _result("A", lar.OPEN, 3.0),
        _result("D", lar.OPEN, 0.1),
        _result("PROVEN", lar.UNKNOWN, None),
        _result("C", lar.UNKNOWN, None),
    ]
    summary = lar.summarize(results)
    assert [g["grade"] for g in summary["grades"]] == ["A", "C", "New", "D"]
    a = summary["grades"][0]
    assert a["count"] == 3 and a["open"] == 2 and a["stopped"] == 1
    assert a["avg_r"] == 1.0
    assert summary["unknown"] == 2
    assert lar.strip_text(summary) == (
        "Working now · A 3 (+1.0R) · C 1 (−0.2R) · NEW 1 (+0.5R) · D 1 (+0.1R) · 2 no data"
    )


def test_strip_text_when_nothing_has_fired():
    assert lar.strip_text(lar.summarize([])) == "Working now · no M5 alerts yet today"


def test_tooltip_line_shapes():
    received = _at(10, 35, 20)
    open_row = dict(_result("A", lar.OPEN, 1.4), symbol="NVDA", best_r=2.1, worst_r=-0.3,
                    received_at=received)
    stopped = dict(_result("B", lar.STOPPED, -1.0), symbol="AMD", side="SHORT", best_r=0.2,
                   worst_r=-1.0, received_at=received)
    unknown = dict(_result("New", lar.UNKNOWN, None), symbol="TSLA", received_at=received,
                   reason="no bars")
    assert lar.tooltip_line(open_row) == "NVDA LONG 10:35 [A] +1.4R now · best +2.1R · worst −0.3R"
    assert lar.tooltip_line(stopped) == "AMD SHORT 10:35 [B] STOPPED −1.0R · best +0.2R"
    assert lar.tooltip_line(unknown) == "TSLA LONG 10:35 [NEW] no data (no bars)"


def test_the_bots_cached_chart_bars_really_are_naive():
    """Pins the fact the naive-zone rule above rests on: every cached M5 series
    goes through `_bars_to_ib` (`_parse_ib_bar_datetime`), which yields NAIVE
    stamps, and `m5_chart_bars` hands them out unchanged. If this ever turns
    aware, `to_ny` converts instead - still correct, but look again."""
    from bounce_bot_lib import legacy

    bars = legacy._bars_to_ib(
        [{"time": "20260922  10:35:00", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 9}]
    )
    chart = legacy.BounceBot._m5_bars_as_chart_dicts(bars, 1)
    assert chart and chart[0]["dt"].tzinfo is None
    assert chart[0]["dt"] == datetime(2026, 9, 22, 10, 35)
