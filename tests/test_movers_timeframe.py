"""The Movers M30 and Daily boards (pure): completed bars, pop ranking, SPY anchors,
Dip-box gates, D1 anchored VWAP (trader 2026-09-29)."""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import movers_timeframe as mt  # noqa: E402

NY = ZoneInfo("America/New_York")
LA = ZoneInfo("America/Los_Angeles")
TODAY = date(2026, 9, 22)  # a Tuesday
NOON = datetime(2026, 9, 22, 12, 0, tzinfo=NY)


def _weekdays(count, end=TODAY):
    days, cursor = [], end
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return sorted(days)


def _m30(day, closes, *, volume=20_000.0, first_open=None):
    """M30 bars from 09:30 NY; open = previous close, range +/- 0.5."""
    out, previous = [], closes[0] if first_open is None else first_open
    for index, close in enumerate(closes):
        out.append({"dt": datetime(day.year, day.month, day.day, 9, 30, tzinfo=NY)
                    + timedelta(minutes=30 * index),
                    "open": previous, "high": max(previous, close) + 0.5,
                    "low": min(previous, close) - 0.5, "close": close, "volume": volume})
        previous = close
    return out


def _daily(days, closes, *, volume=2_000_000.0, volumes=None):
    out, previous = [], closes[0]
    for index, (day, close) in enumerate(zip(days, closes, strict=True)):
        out.append({"dt": datetime(day.year, day.month, day.day), "open": previous,
                    "high": max(previous, close) + 0.5, "low": min(previous, close) - 0.5,
                    "close": close,
                    "volume": volumes[index] if volumes is not None else volume})
        previous = close
    return out


def _m30_history(today_closes, *, price=100.0, sessions=20, volume=20_000.0,
                 today_volume=None):
    prior = []
    for day in _weekdays(sessions + 1)[:-1]:
        prior += _m30(day, [price] * 13, volume=volume, first_open=price)
    return prior + _m30(TODAY, today_closes, volume=today_volume or volume, first_open=price)


# ------------------------------------------------------------------ completed bars
def test_m30_keeps_completed_bars_only_and_reads_naive_stamps_as_market_local():
    bars = _m30(TODAY, [100.0] * 6)  # 09:30 .. 12:00 NY
    kept = mt.normalize_tf_bars("m30", bars, now=NOON)
    assert [b["dt"].strftime("%H:%M") for b in kept][-1] == "11:30"  # 12:00 is forming
    naive = [dict(b, dt=b["dt"].astimezone(LA).replace(tzinfo=None)) for b in bars]
    again = mt.normalize_tf_bars("m30", naive, now=NOON, local_tz=LA)
    assert [b["dt"] for b in again] == [b["dt"] for b in kept]
    assert all(b["dt"].tzinfo is not None for b in again)


def test_d1_drops_todays_forming_bar_until_the_close():
    days = _weekdays(3)
    bars = _daily(days, [100.0, 101.0, 102.0])
    assert [b["dt"].date() for b in mt.normalize_tf_bars("d1", bars, now=NOON)] == days[:2]
    after = datetime(2026, 9, 22, 16, 0, tzinfo=NY)
    assert [b["dt"].date() for b in mt.normalize_tf_bars("d1", bars, now=after)] == days
    aware = [dict(b, dt=datetime.combine(b["dt"].date(), datetime.min.time(), NY)) for b in bars]
    assert len(mt.normalize_tf_bars("d1", aware, now=after)) == 3


# ------------------------------------------------------------------ pop
def test_m30_pop_ranks_3_bar_moves_in_atrs_with_time_of_day_rvol():
    series = {
        "AAA": _m30_history([100.0, 100.0, 100.5, 101.0, 101.5], today_volume=40_000.0),
        "FLAT": _m30_history([100.0] * 5),
        "DOWN": _m30_history([100.0, 100.0, 99.5, 99.0, 98.5]),
        "OLDBARS": _m30_history([100.0, 100.5, 101.0]),  # stops at 10:30: stale
    }
    spy = _m30_history([400.0] * 5, price=400.0)
    board = mt.build_timeframe_board("m30", series, spy, now=NOON)
    assert board["tf"] == "m30" and board["session"] == TODAY.isoformat()
    assert board["as_of"].startswith("2026-09-22T11:30")
    longs = board["pop"]["long"]
    assert [r["symbol"] for r in longs] == ["AAA"]
    assert longs[0]["rvol"] == pytest.approx(2.0)
    assert longs[0]["move15_pct"] == pytest.approx(1.5)
    assert longs[0]["pop_score"] >= mt.POP_MIN_ATR_MOVE
    assert [r["symbol"] for r in board["pop"]["short"]] == ["DOWN"]
    assert board["offered"] == 4 and board["fresh"] == 3


def test_m30_unknown_rvol_is_neutral():
    short_history = _m30_history([100.0, 100.0, 100.5, 101.0, 101.5], sessions=2)
    board = mt.build_timeframe_board("m30", {"NEW": short_history},
                                     _m30_history([400.0] * 5, price=400.0), now=NOON)
    row = board["pop"]["long"][0]
    assert row["rvol"] is None and row["pop_score"] == pytest.approx(
        1.5 / row["atr"])


def test_d1_pop_uses_3_day_moves_and_3_vs_20_day_volume():
    days = _weekdays(30)
    after = datetime(2026, 9, 22, 16, 15, tzinfo=NY)
    closes = [100.0] * 27 + [101.0, 103.0, 105.0]
    volumes = [1_000_000.0] * 27 + [3_000_000.0] * 3
    bars = {"RUN": _daily(days, closes, volumes=volumes), "FLAT": _daily(days, [100.0] * 30)}
    spy = _daily(days, [400.0] * 30)
    board = mt.build_timeframe_board("d1", bars, spy, now=after)
    assert board["session"] == TODAY.isoformat()
    row = board["pop"]["long"][0]
    assert row["symbol"] == "RUN" and [r["symbol"] for r in board["pop"]["long"]] == ["RUN"]
    assert row["move15_pct"] == pytest.approx(5.0)
    assert row["rvol"] == pytest.approx(3.0)
    assert row["vs_spy15_pct"] == pytest.approx(5.0)
    assert row["day_pct"] == pytest.approx((105.0 / 103.0 - 1) * 100)


def test_trend_gate_and_quality_floor_apply_to_the_pop_lists():
    rising = _m30_history([100.0, 100.0, 100.5, 101.0, 101.5])
    series = {"GOOD": rising, "UNDER": rising, "GRAY": rising, "TINY": rising}
    days = _weekdays(201, end=TODAY - timedelta(days=1))
    daily = {"GOOD": _daily(days, [50.0] * 201), "UNDER": _daily(days, [150.0] * 201),
             "TINY": _daily(days, [50.0] * 201)}
    facts = {"TINY": {"market_cap_m": 200.0}}
    board = mt.build_timeframe_board("m30", series, _m30_history([400.0] * 5, price=400.0),
                                     now=NOON, daily_bars=daily, fundamentals=facts)
    names = [r["symbol"] for r in board["pop"]["long"]]
    assert names == ["GOOD", "GRAY"]  # UNDER is below its SMAs, TINY under $1B
    gray = board["pop"]["long"][1]
    assert gray["trend_long"] is None and gray["quality_ok"] is None
    good = board["pop"]["long"][0]
    assert good["trend_long"] is True and good["avg_volume_20d"] == pytest.approx(2e6)


# ------------------------------------------------------------------ anchors
def _spy_m30_rise_drop_rise():
    """9/18 rises to 401 (green run), 9/21 falls to 394 (red run), 9/22 rises to 403."""
    d2, d1 = _weekdays(3)[:2]
    raw = (_m30(d2, [395.0, 396.0, 397.0, 398.0, 399.0, 400.0] + [401.0] * 7, first_open=394.0)
           + _m30(d1, [400.0, 399.0, 398.0, 397.0, 396.0, 395.0] + [394.0] * 7, first_open=401.0)
           + _m30(TODAY, [395.0, 396.0, 397.0, 398.0, 399.0, 400.0, 401.0, 402.0, 403.0],
                  first_open=394.0))
    return mt.normalize_tf_bars("m30", raw, now=datetime(2026, 9, 22, 16, 0, tzinfo=NY))


def test_m30_long_anchor_is_the_top_the_last_big_dip_fell_from():
    # Trader 2026-09-30: M30 longs measure from the last major dip in SPY.
    anchors = mt.tf_anchors("m30", _spy_m30_rise_drop_rise())
    long = anchors["long"]
    assert long["kind"] == "swing"
    # Highest high from the 9/18 green run's start through the 9/21 red run's end;
    # 401.5 ties from 9/18 12:30 to 9/21 09:30 and the tie goes to the later bar.
    assert long["dt"] == "2026-09-21T09:30:00-04:00"
    assert long["price"] == 401.5


def test_m30_short_anchor_is_the_bottom_the_last_big_rip_rose_from():
    short = mt.tf_anchors("m30", _spy_m30_rise_drop_rise())["short"]
    assert short["kind"] == "swing"
    # Lowest low from the 9/21 red run's start through today's green run; later tie wins.
    assert short["dt"] == "2026-09-22T09:30:00-04:00"
    assert short["price"] == 393.5


def test_m30_no_run_falls_back_to_the_window_high_and_no_prior_run_uses_the_lookback():
    prior_day = _weekdays(2)[0]
    raw = (_m30(prior_day, [400.0] * 13, first_open=400.0)
           + _m30(TODAY, [401.0, 402.0, 403.0, 404.0, 405.0, 406.0, 407.0], first_open=400.0))
    spy = mt.normalize_tf_bars("m30", raw, now=datetime(2026, 9, 22, 16, 0, tzinfo=NY))
    anchors = mt.tf_anchors("m30", spy)
    # No red run: longs take the fallback window's high among bars 2+ bars old.
    assert anchors["long"]["kind"] == "window"
    assert anchors["long"]["dt"] == "2026-09-22T11:30:00-04:00"
    assert anchors["long"]["price"] == 405.5
    # Today's green run has no red run before it: its low runs from the lookback start.
    assert anchors["short"]["kind"] == "swing"
    assert anchors["short"]["dt"] == "2026-09-22T09:30:00-04:00"
    assert anchors["short"]["price"] == 399.5
    assert mt.tf_anchors("m30", []) == {"long": None, "short": None}


def test_d1_anchor_searches_60_sessions_and_falls_back_to_20():
    days = _weekdays(80)
    closes = [400.0] * 68 + [399.0, 398.0, 397.0, 396.0, 395.0, 394.0, 393.0] + \
        [394.0, 395.0, 396.0, 397.0, 398.0]
    spy = mt.normalize_tf_bars("d1", _daily(days, closes),
                               now=datetime(2026, 9, 22, 17, 0, tzinfo=NY))
    anchors = mt.tf_anchors("d1", spy)
    assert anchors["long"]["kind"] == "swing"
    assert anchors["long"]["date"] == days[75].isoformat() and anchors["long"]["time"] == ""
    # A red run older than 60 sessions is out of the lookback: the 20-day window's low.
    old = [400.0] * 5 + [399.0, 398.0, 397.0, 396.0, 395.0, 394.0, 393.0] + [394.0] * 68
    spy_old = mt.normalize_tf_bars("d1", _daily(days, old),
                                   now=datetime(2026, 9, 22, 17, 0, tzinfo=NY))
    long = mt.tf_anchors("d1", spy_old)["long"]
    assert long["kind"] == "window"
    assert long["date"] >= days[60].isoformat()


# ------------------------------------------------------------------ Dip boxes
def _dip_series():
    prior_day = _weekdays(2)[0]

    def name(prior_close, today):
        return (_m30(prior_day, [prior_close] * 13, first_open=prior_close)
                + _m30(TODAY, today, first_open=prior_close))

    return {
        "LEAD": name(100.0, [101.0, 102.0, 103.0, 104.0, 105.0]),
        "NODAILY": name(100.0, [101.0, 102.0, 103.0, 104.0, 105.0]),
        "INSIDE": name(100.0, [99.0, 99.2, 99.6, 100.0, 100.3]),
        "LAG": name(100.0, [99.0, 98.0, 97.0, 96.0, 95.0]),
    }


def test_m30_dip_boxes_use_the_m5_gates_and_unknown_smas_keep_names_off():
    prior_day = _weekdays(2)[0]
    spy_raw = (_m30(prior_day, [400.0] * 5 + [399.0, 398.0, 397.0, 396.0, 395.0, 394.0,
                                              393.0, 392.0], first_open=400.0)
               + _m30(TODAY, [393.0, 394.0, 395.0, 396.0, 397.0], first_open=392.0))
    days = _weekdays(201, end=TODAY - timedelta(days=1))
    daily = {"LEAD": _daily(days, [50.0] * 201), "INSIDE": _daily(days, [50.0] * 201),
             "LAG": _daily(days, [200.0] * 201)}
    board = mt.build_timeframe_board("m30", _dip_series(), spy_raw, now=NOON, daily_bars=daily)
    strong = [r["symbol"] for r in board["swing"]["long"]]
    weak = [r["symbol"] for r in board["swing"]["short"]]
    assert strong == ["LEAD"]  # NODAILY: SMAs unknown; INSIDE: below yesterday's high
    assert weak == ["LAG"]
    lead = board["swing"]["long"][0]
    assert lead["dip_score"] >= 0 and lead["last"] > lead["session_vwap"] > 0
    assert lead["last"] > lead["prev_high"]
    assert board["swing"]["short"][0]["dip_score"] < 0
    assert board["swing_anchor"]["long"]["kind"] == "swing"
    assert "_dt" not in board["swing_anchor"]["long"]


def test_d1_dip_strong_needs_the_spy_anchored_vwap_and_the_smas():
    days = _weekdays(220)
    after = datetime(2026, 9, 22, 16, 30, tzinfo=NY)
    spy_closes = [400.0] * 208 + [399.0, 398.0, 397.0, 396.0, 395.0, 394.0, 393.0] + \
        [394.0, 395.0, 396.0, 397.0, 398.0]
    lead = [100.0] * 214 + [101.0, 103.0, 105.0, 107.0, 109.0, 111.0]
    # Ran, then fell back under its anchored VWAP and its 50 SMA: weak, never strong.
    fade = [100.0] * 214 + [101.0, 103.0, 105.0, 107.0, 99.0, 98.0]
    lag = [200.0] * 150 + [100.0] * 64 + [99.0, 97.0, 95.0, 93.0, 91.0, 89.0]
    bars = {"LEAD": _daily(days, lead), "FADE": _daily(days, fade), "LAG": _daily(days, lag),
            "SHORTHIST": _daily(days[-30:], lead[-30:])}
    board = mt.build_timeframe_board("d1", bars, _daily(days, spy_closes), now=after)
    strong = board["swing"]["long"]
    assert [r["symbol"] for r in strong] == ["LEAD"]  # SHORTHIST: SMAs unknown
    assert strong[0]["avwap"] is not None and strong[0]["last"] > strong[0]["avwap"]
    assert [r["symbol"] for r in board["swing"]["short"]] == ["LAG", "FADE"]
    assert all(r["last"] < r["avwap"] for r in board["swing"]["short"])


def test_anchored_vwap_starts_at_the_anchor_bar():
    days = _weekdays(4)
    bars = mt.normalize_tf_bars("d1", _daily(days, [10.0, 20.0, 30.0, 40.0]),
                                now=datetime(2026, 9, 22, 17, 0, tzinfo=NY))
    value = mt.anchored_vwap(bars, bars[2]["dt"])
    # OHLC/4 of the last two bars, equal volume.
    assert value == pytest.approx(((20 + 30.5 + 19.5 + 30) / 4 + (30 + 40.5 + 29.5 + 40) / 4) / 2)
    assert mt.anchored_vwap(bars, datetime(2020, 1, 1, tzinfo=NY)) is None


def test_listed_symbols_and_bad_timeframe():
    board = {"pop": {"long": [{"symbol": "a"}], "short": [{"symbol": "B"}]},
             "swing": {"long": [{"symbol": "A"}], "short": []}}
    assert mt.listed_symbols(board) == ["A", "B"]
    with pytest.raises(ValueError):
        mt.build_timeframe_board("h1", {}, [], now=NOON)


def test_d1_entry_gate_needs_a_known_anchored_vwap_on_the_right_side():
    assert mt.d1_dip_ok(110.0, 100.0, "long", True, None) is True
    assert mt.d1_dip_ok(95.0, 100.0, "long", True, None) is False  # SMAs fine, under AVWAP
    assert mt.d1_dip_ok(110.0, None, "long", True, None) is False  # unknown AVWAP
    assert mt.d1_dip_ok(110.0, 100.0, "long", None, None) is False  # unknown SMA
    assert mt.d1_dip_ok(90.0, 100.0, "short", False, True) is True
    assert mt.d1_dip_ok(90.0, 100.0, "short", False, None) is False
