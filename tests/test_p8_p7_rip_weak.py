"""P8 P7: the Movers board's rally state (Rip-strong / Rip-weak) and the Pop outcome log.

Synthetic bars only. SPY bars are naive Los Angeles wall time (06:30 LA = 09:30 NY),
the shape `bot.m5_chart_bars` hands out.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import movers_scan as ms  # noqa: E402

LA = ZoneInfo("America/Los_Angeles")
NY = ZoneInfo("America/New_York")
PRIOR = date(2026, 9, 21)
TODAY = date(2026, 9, 22)
FLAT_BASELINE = {offset: 100_000.0 for offset in range(78)}


def _bars(day, closes, *, volume=100_000.0, first_open=None, wick=0.05):
    out = []
    previous = closes[0] if first_open is None else first_open
    start = datetime(day.year, day.month, day.day, 6, 30)
    for index, close in enumerate(closes):
        out.append({
            "dt": start + timedelta(minutes=5 * index), "open": previous,
            "high": max(previous, close) + wick, "low": min(previous, close) - wick,
            "close": close, "volume": volume,
        })
        previous = close
    return out


def _series(closes, *, prior_close=100.0):
    return _bars(PRIOR, [prior_close] * 78) + _bars(TODAY, closes, first_open=prior_close)


def _today(bars):
    return [b for b in bars if b["dt"].date() == TODAY]


def _now(n):
    return datetime(TODAY.year, TODAY.month, TODAY.day, 6, 30, tzinfo=LA) + timedelta(minutes=5 * n)


def _state(bars, n):
    return ms.market_state(ms.normalize_bars(bars, now=_now(n), local_tz=LA))


def _plain_rally():
    """Up day, no pullback: the session low is bar 2 (closed above VWAP), then a steady climb."""
    closes = [401.0, 401.2, 401.0, 401.5, 402.0, 402.5, 403.0]
    bars = _series(closes, prior_close=400.0)
    today = _today(bars)
    today[0]["low"] = 400.0
    today[2]["low"] = 399.8  # the swing low, after the first two bars
    return bars, len(closes)


def _pullback_then_rally(bottom=402.0, now_close=403.5):
    """Rally to a 10:15 high, pull back to `bottom` at bar 12, then rally to `now_close`."""
    closes = [400.0, 400.0] + [405.0] * 8 + [404.0, 403.0, bottom, now_close - 0.5, now_close]
    bars = _series(closes, prior_close=400.0)
    today = _today(bars)
    today[0]["volume"] = today[1]["volume"] = 1_000_000.0
    today[9]["high"] = 405.6  # the session high, above VWAP
    return bars, len(closes)


# ------------------------------------------------------------------ market state
def test_plain_rally_lights_rally_from_the_session_low():
    bars, n = _plain_rally()
    state = _state(bars, n)
    assert state.rally is True
    assert state.pullback is False and state.bounce is False
    assert state.state == "up_day"
    assert state.extreme_time == "09:40"  # bar 2
    assert state.extreme_price == pytest.approx(399.8)
    assert state.start_dt is not None and state.start_dt.strftime("%H:%M") == "09:40"
    assert state.spy_from_extreme_pct >= ms.PULLBACK_MIN_PCT
    assert state.to_dict()["rally"] is True


def test_rally_below_the_threshold_lights_nothing():
    closes = [401.0, 401.2, 401.0, 401.1, 401.0, 401.0]
    bars = _series(closes, prior_close=400.0)
    _today(bars)[0]["low"] = 400.9
    _today(bars)[2]["low"] = 400.5  # last 401.0 is only +0.12% off it
    state = _state(bars, len(closes))
    assert state.rally is False and state.start_dt is None


def test_a_low_in_the_first_two_bars_is_not_a_rally_start():
    # Same rule as the pullback's high: the first two bars' extremes are opening noise.
    closes = [401.0, 401.2, 401.3, 401.5, 402.0, 402.5, 403.0]
    bars = _series(closes, prior_close=400.0)
    _today(bars)[0]["low"] = 399.0
    state = _state(bars, len(closes))
    assert state.rally is False and state.start_dt is None


def test_rally_after_a_pullback_starts_at_the_pullback_low_and_the_later_turn_wins():
    bars, n = _pullback_then_rally(bottom=402.0, now_close=403.5)
    state = _state(bars, n)
    # Both qualify: 403.5 is > 0.30% under the 405.6 high AND > 0.30% over the 401.95 low.
    assert (403.5 / 405.6 - 1) * 100 <= -ms.PULLBACK_MIN_PCT
    assert state.rally is True and state.pullback is False
    assert state.extreme_time == "10:30"  # bar 12, the pullback's low
    assert state.extreme_price == pytest.approx(401.95)


def test_rally_ends_when_a_new_down_leg_makes_the_last_bar_the_swing_low():
    closes = [401.0, 401.2, 401.0, 401.5, 402.0, 402.5, 403.0, 401.6]
    bars = _series(closes, prior_close=400.0)
    _today(bars)[0]["low"] = 400.0
    _today(bars)[2]["low"] = 399.8
    state = _state(bars, len(closes))
    assert state.rally is False


def test_a_bounce_off_the_same_low_stays_a_bounce():
    closes = [400.0, 400.0] + [395.0] * 8
    closes += [395.0 + 395.0 * 0.004 * k / 3 for k in (1, 2, 3)]
    bars = _series(closes, prior_close=400.0)
    today = _today(bars)
    today[0]["volume"] = today[1]["volume"] = 1_000_000.0
    today[9]["low"] = 394.4
    state = _state(bars, len(closes))
    assert state.bounce is True and state.rally is False


def test_a_pullback_still_live_after_a_small_bounce_is_not_a_rally():
    bars, n = _pullback_then_rally(bottom=402.0, now_close=402.8)  # +0.21% off the low
    state = _state(bars, n)
    assert state.pullback is True and state.rally is False


# ------------------------------------------------------------------ rip lists
def _rip_board(extra=None):
    spy, n = _pullback_then_rally(bottom=402.0, now_close=403.5)
    base = [100.0] * 13  # through the rally start (bar 12)
    series = {
        "LEAD": _series(base + [100.8, 101.5]),  # beats SPY's +0.37% since the start
        "LAG": _series(base + [100.0, 100.0]),  # flat while SPY rips
        "SINK": _series(base + [99.5, 99.0]),  # falling into the rip
    }
    series.update(extra or {})
    return ms.build_movers_board(
        series, spy, now=_now(n), baselines={s: FLAT_BASELINE for s in series}, local_tz=LA,
    )


def test_rally_fills_rip_lists_by_excess_vs_spy_and_leaves_dip_empty():
    board = _rip_board()
    assert board["state"]["rally"] is True
    assert board["dip"] == {"long": [], "short": []}
    assert [r["symbol"] for r in board["rip"]["long"]] == ["LEAD"]
    assert [r["symbol"] for r in board["rip"]["short"]] == ["SINK", "LAG"]
    assert all(r["dip_score"] < 0 for r in board["rip"]["short"])
    lead = board["rip"]["long"][0]
    assert lead["since_start_pct"] == pytest.approx(1.5)


def test_rip_lists_keep_the_floors_and_top_n():
    cheap = _series([1.0] * 13 + [0.9, 0.8], prior_close=1.0)
    board = _rip_board({"CHEAP": cheap})
    assert "CHEAP" not in [r["symbol"] for r in board["rip"]["short"]]
    many = {f"W{i:02d}": _series([100.0] * 13 + [99.9 - i * 0.01, 99.8 - i * 0.01])
            for i in range(ms.MOVERS_TOP_N + 5)}
    board = _rip_board(many)
    assert len(board["rip"]["short"]) == ms.MOVERS_TOP_N


def test_rip_lists_are_empty_outside_a_rally():
    spy, n = _pullback_then_rally(bottom=402.0, now_close=402.8)
    board = ms.build_movers_board({"X": _series([100.0] * 15)}, spy, now=_now(n), local_tz=LA)
    assert board["rip"] == {"long": [], "short": []}


def test_rip_lists_get_persistence_and_group_tags():
    board = _rip_board()
    memory = ms.apply_persistence(board, {}, session=TODAY)
    top = board["rip"]["short"][0]
    assert top["streak"] == 1 and top["rank_change"] is None
    assert "rip:short" in memory["lists"]
    ms.apply_group_tags(board, {"SINK": "Semiconductors", "LAG": "Semiconductors"})
    assert "rip" in board["groups"]
