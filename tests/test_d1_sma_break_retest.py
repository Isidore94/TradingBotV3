"""`sma_break_retest` (sma_break_retest_v1) and incoming-trendline arms.

Golden fixtures for the SMA break + 15EMA retest D1 event (trader's word
2026-09-24): a completed D1 close through SMA50/100/200 in the armed side's
direction, then - in a LATER session, within 10 sessions - a bar that tags
the D1 15EMA and closes back on the break side. A close back through the SMA
before the retest resets it to waiting-for-break.

Also: a trendline_break_retest armed on an INCOMING scan line (type H- / L+,
no break_date yet) evaluates like one armed on a broken line.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import chart_watch  # noqa: E402
from chart_watch import (  # noqa: E402
    D1_EVENT_KINDS,
    D1_PULLBACK_KINDS,
    D1EventWatch,
    d1_event_levels,
    d1_event_watch_from_dict,
    d1_event_watch_to_dict,
    evaluate_d1_event_watch,
)

START = datetime(2026, 1, 1)


def _bar(index, *, h, low, c):
    return {
        "dt": START + timedelta(days=index),
        "open": float(c),
        "high": float(h),
        "low": float(low),
        "close": float(c),
        "volume": 1000.0,
    }


def _base():
    """57 sessions at 100 then 3 at 94: SMA50 ≈ 99.64, prev close 94 below it."""
    bars = [_bar(i, h=101.0, low=99.0, c=100.0) for i in range(57)]
    bars += [_bar(57 + i, h=95.0, low=93.0, c=94.0) for i in range(3)]
    return bars  # indexes 0..59


ARM_INDEX = 59  # armed on the last base session


def _armed(side="LONG", index=ARM_INDEX):
    return D1EventWatch(
        symbol="NVDA",
        kind="sma_break_retest",
        armed_at=(START + timedelta(days=index)).replace(hour=15),
        side=side,
    )


def _levels(bars, index):
    return d1_event_levels(bars, session=(START + timedelta(days=index)).date())


def _break_up(index):
    return _bar(index, h=101.5, low=94.5, c=101.0)


def _retest_long(bars, index):
    """Tags the D1 15EMA of session `index` and closes back above it and the SMA."""
    levels = _levels(bars, index)
    ema = levels["ema15"]
    close = max(ema, levels["sma50"]) + 1.0
    return _bar(index, h=close + 0.5, low=ema - 0.1, c=close)


def _quiet_long(bars, index):
    """Holds well above the 15EMA: no tag, no fail-back."""
    levels = _levels(bars, index)
    top = max(levels["ema15"], levels["sma50"]) + 3.0
    return _bar(index, h=top + 1.0, low=top - 0.5, c=top + 0.5)


def _now_after(bars):
    return bars[-1]["dt"] + timedelta(days=1, hours=9)


def test_kind_is_labelled_trader_only_and_carries_its_side():
    assert D1_EVENT_KINDS["sma_break_retest"] == "SMA break + 15EMA retest"
    assert "sma_break_retest" not in D1_PULLBACK_KINDS
    assert "sma_break_retest" in chart_watch.D1_TRADER_ONLY_KINDS
    assert chart_watch.SMA_BREAK_RETEST_RULE_VERSION == "sma_break_retest_v1"
    assert chart_watch.SMA_BREAK_RETEST_MAX_SESSIONS == 10
    back = d1_event_watch_from_dict(d1_event_watch_to_dict(_armed("SHORT")))
    assert back.kind == "sma_break_retest" and back.side == "SHORT"


def test_break_then_later_15ema_retest_fires_long():
    bars = _base() + [_break_up(60)]
    bars.append(_retest_long(bars, 61))
    hit = evaluate_d1_event_watch(_armed(), [], bars, now=_now_after(bars))
    assert hit is not None and hit.resolved_side == "long"
    assert "SMA50" in hit.message and "15EMA" in hit.message
    assert "D1 bar" in hit.message
    assert hit.details["rule_version"] == "sma_break_retest_v1"
    assert hit.details["break_date"] == bars[60]["dt"].date().isoformat()
    assert hit.details["retest_date"] == bars[61]["dt"].date().isoformat()


def test_same_session_tag_on_the_break_bar_never_fires():
    bars = _base()
    levels = _levels(bars, 60)
    # Closes through the SMA AND tags the 15EMA on the same bar.
    bars.append(_bar(60, h=101.5, low=levels["ema15"] - 0.2, c=101.0))
    assert evaluate_d1_event_watch(_armed(), [], bars, now=_now_after(bars)) is None


def test_retest_must_come_within_ten_sessions():
    base = _base() + [_break_up(60)]
    inside = list(base)
    for index in range(61, 70):  # nine quiet sessions
        inside.append(_quiet_long(inside, index))
    inside.append(_retest_long(inside, 70))  # 10th session after the break
    assert evaluate_d1_event_watch(_armed(), [], inside, now=_now_after(inside)) is not None

    late = list(base)
    for index in range(61, 71):  # ten quiet sessions
        late.append(_quiet_long(late, index))
    late.append(_retest_long(late, 71))  # 11th session after the break
    assert evaluate_d1_event_watch(_armed(), [], late, now=_now_after(late)) is None


def test_a_close_back_through_the_sma_resets_to_waiting_for_break():
    bars = _base() + [_break_up(60)]
    bars.append(_bar(61, h=100.0, low=95.0, c=96.0))  # fails back under SMA50
    # Tags the 15EMA and closes back over it and over the SMA: without the
    # reset this would be the retest; with it, it is a NEW break session.
    bars.append(_retest_long(bars, 62))
    assert evaluate_d1_event_watch(_armed(), [], bars, now=_now_after(bars)) is None
    bars.append(_retest_long(bars, 63))
    hit = evaluate_d1_event_watch(_armed(), [], bars, now=_now_after(bars))
    assert hit is not None
    assert hit.details["break_date"] == bars[62]["dt"].date().isoformat()


def test_missing_history_is_no_fire():
    bars = _base()[-30:]  # no SMA50 on 30 sessions
    first = bars[0]["dt"]
    tail = [_break_up(60), _bar(61, h=102.0, low=90.0, c=101.5)]
    watch = D1EventWatch(
        symbol="NVDA", kind="sma_break_retest", armed_at=first.replace(hour=15), side="LONG"
    )
    assert evaluate_d1_event_watch(watch, [], bars + tail, now=_now_after(tail)) is None
    assert evaluate_d1_event_watch(_armed(), [], [], now=_now_after(tail)) is None


def test_a_break_on_or_before_the_arm_day_does_not_count():
    bars = _base() + [_break_up(60)]
    bars.append(_retest_long(bars, 61))
    armed_on_break_day = _armed(index=60)
    assert evaluate_d1_event_watch(armed_on_break_day, [], bars, now=_now_after(bars)) is None


def test_wrong_side_or_no_side_never_fires():
    bars = _base() + [_break_up(60)]
    bars.append(_retest_long(bars, 61))
    assert evaluate_d1_event_watch(_armed("SHORT"), [], bars, now=_now_after(bars)) is None
    assert evaluate_d1_event_watch(_armed(""), [], bars, now=_now_after(bars)) is None


def test_short_mirror_fires():
    bars = [_bar(i, h=101.0, low=99.0, c=100.0) for i in range(57)]
    bars += [_bar(57 + i, h=107.0, low=105.0, c=106.0) for i in range(3)]
    bars.append(_bar(60, h=105.5, low=98.5, c=99.0))  # closes down through SMA50
    levels = _levels(bars, 61)
    close = min(levels["ema15"], levels["sma50"]) - 1.0
    bars.append(_bar(61, h=levels["ema15"] + 0.1, low=close - 0.5, c=close))
    hit = evaluate_d1_event_watch(_armed("SHORT"), [], bars, now=_now_after(bars))
    assert hit is not None and hit.resolved_side == "short"


def test_intraday_m5_retest_the_session_after_a_d1_break():
    bars = _base() + [_break_up(60)]
    today = START + timedelta(days=61)
    levels = _levels(bars, 61)
    ema = levels["ema15"]
    close = max(ema, levels["sma50"]) + 1.0
    m5 = {
        "dt": today.replace(hour=10, minute=5),
        "open": close,
        "high": close + 0.3,
        "low": ema - 0.1,
        "close": close,
        "volume": 1000.0,
    }
    hit = evaluate_d1_event_watch(
        _armed(), [m5], bars, now=today.replace(hour=10, minute=15)
    )
    assert hit is not None and "M5 bar" in hit.message
    # Forming M5 bar: preview only.
    assert evaluate_d1_event_watch(_armed(), [m5], bars, now=today.replace(hour=10, minute=8)) is None


# ------------------------------------------------ incoming trendline retest
def _incoming_candidate(lookback_end):
    return {
        "line_id": "d1_trendline:H-:2026-01-05_2026-01-20",
        "type": "H-",
        "start_date": "2026-01-05",
        "end_date": "2026-01-20",
        "start_price": 104.0,
        "end_price": 102.0,
        "current_line_price": 100.0,
        "slope_log_per_bar": 0.0,
        "lookback_start": "2026-01-01",
        "lookback_end": lookback_end,
    }


def test_an_incoming_line_without_a_break_date_is_frozen_and_can_fire():
    bars = [_bar(i, h=99.0, low=97.0, c=98.0) for i in range(30)]
    candidate = _incoming_candidate(bars[-1]["dt"].date().isoformat())
    assert chart_watch._trendline_candidate_is_frozen(candidate)
    # A BREAK-type record still needs its break_date.
    broken = dict(candidate, type="H-break", line_id="d1_trendline:H-break:2026-01-05_2026-01-20")
    assert not chart_watch._trendline_candidate_is_frozen(broken)

    bars.append(_bar(30, h=101.5, low=98.0, c=101.0))  # break up through 100
    bars.append(_bar(31, h=101.2, low=100.1, c=100.6))  # retest, holds
    bars.append(_bar(32, h=102.5, low=100.8, c=102.0))  # confirm
    watch = D1EventWatch(
        symbol="NVDA",
        kind="trendline_break_retest",
        armed_at=bars[29]["dt"].replace(hour=15),
        side="LONG",
        trendline_candidate=candidate,
        trendline_knowledge_at=datetime(2026, 1, 30, 8, 0, tzinfo=timezone.utc),
    )
    hit = evaluate_d1_event_watch(watch, [], bars, now=_now_after(bars))
    assert hit is not None and hit.resolved_side == "long"
    assert "Trendline break + retest" in hit.message


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_incoming_line_type_matches_side(side):
    assert chart_watch.incoming_trendline_type(side) == ("H-" if side == "LONG" else "L+")
