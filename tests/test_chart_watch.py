import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from chart_watch import (  # noqa: E402
    WATCH_KINDS,
    arm_chart_watch,
    evaluate_chart_watch,
    watch_is_stale,
)

DAY = datetime(2026, 7, 24)


def _bar(hour, minute, *, o=100.0, h=100.0, low=100.0, c=100.0, v=1000.0, day=DAY):
    return {
        "dt": day.replace(hour=hour, minute=minute),
        "open": float(o),
        "high": float(h),
        "low": float(low),
        "close": float(c),
        "volume": float(v),
    }


def test_arm_baseline_uses_todays_bars_only_and_includes_forming_bar():
    yesterday = DAY - timedelta(days=1)
    bars = [
        _bar(15, 55, h=200.0, low=90.0, day=yesterday),  # prior session ignored
        _bar(9, 30, h=110.0, low=99.0),
        _bar(9, 35, h=108.0, low=101.0),  # forming at arm time: still counts
    ]
    now = DAY.replace(hour=9, minute=37)
    hod = arm_chart_watch("new_hod", "nvda", "LONG", bars, now=now)
    lod = arm_chart_watch("new_lod", "NVDA", "SHORT", bars, now=now)
    vwap = arm_chart_watch("vwap_bounce", "NVDA", "banana", bars, now=now)

    assert hod.symbol == "NVDA" and hod.baseline == 110.0 and hod.side == "LONG"
    assert lod.baseline == 99.0
    # VWAP bounce has no fixed level; unknown side falls back to WATCH.
    assert vwap.baseline is None and vwap.side == "WATCH"

    with pytest.raises(ValueError):
        arm_chart_watch("teleport", "NVDA", "LONG", bars, now=now)


def test_new_hod_triggers_only_on_completed_post_arm_break():
    bars = [
        _bar(9, 30, h=110.0, low=99.0),
        _bar(9, 35, h=108.0, low=101.0),
    ]
    armed = arm_chart_watch("new_hod", "NVDA", "LONG", bars, now=DAY.replace(hour=9, minute=41))

    # A forming bar above the armed high is preview only - no trigger.
    bars.append(_bar(9, 45, h=111.0, low=104.0, c=110.8))
    assert evaluate_chart_watch(armed, bars, now=DAY.replace(hour=9, minute=49)) is None

    # The same bar completed (09:45 + 5min <= 09:50) fires exactly once.
    hit = evaluate_chart_watch(armed, bars, now=DAY.replace(hour=9, minute=50))
    assert hit is not None
    assert hit.price == 111.0
    assert "New HOD 111.00 > armed day high 110.00" in hit.message

    # A bar that only matches the armed high is not a NEW high.
    equal_only = bars[:2] + [_bar(9, 45, h=110.0, low=104.0)]
    assert evaluate_chart_watch(armed, equal_only, now=DAY.replace(hour=9, minute=50)) is None


def test_extreme_watch_without_cached_bars_builds_baseline_before_firing():
    # Armed before the bot cached any bars for the symbol: pre-arm completed
    # bars tighten the reference, and the first post-arm bar cannot trivially
    # "break" a missing baseline.
    armed = arm_chart_watch("new_lod", "NVDA", "SHORT", [], now=DAY.replace(hour=9, minute=40))
    bars = [
        _bar(9, 30, h=110.0, low=99.0),  # completed pre-arm: becomes baseline
        _bar(9, 45, h=105.0, low=98.5),  # post-arm break of 99.0
    ]
    hit = evaluate_chart_watch(armed, bars, now=DAY.replace(hour=9, minute=55))
    assert hit is not None
    assert hit.price == 98.5
    assert "New LOD 98.50 < armed day low 99.00" in hit.message

    fresh_only = [_bar(9, 45, h=105.0, low=98.5)]
    assert evaluate_chart_watch(armed, fresh_only, now=DAY.replace(hour=9, minute=55)) is None


def test_vwap_bounce_touch_and_reclaim_by_side():
    flat = [
        _bar(9, 30),
        _bar(9, 35),
        _bar(9, 40),
    ]
    # VWAP sits at ~100 after three flat bars; the 09:45 bar tags it from
    # above and closes back over it - the long touch-and-reclaim.
    bounce = _bar(9, 45, o=100.2, h=100.5, low=99.8, c=100.3)
    bars = flat + [bounce]
    now = DAY.replace(hour=9, minute=55)

    long_watch = arm_chart_watch("vwap_bounce", "NVDA", "LONG", flat, now=DAY.replace(hour=9, minute=42))
    hit = evaluate_chart_watch(long_watch, bars, now=now)
    assert hit is not None
    assert hit.price == 100.3
    assert "VWAP bounce (long)" in hit.message

    # The same tape is NOT a short bounce (it closed above VWAP).
    short_watch = arm_chart_watch("vwap_bounce", "NVDA", "SHORT", flat, now=DAY.replace(hour=9, minute=42))
    assert evaluate_chart_watch(short_watch, bars, now=now) is None

    # WATCH side accepts either direction.
    watch_watch = arm_chart_watch("vwap_bounce", "NVDA", "WATCH", flat, now=DAY.replace(hour=9, minute=42))
    assert evaluate_chart_watch(watch_watch, bars, now=now) is not None

    # A bar that never touches VWAP does not fire.
    above = flat + [_bar(9, 45, o=100.6, h=100.9, low=100.5, c=100.8)]
    assert evaluate_chart_watch(long_watch, above, now=now) is None

    # Short flavor: tag VWAP from below, close back under it.
    short_bounce = flat + [_bar(9, 45, o=99.8, h=100.2, low=99.6, c=99.7)]
    hit = evaluate_chart_watch(short_watch, short_bounce, now=now)
    assert hit is not None
    assert "VWAP bounce (short)" in hit.message


def test_pre_arm_bars_never_trigger():
    bars = [
        _bar(9, 30, h=110.0, low=99.0),
        _bar(9, 35, h=112.0, low=101.0),  # day high already 112 before arming
    ]
    armed = arm_chart_watch("new_hod", "NVDA", "LONG", bars, now=DAY.replace(hour=9, minute=41))
    # Nothing new after arming: the pre-arm 112 print must not fire.
    assert evaluate_chart_watch(armed, bars, now=DAY.replace(hour=9, minute=55)) is None


def test_watch_is_stale_next_session():
    armed = arm_chart_watch("new_hod", "NVDA", "LONG", [], now=DAY.replace(hour=15, minute=55))
    assert not watch_is_stale(armed, now=DAY.replace(hour=16, minute=5))
    assert watch_is_stale(armed, now=DAY + timedelta(days=1, hours=9))


def test_watch_kind_labels_cover_all_buttons():
    assert list(WATCH_KINDS) == ["new_hod", "new_lod", "vwap_bounce"]
    assert WATCH_KINDS["new_hod"] == "New HOD"
    assert WATCH_KINDS["new_lod"] == "New LOD"
    assert WATCH_KINDS["vwap_bounce"] == "VWAP bounce"
