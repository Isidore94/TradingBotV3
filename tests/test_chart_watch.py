import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from chart_watch import (  # noqa: E402
    D1EventWatch,
    D1LevelWatch,
    D1_EVENT_KINDS,
    D1_LEVEL_KINDS,
    WATCH_KINDS,
    arm_chart_watch,
    d1_event_levels,
    evaluate_chart_watch,
    evaluate_d1_event_watch,
    evaluate_d1_level_watch,
    load_chart_watches,
    load_d1_event_watches,
    load_d1_level_watches,
    save_chart_watches,
    save_d1_event_watches,
    save_d1_level_watches,
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
    assert list(WATCH_KINDS) == ["new_hod", "new_lod", "vwap_bounce", "band_bounce"]
    assert WATCH_KINDS["new_hod"] == "New HOD"
    assert WATCH_KINDS["new_lod"] == "New LOD"
    assert WATCH_KINDS["vwap_bounce"] == "VWAP bounce"
    assert WATCH_KINDS["band_bounce"] == "σ-band bounce"
    assert set(D1_LEVEL_KINDS) == {"d1_level_above", "d1_level_below"}


def test_band_bounce_touch_and_reclaim_by_side():
    # Two volume bars build VWAP 102 with ±1σ ≈ 102 ± 1.414; the trigger bar
    # carries zero volume so the band at its index is exactly that value.
    base = [
        _bar(9, 30),  # tp 100
        _bar(9, 35, o=100.0, h=108.0, low=100.0, c=108.0),  # tp 104
    ]
    upper = 102.0 + (4000.0 / 2000.0) ** 0.5  # ≈ 103.414
    lower = 102.0 - (4000.0 / 2000.0) ** 0.5  # ≈ 100.586
    now = DAY.replace(hour=9, minute=50)
    armed = DAY.replace(hour=9, minute=41)

    long_hit = base + [
        _bar(9, 45, o=upper + 0.6, h=upper + 0.9, low=upper - 0.2, c=upper + 0.7, v=0.0)
    ]
    long_watch = arm_chart_watch("band_bounce", "NVDA", "LONG", base, now=armed)
    hit = evaluate_chart_watch(long_watch, long_hit, now=now)
    assert hit is not None
    assert "σ-band bounce (long)" in hit.message
    assert hit.resolved_side == "long"

    # The same tape is not a short bounce (never tagged the LOWER band).
    short_watch = arm_chart_watch("band_bounce", "NVDA", "SHORT", base, now=armed)
    assert evaluate_chart_watch(short_watch, long_hit, now=now) is None

    short_hit = base + [
        _bar(9, 45, o=lower - 0.3, h=lower + 0.2, low=lower - 0.6, c=lower - 0.4, v=0.0)
    ]
    hit = evaluate_chart_watch(short_watch, short_hit, now=now)
    assert hit is not None
    assert "σ-band bounce (short)" in hit.message
    assert hit.resolved_side == "short"

    # A bar that stays inside the bands fires neither side.
    inside = base + [_bar(9, 45, o=102.0, h=102.5, low=101.5, c=102.2, v=0.0)]
    watch_watch = arm_chart_watch("band_bounce", "NVDA", "WATCH", base, now=armed)
    assert evaluate_chart_watch(watch_watch, inside, now=now) is None


def test_chart_watches_persist_for_the_same_day_only(tmp_path):
    path = tmp_path / "alert_chart_watches.json"
    bars = [_bar(9, 30, h=110.0, low=99.0)]
    armed = [
        arm_chart_watch("new_hod", "NVDA", "LONG", bars, now=DAY.replace(hour=9, minute=40)),
        arm_chart_watch("band_bounce", "TSLA", "SHORT", [], now=DAY.replace(hour=9, minute=41), source_text="ctx"),
    ]
    save_chart_watches(armed, path, market_date=DAY.date())

    restored = load_chart_watches(path, market_date=DAY.date())
    assert restored == armed  # frozen dataclasses compare by value

    # A new session starts clean.
    assert load_chart_watches(path, market_date=(DAY + timedelta(days=1)).date()) == []
    # Corrupt file degrades to empty.
    path.write_text("{not json", encoding="utf-8")
    assert load_chart_watches(path, market_date=DAY.date()) == []


def test_d1_level_watch_persistence_and_evaluation(tmp_path):
    path = tmp_path / "d1_level_watches.json"
    armed_at = DAY.replace(hour=14, minute=0)  # 2026-07-24 14:00
    above = D1LevelWatch(
        symbol="NVDA", direction="above", level=50.0, armed_at=armed_at, candle_date="2026-07-20"
    )
    below = D1LevelWatch(symbol="TSLA", direction="below", level=20.0, armed_at=armed_at)
    save_d1_level_watches([above, below], path)
    assert load_d1_level_watches(path) == [above, below]
    assert above.kind == "d1_level_above" and below.kind == "d1_level_below"

    def d1_bar(day_offset, high, low):
        return {
            "dt": DAY + timedelta(days=day_offset),
            "open": (high + low) / 2,
            "high": high,
            "low": low,
            "close": (high + low) / 2,
            "volume": 1000.0,
        }

    # The armed day's own D1 bar never triggers (it contains pre-arm prices);
    # a later completed session crossing the level does.
    later = DAY + timedelta(days=3)
    assert (
        evaluate_d1_level_watch(above, [], [d1_bar(0, 51.0, 45.0)], now=later) is None
    )
    hit = evaluate_d1_level_watch(
        above, [], [d1_bar(0, 51.0, 45.0), d1_bar(1, 50.4, 46.0)], now=later
    )
    assert hit is not None
    assert hit.price == 50.4
    assert "D1 level break above 50.00" in hit.message
    assert hit.resolved_side == "long"
    # Today's forming daily bar is preview only.
    assert (
        evaluate_d1_level_watch(above, [], [d1_bar(3, 55.0, 46.0)], now=later) is None
    )

    # Completed M5 evidence covers the armed day itself while scanned.
    m5 = [
        {
            "dt": armed_at.replace(hour=14, minute=30),
            "open": 49.5,
            "high": 50.2,
            "low": 49.4,
            "close": 50.1,
            "volume": 500.0,
        }
    ]
    hit = evaluate_d1_level_watch(above, m5, [], now=armed_at.replace(hour=14, minute=40))
    assert hit is not None and hit.price == 50.2
    # ...but not while the bar is still forming.
    assert (
        evaluate_d1_level_watch(above, m5, [], now=armed_at.replace(hour=14, minute=33))
        is None
    )

    # Short side: a later session probing under the level flags.
    hit = evaluate_d1_level_watch(
        below, [], [d1_bar(1, 22.0, 19.8)], now=later
    )
    assert hit is not None
    assert hit.resolved_side == "short"
    assert "D1 level break below 20.00" in hit.message


# ---------------------------------------------------------------------------
# Persistent D1 event watches: derived-level alerts (15EMA reject, new 5d/20d
# extreme, SMA break) whose references re-derive from the daily store.
# ---------------------------------------------------------------------------
def _daily(day_offset, *, high, low, close):
    return {
        "dt": DAY + timedelta(days=day_offset),
        "open": close,
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": 1000.0,
    }


def test_d1_event_kind_labels_cover_all_buttons():
    assert set(D1_EVENT_KINDS) == {
        "ema15_reject",
        "new_5d_high",
        "new_5d_low",
        "new_20d_high",
        "new_20d_low",
        "sma_break",
    }
    assert all(label for label in D1_EVENT_KINDS.values())
    # Kind namespaces never collide - the feed badge resolves across all three.
    assert not set(D1_EVENT_KINDS) & (set(WATCH_KINDS) | set(D1_LEVEL_KINDS))


def test_d1_event_levels_derive_from_completed_sessions_only():
    # 25 sessions ending the day BEFORE the queried session; the session's own
    # (possibly forming) bar must never feed its reference levels.
    bars = [
        _daily(-offset, high=100.0 + offset, low=90.0 - offset, close=95.0)
        for offset in range(25, 0, -1)
    ]
    bars.append(_daily(0, high=500.0, low=1.0, close=250.0))  # today: ignored
    levels = d1_event_levels(bars, session=DAY.date())

    assert levels["high_5d"] == 105.0  # offsets 1..5
    assert levels["low_5d"] == 85.0
    assert levels["high_20d"] == 120.0
    assert levels["low_20d"] == 70.0
    assert levels["prev_close"] == 95.0
    assert levels["ema15"] == pytest.approx(95.0)  # constant closes
    # 25 sessions cannot support an SMA50/100/200.
    assert "sma50" not in levels and "sma100" not in levels and "sma200" not in levels

    # 50 sessions unlock the SMA50.
    more = [
        _daily(-offset, high=100.0, low=90.0, close=95.0)
        for offset in range(50, 0, -1)
    ]
    assert d1_event_levels(more, session=DAY.date())["sma50"] == pytest.approx(95.0)


def test_new_5d_high_fires_on_completed_post_arm_bar_only():
    daily = [
        _daily(-offset, high=100.0 + offset, low=90.0, close=95.0)
        for offset in range(6, 0, -1)
    ]  # prior 5-session high = 105 (offsets 1..5)
    watch = D1EventWatch(symbol="NVDA", kind="new_5d_high", armed_at=DAY.replace(hour=10, minute=0))
    pre_arm = _bar(9, 35, h=104.0, low=99.0, c=103.0)
    breaker = _bar(10, 5, h=105.5, low=103.0, c=105.2)

    # Pre-arm bars never fire, even above the level.
    assert (
        evaluate_d1_event_watch(
            watch, [pre_arm, _bar(10, 5, h=104.9, low=103.0, c=104.0)], daily,
            now=DAY.replace(hour=10, minute=15),
        )
        is None
    )
    hit = evaluate_d1_event_watch(
        watch, [pre_arm, breaker], daily, now=DAY.replace(hour=10, minute=15)
    )
    assert hit is not None and hit.price == 105.5 and hit.resolved_side == "long"
    assert "New 5-day high: 105.50 > 105.00" in hit.message and "M5 bar" in hit.message
    # A forming breaker is preview only.
    assert (
        evaluate_d1_event_watch(
            watch, [pre_arm, breaker], daily, now=DAY.replace(hour=10, minute=8)
        )
        is None
    )


def test_new_20d_low_fires_and_5d_low_mirrors():
    daily = [
        _daily(-offset, high=100.0, low=90.0 - (offset % 3), close=95.0)
        for offset in range(21, 0, -1)
    ]  # prior lows cycle 88/89/90 -> 20d low = 88
    armed_at = DAY.replace(hour=10, minute=0)
    probe = _bar(10, 5, h=95.0, low=87.5, c=88.0)
    hit = evaluate_d1_event_watch(
        D1EventWatch(symbol="X", kind="new_20d_low", armed_at=armed_at),
        [probe],
        daily,
        now=DAY.replace(hour=10, minute=15),
    )
    assert hit is not None and hit.resolved_side == "short"
    assert "New 20-day low: 87.50 < 88.00" in hit.message

    hit = evaluate_d1_event_watch(
        D1EventWatch(symbol="X", kind="new_5d_low", armed_at=armed_at),
        [probe],
        daily,
        now=DAY.replace(hour=10, minute=15),
    )
    assert hit is not None and "New 5-day low" in hit.message


def test_sma_break_counts_a_gap_over_the_line_once():
    # 50 sessions: 49 closes at 100, last close 99 -> SMA50 = 99.98.
    daily = [
        _daily(-offset, high=101.0, low=98.0, close=(99.0 if offset == 1 else 100.0))
        for offset in range(50, 0, -1)
    ]
    sma50 = (49 * 100.0 + 99.0) / 50.0
    armed_at = DAY.replace(hour=10, minute=0)
    watch = D1EventWatch(symbol="NVDA", kind="sma_break", armed_at=armed_at)

    # Pre-arm gap over the SMA consumes the cross; post-arm bars holding above
    # never "break" it again without first crossing back.
    gap_open = _bar(9, 35, h=101.5, low=100.5, c=101.0)
    hold = _bar(10, 5, h=102.0, low=101.0, c=101.8)
    assert (
        evaluate_d1_event_watch(watch, [gap_open, hold], daily, now=DAY.replace(hour=10, minute=15))
        is None
    )

    # A post-arm close back through the line fires, naming the SMA and side.
    back_under = _bar(10, 10, h=101.9, low=99.0, c=99.5)
    hit = evaluate_d1_event_watch(
        watch, [gap_open, hold, back_under], daily, now=DAY.replace(hour=10, minute=20)
    )
    assert hit is not None and hit.resolved_side == "short"
    assert f"SMA50 break down: closed 99.50 under {sma50:.2f}" in hit.message

    # And the up-cross mirrors (prev close 99 below, post-arm close above).
    up = _bar(10, 5, h=100.5, low=98.5, c=100.3)
    hit = evaluate_d1_event_watch(watch, [up], daily, now=DAY.replace(hour=10, minute=15))
    assert hit is not None and hit.resolved_side == "long"
    assert "SMA50 break up" in hit.message


def test_ema15_reject_touch_and_reclaim_both_ways():
    daily = [
        _daily(-offset, high=101.0, low=99.0, close=100.0)
        for offset in range(20, 0, -1)
    ]  # EMA15 = 100 on constant closes
    armed_at = DAY.replace(hour=10, minute=0)
    watch = D1EventWatch(symbol="NVDA", kind="ema15_reject", armed_at=armed_at)

    # Trades entirely above the line: no tag, no fire.
    assert (
        evaluate_d1_event_watch(
            watch, [_bar(10, 5, h=102.0, low=100.5, c=101.5)], daily,
            now=DAY.replace(hour=10, minute=15),
        )
        is None
    )
    # Dips to the D1 15EMA and closes back above: long rejection.
    hit = evaluate_d1_event_watch(
        watch, [_bar(10, 5, h=101.5, low=99.8, c=100.9)], daily,
        now=DAY.replace(hour=10, minute=15),
    )
    assert hit is not None and hit.resolved_side == "long"
    assert "D1 15EMA rejection (long): tagged 100.00" in hit.message
    # Pops into it and closes back below: short rejection.
    hit = evaluate_d1_event_watch(
        watch, [_bar(10, 5, h=100.4, low=98.9, c=99.2)], daily,
        now=DAY.replace(hour=10, minute=15),
    )
    assert hit is not None and hit.resolved_side == "short"


def test_d1_event_watch_daily_fallback_for_unscanned_symbols():
    # Armed on DAY; no M5 evidence. Store: 6 prior sessions (5d high 105),
    # then a session 2 days later that breaks it. Today's own bar is preview.
    daily = [
        _daily(-offset, high=100.0 + offset, low=90.0, close=95.0)
        for offset in range(6, 0, -1)
    ]
    watch = D1EventWatch(symbol="NVDA", kind="new_5d_high", armed_at=DAY.replace(hour=14, minute=0))
    later = DAY + timedelta(days=4)

    # The armed day's own daily bar never triggers (contains pre-arm prices).
    assert (
        evaluate_d1_event_watch(
            watch, [], daily + [_daily(0, high=200.0, low=90.0, close=150.0)], now=later
        )
        is None
    )
    breaker = _daily(2, high=106.0, low=95.0, close=105.5)
    hit = evaluate_d1_event_watch(watch, [], daily + [breaker], now=later)
    assert hit is not None and hit.resolved_side == "long"
    assert "D1 bar" in hit.message
    # A break on today's (forming) session stays preview.
    assert (
        evaluate_d1_event_watch(
            watch, [], daily + [_daily(4, high=200.0, low=95.0, close=150.0)], now=later
        )
        is None
    )


def test_d1_event_watch_persistence_roundtrip(tmp_path):
    path = tmp_path / "d1_event_watches.json"
    armed_at = DAY.replace(hour=14, minute=0)
    watches = [
        D1EventWatch(symbol="NVDA", kind="new_5d_high", armed_at=armed_at),
        D1EventWatch(symbol="TSLA", kind="sma_break", armed_at=armed_at),
    ]
    save_d1_event_watches(watches, path)
    assert load_d1_event_watches(path) == watches

    # Unknown kinds (e.g. from a future build) are dropped, not crashed on.
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["watches"].append(
        {"symbol": "AMD", "kind": "warp_drive", "armed_at": armed_at.isoformat()}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_d1_event_watches(path) == watches
    assert load_d1_event_watches(tmp_path / "missing.json") == []
