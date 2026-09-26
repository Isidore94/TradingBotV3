"""S16 item 2: machine structure labels from completed daily bars (pure)."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from research_warehouse import exchange_calendar as xcal  # noqa: E402

AS_OF = date(2026, 9, 25)  # a Friday session


def _sessions_before(day: date, count: int) -> list[date]:
    days = [session.session_date for session in xcal.sessions_between(date(2024, 1, 1), day) if session.session_date < day]
    return days[-count:]


def _rows(closes, *, spread=None, days=None) -> list[dict]:
    days = days or _sessions_before(AS_OF, len(closes))
    rows = []
    for index, (day, close) in enumerate(zip(days, closes, strict=True)):
        half = (spread[index] if spread else 1.0) / 2
        rows.append(
            {
                "session_date": day,
                "open": close,
                "high": close + half,
                "low": close - half,
                "close": close,
                "volume": 1_000_000,
            }
        )
    return rows


def _zigzag(count: int, *, trend: float, period: int = 10, amplitude: float = 5.0) -> list[float]:
    closes = []
    for index in range(count):
        phase = index % period
        wave = phase if phase <= period // 2 else period - phase
        closes.append(500.0 + trend * index + amplitude * wave / (period // 2))
    return closes


# ------------------------------------------------------------- weekly --


def test_weekly_bars_are_completed_weeks_only():
    import market_structure as ms

    rows = _rows([100.0 + i for i in range(30)])
    weekly = ms.completed_weekly_bars(rows, AS_OF)
    # AS_OF is a Friday: its own week (Mon 09-21 .. Thu 09-24 in the rows) is not complete.
    assert weekly and weekly[-1]["session_date"] < date(2026, 9, 21)
    last = weekly[-1]
    members = [row for row in rows if date(2026, 9, 14) <= row["session_date"] <= date(2026, 9, 18)]
    assert last["open"] == members[0]["open"] and last["close"] == members[-1]["close"]
    assert last["high"] == max(row["high"] for row in members)
    assert last["low"] == min(row["low"] for row in members)
    # On the next Monday the week of 09-21 is complete.
    assert ms.completed_weekly_bars(rows, date(2026, 9, 28))[-1]["session_date"] == date(2026, 9, 24)


def test_weekly_higher_highs_and_higher_lows_are_counted():
    import market_structure as ms

    rising = ms.weekly_structure(ms.completed_weekly_bars(_rows([100.0 + i for i in range(80)]), AS_OF))
    assert rising["status"] == "ok"
    assert rising["weeks"] == ms.WEEKLY_LOOKBACK
    assert rising["higher_highs"] == rising["higher_lows"] == ms.WEEKLY_LOOKBACK
    assert rising["lower_highs"] == rising["lower_lows"] == 0

    falling = ms.weekly_structure(ms.completed_weekly_bars(_rows([300.0 - i for i in range(80)]), AS_OF))
    assert falling["lower_highs"] == falling["lower_lows"] == ms.WEEKLY_LOOKBACK
    assert falling["higher_highs"] == 0


def test_too_few_weeks_is_unknown():
    import market_structure as ms

    thin = ms.weekly_structure(ms.completed_weekly_bars(_rows([100.0, 101.0, 102.0]), AS_OF))
    assert thin["status"] == "unknown" and thin["higher_highs"] is None


# ------------------------------------------------------------ channel --


def test_a_falling_zigzag_is_a_lower_high_lower_low_channel():
    import market_structure as ms

    channel = ms.daily_channel(_rows(_zigzag(60, trend=-0.3)))
    assert channel["label"] == "lh_ll"
    assert channel["last_pivot_high"]["value"] < channel["prev_pivot_high"]["value"]
    assert channel["last_pivot_low"]["value"] < channel["prev_pivot_low"]["value"]
    assert channel["last_pivot_high"]["date"] > channel["prev_pivot_high"]["date"]


def test_a_rising_zigzag_is_higher_highs_and_higher_lows():
    import market_structure as ms

    assert ms.daily_channel(_rows(_zigzag(60, trend=0.3)))["label"] == "hh_hl"


def test_a_straight_line_has_no_pivots_and_is_unknown():
    import market_structure as ms

    assert ms.daily_channel(_rows([100.0 + i for i in range(60)]))["label"] == "unknown"


def test_the_channel_uses_the_scan_pivot_code():
    import market_structure as ms
    from master_avwap_lib import legacy

    assert ms.pivot_finder() is legacy._find_trendline_pivots


# -------------------------------------------------------- compression --


def test_tight_recent_ranges_read_compressed():
    import market_structure as ms

    spread = [6.0] * 200 + [1.0] * 30
    facts = ms.atr_compression(_rows([100.0] * 230, spread=spread))
    assert facts["status"] == "ok"
    assert facts["percentile"] <= ms.COMPRESSED_PERCENTILE
    assert facts["compressed"] is True

    wide = ms.atr_compression(_rows([100.0] * 230, spread=[1.0] * 200 + [6.0] * 30))
    assert wide["compressed"] is False and wide["percentile"] == 100.0


def test_short_history_compression_is_unknown():
    import market_structure as ms

    facts = ms.atr_compression(_rows([100.0] * 40))
    assert facts["status"] == "unknown" and facts["compressed"] is None


# -------------------------------------------------------------- sma20 --


def test_distance_to_the_20_day_and_its_slope():
    import market_structure as ms

    facts = ms.sma20_facts(_rows([100.0 + i for i in range(40)]))
    # Last close 139, SMA20 of 120..139 = 129.5.
    assert facts["status"] == "ok"
    assert facts["distance_pct"] == round((139 / 129.5 - 1) * 100, 4)
    assert facts["slope_pct"] > 0 and facts["side"] == "above"

    down = ms.sma20_facts(_rows([200.0 - i for i in range(40)]))
    assert down["slope_pct"] < 0 and down["side"] == "below"

    thin = ms.sma20_facts(_rows([100.0] * 10))
    assert thin["status"] == "unknown" and thin["distance_pct"] is None


# ------------------------------------------------------- point in time --


def test_structure_facts_never_read_the_as_of_session_or_later():
    import market_structure as ms

    days = _sessions_before(date(2026, 10, 2), 260)
    rows = _rows(_zigzag(260, trend=-0.2), days=days)
    before = ms.structure_facts([row for row in rows if row["session_date"] < AS_OF], AS_OF)
    crash = [
        {**row, "close": 1.0, "low": 0.5, "high": 1.5} if row["session_date"] >= AS_OF else row
        for row in rows
    ]
    assert ms.structure_facts(crash, AS_OF) == before
    assert set(before) == {"weekly", "daily_channel", "atr", "sma20"}


def test_no_bars_is_unknown_everywhere():
    import market_structure as ms

    facts = ms.structure_facts([], AS_OF)
    assert facts["weekly"]["status"] == "unknown"
    assert facts["daily_channel"]["label"] == "unknown"
    assert facts["atr"]["status"] == "unknown"
    assert facts["sma20"]["status"] == "unknown"
