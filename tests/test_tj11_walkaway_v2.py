"""TJ-11 Part A - walk-away v2 inside `walkaway_day.build`. RED before the build.

Packet `.claude/packets/TJ-11.md` items 2-8; `plan.md` §12.4 "TJ-11" items 1-8;
decision 0021 answers 16, 22 and 24.

Every fixture here is a plain dict. Nothing reads a live store, a clock or the
network. Bars are explicitly zoned; a daily ATR of exactly 2.00 on a 100.00 price
makes one ATR exactly 2%, so every expected number below is exact.

The contract these tests pin, so the builder has nothing to guess
-----------------------------------------------------------------

``walkaway_day.build(session, sources, bars, *, trades=(), claims=(), now=None,
daily_bars=None)`` - the four existing populations are unchanged, and:

``sources["scan_rows"]``
    The session's scan rows as `master_avwap_session_horizon_outcomes` spells
    them (``symbol``, ``side``, ``scan_date``, ``horizon_sessions``,
    ``setup_family``) - four rows per name, one per horizon. The POPULATION
    grain is the distinct ``(symbol, side)``, never the row count. These come
    to Day Review through `day_review_index`, never the 1.1 GB tracker.
``sources["earlier_decisions"]``
    The D1 decisions of earlier sessions, same shape as ``sources["decisions"]``.
``daily_bars``
    ``{symbol: [daily bar, ...]}`` oldest-first from the durable daily store
    (`chart_snapshot.load_d1_bars`, which is the symbol-level DAILY reader the
    desk already uses off the Qt thread - NOT `ui/journal_chart_bars.py`).

``walkaway_day.earlier_sessions(session, *, count=5) -> tuple[str, ...]``
    The ``count`` exchange sessions strictly BEFORE ``session``, oldest first,
    walked on `market_calendar`, never in calendar days.

``WalkawayRow`` gains, all defaulting to ``None`` / ``""``:
    ``against_first_pct``   worst adverse excursion BEFORE the best favourable
                            one, signed negative
    ``at_close_pct``        side-adjusted move to the measured close
    ``ran_after_atr`` / ``against_first_atr`` / ``at_close_atr``
                            the same three moves in ATR(14) units
    ``horizon_moves``       D1 only: ``{1: pct|None, 3: ..., 5: ...}``
    ``real_miss``           `real_miss.verdict`'s answer for this row
    ``instrument``          ``"STK"`` / ``"OPT"`` / ``""``
    ``sessions_held``       closed trades only, EXCHANGE sessions open to close
                            (`market_calendar.trading_days_between`), never
                            calendar days
    ``not_judged_reason``   why a money figure was withheld, never a wrong number

``WalkawayDay`` gains:
    ``earlier_calls``  the fifth population, most-ran first
    ``skill``          ``{"session": WINDOW, "lately": WINDOW}``
    ``sentences``      one deterministic line per table, keyed by population
    ``money``          ``{"n": int, "net": float|None, "line": str}``

``WINDOW`` = ``{"window_sessions": int, "cells": (CELL, ...),
"overlapping": ((population, population), ...), "sentence": str}``

``CELL`` = ``{"population": "liked_or_claimed"|"rejected"|"untouched",
"side": "LONG"|"SHORT", "setup_family": str, "n": int, "measured": int,
"unmeasured": int, "runs": int, "rate": float|None, "low": float|None,
"high": float|None, "reportable": bool}``. ``n`` is the population size and the
three populations PARTITION the scan; ``measured`` is the Wilson denominator
(Q1's rule: the unmeasured are shown, never assumed); ``low``/``high`` are the
ONE Wilson (`swing_headline.wilson_lower_bound`'s z) and ``reportable`` is
``measured >= evidence_stats.MIN_REPORTABLE_N``.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

EASTERN = ZoneInfo("America/New_York")
PACIFIC = ZoneInfo("America/Los_Angeles")

SESSION = "2026-09-18"
STAMP = "2026-09-18T13:00:00-04:00"
#: Flat pre-history: every bar spans exactly 2.00 around a 100.00 close, so
#: Wilder ATR(14) is exactly 2.00 and one ATR is exactly 2%.
ATR = 2.0


# -- fixtures ---------------------------------------------------------------


def _decision(
    *,
    symbol: str,
    side: str = "LONG",
    verdict: str = "like",
    timeframe: str = "M5",
    stamp: str = STAMP,
    session: str = SESSION,
    reason: str = "",
    source: str = "annotations",
    category: str = "chart_review",
    capture_id: str = "",
) -> dict:
    return {
        "session_date": session,
        "symbol": symbol,
        "side": side,
        "category": category,
        "verdict": verdict,
        "source": source,
        "timeframe": timeframe,
        "stamp": stamp,
        "capture_id": capture_id,
        "reason": reason,
    }


def _m5(hour: int, minute: int, *, open_: float, high: float, low: float, close: float) -> dict:
    return {
        "dt": datetime(2026, 9, 18, hour, minute, tzinfo=EASTERN).isoformat(),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    }


def _sessions_ending(session: str, count: int) -> list[str]:
    """`count` exchange sessions ending at `session`, oldest first."""
    from market_calendar import previous_session

    cursor = date.fromisoformat(session)
    out = [cursor.isoformat()]
    for _ in range(count - 1):
        cursor = previous_session(cursor)
        out.append(cursor.isoformat())
    return list(reversed(out))


def _sessions_after(session: str, count: int) -> list[str]:
    """`count` exchange sessions strictly after `session`, oldest first."""
    from market_calendar import is_session

    cursor = date.fromisoformat(session)
    out: list[str] = []
    while len(out) < count:
        cursor += timedelta(days=1)
        if is_session(cursor):
            out.append(cursor.isoformat())
    return out


def _flat_daily(session: str = SESSION, *, bars: int = 15, close: float = 100.0) -> list[dict]:
    """Enough flat history for a Wilder ATR(14) of exactly 2.00."""
    return [
        {"dt": f"{day}T00:00:00", "open": close, "high": close + 1, "low": close - 1, "close": close}
        for day in _sessions_ending(session, bars)
    ]


def _daily_after(session: str, rows: list[tuple[float, float, float, float]]) -> list[dict]:
    days = _sessions_after(session, len(rows))
    return [
        {"dt": f"{day}T00:00:00", "open": o, "high": h, "low": low, "close": c}
        for day, (o, h, low, c) in zip(days, rows)
    ]


def _build(
    *,
    session: str = SESSION,
    decisions=(),
    bars=None,
    trades=(),
    claims=(),
    preference=(),
    outcomes=(),
    scan_rows=(),
    earlier_decisions=(),
    daily_bars=None,
    now=None,
):
    from walkaway_day import build

    return build(
        session,
        sources={
            "decisions": tuple(decisions),
            "preference": tuple(preference),
            "outcomes": tuple(outcomes),
            "scan_rows": tuple(scan_rows),
            "earlier_decisions": tuple(earlier_decisions),
        },
        bars=bars if bars is not None else {},
        trades=tuple(trades),
        claims=tuple(claims),
        now=now or datetime(2026, 9, 19, 8, 0),
        daily_bars=daily_bars or {},
    )


def _all_rows(day):
    return tuple(day.liked_not_traded + day.rejected + day.traded_left_early + day.claimed_d1)


def _row(day, symbol: str):
    for row in _all_rows(day) + tuple(day.earlier_calls):
        if row.symbol == symbol:
            return row
    raise AssertionError(f"{symbol} is in no population: {[r.symbol for r in _all_rows(day)]}")


# -- item 2: three moves, in percent and in ATR ------------------------------


def test_every_row_carries_the_three_moves_in_percent_and_in_atr():
    """Open 100.00, low 98.00, high 110.00, close 106.00, ATR 2.00."""
    bars = {
        "AAA": [
            _m5(9, 40, open_=100.0, high=100.5, low=98.0, close=99.0),
            _m5(9, 45, open_=99.0, high=110.0, low=99.0, close=109.0),
            _m5(15, 55, open_=109.0, high=109.0, low=105.0, close=106.0),
        ]
    }
    day = _build(
        decisions=[_decision(symbol="AAA", verdict="veto", stamp="2026-09-18T09:35:00-04:00")],
        bars=bars,
        daily_bars={"AAA": _flat_daily()},
    )

    row = _row(day, "AAA")
    assert row.ran_after_pct == pytest.approx(10.0)
    assert row.against_first_pct == pytest.approx(-2.0)
    assert row.at_close_pct == pytest.approx(6.0)
    assert row.ran_after_atr == pytest.approx(5.0)
    assert row.against_first_atr == pytest.approx(-1.0)
    assert row.at_close_atr == pytest.approx(3.0)


def test_a_missing_atr_leaves_the_atr_columns_unmeasured_and_the_percents_alone():
    bars = {
        "AAA": [
            _m5(9, 40, open_=100.0, high=100.5, low=98.0, close=99.0),
            _m5(9, 45, open_=99.0, high=110.0, low=99.0, close=109.0),
        ]
    }
    day = _build(
        decisions=[_decision(symbol="AAA", verdict="veto", stamp="2026-09-18T09:35:00-04:00")],
        bars=bars,
        daily_bars={},
    )

    row = _row(day, "AAA")
    assert row.ran_after_pct == pytest.approx(10.0)
    assert row.ran_after_atr is None
    assert row.against_first_atr is None
    assert row.real_miss.startswith("unmeasured:"), row.real_miss


def test_the_worst_adverse_move_counted_is_the_one_before_the_best_favourable_one():
    """A 6% drop AFTER the high is not what you were up against first."""
    bars = {
        "AAA": [
            _m5(9, 40, open_=100.0, high=101.0, low=99.0, close=100.5),
            _m5(9, 45, open_=100.5, high=110.0, low=100.0, close=109.0),
            _m5(9, 50, open_=109.0, high=109.0, low=94.0, close=95.0),
        ]
    }
    day = _build(
        decisions=[_decision(symbol="AAA", verdict="veto", stamp="2026-09-18T09:35:00-04:00")],
        bars=bars,
        daily_bars={"AAA": _flat_daily()},
    )

    assert _row(day, "AAA").against_first_pct == pytest.approx(-1.0)


def test_each_row_carries_the_versioned_real_miss_verdict():
    bars = {
        "AAA": [
            _m5(9, 40, open_=100.0, high=101.2, low=99.5, close=101.0),
            _m5(9, 45, open_=101.0, high=103.0, low=100.8, close=102.8),
        ],
        "BBB": [
            _m5(9, 40, open_=100.0, high=100.2, low=98.6, close=99.0),
            _m5(9, 45, open_=99.0, high=105.0, low=99.0, close=104.5),
        ],
    }
    day = _build(
        decisions=[
            _decision(symbol="AAA", verdict="veto", stamp="2026-09-18T09:35:00-04:00"),
            _decision(symbol="BBB", verdict="veto", stamp="2026-09-18T09:35:00-04:00"),
        ],
        bars=bars,
        daily_bars={"AAA": _flat_daily(), "BBB": _flat_daily()},
    )

    assert _row(day, "AAA").real_miss == "run"
    assert _row(day, "BBB").real_miss == "no_run"


# -- item 2: a D1 decision gets a D1 ruler -----------------------------------


def _d1_veto(symbol: str = "BBB", side: str = "SHORT") -> dict:
    return _decision(symbol=symbol, side=side, verdict="veto", timeframe="D1")


def _bbb_daily(after: list[tuple[float, float, float, float]]) -> list[dict]:
    return _flat_daily() + _daily_after(SESSION, after)


def test_a_d1_decision_is_measured_on_daily_bars_over_one_three_and_five_sessions():
    """Short from a 100.00 close; +3% / +6% / +10% at the three horizons."""
    after = [
        (99.0, 101.0, 96.0, 97.0),   # 09-21  h1: +3%
        (97.0, 97.0, 94.5, 95.0),    # 09-22
        (95.0, 95.0, 93.5, 94.0),    # 09-23  h3: +6%
        (94.0, 94.0, 92.0, 93.0),    # 09-24
        (93.0, 93.0, 89.0, 90.0),    # 09-25  h5: +10%
    ]
    day = _build(
        decisions=[_d1_veto()],
        daily_bars={"BBB": _bbb_daily(after)},
        now=datetime(2026, 9, 28, 8, 0),
    )

    row = _row(day, "BBB")
    assert dict(row.horizon_moves) == {
        1: pytest.approx(3.0),
        3: pytest.approx(6.0),
        5: pytest.approx(10.0),
    }
    # Best favourable excursion is the 89.00 low; worst adverse before it the
    # 101.00 high. ATR is 2.00, so ATR units are half the percent.
    assert row.ran_after_pct == pytest.approx(11.0)
    assert row.against_first_pct == pytest.approx(-1.0)
    assert row.at_close_pct == pytest.approx(10.0)
    assert row.ran_after_atr == pytest.approx(5.5)
    assert row.state == "measured"


def test_a_d1_veto_is_pending_with_its_date_until_the_horizon_closes():
    """09-21 has closed; 09-23 and 09-25 have not. Never a zero."""
    after = [(99.0, 101.0, 96.0, 97.0)]
    day = _build(
        decisions=[_d1_veto()],
        daily_bars={"BBB": _bbb_daily(after)},
        now=datetime(2026, 9, 22, 8, 0, tzinfo=EASTERN),
    )

    row = _row(day, "BBB")
    assert row.state == "pending 2026-09-25", row.state
    assert dict(row.horizon_moves)[1] == pytest.approx(3.0)
    assert dict(row.horizon_moves)[3] is None
    assert dict(row.horizon_moves)[5] is None
    assert row.ran_after_pct is None
    assert row.at_close_pct is None


def test_a_d1_decision_never_reads_zero_while_its_horizon_is_open():
    day = _build(
        decisions=[_d1_veto()],
        daily_bars={"BBB": _flat_daily()},
        now=datetime(2026, 9, 21, 8, 0, tzinfo=EASTERN),
    )

    row = _row(day, "BBB")
    for value in (row.ran_after_pct, row.against_first_pct, row.at_close_pct,
                  row.ran_after_atr, row.at_close_atr):
        assert value is None, value
    assert row.state.startswith("pending "), row.state


def test_a_d1_decision_never_uses_the_sessions_five_minute_tape():
    """~95% of the day's decisions are D1 calls (decision 0021 amendment)."""
    intraday = {
        "BBB": [
            _m5(9, 40, open_=100.0, high=100.0, low=80.0, close=81.0),
            _m5(15, 55, open_=81.0, high=82.0, low=80.0, close=81.0),
        ]
    }
    after = [(99.0, 101.0, 96.0, 97.0), (97.0, 97.0, 94.5, 95.0),
             (95.0, 95.0, 93.5, 94.0), (94.0, 94.0, 92.0, 93.0),
             (93.0, 93.0, 89.0, 90.0)]
    day = _build(
        decisions=[_d1_veto()],
        bars=intraday,
        daily_bars={"BBB": _bbb_daily(after)},
        now=datetime(2026, 9, 28, 8, 0),
    )

    # The M5 tape would say the short ran 20%. The D1 ruler says 11%.
    assert _row(day, "BBB").ran_after_pct == pytest.approx(11.0)


# -- item 3: earlier calls, now ---------------------------------------------


def test_the_earlier_calls_window_walks_five_exchange_sessions_across_a_holiday():
    """Labor Day 2026 is Monday 2026-09-07, so it is skipped, not counted."""
    from walkaway_day import earlier_sessions

    assert earlier_sessions("2026-09-08") == (
        "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04",
    )
    assert earlier_sessions("2026-09-18") == (
        "2026-09-11", "2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17",
    )
    assert "2026-09-19" not in earlier_sessions("2026-09-21")
    assert "2026-09-20" not in earlier_sessions("2026-09-21")


def test_earlier_calls_are_a_fifth_population_moved_to_the_selected_sessions_close():
    """A Wednesday veto that ran two ATR by Friday is on Friday's table."""
    earlier = [
        _d1_veto(symbol="CCC") | {"session_date": "2026-09-16", "stamp": "2026-09-16T13:00:00-04:00"},
        _d1_veto(symbol="DDD") | {"session_date": "2026-09-17", "stamp": "2026-09-17T13:00:00-04:00"},
    ]
    # Both shorts from a 100.00 close; CCC closes the session at 94, DDD at 98.
    ccc = _flat_daily("2026-09-16") + _daily_after(
        "2026-09-16", [(99.0, 99.0, 95.0, 96.0), (96.0, 96.0, 93.0, 94.0)])
    ddd = _flat_daily("2026-09-17") + _daily_after("2026-09-17", [(99.0, 100.0, 97.0, 98.0)])
    day = _build(
        earlier_decisions=earlier,
        daily_bars={"CCC": ccc, "DDD": ddd},
        now=datetime(2026, 9, 19, 8, 0),
    )

    assert [row.symbol for row in day.earlier_calls] == ["CCC", "DDD"], "most-ran first"
    assert day.earlier_calls[0].at_close_pct == pytest.approx(6.0)
    assert day.earlier_calls[1].at_close_pct == pytest.approx(2.0)
    assert day.earlier_calls[0].real_miss in {"run", "no_run"}


def test_the_earlier_calls_are_not_counted_among_the_sessions_own_decisions():
    earlier = [_d1_veto(symbol="CCC") | {"session_date": "2026-09-17",
                                         "stamp": "2026-09-17T13:00:00-04:00"}]
    day = _build(
        decisions=[_decision(symbol="AAA", verdict="veto")],
        earlier_decisions=earlier,
        daily_bars={"CCC": _flat_daily("2026-09-17"), "AAA": _flat_daily()},
    )

    assert len(_all_rows(day)) == 1
    assert len(day.earlier_calls) == 1


# -- item 4: the skill line --------------------------------------------------


def _scan_rows(names, side: str = "LONG", family: str = "general", session: str = SESSION):
    """Four rows per name, exactly as the live horizon-outcomes file holds them."""
    return [
        {
            "symbol": symbol, "side": side, "scan_date": session,
            "horizon_sessions": str(horizon), "setup_family": family,
        }
        for symbol in names
        for horizon in (1, 3, 5)
    ] + [
        {
            "symbol": symbol, "side": side, "scan_date": session,
            "horizon_sessions": "10", "setup_family": family,
        }
        for symbol in names
    ]


def _run_daily(symbol_session: str = SESSION, *, run: bool) -> list[dict]:
    """Five closed daily sessions that are, or are not, a REAL_MISS_V1 run."""
    if run:
        after = [(100.0, 103.0, 99.9, 102.5)] + [(102.5, 103.0, 102.0, 102.5)] * 4
    else:
        after = [(100.0, 100.5, 98.5, 98.6)] + [(98.6, 99.0, 98.0, 98.5)] * 4
    return _flat_daily(symbol_session) + _daily_after(symbol_session, after)


def _cell(day, population: str, *, side: str = "LONG", window: str = "session", family: str = ""):
    cells = [
        cell for cell in day.skill[window]["cells"]
        if cell["population"] == population and cell["side"] == side
        and cell.get("setup_family", "") == family
    ]
    assert len(cells) == 1, (population, side, family, day.skill[window]["cells"])
    return cells[0]


def test_the_three_skill_populations_partition_the_scan_rows_for_a_side():
    """Six names, four horizon rows each: n sums to SIX, not twenty-four."""
    names = ["S1", "S2", "S3", "S4", "S5", "S6"]
    decisions = [
        _decision(symbol="S1", verdict="like", timeframe="D1"),
        _decision(symbol="S2", verdict="like", timeframe="D1"),
        _decision(symbol="S3", verdict="veto", timeframe="D1"),
        _decision(symbol="S4", verdict="veto", timeframe="D1"),
    ]
    claims = [{
        "symbol": "S5", "side": "LONG", "action": "claim", "session_date": SESSION,
        "claimed_setup_id": "trendline_break", "horizon": "d1", "annotation_ref": "",
    }]
    day = _build(
        decisions=decisions,
        claims=claims,
        scan_rows=_scan_rows(names),
        daily_bars={name: _run_daily(run=False) for name in names},
        now=datetime(2026, 9, 28, 8, 0),
    )

    liked = _cell(day, "liked_or_claimed")
    rejected = _cell(day, "rejected")
    untouched = _cell(day, "untouched")
    assert (liked["n"], rejected["n"], untouched["n"]) == (3, 2, 1)
    assert liked["n"] + rejected["n"] + untouched["n"] == 6


def test_an_untouched_name_comes_from_the_scan_rows_and_never_from_the_tracker():
    names = ["S1", "S2"]
    day = _build(
        decisions=[_decision(symbol="S1", verdict="like", timeframe="D1")],
        scan_rows=_scan_rows(names),
        daily_bars={name: _run_daily(run=True) for name in names},
        now=datetime(2026, 9, 28, 8, 0),
    )

    assert _cell(day, "untouched")["n"] == 1
    assert _cell(day, "untouched")["runs"] == 1


def test_an_unmeasured_name_is_shown_and_never_assumed_into_the_rate():
    """Q1's rule: the denominator is what was MEASURED."""
    names = ["S1", "S2", "S3"]
    day = _build(
        scan_rows=_scan_rows(names),
        daily_bars={"S1": _run_daily(run=True), "S2": _run_daily(run=False)},
        now=datetime(2026, 9, 28, 8, 0),
    )

    cell = _cell(day, "untouched")
    assert cell["n"] == 3
    assert cell["measured"] == 2
    assert cell["unmeasured"] == 1
    assert cell["rate"] == pytest.approx(0.5)


def _wide_skill_day():
    """100 likes at 90% runs, 100 vetoes at 5%, 100 untouched at 12%."""
    liked = [f"L{i:03d}" for i in range(100)]
    vetoed = [f"V{i:03d}" for i in range(100)]
    untouched = [f"U{i:03d}" for i in range(100)]
    runs = dict.fromkeys(liked[:90], True) | dict.fromkeys(liked[90:], False)
    runs |= dict.fromkeys(vetoed[:5], True) | dict.fromkeys(vetoed[5:], False)
    runs |= dict.fromkeys(untouched[:12], True) | dict.fromkeys(untouched[12:], False)
    decisions = (
        [_decision(symbol=name, verdict="like", timeframe="D1") for name in liked]
        + [_decision(symbol=name, verdict="veto", timeframe="D1") for name in vetoed]
    )
    return _build(
        decisions=decisions,
        scan_rows=_scan_rows(liked + vetoed + untouched),
        daily_bars={name: _run_daily(run=is_run) for name, is_run in runs.items()},
        now=datetime(2026, 9, 28, 8, 0),
    )


def test_every_skill_cell_carries_n_and_the_one_wilson_interval():
    from swing_headline import wilson_lower_bound

    day = _wide_skill_day()
    liked = _cell(day, "liked_or_claimed")

    assert (liked["n"], liked["measured"], liked["runs"]) == (100, 100, 90)
    assert liked["rate"] == pytest.approx(0.90)
    assert liked["low"] == pytest.approx(wilson_lower_bound(90, 100))
    assert liked["high"] == pytest.approx(0.9447708629393249)
    assert liked["low"] < liked["rate"] < liked["high"]


def test_overlapping_intervals_are_flagged_and_disjoint_ones_are_not():
    """Vetoes 5% and untouched 12% overlap; likes at 90% overlap neither."""
    day = _wide_skill_day()

    overlapping = {tuple(sorted(pair)) for pair in day.skill["session"]["overlapping"]}
    assert ("rejected", "untouched") in overlapping
    assert ("liked_or_claimed", "rejected") not in overlapping
    assert ("liked_or_claimed", "untouched") not in overlapping
    assert "overlap" in day.skill["session"]["sentence"].lower()


def test_the_skill_line_is_cut_by_side_and_the_lately_window_is_lately_sessions():
    from evidence_stats import LATELY_SESSIONS

    day = _wide_skill_day()

    assert day.skill["session"]["window_sessions"] == 1
    assert day.skill["lately"]["window_sessions"] == LATELY_SESSIONS
    assert {cell["side"] for cell in day.skill["session"]["cells"]} == {"LONG"}


def test_nothing_is_named_under_the_reportable_floor():
    from evidence_stats import MIN_REPORTABLE_N

    names = ["S1", "S2", "S3", "S4"]
    day = _build(
        decisions=[_decision(symbol="S1", verdict="like", timeframe="D1")],
        scan_rows=_scan_rows(names),
        daily_bars={name: _run_daily(run=True) for name in names},
        now=datetime(2026, 9, 28, 8, 0),
    )

    for cell in day.skill["session"]["cells"]:
        assert cell["measured"] < MIN_REPORTABLE_N
        assert cell["reportable"] is False
    assert _wide_skill_day().skill["session"]["cells"][0]["reportable"] is True


def test_a_setup_family_cut_appears_only_where_n_allows_it():
    """Two families, one of them a single name - the thin cut is not shown."""
    fat = [f"F{i:03d}" for i in range(40)]
    thin = ["T1"]
    rows = _scan_rows(fat, family="avwap_band_bounce") + _scan_rows(thin, family="top_pattern_tracking")
    day = _build(
        scan_rows=rows,
        daily_bars={name: _run_daily(run=True) for name in fat + thin},
        now=datetime(2026, 9, 28, 8, 0),
    )

    families = {cell.get("setup_family", "") for cell in day.skill["session"]["cells"]}
    assert "avwap_band_bounce" in families
    assert "top_pattern_tracking" not in families


def test_no_r_statistic_selects_what_the_skill_line_shows():
    """Gate #43's rule: size and name order the view, never a result."""
    day = _wide_skill_day()
    text = json.dumps(day.skill["session"], default=str).lower()

    for banned in ("expected_r", "avg_r", "_r\"", "r_multiple", "profit_factor"):
        assert banned not in text, banned


# -- item 5: one deterministic sentence above each table ---------------------


def _four_vetoes():
    """AAA and BBB run; three of the four share the code `extended`."""
    bars = {
        "AAA": [_m5(9, 40, open_=100.0, high=101.2, low=99.5, close=101.0),
                _m5(9, 45, open_=101.0, high=103.0, low=100.8, close=102.8)],
        "BBB": [_m5(9, 40, open_=100.0, high=101.2, low=99.5, close=101.0),
                _m5(9, 45, open_=101.0, high=104.0, low=100.8, close=103.8)],
        "CCC": [_m5(9, 40, open_=100.0, high=100.2, low=98.6, close=99.0),
                _m5(9, 45, open_=99.0, high=99.5, low=98.0, close=98.5)],
        "DDD": [_m5(9, 40, open_=100.0, high=100.2, low=98.6, close=99.0),
                _m5(9, 45, open_=99.0, high=99.5, low=98.0, close=98.5)],
    }
    decisions = [
        _decision(symbol="AAA", verdict="veto", reason="extended",
                  stamp="2026-09-18T09:35:00-04:00"),
        _decision(symbol="BBB", verdict="veto", reason="extended",
                  stamp="2026-09-18T09:35:00-04:00"),
        _decision(symbol="CCC", verdict="veto", reason="extended",
                  stamp="2026-09-18T09:35:00-04:00"),
        _decision(symbol="DDD", verdict="veto", reason="sma_incoming",
                  stamp="2026-09-18T09:35:00-04:00"),
    ]
    return _build(
        decisions=decisions,
        bars=bars,
        daily_bars={name: _flat_daily() for name in ("AAA", "BBB", "CCC", "DDD")},
    )


def test_each_table_gets_one_sentence_carrying_its_own_counts():
    day = _four_vetoes()

    sentence = day.sentences["rejected"]
    assert sentence.startswith("You vetoed 4."), sentence
    assert "2" in sentence and "real miss" in sentence.lower(), sentence
    assert "3" in sentence and "extended" in sentence, sentence


def test_the_sentence_is_deterministic():
    assert _four_vetoes().sentences["rejected"] == _four_vetoes().sentences["rejected"]


def test_overlapping_veto_codes_are_never_summed():
    """Three `extended` and one `sma_incoming` is three, never four or five."""
    sentence = _four_vetoes().sentences["rejected"]

    assert "4 share" not in sentence, sentence
    assert "5 share" not in sentence, sentence


def test_every_population_has_a_sentence_even_when_it_is_empty():
    day = _build(decisions=[])

    for name in ("liked_not_traded", "rejected", "traded_left_early", "claimed_d1",
                 "earlier_calls"):
        assert isinstance(day.sentences[name], str) and day.sentences[name].strip(), name


# -- item 6: instrument-aware rows -------------------------------------------


def _closed_trade(**overrides) -> dict:
    trade = {
        "trade_id": "T1",
        "symbol": "AAA",
        "direction": "LONG",
        "status": "closed",
        "security_type": "STK",
        "opened_at": "2026-09-18T09:40:00-04:00",
        "closed_at": "2026-09-18T10:00:00-04:00",
        "last_closing_leg_at": "2026-09-18T10:00:00-04:00",
        "net_pnl": 120.0,
    }
    trade.update(overrides)
    return trade


def _trade_bars():
    return {SESSION: {"AAA": [
        _m5(10, 5, open_=100.0, high=110.0, low=99.0, close=109.0),
        _m5(15, 55, open_=109.0, high=109.0, low=105.0, close=106.0),
    ]}}


def test_a_stock_trade_closed_in_the_session_still_gets_left_on_the_table():
    day = _build(trades=[_closed_trade()], bars=_trade_bars(),
                 daily_bars={"AAA": _flat_daily()})

    row = day.traded_left_early[0]
    assert row.left_on_table_pct == pytest.approx(10.0)
    assert row.not_judged_reason in (None, "")


def test_an_option_trade_never_gets_a_left_on_the_table_number():
    """`journal_exposure`'s rule: a long option is never a bullish setup."""
    day = _build(
        trades=[_closed_trade(security_type="OPT", symbol="AAA")],
        bars=_trade_bars(),
        daily_bars={"AAA": _flat_daily()},
    )

    row = day.traded_left_early[0]
    assert row.left_on_table_pct is None
    assert row.instrument == "OPT"
    assert row.not_judged_reason and "option" in row.not_judged_reason.lower()
    assert row.you_made == pytest.approx(120.0), "premium kept is still reported"
    assert row.sessions_held == 0


def test_a_position_held_past_five_sessions_is_not_judged_by_todays_tape():
    """Opened 09-04, closed 09-18: nine sessions, Labor Day not counted."""
    day = _build(
        trades=[_closed_trade(opened_at="2026-09-04T09:40:00-04:00")],
        bars=_trade_bars(),
        daily_bars={"AAA": _flat_daily()},
    )

    row = day.traded_left_early[0]
    assert row.left_on_table_pct is None
    assert row.not_judged_reason and "held" in row.not_judged_reason.lower()
    assert row.sessions_held == 9


def test_a_row_the_desk_cannot_judge_says_so_and_never_prints_a_zero():
    day = _build(
        trades=[_closed_trade(security_type="OPT")],
        bars={},
        daily_bars={},
    )

    row = day.traded_left_early[0]
    assert row.left_on_table_pct is None
    assert row.not_judged_reason


# -- item 8: every money line carries its n ----------------------------------


def test_the_money_line_carries_its_n_and_says_too_few_to_call_under_the_floor():
    from evidence_stats import MIN_REPORTABLE_N

    day = _build(trades=[_closed_trade()], bars=_trade_bars(),
                 daily_bars={"AAA": _flat_daily()})

    assert day.money["n"] == 1
    assert day.money["n"] < MIN_REPORTABLE_N
    assert "too few to call" in day.money["line"].lower()


def test_the_money_line_reports_its_n_above_the_floor():
    trades = [
        _closed_trade(trade_id=f"T{i}", symbol=f"S{i:02d}", net_pnl=10.0)
        for i in range(30)
    ]
    day = _build(trades=trades, bars={SESSION: {}}, daily_bars={})

    assert day.money["n"] == 30
    assert "too few to call" not in day.money["line"].lower()
    assert "30" in day.money["line"]


# -- item 7: the session stamp, read forward, files untouched ----------------


def test_a_saturday_stamped_decision_is_read_as_the_next_sessions():
    """Friday 21:04 Pacific carries `session_date` 2026-09-19, a Saturday."""
    saturday = _d1_veto(symbol="EEE") | {
        "session_date": "2026-09-19",
        "stamp": "2026-09-18T21:04:28-07:00",
    }
    day = _build(
        session="2026-09-21",
        decisions=[saturday],
        daily_bars={"EEE": _flat_daily("2026-09-21")},
        now=datetime(2026, 9, 21, 8, 0, tzinfo=EASTERN),
    )

    assert [row.symbol for row in day.rejected] == ["EEE"]


def test_a_saturday_stamped_row_never_lands_on_the_friday_before_it():
    saturday = _d1_veto(symbol="EEE") | {
        "session_date": "2026-09-19",
        "stamp": "2026-09-18T21:04:28-07:00",
    }
    day = _build(session=SESSION, decisions=[saturday], daily_bars={"EEE": _flat_daily()})

    assert day.rejected == ()


def test_reading_a_saturday_row_forward_leaves_the_annotation_file_byte_identical(tmp_path):
    """Existing rows are NEVER rewritten - the reader maps, the file does not move.

    AMENDED by the lead, 2026-09-19 (reviewer NO-GO, blocker 1): this test used
    to ask for the forward mapping through `load_annotations`' DEFAULT, which
    the packet never granted. That default is what `pick_feedback` - and so the
    setups-table hide, the chart-cycling skip and the review queue's "Reviewed
    today" mark - joins on, and moving it hid 12 symbols the base desk showed.
    TJ-11's mapping is now an explicit OPT-IN (`by_decision_session=True`) and
    the default path is byte-identical to base. The byte-identical FILE
    assertion below is the tester's and is unchanged.
    """
    from ui.annotations import store

    target = tmp_path / "trader_annotations.jsonl"
    rows = [
        {"schema_version": 1, "event_id": "a", "event_type": "veto", "symbol": "EEE",
         "side": "SHORT", "session_date": "2026-09-19", "timeframe": "D1",
         "created_at": "2026-09-18T21:04:28.734007-07:00", "source": "chart_review",
         "reason_code": "extended"},
        {"schema_version": 1, "event_id": "b", "event_type": "veto", "symbol": "FFF",
         "side": "LONG", "session_date": "2026-09-21", "timeframe": "D1",
         "created_at": "2026-09-21T13:00:00-04:00", "source": "chart_review",
         "reason_code": "extended"},
    ]
    target.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    before = target.read_bytes()

    found = store.load_annotations(
        target, session_date="2026-09-21", by_decision_session=True
    )

    assert sorted(row["symbol"] for row in found) == ["EEE", "FFF"]
    assert target.read_bytes() == before
    # And the DEFAULT still answers exactly what base answered: the Saturday
    # row is not Monday's for anybody but TJ-11.
    base = store.load_annotations(target, session_date="2026-09-21")
    assert [row["symbol"] for row in base] == ["FFF"]
    assert target.read_bytes() == before


def test_a_new_annotation_written_after_the_close_carries_the_next_sessions_date(monkeypatch):
    """The 18 live rows stamped 2026-09-19 are the defect this closes.

    The wall clock is pinned to a DIFFERENT date so the answer cannot come from
    `get_market_session_window()` being accidentally right on the day the suite
    happens to run: the stamp decides the session, not `now`.

    AMENDED by the lead, 2026-09-19 (reviewer NO-GO, blocker 1): the session a
    decision belongs to is an ADDITIVE field, `decision_session`. `session_date`
    keeps exactly the value and the meaning base writes, because
    `review_learning` (the veto cohort behind `review_policy.json`), the three
    cohort graders, `daily_recap_reader` and `pick_feedback` all join on it by
    exact match. Both halves are asserted here.
    """
    import market_session
    from ui.annotations import store

    monkeypatch.setattr(
        market_session, "get_market_session_window",
        lambda *a, **k: type("_W", (), {"market_date": date(2026, 9, 18)})(),
    )

    row = store.build_annotation(
        store.EVENT_LIKE_CLAIM,
        symbol="EBAY",
        side="LONG",
        like_mode="quick",
        timeframe="D1",
        created_at=datetime(2026, 9, 18, 21, 4, 28, tzinfo=PACIFIC),
    )

    assert row["decision_session"] == "2026-09-21"
    # EXACTLY what base writes, which is the market session WINDOW's date and
    # never the stamp - pinned to 2026-09-18 by the monkeypatch above. The
    # stamp decides `decision_session` and nothing else, so every live reader
    # joining on `session_date` sees what it has always seen.
    assert row["session_date"] == "2026-09-18"
    assert row["created_at"].startswith("2026-09-18T21:04:28")


def test_an_explicit_session_date_is_still_honoured_exactly():
    from ui.annotations import store

    row = store.build_annotation(
        store.EVENT_LIKE_CLAIM, symbol="EBAY", side="LONG", like_mode="quick",
        session_date="2026-09-18",
        created_at=datetime(2026, 9, 18, 21, 4, 28, tzinfo=PACIFIC),
    )

    assert row["session_date"] == "2026-09-18"


# -- the totals still hold ---------------------------------------------------


def test_the_population_totals_still_equal_the_decision_count():
    decisions = [
        _decision(symbol="AAA", verdict="like", stamp="2026-09-18T09:35:00-04:00"),
        _decision(symbol="BBB", verdict="veto", stamp="2026-09-18T09:36:00-04:00"),
        _decision(symbol="CCC", verdict="pass", stamp="2026-09-18T09:37:00-04:00"),
        _decision(symbol="DDD", verdict="like", timeframe="D1",
                  stamp="2026-09-18T09:38:00-04:00"),
    ]
    day = _build(
        decisions=decisions,
        bars={name: [_m5(9, 45, open_=100.0, high=101.0, low=99.0, close=100.5)]
              for name in ("AAA", "BBB", "CCC", "DDD")},
        daily_bars={name: _flat_daily() for name in ("AAA", "BBB", "CCC", "DDD")},
        scan_rows=_scan_rows(["AAA", "BBB", "CCC", "DDD", "EEE"]),
        now=datetime(2026, 9, 28, 8, 0),
    )

    assert len(_all_rows(day)) == len(decisions)
    skill_n = sum(cell["n"] for cell in day.skill["session"]["cells"]
                  if cell.get("setup_family", "") == "" and cell["side"] == "LONG")
    assert skill_n == 5, "the skill line partitions the SCAN, not the decisions"
