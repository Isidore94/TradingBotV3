"""Movers board pure model (`scripts/movers_scan.py`): hand-built bar fixtures.

Bars are naive MARKET-LOCAL time in Los Angeles (06:30 local = 09:30 NY), the
same shape `bot.m5_chart_bars` hands out, so every test also proves the naive
-> New York conversion.
"""

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

import movers_scan as ms  # noqa: E402

LA = ZoneInfo("America/Los_Angeles")
NY = ZoneInfo("America/New_York")
PRIOR = date(2026, 9, 21)
TODAY = date(2026, 9, 22)


def _bars(day, closes, *, volume=100_000.0, volumes=None, rng=1.0, first_open=None):
    """Consecutive M5 bars from 06:30 LA; open = previous close, range +/- rng/2."""
    out = []
    previous = closes[0] if first_open is None else first_open
    start = datetime(day.year, day.month, day.day, 6, 30)
    for index, close in enumerate(closes):
        open_ = previous
        out.append({
            "dt": start + timedelta(minutes=5 * index),
            "open": open_,
            "high": max(open_, close) + rng / 2,
            "low": min(open_, close) - rng / 2,
            "close": close,
            "volume": volumes[index] if volumes is not None else volume,
        })
        previous = close
    return out


def _now(bar_count, extra_seconds=0):
    """Aware LA 'now' right after `bar_count` bars of TODAY have completed."""
    start = datetime(TODAY.year, TODAY.month, TODAY.day, 6, 30, tzinfo=LA)
    return start + timedelta(minutes=5 * bar_count, seconds=extra_seconds)


def _series(today_closes, *, prior_close=100.0, today_volumes=None):
    prior = _bars(PRIOR, [prior_close] * 78)
    return prior + _bars(TODAY, today_closes, volumes=today_volumes, first_open=prior_close)


FLAT_BASELINE = {offset: 100_000.0 for offset in range(78)}


def _flat_spy(n=20):
    return _series([400.0] * n, prior_close=400.0)


def _board(symbols, *, spy=None, n=20, baselines=None, focus=None, now=None, earnings=None):
    return ms.build_movers_board(
        symbols,
        _flat_spy(n) if spy is None else spy,
        now=now or _now(n),
        baselines=baselines,
        focus_by_side=focus,
        local_tz=LA,
        earnings=earnings,
    )


def _pop(closes_tail, n=20, base=100.0):
    return [base] * (n - len(closes_tail)) + closes_tail


# ------------------------------------------------------------------ pop ranking
def test_pop_ranks_by_atr_move_weighted_by_rvol():
    n = 20
    loud = [100_000.0] * (n - 3) + [300_000.0] * 3
    symbols = {
        "AAA": _series(_pop([101.0, 102.0, 103.0])),  # +3 over 3 bars, RVOL 1
        "BBB": _series(_pop([100.33, 100.67, 101.0])),  # +1, RVOL 1
        "CCC": _series(_pop([100.33, 100.67, 101.0]), today_volumes=loud),  # +1, RVOL 3
        "DDD": _series(_pop([99.0, 98.0, 97.0])),  # -3
    }
    baselines = {sym: FLAT_BASELINE for sym in symbols}
    board = _board(symbols, baselines=baselines)
    longs = [row["symbol"] for row in board["pop"]["long"]]
    assert longs == ["AAA", "CCC", "BBB"]
    assert [row["symbol"] for row in board["pop"]["short"]] == ["DDD"]
    ccc = next(row for row in board["pop"]["long"] if row["symbol"] == "CCC")
    assert ccc["rvol"] == pytest.approx(3.0)
    assert ccc["move15_pct"] == pytest.approx(1.0)
    assert ccc["day_pct"] == pytest.approx(1.0)


def test_rvol_none_is_shown_blank_and_never_gets_a_bonus():
    n = 20
    loud = [100_000.0] * (n - 3) + [300_000.0] * 3
    same_move = _pop([100.33, 100.67, 101.0])
    symbols = {
        "LOUD": _series(same_move, today_volumes=loud),
        "NOBASE": _series(same_move, today_volumes=loud),  # loud, but unmeasurable
    }
    board = _board(symbols, baselines={"LOUD": FLAT_BASELINE})
    rows = {row["symbol"]: row for row in board["pop"]["long"]}
    assert rows["NOBASE"]["rvol"] is None
    assert [row["symbol"] for row in board["pop"]["long"]] == ["LOUD", "NOBASE"]
    assert rows["NOBASE"]["pop_score"] < rows["LOUD"]["pop_score"]


def test_price_and_session_volume_floors_keep_a_name_off_the_ranked_lists():
    symbols = {
        "CHEAP": _series(_pop([4.33, 4.67, 5.0], base=4.0), prior_close=4.0),
        "THIN": _series(_pop([100.33, 100.67, 101.0]), today_volumes=[100.0] * 20),
        "OK": _series(_pop([100.33, 100.67, 101.0])),
    }
    board = _board(symbols, baselines={s: FLAT_BASELINE for s in symbols})
    assert [row["symbol"] for row in board["pop"]["long"]] == ["OK"]


def test_forming_bar_is_excluded():
    n = 20
    closes = _pop([100.33, 100.67, 101.0], n=n)
    forming = _series(closes + [150.0])  # a 21st bar still forming
    spy = _series([400.0] * (n + 1), prior_close=400.0)
    now = _now(n, extra_seconds=120)  # 2 minutes into bar 21
    with_forming = _board({"XYZ": forming}, spy=spy, now=now)
    without = _board({"XYZ": _series(closes)}, spy=_flat_spy(n), now=now)
    assert with_forming["pop"] == without["pop"]
    assert with_forming["pop"]["long"][0]["last"] == pytest.approx(101.0)


def test_naive_market_local_bars_match_aware_new_york_bars():
    closes = _pop([100.33, 100.67, 101.0])
    naive = _series(closes)
    aware = [
        dict(bar, dt=bar["dt"].replace(tzinfo=LA).astimezone(NY)) for bar in naive
    ]
    a = _board({"XYZ": naive})
    b = _board({"XYZ": aware})
    assert a["pop"] == b["pop"]
    assert a["pop"]["long"][0]["move15_pct"] == pytest.approx(1.0)


def test_stale_series_is_not_ranked():
    n = 20
    fresh = _series(_pop([100.33, 100.67, 101.0], n=n))
    stale = _series(_pop([101.0, 102.0, 103.0], n=n - 2))  # ends 2 bars earlier
    board = _board({"FRESH": fresh, "STALE": stale})
    assert [row["symbol"] for row in board["pop"]["long"]] == ["FRESH"]


def test_yesterdays_bars_are_not_today():
    """A series that stopped yesterday has no bars today (never yesterday's move)."""
    only_prior = _bars(PRIOR, _pop([100.33, 100.67, 101.0], n=78))
    spy_prior = _bars(PRIOR, [400.0] * 78)
    board = _board({"OLD": only_prior}, spy=spy_prior, focus={"long": ["OLD"]})
    assert board["state"]["state"] == "unknown"
    assert board["pop"] == {"long": [], "short": []}
    assert board["mine"]["long"][0]["note"] == "no bars today"


# ------------------------------------------------------------------ market state
def _spy_pullback(drop_pct, *, high_index=9):
    """SPY: heavy-volume open at 400, rally to 405 by `high_index`, then fall."""
    closes = [400.0, 400.0] + [405.0] * (high_index - 1)
    closes[high_index] = 405.0
    peak = 405.0
    target = peak * (1 - drop_pct / 100.0)
    closes += [peak - (peak - target) * k / 3 for k in (1, 2, 3)]
    volumes = [1_000_000.0, 1_000_000.0] + [100_000.0] * (len(closes) - 2)
    bars = _series(closes, prior_close=400.0, today_volumes=volumes)
    # Make the high unambiguous: only `high_index` pokes above.
    today = [b for b in bars if b["dt"].date() == TODAY]
    for index, bar in enumerate(today):
        bar["high"] = max(bar["open"], bar["close"]) + (0.6 if index == high_index else 0.05)
        bar["low"] = min(bar["open"], bar["close"]) - 0.05
    return bars, len(closes)


def _state(spy_bars, n):
    spy = ms.normalize_bars(spy_bars, now=_now(n), local_tz=LA)
    return ms.market_state(spy)


def test_pullback_on_at_or_beyond_threshold_and_records_start():
    bars, n = _spy_pullback(0.45)
    state = _state(bars, n)
    assert state.state == "up_day"
    assert state.pullback is True
    assert state.extreme_time == "10:15"  # bar 9 after 09:30 NY
    assert state.spy_from_extreme_pct <= -ms.PULLBACK_MIN_PCT


def test_pullback_off_below_threshold():
    bars, n = _spy_pullback(0.10)
    state = _state(bars, n)
    assert state.state == "up_day"
    assert state.pullback is False
    assert state.start_dt is None


def test_pullback_off_when_high_is_in_the_first_two_bars():
    closes = [400.0, 406.0, 405.0, 404.0, 403.5, 403.0]
    volumes = [2_000_000.0, 100_000.0, 100_000.0, 100_000.0, 100_000.0, 100_000.0]
    bars = _series(closes, prior_close=400.0, today_volumes=volumes)
    state = _state(bars, len(closes))
    assert state.state == "up_day"
    assert state.pullback is False


def test_missing_spy_is_unknown_and_lights_nothing():
    closes = _pop([100.33, 100.67, 101.0])
    board = _board({"XYZ": _series(closes)}, spy=[])
    assert board["state"]["state"] == "unknown"
    assert board["state"]["pullback"] is False
    assert board["dip"] == {"long": [], "short": []}
    assert board["pop"]["long"][0]["vs_spy15_pct"] is None


def test_bounce_mirror_on_a_down_day():
    closes = [400.0, 400.0] + [395.0] * 8
    closes += [395.0 + 395.0 * 0.004 * k / 3 for k in (1, 2, 3)]
    volumes = [1_000_000.0, 1_000_000.0] + [100_000.0] * (len(closes) - 2)
    bars = _series(closes, prior_close=400.0, today_volumes=volumes)
    today = [b for b in bars if b["dt"].date() == TODAY]
    for index, bar in enumerate(today):
        bar["low"] = min(bar["open"], bar["close"]) - (0.6 if index == 9 else 0.05)
        bar["high"] = max(bar["open"], bar["close"]) + 0.05
    state = _state(bars, len(closes))
    assert state.state == "down_day"
    assert state.bounce is True and state.pullback is False


# ------------------------------------------------------------------ dip-strong
def test_dip_strong_ranks_excess_vs_spy_and_drops_the_weaker_names():
    spy, n = _spy_pullback(0.45)
    base = [100.0] * 10  # through the start bar (index 9)
    holds = _series(base + [100.2, 100.3, 100.4])  # up while SPY falls
    flat = _series(base + [100.0, 100.0, 100.0])  # flat: still beats SPY
    sinks = _series(base + [99.0, 98.5, 98.0])  # fell harder than SPY
    board = _board(
        {"HOLD": holds, "FLAT": flat, "SINK": sinks},
        spy=spy, n=n, baselines={s: FLAT_BASELINE for s in ("HOLD", "FLAT", "SINK")},
    )
    assert board["state"]["pullback"] is True
    assert [row["symbol"] for row in board["dip"]["long"]] == ["HOLD", "FLAT"]
    hold = board["dip"]["long"][0]
    assert hold["since_start_pct"] == pytest.approx(0.4)
    assert hold["dip_score"] > 0


def test_dip_strong_needs_a_bar_at_the_start():
    spy, n = _spy_pullback(0.45)
    late = _series([100.0] * 10 + [100.2, 100.3, 100.4])
    late = [bar for bar in late if not (bar["dt"].date() == TODAY and bar["dt"].minute == 15
                                        and bar["dt"].hour == 7)]
    board = _board({"LATE": late}, spy=spy, n=n)
    assert board["dip"]["long"] == []


def test_dip_lists_empty_without_a_pullback():
    spy, n = _spy_pullback(0.10)
    board = _board({"HOLD": _series([100.0] * 10 + [100.2, 100.3, 100.4])}, spy=spy, n=n)
    assert board["dip"] == {"long": [], "short": []}


# ------------------------------------------------------------------ RVOL baseline
def test_build_rvol_baseline_is_mean_per_offset_and_missing_is_not_zero():
    history = []
    for k in range(6):
        day = date(2026, 9, 10 + k)
        bars = _bars(day, [100.0] * 4, volumes=[1000.0 * (k + 1)] * 4)
        if k == 0:
            bars = bars[:2]  # an early-ending session: offsets 2-3 missing
        history += bars
    baseline = ms.build_rvol_baseline(history, before=TODAY, local_tz=LA)
    assert baseline[0] == pytest.approx(sum(1000.0 * (k + 1) for k in range(6)) / 6)
    assert baseline[3] == pytest.approx(sum(1000.0 * (k + 1) for k in range(1, 6)) / 5)


def test_build_rvol_baseline_needs_enough_sessions_and_ignores_today():
    history = []
    for k in range(4):
        history += _bars(date(2026, 9, 14 + k), [100.0] * 3)
    history += _bars(TODAY, [100.0] * 3)
    assert ms.build_rvol_baseline(history, before=TODAY, local_tz=LA) is None


def test_recent_rvol_none_when_an_offset_is_missing():
    today = ms.normalize_bars(_bars(TODAY, [100.0] * 4, volume=200.0), now=_now(4), local_tz=LA)
    assert ms.recent_rvol(today, {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0}) == pytest.approx(2.0)
    assert ms.recent_rvol(today, {0: 100.0, 1: 100.0, 3: 100.0}) is None
    assert ms.recent_rvol(today, None) is None
    assert ms.rvol_weight(None) == 1.0


# ------------------------------------------------------------------ my names
def test_my_names_include_unmeasured_focus_and_sort_by_mode():
    symbols = {
        "AAA": _series(_pop([101.0, 102.0, 103.0])),
        "BBB": _series(_pop([100.33, 100.67, 101.0])),
    }
    board = _board(symbols, focus={"long": ["bbb", "AAA", "GONE"], "short": []})
    mine = board["mine"]["long"]
    assert [row["symbol"] for row in mine] == ["BBB", "AAA", "GONE"]
    assert mine[2]["note"] == "no bars" and mine[2]["pop_score"] is None
    ordered = ms.sort_mine(mine, "pop", "long")
    assert [row["symbol"] for row in ordered] == ["AAA", "BBB", "GONE"]


# ------------------------------------------------------------------ freshness vs now
def test_reviewer_repro_bars_end_0955_now_1140_is_unknown_and_nothing_fresh():
    six = _pop([100.33, 100.67, 101.0], n=6)  # last bar starts 09:55 NY
    now = datetime(2026, 9, 22, 11, 40, tzinfo=NY)
    board = ms.build_movers_board(
        {"XYZ": _series(six)}, _series([400.0] * 6, prior_close=400.0),
        now=now, baselines={"XYZ": FLAT_BASELINE}, local_tz=LA,
    )
    assert board["state"]["state"] == "unknown"
    assert board["state"]["reason"] == "SPY bars stale"
    assert board["pop"] == {"long": [], "short": []}
    assert board["fresh"] == 0 and board["offered"] == 1
    assert board["as_of_stale"] is True
    assert board["as_of"].startswith("2026-09-22T09:55:00")


def test_fresh_symbols_with_stale_spy_light_nothing():
    n = 20
    spy_old = _series([400.0] * (n - 3), prior_close=400.0)
    board = _board({"AAA": _series(_pop([101.0, 102.0, 103.0], n=n))}, spy=spy_old, n=n)
    assert board["state"]["state"] == "unknown"
    assert board["fresh"] == 1  # the symbol is fresh; SPY is not


def test_as_of_is_spy_last_completed_bar_and_fresh_count():
    n = 20
    fresh = _series(_pop([100.33, 100.67, 101.0], n=n))
    stale = _series(_pop([101.0, 102.0, 103.0], n=n - 2))
    board = _board({"FRESH": fresh, "STALE": stale}, now=_now(n, extra_seconds=30))
    last_start = datetime(2026, 9, 22, 6, 30, tzinfo=LA) + timedelta(minutes=5 * (n - 1))
    assert datetime.fromisoformat(board["as_of"]) == last_start
    assert board["as_of_stale"] is False
    assert (board["fresh"], board["offered"]) == (1, 2)


def test_freshness_cutoff_is_floor5_minus_two_bars():
    now = datetime(2026, 9, 22, 10, 43, 10, tzinfo=NY)
    assert ms.freshness_cutoff(now) == datetime(2026, 9, 22, 10, 30, tzinfo=NY)


# ------------------------------------------------------------------ B: stretch / level
def test_level_fields_hod_break_and_distance_in_atr():
    n = 20
    board = _board({"AAA": _series(_pop([101.0, 102.0, 103.0], n=n))}, n=n)
    row = board["pop"]["long"][0]
    assert row["hod_break"] is True and row["lod_break"] is False
    assert row["from_hod_atr"] == pytest.approx((103.0 - 103.5) / row["atr"])
    assert row["from_vwap_atr"] > 0
    assert row["prev_high"] == pytest.approx(100.5) and row["prev_low"] == pytest.approx(99.5)
    assert row["session_vwap"] is not None


def test_ext_tag_past_the_named_atr_limit_and_ranking_unchanged():
    n = 20
    far = _series(_pop([104.0, 108.0, 112.0], n=n))  # far above VWAP
    near = _series(_pop([100.33, 100.67, 101.0], n=n))
    board = _board({"FAR": far, "NEAR": near}, n=n)
    rows = {r["symbol"]: r for r in board["pop"]["long"]}
    assert rows["FAR"]["ext_up"] is True and rows["FAR"]["from_vwap_atr"] > ms.EXT_ATR
    assert rows["NEAR"]["ext_up"] is False
    assert [r["symbol"] for r in board["pop"]["long"]] == ["FAR", "NEAR"]  # info only


def test_hod_break_false_when_last_bar_is_below_the_session_high():
    n = 20
    board = _board({"AAA": _series(_pop([102.0, 103.0, 102.6], n=n))}, n=n)
    row = board["pop"]["long"][0]
    assert row["hod_break"] is False and row["from_hod_atr"] < 0


# ------------------------------------------------------------------ C: persistence
def test_persistence_counts_ticks_and_rank_changes_and_resets_per_session():
    def board(order):
        return {"pop": {"long": [{"symbol": s} for s in order], "short": []}, "dip": {}}

    memory: dict = {}
    first = board(["AAA", "BBB"])
    memory = ms.apply_persistence(first, memory, session=TODAY)
    assert [(r["streak"], r["rank_change"]) for r in first["pop"]["long"]] == [(1, None), (1, None)]
    second = board(["BBB", "AAA", "CCC"])
    memory = ms.apply_persistence(second, memory, session=TODAY)
    rows = {r["symbol"]: r for r in second["pop"]["long"]}
    assert (rows["BBB"]["streak"], rows["BBB"]["rank_change"]) == (2, 1)
    assert (rows["AAA"]["streak"], rows["AAA"]["rank_change"]) == (2, -1)
    assert (rows["CCC"]["streak"], rows["CCC"]["rank_change"]) == (1, None)
    third = board(["BBB"])
    ms.apply_persistence(third, memory, session=date(2026, 9, 23))
    assert third["pop"]["long"][0]["streak"] == 1  # new session starts over


# ------------------------------------------------------------------ D: group tag
def test_group_tag_needs_three_in_the_top_list():
    board = {"pop": {"long": [{"symbol": s} for s in ("NVDA", "AMD", "MU", "AAPL", "XOM")],
                     "short": [{"symbol": s} for s in ("NVDA", "AMD")]},
             "dip": {"long": [], "short": []}}
    industry = {"NVDA": "Semiconductors", "AMD": "Semiconductors", "MU": "Semiconductors",
                "AAPL": "Consumer Electronics"}
    ms.apply_group_tags(board, industry)
    rows = {r["symbol"]: r for r in board["pop"]["long"]}
    assert rows["NVDA"]["group"] == ms.short_group("Semiconductors")
    assert rows["AAPL"]["group"] == "" and rows["XOM"]["group"] == ""
    assert board["groups"]["pop"]["long"] == [[ms.short_group("Semiconductors"), 3]]
    assert board["groups"]["pop"]["short"] == []  # only two


# ------------------------------------------------------------------ E: earnings tag
def test_earnings_flags_today_bmo_and_yesterday_amc_only():
    events = [
        {"ticker": "AAA", "earnings_date": "2026-09-22", "release_session": "BMO"},
        {"ticker": "BBB", "earnings_date": "2026-09-21", "release_session": "AMC"},
        {"ticker": "CCC", "earnings_date": "2026-09-21", "release_session": "BMO"},
        {"ticker": "DDD", "earnings_date": "2026-09-22", "release_session": "AMC"},
        {"ticker": "EEE", "earnings_date": "2026-09-22", "release_session": "TBD"},
    ]
    flags = ms.earnings_symbols(events, today=TODAY, previous=PRIOR)
    assert flags == {"AAA", "BBB", "DDD", "EEE"}


def test_board_marks_er_rows():
    board = _board({"AAA": _series(_pop([101.0, 102.0, 103.0]))}, earnings={"AAA"})
    assert board["pop"]["long"][0]["er"] is True


# ------------------------------------------------------------------ pullback below VWAP
def test_pullback_stays_on_when_spy_dips_below_vwap_above_the_open():
    closes = [400.0, 401.0] + [405.0] * 8 + [404.4, 403.8, 403.2]
    volumes = [100_000.0, 100_000.0] + [1_000_000.0] * 8 + [100_000.0] * 3
    bars = _series(closes, prior_close=400.0, today_volumes=volumes)
    today = [b for b in bars if b["dt"].date() == TODAY]
    for index, bar in enumerate(today):
        bar["high"] = max(bar["open"], bar["close"]) + (0.6 if index == 9 else 0.05)
        bar["low"] = min(bar["open"], bar["close"]) - 0.05
    state = _state(bars, len(closes))
    assert state.spy_last < state.spy_vwap  # below VWAP now
    assert state.spy_last > 400.0  # still above the open
    assert state.state == "up_day" and state.pullback is True
