"""p9 long setups: the leader pullback and the post-earnings drift (`long_setups`).

Per rule: it fires, it does not, it is point in time, and the market gate holds a row
back without hiding it. The trader, 2026-09-26: "Give them their own setup ... The bot
should really promote these."
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import long_setups as ls  # noqa: E402


def _days(count: int) -> list[str]:
    out, day = [], date(2025, 1, 2)
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day.isoformat())
        day += timedelta(days=1)
    return out


def _bars(closes, volumes=None, *, spread=0.5):
    days = _days(len(closes))
    volumes = volumes or [1_000_000] * len(closes)
    return [{"date": d, "open": c, "high": c + spread, "low": c - spread, "close": c, "volume": v}
            for d, c, v in zip(days, closes, volumes, strict=True)]


def _leader(pullback_sessions=10, step=0.012, pullback_volume=600_000, run_bars=290):
    closes = [50.0 + 0.25 * i for i in range(run_bars)]
    peak = closes[-1]
    closes += [peak * (1 - step * k) for k in range(1, pullback_sessions + 1)]
    volumes = [1_000_000] * run_bars + [pullback_volume] * pullback_sessions
    return _bars(closes, volumes)


# --- leader pullback

def test_leader_pullback_fires_with_entry_stop_and_exit():
    bars = _leader()
    row = ls.leader_pullback(bars, atr=2.0)
    assert row is not None and row["setup"] == ls.LEADER_PULLBACK
    close = bars[-1]["close"]
    assert row["entry_limit"] == round(close - 0.25 * 2.0, 2)
    assert row["target"] == round(close - 0.5 + 2.0, 2)
    assert row["stop"] < row["entry_limit"]
    assert row["time_exit_sessions"] == 10
    assert "take +1 ATR" in row["exit"] and "10 sessions" in row["exit"]
    assert any("52-week high" in reason for reason in row["reasons"])
    assert row["leader"] is False


def test_leader_bonus_raises_strength():
    bars = _leader()
    plain = ls.leader_pullback(bars, atr=2.0)
    leader = ls.leader_pullback(bars, atr=2.0, sector_top_third=True, top_pattern=True)
    assert leader["leader"] is True
    assert leader["strength"] == plain["strength"] + 2


@pytest.mark.parametrize("bars", [
    _leader(pullback_volume=1_500_000),          # the pullback is heavier than the run
    _leader(pullback_sessions=2, step=0.01),     # only 2-3% off the high
    _leader(pullback_sessions=10, step=0.03),    # ~30% off the high: too deep
])
def test_leader_pullback_does_not_fire(bars):
    assert ls.leader_pullback(bars, atr=2.0) is None


def test_leader_pullback_needs_the_200_day():
    bars = _leader()
    # A crash long ago puts the 200-day above today's close.
    crashed = [dict(bar) for bar in bars]
    for bar in crashed[-200:-40]:
        for key in ("open", "high", "low", "close"):
            bar[key] += 60.0
    assert ls.leader_pullback(crashed, atr=2.0) is None


def test_strong_by_rs_alone_when_the_52_week_high_is_unknown():
    # 210 slow bars: no 252-session window (52w unknown) and no 30% run.
    closes = [100.0 * (1.002 ** i) for i in range(200)]
    peak = closes[-1]
    closes += [peak * (1 - 0.011 * k) for k in range(1, 11)]
    bars = _bars(closes, [1_000_000] * 200 + [500_000] * 10, spread=0.2)
    assert ls.made_52w_high(bars) is None
    assert ls.leader_pullback(bars, atr=1.0) is None
    row = ls.leader_pullback(bars, atr=1.0, rs_percentile=0.95)
    assert row is not None and any("top 10%" in reason for reason in row["reasons"])


def test_missing_data_is_no_setup():
    bars = _leader()
    holed = [dict(bar) for bar in bars]
    holed[-3]["close"] = None
    assert ls.leader_pullback(holed, atr=2.0) is None
    no_volume = [dict(bar) for bar in bars]
    no_volume[-2]["volume"] = None
    assert ls.leader_pullback(no_volume, atr=2.0) is None
    assert ls.leader_pullback([], atr=2.0) is None


def test_leader_pullback_is_point_in_time():
    bars = _leader()
    today = ls.leader_pullback(bars, atr=2.0)
    # A later bar that rips back to the high is not read by the earlier session's answer.
    later = bars + [{**bars[-1], "date": "2099-01-02", "close": 130.0, "high": 131.0}]
    assert ls.leader_pullback(later[:-1], atr=2.0) == today


# --- post-earnings drift

def _earnings(after=5, gap_open=110.0, gap_close=114.0, gap_low=109.0, gap_high=115.0, hold=113.5):
    closes = [100.0] * 60
    bars = _bars(closes)
    days = _days(60 + 1 + after)
    bars.append({"date": days[60], "open": gap_open, "high": gap_high, "low": gap_low, "close": gap_close,
                 "volume": 3_000_000})
    for k in range(after):
        bars.append({"date": days[61 + k], "open": hold, "high": hold + 1, "low": hold - 1, "close": hold,
                     "volume": 1_000_000})
    return bars, days[60]


def test_post_earnings_drift_fires():
    bars, gap_day = _earnings()
    row = ls.post_earnings_drift(bars, gap_date=gap_day, gap_is_up=True, gap_atr_multiple=2.5, atr=3.0)
    assert row is not None and row["setup"] == ls.POST_EARNINGS_DRIFT
    assert row["sessions_after_gap"] == 5
    assert row["entry_limit"] == round(113.5 - 0.75, 2)
    assert row["stop"] == round(109.0 - 0.01, 2) and row["stop_basis"] == "under the gap-day low"


def test_the_stop_is_capped_at_one_and_a_half_atr():
    bars, gap_day = _earnings()
    row = ls.post_earnings_drift(bars, gap_date=gap_day, gap_is_up=True, gap_atr_multiple=2.5, atr=2.0)
    # The gap-day low is 4 points under a 113.00 entry; 1.5 ATR is 3.
    assert row["entry_limit"] == 113.0
    assert row["stop"] == 110.0 and row["stop_basis"] == "1.5 ATR under the entry"


@pytest.mark.parametrize("kwargs, earnings", [
    ({"gap_is_up": False, "gap_atr_multiple": 2.5}, {}),          # a gap down
    ({"gap_is_up": True, "gap_atr_multiple": 0.8}, {}),           # under 1 ATR
    ({"gap_is_up": True, "gap_atr_multiple": 2.5}, {"gap_close": 110.0}),  # closed in the lower half
    ({"gap_is_up": True, "gap_atr_multiple": 2.5}, {"after": 3}),  # too soon
    ({"gap_is_up": True, "gap_atr_multiple": 2.5}, {"after": 8}),  # too late
    ({"gap_is_up": True, "gap_atr_multiple": 2.5}, {"hold": 108.5}),  # lost the gap-day low
    ({"gap_is_up": None, "gap_atr_multiple": 2.5}, {}),           # unknown direction
])
def test_post_earnings_drift_does_not_fire(kwargs, earnings):
    bars, gap_day = _earnings(**earnings)
    assert ls.post_earnings_drift(bars, gap_date=gap_day, atr=2.0, **kwargs) is None


def test_post_earnings_drift_is_point_in_time():
    bars, gap_day = _earnings(after=7)
    # On session 3 after the gap it is too early; the same bars through session 4 fire.
    assert ls.post_earnings_drift(bars[:-4], gap_date=gap_day, gap_is_up=True, gap_atr_multiple=2.5, atr=2.0) is None
    assert ls.post_earnings_drift(bars[:-3], gap_date=gap_day, gap_is_up=True, gap_atr_multiple=2.5, atr=2.0)


# --- one scan: the market gate, ranking, staleness

def _scan(working="yes", **row):
    bars = _leader()
    feature = {"symbol": "LEAD", "side": "LONG", "perm_regime_working": working,
               "perm_regime_working_rule": "trader", **row}
    return ls.build_rows(bars_by_symbol={"LEAD": bars}, spy_bars=bars, feature_rows=[feature],
                         atr_by_symbol={"LEAD": 2.0}, as_of=bars[-1]["date"])


def test_market_working_promotes_the_row():
    payload = _scan("yes")
    (row,) = payload["rows"]
    assert row["promoted"] is True and row["status"] == ls.STATUS_READY
    assert payload["market_working"] == "yes"


@pytest.mark.parametrize("working", ["no", "unknown", ""])
def test_market_not_working_keeps_the_row_but_waits(working):
    (row,) = _scan(working)["rows"]
    assert row["promoted"] is False
    assert row["status"] == "waiting for the market"


def test_market_gate_reads_the_trader_regime_first():
    # SPY above a rising 20-day, but the trader says bear: not working.
    rows = [{"perm_regime_trader": "bear", "perm_spy_vs_sma20_pct": 2.0, "perm_spy_sma20_slope_pct": 1.0}]
    assert ls.market_gate(rows) == ("no", "trader")
    rows = [{"perm_regime_trader": "", "perm_spy_vs_sma20_pct": 2.0, "perm_spy_sma20_slope_pct": 1.0}]
    assert ls.market_gate(rows) == ("yes", "spy_above_rising_sma20")


def test_leader_bonus_from_the_scan_rows():
    rows = _scan("yes", setup_family="top_pattern_tracking")["rows"]
    assert rows[0]["leader"] is True
    assert "leader: top-pattern tracking name" in rows[0]["reasons"]


def test_a_stale_name_is_skipped():
    bars = _leader()
    payload = ls.build_rows(bars_by_symbol={"LEAD": bars[:-1]}, spy_bars=bars, feature_rows=[],
                            atr_by_symbol={"LEAD": 2.0}, as_of=bars[-1]["date"])
    assert payload["rows"] == []


def test_rank_is_by_strength():
    rows = [{"symbol": "B", "setup": ls.LEADER_PULLBACK, "strength": 1.0},
            {"symbol": "A", "setup": ls.POST_EARNINGS_DRIFT, "strength": 3.0},
            {"symbol": "C", "setup": ls.LEADER_PULLBACK, "strength": 3.0}]
    assert [row["symbol"] for row in ls.rank(rows)] == ["C", "A", "B"]


# --- grading history

def test_settle_uses_the_limit_fill_and_waits_for_the_target_session():
    bars = _bars([100.0] * 10)
    history = [{"symbol": "X", "as_of": bars[2]["date"], "entry_limit": 99.8}]
    # Session 1's low (99.5) touches 99.8: filled at the limit.
    settled = ls.settle(history, {"X": bars}, bars)
    assert settled[0]["outcome"] == "filled" and settled[0]["fill"] == 99.8
    assert settled[0]["target_session"] == bars[7]["date"]
    assert settled[0]["return_pct"] == pytest.approx((100.0 / 99.8 - 1) * 100, abs=1e-3)
    # Not enough completed bars after the scan session: left unknown.
    unsettled = ls.settle(history, {"X": bars[:7]}, bars)
    assert "outcome" not in unsettled[0]
    no_fill = ls.settle([{"symbol": "X", "as_of": bars[2]["date"], "entry_limit": 90.0}], {"X": bars}, bars)
    assert no_fill[0]["outcome"] == "no_fill" and "return_pct" not in no_fill[0]


def test_upsert_replaces_the_same_session():
    old = [{"symbol": "A", "as_of": "2026-09-24"}, {"symbol": "B", "as_of": "2026-09-25"}]
    new = [{"symbol": "C", "as_of": "2026-09-25", "setup": ls.LEADER_PULLBACK}]
    merged = ls.upsert_history(old, new)
    assert [row["symbol"] for row in merged] == ["A", "C"]


# --- the words and the Focus candidates

def test_phone_line_only_for_promoted_rows():
    assert ls.phone_line(_scan("no")) == ""
    line = ls.phone_line(_scan("yes"))
    assert line.startswith("Long leaders ") and "LEAD (leader pullback, limit" in line


def test_tracker_lines_head_and_rows():
    lines = ls.tracker_lines(_scan("no"))
    assert "waiting for the market" in lines[1]
    assert lines[0].startswith("Long leaders (scan session")
    assert ls.tracker_lines(None) == ["Long leaders: no scan has published long setups yet."]


def test_focus_candidates_are_promoted_and_fresh():
    payload = _scan("yes")
    as_of = date.fromisoformat(payload["as_of"])
    got = ls.focus_candidates(payload, today=as_of + timedelta(days=3))
    assert [row["symbol"] for row in got["longs"]] == ["LEAD"] and got["shorts"] == []
    assert got["longs"][0]["score"] > ls.FOCUS_SCORE_BASE
    assert ls.focus_candidates(payload, today=as_of + timedelta(days=5))["longs"] == []
    assert ls.focus_candidates(_scan("no"), today=as_of)["longs"] == []


def test_runner_bars_drop_a_forming_bar():
    from master_avwap_lib import runner

    frame = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-09-24", "2026-09-25", "2026-09-26"]),
        "open": [1.0, 2.0, 3.0], "high": [1.5, 2.5, 3.5], "low": [0.5, 1.5, 2.5],
        "close": [1.2, 2.2, 3.2], "volume": [10, 20, 30],
    })
    bars = runner._long_setup_bars(frame, "2026-09-25")
    assert [bar["date"] for bar in bars] == ["2026-09-24", "2026-09-25"]
    assert runner._long_setup_bars(frame, None) == []
