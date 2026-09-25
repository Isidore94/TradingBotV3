"""Journal page stats (overhaul 2026-09-23): stat cards, long vs short, time
breakdowns and the calendar's per-day numbers, on fixed trades.

Every expected number here is worked out by hand from the rows below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_analytics as ja  # noqa: E402


def _trade(tid, pnl, *, opened, closed="", direction="LONG", status="CLOSED", **extra):
    row = {
        "trade_id": tid,
        "direction": direction,
        "status": status,
        "opened_at": opened,
        "closed_at": closed,
        "trade_date": (closed or opened)[:10],
        "net_pnl": pnl,
        "currency": "USD",
    }
    row.update(extra)
    return row


# Close order: +100, -50, -30, +200, 0, -80 -> equity 100, 50, 20, 220, 220, 140.
TRADES = [
    _trade("a", 100.0, opened="2026-09-14T09:35:00-04:00", closed="2026-09-14T09:38:00-04:00"),
    _trade("b", -50.0, opened="2026-09-14T10:05:00-04:00", closed="2026-09-14T10:25:00-04:00",
           direction="SHORT"),
    _trade("c", -30.0, opened="2026-09-15T11:00:00-04:00", closed="2026-09-15T12:30:00-04:00"),
    _trade("d", 200.0, opened="2026-09-15T13:00:00-04:00", closed="2026-09-17T10:00:00-04:00",
           direction="SHORT"),
    _trade("e", 0.0, opened="2026-09-17T14:00:00-04:00", closed="2026-09-17T15:30:00-04:00"),
    _trade("f", -80.0, opened="2026-09-01T09:45:00-04:00", closed="2026-09-18T11:00:00-04:00"),
    _trade("g", 999.0, opened="2026-09-18T09:31:00-04:00", status="OPEN"),
]


class TestTradePerformanceStats:
    def test_headline_numbers(self):
        stats = ja.trade_performance_stats(TRADES)
        assert stats["closed"] == 6
        assert stats["wins"] == 2 and stats["losses"] == 3 and stats["breakeven"] == 1
        assert stats["win_rate"] == pytest.approx(2 / 6)
        assert stats["net_pnl"] == pytest.approx(140.0)
        assert stats["profit_factor"] == pytest.approx(300.0 / 160.0)
        assert stats["expectancy"] == pytest.approx(140.0 / 6)
        assert stats["avg_win"] == pytest.approx(150.0)
        assert stats["avg_loss"] == pytest.approx(-160.0 / 3)
        assert stats["largest_win"] == pytest.approx(200.0)
        assert stats["largest_loss"] == pytest.approx(-80.0)

    def test_drawdown_and_streaks_follow_close_order(self):
        stats = ja.trade_performance_stats(TRADES)
        # Peak 100 -> trough 20 is -80; peak 220 -> 140 is also -80.
        assert stats["max_drawdown"] == pytest.approx(-80.0)
        assert stats["max_loss_streak"] == 2
        assert stats["max_win_streak"] == 1
        assert stats["current_streak"] == -1

    def test_a_missing_pnl_is_unpriced_not_zero(self):
        rows = [
            _trade("x", 50.0, opened="2026-09-14T09:35:00-04:00", closed="2026-09-14T09:40:00-04:00"),
            _trade("y", None, opened="2026-09-14T09:45:00-04:00", closed="2026-09-14T09:50:00-04:00"),
        ]
        stats = ja.trade_performance_stats(rows)
        assert stats["closed"] == 1
        assert stats["unpriced"] == 1
        assert stats["win_rate"] == 1.0

    def test_no_closed_trades_is_all_unknown(self):
        stats = ja.trade_performance_stats([TRADES[-1]])
        assert stats["closed"] == 0
        assert stats["net_pnl"] is None
        assert stats["win_rate"] is None
        assert stats["max_drawdown"] is None

    def test_avg_r_uses_the_journals_one_r(self):
        rows = [
            _trade("r1", 100.0, opened="2026-09-14T09:35:00-04:00", closed="2026-09-14T09:40:00-04:00",
                   net_pnl_cad=100.0, planned_risk=50.0),
            _trade("r2", -25.0, opened="2026-09-14T09:45:00-04:00", closed="2026-09-14T09:50:00-04:00",
                   net_pnl_cad=-25.0, planned_risk=50.0),
            _trade("r3", 10.0, opened="2026-09-14T10:45:00-04:00", closed="2026-09-14T10:50:00-04:00"),
        ]
        stats = ja.trade_performance_stats(rows)
        assert stats["avg_r"] == pytest.approx((2.0 - 0.5) / 2)
        assert stats["r_trades"] == 2


def test_r_of_a_usd_trade_divides_the_usd_pnl_by_the_usd_risk():
    """P8-P5: R is net P&L in the trade's currency / planned_risk in that currency."""
    row = _trade("usd", 100.0, opened="2026-09-14T09:35:00-04:00", closed="2026-09-14T09:40:00-04:00",
                 net_pnl_cad=140.0, planned_risk=50.0)
    assert ja.trade_r_multiple(row) == pytest.approx(2.0)
    assert ja.trade_r_multiple({**row, "planned_risk": None}) is None
    assert ja.trade_r_multiple({**row, "net_pnl": None}) is None


def test_long_and_short_side_by_side():
    split = ja.direction_split_stats(TRADES)
    assert set(split) == {"LONG", "SHORT"}
    assert split["LONG"]["closed"] == 4
    assert split["LONG"]["net_pnl"] == pytest.approx(-10.0)
    assert split["SHORT"]["closed"] == 2
    assert split["SHORT"]["net_pnl"] == pytest.approx(150.0)
    assert split["SHORT"]["win_rate"] == pytest.approx(0.5)


class TestTimeBreakdowns:
    def test_weekday_of_entry_in_natural_order(self):
        groups = ja.time_breakdown_groups(TRADES)
        weekday = {row["label"]: row for row in groups["weekday (entry)"]}
        assert [row["label"] for row in groups["weekday (entry)"]] == ["Mon", "Tue", "Thu", "Fri"]
        assert weekday["Mon"]["trades"] == 2  # a, b
        assert weekday["Mon"]["net_pnl"] == pytest.approx(50.0)
        assert weekday["Mon"]["expectancy"] == pytest.approx(25.0)
        assert weekday["Tue"]["closed"] == 3  # c, d, f (f opened Tue Sep 1)

    def test_hour_of_entry_is_market_time(self):
        rows = [
            _trade("utc", 10.0, opened="2026-09-14T13:40:00+00:00", closed="2026-09-14T13:50:00+00:00"),
        ]
        groups = ja.time_breakdown_groups(rows)
        assert [row["label"] for row in groups["hour of entry"]] == ["09:00-10:00 ET"]

    def test_hold_time_buckets(self):
        labels = {row["trade_id"]: ja.hold_time_bucket(row) for row in TRADES}
        assert labels == {
            "a": "under 5 min",
            "b": "5-30 min",
            "c": "30 min-2 h",
            "d": "overnight, 1-5 days",
            "e": "30 min-2 h",
            "f": "over 5 days",
            "g": None,
        }
        order = [row["label"] for row in ja.time_breakdown_groups(TRADES)["hold time"]]
        assert order == ["under 5 min", "5-30 min", "30 min-2 h", "overnight, 1-5 days", "over 5 days"]

    def test_a_refused_total_stays_refused_in_the_buckets(self):
        groups = ja.time_breakdown_groups(TRADES, "")
        assert all(row["net_pnl"] is None for rows in groups.values() for row in rows)


def test_calendar_day_stats_counts_trades_and_outcomes():
    days = ja.calendar_day_stats(TRADES)
    assert days["2026-09-14"] == {"net": pytest.approx(50.0), "trades": 2, "wins": 1, "losses": 1}
    assert days["2026-09-17"]["trades"] == 2  # d closes, e is breakeven
    assert days["2026-09-17"]["wins"] == 1 and days["2026-09-17"]["losses"] == 0
    assert "2026-09-18" in days and days["2026-09-18"]["trades"] == 1  # the open trade is not
    # Same days and totals as the older calendar function.
    legacy = ja.calendar_pnl_by_day(TRADES)
    assert {day: entry["net"] for day, entry in days.items()} == pytest.approx(legacy)


def test_group_expectancy():
    assert ja.group_expectancy({"net_pnl": 90.0, "closed": 3}) == pytest.approx(30.0)
    assert ja.group_expectancy({"net_pnl": None, "closed": 3}) is None
    assert ja.group_expectancy({"net_pnl": 5.0, "closed": 0}) is None


@pytest.mark.parametrize(
    ("mode", "key", "currencies", "expected"),
    [
        ("CAD", "net_pnl_cad", ["USD", "CAD"], "CAD"),
        ("USD", "net_pnl_usd", ["USD", "CAD"], "USD"),
        ("USD", "net_pnl_usd_estimated", ["USD", "CAD"], "USD (estimate)"),
        ("Native", "net_pnl", ["USD"], "USD"),
        ("USD", "net_pnl", ["USD"], "USD"),
        ("CAD", "", ["USD", "CAD"], "no total (mixed currencies)"),
    ],
)
def test_the_currency_label_names_what_the_totals_are_in(mode, key, currencies, expected):
    assert ja.pnl_currency_label(mode, key, currencies) == expected
