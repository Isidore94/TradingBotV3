"""Analytics tab stats (overhaul 2026-09-23): stat cards, Long vs Short, a
breakdown table with avg win / avg loss / expectancy, and time breakdowns.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6")

from ui.models.journal import JournalTrade  # noqa: E402
from ui.panels.journal import analytics_tab as at  # noqa: E402
from ui.services import journal_feed  # noqa: E402


def _row(tid, pnl, *, day, opened, closed, direction="LONG", status="CLOSED", tags=""):
    return {
        "trade_id": tid, "trade_date": day, "direction": direction, "status": status,
        "opened_at": f"{day}T{opened}-04:00", "closed_at": f"{day}T{closed}-04:00" if closed else "",
        "net_pnl": pnl, "net_pnl_cad": pnl, "currency": "CAD", "symbol": "SHOP",
        "setup_tags": tags, "tag_status": "confirmed",
    }


def test_format_stat():
    assert at.format_stat(None, "money") == "-"
    assert at.format_stat(1234.5, "money") == "+1,234.50"
    assert at.format_stat(-5, "money") == "-5.00"
    assert at.format_stat(0.567, "pct") == "57%"
    assert at.format_stat(1.456, "ratio") == "1.46"
    assert at.format_stat(0.5, "r") == "+0.50R"
    assert at.format_stat(7, "count") == "7"


def test_stat_cards_cover_what_online_journals_show():
    keys = [key for key, _title in at.STAT_CARDS]
    for wanted in ("net_pnl", "win_rate", "profit_factor", "expectancy", "avg_win", "avg_loss",
                   "largest_win", "largest_loss", "max_drawdown", "streaks", "closed"):
        assert wanted in keys
    stats = {
        "trades": 5, "closed": 4, "unpriced": 0, "wins": 3, "losses": 1, "breakeven": 0,
        "win_rate": 0.75, "net_pnl": 200.0, "profit_factor": 3.0, "expectancy": 50.0,
        "avg_win": 100.0, "avg_loss": -100.0, "payoff_ratio": 1.0, "largest_win": 150.0,
        "largest_loss": -100.0, "max_drawdown": -100.0, "max_win_streak": 2,
        "max_loss_streak": 1, "current_streak": 2, "avg_r": None, "r_trades": 0,
    }
    values = at.stat_card_values(stats)
    assert values["net_pnl"] == ("+200.00", "", "win")
    assert values["closed"][1] == "3W 1L, 1 open"
    assert values["profit_factor"][2] == "win"
    assert values["largest_loss"][2] == "loss"
    assert values["streaks"][:2] == ("2W / 1L", "now 2 win(s) in a row")
    assert values["avg_r"][0] == "-"


def test_group_rows_carry_avg_win_avg_loss_and_expectancy():
    row = {"label": "bull flag", "trades": 4, "closed": 4, "win_rate": 0.5,
           "profit_factor": 2.0, "avg_win": 100.0, "avg_loss": -50.0, "net_pnl": 100.0}
    assert at.group_table_row(row) == [
        "bull flag", "4", "4", "50%", "2.00", "+100.00", "-50.00", "+25.00", "+100.00",
    ]
    refused = {**row, "net_pnl": None}
    assert at.group_table_row(refused)[5:] == ["-", "-", "-", "-"]


def test_the_curve_steps_in_close_order_within_a_day():
    """trade_id order put a later close first and drew a peak that never happened."""
    rows = [
        _row("a-late", -100.0, day="2026-09-14", opened="09:40:00", closed="15:00:00"),
        _row("z-early", 300.0, day="2026-09-14", opened="09:35:00", closed="10:00:00"),
    ]
    trades = [JournalTrade.from_mapping(row) for row in rows]
    curve = journal_feed.equity_curve(trades, "CAD")
    assert [value for _day, value in curve] == [pytest.approx(300.0), pytest.approx(200.0)]


class _Header:
    currency_mode = "CAD"

    def query(self):
        return {}

    def date_bounds(self):
        return (None, None)


@pytest.fixture
def tab(monkeypatch):
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    base = date.today() - timedelta(days=date.today().weekday() + 7)  # last week's Monday
    mon, tue = base.isoformat(), (base + timedelta(days=1)).isoformat()
    rows = [
        _row("1", 120.0, day=mon, opened="09:35:00", closed="09:37:00", tags="bull flag"),
        _row("2", -40.0, day=mon, opened="10:05:00", closed="10:20:00", direction="SHORT",
             tags="bull flag"),
        _row("3", 60.0, day=tue, opened="13:00:00", closed="14:30:00", direction="SHORT"),
        _row("4", 0.0, day=tue, opened="15:00:00", closed="", status="OPEN"),
    ]
    trades = [JournalTrade.from_mapping(row) for row in rows]
    monkeypatch.setattr(journal_feed, "load_trades", lambda **_kw: trades)
    widget = at.AnalyticsTab(_Header())
    widget.reload()
    yield widget
    widget.deleteLater()


def test_the_cards_and_long_short_table_render(tab):
    assert tab.stat_cards["net_pnl"].value_label.text() == "+140.00"
    assert tab.stat_cards["net_pnl"].property("tone") == "win"
    assert tab.stat_cards["win_rate"].value_label.text() == "67%"
    assert tab.stat_cards["profit_factor"].value_label.text() == "4.50"
    assert tab.stat_cards["expectancy"].value_label.text() == "+46.67"
    assert tab.currency_badge.text() == "Numbers in CAD"
    assert "CAD" in tab.headline.text()
    headers = [tab.side_table.horizontalHeaderItem(i).text() for i in range(2)]
    assert headers == ["Long", "Short"]
    net_row = [label for label, _k, _kind in at.SIDE_ROWS].index("Net P&L")
    assert tab.side_table.item(net_row, 0).text() == "+120.00"
    assert tab.side_table.item(net_row, 1).text() == "+20.00"


def test_time_breakdowns_are_offered_and_the_table_shows_one_group(tab):
    names = [tab.group_picker.itemText(i) for i in range(tab.group_picker.count())]
    for wanted in ("weekday (entry)", "hour of entry", "hold time", "my setups", "provisional setups"):
        assert wanted in names
    tab.group_picker.setCurrentText("weekday (entry)")
    labels = [tab.groups_table.item(r, 0).text() for r in range(tab.groups_table.rowCount())]
    assert labels == ["Mon", "Tue"]
    headers = [tab.groups_table.horizontalHeaderItem(i).text() for i in range(tab.groups_table.columnCount())]
    assert headers == list(at.GROUP_TABLE_COLUMNS)
    tab.group_picker.setCurrentText("my setups")
    labels = {tab.groups_table.item(r, 0).text() for r in range(tab.groups_table.rowCount())}
    assert labels == {"bull flag", "untagged"}, "provisional buckets never share the confirmed table"
