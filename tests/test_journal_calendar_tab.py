"""The Journal calendar redo (2026-09-23): a Mon-Fri grid with weekly totals,
tinted day cells with trade counts, a month summary and a 12-month strip.
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

from ui.panels.journal import calendar_tab as ct  # noqa: E402


def _day(net, trades=1, wins=None, losses=None):
    return {
        "net": net,
        "trades": trades,
        "wins": wins if wins is not None else (1 if net > 0 else 0),
        "losses": losses if losses is not None else (1 if net < 0 else 0),
    }


# September 2026 starts on a Tuesday; the 5th/6th are a weekend.
DAYS = {
    "2026-09-01": _day(100.0, 2, 2, 0),
    "2026-09-03": _day(-40.0, 3, 1, 2),
    "2026-09-06": _day(10.0),  # a Sunday trade
    "2026-09-08": _day(0.0, 1, 0, 0),
    "2026-10-01": _day(55.0),
    "2025-09-01": _day(999.0),
}


class TestMonthLayout:
    def test_weeks_are_monday_to_friday_with_a_total(self):
        layout = ct.month_layout(DAYS, 2026, 9)
        weeks = layout["weeks"]
        assert all(len(week["days"]) == 5 for week in weeks)
        assert weeks[0]["days"][0] is None  # Monday Aug 31 is not in September
        assert weeks[0]["days"][1]["date"] == "2026-09-01"
        # The Sunday trade counts in its week, never dropped.
        assert weeks[0]["net"] == pytest.approx(70.0)
        assert weeks[0]["trades"] == 6
        assert weeks[1]["net"] == pytest.approx(0.0)
        assert weeks[2]["net"] is None

    def test_cells_carry_count_wins_losses_and_tone(self):
        first = ct.month_layout(DAYS, 2026, 9)["weeks"][0]["days"]
        tuesday, wednesday, thursday = first[1], first[2], first[3]
        assert tuesday["trades"] == 2 and tuesday["wins"] == 2
        assert tuesday["tone"] == "win_strong"  # the month's biggest day
        assert thursday["tone"] == "loss"
        assert wednesday["net"] is None and wednesday["tone"] == "none"
        breakeven = ct.month_layout(DAYS, 2026, 9)["weeks"][1]["days"][1]
        assert breakeven["date"] == "2026-09-08"
        assert breakeven["tone"] == "flat", "breakeven must not look like a no-trade day"

    def test_month_summary(self):
        summary = ct.month_layout(DAYS, 2026, 9)["summary"]
        assert summary["net"] == pytest.approx(70.0)
        assert summary["days_traded"] == 4
        assert (summary["green_days"], summary["red_days"], summary["flat_days"]) == (2, 1, 1)
        assert summary["best"] == ("2026-09-01", 100.0)
        assert summary["worst"] == ("2026-09-03", -40.0)
        assert summary["weekend_days"] == 1
        text = ct.month_summary_text(summary, "CAD")
        assert "Net +70.00 CAD" in text and "2 green, 1 red, 1 flat" in text
        assert "Best Sep 1 +100" in text and "Worst Sep 3 -40" in text
        assert "weekend" in text

    def test_an_empty_month_says_so(self):
        layout = ct.month_layout({}, 2026, 2)
        assert layout["summary"]["net"] is None
        assert ct.month_summary_text(layout["summary"], "CAD") == "No closed trades this month."


def test_year_strip_has_twelve_months_and_blank_is_not_zero():
    months = ct.year_month_totals(DAYS, 2026)
    assert [m["label"] for m in months][:3] == ["Jan", "Feb", "Mar"]
    assert len(months) == 12
    sep, octo = months[8], months[9]
    assert sep["net"] == pytest.approx(70.0) and sep["days"] == 4
    assert (sep["green"], sep["red"]) == (2, 1)
    assert octo["net"] == pytest.approx(55.0)
    assert months[0]["net"] is None, "a month with no trading is unknown, not zero"


@pytest.mark.parametrize(
    ("value", "scale", "tone"),
    [(None, 10, "none"), (0.0, 10, "flat"), (3.0, 10, "win"), (6.0, 10, "win_strong"),
     (-3.0, 10, "loss"), (-5.0, 10, "loss_strong"), (4.0, 0, "win")],
)
def test_pnl_tone(value, scale, tone):
    assert ct.pnl_tone(value, scale) == tone


def test_format_pnl_is_signed():
    assert ct.format_pnl(1234.4) == "+1,234"
    assert ct.format_pnl(-56.0) == "-56"
    assert ct.format_pnl(0.001) == "0"
    assert ct.format_pnl(None) == "-"


# ---------------------------------------------------------------------------
# The widget, against a stub header and a stubbed feed
# ---------------------------------------------------------------------------


class _Header:
    currency_mode = "CAD"

    def __init__(self):
        self.asked = []

    def query(self):
        return {"date_from": date.today() - timedelta(days=30), "date_to": date.today(), "symbol": ""}


@pytest.fixture
def tab(monkeypatch):
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    calls = []

    def fake_month_data(**kwargs):
        calls.append(kwargs)
        return {"days": DAYS, "pnl_key": "net_pnl_cad", "note": "", "currency": "CAD"}

    monkeypatch.setattr(ct.journal_feed, "calendar_month_data", fake_month_data)
    widget = ct.CalendarTab(_Header())
    widget.year_input.setCurrentText("2026")
    widget.month_input.setCurrentIndex(8)
    widget.reload()
    widget.calls = calls
    yield widget
    widget.deleteLater()


def test_the_calendar_reads_the_whole_selected_year_not_the_header_range(tab):
    last = tab.calls[-1]
    assert last["date_from"] == date(2026, 1, 1)
    assert last["date_to"] == date(2026, 12, 31)
    assert last["currency_mode"] == "CAD"


def test_the_grid_has_no_weekend_columns_and_a_week_total(tab):
    heads = [
        tab.grid_layout.itemAtPosition(0, column).widget().text() for column in range(6)
    ]
    assert heads == ["Mon", "Tue", "Wed", "Thu", "Fri", "Week"]
    tuesday = tab.day_cells[0][1]
    assert tuesday.net_label.text() == "+100"
    assert "2 trades" in tuesday.detail_label.text() and "2W 0L" in tuesday.detail_label.text()
    assert tuesday.property("tone") == "win_strong"
    assert tab.week_cells[0].net_label.text() == "+70"
    assert "Numbers in CAD" == tab.currency_badge.text()
    assert "Net +70.00 CAD" in tab.summary.text()


def test_clicking_a_day_emits_it_and_a_month_card_opens_that_month(tab):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    seen: list[str] = []
    tab.daySelected.connect(seen.append)
    QTest.mouseClick(tab.day_cells[0][3], Qt.LeftButton)
    assert seen == ["2026-09-03"]
    QTest.mouseClick(tab.day_cells[0][0], Qt.LeftButton)  # outside the month
    assert seen == ["2026-09-03"]

    QTest.mouseClick(tab.month_cards[9], Qt.LeftButton)
    assert tab.month_input.currentIndex() == 9
    assert tab.month_cards[9].property("selected") is True


def test_prev_and_next_step_across_the_year(tab):
    tab.month_input.setCurrentIndex(0)
    tab._step_month(-1)
    assert (tab.year_input.currentText(), tab.month_input.currentIndex()) == ("2025", 11)
    assert tab.calls[-1]["date_from"] == date(2025, 1, 1)
    tab._step_month(1)
    assert (tab.year_input.currentText(), tab.month_input.currentIndex()) == ("2026", 0)
