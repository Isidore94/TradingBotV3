"""P8-P5 (trader's go 2026-09-25): the journal's truths in plain words.

Stocks vs options, longs vs shorts and confirmed setups on Day Review, the
Week Review coach card and the Journal Analytics tab; the bot's grade of each
traded setup as of the entry; thin rows and a 4-week rollup on the week card;
and the exit scoreboard.
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

NY = ZoneInfo("America/New_York")


def _trade(tid, net, *, cad=None, day="2026-09-14", opened="10:00:00", closed="11:00:00",
           direction="LONG", security_type="STK", symbol="SHOP", tags="", tag_status="confirmed",
           status="CLOSED", **extra):
    row = {
        "trade_id": tid, "symbol": symbol, "security_type": security_type, "direction": direction,
        "status": status, "trade_date": day,
        "opened_at": f"{day}T{opened}-04:00", "closed_at": f"{day}T{closed}-04:00" if closed else "",
        "net_pnl": net, "net_pnl_cad": net if cad is None else cad, "currency": "USD",
        "setup_tags": tags, "tag_status": tag_status,
    }
    row.update(extra)
    return row


TRADES = [
    _trade("s1", 100.0, cad=140.0),
    _trade("s2", -50.0, cad=-70.0, direction="SHORT"),
    _trade("o1", 300.0, cad=420.0, security_type="OPT", symbol="SHOP  260918C00100000", direction="LONG"),
    _trade("o2", -20.0, cad=-28.0, security_type="OPT", symbol="SHOP  260918P00090000", direction="SHORT"),
    _trade("open", 999.0, cad=999.0, status="OPEN", closed=""),
]


# ---------------------------------------------------------------------------
# step 2: the truth lines and the two new breakdown groups
# ---------------------------------------------------------------------------
def test_split_lines_say_stocks_options_longs_and_shorts_in_the_surfaces_currency():
    import journal_truth

    lines = journal_truth.split_lines(TRADES, "net_pnl_cad", "CAD")
    assert lines == [
        "Stocks: 2 trades, +$70 CAD. Options: 2 trades, +$392 CAD.",
        "Longs: 2, +$560 CAD. Shorts: 2, -$98 CAD.",
    ]


def test_a_refused_total_is_money_unknown_never_zero():
    import journal_truth

    lines = journal_truth.split_lines(TRADES, "", "")
    assert lines[0] == "Stocks: 2 trades, money unknown. Options: 2 trades, money unknown."


def test_setup_lines_count_confirmed_tags_only():
    import journal_truth

    rows = [
        _trade("a", 100.0, tags="avwap_band_bounce | favorite_setup"),
        _trade("b", -40.0, tags="avwap_band_bounce | favorite_setup"),
        _trade("c", 500.0, tags="new_5d_high", tag_status="provisional"),
        _trade("d", 10.0, tags="vetoed:too_extended"),
        _trade("e", 10.0),
    ]
    lines = journal_truth.setup_lines(rows, "net_pnl_cad", "CAD")
    assert lines == [
        "By setup (confirmed only): avwap band bounce, n 2, win 50%, expectancy +$30 CAD.",
        "2 of 5 closed trades have a confirmed setup.",
    ]
    none = journal_truth.setup_lines([_trade("x", 1.0)], "net_pnl_cad", "CAD")
    assert none == ["By setup (confirmed only): none yet. 0 of 1 closed trade have a confirmed setup."]


def test_instrument_and_direction_are_breakdown_groups():
    import journal_analytics as ja

    assert ja.DERIVED_GROUPS["instrument"] is ja.trade_instrument
    assert ja.DERIVED_GROUPS["direction"] is ja.trade_direction
    rows = {row["label"]: row for row in ja.derived_group_summary(TRADES, "instrument", pnl_key="net_pnl_cad")}
    assert rows["STK"]["closed"] == 2 and rows["STK"]["net_pnl"] == pytest.approx(70.0)
    assert rows["OPT"]["closed"] == 2 and rows["OPT"]["net_pnl"] == pytest.approx(392.0)
    summary = ja.build_analytics_summary(TRADES, "CAD")
    assert {row["label"] for row in summary["groups"]["instrument"]} == {"STK", "OPT"}


@pytest.fixture
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class _Header:
    currency_mode = "CAD"

    def query(self):
        return {}

    def date_bounds(self):
        return (None, None)


def test_the_analytics_tab_says_the_truth_lines_in_its_currency(qapp, monkeypatch):
    from ui.models.journal import JournalTrade
    from ui.panels.journal import analytics_tab as at
    from ui.services import journal_feed

    trades = [JournalTrade.from_mapping(row) for row in TRADES]
    monkeypatch.setattr(journal_feed, "load_trades", lambda **_kw: trades)
    tab = at.AnalyticsTab(_Header())
    try:
        tab.reload()
        text = tab.truth_note.text()
        assert "Stocks: 2 trades, +$70 CAD. Options: 2 trades, +$392 CAD." in text
        assert "Longs: 2, +$560 CAD. Shorts: 2, -$98 CAD." in text
        assert "By setup (confirmed only): none yet." in text
        names = [tab.group_picker.itemText(i) for i in range(tab.group_picker.count())]
        assert "instrument" in names and "direction" in names
    finally:
        tab.shutdown()
        tab.deleteLater()


def test_day_review_truth_covers_the_last_20_sessions_in_cad():
    import walkaway_day
    from ui.services.day_review_service import PAYLOAD_KEYS, DayReviewService, empty_payload

    assert "truth" in PAYLOAD_KEYS and empty_payload("2026-09-22")["truth"] == {}
    session = "2026-09-22"
    first = walkaway_day.earlier_sessions(session, count=19)[0]
    before = (date.fromisoformat(first) - timedelta(days=1)).isoformat()
    rows = [
        _trade("in1", 100.0, cad=140.0, day=session),
        _trade("in2", -10.0, cad=-14.0, day=first, direction="SHORT"),
        _trade("old", 1000.0, cad=1400.0, day=before),
    ]
    truth = DayReviewService._truth(session, rows, [rows[0]])
    assert truth["lines"][0] == f"Last 20 sessions ({first} to {session}):"
    assert "Stocks: 2 trades, +$126 CAD. Options: 0 trades." in truth["lines"]
    assert set(truth["grades"]) == {"in1"}
    unread = DayReviewService._truth(session, None, [])
    assert unread["lines"] == ["The journal was not read, so the 20-session lines are unknown."]


def test_the_day_review_details_show_the_truth_lines(qapp, monkeypatch):
    from ui.panels import day_review_panel
    from ui.services.day_review_service import empty_payload

    class _Service:
        def read_day(self, session_date, **_kwargs):
            return empty_payload(session_date)

    widget = day_review_panel.DayReviewPanel(service=_Service(), clock=lambda: datetime(2026, 9, 23, 7, 30))
    try:
        payload = empty_payload("2026-09-22")
        payload["truth"] = {"lines": ["Last 20 sessions (a to b):", "Stocks: 1 trade, +$5 CAD."], "grades": {}}
        widget.render(payload)
        assert widget.truth_note.text() == "Last 20 sessions (a to b):\nStocks: 1 trade, +$5 CAD."
    finally:
        widget.shutdown()
        widget.deleteLater()


def test_the_week_card_says_the_week_and_the_4_week_rollup(qapp, tmp_path):
    import week_coach
    from ui.widgets.week_coach_card import WeekCoachCard

    week = "2026-W38"  # Mon 2026-09-14
    rows = [
        _trade("w", 100.0, cad=140.0, day="2026-09-14"),
        _trade("r", -10.0, cad=-14.0, day="2026-08-25", security_type="OPT",
               symbol="SHOP  260918C00100000"),
        _trade("x", 5.0, cad=7.0, day="2026-08-10"),  # before the rollup
    ]
    view = week_coach.read_view(week, root=tmp_path, trades_loader=lambda: rows,
                                questions_path=tmp_path / "q.jsonl", answers_path=tmp_path / "a.jsonl")
    truth = view["truth"]
    assert truth["rollup_weeks"] == ["2026-W35", "2026-W36", "2026-W37", "2026-W38"]
    assert truth["lines"][0] == "Stocks: 1 trade, +$140 CAD. Options: 0 trades."
    assert truth["rollup_lines"][0] == "Stocks: 1 trade, +$140 CAD. Options: 1 trade, -$14 CAD."
    card = WeekCoachCard(read=lambda *_a, **_k: view)
    try:
        card.render(view)
        text = card.truth_label.text()
        assert text.startswith("This week:\nStocks: 1 trade, +$140 CAD.")
        assert "Last 4 weeks (2026-W35 to 2026-W38):" in text
    finally:
        card.deleteLater()


def test_an_unreadable_journal_is_said_on_the_week_card():
    import week_coach

    def boom():
        raise OSError("locked")

    truth = week_coach.truth_view(["2026-W38"], ["2026-W38"], trades_loader=boom)
    assert truth == {"error": "the journal could not be read: locked"}
