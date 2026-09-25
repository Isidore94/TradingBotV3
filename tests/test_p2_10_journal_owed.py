"""P2-10: the journal's owed items (made-up entries out of totals, one store owner)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

SESSION = "2026-09-22"

# One real trade, one made-up entry (SYNTHETIC_OPEN), one TFSA stock short.
# Before P2-10 the totals were 100 - 40 + 341.57 + 25 = 426.57; after, only 60.00.
REAL = {"trade_id": "r1", "symbol": "ABC", "status": "CLOSED", "trade_date": SESSION,
        "tag_status": "confirmed", "setup_tags": "x", "net_pnl": 100.0, "net_pnl_cad": 100.0}
REAL_LOSS = {"trade_id": "r2", "symbol": "DEF", "status": "CLOSED", "trade_date": SESSION,
             "tag_status": "confirmed", "setup_tags": "x", "net_pnl": -40.0, "net_pnl_cad": -40.0}
MADE_UP = {"trade_id": "m1", "symbol": "SMH", "status": "CLOSED", "trade_date": SESSION,
           "tag_status": "confirmed", "setup_tags": "x", "net_pnl": 341.57,
           "net_pnl_cad": 341.57, "synthetic_entry": True}
TFSA_SHORT = {"trade_id": "m2", "symbol": "CRDO", "status": "CLOSED", "trade_date": SESSION,
              "tag_status": "confirmed", "setup_tags": "x", "net_pnl": 25.0, "net_pnl_cad": 25.0,
              "direction": "SHORT", "security_type": "STK", "account_tax_status": "TAX_FREE"}
ROWS = [REAL, REAL_LOSS, MADE_UP, TFSA_SHORT]


def _rows():
    return [dict(row) for row in ROWS]


def test_day_review_glance_keeps_made_up_trades_but_leaves_them_out_of_pnl():
    import day_report_card

    glance = day_report_card.glance({"trades": _rows()})

    assert glance["trades"] == 4  # kept
    assert glance["pnl"] == 60.0  # before: 426.57
    assert glance["pnl_counted"] == 2
    assert glance["wins"] == 1 and glance["losses"] == 1
    assert glance["pnl_not_counted"] == 2


def test_day_review_sparkline_leaves_made_up_trades_out():
    from ui.services.day_review_service import _pnl_by_session

    assert _pnl_by_session((SESSION,), _rows()) == ((SESSION, 60.0),)
    only_made_up = [dict(MADE_UP)]
    assert _pnl_by_session((SESSION,), only_made_up) == ((SESSION, None),)


def test_weekend_prep_week_line_leaves_made_up_trades_out():
    import weekend_verdict

    line = weekend_verdict.journal_week_line(_rows())

    assert line.n == 2
    assert "+60.00" in line.text


def test_recap_day_record_total_leaves_made_up_trades_out_but_keeps_the_rows():
    import day_session_record

    section = day_session_record._trades({"payload": {"trades": _rows()}}, [])

    assert section["n"] == 4
    assert len(section["rows"]) == 4
    assert section["net_pnl"] == 60.0  # before: 426.57
    made_up = next(row for row in section["rows"] if row["trade_id"] == "m1")
    assert made_up["net_pnl"] == 341.57  # the trade's own number stays visible
    assert made_up["counted_in_pnl"] is False


# ---------------------------------------------------------------------------
# step 2: the Mentor rule lane never reads the journal on the Qt thread
# ---------------------------------------------------------------------------
def _size_rule():
    return {"for_date": SESSION, "rule_id": "rc-1", "set_on": "2026-09-21",
            "text": "size down in chop", "tag": "size_down_in_chop", "streak": 1}


def _sized_trade(trade_id, day, qty, *, opened="09:50"):
    return {
        "trade_id": trade_id, "symbol": "ZETA", "direction": "LONG", "status": "CLOSED",
        "opened_at": f"{day}T{opened}:00-04:00", "closed_at": f"{day}T11:00:00-04:00",
        "trade_date": day, "net_pnl": 10.0, "average_entry_price": 10.0,
        "average_exit_price": 10.1, "quantity_opened": qty,
    }


class _CountingStore:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.calls = 0

    def list_trades(self, **_kwargs):
        self.calls += 1
        return list(self.rows)


def test_rule_lane_reads_no_journal_on_the_qt_thread():
    import types

    import pytest

    pytest.importorskip("PySide6")
    from ui.app import MainWindow

    store = _CountingStore()
    host = types.SimpleNamespace(
        rule_chip=types.SimpleNamespace(info=_size_rule), _regime_timeline=[],
    )
    MainWindow._mentor_rule_lane(host, store, SESSION, [_sized_trade("t1", SESSION, 500)])
    assert store.calls == 0  # before: one list_trades call on the Qt thread


def test_rule_lane_uses_the_worker_baseline_for_its_own_session_only():
    import types
    from datetime import datetime, timedelta, timezone

    import pytest

    pytest.importorskip("PySide6")
    from ui.app import MainWindow

    earlier = [_sized_trade(f"e{i}", f"2026-09-{10 + i:02d}", 100) for i in range(6)]
    store = _CountingStore(earlier)
    # The worker's read: one bounded journal query, the median entry notional.
    baseline = MainWindow._read_rule_size_baseline(SESSION, store=store)
    assert baseline == {"session": SESSION, "median": 1000.0}
    assert store.calls == 1

    chop = datetime(2026, 9, 22, 9, 35, tzinfo=timezone(timedelta(hours=-4)))
    host = types.SimpleNamespace(
        rule_chip=types.SimpleNamespace(info=_size_rule),
        _regime_timeline=[(chop, "neutral_chop")],
        _rule_size_baseline=baseline,
    )
    big = [_sized_trade("t1", SESSION, 500)]
    rows = MainWindow._mentor_rule_lane(host, None, SESSION, big)
    assert [row["trade_id"] for row in rows] == ["t1"]
    # A baseline for another session is not used: the size check says nothing.
    host._rule_size_baseline = {"session": "2026-09-21", "median": 1000.0}
    assert MainWindow._mentor_rule_lane(host, None, SESSION, big) == []
    # Another rule clears the baseline without starting a read.
    MainWindow._refresh_rule_size_baseline(host, {"tag": "hold_winners"})
    assert host._rule_size_baseline is None
