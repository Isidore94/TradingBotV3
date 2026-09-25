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
