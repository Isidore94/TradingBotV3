import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_journal_trade_from_mapping_parses_core_fields():
    from ui.models.journal import JournalTrade

    trade = JournalTrade.from_mapping(
        {
            "trade_id": "abc123",
            "trade_date": "2026-06-20T00:00:00",
            "symbol": "nvda",
            "direction": "long",
            "status": "closed",
            "quantity_closed": 100,
            "average_entry_price": 120.5,
            "average_exit_price": 128.0,
            "net_pnl": 740.0,
            "commission": 1.0,
            "fees": 0.5,
            "account_label": "Main",
            "broker": "IBKR",
            "display_tags": "EMA15_RETEST",
        }
    )

    assert trade.symbol == "NVDA"
    assert trade.direction == "LONG"
    assert trade.is_closed
    assert trade.quantity == 100
    assert trade.net_pnl == 740.0
    assert trade.fees == 1.5  # commission + fees
    assert trade.trade_date == "2026-06-20"
    assert trade.tags == "EMA15_RETEST"


def test_journal_trade_handles_missing_costs_and_tags():
    from ui.models.journal import JournalTrade

    trade = JournalTrade.from_mapping(
        {"trade_id": "x", "symbol": "AAPL", "status": "OPEN", "auto_tag_summary": "AVWAP"}
    )

    assert trade.fees is None
    assert trade.net_pnl is None
    assert not trade.is_closed
    assert trade.tags == "AVWAP"  # falls back to auto-tag summary
