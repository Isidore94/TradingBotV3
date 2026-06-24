import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_theta_row_from_mapping_formats_core_fields():
    from ui.models.theta import ThetaRow

    row = ThetaRow.from_mapping(
        {
            "symbol": "aapl",
            "play_type": "pcs",
            "score": "88",
            "support_count": "4",
            "recommended_strike": "190",
            "recommended_long_strike": "185",
            "recommended_credit": "1.23",
        }
    )

    assert row.symbol == "AAPL"
    assert row.play_label == "PCS"
    assert row.score == 88
    assert row.support_count == 4
    assert row.recommended_credit == 1.23


def test_market_prep_section_helpers_count_symbols_from_payload_and_text():
    from ui.services.market_prep_feed import section_copy_text, section_symbol_count

    definition = {"empty_message": "None"}
    section = {"symbols": ["AAPL", "MSFT", ""], "copy_text": "AAPL, MSFT"}

    text = section_copy_text(section, definition)

    assert text == "AAPL, MSFT"
    assert section_symbol_count(section, text) == 2
    assert section_symbol_count({}, "NVDA, TSLA AMD") == 3


def test_human_focus_pick_helpers_format_today_rows(tmp_path):
    from ui.services.market_prep_feed import (
        human_focus_pick_count,
        human_focus_pick_text,
        load_human_focus_daily_picks,
    )

    path = tmp_path / "human_focus_daily_picks.csv"
    path.write_text(
        "\n".join(
            [
                "trade_date,symbol,side,source,snapshotted_at,active_at_snapshot",
                "2026-06-01,NVDA,LONG,focus_pick,2026-06-01T09:35:00,1",
                "2026-06-01,TSLA,SHORT,focus_pick,2026-06-01T09:35:00,1",
                "2026-06-02,AAPL,LONG,focus_pick,2026-06-02T09:35:00,1",
            ]
        ),
        encoding="utf-8",
    )

    rows = load_human_focus_daily_picks(trade_date="2026-06-01", path=path)

    assert human_focus_pick_count(rows) == 2
    assert human_focus_pick_text(rows) == "LONG: NVDA\nSHORT: TSLA"
