import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_setup_rows_from_run_result_include_theta_and_study_rows():
    from ui.services.data_feed import rows_from_run_result

    rows = rows_from_run_result(
        {
            "tracked_rows": [
                {
                    "symbol": "AAPL",
                    "side": "LONG",
                    "score": 91.25,
                    "priority_bucket": "favorite_setup",
                    "setup_tags": ["EMA15_RETEST", "AVWAP_SUPPORT"],
                    "current_band_zone": "VWAP to UPPER_1",
                    "expected_r": 1.42,
                    "days_to_next_earnings": 12,
                    "support_count": 4,
                }
            ],
            "theta_put_rows": [
                {
                    "symbol": "AAPL",
                    "best_option": {
                        "strike": 190,
                        "credit": 1.25,
                        "market_days": 18,
                    },
                }
            ],
            "hv_level_study_rows": [
                {
                    "symbol": "MSFT",
                    "side": "SHORT",
                    "score": 52,
                    "setup_tags": ["HV_LEVEL_BLOCK"],
                    "hv_level_nearby_count": 1,
                }
            ],
        }
    )

    by_symbol = {row.symbol: row for row in rows}
    assert by_symbol["AAPL"].bucket_label == "Favorite"
    assert by_symbol["AAPL"].theta == "Put 190.0 strike @ 1.25 18d"
    assert by_symbol["AAPL"].supports_text == "4"
    assert by_symbol["MSFT"].bucket_label == "Study"


def test_copy_symbols_ranked_preserves_order_while_lists_alphabetize():
    from ui.models.setup import SetupRow
    from ui.services.data_feed import copy_symbols

    rows = [
        SetupRow(symbol="NVDA", side="LONG", bucket="high_conviction"),
        SetupRow(symbol="AAPL", side="LONG", bucket="favorite_setup"),
        SetupRow(symbol="TSLA", side="SHORT", bucket="near_favorite_zone"),
        SetupRow(symbol="MSFT", side="LONG", bucket="study"),
    ]

    # "Ranked" must preserve the incoming (rank) order; the others alphabetize.
    assert copy_symbols(rows, "ranked") == "NVDA, AAPL, TSLA, MSFT"
    assert copy_symbols(rows, "longs") == "AAPL, MSFT, NVDA"
    assert copy_symbols(rows, "favorites") == "AAPL, NVDA"
    assert copy_symbols(rows, "active") == "AAPL, NVDA, TSLA"


def test_priority_report_parser_reads_existing_ranked_lines(tmp_path):
    from ui.services.data_feed import load_setup_rows_from_priority_report

    report = tmp_path / "priority.txt"
    report.write_text(
        "Overall score rankings\n"
        "----------------------\n"
        "  1. NVDA   LONG  score=88.5  bucket=favorite     family=earnings gap             zone=VWAP to UPPER_1   trend=UP       clean\n",
        encoding="utf-8",
    )

    rows = load_setup_rows_from_priority_report(report)

    assert len(rows) == 1
    assert rows[0].symbol == "NVDA"
    assert rows[0].side == "LONG"
    assert rows[0].bucket_label == "Favorite"
    assert rows[0].key_level == "VWAP to UPPER_1"
