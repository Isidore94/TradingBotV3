import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_setup_rows_from_run_result_include_theta_and_tradeworthy_study_rows():
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
            # Context-overlay studies must NOT reach the desk.
            "hv_level_study_rows": [
                {"symbol": "MSFT", "side": "SHORT", "score": 52, "setup_tags": ["HV_LEVEL_BLOCK"]}
            ],
            # Entry-pattern studies show, labelled as their ACTUAL setup.
            "second_dev_breakout_study_rows": [
                {
                    "symbol": "IBM",
                    "side": "LONG",
                    "score": 90,
                    "priority_bucket": "study_2nddev_breakout",
                    "setup_family": "2nddev_breakout",
                }
            ],
            # A clone of a symbol/side already on the board is skipped.
            "playbook_study_rows": [
                {
                    "symbol": "NVDA",
                    "side": "LONG",
                    "score": 120,
                    "priority_bucket": "study_playbook",
                    "setup_family": "playbook_volume_thrust",
                },
                {
                    "symbol": "AAPL",
                    "side": "LONG",
                    "score": 91.25,
                    "priority_bucket": "study_playbook",
                    "setup_family": "playbook_volume_thrust",
                },
            ],
        }
    )

    by_symbol = {row.symbol: row for row in rows}
    assert by_symbol["AAPL"].bucket_label == "Favorite"
    assert by_symbol["AAPL"].theta == "Put 190.0 strike @ 1.25 18d"
    assert by_symbol["AAPL"].supports_text == "4"
    assert by_symbol["NVDA"].bucket_label == "Volume Thrust"
    assert by_symbol["IBM"].bucket_label == "2nd-Dev Break"
    assert "MSFT" not in by_symbol
    assert sum(1 for row in rows if row.symbol == "AAPL") == 1


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


def test_focus_payload_merges_bucket_duplicates_but_preserves_real_theses():
    from ui.services.data_feed import _rows_from_focus_payload

    common = {
        "symbol": "NVDA",
        "side": "LONG",
        "setup_family": "earnings_gap_retest",
        "anchor_date": "2026-06-01",
        "score": 120,
    }
    rows = _rows_from_focus_payload(
        {
            "high_conviction": [{**common, "priority_bucket": "high_conviction"}],
            "favorites": [{**common, "priority_bucket": "favorite_setup"}],
            "near_favorite_zones": [
                {
                    **common,
                    "anchor_date": "2026-05-01",
                    "priority_bucket": "near_favorite_zone",
                },
                {
                    **common,
                    "symbol": "AMD",
                    "setup_family": "weekly_ema8_hold_retest",
                    "priority_bucket": "near_favorite_zone",
                },
            ],
        }
    )

    nvda = [row for row in rows if row.symbol == "NVDA"]
    assert len(nvda) == 2, "a separate anchor remains a separate opportunity"
    merged = next(row for row in nvda if row.bucket == "high_conviction")
    assert merged.bucket_display == "High Conviction / Favorite"
    assert any(row.symbol == "AMD" for row in rows)


def test_run_result_duplicate_promotes_the_primary_bucket_by_precedence():
    from ui.services.data_feed import rows_from_run_result

    common = {
        "symbol": "AAPL",
        "side": "LONG",
        "setup_family": "avwap_retest",
        "anchor_date": "2026-07-01",
    }
    rows = rows_from_run_result(
        {
            "tracked_rows": [
                {**common, "priority_bucket": "favorite_setup"},
                {**common, "priority_bucket": "high_conviction"},
            ]
        }
    )

    assert len(rows) == 1
    assert rows[0].bucket == "high_conviction"
    assert rows[0].bucket_display == "High Conviction / Favorite"


def test_priority_report_parser_reads_ranked_expected_r_section(tmp_path):
    from ui.services.data_feed import load_setup_rows_from_priority_report

    # The parser reads the machine-readable "Ranked by Expected-R (blended)"
    # section (raw internal bucket names) and enriches key level from the band
    # zone in "Overall score rankings".
    report = tmp_path / "priority.txt"
    report.write_text(
        "Overall score rankings\n"
        "----------------------\n"
        "  1. NVDA   LONG  score=88   bucket=Favorite setup family=earnings gap            zone=VWAP to UPPER_1   trend=UP       clean\n"
        "\n"
        "Ranked by Expected-R (blended)\n"
        "------------------------------\n"
        "NVDA LONG ExpR=+1.20R score=88 WR=60%  PF=3.2  n=19  family=earnings gap bucket=favorite_setup\n",
        encoding="utf-8",
    )

    rows = load_setup_rows_from_priority_report(report)

    assert len(rows) == 1
    assert rows[0].symbol == "NVDA"
    assert rows[0].side == "LONG"
    assert rows[0].bucket == "favorite_setup"
    assert rows[0].bucket_label == "Favorite"
    assert rows[0].expected_r == 1.2
    assert rows[0].key_level == "VWAP to UPPER_1"


def test_sma_track_bucket_is_first_class_setup_bucket(tmp_path):
    from ui.models.setup import DEFAULT_SETUP_BUCKET_FILTER_LABELS, SetupRow
    from ui.services.data_feed import load_setup_rows_from_priority_report

    assert SetupRow(symbol="SMAT", bucket="sma_breakout_tracking").bucket_label == "SMA Track"
    assert "SMA Track" in DEFAULT_SETUP_BUCKET_FILTER_LABELS
    assert "Stdev Track" in DEFAULT_SETUP_BUCKET_FILTER_LABELS

    report = tmp_path / "priority.txt"
    report.write_text(
        "Ranked by Expected-R (blended)\n"
        "------------------------------\n"
        "SMAT LONG ExpR=+0.50R score=72 family=sma breakout retest bucket=sma_breakout_tracking\n",
        encoding="utf-8",
    )

    rows = load_setup_rows_from_priority_report(report)

    assert len(rows) == 1
    assert rows[0].symbol == "SMAT"
    assert rows[0].bucket == "sma_breakout_tracking"
    assert rows[0].bucket_label == "SMA Track"
