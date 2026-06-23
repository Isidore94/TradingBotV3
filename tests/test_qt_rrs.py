import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _payload():
    return {
        "threshold": 2.0,
        "results": [
            ("RS", "NVDA", 4.81, 2.1),
            ("RS", "MSFT", 1.2, 0.4),  # below threshold -> filtered
            ("RW", "TSLA", -3.51, 1.3),
            ("RS", "AVGO", 3.92, 1.7),
        ],
        "results_sector": [{"signal": "RS", "symbol": "XLK", "rrs": 2.6, "power_index": 1.1}],
    }


def test_rrs_rows_filter_threshold_and_rank_strongest_first():
    from ui.models.rrs import rrs_rows

    rows = rrs_rows(_payload(), "SPY")
    assert [row.symbol for row in rows] == ["NVDA", "AVGO", "TSLA"]  # MSFT dropped
    assert rows[0].side == "RS" and rows[0].rrs == 4.81
    assert rows[-1].side == "RW" and rows[-1].rrs == -3.51


def test_rrs_rows_accept_dict_rows_and_scope_selection():
    from ui.models.rrs import rrs_rows

    rows = rrs_rows(_payload(), "Sector")
    assert len(rows) == 1
    assert rows[0].symbol == "XLK"
    assert rows[0].power == 1.1


def test_rrs_rows_tolerate_missing_payload():
    from ui.models.rrs import rrs_rows

    assert rrs_rows(None, "SPY") == []
    assert rrs_rows({}, "SPY") == []
