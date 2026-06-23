import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _demo_payload():
    return {
        "ticker": "NVDA",
        "metadata": {"company": "NVIDIA", "sector": "Semiconductors"},
        "peer_reason": "GPU peers",
        "markdown": "# NVDA report",
        "target_earnings": [
            {"date": "2026-07-01", "ticker": "NVDA", "company": "NVIDIA", "time": "AMC", "importance": "High"}
        ],
        "peer_earnings": [
            {"date": "2026-07-02", "ticker": "AMD", "company": "AMD", "time": "AMC", "importance": "Med"}
        ],
        "target_headlines": [{"title": "Chips rally", "source": "WSJ", "published": "2026-06-22"}],
        "industry_headlines": [],
    }


def test_format_earnings_rows_tags_scope_for_target_and_peers():
    from ui.services.ticker_lookup_feed import format_earnings_rows

    rows = format_earnings_rows(_demo_payload())
    assert [row["scope"] for row in rows] == ["Target", "Peer"]
    assert rows[0]["ticker"] == "NVDA"
    assert rows[1]["ticker"] == "AMD"


def test_headlines_and_summary_render_expected_text():
    from ui.services.ticker_lookup_feed import format_headlines, lookup_summary

    payload = _demo_payload()
    headlines = format_headlines(payload)
    assert "Chips rally" in headlines
    assert "WSJ" in headlines

    summary = lookup_summary(payload)
    assert "NVDA" in summary
    assert "NVIDIA" in summary
    assert "Peer context: GPU peers" in summary


def test_format_helpers_tolerate_empty_payload():
    from ui.services.ticker_lookup_feed import format_earnings_rows, format_headlines, lookup_summary

    assert format_earnings_rows({}) == []
    assert format_headlines({}) == "No headlines returned."
    assert lookup_summary({}) == "No ticker loaded."
