"""One R everywhere: `journal_analytics.trade_r_multiple`, native currency."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# A USD loser: risk typed in USD, so native R is -150 / 100 = -1.5.
# The old CAD reading was -210 / 100 = -2.1.
USD_TRADE = {
    "trade_id": "t-usd", "symbol": "AMD", "direction": "LONG", "status": "CLOSED",
    "opened_at": "2026-09-22T09:45:00-04:00", "closed_at": "2026-09-22T10:30:00-04:00",
    "net_pnl": -150.0, "net_pnl_cad": -210.0, "planned_risk": 100.0, "currency": "USD",
    "setup_tags": "",
}
NATIVE_R = -1.5
CAD_R = -2.1


def _journal_r() -> float:
    from journal_analytics import trade_r_multiple

    value = trade_r_multiple(USD_TRADE)
    assert value == pytest.approx(NATIVE_R)
    return value


def _rule_loop_text() -> str:
    import recap_rule_loop

    return recap_rule_loop._check(
        "respect_stop", USD_TRADE, session=date(2026, 9, 22), winners=(),
        size_median=None, timeline=None,
    )


def test_recap_rule_loop_reads_the_journals_native_r():
    import recap_rule_loop

    text = _rule_loop_text()
    assert f"{_journal_r():+.1f}R" in text
    assert f"{CAD_R:+.1f}R" not in text
    assert not hasattr(recap_rule_loop, "trade_r"), "no local R definition"


def test_preference_trade_outcomes_reads_the_journals_native_r():
    import preference_trade_outcomes as report

    assert report._canonical_r(USD_TRADE) == f"{_journal_r():.4f}"
    assert report._canonical_r(USD_TRADE) != f"{CAD_R:.4f}"


def test_day_session_record_reads_the_journals_native_r():
    import day_session_record as dsr

    row = dsr._trades({"payload": {"trades": [USD_TRADE]}}, [])["rows"][0]
    assert row["r_multiple"] == pytest.approx(_journal_r())
    assert row["r_multiple"] != pytest.approx(CAD_R)
    # The money column stays CAD; only R is native.
    assert row["net_pnl_cad"] == pytest.approx(-210.0)


def test_the_three_readers_agree_with_each_other():
    import day_session_record as dsr
    import preference_trade_outcomes as report

    from_record = dsr._trades({"payload": {"trades": [USD_TRADE]}}, [])["rows"][0]["r_multiple"]
    from_report = float(report._canonical_r(USD_TRADE))
    assert from_record == pytest.approx(from_report)
    assert f"{from_record:+.1f}R" in _rule_loop_text()
