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


# ---------------------------------------------------------------------------
# day_session_record migration: records written with the CAD R
# ---------------------------------------------------------------------------
SESSION = "2026-09-22"


def _built():
    from datetime import datetime, timedelta, timezone

    return datetime(2026, 9, 23, 1, 0, tzinfo=timezone(timedelta(hours=-4)))


def _new_record():
    import day_session_record as dsr

    return dsr.build_record(SESSION, {"payload": {"session_date": SESSION, "trades": [USD_TRADE]}}, built_at=_built())


def _write_old_shape(root: Path) -> dict:
    """The record exactly as the CAD-R code wrote it: no `r_definition`, CAD R, its own hash."""
    import json

    import day_session_record as dsr

    old = _new_record()
    old.pop("r_definition", None)
    old["trades"]["rows"][0]["r_multiple"] = CAD_R
    old["content_hash"] = dsr._content_hash(old)
    path = dsr.record_path(SESSION, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(old, sort_keys=True), encoding="utf-8")
    return old


def test_a_new_record_says_its_r_is_native():
    import day_session_record as dsr

    assert _new_record()["r_definition"] == dsr.R_DEFINITION == "native"


def test_an_old_cad_record_still_loads_with_native_r(tmp_path):
    import day_session_record as dsr

    _write_old_shape(tmp_path)
    record = dsr.read_record(SESSION, root=tmp_path)
    assert record is not None, "an old-shape record is not corrupt"
    assert record["r_definition"] == "native"
    assert record["trades"]["rows"][0]["r_multiple"] == pytest.approx(NATIVE_R)


def test_a_rebuild_replaces_the_old_cad_record(tmp_path):
    import json

    import day_session_record as dsr

    old = _write_old_shape(tmp_path)
    new = _new_record()
    assert new["content_hash"] != old["content_hash"]
    result = dsr.write_record(new, root=tmp_path)
    assert result["changed"] is True
    on_disk = json.loads(dsr.record_path(SESSION, root=tmp_path).read_text(encoding="utf-8"))
    assert on_disk["r_definition"] == "native"
    assert on_disk["content_hash"] == new["content_hash"]
    assert on_disk["trades"]["rows"][0]["r_multiple"] == pytest.approx(NATIVE_R)


def test_a_week_built_over_an_old_record_uses_native_r(tmp_path):
    import day_session_record as dsr

    _write_old_shape(tmp_path)
    week = dsr.week_key(SESSION)
    dsr.write_week(week, root=tmp_path, built_at=_built())
    body = dsr._read_json(dsr.week_path(week, root=tmp_path))
    assert body["r_definition"] == "native"
    assert body["by_origin"][0]["avg_r"] == pytest.approx(NATIVE_R)


def test_an_old_week_file_is_read_and_rebuilt_as_native(tmp_path):
    import json

    import day_session_record as dsr
    import week_coach

    _write_old_shape(tmp_path)
    week = dsr.week_key(SESSION)
    records = [dsr._read_json(dsr.record_path(SESSION, root=tmp_path))]
    stale = dsr.build_week(week, records, built_at=_built())
    stale.pop("r_definition", None)
    for row in stale["by_origin"]:
        row["avg_r"] = CAD_R
    dsr.week_path(week, root=tmp_path).write_text(json.dumps(stale), encoding="utf-8")

    loaded = week_coach.load_week(week, root=tmp_path)
    assert loaded["r_definition"] == "native"
    assert loaded["by_origin"][0]["avg_r"] == pytest.approx(NATIVE_R)

    assert dsr.stale_weeks(root=tmp_path) == [week]
    dsr.refresh_stale_weeks(root=tmp_path, built_at=_built())
    assert dsr.stale_weeks(root=tmp_path) == []
    body = dsr._read_json(dsr.week_path(week, root=tmp_path))
    assert body["by_origin"][0]["avg_r"] == pytest.approx(NATIVE_R)
