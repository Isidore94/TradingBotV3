"""P8 B4: which recent trades still lack a stop, a confirmed setup or a thesis."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

from tj14b_support import add_round_trip, new_store  # noqa: E402

TODAY = date(2026, 9, 25)


def _trade(trade_id, opened_at, **extra):
    row = {
        "trade_id": trade_id,
        "symbol": trade_id.upper(),
        "opened_at": opened_at,
        "trade_date": opened_at[:10],
        "status": "CLOSED",
        "setup_tags": "",
        "tag_status": "confirmed",
        "notes": "",
        "planned_stop": None,
    }
    row.update(extra)
    return row


def test_rows_are_oldest_first_with_what_each_misses():
    import journal_missing_inputs as mi

    trades = [
        _trade("new", "2026-09-20T07:31:00", planned_stop=9.5, setup_tags="vwap_bounce", notes="held the 20"),
        _trade("mid", "2026-09-10T07:31:00", planned_stop=9.5),
        _trade("old", "2026-09-01T07:31:00"),
    ]
    result = mi.missing_inputs(trades, {}, today=TODAY)

    assert [row["trade_id"] for row in result["rows"]] == ["old", "mid"]
    assert result["rows"][0] == {
        "trade_id": "old",
        "symbol": "OLD",
        "opened_at": "2026-09-01T07:31:00",
        "trade_date": "2026-09-01",
        "missing": ["stop", "setup", "thesis"],
    }
    assert result["rows"][1]["missing"] == ["setup", "thesis"]
    assert result["counts"] == {
        "stop": 1,
        "setup": 2,
        "thesis": 2,
        "trades": 2,
        "stop_or_setup": 2,
    }
    assert result["since"] == "2026-08-26"


def test_a_trade_opened_before_the_window_is_left_out():
    import journal_missing_inputs as mi

    trades = [_trade("gone", "2026-08-25T07:31:00"), _trade("in", "2026-08-26T07:31:00")]
    result = mi.missing_inputs(trades, {}, today=TODAY, since_days=30)
    assert [row["trade_id"] for row in result["rows"]] == ["in"]
    assert mi.missing_inputs(trades, {}, today=TODAY, since_days=31)["counts"]["trades"] == 2


def test_a_provisional_or_needs_review_tag_is_not_a_setup():
    import journal_missing_inputs as mi

    trades = [
        _trade("prov", "2026-09-10T07:31:00", setup_tags="vwap_bounce", tag_status="provisional"),
        _trade("review", "2026-09-11T07:31:00", setup_tags="vwap_bounce", tag_status="needs_review"),
        _trade("mine", "2026-09-12T07:31:00", setup_tags="vwap_bounce", tag_status="confirmed"),
    ]
    rows = {row["trade_id"]: row["missing"] for row in mi.missing_inputs(trades, {}, today=TODAY)["rows"]}
    assert "setup" in rows["prov"]
    assert "setup" in rows["review"]
    assert "setup" not in rows["mine"]


def test_absent_data_counts_as_missing_and_nothing_is_inferred():
    """A row with no annotation columns at all misses all three; a blank string
    stop or whitespace notes are not answers."""
    import journal_missing_inputs as mi

    bare = {"trade_id": "bare", "symbol": "B", "opened_at": "2026-09-10T07:31:00"}
    blanks = _trade("blank", "2026-09-11T07:31:00", planned_stop="", notes="   ", setup_tags="")
    rows = {r["trade_id"]: r["missing"] for r in mi.missing_inputs([bare, blanks], None, today=TODAY)["rows"]}
    assert rows == {"bare": ["stop", "setup", "thesis"], "blank": ["stop", "setup", "thesis"]}


def test_annotations_win_over_the_row_and_mentor_answers_close_a_field():
    import journal_missing_inputs as mi

    trades = [_trade("a", "2026-09-10T07:31:00")]
    annotations = {
        "a": {
            "planned_stop": 9.5,
            "setup_tags": "vwap_bounce",
            "tag_status": "confirmed",
            "answered": ["thesis"],
        }
    }
    assert mi.missing_inputs(trades, annotations, today=TODAY)["rows"] == []
    only_answer = {"a": {"answered": ["stop"]}}
    assert mi.missing_inputs(trades, only_answer, today=TODAY)["rows"][0]["missing"] == ["setup", "thesis"]


def test_the_chip_counts_stop_or_setup_and_points_at_the_oldest_such_trade():
    import journal_missing_inputs as mi

    trades = [
        _trade("thesis_only", "2026-09-01T07:31:00", planned_stop=9.5, setup_tags="x"),
        _trade("no_stop", "2026-09-05T07:31:00", setup_tags="x", notes="n"),
        _trade("no_setup", "2026-09-06T07:31:00", planned_stop=9.5, notes="n"),
    ]
    result = mi.missing_inputs(trades, {}, today=TODAY)
    assert result["counts"]["stop_or_setup"] == 2
    assert mi.oldest_chip_row(result)["trade_id"] == "no_stop"
    assert mi.chip_text(result) == "Inputs: 2 trades missing stop/setup"

    one = mi.missing_inputs(trades[1:2], {}, today=TODAY)
    assert mi.chip_text(one) == "Inputs: 1 trade missing stop/setup"
    none = mi.missing_inputs(trades[:1], {}, today=TODAY)
    assert mi.chip_text(none) == ""
    assert mi.oldest_chip_row(none) is None
    assert mi.chip_text(None) == ""


def test_load_reads_the_store_and_the_mentor_answers(tmp_path):
    import journal_missing_inputs as mi
    import trade_mentor_trade_check as check

    store = new_store(tmp_path)
    old = add_round_trip(store, "AAA", day="2026-09-10", entry_hour=7)
    new = add_round_trip(store, "BBB", day="2026-09-11", entry_hour=7)
    store.save_risk_fields(new, planned_stop=9.5)
    check.save_answers(store, old, {"thesis": {"state": check.ANSWER_NOT_SUPPLIED, "text": "bounce"}})

    result = mi.load(store, today=TODAY)

    rows = {row["trade_id"]: row["missing"] for row in result["rows"]}
    assert rows == {old: ["stop", "setup"], new: ["setup", "thesis"]}
    assert [row["trade_id"] for row in result["rows"]] == [old, new]
    assert mi.oldest_chip_row(result)["trade_id"] == old


def _contents(db):
    import sqlite3

    conn = sqlite3.connect(db)
    try:
        # The store's normal open re-stamps `meta.last_migration_at`; every
        # other row must be untouched by a read.
        return sorted(line for line in conn.iterdump() if "last_migration_at" not in line)
    finally:
        conn.close()


def test_the_cli_summary_prints_the_counts_and_writes_nothing(tmp_path, capsys):
    from datetime import timedelta

    import journal_missing_inputs as mi

    store = new_store(tmp_path)
    day = (date.today() - timedelta(days=3)).isoformat()
    trade_id = add_round_trip(store, "AAA", day=day, entry_hour=7)
    db = store.db_path
    before = _contents(db)

    assert mi.main(["--summary", "--db", str(db)]) == 0

    out = capsys.readouterr().out
    assert "missing an input: 1" in out
    assert "missing stop:   1" in out
    assert "missing setup:  1" in out
    assert "missing thesis: 1" in out
    assert "missing stop or setup (the chip): 1" in out
    assert trade_id not in out  # --summary prints counts, not rows
    # Read-only: the store's normal open may touch the file, never its rows.
    assert _contents(db) == before

    assert mi.main(["--db", str(db)]) == 0
    assert trade_id in capsys.readouterr().out
    assert _contents(db) == before
