"""TJ-9Q review round - the blocker and the five hardenings, 2026-09-19.

The reviewer reproduced every number on a copy of the live journal and returned
NO-GO on one blocker: **the import switch flipped ON even when a position had
been REFUSED**. A refused position keeps the old spelling, so the very next fill
that tried to close it opened a SECOND position for the same contract and the
refused one could never close. Each test below is the reviewer's own
reproduction, driven through ``main(argv)``.

The hardenings, one test each: the machine's trade-keyed rows are carried across
the re-key (23 of 24 AI enrichment rows and 98 of 216 note verdicts were being
left on dead trade ids); a read-only journal gets a report instead of a
traceback; ``--apply`` refuses while the overnight runner holds its lock;
``reclassify_executions`` refuses a row that is not a Questrade fill;
page one is short enough to decide on; and an interrupted apply is repaired by
running it again.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tj9q_support import (  # noqa: E402
    AAOI_BUY_BACK_1,
    AAOI_BUY_BACK_2,
    AAOI_PUT,
    AAOI_SELL_1,
    AAOI_SELL_2,
    ACCOUNT_NUMBER,
    BE_PUT,
    MARA_SHORT,
    import_payloads,
    ib_flex_execution,
    new_store,
    store_old_convention,
    trade_for,
)

#: The reviewer's exact shape: one contract, sold, bought back, sold again and
#: bought back again. One OPEN position today; two closed round trips once the
#: sides are right, each holding half of the old position's executions.
INTERLEAVED = [
    dict(AAOI_SELL_1, timestamp="2026-06-10T12:53:34.905000-04:00"),
    dict(AAOI_BUY_BACK_1, timestamp="2026-06-10T13:10:00.000000-04:00"),
    dict(AAOI_SELL_2, timestamp="2026-06-11T10:05:00.000000-04:00"),
    dict(AAOI_BUY_BACK_2, timestamp="2026-06-11T14:22:00.000000-04:00"),
]

#: A fill that arrives AFTER the apply - a new sale of the same contract.
NEW_SALE = dict(AAOI_SELL_1, id=7099, exchangeExecId="EXEC-7099",
                timestamp="2026-06-16T10:00:00.000000-04:00")


@pytest.fixture
def local_settings_restored():
    import project_paths

    path = Path(project_paths.LOCAL_SETTINGS_FILE)
    before = path.read_bytes() if path.is_file() else None
    try:
        yield
    finally:
        if before is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(before)
        project_paths.invalidate_local_settings_cache()


def _cli(*args: str) -> int:
    import journal_reclassify

    return journal_reclassify.main(list(args))


def _switch_value():
    import journal_importers
    import project_paths

    return project_paths.get_local_setting(journal_importers.QUESTRADE_INSTRUMENT_SETTING, None)


def _refused_journal(tmp_path: Path):
    """A journal holding the refusal case and one position that is fine."""
    store = new_store(tmp_path)
    store_old_convention(store, INTERLEAVED + BE_PUT)
    old_id = trade_for(store, "AAOI18JUN26P120.00")["trade_id"]
    store.save_trade_annotation(
        old_id, setup_tags="sold put into strength", notes="", label_provenance=""
    )
    return store, Path(store.db_path)


# --------------------------------------------------------------------------
# THE BLOCKER
# --------------------------------------------------------------------------


def test_a_refused_position_keeps_the_import_switch_off(tmp_path, local_settings_restored):
    """The blocker, in one sentence: a run that refused is not a clean run.

    AAOI is refused (its annotation cannot be carried without guessing), so the
    contract is still spelled the old way in `raw_executions`. Turning the
    import convention on now would file the next AAOI fill into a DIFFERENT
    group from the position it belongs to.
    """
    import journal_reclassify

    store, db = _refused_journal(tmp_path)

    code = _cli("--db", str(db), "--apply")

    assert code == journal_reclassify.EXIT_SWITCH_STAYED_OFF
    assert code != 0
    assert _switch_value() is None, "the switch was flipped on a run that refused a position"
    import journal_importers

    assert journal_importers.questrade_instrument_from_symbol_enabled() is False


def test_the_run_that_refused_says_so_and_names_the_position(
    tmp_path, capsys, local_settings_restored
):
    """The trader has to be told which position is holding the switch back."""
    store, db = _refused_journal(tmp_path)

    _cli("--db", str(db), "--apply")

    out = capsys.readouterr().out
    assert "REFUSED" in out
    assert "AAOI18JUN26P120.00" in out
    assert "OFF" in out
    assert "half in each convention" in out


def test_what_the_refused_run_did_reclassify_stays_reclassified(
    tmp_path, local_settings_restored
):
    """The rows that could move ARE moved and are correct - only the switch is
    held back. A run that rolled the good work back would teach the trader to
    dread running it."""
    store, db = _refused_journal(tmp_path)

    _cli("--db", str(db), "--apply")

    be = trade_for(store, "BE2JUL26P260.00")
    assert be["security_type"] == "OPT"
    assert be["direction"] == "SHORT"
    assert be["status"] == "CLOSED"
    with store.connection() as conn:
        aaoi = {
            (str(row[0]), str(row[1]))
            for row in conn.execute(
                "SELECT security_type, side FROM raw_executions WHERE symbol = ?",
                ("AAOI18JUN26P120.00",),
            )
        }
    assert aaoi == {("UNKNOWN", "STO"), ("UNKNOWN", "BTC")}


def test_after_a_refused_run_a_new_fill_does_not_split_the_contract(
    tmp_path, local_settings_restored
):
    """The reviewer's reproduction, end to end.

    With the switch left ON by a refused run, one new `STO` fill through the
    real import seam gave that contract TWO positions - `OPT SHORT OPEN 1/0`
    beside `UNKNOWN LONG OPEN 4/0` - and the refused one could never close.
    With the switch OFF the new fill lands in the position it belongs to.
    """
    store, db = _refused_journal(tmp_path)

    _cli("--db", str(db), "--apply")
    import_payloads(store, [NEW_SALE])

    matches = [row for row in store.list_trades() if row["symbol"] == "AAOI18JUN26P120.00"]
    assert len(matches) == 1, f"the contract was split into {len(matches)} positions"
    assert matches[0]["security_type"] == "UNKNOWN"


def test_an_open_unknown_position_left_behind_also_keeps_the_switch_off(
    tmp_path, local_settings_restored
):
    """The other route to the same split: a position this pass could not TYPE
    at all. Here the payload's symbol and side disagree, so the classifier
    refuses to guess and the position stays UNKNOWN and open."""
    import journal_reclassify

    store = new_store(tmp_path)
    confused = dict(MARA_SHORT, id=7098, exchangeExecId="EXEC-7098", side="BTO")
    store_old_convention(store, AAOI_PUT + [confused])

    code = _cli("--db", str(db_of(store)), "--apply")

    assert code == journal_reclassify.EXIT_SWITCH_STAYED_OFF
    assert _switch_value() is None
    # ...and the sold put still came out right.
    assert trade_for(store, "AAOI18JUN26P120.00")["direction"] == "SHORT"


def db_of(store) -> Path:
    return Path(store.db_path)


def test_a_clean_run_still_turns_the_switch_on(tmp_path, local_settings_restored):
    """Nothing refused and nothing left open and UNKNOWN: the whole point."""
    store = new_store(tmp_path)
    store_old_convention(store, AAOI_PUT + BE_PUT)

    assert _cli("--db", str(db_of(store)), "--apply") == 0

    assert _switch_value() is True


# --------------------------------------------------------------------------
# A. the machine's rows are carried, and counted
# --------------------------------------------------------------------------


def _machine_rows(store):
    with store.connection() as conn:
        live = {str(row[0]) for row in conn.execute("SELECT trade_id FROM trades")}
        enrichment = [str(row[0]) for row in conn.execute("SELECT trade_id FROM ai_trade_enrichment")]
        verdicts = [str(row[0]) for row in conn.execute("SELECT trade_id FROM note_lane_verdicts")]
    return live, enrichment, verdicts


def test_the_ai_narration_and_the_note_verdict_ride_across_the_re_key(
    tmp_path, local_settings_restored
):
    """`list_ai_enrichment` and the note-lane join match the LITERAL trade id,
    so a re-key that leaves them behind empties the Journal page's narration
    without saying a word. Measured on the live copy before the fix: 23 of 24
    enrichment rows and 98 of 216 note verdicts were left on dead ids."""
    store = new_store(tmp_path)
    store_old_convention(store, AAOI_PUT + BE_PUT)
    old_id = trade_for(store, "AAOI18JUN26P120.00")["trade_id"]
    store.save_ai_enrichment(
        trade_id=old_id,
        session_date="2026-06-15",
        summary="sold the put into strength and bought it back cheap",
        model="local",
    )
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO note_lane_verdicts(trade_id, note_lane_json, updated_at) "
            "VALUES(?, ?, ?)",
            (old_id, '{"lane": "trader_note"}', "2026-06-15T10:00:00-07:00"),
        )

    assert _cli("--db", str(db_of(store)), "--apply") == 0

    new_id = trade_for(store, "AAOI18JUN26P120.00")["trade_id"]
    assert new_id != old_id
    live, enrichment, verdicts = _machine_rows(store)
    assert enrichment == [new_id]
    assert verdicts == [new_id]
    assert store.list_ai_enrichment(new_id), "the Journal page would show no narration"
    assert set(enrichment) <= live and set(verdicts) <= live


def test_an_event_that_was_re_keyed_still_resolves_to_its_trade(
    tmp_path, local_settings_restored
):
    """`opportunity_events` is immutable by design and is never rewritten, so
    the carry is a `trade_aliases` row - the documented way an event reaches the
    trade it belongs to."""
    store = new_store(tmp_path)
    store_old_convention(store, AAOI_PUT)
    old_id = trade_for(store, "AAOI18JUN26P120.00")["trade_id"]
    with store.connection() as conn:
        events = [
            str(row[0])
            for row in conn.execute(
                "SELECT trade_id FROM opportunity_events WHERE trade_id = ?", (old_id,)
            )
        ]
    assert events, "the rebuild records its own lifecycle events"

    assert _cli("--db", str(db_of(store)), "--apply") == 0

    new_id = trade_for(store, "AAOI18JUN26P120.00")["trade_id"]
    assert store.resolve_trade_id(old_id) == new_id


def test_the_report_says_how_many_machine_rows_were_carried_and_how_many_were_not(
    tmp_path, capsys, local_settings_restored
):
    store = new_store(tmp_path)
    store_old_convention(store, AAOI_PUT)
    old_id = trade_for(store, "AAOI18JUN26P120.00")["trade_id"]
    store.save_ai_enrichment(trade_id=old_id, session_date="2026-06-15", summary="x")

    _cli("--db", str(db_of(store)), "--apply")

    out = capsys.readouterr().out
    assert "ai_trade_enrichment" in out
    assert "note_lane_verdicts" in out
    assert "opportunity_events" in out
    assert "carried" in out and "left where they were" in out


def test_a_position_whose_ai_narration_cannot_be_carried_is_refused_too(
    tmp_path, local_settings_restored
):
    """Nothing regenerates `ai_trade_enrichment`. A tie there is the same kind
    of loss as a tie on the trader's own tag, so it refuses the POSITION - the
    row stays reachable and the switch stays off - instead of either guessing or
    abandoning the whole repair. (The note-lane verdicts are DERIVED and the
    events are reached through an alias; only this one is gone for good.)"""
    import journal_reclassify

    store = new_store(tmp_path)
    store_old_convention(store, INTERLEAVED + BE_PUT)
    old_id = trade_for(store, "AAOI18JUN26P120.00")["trade_id"]
    store.save_ai_enrichment(
        trade_id=old_id, session_date="2026-06-11", summary="rolled the same strike twice"
    )

    assert _cli("--db", str(db_of(store)), "--apply") == journal_reclassify.EXIT_SWITCH_STAYED_OFF

    live, enrichment, _ = _machine_rows(store)
    assert enrichment == [old_id]
    assert set(enrichment) <= live, "the narration was left pointing at nothing"
    assert trade_for(store, "BE2JUL26P260.00")["security_type"] == "OPT"


def test_a_note_verdict_that_cannot_be_carried_is_dropped_not_left_to_rot(
    tmp_path, local_settings_restored
):
    """`refresh_auto_tags` already DELETEs a verdict whose trade is gone -
    "this table is derived and nothing downstream may read a row whose trade is
    gone" - so a verdict the overlap rule cannot place is dropped under that
    same rule, counted, and named. The note lane writes it again from the
    trader's own note."""
    store = new_store(tmp_path)
    store_old_convention(store, INTERLEAVED + BE_PUT)
    old_id = trade_for(store, "AAOI18JUN26P120.00")["trade_id"]
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO note_lane_verdicts(trade_id, note_lane_json, updated_at) "
            "VALUES(?, '{}', '2026-06-11T10:00:00-07:00')",
            (old_id,),
        )

    assert _cli("--db", str(db_of(store)), "--apply") in (0, 4)

    live, _, verdicts = _machine_rows(store)
    assert all(trade_id in live for trade_id in verdicts), verdicts
    # ...and the run was not abandoned over it: BE moved.
    assert trade_for(store, "BE2JUL26P260.00")["security_type"] == "OPT"


def test_a_position_that_only_gains_a_type_is_not_on_page_one_even_when_re_keyed(
    tmp_path, capsys
):
    """A re-key renames every trade in a position, and sorting by trade id then
    hands them back in a different order. Comparing the LIST rather than the SET
    of trades put nine positions on page one of the live journal where four
    belong. Two round trips in one symbol, none of whose money moves."""
    store = new_store(tmp_path)
    two_round_trips = [
        dict(MARA_SHORT, id=7101, exchangeExecId="EXEC-7101", side="Short", quantity=10,
             timestamp="2026-09-01T10:00:00.000000-04:00"),
        dict(MARA_SHORT, id=7102, exchangeExecId="EXEC-7102", side="Cov", quantity=10,
             timestamp="2026-09-01T11:00:00.000000-04:00"),
        dict(MARA_SHORT, id=7103, exchangeExecId="EXEC-7103", side="Short", quantity=5,
             timestamp="2026-09-02T10:00:00.000000-04:00"),
        dict(MARA_SHORT, id=7104, exchangeExecId="EXEC-7104", side="Cov", quantity=5,
             timestamp="2026-09-02T11:00:00.000000-04:00"),
    ]
    store_old_convention(store, AAOI_PUT + two_round_trips)

    _cli("--db", str(db_of(store)))

    out = capsys.readouterr().out
    head = out.split("DETAIL")[0]
    assert "AAOI18JUN26P120.00" in head
    assert "MARA" not in head, "an unchanged equity position reached page one"
    assert "money or direction moves: 1" in head


def test_the_same_trades_in_a_different_order_are_not_a_change(tmp_path):
    """The rule itself, deterministically: two trades, same direction, same
    status, same money, handed back the other way round by the rebuild. That is
    a position that did not change, and only its instrument type moved."""
    from journal_reclassify import _position_change

    def trade(trade_id, direction, pnl, security_type):
        return {
            "trade_id": trade_id,
            "broker": "QUESTRADE",
            "account_number": "A1",
            "symbol": "MARA",
            "security_type": security_type,
            "currency": "USD",
            "direction": direction,
            "status": "CLOSED",
            "net_pnl": pnl,
        }

    before = [trade("aaa", "SHORT", 12.5, "UNKNOWN"), trade("bbb", "LONG", -4.0, "UNKNOWN")]
    after_same = [trade("zzz", "LONG", -4.0, "STK"), trade("yyy", "SHORT", 12.5, "STK")]
    after_moved = [trade("zzz", "LONG", -4.0, "STK"), trade("yyy", "SHORT", 99.0, "STK")]

    assert _position_change(before, after_same) == "type"
    assert _position_change(before, after_moved) == "matters"
    assert _position_change(before, list(reversed(before))) == ""


def test_a_run_that_would_strand_more_machine_rows_is_refused_and_restored(
    tmp_path, monkeypatch, local_settings_restored
):
    """The verify step is what makes the carry a promise rather than a hope: if
    a table came out with MORE dead references than it went in with, the journal
    is put back byte for byte."""
    import journal_reclassify

    store = new_store(tmp_path)
    store_old_convention(store, AAOI_PUT)
    old_id = trade_for(store, "AAOI18JUN26P120.00")["trade_id"]
    store.save_ai_enrichment(trade_id=old_id, session_date="2026-06-15", summary="x")
    db = db_of(store)
    before = db.read_bytes()

    real = journal_reclassify._machine_dead_counts

    def _pretend(snapshot_store):
        counts = dict(real(snapshot_store))
        with snapshot_store.connection() as conn:
            typed = conn.execute(
                "SELECT COUNT(*) FROM raw_executions WHERE security_type = 'OPT'"
            ).fetchone()[0]
        if typed:  # the AFTER snapshot
            counts["ai_trade_enrichment"] = counts.get("ai_trade_enrichment", 0) + 5
        return counts

    monkeypatch.setattr(journal_reclassify, "_machine_dead_counts", _pretend)

    assert _cli("--db", str(db), "--apply") == journal_reclassify.EXIT_VERIFY_FAILED

    assert db.read_bytes() == before
    assert _switch_value() is None


# --------------------------------------------------------------------------
# B. a read-only journal gets a report, not a traceback
# --------------------------------------------------------------------------


def test_a_read_only_journal_can_still_be_read(tmp_path, capsys):
    """`shutil.copy2` preserves the read-only attribute, and sqlite then raises
    "attempt to write a readonly database" when the WORKING COPY is opened. The
    bit is cleared on the copy; the file the trader protected is never touched.
    """
    import os
    import stat as stat_module

    store = new_store(tmp_path)
    store_old_convention(store, AAOI_PUT)
    db = db_of(store)
    mode_before = db.stat().st_mode
    os.chmod(db, stat_module.S_IREAD)
    try:
        assert _cli("--db", str(db)) == 0
    finally:
        os.chmod(db, mode_before)

    out = capsys.readouterr().out
    assert "241.03" in out
    assert "Traceback" not in out
    # and it is still read-only afterwards - we changed nothing about it
    assert db.stat().st_mode == mode_before


# --------------------------------------------------------------------------
# C. what "busy" actually means
# --------------------------------------------------------------------------


def test_apply_refuses_while_the_overnight_runner_is_going(
    tmp_path, monkeypatch, capsys, local_settings_restored
):
    """The nightly journal import runs INSIDE the overnight runner, and nothing
    in it locks the journal file - so the per-path lock this tool takes would
    never have noticed it."""
    import journal_reclassify
    from ai_jobs.runner import RUNNER_LOCK_KEY
    from local_writer_lock import local_writer_lock

    store = new_store(tmp_path)
    store_old_convention(store, AAOI_PUT)
    db = db_of(store)
    before = db.read_bytes()
    monkeypatch.setattr(journal_reclassify, "_is_the_desks_journal", lambda path: True)
    monkeypatch.setattr(journal_reclassify, "_desk_is_running", lambda: False)

    held = threading.Event()
    release = threading.Event()

    def _hold():
        with local_writer_lock(RUNNER_LOCK_KEY, timeout_seconds=5.0):
            held.set()
            release.wait(30.0)

    holder = threading.Thread(target=_hold, name="tj9q-runner-lock", daemon=True)
    holder.start()
    try:
        assert held.wait(10.0)
        code = _cli("--db", str(db), "--apply")
    finally:
        release.set()
        holder.join(10.0)

    assert code == journal_reclassify.EXIT_BUSY
    assert db.read_bytes() == before
    assert "overnight AI runner" in capsys.readouterr().err


def test_apply_refuses_while_the_desk_is_open(
    tmp_path, monkeypatch, capsys, local_settings_restored
):
    import journal_reclassify

    store = new_store(tmp_path)
    store_old_convention(store, AAOI_PUT)
    db = db_of(store)
    before = db.read_bytes()
    monkeypatch.setattr(journal_reclassify, "_is_the_desks_journal", lambda path: True)
    monkeypatch.setattr(journal_reclassify, "_desk_is_running", lambda: True)
    monkeypatch.setattr(journal_reclassify, "_nightly_jobs_are_running", lambda: False)

    assert _cli("--db", str(db), "--apply") == journal_reclassify.EXIT_BUSY

    assert db.read_bytes() == before
    assert "Trading Desk is running" in capsys.readouterr().err


def test_a_scratch_copy_is_nobodys_journal_and_is_not_blocked(
    tmp_path, monkeypatch, local_settings_restored
):
    """A copy in a scratch directory is not the desk's journal, so a desk that
    happens to be open does not block it - otherwise this tool could never be
    tested on the case it exists for."""
    import journal_reclassify

    store = new_store(tmp_path)
    store_old_convention(store, AAOI_PUT)
    monkeypatch.setattr(journal_reclassify, "_desk_is_running", lambda: True)
    monkeypatch.setattr(journal_reclassify, "_nightly_jobs_are_running", lambda: True)

    assert journal_reclassify.busy_reasons(db_of(store)) == []
    assert _cli("--db", str(db_of(store)), "--apply") == 0


# --------------------------------------------------------------------------
# D. the store's seam refuses a row that is not a Questrade fill
# --------------------------------------------------------------------------


def test_the_reclassify_seam_refuses_a_row_from_another_broker(tmp_path):
    """This vocabulary is one broker's. An IBKR option re-typed by it would be
    a silent wrong number, so the seam checks rather than trusting its caller -
    and the SQL is constrained as well."""
    store = new_store(tmp_path)
    store.upsert_executions(
        [
            ib_flex_execution(
                "FLEX-1", symbol="AAOI", side="BUY", quantity=100, price=10.0,
                commission=1.0, net_amount=-1001.0, timestamp="2026-06-10T10:30:00-07:00",
            )
        ]
    )
    uid = f"IBKR:{ACCOUNT_NUMBER}:FLEX-1"

    with pytest.raises(ValueError, match="another broker"):
        store.reclassify_executions(
            [{"execution_uid": uid, "security_type": "OPT", "side": "SELL", "multiplier": 100.0}]
        )

    with store.connection() as conn:
        row = conn.execute(
            "SELECT security_type, side, multiplier FROM raw_executions WHERE execution_uid = ?",
            (uid,),
        ).fetchone()
    assert (str(row[0]), str(row[1]), float(row[2])) == ("STK", "BUY", 1.0)


# --------------------------------------------------------------------------
# E. page one is short enough to decide on
# --------------------------------------------------------------------------


def test_page_one_holds_the_decision_and_the_long_list_is_behind_verbose(
    tmp_path, capsys
):
    """609 lines of `UNKNOWN -> STK` is not a report a person reads. What
    changes money or direction, what was refused, what the switch will do and
    where the backup is come FIRST; the type-only moves collapse into a count.
    """
    store = new_store(tmp_path)
    store_old_convention(store, AAOI_PUT + BE_PUT + [MARA_SHORT])

    _cli("--db", str(db_of(store)))
    brief = capsys.readouterr().out
    _cli("--db", str(db_of(store)), "--verbose")
    full = capsys.readouterr().out

    head = brief.splitlines()[:30]
    joined = "\n".join(head)
    assert "WHAT CHANGES" in joined
    assert "AAOI18JUN26P120.00" in joined
    assert "241.03" in joined
    # MARA only gains a type: it is counted, not listed, until --verbose.
    assert "MARA" not in brief.split("DETAIL")[0]
    assert "only gain an instrument type" in brief
    assert len(full.splitlines()) > len(brief.splitlines())
    assert "MARA" in full
    for heading in ("REFUSED", "THE SWITCH", "BACKUP AND UNDO"):
        assert heading in brief


def test_page_one_says_where_the_backup_is_and_how_to_undo(
    tmp_path, capsys, local_settings_restored
):
    store = new_store(tmp_path)
    store_old_convention(store, AAOI_PUT)

    assert _cli("--db", str(db_of(store)), "--apply") == 0

    out = capsys.readouterr().out
    assert "BACKUP AND UNDO" in out
    assert ".pre-tj9q-" in out
    assert "copy that file back" in out


# --------------------------------------------------------------------------
# F. an interrupted apply is repaired by running it again
# --------------------------------------------------------------------------


def test_the_report_says_what_to_do_if_the_power_goes_out(tmp_path, capsys):
    store = new_store(tmp_path)
    store_old_convention(store, AAOI_PUT)

    _cli("--db", str(db_of(store)))

    out = capsys.readouterr().out
    assert "loses power" in out
    assert "run --apply again" in out


def test_running_apply_again_repairs_a_journal_whose_rows_moved_but_never_rebuilt(
    tmp_path, local_settings_restored
):
    """The interrupted state, made by hand: the rows carry the new spelling and
    the trades were never rebuilt, so the position still reads LONG and OPEN."""
    store = new_store(tmp_path)
    store_old_convention(store, AAOI_PUT)
    db = db_of(store)
    with store.connection() as conn:
        conn.execute(
            "UPDATE raw_executions SET security_type = 'OPT', multiplier = 100.0, "
            "side = CASE side WHEN 'STO' THEN 'SELL' WHEN 'BTC' THEN 'BUY' ELSE side END "
            "WHERE symbol = ?",
            ("AAOI18JUN26P120.00",),
        )
    assert trade_for(store, "AAOI18JUN26P120.00")["direction"] == "LONG"

    assert _cli("--db", str(db), "--apply") == 0

    trade = trade_for(store, "AAOI18JUN26P120.00")
    assert trade["direction"] == "SHORT"
    assert trade["status"] == "CLOSED"
    assert trade["net_pnl"] == pytest.approx(241.034335, abs=1e-6)
    assert _switch_value() is True
