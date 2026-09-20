"""TJ-9Q items 3 and 4 - the stored rows move only through a tested CLI.

Red before the fix, on `claude/tj9q-questrade-instrument` at `3c9d837a`, except
where a test says GUARD. `python -m journal_reclassify` does not exist yet, so
most of these fail at the import.

WHAT IS AT STAKE IN THE LIVE STORE (read-only copy, 2026-09-19)
----------------------------------------------------------------
Reclassifying `security_type` changes `journal_identity.group_key`, which
changes `journal_store._new_trade_state`'s `trade_id` - a sha1 over the group
key, the anchor execution's uid, the direction and the occurrence. Every one of
the trader's **98 Questrade trades** is re-keyed by a full reclassify, and
**72 of them carry a `trade_annotations` row**: 1 confirmed, 33 provisional,
151 `needs_review` across both brokers, and `label_provenance` is PRESENT AND
EMPTY on all 185. The re-key pass (`JournalStore._rekey_annotations`) carries an
annotation onto the rebuilt trade that shares the most executions with the old
one - and REFUSES when two rebuilt trades tie, which is exactly what a sold put
that was opened, closed, opened and closed again produces.

`raw_executions.net_amount` is NULL on all 226 Questrade rows, so the tax
report already excludes every Questrade position ("a fill carries no
broker-stated amount"); the 387 IBKR Flex rows are the ones it adds up. The
guard below is therefore built on an IBKR position: the tax number is the
broker's, and a Questrade reclassify may not touch it.
"""

from __future__ import annotations

import hashlib
import sys
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
    BE_PUT,
    MARA_SHORT,
    QBTS_PUT,
    broker_pnl,
    ib_flex_execution,
    import_payloads,
    new_store,
    store_old_convention,
    trade_for,
)

SOLD_PUTS = AAOI_PUT + BE_PUT + QBTS_PUT


@pytest.fixture
def local_settings_restored():
    """The CLI's `--apply` flips a machine setting. Put it back afterwards.

    `local_settings.json` is shared by the whole pytest session, so a test that
    leaves the instrument flag on would silently change what every later test
    imports. The whole file is snapshotted rather than one key, because the key
    is the builder's to name.
    """
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


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cli(*args: str) -> int:
    import journal_reclassify

    return journal_reclassify.main(list(args))


def _annotation_rows(store) -> list[dict]:
    with store.connection() as conn:
        rows = conn.execute(
            "SELECT trade_id, setup_tags, tag_status, label_provenance, notes "
            "FROM trade_annotations ORDER BY trade_id"
        ).fetchall()
    return [{key: row[key] for key in row.keys()} for row in rows]


def _security_types(store, symbol: str) -> set[str]:
    with store.connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT security_type FROM raw_executions WHERE symbol = ?", (symbol,)
        ).fetchall()
    return {str(row[0]) for row in rows}


def _sides(store, symbol: str) -> set[str]:
    with store.connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT side FROM raw_executions WHERE symbol = ?", (symbol,)
        ).fetchall()
    return {str(row[0]) for row in rows}


def _seeded(tmp_path: Path):
    """A journal in the shape the trader's is in: old rows, old convention."""
    store = new_store(tmp_path)
    store_old_convention(store, SOLD_PUTS + [MARA_SHORT])
    return store, Path(store.db_path)


# --------------------------------------------------------------------------
# Item 3 - the dry run
# --------------------------------------------------------------------------


def test_the_dry_run_changes_not_one_byte_of_the_database(tmp_path):
    """Dry run is the DEFAULT, and a default that writes is not a dry run."""
    store, db = _seeded(tmp_path)
    before = _sha(db)

    assert _cli("--db", str(db)) == 0

    assert _sha(db) == before


def test_the_dry_run_asked_for_by_name_also_changes_nothing(tmp_path):
    store, db = _seeded(tmp_path)
    before = _sha(db)

    assert _cli("--db", str(db), "--dry-run") == 0

    assert _sha(db) == before


def test_the_dry_run_shows_the_before_and_the_after_of_a_sold_put(tmp_path, capsys):
    """The trader has to be able to read what `--apply` would do before it does
    it: the group key, the direction, the status and the money, on both sides of
    the change. AAOI goes from an OPEN LONG worth -3.97 to a CLOSED SHORT worth
    +241.03."""
    store, db = _seeded(tmp_path)

    assert _cli("--db", str(db)) == 0

    out = capsys.readouterr().out
    assert "AAOI18JUN26P120.00" in out
    assert "LONG" in out and "SHORT" in out
    assert "OPEN" in out and "CLOSED" in out
    assert "UNKNOWN" in out and "OPT" in out
    assert "241.03" in out
    assert any(token in out for token in ("-3.97", "-3.966", "-3.9657", "-3.965665"))


def test_the_dry_run_counts_the_open_unknown_positions_a_new_fill_could_close(tmp_path, capsys):
    """Item 4's gate input, and it is a COUNT of POSITIONS, not of fills.

    This fixture holds four open `UNKNOWN` Questrade positions - three sold puts
    and the MARA short - so the desk may not start classifying new fills yet.
    The wording is the builder's; the four names and the number are not.
    """
    import re

    store, db = _seeded(tmp_path)

    assert _cli("--db", str(db)) == 0

    out = capsys.readouterr().out
    for symbol in ("AAOI18JUN26P120.00", "BE2JUL26P260.00", "QBTS26JUN26P22.50", "MARA"):
        assert symbol in out, symbol
    assert re.search(r"\b4\b.{0,60}open|open.{0,60}\b4\b", out, re.IGNORECASE | re.DOTALL)
    assert "UNKNOWN" in out


# --------------------------------------------------------------------------
# Item 3 - --apply
# --------------------------------------------------------------------------


def test_apply_takes_a_timestamped_backup_before_it_writes(tmp_path, local_settings_restored):
    """A byte-exact copy of the database as it was, beside it, named after it.

    Not a promise about the format: a file whose bytes hash to the pre-apply
    database and whose name carries a run of digits is a backup a person can
    find and a person can restore.
    """
    store, db = _seeded(tmp_path)
    before = _sha(db)
    existing = {item.resolve() for item in db.parent.rglob("*")}

    assert _cli("--db", str(db), "--apply") == 0

    created = [item for item in db.parent.rglob("*") if item.resolve() not in existing]
    backups = [item for item in created if item.is_file() and _sha(item) == before]
    assert backups, f"no byte-exact backup among {[item.name for item in created]}"
    assert any(db.stem in item.name for item in backups)
    assert any(sum(char.isdigit() for char in item.name) >= 8 for item in backups)
    # And it really did change the database it backed up.
    assert _sha(db) != before


def test_apply_makes_the_stored_sold_puts_short_closed_and_worth_what_the_broker_says(
    tmp_path, local_settings_restored
):
    """The stored half of the golden: the same three positions, but re-keyed in
    place from rows that were written under the old convention."""
    store, db = _seeded(tmp_path)

    assert _cli("--db", str(db), "--apply") == 0

    for symbol, payloads in (
        ("AAOI18JUN26P120.00", AAOI_PUT),
        ("BE2JUL26P260.00", BE_PUT),
        ("QBTS26JUN26P22.50", QBTS_PUT),
    ):
        trade = trade_for(store, symbol)
        assert trade["direction"] == "SHORT", symbol
        assert trade["status"] == "CLOSED", symbol
        assert trade["security_type"] == "OPT", symbol
        assert trade["net_pnl"] == pytest.approx(broker_pnl(payloads), abs=1e-4), symbol


def test_apply_leaves_every_broker_stated_amount_and_the_tax_number_alone(
    tmp_path, local_settings_restored
):
    """The tax number is the BROKER's: it is summed from `net_amount`, which
    this packet never writes and never recomputes.

    The reportable positions and the account totals are identical across the
    apply. The EXCLUDED list is deliberately not compared: a sold put that
    stops being read as 4 long contracts stops being excluded for "still open"
    and starts being excluded for "a fill carries no broker-stated amount" -
    a better sentence about the same zero dollars.
    """
    import journal_tax_report

    store, db = _seeded(tmp_path)
    store.upsert_executions(
        [
            ib_flex_execution(
                "FLEX-OPEN", symbol="AAOI", side="BUY", quantity=100, price=10.0,
                commission=1.0, net_amount=-1001.0, timestamp="2026-06-10T10:30:00-07:00",
            ),
            ib_flex_execution(
                "FLEX-CLOSE", symbol="AAOI", side="SELL", quantity=100, price=11.0,
                commission=-2.17, net_amount=1102.17, timestamp="2026-06-11T10:30:00-07:00",
            ),
        ]
    )
    store.rebuild_trades(refresh_tags=False)

    def amounts() -> dict[str, object]:
        with store.connection() as conn:
            return {
                str(row[0]): row[1]
                for row in conn.execute("SELECT execution_uid, net_amount FROM raw_executions")
            }

    before_amounts = amounts()
    before_report = journal_tax_report.build_tax_report(store, year=2026)

    assert _cli("--db", str(db), "--apply") == 0

    after_report = journal_tax_report.build_tax_report(store, year=2026)
    assert amounts() == before_amounts
    # LEAD AMENDMENT 2026-09-19: the tester wrote `["reported"]`, a key
    # `journal_tax_report.build_tax_report` has never returned (its keys are
    # `positions`, `positions_reported`, `by_account`, `realised_cad`,
    # `positions_excluded`, `excluded`, `source`). The intent - the tax number does
    # not move - is pinned through the keys that exist, and more of them.
    assert after_report["positions_reported"] == before_report["positions_reported"]
    assert after_report["realised_cad"] == before_report["realised_cad"]
    assert after_report["positions_excluded"] == before_report["positions_excluded"]
    assert after_report["by_account"] == before_report["by_account"]
    # And the IBKR commission credit is still a credit.
    with store.connection() as conn:
        row = conn.execute(
            "SELECT commission FROM raw_executions WHERE execution_uid LIKE '%FLEX-CLOSE'"
        ).fetchone()
    assert float(row[0]) == pytest.approx(-2.17)


def test_a_confirmed_tag_and_its_provenance_ride_across_the_re_key(
    tmp_path, local_settings_restored
):
    """The trader owns `trade_annotations`, and a re-key is the machine's doing.

    The AAOI position's `trade_id` must change - its group key moves from
    UNKNOWN to OPT - and the trader's confirmed words, their note and the
    provenance of the label must arrive on the new id unaltered.
    """
    import trade_origin

    store, db = _seeded(tmp_path)
    old_id = trade_for(store, "AAOI18JUN26P120.00")["trade_id"]
    store.save_trade_annotation(
        old_id,
        setup_tags="sold put into strength",
        notes="kept the premium",
        label_provenance=trade_origin.SAME_SESSION,
    )

    assert _cli("--db", str(db), "--apply") == 0

    rows = _annotation_rows(store)
    assert len(rows) == 1
    row = rows[0]
    assert row["trade_id"] != old_id
    assert row["trade_id"] == trade_for(store, "AAOI18JUN26P120.00")["trade_id"]
    assert row["setup_tags"] == "sold put into strength"
    assert row["notes"] == "kept the premium"
    assert row["tag_status"] == "confirmed"
    assert row["label_provenance"] == trade_origin.SAME_SESSION


def test_an_older_annotation_arrives_with_its_empty_provenance_still_empty(
    tmp_path, local_settings_restored
):
    """Every one of the 185 annotation rows in the live store has
    `label_provenance` PRESENT AND EMPTY - the column was added after they were
    written. Empty means UNRECORDED, and a re-key may not fill it in: a
    provenance invented during a maintenance pass would be a claim about when
    the trader made a call, made by a script that was not there.
    """
    store, db = _seeded(tmp_path)
    old_id = trade_for(store, "AAOI18JUN26P120.00")["trade_id"]
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO trade_annotations(trade_id, setup_tags, notes, updated_at, "
            "tag_status, label_provenance) VALUES(?, ?, '', ?, 'confirmed', '')",
            (old_id, "sold put into strength", "2026-06-15T10:00:00-07:00"),
        )

    assert _cli("--db", str(db), "--apply") == 0

    rows = _annotation_rows(store)
    assert len(rows) == 1
    assert rows[0]["label_provenance"] == ""
    assert rows[0]["setup_tags"] == "sold put into strength"
    assert rows[0]["trade_id"] == trade_for(store, "AAOI18JUN26P120.00")["trade_id"]


def test_a_trade_whose_annotation_cannot_be_carried_is_refused_not_dropped(
    tmp_path, local_settings_restored
):
    """Sold, bought back, sold again, bought back again - one contract, four
    fills, interleaved.

    Today those four fills are one OPEN position. Corrected, they are TWO closed
    round trips, each holding two of the old position's four executions - so
    `_rekey_annotations` sees a TIE and will not guess which of them the
    trader's note belongs to. The CLI's answer is to leave that position exactly
    as it found it and say so; the answer it may never give is to reclassify it
    and let the annotation fall off.
    """
    interleaved = [
        dict(AAOI_SELL_1, timestamp="2026-06-10T12:53:34.905000-04:00"),
        dict(AAOI_BUY_BACK_1, timestamp="2026-06-10T13:10:00.000000-04:00"),
        dict(AAOI_SELL_2, timestamp="2026-06-11T10:05:00.000000-04:00"),
        dict(AAOI_BUY_BACK_2, timestamp="2026-06-11T14:22:00.000000-04:00"),
    ]
    store = new_store(tmp_path)
    store_old_convention(store, interleaved + BE_PUT)
    db = Path(store.db_path)
    old_id = trade_for(store, "AAOI18JUN26P120.00")["trade_id"]
    store.save_trade_annotation(
        old_id, setup_tags="sold put into strength", notes="", label_provenance="",
    )
    before = _annotation_rows(store)

    assert _cli("--db", str(db), "--apply") == 0

    after = _annotation_rows(store)
    assert len(after) == len(before) == 1
    live_ids = {str(row["trade_id"]) for row in store.list_trades()}
    assert after[0]["trade_id"] in live_ids, "the trader's tag was left pointing at nothing"
    assert after[0]["setup_tags"] == "sold put into strength"
    # Refused: this position's rows are left exactly as they were found.
    assert _security_types(store, "AAOI18JUN26P120.00") == {"UNKNOWN"}
    assert _sides(store, "AAOI18JUN26P120.00") == {"STO", "BTC"}
    # And the refusal is one position, not the whole run: BE was reclassified.
    assert _security_types(store, "BE2JUL26P260.00") == {"OPT"}
    assert trade_for(store, "BE2JUL26P260.00")["direction"] == "SHORT"


# --------------------------------------------------------------------------
# Item 4 - no half-applied desk
# --------------------------------------------------------------------------


def test_the_instrument_switch_ships_off():
    """One named constant, and its shipped value is the old behaviour.

    `journal_importers` is where it lives because that is the module the packet
    puts the Questrade seam in; the CLI's `--apply` turns it on through
    `local_settings`, which is why the constant is the DEFAULT and never the
    effective value.
    """
    import journal_importers

    assert journal_importers.QUESTRADE_INSTRUMENT_FROM_SYMBOL is False


def test_with_the_switch_off_a_new_fill_is_stored_exactly_as_it_is_today(tmp_path):
    """Byte-for-byte the same stored row, and the same assembled trade id, as a
    row written by the code that is on the desk right now.

    The left-hand store is written by hand from the live values; the right-hand
    one is written by the real import seam. While the switch is off they must
    agree on every field that decides identity or money.
    """
    old = new_store(tmp_path / "old")
    store_old_convention(old, AAOI_PUT)
    fresh = new_store(tmp_path / "fresh")
    import_payloads(fresh, AAOI_PUT)

    def rows(store):
        with store.connection() as conn:
            return [
                {
                    key: row[key]
                    for key in (
                        "execution_uid", "symbol", "security_type", "side",
                        "quantity", "price", "commission", "fees", "multiplier",
                        "currency", "timestamp", "trade_date", "net_amount",
                    )
                }
                for row in conn.execute("SELECT * FROM raw_executions ORDER BY execution_uid")
            ]

    assert rows(fresh) == rows(old)
    assert trade_for(fresh, "AAOI18JUN26P120.00")["trade_id"] == (
        trade_for(old, "AAOI18JUN26P120.00")["trade_id"]
    )


def test_a_new_closing_fill_joins_the_old_position_rather_than_splitting_it(tmp_path):
    """GUARD, green today and the reason the switch exists.

    A position opened under the old convention and a fill that arrives under the
    new one must never become two positions. While the switch is off the new
    fill lands in the same group - one trade for the contract, not two - even
    though it does not yet close it.
    """
    store = new_store(tmp_path)
    store_old_convention(store, [AAOI_SELL_1, AAOI_SELL_2])
    import_payloads(store, [AAOI_BUY_BACK_1])

    trade = trade_for(store, "AAOI18JUN26P120.00")

    assert trade["security_type"] == "UNKNOWN"


def test_after_apply_a_new_closing_fill_closes_the_position_it_belongs_to(
    tmp_path, local_settings_restored
):
    """The other side of item 4: once the stored rows have moved, the switch is
    on and the next fill closes the position instead of starting a second one.

    Two contracts sold under the old convention, `--apply`, then the two
    buy-backs arriving as new fills: one CLOSED short worth what the broker
    says, never two positions in two conventions.
    """
    store = new_store(tmp_path)
    store_old_convention(store, [AAOI_SELL_1, AAOI_SELL_2])
    db = Path(store.db_path)

    assert _cli("--db", str(db), "--apply") == 0
    import_payloads(store, [AAOI_BUY_BACK_1, AAOI_BUY_BACK_2])

    trade = trade_for(store, "AAOI18JUN26P120.00")
    assert trade["security_type"] == "OPT"
    assert trade["direction"] == "SHORT"
    assert trade["status"] == "CLOSED"
    assert trade["quantity_closed"] == pytest.approx(2.0)
    assert trade["net_pnl"] == pytest.approx(broker_pnl(AAOI_PUT), abs=1e-4)
