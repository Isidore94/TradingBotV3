"""TJ-9Q, the builder's own tests - the edges the tester's set does not cover.

Added 2026-09-19 alongside the fix. Nothing here weakens or replaces one of the
tester's assertions; each one is a fact the packet asks for that no red test
pins:

* the tax number across ``--apply``, read through the keys
  ``journal_tax_report.build_tax_report`` actually returns (the tester's
  ``test_apply_leaves_every_broker_stated_amount_and_the_tax_number_alone``
  asks for ``report["reported"]``, which that function has never returned - see
  the handoff);
* the CSV statement path, which shares ``normalize_side`` and therefore shares
  the switch: with it off an ``STO`` line is still dropped, with it on the same
  line is imported;
* a stored ``COV`` row's cash, which is corrected in
  ``journal_file_authority`` itself rather than only through a re-import;
* the classifier's refusals - a disagreement between the symbol and the side is
  ``UNKNOWN``, not a guess;
* the CLI's two refusals: the live data folder, and a journal somebody else is
  writing.
"""

from __future__ import annotations

import json
import sys
import threading
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tj9q_support import (  # noqa: E402
    AAOI_PUT,
    ACCOUNT_NUMBER,
    BE_PUT,
    QBTS_PUT,
    SMPL_COVER,
    ib_flex_execution,
    new_store,
    store_old_convention,
)

SOLD_PUTS = AAOI_PUT + BE_PUT + QBTS_PUT


@pytest.fixture
def local_settings_restored():
    """``--apply`` writes a machine setting. Put the whole file back after."""
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


def _seeded(tmp_path: Path):
    store = new_store(tmp_path)
    store_old_convention(store, SOLD_PUTS)
    return store, Path(store.db_path)


# --------------------------------------------------------------------------
# The tax number, through the keys the tax report actually returns
# --------------------------------------------------------------------------


def test_the_tax_number_is_the_same_before_and_after_the_apply(tmp_path, local_settings_restored):
    """``build_tax_report`` sums ``raw_executions.net_amount``. This packet never
    writes that column, so every number it returns is identical across the run.

    The EXCLUDED list is deliberately not compared: a sold put that stops being
    read as four long contracts stops being excluded for "still open" and starts
    being excluded for "a fill carries no broker-stated amount" - a better
    sentence about the same zero dollars.
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
    before = journal_tax_report.build_tax_report(store, year=2026)

    assert _cli("--db", str(db), "--apply") == 0

    after = journal_tax_report.build_tax_report(store, year=2026)
    assert after["positions"] == before["positions"]
    assert after["positions_reported"] == before["positions_reported"]
    assert after["by_account"] == before["by_account"]
    assert after["realised_cad"] == before["realised_cad"]
    assert after["source"] == before["source"]
    # And the IBKR commission CREDIT is still a credit.
    with store.connection() as conn:
        row = conn.execute(
            "SELECT commission FROM raw_executions WHERE execution_uid LIKE '%FLEX-CLOSE'"
        ).fetchone()
    assert float(row[0]) == pytest.approx(-2.17)


# --------------------------------------------------------------------------
# The blast radius: the CSV statement path shares the side map
# --------------------------------------------------------------------------


def _statement_row(action: str):
    from journal_statement_import import StatementRow

    return StatementRow(
        sequence=1,
        transaction_date=date(2026, 6, 10),
        settlement_date=date(2026, 6, 12),
        action=action,
        symbol="AAOI",
        description="AAOI CALL 18JUN26 120.00",
        quantity=1.0,
        price=1.45,
        gross_amount=145.0,
        commission=0.99,
        net_amount=144.01,
        currency="USD",
        account_number=ACCOUNT_NUMBER,
        activity_type="Trades",
        account_type="Margin",
    )


def test_with_the_switch_off_a_statement_line_that_says_sto_is_still_dropped(monkeypatch):
    """Today ``_execution_from_row`` drops any line whose normalized side is not
    BUY or SELL, and ``STO`` was not in the map - so the line was dropped. While
    the switch is off it still is, because ``fill_signature`` carries the
    normalized side: a row imported under one spelling would re-import as a
    second execution under the other."""
    import journal_statement_import

    monkeypatch.setattr(
        journal_statement_import, "questrade_instrument_from_symbol_enabled", lambda: False
    )

    assert journal_statement_import._execution_from_row(_statement_row("STO")) is None
    assert journal_statement_import._execution_from_row(_statement_row("BTC")) is None
    assert journal_statement_import._execution_from_row(_statement_row("Cov")) is None
    # The half of the vocabulary that always worked is untouched either way.
    assert journal_statement_import._execution_from_row(_statement_row("Buy")) is not None


def test_with_the_switch_on_the_same_statement_line_is_a_real_fill(monkeypatch):
    """It is a fill the trader really made, and once the stored rows are on the
    corrected vocabulary there is nothing left for it to collide with."""
    import journal_statement_import

    monkeypatch.setattr(
        journal_statement_import, "questrade_instrument_from_symbol_enabled", lambda: True
    )

    sold = journal_statement_import._execution_from_row(_statement_row("STO"))
    bought_back = journal_statement_import._execution_from_row(_statement_row("BTC"))
    covered = journal_statement_import._execution_from_row(_statement_row("Cov"))

    assert sold is not None and sold.side == "SELL"
    assert bought_back is not None and bought_back.side == "BUY"
    assert covered is not None and covered.side == "BUY"
    # The money the statement stated is carried across untouched.
    assert sold.net_amount == pytest.approx(144.01)
    assert sold.commission == pytest.approx(0.99)


# --------------------------------------------------------------------------
# A stored cover, read by the comparison that decides whose day it is
# --------------------------------------------------------------------------


def test_a_stored_cover_costs_cash_even_before_the_rows_are_moved(tmp_path):
    """The 25 covers in the live journal say ``COV`` today and will until the
    trader runs ``--apply``. ``_BUY_SIDES`` held ``COVER`` and not ``COV``, so
    every one of them was counted as money coming IN - and this is the read that
    decides whether a broker's own statement disagrees with us. Correcting the
    word corrects the money for the rows that are already stored."""
    from journal_file_authority import signed_cash

    store = new_store(tmp_path)
    store_old_convention(store, [SMPL_COVER])
    with store.connection() as conn:
        row = conn.execute(
            "SELECT * FROM raw_executions WHERE execution_uid = ?",
            (f"QT:{ACCOUNT_NUMBER}:{SMPL_COVER['id']}",),
        ).fetchone()
    stored = {key: row[key] for key in row.keys()}

    assert stored["side"] == "COV"  # not yet moved: this is today's row
    assert signed_cash(stored) == pytest.approx(-366.9845, abs=1e-9)


# --------------------------------------------------------------------------
# The classifier refuses rather than guesses
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symbol",
    ["AAOI18JUN26P120.00", "BE2JUL26P260.00", "QBTS26JUN26P22.50", "QQQ26JUN26P700.00"],
)
def test_the_four_recorded_contracts_are_read_as_contracts(symbol):
    from journal_importers import classify_questrade_security_type

    assert classify_questrade_security_type({"symbol": symbol, "side": "STO"}) == "OPT"


@pytest.mark.parametrize("symbol", ["MARA", "SMPL", "DKS", "BRK.B", "SPY"])
def test_a_plain_ticker_with_an_equity_side_is_stock(symbol):
    from journal_importers import classify_questrade_security_type

    assert classify_questrade_security_type({"symbol": symbol, "side": "Short"}) == "STK"


def test_a_symbol_and_a_side_that_disagree_stay_unknown():
    """Both fields say it independently, which is what makes them a check on
    each other. When they do not agree, the honest answer is that we do not
    know - an UNKNOWN position is visible and fixable, a wrong one is silent."""
    from journal_importers import classify_questrade_security_type

    assert classify_questrade_security_type({"symbol": "AAOI18JUN26P120.00", "side": "Buy"}) == "UNKNOWN"
    assert classify_questrade_security_type({"symbol": "MARA", "side": "BTO"}) == "UNKNOWN"
    assert classify_questrade_security_type({"symbol": "", "side": "Short"}) == "UNKNOWN"
    assert classify_questrade_security_type({"symbol": "MARA"}) == "UNKNOWN"


def test_nothing_is_read_from_a_tickers_length():
    """Four letters is not an option and two is not a stock. The only things
    read are the option spelling and the side vocabulary."""
    from journal_importers import classify_questrade_security_type, is_questrade_option_symbol

    assert not is_questrade_option_symbol("AAOI")
    assert not is_questrade_option_symbol("BE")
    assert is_questrade_option_symbol("BE2Jul26P260.00")
    assert classify_questrade_security_type({"symbol": "AAOI", "side": "Buy"}) == "STK"


# --------------------------------------------------------------------------
# The store's one additive seam
# --------------------------------------------------------------------------


def test_the_reclassify_seam_moves_three_fields_and_no_money(tmp_path):
    """``JournalStore.reclassify_executions`` is the only way stored rows move.
    It is handed what to set and it touches nothing else - and the rebuild it
    ends with is what makes the position read correctly."""
    store = new_store(tmp_path)
    store_old_convention(store, BE_PUT)
    uid = f"QT:{ACCOUNT_NUMBER}:{BE_PUT[0]['id']}"
    with store.connection() as conn:
        before = {key: row[key] for row in conn.execute(
            "SELECT * FROM raw_executions WHERE execution_uid = ?", (uid,)
        ) for key in row.keys()}

    report = store.reclassify_executions(
        [{"execution_uid": uid, "security_type": "OPT", "side": "SELL", "multiplier": 100.0}],
        refresh_tags=False,
    )

    with store.connection() as conn:
        after = {key: row[key] for row in conn.execute(
            "SELECT * FROM raw_executions WHERE execution_uid = ?", (uid,)
        ) for key in row.keys()}
    assert report["executions_updated"] == 1
    assert after["security_type"] == "OPT"
    assert after["side"] == "SELL"
    assert after["multiplier"] == pytest.approx(100.0)
    for column in (
        "quantity", "price", "commission", "fees", "gross_amount", "net_amount",
        "symbol", "raw_json", "execution_uid", "currency", "timestamp", "trade_date",
    ):
        assert after[column] == before[column], column


# --------------------------------------------------------------------------
# The CLI's two refusals
# --------------------------------------------------------------------------


def test_the_cli_refuses_to_apply_inside_the_live_data_folder(tmp_path, monkeypatch, capsys):
    """``--apply`` on the trader's own journal is the TRADER's act. No agent and
    no nightly job runs it, and the guard is the path, not a promise."""
    import journal_reclassify

    store, db = _seeded(tmp_path)
    before = db.read_bytes()
    monkeypatch.setattr(journal_reclassify, "LIVE_DATA_ROOTS", (tmp_path,))

    assert journal_reclassify.main(["--db", str(db), "--apply"]) == 2

    assert db.read_bytes() == before
    assert "trader" in capsys.readouterr().err
    # The dry run is still allowed to read it.
    assert journal_reclassify.main(["--db", str(db)]) == 0
    assert db.read_bytes() == before


def test_the_cli_refuses_to_apply_while_another_writer_holds_the_journal(tmp_path, capsys):
    """The desk and the nightly import write this file. A reclassify that ran
    beside one of them would rebuild from rows that are still arriving."""
    from local_writer_lock import local_writer_lock, lock_key_for_path

    store, db = _seeded(tmp_path)
    before = db.read_bytes()
    held = threading.Event()
    release = threading.Event()

    def _hold():
        with local_writer_lock(lock_key_for_path(db), timeout_seconds=5.0):
            held.set()
            release.wait(30.0)

    holder = threading.Thread(target=_hold, name="tj9q-lock-holder", daemon=True)
    holder.start()
    try:
        assert held.wait(10.0), "the lock holder never started"
        code = _cli("--db", str(db), "--apply")
    finally:
        release.set()
        holder.join(10.0)

    assert code == 3
    assert db.read_bytes() == before
    assert "writing this journal" in capsys.readouterr().err


# --------------------------------------------------------------------------
# What the dry run says about a number nobody stated
# --------------------------------------------------------------------------


def test_the_dry_run_says_the_broker_stated_no_net_amount_rather_than_printing_zero(
    tmp_path, capsys
):
    """Questrade sends neither ``netAmount`` nor ``grossAmount``. An absent
    number is reported as absent: printing 0.00 for it would be the machine
    inventing a broker statement."""
    store, db = _seeded(tmp_path)

    assert _cli("--db", str(db)) == 0

    out = capsys.readouterr().out
    assert "net_amount is ABSENT" in out
    assert "broker net_amount: not stated by Questrade" in out
    # No line pretends a stated amount of zero. (Strike prices are not money:
    # the check is on the money columns' own lines.)
    money_lines = [line for line in out.splitlines() if "net_amount" in line or "broker  cash" in line]
    assert money_lines
    for line in money_lines:
        assert "0.000000" not in line, line


def test_the_dry_run_lists_the_days_whose_file_authority_cash_moves(tmp_path, capsys):
    """Both halves of this change reach ``journal_file_authority.signed_cash``,
    so the trader sees which (account, day) comparisons move before any of them
    do. BE's sale on 2026-06-22 goes from 3.45 of cash to 345.00."""
    store, db = _seeded(tmp_path)

    assert _cli("--db", str(db)) == 0

    out = capsys.readouterr().out
    assert "File-authority signed cash that moves" in out
    assert "2026-06-22" in out
    assert "344.0029" in out


def test_the_dry_run_is_the_default_and_writes_no_local_setting(tmp_path):
    """The switch is turned on by ``--apply`` and by nothing else."""
    import journal_importers
    import project_paths

    store, db = _seeded(tmp_path)

    assert _cli("--db", str(db)) == 0

    assert (
        project_paths.get_local_setting(journal_importers.QUESTRADE_INSTRUMENT_SETTING, None)
        is None
    )
    assert journal_importers.questrade_instrument_from_symbol_enabled() is False


def test_apply_turns_the_switch_on_only_after_the_rebuild_verified(
    tmp_path, local_settings_restored
):
    """The key is the LAST line of the run. Until it is written, the desk and
    the nightly import store exactly what they stored yesterday."""
    import journal_importers
    import project_paths

    store, db = _seeded(tmp_path)
    assert journal_importers.questrade_instrument_from_symbol_enabled() is False

    assert _cli("--db", str(db), "--apply") == 0

    assert (
        project_paths.get_local_setting(journal_importers.QUESTRADE_INSTRUMENT_SETTING) is True
    )
    assert journal_importers.questrade_instrument_from_symbol_enabled() is True
    # And the rows really did move, in the same run.
    with store.connection() as conn:
        types = {
            str(row[0])
            for row in conn.execute("SELECT DISTINCT security_type FROM raw_executions")
        }
    assert types == {"OPT"}


def test_a_payload_that_states_its_type_is_still_believed_over_the_symbol(tmp_path):
    """``get_positions`` does send a ``securityType``, and a future endpoint may.
    The classifier is a fallback for the endpoint that sends none."""
    from journal_importers import classify_questrade_security_type

    payload = dict(AAOI_PUT[0])
    payload["securityType"] = "Stock"

    assert classify_questrade_security_type(payload) == "STK"
    assert json.loads(json.dumps(payload))["symbol"] == "AAOI18Jun26P120.00"
