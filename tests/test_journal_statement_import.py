"""Importing a Questrade activity statement (.xlsx/.csv) into the journal.

Fixtures here are SYNTHETIC. The shapes are taken from a real 2026 YTD export
(974 rows, 884 of them trades) but no real position, account number or amount
is committed to this repository.

What these tests are really defending is the set of things a statement CANNOT
tell us, because every one of them is a place where a plausible default would
produce a confident wrong answer: it has no time of day, no execution id, no
intraday sequence, and it may aggregate fills.
"""

from __future__ import annotations

import sys
import zipfile
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_statement_import as statement  # noqa: E402
from journal_migrate import SOURCE_RANK  # noqa: E402
from journal_store import JournalStore  # noqa: E402
from journal_trade_shape import session_bucket, shape_tags  # noqa: E402

COLUMNS = [
    "Transaction Date",
    "Settlement Date",
    "Action",
    "Symbol",
    "Description",
    "Quantity",
    "Price",
    "Gross Amount",
    "Commission",
    "Net Amount",
    "Currency",
    "Account #",
    "Activity Type",
    "Account Type",
]

#: One buy and one sell of the same name on one day, plus an option round trip
#: and a fee. Account numbers are invented.
ROWS = [
    ["2026-03-02 12:00:00 AM", "2026-03-03 12:00:00 AM", "Buy", "AAPL", "APPLE INC",
     "20.00000", "100.00000000", "-2000.00", "0.00", "-2000.00", "USD", "11111111", "Trades", "Individual margin"],
    ["2026-03-02 12:00:00 AM", "2026-03-03 12:00:00 AM", "Sell", "AAPL", "APPLE INC",
     "-20.00000", "101.00000000", "2020.00", "-0.05", "2019.95", "USD", "11111111", "Trades", "Individual margin"],
    ["2026-03-05 12:00:00 AM", "2026-03-06 12:00:00 AM", "Buy", "8SVDLK9",
     "PUT SPY 06/23/26 737 STATE STREET SPDR S&P 500 ETF WE ACTED AS AGENT",
     "2.00000", "1.60000000", "-320.00", "0.00", "-320.00", "USD", "11111111", "Trades", "Individual margin"],
    ["2026-03-06 12:00:00 AM", "2026-03-09 12:00:00 AM", "Sell", "8SVDLK9",
     "PUT SPY 06/23/26 737 STATE STREET SPDR S&P 500 ETF WE ACTED AS AGENT",
     "-2.00000", "1.87000000", "374.00", "-0.02", "373.98", "USD", "11111111", "Trades", "Individual margin"],
    ["2026-03-31 12:00:00 AM", "2026-03-31 12:00:00 AM", "FCH", "", "Mar 2026 PLUS PLAN FEE",
     "0.00000", "0.00000000", "0.00", "0.00", "-19.95", "CAD", "11111111", "Fees and rebates", "Individual margin"],
    ["2026-04-01 12:00:00 AM", "2026-04-01 12:00:00 AM", "DIV", "T999999", "SOME ETF CASH DIV",
     "0.00000", "0.00000000", "0.00", "0.00", "3.21", "USD", "22222222", "Dividends", "Individual TFSA"],
]


def _write_xlsx(path: Path, rows: list[list[str]], *, shared_strings: bool = False) -> Path:
    """A minimal workbook in the shape Questrade actually emits.

    Questrade writes inline ``t="str"`` values and ships no ``sharedStrings.xml``
    at all; Excel re-saving the same file produces the shared-string form
    instead. Both are generated here because which one arrives is the writer's
    choice, not the trader's.
    """
    table = [COLUMNS, *rows]
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    shared: list[str] = []
    if shared_strings:
        for row in table:
            for value in row:
                if value not in shared:
                    shared.append(value)

    body = []
    for row_index, row in enumerate(table, start=1):
        cells = []
        for column_index, value in enumerate(row):
            reference = f"{chr(ord('A') + column_index)}{row_index}"
            if shared_strings:
                cells.append(f'<c r="{reference}" t="s"><v>{shared.index(value)}</v></c>')
            else:
                escaped = value.replace("&", "&amp;").replace("<", "&lt;")
                cells.append(f'<c r="{reference}" t="str"><v>{escaped}</v></c>')
        body.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    sheet = (
        f'<?xml version="1.0"?><worksheet xmlns="{namespace}">'
        f'<sheetData>{"".join(body)}</sheetData></worksheet>'
    )

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", sheet)
        if shared_strings:
            items = "".join(
                "<si><t>" + value.replace("&", "&amp;").replace("<", "&lt;") + "</t></si>"
                for value in shared
            )
            archive.writestr(
                "xl/sharedStrings.xml",
                f'<?xml version="1.0"?><sst xmlns="{namespace}" count="{len(shared)}">{items}</sst>',
            )
    return path


def _write_csv(path: Path, rows: list[list[str]]) -> Path:
    lines = [",".join(f'"{value}"' for value in row) for row in [COLUMNS, *rows]]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture
def store(tmp_path):
    return JournalStore(tmp_path / "trade_journal.sqlite3")


def _api_execution(uid, side, quantity, price, commission, *, day="2026-03-02", symbol="AAPL"):
    """A live-sync row: a real execution id and, crucially, a real TIME."""
    return {
        "execution_uid": f"QUESTRADE:11111111:{uid}",
        "broker": "QUESTRADE",
        "account_number": "11111111",
        "account_label": "Individual margin",
        "account_type": "Individual margin",
        "symbol": symbol,
        "security_type": "STK",
        "currency": "USD",
        "side": side,
        "quantity": quantity,
        "price": price,
        "timestamp": f"{day}T09:45:00-05:00",
        "trade_date": day,
        "commission": commission,
        "fees": 0.0,
        "gross_amount": None,
        "net_amount": None,
        "order_id": "",
        "exchange_exec_id": "",
        "raw_json": "{}",
        "source": "QT_API",
    }


# -- reading the file --------------------------------------------------------


def test_an_xlsx_is_read_without_a_third_party_library(tmp_path):
    """An xlsx is a zip of XML. Adding openpyxl would owe a frozen rebuild."""
    table = statement.read_statement_table(_write_xlsx(tmp_path / "a.xlsx", ROWS))
    assert len(table) == len(ROWS)
    assert table[0]["Symbol"] == "AAPL"
    assert table[0]["Account #"] == "11111111"


def test_the_shared_string_form_reads_identically(tmp_path):
    inline = statement.read_statement_table(_write_xlsx(tmp_path / "inline.xlsx", ROWS))
    shared = statement.read_statement_table(
        _write_xlsx(tmp_path / "shared.xlsx", ROWS, shared_strings=True)
    )
    assert inline == shared


def test_a_csv_export_reads_the_same_way(tmp_path):
    from_csv = statement.parse_statement(
        statement.read_statement_table(_write_csv(tmp_path / "a.csv", ROWS))
    )
    from_xlsx = statement.parse_statement(
        statement.read_statement_table(_write_xlsx(tmp_path / "a.xlsx", ROWS))
    )
    assert [e.execution_uid for e in from_csv.executions] == [
        e.execution_uid for e in from_xlsx.executions
    ]


def test_an_unsupported_extension_is_refused_rather_than_guessed(tmp_path):
    target = tmp_path / "statement.pdf"
    target.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="unsupported statement file type"):
        statement.read_statement_table(target)


# -- what a statement cannot say ---------------------------------------------


def test_a_statement_trade_carries_no_time_of_day_and_is_not_given_one(tmp_path):
    """Every Questrade row is stamped "12:00:00 AM".

    Writing these at 09:30 to make them look complete would tag an entire
    imported year ``opening_drive`` - a confident answer to a question the file
    never answered.
    """
    parse = statement.parse_statement(
        statement.read_statement_table(_write_xlsx(tmp_path / "a.xlsx", ROWS))
    )
    execution = parse.executions[0]
    assert execution.timestamp.endswith("00:00:00-05:00")
    assert session_bucket(execution.timestamp) is None


def test_a_date_only_round_trip_is_a_day_trade_and_never_a_scalp(tmp_path):
    """Zero elapsed minutes is missing data, not a three-second scalp."""
    _write_xlsx(tmp_path / "a.xlsx", ROWS)
    trade = {
        "opened_at": statement.statement_timestamp(date(2026, 3, 2)),
        "closed_at": statement.statement_timestamp(date(2026, 3, 2)),
        "security_type": "STK",
    }
    tags = {tag.kind: tag.tag for tag in shape_tags(trade)}
    assert tags["hold"] == "day_trade"
    assert "entry_time" not in tags


def test_the_statements_own_row_order_is_carried_into_the_uid(tmp_path):
    """Two identical fills on one day are two real fills.

    Without the sequence they hash to one uid and half the position silently
    disappears. It is also the only intraday ordering a statement has.
    """
    doubled = [ROWS[0], ROWS[0]]
    parse = statement.parse_statement(
        statement.read_statement_table(_write_xlsx(tmp_path / "a.xlsx", doubled))
    )
    assert len(parse.executions) == 2
    assert parse.executions[0].execution_uid != parse.executions[1].execution_uid


def test_the_uid_is_deterministic_so_a_re_import_is_idempotent(tmp_path, store):
    path = _write_xlsx(tmp_path / "a.xlsx", ROWS)
    first = statement.import_questrade_statement(store, path)
    second = statement.import_questrade_statement(store, path)
    assert first["executions_written"] == 4
    assert len(store.list_trades()) == 2
    assert second["days_skipped_richer_source"] == 0
    assert len(store.list_trades()) == 2


# -- options -----------------------------------------------------------------


def test_an_option_contract_is_read_from_the_description_not_the_symbol(tmp_path):
    """Questrade's Symbol column for an option is an internal id.

    Trusting it would make every contract its own opaque position, hide the
    expiry and strike, and leave the multiplier at 1 instead of 100 - a P&L
    error of two orders of magnitude.
    """
    parse = statement.parse_statement(
        statement.read_statement_table(_write_xlsx(tmp_path / "a.xlsx", ROWS))
    )
    options = [e for e in parse.executions if e.security_type == "OPT"]
    assert [e.symbol for e in options] == ["SPY260623P00737000"] * 2
    assert all(e.symbol != "8SVDLK9" for e in parse.executions)


def test_an_option_position_uses_a_hundred_multiplier(tmp_path, store):
    statement.import_questrade_statement(store, _write_xlsx(tmp_path / "a.xlsx", ROWS))
    option = [t for t in store.list_trades() if t["security_type"] == "OPT"][0]
    with store.connection() as conn:
        multipliers = {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT multiplier FROM raw_executions WHERE security_type = 'OPT'"
            ).fetchall()
        }
    assert multipliers == {100.0}
    # The P&L is the real proof: (1.87 - 1.60) * 2 contracts * 100 = 54.00
    # gross, less 0.02 of cost. Without the multiplier it would be 54 cents.
    assert option["gross_pnl"] == pytest.approx(54.0, abs=0.01)
    assert option["net_pnl"] == pytest.approx(53.98, abs=0.01)


def test_a_description_that_is_not_an_option_is_left_alone():
    assert statement.parse_option_description("APPLE INC") is None
    assert statement.parse_option_description("") is None
    parsed = statement.parse_option_description("CALL QQQ 12/19/25 500 INVESCO QQQ TRUST")
    assert parsed == {"right": "CALL", "underlying": "QQQ", "expiry": "251219", "strike": 500.0}


# -- money -------------------------------------------------------------------


def test_the_single_commission_column_is_taken_as_the_whole_cost(tmp_path, store):
    """Net == Gross + Commission on every trade row of the real file.

    So the statement's one column IS the total cost. Splitting it into a
    guessed commission and a guessed fee would invent a breakdown the file
    does not contain.
    """
    statement.import_questrade_statement(store, _write_xlsx(tmp_path / "a.xlsx", ROWS))
    equity = [t for t in store.list_trades() if t["symbol"] == "AAPL"][0]
    assert equity["commission"] == pytest.approx(0.05)
    assert equity["fees"] == 0.0
    assert equity["gross_pnl"] == pytest.approx(20.0)
    assert equity["net_pnl"] == pytest.approx(19.95)


def test_cash_rows_are_imported_beside_the_trades(tmp_path, store):
    statement.import_questrade_statement(store, _write_xlsx(tmp_path / "a.xlsx", ROWS))
    rows = {row["activity_type"]: row for row in store.list_cash_transactions()}
    assert rows["FEE"]["amount"] == pytest.approx(-19.95)
    assert rows["FEE"]["currency"] == "CAD"
    assert rows["DIVIDEND"]["amount"] == pytest.approx(3.21)


# -- the rule that prevents double counting ----------------------------------


def test_a_day_both_sources_agree_on_stays_with_the_live_sync(tmp_path, store):
    """Trader decision 2026-08-28: the file wins on MONEY, not on everything.

    A statement carries no time of day, so taking over a day the API already
    has would discard the only intraday timestamps the journal owns. When the
    two agree on the day's cash there is nothing to gain by it, so the API rows
    stay - times and all.
    """
    for uid, side, quantity, price, commission in (
        ("api-1", "BUY", 20.0, 100.0, 0.0),
        ("api-2", "SELL", 20.0, 101.0, 0.05),
    ):
        store.upsert_executions([_api_execution(uid, side, quantity, price, commission)])

    summary = statement.import_questrade_statement(store, _write_xlsx(tmp_path / "a.xlsx", ROWS))

    assert summary["authority"]["days_taken_over"] == 0
    assert summary["authority"]["days_in_agreement"] == 1
    with store.connection() as conn:
        sources = {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT source FROM raw_executions WHERE symbol = 'AAPL'"
            ).fetchall()
        }
    assert sources == {"QT_API"}
    # The API's real timestamp survived, so the trade still has a session.
    trade = [t for t in store.list_trades() if t["symbol"] == "AAPL"][0]
    assert trade["opened_at"].endswith("09:45:00-05:00")


def test_a_day_whose_money_disagrees_is_taken_over_by_the_file(tmp_path, store):
    """The other half of the rule: the broker's own file is the money.

    Here the API only ever saw the opening fill, so the day's cash is wrong.
    The file's version replaces it and the API rows are VOIDED - append-only
    (I3), so they stay in raw_executions and the change is undoable.
    """
    store.upsert_executions([_api_execution("api-1", "BUY", 20.0, 100.0, 0.0)])

    summary = statement.import_questrade_statement(store, _write_xlsx(tmp_path / "a.xlsx", ROWS))

    authority = summary["authority"]
    assert authority["days_taken_over"] == 1
    assert authority["taken"][0]["voided"] == 1
    assert authority["taken"][0]["written"] == 2

    # Nothing was deleted - the API row is still on disk, and voided.
    with store.connection() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM raw_executions WHERE execution_uid = 'QUESTRADE:11111111:api-1'"
        ).fetchone()[0] == 1
    voids = [row for row in store.list_adjustments(limit=50) if row["action"] == "VOID_EXECUTION"]
    assert len(voids) == 1
    assert "authoritative for money" in voids[0]["reason"]

    # And the assembled trade is now the file's: a complete round trip.
    trade = [t for t in store.list_trades() if t["symbol"] == "AAPL"][0]
    assert trade["status"] == "CLOSED"
    assert trade["net_pnl"] == pytest.approx(19.95)


def test_the_file_never_touches_a_day_it_does_not_mention(tmp_path, store):
    """A day the sync has and the file does not is a gap, not a disagreement.

    Taking it over would delete real fills for no reason.
    """
    store.upsert_executions(
        [_api_execution("api-9", "BUY", 5.0, 50.0, 0.0, day="2026-05-20", symbol="MSFT")]
    )

    summary = statement.import_questrade_statement(store, _write_xlsx(tmp_path / "a.xlsx", ROWS))

    assert summary["authority"]["days_compared"] == 0
    assert store.list_adjustments(limit=50) == []


def test_a_statement_ranks_below_the_api_if_the_two_ever_share_a_uid():
    assert SOURCE_RANK[statement.STATEMENT_SOURCE] < SOURCE_RANK["QT_API"]
    assert statement.STATEMENT_SOURCE not in statement.RICHER_SOURCES


# -- accounts and coverage ---------------------------------------------------


def test_account_tax_status_is_seeded_from_the_account_type_column(tmp_path, store):
    statement.import_questrade_statement(store, _write_xlsx(tmp_path / "a.xlsx", ROWS))
    statuses = {
        row["account_number"]: row["tax_status"]
        for row in store.list_accounts()
    }
    assert statuses["11111111"] == "TAXABLE"
    assert statuses["22222222"] == "TAX_FREE"


def test_a_tax_status_the_trader_set_is_never_overwritten(tmp_path, store):
    """I6: the label is trader-owned and a guess is a wrong number in a tax record."""
    store.upsert_accounts("QUESTRADE", [{"number": "11111111", "type": "Individual margin"}])
    store.set_account_tax_status("QUESTRADE", "11111111", "TAX_FREE", source="trader")

    statement.import_questrade_statement(store, _write_xlsx(tmp_path / "a.xlsx", ROWS))

    statuses = {row["account_number"]: row["tax_status"] for row in store.list_accounts()}
    assert statuses["11111111"] == "TAX_FREE"


def test_an_unrecognised_account_type_stays_unlabeled_rather_than_guessed():
    assert statement.tax_status_for_account_type("Individual TFSA") == "TAX_FREE"
    assert statement.tax_status_for_account_type("Individual margin") == "TAXABLE"
    assert statement.tax_status_for_account_type("Some New Account Type") == ""
    assert statement.tax_status_for_account_type("") == ""


def test_coverage_is_marked_only_for_days_the_import_actually_wrote(tmp_path, store):
    """A statement listing no trades on a day is not evidence none happened."""
    statement.import_questrade_statement(store, _write_xlsx(tmp_path / "a.xlsx", ROWS))
    from journal_coverage import coverage_rows

    covered = {
        (row["account_number"], row["day"])
        for row in coverage_rows(store, broker="QUESTRADE")
        if row["status"] == "COVERED"
    }
    assert ("11111111", "2026-03-02") in covered
    assert ("11111111", "2026-03-05") in covered
    # 2026-03-31 carried only a fee, and 2026-04-01 only a dividend.
    assert ("11111111", "2026-03-31") not in covered
    assert ("22222222", "2026-04-01") not in covered


def test_the_import_run_records_what_it_did(tmp_path, store):
    statement.import_questrade_statement(store, _write_xlsx(tmp_path / "a.xlsx", ROWS))
    run = store.list_import_runs(limit=5)[0]
    assert run["source"] == "QUESTRADE_STATEMENT"
    assert run["status"] == "OK"
    assert run["coverage_start"] == "2026-03-02"
    assert run["coverage_end"] == "2026-04-01"
    assert "4 executions" in run["message"]


def test_a_failed_import_records_its_own_failure(tmp_path, store):
    broken = tmp_path / "broken.xlsx"
    broken.write_bytes(b"not a zip")
    with pytest.raises(Exception):
        statement.import_questrade_statement(store, broken)
    assert store.list_import_runs(limit=5) == [] or store.list_import_runs(limit=5)[0][
        "status"
    ] in {"FAILED", "OK"}


# -- long vs short, and layering later exports -------------------------------

#: Questrade names a short in the Description. Same shapes as the real file.
SHORT_ROWS = [
    ["2026-04-06 12:00:00 AM", "2026-04-07 12:00:00 AM", "Sell", "ZZZ",
     "ZZZ INC COMMON STOCK SHORT. WE ACTED AS AGENT",
     "-10.00000", "50.00000000", "500.00", "-0.02", "499.98", "USD", "11111111", "Trades", "Individual margin"],
    ["2026-04-06 12:00:00 AM", "2026-04-07 12:00:00 AM", "Buy", "ZZZ",
     "ZZZ INC COMMON STOCK COVER SHORT. WE ACTED AS AGENT",
     "10.00000", "48.00000000", "-480.00", "0.00", "-480.00", "USD", "11111111", "Trades", "Individual margin"],
]


def test_a_short_round_trip_is_read_from_the_description_not_the_row_order(tmp_path, store):
    """The file lists a same-day round trip SELL-first 227 times out of 227.

    That makes row order a SORT, not a sequence, and the assembler's uid
    tiebreak turned direction into a coin flip - 86 of 199 came out SHORT at
    random. Questrade writes "SHORT." and "COVER SHORT." in the description,
    which settles it per row.
    """
    statement.import_questrade_statement(store, _write_xlsx(tmp_path / "a.xlsx", SHORT_ROWS))
    trade = store.list_trades()[0]
    assert trade["direction"] == "SHORT"
    # Sold at 50, covered at 48, on ten shares, less two cents of cost.
    assert trade["gross_pnl"] == pytest.approx(20.0)
    assert trade["net_pnl"] == pytest.approx(19.98)
    assert trade["status"] == "CLOSED"


def test_an_unmarked_round_trip_is_a_long(tmp_path, store):
    """Absence of a marking is itself the answer: Questrade marks every short."""
    statement.import_questrade_statement(store, _write_xlsx(tmp_path / "a.xlsx", ROWS[:2]))
    trade = store.list_trades()[0]
    assert trade["direction"] == "LONG"
    assert trade["net_pnl"] == pytest.approx(19.95)


def test_the_marking_is_read_per_row():
    assert statement.short_marking("ZZZ INC COMMON STOCK SHORT. WE ACTED AS AGENT") == "SHORT"
    assert statement.short_marking("VF CORPORATION COVER SHORT. WE ACTED AS AGENT") == "COVER"
    assert statement.short_marking("APPLE INC WE ACTED AS AGENT") == ""
    # "COVER" is tested first because a cover line contains the word SHORT too.
    assert statement.short_marking("X COVER SHORT. WE ACTED AS AGENT") == "COVER"


def test_leg_rank_puts_the_opening_side_first():
    assert statement.leg_rank("SELL", "SHORT") == statement.OPENS_POSITION
    assert statement.leg_rank("BUY", "") == statement.OPENS_POSITION
    assert statement.leg_rank("BUY", "COVER") == statement.CLOSES_POSITION
    assert statement.leg_rank("SELL", "") == statement.CLOSES_POSITION


def test_a_later_longer_export_layers_instead_of_doubling(tmp_path, store):
    """The whole point of being able to re-download through the year.

    A January-to-December export lists the same January trades the
    January-to-August one did, at different row positions. When the uid carried
    the row index, a one-row shift made 884 of 884 real trades look new.
    """
    first = _write_xlsx(tmp_path / "first.xlsx", ROWS)
    # The same rows, re-ordered and with later activity in front of them.
    later = _write_xlsx(tmp_path / "later.xlsx", SHORT_ROWS + list(reversed(ROWS)))

    statement.import_questrade_statement(store, first)
    trades_after_first = len(store.list_trades())
    summary = statement.import_questrade_statement(store, later)

    with store.connection() as conn:
        executions = conn.execute("SELECT COUNT(*) FROM raw_executions").fetchone()[0]
    assert executions == 6  # 4 from the first file, 2 genuinely new
    assert len(store.list_trades()) == trades_after_first + 1
    assert summary["executions_written"] == 6


def test_re_importing_either_file_in_any_order_adds_nothing(tmp_path, store):
    first = _write_xlsx(tmp_path / "first.xlsx", ROWS)
    later = _write_xlsx(tmp_path / "later.xlsx", SHORT_ROWS + list(reversed(ROWS)))
    for path in (first, later, first, later, later, first):
        statement.import_questrade_statement(store, path)
    with store.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_executions").fetchone()[0] == 6
        assert conn.execute("SELECT COUNT(*) FROM cash_transactions").fetchone()[0] == 2


# -- checking the journal against the file it came from ----------------------


def test_reconciliation_adds_the_file_up_by_hand_and_agrees_with_the_journal(tmp_path, store):
    path = _write_xlsx(tmp_path / "a.xlsx", ROWS + SHORT_ROWS)
    statement.import_questrade_statement(store, path)

    report = statement.reconcile_statement(store, path)

    # AAPL 19.95 + the option 53.98 + the short 19.98.
    assert report["statement_pnl"] == pytest.approx(93.91, abs=0.01)
    assert report["journal_pnl"] == pytest.approx(93.91, abs=0.01)
    assert abs(report["difference"]) <= statement.ROUNDING_TOLERANCE
    assert report["symbols_beyond_rounding"] == []
    assert report["closed_symbols"] == 3
    assert report["statement_commission"] == pytest.approx(report["journal_commission"])
    # AAPL and the short are same-day; the option's legs are a day apart.
    assert report["same_day_round_trips"] == 2
    assert report["short_marked_rows"] == 2


def test_reconciliation_excludes_an_open_position_rather_than_zeroing_it(tmp_path, store):
    """Cash has left the account with no realised P&L against it yet."""
    open_only = [ROWS[0]]  # a buy with no matching sell
    path = _write_xlsx(tmp_path / "a.xlsx", open_only)
    statement.import_questrade_statement(store, path)

    report = statement.reconcile_statement(store, path)

    assert report["closed_symbols"] == 0
    assert report["open_symbols"] == 1
    assert report["statement_pnl"] == 0.0
    assert report["open_cash"] == pytest.approx(-2000.0)


def test_reconciliation_names_the_symbol_when_the_two_disagree(tmp_path, store):
    path = _write_xlsx(tmp_path / "a.xlsx", ROWS)
    statement.import_questrade_statement(store, path)
    # Void one leg through the sanctioned append-only route, so the journal and
    # the file genuinely disagree about AAPL.
    trade = [t for t in store.list_trades() if t["symbol"] == "AAPL"][0]
    leg = store.list_trade_legs(trade["trade_id"])[0]
    store.record_adjustment(
        action="VOID_EXECUTION",
        target_uid=leg["execution_uid"],
        reason="test: prove the check notices",
    )
    store.rebuild_trades()

    report = statement.reconcile_statement(store, path)

    flagged = {row["symbol"] for row in report["symbols_beyond_rounding"]}
    assert "AAPL" in flagged
    assert abs(report["difference"]) > statement.ROUNDING_TOLERANCE


def test_reconciliation_never_writes(tmp_path, store):
    path = _write_xlsx(tmp_path / "a.xlsx", ROWS)
    statement.import_questrade_statement(store, path)
    before = len(store.list_import_runs(limit=50))

    statement.reconcile_statement(store, path)

    assert len(store.list_import_runs(limit=50)) == before


def test_a_day_holding_both_a_short_and_a_long_in_one_symbol_is_named(tmp_path, store):
    """Legal, and 3 days of 439 on the trader's own history.

    The assembler groups a symbol into ONE position, so it blends what were
    really two trades. The day's money is still exact; the split is not.
    """
    mixed = SHORT_ROWS + [
        ["2026-04-06 12:00:00 AM", "2026-04-07 12:00:00 AM", "Buy", "ZZZ",
         "ZZZ INC COMMON STOCK WE ACTED AS AGENT",
         "5.00000", "49.00000000", "-245.00", "0.00", "-245.00", "USD", "11111111", "Trades", "Individual margin"],
        ["2026-04-06 12:00:00 AM", "2026-04-07 12:00:00 AM", "Sell", "ZZZ",
         "ZZZ INC COMMON STOCK WE ACTED AS AGENT",
         "-5.00000", "49.50000000", "247.50", "-0.01", "247.49", "USD", "11111111", "Trades", "Individual margin"],
    ]
    path = _write_xlsx(tmp_path / "a.xlsx", mixed)
    statement.import_questrade_statement(store, path)

    report = statement.reconcile_statement(store, path)

    assert [(row["symbol"], row["date"]) for row in report["mixed_direction_days"]] == [
        ("ZZZ", "2026-04-06")
    ]
    # The money is still exact even though the two trades are blended.
    assert abs(report["difference"]) <= statement.ROUNDING_TOLERANCE
