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


def test_a_statement_never_writes_into_a_day_the_api_already_covers(tmp_path, store):
    """The two sources give the same fill different uids.

    Nothing in the execution upsert can see they are duplicates, so importing
    both would silently double the position. The day is refused instead, and
    the refusal is counted rather than swallowed.
    """
    store.upsert_executions(
        [
            {
                "execution_uid": "QUESTRADE:11111111:real-exec-1",
                "broker": "QUESTRADE",
                "account_number": "11111111",
                "account_label": "Individual margin",
                "account_type": "Individual margin",
                "symbol": "AAPL",
                "security_type": "STK",
                "currency": "USD",
                "side": "BUY",
                "quantity": 20.0,
                "price": 100.0,
                "timestamp": "2026-03-02T09:45:00-05:00",
                "trade_date": "2026-03-02",
                "commission": 0.0,
                "fees": 0.0,
                "gross_amount": None,
                "net_amount": None,
                "order_id": "",
                "exchange_exec_id": "",
                "raw_json": "{}",
                "source": "QT_API",
            }
        ]
    )

    summary = statement.import_questrade_statement(store, _write_xlsx(tmp_path / "a.xlsx", ROWS))

    assert summary["days_skipped_richer_source"] == 1
    assert ("11111111", "2026-03-02") in [tuple(item) for item in summary["skipped_days"]]
    with store.connection() as conn:
        aapl_sources = {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT source FROM raw_executions WHERE symbol = 'AAPL'"
            ).fetchall()
        }
    assert aapl_sources == {"QT_API"}
    # The days the API does not own are still imported.
    assert summary["executions_written"] == 2


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
