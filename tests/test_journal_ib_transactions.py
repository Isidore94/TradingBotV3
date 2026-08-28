"""Importing an IBKR Transaction History file.

Fixtures are SYNTHETIC. The shapes come from a real 2025-2026 export (803 rows,
609 of them fills) but no real account number or amount is committed here.

Three things separate IBKR's file from Questrade's, and each is a place where
carrying the Questrade reading over would produce a confident wrong number: it
is a SECTIONED csv, its money is in the BASE currency while its prices are not,
and its account numbers arrive MASKED.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_ib_transactions as ib  # noqa: E402
from journal_store import JournalStore  # noqa: E402
from journal_trade_shape import session_bucket  # noqa: E402

HEADER = (
    "Transaction History,Header,Date,Account,Description,Transaction Type,Symbol,"
    "Quantity,Price,Price Currency,Gross Amount ,Commission,Net Amount"
)

#: A USD stock round trip, a USD option round trip, an assignment, a fee, and a
#: commission CREDIT. Base currency is CAD, so every Gross/Net below is CAD
#: while every Price is USD - the file's own defining awkwardness.
LINES = [
    "Statement,Header,Field Name,Field Value",
    "Statement,Data,Title,Transaction History",
    "Summary,Header,Field Name,Field Value",
    "Summary,Data,Base Currency,CAD",
    HEADER,
    # 10 shares at 100 USD, booked at a rate of 1.40 -> 1400 CAD. Commission
    # -1.40 CAD, i.e. 1.00 USD.
    "Transaction History,Data,2026-03-02,U***2524,SOME CO,Buy,ZZZ,10.0,100.0,USD,-1400.0,-1.4,-1401.4",
    "Transaction History,Data,2026-03-02,U***2524,SOME CO,Sell,ZZZ,-10.0,110.0,USD,1540.0,-1.4,1538.6",
    # 2 contracts at 1.50 USD -> 2 * 1.50 * 100 = 300 USD -> 420 CAD at 1.40.
    "Transaction History,Data,2026-03-05,U***2524,OPT,Sell,QQQ  260320P00500000,-2.0,1.5,USD,420.0,-2.8,417.2",
    # Bought back at 0.50, and this row carries a commission CREDIT (+0.70 CAD).
    "Transaction History,Data,2026-03-06,U***2524,OPT,Buy,QQQ  260320P00500000,2.0,0.5,USD,-140.0,0.7,-139.3",
    "Transaction History,Data,2026-03-09,U***2524,Buy 100 SOME ETF (Assignment),Assignment,AAA,100.0,20.0,USD,-2800.0,-,-2800.0",
    "Transaction History,Data,2026-03-31,U***2524,Monthly fee,Other Fee,-,-,-,-,-,-,-11.5",
    "Transaction History,Data,2026-03-31,U***2524,GST,Sales Tax,-,-,-,-,-,-,-1.5",
]


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def store(tmp_path):
    return JournalStore(tmp_path / "trade_journal.sqlite3")


@pytest.fixture
def ib_file(tmp_path):
    return _write(tmp_path / "U9992524.TRANSACTIONS.20250101.20260827.csv", LINES)


# -- the sectioned file ------------------------------------------------------


def test_each_section_keeps_its_own_header(ib_file):
    """A plain DictReader applies the first header to every later table."""
    sections = ib.read_ib_sections(ib_file)
    assert set(sections) == {"Statement", "Summary", "Transaction History"}
    assert sections["Summary"][0] == {"Field Name": "Base Currency", "Field Value": "CAD"}
    assert sections["Transaction History"][0]["Symbol"] == "ZZZ"
    assert sections["Transaction History"][0]["Price Currency"] == "USD"


def test_an_ib_file_is_recognised_from_its_contents_not_its_name(ib_file, tmp_path):
    assert ib.looks_like_ib_transactions(ib_file)
    other = _write(tmp_path / "questrade.csv", ["Transaction Date,Action,Symbol", "x,Buy,AAPL"])
    assert not ib.looks_like_ib_transactions(other)


def test_the_base_currency_is_read_from_the_summary(ib_file):
    assert ib.read_ib_file(ib_file).base_currency == "CAD"


# -- base currency vs price currency -----------------------------------------


def test_the_row_implies_its_own_fx_rate(ib_file):
    """Gross is CAD, price is USD; their ratio is IB's rate for that trade."""
    parse = ib.read_ib_file(ib_file)
    stock = [e for e in parse.executions if e.symbol == "ZZZ"][0]
    import json

    assert json.loads(stock.raw_json)["implied_fx_rate"] == pytest.approx(1.40)


def test_the_option_multiplier_is_inside_the_implied_rate(ib_file):
    """420 CAD over 2 x 1.50 USD is 140 without the multiplier, 1.40 with it.

    Getting this wrong would not just misprice the option - it would make the
    implied rate a hundred times too large and corrupt the commission with it.
    """
    parse = ib.read_ib_file(ib_file)
    option = [e for e in parse.executions if e.security_type == "OPT"][0]
    import json

    assert json.loads(option.raw_json)["implied_fx_rate"] == pytest.approx(1.40)


def test_the_cost_is_converted_into_the_trades_own_currency(ib_file):
    """A USD gross reduced by a CAD commission is a plausible wrong number."""
    parse = ib.read_ib_file(ib_file)
    stock = [e for e in parse.executions if e.symbol == "ZZZ" and e.side == "BUY"][0]
    assert stock.currency == "USD"
    assert stock.commission == pytest.approx(1.0)  # 1.40 CAD at 1.40


def test_a_commission_credit_survives_as_a_credit(ib_file, store):
    """18 of 609 rows on the trader's real file carry a rebate.

    ``abs()`` turned each into a charge, overstating the year's cost by twice
    the credit - which was the whole $2.17 by which that file and the journal
    disagreed.
    """
    parse = ib.read_ib_file(ib_file)
    credit = [
        e for e in parse.executions if e.security_type == "OPT" and e.side == "BUY"
    ][0]
    assert credit.commission == pytest.approx(-0.5)  # +0.70 CAD at 1.40

    ib.import_ib_transaction_file(store, _write(Path(store.db_path).parent / "f.csv", LINES))
    option = [t for t in store.list_trades() if t["security_type"] == "OPT"][0]
    # Sold 2 at 1.50 and bought back at 0.50: 200 USD gross on a short.
    # Costs: 2.00 charged on the sell, 0.50 CREDITED on the buy -> 1.50 net.
    assert option["gross_pnl"] == pytest.approx(200.0)
    assert option["commission"] == pytest.approx(1.5)
    assert option["net_pnl"] == pytest.approx(198.5)


def test_a_rate_that_cannot_be_derived_leaves_the_cost_unscaled_and_says_so(tmp_path):
    """Never scale by a guess. The row records that it was not converted."""
    lines = LINES[:5] + [
        "Transaction History,Data,2026-03-02,U***2524,SOME CO,Buy,ZZZ,10.0,0.0,USD,-,-1.4,-1.4",
    ]
    parse = ib.read_ib_file(_write(tmp_path / "x.csv", lines))
    import json

    assert len(parse.executions) == 1
    payload = json.loads(parse.executions[0].raw_json)
    assert payload["implied_fx_rate"] is None
    assert payload["cost_converted_to_native"] is False


# -- masked accounts ---------------------------------------------------------


def test_a_mask_matches_only_an_account_of_the_same_length(tmp_path):
    assert ib.mask_matches("U***7396", "U4867396")
    assert not ib.mask_matches("U***7396", "U12345697396")
    assert not ib.mask_matches("U***7396", "U4867395")
    assert not ib.mask_matches("", "U4867396")


def test_a_mask_resolves_only_when_exactly_one_account_fits():
    """A guessed account merges or splits a position on a hunch."""
    assert ib.resolve_account_number("U***7396", ["U4867396", "U1112524"]) == "U4867396"
    # Two accounts fit: refuse rather than pick.
    assert ib.resolve_account_number("U***7396", ["U4867396", "U1117396"]) == "U***7396"
    # None fit.
    assert ib.resolve_account_number("U***7396", ["U1112524"]) == "U***7396"
    # Already unmasked.
    assert ib.resolve_account_number("U4867396", []) == "U4867396"


def test_the_filename_is_another_candidate_and_never_an_override(tmp_path):
    """An IBKR export is named for one account but can hold rows for several."""
    assert ib.account_hint_from_filename(Path("U4867396.TRANSACTIONS.2025.csv")) == "U4867396"
    # A copied or prefixed filename still yields the hint.
    assert ib.account_hint_from_filename(Path("36558c4a-U4867396.TRANSACTIONS.csv")) == "U4867396"
    assert ib.account_hint_from_filename(Path("statement.csv")) == ""
    # The hint fits this mask, so it resolves; a different mask is untouched.
    assert ib.resolve_account_number("U***7396", [], filename_hint="U4867396") == "U4867396"
    assert ib.resolve_account_number("U***2524", [], filename_hint="U4867396") == "U***2524"


def test_an_unresolved_account_is_reported_rather_than_invented(ib_file, store):
    summary = ib.import_ib_transaction_file(store, ib_file)
    # The fixture's filename names U9992524, which fits U***2524.
    assert summary["unresolved_accounts"] == []
    assert summary["accounts"] == ["U9992524"]


def test_a_mask_nothing_can_resolve_keeps_its_masked_form(tmp_path, store):
    path = _write(tmp_path / "statement.csv", LINES)  # no account in the name
    summary = ib.import_ib_transaction_file(store, path)
    assert summary["unresolved_accounts"] == ["U***2524"]
    assert summary["accounts"] == ["U***2524"]


# -- fills -------------------------------------------------------------------


def test_an_assignment_is_a_real_fill(ib_file):
    """An assigned option becomes a stock position through a real buy.

    Dropping it leaves the position open forever with nothing to close it.
    """
    parse = ib.read_ib_file(ib_file)
    assignment = [e for e in parse.executions if e.symbol == "AAA"][0]
    assert assignment.side == "BUY"
    assert assignment.quantity == pytest.approx(100.0)


def test_the_side_of_an_assignment_comes_from_its_description():
    row = ib.IBRow(
        sequence=0, trade_date=__import__("datetime").date(2026, 3, 9), account="U1",
        description="Sell 100 SOME ETF (Assignment)", transaction_type="Assignment",
        symbol="AAA", quantity=100.0, price=20.0, price_currency="USD",
        gross_amount=-2800.0, commission=None, net_amount=-2800.0,
    )
    assert ib.side_for(row) == "SELL"
    row.description = "(Assignment)"
    assert ib.side_for(row) == "BUY"  # falls back to the quantity's sign
    row.transaction_type = "Other Fee"
    assert ib.side_for(row) == ""


def test_options_keep_their_occ_identity(ib_file):
    parse = ib.read_ib_file(ib_file)
    assert [e.symbol for e in parse.executions if e.security_type == "OPT"] == [
        "QQQ260320P00500000"
    ] * 2


def test_an_ib_file_carries_no_time_of_day_either(ib_file):
    execution = ib.read_ib_file(ib_file).executions[0]
    assert session_bucket(execution.timestamp) is None


# -- writing to the store ----------------------------------------------------


def test_cash_rows_are_imported_in_the_base_currency(ib_file, store):
    """Unlike a trade, a cash row carries no price to imply a rate from."""
    ib.import_ib_transaction_file(store, ib_file)
    rows = {row["description"]: row for row in store.list_cash_transactions()}
    assert rows["Monthly fee"]["activity_type"] == "FEE"
    assert rows["Monthly fee"]["currency"] == "CAD"
    assert rows["Monthly fee"]["amount"] == pytest.approx(-11.5)
    assert rows["GST"]["activity_type"] == "FEE"


def test_the_round_trips_assemble_with_the_right_direction(ib_file, store):
    ib.import_ib_transaction_file(store, ib_file)
    by_symbol = {t["symbol"]: t for t in store.list_trades()}
    assert by_symbol["ZZZ"]["direction"] == "LONG"
    assert by_symbol["ZZZ"]["net_pnl"] == pytest.approx(98.0)  # 100 gross less 2 USD
    assert by_symbol["QQQ260320P00500000"]["direction"] == "SHORT"


def test_re_importing_the_same_file_adds_nothing(ib_file, store):
    first = ib.import_ib_transaction_file(store, ib_file)
    ib.import_ib_transaction_file(store, ib_file)
    with store.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_executions").fetchone()[0] == (
            first["executions_written"]
        )
        assert conn.execute("SELECT COUNT(*) FROM cash_transactions").fetchone()[0] == 2


def test_a_file_never_writes_into_a_day_flex_already_covers(ib_file, store):
    store.upsert_executions(
        [
            {
                "execution_uid": "IBKR:U9992524:flex-1", "broker": "IBKR",
                "account_number": "U9992524", "account_label": "", "account_type": "",
                "symbol": "ZZZ", "security_type": "STK", "currency": "USD", "side": "BUY",
                "quantity": 10.0, "price": 100.0, "timestamp": "2026-03-02T09:45:00-05:00",
                "trade_date": "2026-03-02", "commission": 1.0, "fees": 0.0,
                "gross_amount": None, "net_amount": None, "order_id": "",
                "exchange_exec_id": "", "raw_json": "{}", "source": "IBKR_FLEX",
            }
        ]
    )

    summary = ib.import_ib_transaction_file(store, ib_file)

    assert summary["days_skipped_richer_source"] == 1
    with store.connection() as conn:
        sources = {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT source FROM raw_executions WHERE symbol = 'ZZZ'"
            ).fetchall()
        }
    assert sources == {"IBKR_FLEX"}


def test_the_import_run_records_what_it_did(ib_file, store):
    ib.import_ib_transaction_file(store, ib_file)
    run = store.list_import_runs(limit=5)[0]
    assert run["source"] == "IBKR_FILE"
    assert run["status"] == "OK"
    assert run["coverage_start"] == "2026-03-02"
    assert run["coverage_end"] == "2026-03-31"


def test_nothing_in_the_file_is_silently_dropped(ib_file):
    parse = ib.read_ib_file(ib_file)
    assert parse.skipped == []
    assert len(parse.executions) == 5
    assert len(parse.cash) == 2


def test_the_feed_routes_a_file_to_its_broker_by_content(tmp_path, monkeypatch):
    """Both brokers ship .csv and the file name is whatever it was saved as.

    Asking the trader to pick the broker would be asking them to get it right
    every time; the file says which it is.
    """
    from ui.services import journal_feed

    store = JournalStore(tmp_path / "trade_journal.sqlite3")
    monkeypatch.setattr(journal_feed, "_STORE", store)
    monkeypatch.setattr(journal_feed, "_store", lambda: store)

    ib_path = _write(tmp_path / "anything.csv", LINES)
    summary = journal_feed.import_broker_statement(ib_path)
    assert summary["broker"] == "IBKR"

    questrade = _write(
        tmp_path / "also-anything.csv",
        [
            "Transaction Date,Settlement Date,Action,Symbol,Description,Quantity,Price,"
            "Gross Amount,Commission,Net Amount,Currency,Account #,Activity Type,Account Type",
            "2026-03-02 12:00:00 AM,2026-03-03 12:00:00 AM,Buy,AAPL,APPLE INC,20.00000,"
            "100.00000000,-2000.00,0.00,-2000.00,USD,11111111,Trades,Individual margin",
        ],
    )
    summary = journal_feed.import_broker_statement(questrade)
    assert summary.get("broker") != "IBKR"
    assert summary["executions_written"] == 1
