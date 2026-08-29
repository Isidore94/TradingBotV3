"""Realised P&L for a tax year, from the broker's own money.

Trader decision 2026-08-28: *"Statement is source of truth for final pnl/tax
purposes."* Everywhere else the journal RECOMPUTES a trade's P&L, which is what
makes per-setup statistics possible and is also arithmetic of our own — it
drifts from the broker's cent-rounded figures by $0.24 across the trader's year.
This report recomputes nothing.

What is really being defended here is the set of things it REFUSES to report,
because a tax figure that quietly interpolates is worse than one that names the
symbol it cannot answer for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_tax_report as tax  # noqa: E402
from journal_store import JournalStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    store = JournalStore(tmp_path / "trade_journal.sqlite3")
    store.upsert_accounts("QUESTRADE", [{"number": "ACCT", "type": "Individual margin"}])
    store.set_account_tax_status("QUESTRADE", "ACCT", "TAXABLE", source="trader")
    return store


def _fill(uid, side, quantity, price, net, *, day="2026-03-02", symbol="AAPL", commission=0.0, **extra):
    """One broker fill. ``net`` is the broker's OWN statement of the cash."""
    row = {
        "execution_uid": uid, "broker": "QUESTRADE", "account_number": "ACCT",
        "account_label": "", "account_type": "", "symbol": symbol,
        "security_type": "STK", "currency": "USD", "side": side,
        "quantity": quantity, "price": price, "timestamp": f"{day}T00:00:00-05:00",
        "trade_date": day, "commission": commission, "fees": 0.0,
        "gross_amount": None, "net_amount": net, "order_id": "",
        "exchange_exec_id": "", "raw_json": "{}", "source": "QT_STATEMENT",
    }
    row.update(extra)
    return row


def _closed(store, **kwargs):
    store.upsert_executions(
        [
            _fill("b1", "BUY", 10.0, 100.0, -1000.0, commission=0.0, **kwargs),
            _fill("s1", "SELL", 10.0, 110.0, 1099.95, commission=0.05, **kwargs),
        ]
    )
    store.rebuild_trades(refresh_tags=False)


# -- the number itself -------------------------------------------------------


def test_realised_is_the_sum_of_what_the_broker_said(store):
    """No cost-basis model, because a flat position does not need one.

    Every share bought was sold, so the fills' own cash amounts add up to the
    realised P&L exactly.
    """
    _closed(store)

    report = tax.build_tax_report(store)

    assert report["positions_reported"] == 1
    position = report["positions"][0]
    assert position["proceeds"] == pytest.approx(1099.95)
    assert position["cost"] == pytest.approx(-1000.0)
    assert position["realised"] == pytest.approx(99.95)
    assert position["commission"] == pytest.approx(0.05)
    assert report["source"].startswith("broker-stated")


def test_it_does_not_recompute_price_times_quantity(store):
    """The whole point: if the broker's figure and ours disagree, theirs wins.

    The fills below state a net the multiplication would not produce, and the
    report must return the broker's number rather than 'correcting' it.
    """
    store.upsert_executions(
        [
            _fill("b1", "BUY", 10.0, 100.0, -1000.07),
            _fill("s1", "SELL", 10.0, 110.0, 1100.03),
        ]
    )
    store.rebuild_trades(refresh_tags=False)

    report = tax.build_tax_report(store)

    assert report["positions"][0]["realised"] == pytest.approx(99.96)


def test_accounts_are_reported_separately_with_their_tax_status(store):
    """A taxable and a tax-free account may never be added together (I6)."""
    _closed(store)
    store.upsert_accounts("QUESTRADE", [{"number": "TFSA", "type": "Individual TFSA"}])
    store.set_account_tax_status("QUESTRADE", "TFSA", "TAX_FREE", source="trader")
    store.upsert_executions(
        [
            _fill("b2", "BUY", 5.0, 20.0, -100.0, account_number="TFSA", symbol="MSFT"),
            _fill("s2", "SELL", 5.0, 25.0, 125.0, account_number="TFSA", symbol="MSFT"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)

    report = tax.build_tax_report(store)

    assert report["by_account"]["ACCT"]["tax_status"] == "TAXABLE"
    assert report["by_account"]["TFSA"]["tax_status"] == "TAX_FREE"
    assert report["by_account"]["TFSA"]["realised_by_currency"]["USD"] == pytest.approx(25.0)


def test_currencies_are_kept_apart(store):
    _closed(store)
    store.upsert_executions(
        [
            _fill("b3", "BUY", 2.0, 10.0, -20.0, symbol="ENB", currency="CAD"),
            _fill("s3", "SELL", 2.0, 12.0, 24.0, symbol="ENB", currency="CAD"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)

    report = tax.build_tax_report(store)

    money = report["by_account"]["ACCT"]["realised_by_currency"]
    assert money["USD"] == pytest.approx(99.95)
    assert money["CAD"] == pytest.approx(4.0)


# -- what it refuses to report -----------------------------------------------


def test_an_open_position_contributes_nothing(store):
    """Cash has left the account with no realised P&L against it yet."""
    store.upsert_executions([_fill("b1", "BUY", 10.0, 100.0, -1000.0)])
    store.rebuild_trades(refresh_tags=False)

    report = tax.build_tax_report(store)

    assert report["positions_reported"] == 0
    assert report["positions_excluded"] == 1
    assert report["excluded"][0]["reason"] == "still open"


def test_a_position_whose_opening_fill_is_missing_is_named_not_guessed(store):
    """A SYNTHETIC_OPEN means the proceeds are real and the cost basis is not.

    On the trader's data this was 23 positions, and importing the earlier
    statement took it to 5 - which is why the reason says which file fixes it.
    """
    # Selling more than the journal knows was opened: the leftover has to open
    # a position the journal never saw the entry for. Buying it back leaves the
    # symbol flat, so only the invented leg keeps it out of the total.
    store.upsert_executions(
        [
            _fill("b1", "BUY", 5.0, 100.0, -500.0, day="2026-03-02"),
            _fill("s1", "SELL", 10.0, 110.0, 1100.0, day="2026-03-03"),
            _fill("b2", "BUY", 5.0, 105.0, -525.0, day="2026-03-04"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)

    report = tax.build_tax_report(store)

    assert report["positions_reported"] == 0
    assert "opening fill is missing" in report["excluded"][0]["reason"]


def test_a_fill_with_no_broker_amount_disqualifies_its_position(store):
    """Mixing a stated figure with a recomputed one is neither number.

    The IBKR socket path records no net_amount at all.
    """
    store.upsert_executions(
        [
            _fill("b1", "BUY", 10.0, 100.0, -1000.0),
            _fill("s1", "SELL", 10.0, 110.0, None, source="IBKR_SOCKET"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)

    report = tax.build_tax_report(store)

    assert report["positions_reported"] == 0
    assert "no broker-stated amount" in report["excluded"][0]["reason"]


def test_a_voided_execution_never_reaches_a_tax_total(store):
    """Retired by a correction or by the file-authority rule.

    It no longer describes the account, so counting it would report a fill the
    trader has already said did not happen that way.
    """
    _closed(store)
    store.upsert_executions([_fill("s2", "SELL", 10.0, 999.0, 9990.0, symbol="AAPL")])
    store.record_adjustment(
        action="VOID_EXECUTION", target_uid="s2", reason="test: not a real fill"
    )
    store.rebuild_trades()

    report = tax.build_tax_report(store)

    assert report["positions_reported"] == 1
    assert report["positions"][0]["realised"] == pytest.approx(99.95)


# -- CAD, the tax currency ---------------------------------------------------


def test_cad_is_converted_per_fill_at_the_booked_rate(store):
    """Never one rate for the year, and never a broker's internal rate."""
    from journal_fx import seed_rate

    seed_rate(store, day="2026-03-02", currency="USD", rate_to_cad=1.40)
    _closed(store)

    report = tax.build_tax_report(store)

    # (-1000.00 + 1099.95) x 1.40
    assert report["positions"][0]["realised_cad"] == pytest.approx(139.93)
    assert report["realised_cad"] == pytest.approx(139.93)
    assert report["cad_complete"] is True


def test_an_unbooked_rate_withholds_the_cad_total_rather_than_guessing(store):
    _closed(store)

    report = tax.build_tax_report(store)

    assert report["positions"][0]["realised_cad"] is None
    assert report["realised_cad"] is None
    assert report["cad_complete"] is False
    assert "2026-03-02" in report["unbooked_rate_days"]


def test_a_cad_position_needs_no_rate(store):
    store.upsert_executions(
        [
            _fill("b3", "BUY", 2.0, 10.0, -20.0, symbol="ENB", currency="CAD"),
            _fill("s3", "SELL", 2.0, 12.0, 24.0, symbol="ENB", currency="CAD"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)

    report = tax.build_tax_report(store)

    assert report["positions"][0]["realised_cad"] == pytest.approx(4.0)


# -- the window --------------------------------------------------------------


def test_a_year_selects_by_the_positions_own_days(store):
    _closed(store, day="2026-03-02")
    store.upsert_executions(
        [
            _fill("b9", "BUY", 1.0, 5.0, -5.0, day="2025-06-01", symbol="OLD"),
            _fill("s9", "SELL", 1.0, 7.0, 7.0, day="2025-06-02", symbol="OLD"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)

    assert {row["symbol"] for row in tax.build_tax_report(store, year=2026)["positions"]} == {"AAPL"}
    assert {row["symbol"] for row in tax.build_tax_report(store, year=2025)["positions"]} == {"OLD"}


def test_a_position_spanning_the_year_end_is_not_cut_in_half(store):
    """Reporting December's buy and January's sell separately would invent a
    cost basis for one and proceeds for the other."""
    store.upsert_executions(
        [
            _fill("b1", "BUY", 10.0, 100.0, -1000.0, day="2025-12-30"),
            _fill("s1", "SELL", 10.0, 110.0, 1100.0, day="2026-01-05"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)

    report = tax.build_tax_report(store, year=2026)

    assert report["positions_reported"] == 1
    assert report["positions"][0]["realised"] == pytest.approx(100.0)
    assert report["positions"][0]["first_day"] == "2025-12-30"


# -- the cross-check and the export ------------------------------------------


def test_the_cross_check_puts_both_routes_side_by_side(store):
    _closed(store)
    report = tax.build_tax_report(store)

    check = tax.cross_check_against_journal(store, report)

    assert check["broker_stated"] == pytest.approx(99.95)
    assert check["journal_recomputed"] == pytest.approx(99.95)
    assert check["difference"] == pytest.approx(0.0, abs=0.0001)
    assert check["accounts"][0]["account"] == "ACCT"


def test_the_export_lists_the_excluded_positions_and_their_reasons(store, tmp_path):
    _closed(store)
    store.upsert_executions([_fill("b8", "BUY", 3.0, 10.0, -30.0, symbol="OPEN")])
    store.rebuild_trades(refresh_tags=False)
    report = tax.build_tax_report(store)

    target = tax.export_tax_csv(report, tmp_path / "tax.csv")

    text = target.read_text(encoding="utf-8")
    assert "reported,QUESTRADE,ACCT,AAPL" in text
    assert "excluded" in text and "still open" in text


def test_the_summary_names_what_it_could_not_count(store):
    _closed(store)
    store.upsert_executions([_fill("b8", "BUY", 3.0, 10.0, -30.0, symbol="OPEN")])
    store.rebuild_trades(refresh_tags=False)
    report = tax.build_tax_report(store)

    text = tax.describe_tax_report(report, tax.cross_check_against_journal(store, report))

    assert "1 position(s) not counted" in text
    assert "still open" in text
    assert "TAXABLE" in text
    assert "Cross-check" in text
