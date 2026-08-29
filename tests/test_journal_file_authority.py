"""When a broker file disagrees with the live sync, the file wins — on money.

Trader decision 2026-08-28, taken after the cost of the blunt version was
measured: neither broker's file carries a time of day, so letting a file take
over every day it covers would discard the only intraday timestamps the journal
has. The rule is therefore split — the sync keeps a day the two AGREE on, the
file takes a day they do not.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_file_authority as authority  # noqa: E402
from journal_store import JournalStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return JournalStore(tmp_path / "trade_journal.sqlite3")


def _row(uid, side, quantity, price, commission=0.0, *, source="QT_API", day="2026-03-02", **extra):
    row = {
        "execution_uid": uid, "broker": "QUESTRADE", "account_number": "ACCT",
        "account_label": "", "account_type": "", "symbol": "AAPL",
        "security_type": "STK", "currency": "USD", "side": side,
        "quantity": quantity, "price": price, "timestamp": f"{day}T09:45:00-05:00",
        "trade_date": day, "commission": commission, "fees": 0.0,
        "gross_amount": None, "net_amount": None, "order_id": "",
        "exchange_exec_id": "", "raw_json": "{}", "source": source,
    }
    row.update(extra)
    return row


# -- what "agreement" means --------------------------------------------------


def test_cash_is_computed_not_read_off_a_column():
    """The two brokers report Gross/Net in different currencies.

    Questrade reports in the trade's currency and IBKR in the account's base
    currency, so their columns are not comparable to each other. Quantity,
    price and cost are.
    """
    assert authority.signed_cash(_row("a", "BUY", 10.0, 100.0, 1.0)) == pytest.approx(-1001.0)
    assert authority.signed_cash(_row("b", "SELL", 10.0, 110.0, 1.0)) == pytest.approx(1099.0)


def test_an_option_multiplier_reaches_the_cash():
    contract = _row("c", "SELL", 2.0, 1.5, 0.0, security_type="OPT")
    assert authority.signed_cash(contract) == pytest.approx(300.0)


def test_a_commission_credit_increases_the_cash():
    """A rebate is money in, and the sign survives all the way here."""
    assert authority.signed_cash(_row("d", "SELL", 10.0, 100.0, -0.5)) == pytest.approx(1000.5)


def test_cash_is_grouped_by_account_and_day():
    totals = authority.cash_by_day(
        [
            _row("a", "BUY", 10.0, 100.0),
            _row("b", "SELL", 10.0, 110.0),
            _row("c", "BUY", 1.0, 5.0, day="2026-03-03"),
        ]
    )
    assert totals[("ACCT", date(2026, 3, 2))] == (pytest.approx(100.0), 2)
    assert totals[("ACCT", date(2026, 3, 3))] == (pytest.approx(-5.0), 1)


def test_the_tolerance_grows_with_the_number_of_fills():
    """Questrade rounds each row to the cent, so a busy day drifts more.

    A flat threshold would either fire on rounding or miss a real difference on
    a quiet day.
    """
    quiet = authority.DayCash("A", date(2026, 3, 2), 100.0, 100.0, 1, 1)
    busy = authority.DayCash("A", date(2026, 3, 2), 100.0, 100.0, 40, 40)
    assert busy.tolerance > quiet.tolerance
    assert quiet.tolerance == pytest.approx(authority.TOLERANCE_BASE + authority.TOLERANCE_PER_FILL)


def test_rounding_is_not_a_disagreement():
    same = authority.DayCash("A", date(2026, 3, 2), 100.005, 100.0, 1, 1)
    assert not same.disagrees
    real = authority.DayCash("A", date(2026, 3, 2), 200.0, 100.0, 1, 1)
    assert real.disagrees


# -- which days are even looked at -------------------------------------------


def test_only_days_both_sources_cover_are_compared(store):
    """A day the file alone covers is a gap the ordinary import fills.

    A day the SYNC alone covers is not a disagreement either - taking it over
    would delete real fills.
    """
    store.upsert_executions([_row("api-1", "BUY", 10.0, 100.0, day="2026-03-02")])
    file_rows = [
        _row("f-1", "BUY", 10.0, 100.0, source="QT_STATEMENT", day="2026-03-05"),
    ]

    comparisons = authority.compare_days(
        store, broker="QUESTRADE", file_executions=file_rows, sources=["QT_API"]
    )

    assert comparisons == []


def test_a_shared_day_is_compared(store):
    store.upsert_executions([_row("api-1", "BUY", 10.0, 100.0)])
    file_rows = [_row("f-1", "BUY", 10.0, 100.0, source="QT_STATEMENT")]

    comparisons = authority.compare_days(
        store, broker="QUESTRADE", file_executions=file_rows, sources=["QT_API"]
    )

    assert len(comparisons) == 1
    assert not comparisons[0].disagrees


# -- taking a day over -------------------------------------------------------


def test_an_agreeing_day_is_left_alone_so_its_times_survive(store):
    store.upsert_executions(
        [_row("api-1", "BUY", 10.0, 100.0, 1.0), _row("api-2", "SELL", 10.0, 110.0, 1.0)]
    )
    file_rows = [
        _row("f-1", "BUY", 10.0, 100.0, 1.0, source="QT_STATEMENT"),
        _row("f-2", "SELL", 10.0, 110.0, 1.0, source="QT_STATEMENT"),
    ]

    report = authority.apply_file_authority(
        store, broker="QUESTRADE", file_executions=file_rows, sources=["QT_API"], label="f.xlsx"
    )

    assert report["days_taken_over"] == 0
    assert report["days_in_agreement"] == 1
    assert store.list_adjustments(limit=10) == []


def test_a_disagreeing_day_is_voided_and_replaced(store):
    store.upsert_executions([_row("api-1", "BUY", 10.0, 100.0, 1.0)])
    file_rows = [
        _row("f-1", "BUY", 10.0, 100.0, 1.0, source="QT_STATEMENT"),
        _row("f-2", "SELL", 10.0, 110.0, 1.0, source="QT_STATEMENT"),
    ]

    report = authority.apply_file_authority(
        store, broker="QUESTRADE", file_executions=file_rows, sources=["QT_API"], label="f.xlsx"
    )

    assert report["days_taken_over"] == 1
    assert report["taken"][0]["voided"] == 1
    assert report["taken"][0]["written"] == 2
    voids = [row for row in store.list_adjustments(limit=10) if row["action"] == "VOID_EXECUTION"]
    assert len(voids) == 1
    assert "authoritative for money" in voids[0]["reason"]
    assert f"{report['days'][0]['difference']:+.2f}" in voids[0]["reason"]


def test_nothing_is_ever_deleted(store):
    """I3: a broker row is never destructively edited or removed.

    The void is append-only, so the day is recoverable by a superseding record
    - which matters because the trader can change their mind about a day.
    """
    store.upsert_executions([_row("api-1", "BUY", 10.0, 100.0, 1.0)])
    file_rows = [
        _row("f-1", "BUY", 10.0, 100.0, 1.0, source="QT_STATEMENT"),
        _row("f-2", "SELL", 10.0, 110.0, 1.0, source="QT_STATEMENT"),
    ]

    authority.apply_file_authority(
        store, broker="QUESTRADE", file_executions=file_rows, sources=["QT_API"], label="f.xlsx"
    )

    with store.connection() as conn:
        surviving = conn.execute(
            "SELECT source FROM raw_executions WHERE execution_uid = 'api-1'"
        ).fetchall()
    assert [row[0] for row in surviving] == ["QT_API"]


def test_a_dry_run_measures_without_writing(store):
    """What the "Check a statement..." button uses: see before you move."""
    store.upsert_executions([_row("api-1", "BUY", 10.0, 100.0, 1.0)])
    file_rows = [
        _row("f-1", "BUY", 10.0, 100.0, 1.0, source="QT_STATEMENT"),
        _row("f-2", "SELL", 10.0, 110.0, 1.0, source="QT_STATEMENT"),
    ]

    report = authority.apply_file_authority(
        store,
        broker="QUESTRADE",
        file_executions=file_rows,
        sources=["QT_API"],
        label="f.xlsx",
        dry_run=True,
    )

    assert report["dry_run"] is True
    assert report["days_taken_over"] == 1
    assert report["taken"] == []
    assert store.list_adjustments(limit=10) == []
    with store.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_executions").fetchone()[0] == 1


def test_another_brokers_days_are_not_touched(store):
    store.upsert_executions(
        [_row("ib-1", "BUY", 10.0, 100.0, 1.0, broker="IBKR", source="IBKR_FLEX")]
    )
    file_rows = [_row("f-1", "SELL", 99.0, 999.0, source="QT_STATEMENT")]

    report = authority.apply_file_authority(
        store, broker="QUESTRADE", file_executions=file_rows, sources=["QT_API"], label="f.xlsx"
    )

    assert report["days_compared"] == 0
    assert store.list_adjustments(limit=10) == []


def test_the_summary_says_what_happened():
    assert "agree" in authority.describe_authority(
        {"days_compared": 4, "days_in_agreement": 4, "days_taken_over": 0, "days": []}
    )
    text = authority.describe_authority(
        {
            "days_compared": 4,
            "days_in_agreement": 3,
            "days_taken_over": 1,
            "days": [{"account": "A", "day": "2026-03-02", "difference": -12.5}],
            "dry_run": True,
        }
    )
    assert "would take over" in text and "2026-03-02" in text
    assert "No day" in authority.describe_authority({"days_compared": 0})
