"""R7 §9 steps 11-13 - everything the Journal tabs read, tested without Qt.

Spec §7 puts all data access behind ``ui.services.journal_feed``. That is what
makes this file possible: the tabs hold no store and no SQL, so the behaviour
that matters - tax grouping, currency honesty, R multiples, prefill, fee
separation - is testable as plain functions.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_fx as fx  # noqa: E402
from journal_store import JournalStore  # noqa: E402
from ui.models.journal import JournalTrade  # noqa: E402
from ui.services import journal_feed  # noqa: E402


def test_accounts_does_not_turn_a_store_failure_into_an_empty_journal(monkeypatch):
    monkeypatch.setattr(
        journal_feed, "_store", lambda: (_ for _ in ()).throw(RuntimeError("migration failed"))
    )
    with pytest.raises(RuntimeError, match="migration failed"):
        journal_feed.accounts()


@pytest.fixture
def feed(tmp_path, monkeypatch):
    """A feed bound to a throwaway store. Never the live journal."""
    store = JournalStore(tmp_path / "trade_journal.sqlite3")
    monkeypatch.setattr(journal_feed, "_STORE", store)
    monkeypatch.setattr(journal_feed, "_store", lambda: store)
    return store


def _execution(uid, side, quantity, price, day, **overrides):
    row = {
        "execution_uid": uid, "broker": "QUESTRADE", "account_number": "51830546",
        "account_label": "TFSA", "account_type": "TFSA", "symbol": "AAPL",
        "security_type": "STK", "currency": "USD", "side": side, "quantity": quantity,
        "price": price, "timestamp": f"{day}T09:31:00-07:00", "trade_date": day,
        "commission": 4.95, "fees": 0.05, "gross_amount": None, "net_amount": None,
        "order_id": "", "exchange_exec_id": "", "raw_json": "{}",
    }
    row.update(overrides)
    return row


def _round_trip(store, uid, day, entry, exit_price, **overrides):
    store.upsert_executions(
        [
            _execution(f"QT:{uid}a", "BUY", 100, entry, day, **overrides),
            _execution(f"QT:{uid}b", "SELL", 100, exit_price, day, **overrides),
        ]
    )


# ---------------------------------------------------------------------------
# I6 - accounts, tax grouping, and the blended badge
# ---------------------------------------------------------------------------


def test_the_four_confirmed_accounts_are_labeled_by_the_trader(feed):
    """The trader stated all four on 2026-08-15, so none is a guess."""
    from journal_migrate import TRADER_CONFIRMED_TAX_STATUS

    assert TRADER_CONFIRMED_TAX_STATUS == {
        ("QUESTRADE", "51830546"): "TAX_FREE",
        ("QUESTRADE", "29347316"): "TAXABLE",
        ("IBKR", "U4867396"): "TAX_FREE",
        ("IBKR", "U5102524"): "TAXABLE",
    }


def test_a_confirmed_status_beats_the_account_type_guess(tmp_path):
    """A stated fact outranks an inference from an account-type string."""
    store = JournalStore(tmp_path / "j.sqlite3")
    store.upsert_accounts("IBKR", [{"number": "U4867396", "type": "Margin"}])
    with store.connection() as conn:
        conn.execute("UPDATE accounts SET tax_status = 'TAXABLE', tax_status_source = 'auto'")
    from journal_migrate import migrate_to_v3

    with store.connection() as conn:
        migrate_to_v3(conn)
    labels = {row["account_number"]: (row["tax_status"], row["tax_status_source"])
              for row in store.list_accounts()}
    assert labels["U4867396"] == ("TAX_FREE", "trader")


def test_an_unfunded_account_is_kept_and_labeled(feed):
    """U4867396 is empty today. Zero balance is not zero history.

    An account that drops out of the tax grouping is an account whose past
    trades quietly stop being counted.
    """
    feed.set_account_tax_status("IBKR", "U4867396", "TAX_FREE")
    tree = journal_feed.account_tree()
    tax_free = [group for group in tree if group["tax_status"] == "TAX_FREE"]
    assert tax_free and any(a["account_number"] == "U4867396" for a in tax_free[0]["accounts"])


def test_an_unlabeled_account_gets_its_own_group_rather_than_a_guess(feed):
    feed.upsert_accounts("QUESTRADE", [{"number": "99999999", "type": "SomethingNew"}])
    groups = {group["tax_status"]: group for group in journal_feed.account_tree()}
    assert "" in groups and groups[""]["label"] == "Unlabeled"


def test_a_selection_spanning_tax_groups_is_flagged_for_the_badge(feed):
    feed.set_account_tax_status("QUESTRADE", "51830546", "TAX_FREE")
    feed.set_account_tax_status("QUESTRADE", "29347316", "TAXABLE")
    assert journal_feed.selection_spans_tax_groups(
        [("QUESTRADE", "51830546"), ("QUESTRADE", "29347316")]
    ) is True
    assert journal_feed.selection_spans_tax_groups([("QUESTRADE", "51830546")]) is False


def test_an_import_never_overwrites_a_trader_label(feed):
    """I7, at the one place an import touches the accounts table."""
    feed.set_account_tax_status("QUESTRADE", "51830546", "TAX_FREE")
    feed.upsert_accounts("QUESTRADE", [{"number": "51830546", "type": "Margin", "name": "Renamed"}])
    row = {a["account_number"]: a for a in feed.list_accounts()}["51830546"]
    assert (row["tax_status"], row["tax_status_source"]) == ("TAX_FREE", "trader")
    assert row["account_label"] == "Renamed", "the label itself still refreshes"


def test_an_unsupported_tax_status_is_refused(feed):
    with pytest.raises(ValueError, match="unsupported tax status"):
        feed.set_account_tax_status("QUESTRADE", "51830546", "SORT_OF_TAXABLE")


# ---------------------------------------------------------------------------
# The shared header: date range, currency
# ---------------------------------------------------------------------------


def test_the_date_presets_resolve_to_real_windows():
    today = date.today()
    assert journal_feed.date_range_bounds("7d") == (today - timedelta(days=7), today)
    assert journal_feed.date_range_bounds("YTD")[0] == date(today.year, 1, 1)
    assert journal_feed.date_range_bounds("All") == (None, None)
    quarter_start = journal_feed.date_range_bounds("QTD")[0]
    assert quarter_start.month in {1, 4, 7, 10} and quarter_start.day == 1


def test_a_date_range_filters_the_trades(feed):
    _round_trip(feed, "old", "2026-07-01", 100.0, 110.0, symbol="OLD")
    _round_trip(feed, "new", "2026-08-05", 100.0, 110.0, symbol="NEW")
    feed.rebuild_trades(refresh_tags=False)
    symbols = {t.symbol for t in journal_feed.load_trades(date_from="2026-08-01", date_to="2026-08-31")}
    assert symbols == {"NEW"}


def test_the_account_filter_is_exact_and_not_a_label_match(feed):
    _round_trip(feed, "a", "2026-08-05", 100.0, 110.0)
    _round_trip(feed, "b", "2026-08-05", 100.0, 110.0,
                account_number="29347316", account_label="Margin", symbol="MSFT")
    feed.rebuild_trades(refresh_tags=False)
    picked = journal_feed.load_trades(accounts_filter=[("QUESTRADE", "29347316")])
    assert {t.symbol for t in picked} == {"MSFT"}


def test_the_currency_toggle_never_relabels_an_unconverted_number(feed):
    """I5 at the render seam: 'unconverted' is a state, not a zero."""
    _round_trip(feed, "u", "2026-08-05", 100.0, 110.0)
    feed.rebuild_trades(refresh_tags=False)
    trade = journal_feed.load_trades()[0]

    native, label = journal_feed.convert_amount(trade, "Native")
    assert native is not None and label == "USD"
    value, label = journal_feed.convert_amount(trade, "CAD")
    assert value is None and label == "unconverted"

    fx.seed_rate(feed, day="2026-08-05", currency="USD", rate_to_cad=1.37)
    feed.book_cad_values()
    trade = journal_feed.load_trades()[0]
    value, label = journal_feed.convert_amount(trade, "CAD")
    assert value == pytest.approx(trade.raw["net_pnl"] * 1.37) and label == "CAD"


# ---------------------------------------------------------------------------
# R fields (I7) and the prefill
# ---------------------------------------------------------------------------


def test_the_r_fields_survive_a_note_being_saved(feed):
    _round_trip(feed, "r", "2026-08-05", 100.0, 110.0)
    feed.rebuild_trades(refresh_tags=False)
    trade_id = journal_feed.load_trades()[0].trade_id

    journal_feed.save_risk_fields(trade_id, planned_entry=100.0, planned_stop=98.0, planned_risk=200.0)
    journal_feed.save_annotation(trade_id, setup_tags="avwap-reclaim", notes="held it")
    trade = journal_feed.load_trades()[0]
    assert trade.raw["planned_risk"] == pytest.approx(200.0)
    assert trade.raw["setup_tags"] == "avwap-reclaim"


def test_the_r_multiple_is_computed_in_cad_not_in_whatever_currency(feed):
    """An R from a native P&L and a risk typed in dollars mixes currencies - B8 again."""
    _round_trip(feed, "r", "2026-08-05", 100.0, 110.0)
    fx.seed_rate(feed, day="2026-08-05", currency="USD", rate_to_cad=1.5)
    feed.rebuild_trades(refresh_tags=False)
    trade_id = journal_feed.load_trades()[0].trade_id
    journal_feed.save_risk_fields(trade_id, planned_risk=300.0)

    trade = journal_feed.load_trades()[0]
    # net_pnl 990.10 USD -> 1485.15 CAD; / 300 risk
    assert journal_feed.r_multiple(trade) == pytest.approx(trade.raw["net_pnl_cad"] / 300.0)


def test_an_r_multiple_without_a_risk_is_none_not_zero(feed):
    _round_trip(feed, "r", "2026-08-05", 100.0, 110.0)
    feed.rebuild_trades(refresh_tags=False)
    assert journal_feed.r_multiple(journal_feed.load_trades()[0]) is None


def test_the_prefill_only_fires_on_a_unique_match(feed, tmp_path, monkeypatch):
    """Same-symbol re-entries in one day are ambiguous (spec §11).

    A prefilled stop the trader did not set is a fabricated R, so anything but
    exactly one candidate declines rather than picking.
    """
    events_path = tmp_path / "alert_review_events.jsonl"
    monkeypatch.setattr(
        journal_feed, "_armed_alert_events",
        lambda: [
            json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ] if events_path.is_file() else [],
    )
    _round_trip(feed, "p", "2026-08-05", 100.0, 110.0)
    feed.rebuild_trades(refresh_tags=False)
    trade = journal_feed.load_trades()[0]

    events_path.write_text(
        json.dumps({"symbol": "AAPL", "side": "LONG", "entry": 100.0, "stop": 98.0,
                    "occurred_at": "2026-08-05T09:30:00"}) + "\n",
        encoding="utf-8",
    )
    suggestion = journal_feed.suggest_planned_risk(trade)
    assert suggestion["planned_stop"] == pytest.approx(98.0)
    assert suggestion["planned_risk"] == pytest.approx(200.0)
    assert suggestion["risk_source"] == "alert_prefill"

    events_path.write_text(
        events_path.read_text(encoding="utf-8")
        + json.dumps({"symbol": "AAPL", "side": "LONG", "entry": 101.0, "stop": 99.0,
                      "occurred_at": "2026-08-05T11:30:00"}) + "\n",
        encoding="utf-8",
    )
    assert journal_feed.suggest_planned_risk(trade) is None, "two candidates is not a match"


def test_a_missing_alert_log_is_not_an_error(feed):
    _round_trip(feed, "p", "2026-08-05", 100.0, 110.0)
    feed.rebuild_trades(refresh_tags=False)
    assert journal_feed.suggest_planned_risk(journal_feed.load_trades()[0]) is None


# ---------------------------------------------------------------------------
# Auto-tag review, corrections, manual entry
# ---------------------------------------------------------------------------


def test_accepting_a_suggestion_appends_rather_than_replaces(feed):
    _round_trip(feed, "t", "2026-08-05", 100.0, 110.0)
    feed.rebuild_trades(refresh_tags=False)
    trade_id = journal_feed.load_trades()[0].trade_id
    journal_feed.save_annotation(trade_id, setup_tags="mine", notes="")

    combined = journal_feed.accept_auto_tags(trade_id, ["avwap-reclaim"])
    assert combined == "mine; avwap-reclaim"
    assert any(row["setup_tag"] == "avwap-reclaim" for row in feed.list_tag_corrections())


def test_accepting_the_same_tag_twice_does_not_duplicate_it(feed):
    _round_trip(feed, "t", "2026-08-05", 100.0, 110.0)
    feed.rebuild_trades(refresh_tags=False)
    trade_id = journal_feed.load_trades()[0].trade_id
    journal_feed.accept_auto_tags(trade_id, ["avwap-reclaim"])
    assert journal_feed.accept_auto_tags(trade_id, ["avwap-reclaim"]) == "avwap-reclaim"


def test_a_correction_from_the_ui_reaches_the_rebuild(feed):
    _round_trip(feed, "c", "2026-08-05", 100.0, 110.0)
    feed.rebuild_trades(refresh_tags=False)
    journal_feed.record_adjustment(
        action="EDIT_EXECUTION", target_uid="QT:cb", payload={"price": 112.0},
        reason="statement says 112.00",
    )
    feed.rebuild_trades(refresh_tags=False)
    assert journal_feed.load_trades()[0].raw["gross_pnl"] == pytest.approx(1200.0)


def test_the_group_key_for_a_trade_is_what_a_force_close_targets(feed):
    _round_trip(feed, "g", "2026-08-05", 100.0, 110.0)
    feed.rebuild_trades(refresh_tags=False)
    trade = journal_feed.load_trades()[0]
    assert journal_feed.group_key_for(trade) == "QUESTRADE|51830546|AAPL|STK|USD"


def test_a_hand_entered_fill_lands_in_a_real_account_not_in_manual(feed):
    """The deferred half of §5 fix 3: the pickers exist so this is possible.

    ``broker="MANUAL"`` was the old default and made every hand-entered fill an
    orphan that could never attach to the position it belonged to (B3).
    """
    _round_trip(feed, "m", "2026-08-05", 100.0, 110.0)
    journal_feed.add_manual_execution(
        {
            "broker": "QUESTRADE", "account_number": "51830546", "symbol": "AAPL",
            "security_type": "STK", "currency": "USD", "side": "BUY", "quantity": 50,
            "price": 105.0, "timestamp": "2026-08-06T09:31:00-07:00", "execution_id": "hand-1",
        }
    )
    feed.rebuild_trades(refresh_tags=False)
    trades = journal_feed.load_trades()
    assert len(trades) == 2
    assert {t.raw["broker"] for t in trades} == {"QUESTRADE"}, "no MANUAL orphan"


# ---------------------------------------------------------------------------
# Analytics, calendar, fees
# ---------------------------------------------------------------------------


def test_the_equity_curve_refuses_a_mixed_selection_it_cannot_fully_convert(feed):
    """Omitting one trade would make the final curve total look complete."""
    _round_trip(feed, "u", "2026-08-05", 100.0, 110.0)
    _round_trip(feed, "c", "2026-08-06", 80.0, 85.0, currency="CAD", symbol="SHOP.TO")
    feed.rebuild_trades(refresh_tags=False)
    trades = journal_feed.load_trades()

    curve = journal_feed.equity_curve(trades, "CAD")
    assert curve == []
    assert journal_feed.unconvertible_count(trades, "CAD") == 1


def test_the_calendar_returns_pnl_by_day(feed):
    _round_trip(feed, "d", "2026-08-05", 100.0, 110.0)
    feed.rebuild_trades(refresh_tags=False)
    calendar = journal_feed.calendar_pnl_by_day()
    assert calendar.get("2026-08-05") == pytest.approx(990.0)  # 1000 gross - 9.90 commission - 0.10 fees


def test_calendar_and_analytics_use_the_same_selected_currency(feed):
    _round_trip(feed, "usd", "2026-08-05", 100.0, 110.0, symbol="AAPL")
    _round_trip(feed, "cad", "2026-08-06", 80.0, 85.0, currency="CAD", symbol="SHOP.TO")
    feed.rebuild_trades(refresh_tags=False)
    fx.seed_rate(feed, day="2026-08-05", currency="USD", rate_to_cad=1.4)
    feed.book_cad_values()
    trades = journal_feed.load_trades()

    calendar = journal_feed.calendar_pnl_by_day(currency_mode="CAD")
    summary = journal_feed.analytics_summary(trades, "CAD")

    assert sum(calendar.values()) == pytest.approx(summary["overall"]["net_pnl"])
    assert journal_feed.equity_curve(trades, "CAD")[-1][1] == pytest.approx(sum(calendar.values()))


def test_fee_totals_never_add_trade_costs_to_cash_fees(feed):
    """One is already inside net P&L and the other is not; one column would double-count."""
    _round_trip(feed, "f", "2026-08-05", 100.0, 110.0)
    feed.rebuild_trades(refresh_tags=False)
    feed.upsert_cash_transactions(
        [
            {"txn_uid": "x1", "broker": "QUESTRADE", "account_number": "51830546",
             "txn_date": "2026-08-05", "activity_type": "FEE", "amount": -3.0, "currency": "USD"},
            {"txn_uid": "x2", "broker": "QUESTRADE", "account_number": "51830546",
             "txn_date": "2026-08-05", "activity_type": "DIVIDEND", "amount": 12.5, "currency": "USD"},
        ]
    )
    rows = journal_feed.fee_totals()
    trade_row = [row for row in rows if row["commission"]][0]
    assert trade_row["commission"] == pytest.approx(9.9)
    assert trade_row["fees"] == pytest.approx(0.1)
    cash_row = [row for row in rows if row["cash_fees"] or row["dividends"]][0]
    assert cash_row["cash_fees"] == pytest.approx(-3.0)
    assert cash_row["dividends"] == pytest.approx(12.5)
    assert len(rows) == 1, "trade costs and cash activity share the broker/account-number key"


def test_fee_totals_apply_the_header_date_and_account_filters_to_cash_rows(feed):
    _round_trip(feed, "f", "2026-08-05", 100.0, 110.0)
    feed.rebuild_trades(refresh_tags=False)
    feed.upsert_cash_transactions(
        [
            {"txn_uid": "wanted", "broker": "QUESTRADE", "account_number": "51830546",
             "txn_date": "2026-08-05", "activity_type": "FEE", "amount": -3.0, "currency": "USD"},
            {"txn_uid": "other-account", "broker": "QUESTRADE", "account_number": "29347316",
             "txn_date": "2026-08-05", "activity_type": "FEE", "amount": -99.0, "currency": "USD"},
            {"txn_uid": "old", "broker": "QUESTRADE", "account_number": "51830546",
             "txn_date": "2025-01-01", "activity_type": "FEE", "amount": -88.0, "currency": "USD"},
        ]
    )

    rows = journal_feed.fee_totals(
        date_from="2026-08-01", date_to="2026-08-31",
        accounts_filter=[("QUESTRADE", "51830546")],
    )

    assert len(rows) == 1
    assert rows[0]["account"] == "51830546" and rows[0]["cash_fees"] == pytest.approx(-3.0)


def test_the_fee_export_writes_every_column(feed, tmp_path):
    _round_trip(feed, "f", "2026-08-05", 100.0, 110.0)
    feed.rebuild_trades(refresh_tags=False)
    target = journal_feed.export_fees_csv(tmp_path / "fees.csv")
    header = target.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == [
        "broker", "account", "currency", "commission", "fees", "cash_fees", "dividends"
    ]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_the_coverage_grid_has_a_row_per_account_and_a_column_per_day(feed):
    import journal_coverage

    journal_coverage.mark_range(
        feed, broker="QUESTRADE", account_number="51830546",
        start=date.today() - timedelta(days=3), end=date.today(), status=journal_coverage.COVERED,
    )
    grid = journal_feed.coverage_grid(days=7)
    assert len(grid["days"]) == 8
    assert ("QUESTRADE", "51830546") in grid["accounts"]


def test_the_health_tab_can_see_the_gaps_without_running_an_import(feed):
    import journal_coverage

    journal_coverage.mark_coverage(
        feed, broker="QUESTRADE", account_number="51830546",
        day=date.today() - timedelta(days=2), status=journal_coverage.FAILED, message="503",
    )
    gaps = journal_feed.find_coverage_gaps(days=5)
    assert gaps and gaps[0]["broker"] == "QUESTRADE"


def test_the_reconciliation_report_is_readable_from_the_feed(feed):
    import journal_reconcile

    _round_trip(feed, "h", "2026-08-05", 100.0, 110.0)
    feed.upsert_executions([_execution("QT:open", "BUY", 10, 100.0, "2026-08-06", symbol="NVDA")])
    feed.rebuild_trades(refresh_tags=False)
    journal_reconcile.reconcile(feed, [], brokers=["QUESTRADE"])
    report = journal_feed.last_reconciliation()
    assert report and report["suggestions"]


def test_fx_coverage_is_reportable(feed):
    _round_trip(feed, "u", "2026-08-05", 100.0, 110.0)
    feed.rebuild_trades(refresh_tags=False)
    assert journal_feed.fx_coverage()["unconverted"] == [{"currency": "USD", "trades": 1}]
