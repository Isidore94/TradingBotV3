"""R7 §9 steps 11-13 - the Journal shell and its five tabs, constructed for real.

Offline by construction: the feed is bound to a temporary store, and no test
here reaches a broker, the network, or the live journal database. What these
cover is the wiring the pure-function tests in ``test_journal_feed.py`` cannot -
that the tabs exist, load, render what the feed hands them, and route the
signals between each other.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt

PySide6 = pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

from journal_store import JournalStore  # noqa: E402
from ui.services import journal_feed  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A throwaway journal. Never the live one."""
    journal = JournalStore(tmp_path / "trade_journal.sqlite3")
    monkeypatch.setattr(journal_feed, "_STORE", journal)
    monkeypatch.setattr(journal_feed, "_store", lambda: journal)
    return journal


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


@pytest.fixture
def populated(store):
    store.upsert_executions(
        [
            _execution("QT:1", "BUY", 100, 150.0, "2026-08-03"),
            _execution("QT:2", "SELL", 100, 160.0, "2026-08-03"),
            _execution("QT:3", "BUY", 50, 90.0, "2026-08-05",
                       symbol="AMD", account_number="29347316", account_label="Margin"),
        ]
    )
    store.set_account_tax_status("QUESTRADE", "51830546", "TAX_FREE")
    store.set_account_tax_status("QUESTRADE", "29347316", "TAXABLE")
    store.rebuild_trades(refresh_tags=False)
    return store


@pytest.fixture
def panel(qapp, populated):
    from ui.panels.journal_panel import JournalPanel

    widget = JournalPanel()
    yield widget
    widget.shutdown()
    widget.deleteLater()


# ---------------------------------------------------------------------------
# The shell
# ---------------------------------------------------------------------------


def test_the_journal_is_five_tabs_over_one_header(panel):
    assert [panel.tabs.tabText(i) for i in range(panel.tabs.count())] == [
        "Trades", "Calendar", "Analytics", "Health", "Fees"
    ]
    assert panel.header is panel.trades_tab._header is panel.fees_tab._header


def test_the_panel_keeps_the_surface_the_app_depends_on(panel):
    for name in ("statusChanged", "shutdown"):
        assert hasattr(panel, name), f"ui/app.py calls {name}"
    for dead_shell_method in ("rebuild_trades", "export_csv", "refresh"):
        assert not hasattr(panel, dead_shell_method)


def test_first_open_runs_real_store_initialization_off_the_gui_thread(qapp, tmp_path, monkeypatch):
    import journal_store
    from PySide6.QtCore import QThread
    from ui.panels.journal_panel import JournalPanel

    real_store = journal_store.JournalStore
    called_on = []

    def tracked_store(*args, **kwargs):
        called_on.append(QThread.currentThread())
        return real_store(*args, **kwargs)

    monkeypatch.setattr(journal_store, "JournalStore", tracked_store)
    monkeypatch.setattr(journal_feed, "_STORE", None)
    db_path = tmp_path / "first-open.sqlite3"
    monkeypatch.setattr(journal_feed, "journal_db_path", lambda: db_path)

    widget = JournalPanel()
    try:
        assert widget._migration_worker is None
        assert "dry-run first" in widget.migration_status.text()
        widget.prepare_button.click()
        assert widget._migration_worker.wait(10000)
        qapp.processEvents()
        assert called_on and called_on[0] is not qapp.thread()
        assert db_path.is_file()
        assert widget.tabs.isEnabled()
        assert "migration completed" in widget.migration_status.text().lower()
    finally:
        widget.shutdown()
        widget.deleteLater()


def test_fresh_process_opens_an_already_v3_database_without_prepare_gate(qapp, tmp_path, monkeypatch):
    from ui.panels.journal_panel import JournalPanel

    db_path = tmp_path / "already-v3.sqlite3"
    JournalStore(db_path)
    monkeypatch.setattr(journal_feed, "_STORE", None)
    monkeypatch.setattr(journal_feed, "journal_db_path", lambda: db_path)

    widget = JournalPanel()
    try:
        assert widget._migration_worker is None
        assert widget.tabs.isEnabled()
        assert not widget.prepare_button.isVisible()
        assert journal_feed._STORE is not None
    finally:
        widget.shutdown()
        widget.deleteLater()


def test_migration_failure_stays_visible_instead_of_claiming_no_accounts(qapp, monkeypatch):
    from ui.panels.journal_panel import JournalPanel

    monkeypatch.setattr(journal_feed, "_STORE", None)
    monkeypatch.setattr(
        journal_feed, "initialize_store", lambda: (_ for _ in ()).throw(RuntimeError("backup refused"))
    )
    widget = JournalPanel()
    try:
        widget.prepare_button.click()
        assert widget._migration_worker.wait(5000)
        qapp.processEvents()
        assert "migration failed" in widget.migration_status.text().lower()
        assert "backup refused" in widget.migration_status.text()
        assert widget.header.account_button.text() != "No accounts"
        assert not widget.tabs.isEnabled()
    finally:
        widget.shutdown()
        widget.deleteLater()


def test_only_the_visible_tab_reloads(panel, monkeypatch):
    """Analytics and Health are the expensive ones; reloading all five on every
    click of the account tree is work nobody is looking at."""
    calls: list[str] = []
    for name in ("trades_tab", "calendar_tab", "analytics_tab", "health_tab", "fees_tab"):
        tab = getattr(panel, name)
        monkeypatch.setattr(tab, "reload", lambda n=name: calls.append(n))
    panel.header.selectionChanged.emit()
    assert calls == ["trades_tab"]


# ---------------------------------------------------------------------------
# The shared header (I6)
# ---------------------------------------------------------------------------


def test_the_account_tree_groups_by_tax_treatment(panel):
    labels = [action.text() for action in panel.header.account_menu.actions()]
    assert any("51830546" in text for text in labels)
    assert any("29347316" in text for text in labels)


def test_a_blended_selection_is_badged_and_a_single_group_is_not(panel):
    """I6: never silently. A badge, not a refusal - the trader may look at
    everything at once and may not do it by accident."""
    assert panel.header.blend_badge.isVisibleTo(panel.header) is True

    for action in panel.header.account_menu.actions():
        if "29347316" in action.text():
            action.setChecked(False)
    assert panel.header.blend_badge.isVisibleTo(panel.header) is False


def test_all_accounts_selected_means_no_filter_not_an_empty_list(panel):
    """None and [] must stay distinguishable: [] is what the trader sees after
    unchecking the last account, and an empty journal is the correct answer."""
    assert panel.header.selected_accounts is None
    for action in panel.header.account_menu.actions():
        if action.isCheckable():
            action.setChecked(False)
    assert panel.header.selected_accounts == []


def test_the_date_preset_switches_the_custom_pickers_on_and_off(panel):
    panel.header.range_input.setCurrentText("Custom")
    assert panel.header.date_from.isVisibleTo(panel.header) is True
    panel.header.range_input.setCurrentText("30d")
    assert panel.header.date_from.isVisibleTo(panel.header) is False


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------


def test_the_trades_table_renders_what_the_feed_returns(panel):
    panel.header.range_input.setCurrentText("All")
    panel.trades_tab.reload()
    assert panel.trades_tab.table.rowCount() == 2
    symbols = {panel.trades_tab.table.item(row, 1).text() for row in range(2)}
    assert symbols == {"AAPL", "AMD"}


def test_header_symbol_status_and_direction_filters_reach_the_store(panel):
    panel.header.range_input.setCurrentText("All")
    panel.header.symbol_input.setText("amd")
    panel.header.status_input.setCurrentText("OPEN")
    panel.header.direction_input.setCurrentText("LONG")

    panel.trades_tab.reload()

    assert panel.trades_tab.table.rowCount() == 1
    assert panel.trades_tab.table.item(0, 1).text() == "AMD"
    assert panel.trades_tab.table.item(0, 3).text() == "OPEN"


def test_an_unconverted_pnl_reads_as_unconverted_and_never_as_a_number(panel):
    """I5 at the render seam. The fixture books no FX rate, so CAD is unavailable."""
    panel.header.range_input.setCurrentText("All")
    panel.header.currency_input.setCurrentText("CAD")
    panel.trades_tab.reload()
    cells = {panel.trades_tab.table.item(row, 5).text() for row in range(panel.trades_tab.table.rowCount())}
    assert cells == {"unconverted"}


def test_selecting_a_trade_fills_the_detail_pane_and_its_legs(panel):
    panel.header.range_input.setCurrentText("All")
    panel.trades_tab.reload()
    panel.trades_tab.table.selectRow(0)
    assert panel.trades_tab._current is not None
    assert panel.trades_tab.legs_table.rowCount() >= 1


def test_structured_trade_review_is_captured_through_the_real_store(panel, populated):
    panel.header.range_input.setCurrentText("All")
    panel.trades_tab.reload()
    panel.trades_tab.table.selectRow(0)
    trade_id = panel.trades_tab._current.trade_id
    panel.trades_tab.review_outcome.setCurrentText("Poor exit discipline")
    panel.trades_tab.decision_reason.setText("Moved the stop without a new level")

    panel.trades_tab._save_review()

    stored = populated.latest_trade_review(trade_id)
    assert stored["payload"]["review_outcome"] == "Poor exit discipline"
    assert stored["reason"] == "Moved the stop without a new level"


def test_the_r_readout_says_what_is_missing_rather_than_showing_a_zero(panel):
    panel.header.range_input.setCurrentText("All")
    panel.trades_tab.reload()
    panel.trades_tab.table.selectRow(0)
    assert "needs risk" in panel.trades_tab.r_readout.text()


def test_prefill_declines_out_loud_when_nothing_matches(panel, monkeypatch):
    """A silent no-op would read as "no alert found" when the honest answer may
    be "more than one, and I will not guess"."""
    messages: list[str] = []
    panel.trades_tab.statusChanged.connect(messages.append)
    monkeypatch.setattr(journal_feed, "suggest_planned_risk", lambda trade: None)
    panel.header.range_input.setCurrentText("All")
    panel.trades_tab.reload()
    panel.trades_tab.table.selectRow(0)
    panel.trades_tab._prefill_risk()
    assert messages and "unambiguous match" in messages[-1]


def test_saving_a_plan_stores_it_and_the_r_appears(panel, populated):
    import journal_fx

    panel.header.range_input.setCurrentText("All")
    panel.trades_tab.reload()
    # The closed AAPL trade specifically: the table sorts newest first, and the
    # AMD row is an open position on a day this test books no rate for.
    aapl_row = next(
        row for row in range(panel.trades_tab.table.rowCount())
        if panel.trades_tab.table.item(row, 1).text() == "AAPL"
    )
    panel.trades_tab.table.selectRow(aapl_row)
    trade_id = panel.trades_tab._current.trade_id
    journal_fx.seed_rate(populated, day="2026-08-03", currency="USD", rate_to_cad=1.4)
    populated.book_cad_values()

    panel.trades_tab.planned_risk.setValue(500.0)
    panel.trades_tab._save_risk()
    stored = {t.trade_id: t for t in journal_feed.load_trades()}[trade_id]
    assert stored.raw["planned_risk"] == pytest.approx(500.0)
    assert journal_feed.r_multiple(stored) is not None


def test_the_corrections_dialog_refuses_to_accept_without_a_reason(qapp, panel):
    """The audit trail behind a tax filing does not take unexplained entries."""
    from PySide6.QtWidgets import QDialogButtonBox

    from ui.panels.journal.trades_tab import CorrectionsDialog

    panel.header.range_input.setCurrentText("All")
    panel.trades_tab.reload()
    panel.trades_tab.table.selectRow(0)
    dialog = CorrectionsDialog(panel.trades_tab._current)
    try:
        assert dialog.buttons.button(QDialogButtonBox.Ok).isEnabled() is False
        dialog.reason_input.setPlainText("the statement says 161.00")
        assert dialog.buttons.button(QDialogButtonBox.Ok).isEnabled() is True
    finally:
        dialog.deleteLater()


def test_a_force_close_targets_the_position_and_an_edit_targets_a_leg(qapp, panel):
    from ui.panels.journal.trades_tab import CorrectionsDialog

    panel.header.range_input.setCurrentText("All")
    panel.trades_tab.reload()
    panel.trades_tab.table.selectRow(0)
    dialog = CorrectionsDialog(panel.trades_tab._current)
    try:
        dialog.action_input.setCurrentIndex(
            [dialog.action_input.itemData(i) for i in range(dialog.action_input.count())].index("FORCE_CLOSE")
        )
        assert "|" in str(dialog.target_input.currentData()), "a group key, not an execution uid"
        dialog.action_input.setCurrentIndex(
            [dialog.action_input.itemData(i) for i in range(dialog.action_input.count())].index("EDIT_EXECUTION")
        )
        assert str(dialog.target_input.currentData()).startswith("QT:")
    finally:
        dialog.deleteLater()


def test_the_manual_execution_dialog_offers_real_accounts_not_manual(qapp, panel):
    """§5 fix 3's deferred half. broker="MANUAL" made every hand-entered fill an
    orphan that could never attach to the position it belonged to."""
    from ui.panels.journal.trades_tab import ManualExecutionDialog

    dialog = ManualExecutionDialog()
    try:
        brokers = [dialog.broker_input.itemText(i) for i in range(dialog.broker_input.count())]
        assert "QUESTRADE" in brokers and "MANUAL" not in brokers
        accounts = [dialog.account_input.itemData(i) for i in range(dialog.account_input.count())]
        assert "51830546" in accounts
    finally:
        dialog.deleteLater()


# ---------------------------------------------------------------------------
# Calendar, Analytics, Health, Fees
# ---------------------------------------------------------------------------


def test_the_calendar_paints_a_day_and_clicking_it_filters_trades(panel):
    panel.header.range_input.setCurrentText("All")
    panel.calendar_tab.reload()
    panel.calendar_tab.month_input.setCurrentIndex(7)  # August
    panel.calendar_tab.year_input.setCurrentText("2026")

    seen: list[str] = []
    panel.calendar_tab.daySelected.connect(seen.append)
    panel.calendar_tab.daySelected.emit("2026-08-03")
    assert seen == ["2026-08-03"]
    assert panel.tabs.currentWidget() is panel.trades_tab
    assert panel.header.range_input.currentText() == "Custom"


def test_analytics_says_why_a_total_is_missing_instead_of_showing_a_wrong_one(panel, populated):
    """B8, rendered. Mixed currencies with anything unconverted refuses to total."""
    populated.upsert_executions(
        [
            _execution("QT:c1", "BUY", 100, 80.0, "2026-08-04", symbol="SHOP.TO", currency="CAD"),
            _execution("QT:c2", "SELL", 100, 85.0, "2026-08-04", symbol="SHOP.TO", currency="CAD"),
        ]
    )
    populated.rebuild_trades(refresh_tags=False)
    panel.header.range_input.setCurrentText("All")
    panel.analytics_tab.reload()
    assert panel.analytics_tab.currency_note.isVisibleTo(panel.analytics_tab) is True
    assert "no booked FX rate" in panel.analytics_tab.currency_note.text()
    assert "net not shown" in panel.analytics_tab.headline.text()


def test_the_equity_curve_reports_what_it_had_to_leave_out(panel, populated):
    populated.upsert_executions(
        [
            _execution("QT:c1", "BUY", 100, 80.0, "2026-08-04", symbol="SHOP.TO", currency="CAD"),
            _execution("QT:c2", "SELL", 100, 85.0, "2026-08-04", symbol="SHOP.TO", currency="CAD"),
        ]
    )
    populated.rebuild_trades(refresh_tags=False)
    panel.header.range_input.setCurrentText("All")
    panel.header.currency_input.setCurrentText("CAD")
    panel.analytics_tab.reload()
    assert "not in the curve" in panel.analytics_tab.currency_note.text()
    assert panel.analytics_tab.curve_table.rowCount() == 0, (
        "a mixed selection with an unconverted trade refuses the total instead of silently omitting it"
    )


def test_analytics_walkaway_renders_structured_engine_output(panel):
    panel.analytics_tab._on_walkaway_done(
        {"journal_rows": [], "focus_rows": [], "skipped_non_equity": 0}
    )
    assert "WALKAWAY ANALYSIS" in panel.analytics_tab.walkaway_output.text()
    assert "journal_rows" not in panel.analytics_tab.walkaway_output.text()
    assert panel.analytics_tab.export_button.text() == "Export trades CSV"


def test_health_shows_the_coverage_grid_and_names_the_gaps(panel, populated):
    import journal_coverage
    from datetime import date, timedelta

    journal_coverage.mark_coverage(
        populated, broker="QUESTRADE", account_number="51830546",
        day=date.today() - timedelta(days=2), status=journal_coverage.FAILED, message="503",
    )
    panel.health_tab.reload()
    assert panel.health_tab.coverage_table.rowCount() >= 1
    assert "uncovered session day" in panel.health_tab.gaps_label.text()


def test_health_reports_unconverted_trades_as_excluded_not_as_zero(panel):
    panel.health_tab.reload()
    assert "rather than counted as zero" in panel.health_tab.fx_label.text()


def test_health_offers_the_flex_fields_the_qt_panel_never_had(panel):
    """A1/A9: the only complete import path used to be a CLI the trader never ran."""
    for widget in (panel.health_tab.flex_token_input, panel.health_tab.flex_query_input):
        assert widget is not None
    assert panel.health_tab.pull_button.text() == "Pull today now"
    assert panel.health_tab.backfill_button.text() == "Backfill Questrade gaps"
    assert panel.health_tab.retry_button.text() == "Retry failed Questrade days"


def test_health_routes_the_three_broker_actions_distinctly(panel, monkeypatch, qapp):
    calls = []
    monkeypatch.setattr(
        journal_feed, "pull_today",
        lambda: calls.append("pull") or {"source_results": [], "messages": ["today done"]},
    )
    monkeypatch.setattr(
        journal_feed, "self_heal_gaps",
        lambda *, failed_only=False: calls.append("failed" if failed_only else "gaps")
        or {"repaired": [], "failed": [], "exhausted": [], "unsupported_brokers": ["IBKR"]},
    )

    for action in ("pull", "gaps", "failed"):
        panel.health_tab._start_task(action)
        assert panel.health_tab._task.wait(5000)
        qapp.processEvents()

    assert calls == ["pull", "gaps", "failed"]


def test_health_lists_a_reconciliation_and_its_suggestions(panel, populated):
    import journal_reconcile

    journal_reconcile.reconcile(populated, [], brokers=["QUESTRADE"])
    panel.health_tab.reload()
    assert "mismatch" in panel.health_tab.reconcile_label.text()
    assert panel.health_tab.suggestions_list.count() >= 1


def test_health_keeps_force_close_suggestions_aligned_with_their_mismatch_rows(
    panel, populated, monkeypatch
):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QMessageBox
    import journal_reconcile

    first_key = "QUESTRADE|51830546|AAPL|STK|USD"
    second_key = "QUESTRADE|51830546|MSFT|STK|USD"
    suggestion = {
        "action": "FORCE_CLOSE",
        "group_key": second_key,
        "broker": "QUESTRADE",
        "symbol": "MSFT",
        "reason": "broker reports flat",
    }
    journal_reconcile.store_report(
        populated,
        {
            "positions_checked": 2,
            "checked_at": "2026-08-15T12:00:00-07:00",
            "mismatched": [
                {"kind": "QUANTITY_MISMATCH", "group_key": first_key, "broker": "QUESTRADE",
                 "symbol": "AAPL", "journal_quantity": 10, "broker_quantity": 5},
                {"kind": "JOURNAL_OPEN_BROKER_FLAT", "group_key": second_key,
                 "broker": "QUESTRADE", "symbol": "MSFT", "journal_quantity": 20,
                 "broker_quantity": 0},
            ],
            "suggestions": [suggestion],
        },
    )
    panel.health_tab.reload()

    assert panel.health_tab.suggestions_list.item(0).data(Qt.UserRole) is None
    assert panel.health_tab.suggestions_list.item(1).data(Qt.UserRole)["symbol"] == "MSFT"
    confirmed = []
    monkeypatch.setattr(
        "ui.panels.journal.health_tab.QMessageBox.question", lambda *a, **k: QMessageBox.Yes
    )
    monkeypatch.setattr(
        journal_feed, "confirm_reconciliation_suggestion",
        lambda selected, **kwargs: confirmed.append(selected),
    )
    panel.health_tab.suggestions_list.setCurrentRow(1)
    panel.health_tab._confirm_suggestion()
    assert confirmed == [suggestion]


def test_fees_never_adds_trade_costs_to_cash_fees(panel, populated):
    populated.upsert_cash_transactions(
        [
            {"txn_uid": "c1", "broker": "QUESTRADE", "account_number": "51830546",
             "txn_date": "2026-08-05", "activity_type": "FEE", "amount": -3.0, "currency": "USD"},
        ]
    )
    panel.header.range_input.setCurrentText("All")
    panel.fees_tab.reload()
    headers = [
        panel.fees_tab.totals_table.horizontalHeaderItem(i).text()
        for i in range(panel.fees_tab.totals_table.columnCount())
    ]
    assert "Commission" in headers and "Cash fees" in headers
    assert "never added" in panel.fees_tab.note.text()


def test_every_tab_survives_an_empty_journal(qapp, store):
    """A brand-new desk opens this tab before importing anything."""
    from ui.panels.journal_panel import JournalPanel

    widget = JournalPanel()
    try:
        for index in range(widget.tabs.count()):
            widget.tabs.setCurrentIndex(index)
        assert widget.trades_tab.table.rowCount() == 0
    finally:
        widget.shutdown()
        widget.deleteLater()
