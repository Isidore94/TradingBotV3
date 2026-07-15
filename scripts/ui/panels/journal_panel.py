from __future__ import annotations

from PySide6.QtCore import QItemSelection, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from journal_importers import QUESTRADE_REFRESH_TOKEN_SETTING, QuestradeImporter
from project_paths import get_local_setting, save_local_setting
from ui.models.journal import JournalTrade
from ui.models.journal_table_model import ROW_ROLE, JournalFilterProxyModel, JournalTableModel
from ui.services import journal_feed
from ui.services.journal_import_helpers import DEFAULT_QUESTRADE_PULL_DAYS
from ui.services.journal_import_service import JournalImportService
from ui.widgets.data_table import DataTable
from ui.widgets.empty_state import EmptyState
from ui.widgets.kpi_tile import KpiTile
from ui.widgets.section_header import SectionHeader


class JournalPanel(QFrame):
    statusChanged = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self._trades: list[JournalTrade] = []
        self.import_service = JournalImportService(self)
        self.import_service.started.connect(self._on_import_started)
        self.import_service.finished.connect(self._on_import_finished)
        self.import_service.failed.connect(self._on_import_failed)

        self.model = JournalTableModel()
        self.proxy = JournalFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.table = DataTable()
        self.table.setModel(self.proxy)
        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)

        self.empty_state = EmptyState(
            "No journaled trades yet",
            "Import executions through the journal pipeline, then refresh. Closed trades show realized P&L, tags, and notes.",
            "Rebuild From Executions",
        )
        self.empty_state.action_button.clicked.connect(self.rebuild_trades)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.empty_state)
        self.stack.addWidget(self.table)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter symbol, tag, account, note")
        self.search_input.textChanged.connect(self._apply_filters)
        self.broker_input = QComboBox()
        self.broker_input.currentTextChanged.connect(self._reload)
        self.account_input = QComboBox()
        self.account_input.currentTextChanged.connect(self._reload)
        self.status_input = QComboBox()
        self.status_input.addItems(["ALL", "CLOSED", "OPEN"])
        self.status_input.currentTextChanged.connect(self._apply_filters)
        self.direction_input = QComboBox()
        self.direction_input.addItems(["ALL", "LONG", "SHORT"])
        self.direction_input.currentTextChanged.connect(self._apply_filters)

        self.broker_sync_toggle = QPushButton("Broker sync >")
        self.broker_sync_toggle.setCheckable(True)
        self.broker_sync_toggle.toggled.connect(self._toggle_broker_sync)
        self.broker_sync_body = self._build_broker_sync_body()
        self.broker_sync_body.setVisible(False)

        self.trades_tile = KpiTile("Trades", "0")
        self.winrate_tile = KpiTile("Win Rate", "-")
        self.pnl_tile = KpiTile("Net P&L", "-")
        self.pf_tile = KpiTile("Profit Factor", "-")

        self.detail = JournalDetailPanel()
        self.detail.annotationSaved.connect(self._on_annotation_saved)

        self.status_label = QLabel("")
        self.status_label.setObjectName("MutedLabel")

        self._build_layout()
        self._populate_filter_options()
        self._reload()

    def _build_layout(self) -> None:
        header = SectionHeader("Trade Journal", "Imported executions grouped into trades with realized P&L, tags, and notes.")
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._reload)
        rebuild_button = QPushButton("Rebuild")
        rebuild_button.clicked.connect(self.rebuild_trades)
        export_button = QPushButton("Export CSV")
        export_button.clicked.connect(self.export_csv)
        for button in (refresh_button, rebuild_button, export_button):
            header.add_action(button)

        kpi_row = QHBoxLayout()
        kpi_row.setContentsMargins(0, 0, 0, 0)
        kpi_row.setSpacing(8)
        for tile in (self.trades_tile, self.winrate_tile, self.pnl_tile, self.pf_tile):
            kpi_row.addWidget(tile)
        kpi_row.addStretch(1)

        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.setSpacing(8)
        filter_row.addWidget(self.search_input, 2)
        filter_row.addWidget(QLabel("Broker"))
        filter_row.addWidget(self.broker_input)
        filter_row.addWidget(QLabel("Account"))
        filter_row.addWidget(self.account_input)
        filter_row.addWidget(QLabel("Status"))
        filter_row.addWidget(self.status_input)
        filter_row.addWidget(QLabel("Side"))
        filter_row.addWidget(self.direction_input)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)
        body.addWidget(self.stack, 3)
        body.addWidget(self.detail, 2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(header)
        layout.addWidget(self.broker_sync_toggle)
        layout.addWidget(self.broker_sync_body)
        layout.addLayout(kpi_row)
        layout.addLayout(filter_row)
        layout.addLayout(body, 1)
        layout.addWidget(self.status_label)

    def _build_broker_sync_body(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("InfoDrawer")

        self.questrade_refresh_input = QLineEdit()
        self.questrade_refresh_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.questrade_refresh_input.setPlaceholderText("Paste Questrade refresh token")
        self.questrade_refresh_input.setText(str(get_local_setting(QUESTRADE_REFRESH_TOKEN_SETTING, "") or ""))

        self.questrade_save_button = QPushButton("Save")
        self.questrade_save_button.clicked.connect(self._save_questrade_refresh_token)
        self.questrade_clear_button = QPushButton("Clear")
        self.questrade_clear_button.clicked.connect(self._clear_questrade_refresh_token)

        self.questrade_days_input = QSpinBox()
        self.questrade_days_input.setRange(1, 31)
        self.questrade_days_input.setValue(DEFAULT_QUESTRADE_PULL_DAYS)
        self.questrade_days_input.setPrefix("Days ")

        self.questrade_pull_button = QPushButton("Pull New Trades")
        self.questrade_pull_button.setObjectName("PrimaryButton")
        self.questrade_pull_button.clicked.connect(self._pull_new_questrade_trades)

        self.questrade_status_label = QLabel("")
        self.questrade_status_label.setObjectName("MutedLabel")
        self.questrade_status_label.setWordWrap(True)

        note = QLabel("Questrade rotates refresh tokens; paste the current token here when it changes.")
        note.setObjectName("MutedLabel")
        note.setWordWrap(True)

        token_row = QHBoxLayout()
        token_row.setContentsMargins(0, 0, 0, 0)
        token_row.setSpacing(8)
        token_row.addWidget(QLabel("Questrade refresh token"))
        token_row.addWidget(self.questrade_refresh_input, 1)
        token_row.addWidget(self.questrade_save_button)
        token_row.addWidget(self.questrade_clear_button)

        pull_row = QHBoxLayout()
        pull_row.setContentsMargins(0, 0, 0, 0)
        pull_row.setSpacing(8)
        pull_row.addWidget(self.questrade_days_input)
        pull_row.addWidget(self.questrade_pull_button)
        pull_row.addStretch(1)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        layout.addWidget(SectionHeader("Broker Sync", "Questrade import settings and on-demand trade pull."))
        layout.addLayout(token_row)
        layout.addWidget(note)
        layout.addLayout(pull_row)
        layout.addWidget(self.questrade_status_label)

        self._refresh_questrade_status()
        return frame

    def _populate_filter_options(self) -> None:
        for combo, column in ((self.broker_input, "broker"), (self.account_input, "account_label")):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("All")
            for value in journal_feed.distinct_values(column):
                if value:
                    combo.addItem(str(value))
            combo.blockSignals(False)

    def _reload(self) -> None:
        try:
            trades = journal_feed.load_trades(
                broker=self.broker_input.currentText() or "All",
                account=self.account_input.currentText() or "All",
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._set_status(f"Journal load failed: {exc}")
            trades = []
        self._trades = trades
        self.model.set_rows(trades)
        self._apply_filters()
        self.stack.setCurrentWidget(self.table if trades else self.empty_state)
        if trades:
            self.table.fit_columns()
        self._update_kpis(trades)
        self.detail.show_analytics(journal_feed.analytics_text(trades) if trades else "No journaled trades yet.")
        self._set_status(f"Loaded {len(trades)} trade(s) from {journal_feed.journal_db_path().name}.")

    def rebuild_trades(self) -> None:
        try:
            count = journal_feed.rebuild_trades()
        except Exception as exc:
            QMessageBox.warning(self, "Journal Rebuild Failed", str(exc))
            return
        self._set_status(f"Rebuilt {count} trade(s) from stored executions.")
        self._populate_filter_options()
        self._reload()

    def export_csv(self) -> None:
        try:
            path = journal_feed.export_trades_csv()
        except Exception as exc:
            QMessageBox.warning(self, "Journal Export Failed", str(exc))
            return
        self._set_status(f"Exported trades to {path}")

    def shutdown(self) -> None:
        self.import_service.shutdown()

    def _toggle_broker_sync(self, visible: bool) -> None:
        self.broker_sync_body.setVisible(visible)
        self.broker_sync_toggle.setText("Broker sync v" if visible else "Broker sync >")
        if visible:
            self._refresh_questrade_status()

    def _save_questrade_refresh_token(self) -> None:
        save_local_setting(QUESTRADE_REFRESH_TOKEN_SETTING, self.questrade_refresh_input.text().strip())
        self._refresh_questrade_status()
        self._set_status("Saved Questrade refresh token.")

    def _clear_questrade_refresh_token(self) -> None:
        self.questrade_refresh_input.clear()
        save_local_setting(QUESTRADE_REFRESH_TOKEN_SETTING, "")
        self._refresh_questrade_status()
        self._set_status("Cleared Questrade refresh token.")

    def _pull_new_questrade_trades(self) -> None:
        if self.import_service.running:
            self._set_status("Questrade import is already running.")
            return
        self._save_questrade_refresh_token()
        days = self.questrade_days_input.value()
        if not self.import_service.pull_recent_questrade(days):
            self._set_status("Questrade import is already running.")

    def _refresh_questrade_status(self) -> None:
        try:
            lines = QuestradeImporter().status_lines()
        except Exception as exc:  # pragma: no cover - defensive
            lines = [f"Questrade status unavailable: {exc}"]
        self.questrade_status_label.setText("\n".join(lines))

    def _set_import_controls_enabled(self, enabled: bool) -> None:
        self.questrade_refresh_input.setEnabled(enabled)
        self.questrade_save_button.setEnabled(enabled)
        self.questrade_clear_button.setEnabled(enabled)
        self.questrade_days_input.setEnabled(enabled)
        self.questrade_pull_button.setEnabled(enabled)

    def _on_import_started(self, days: int) -> None:
        self._set_import_controls_enabled(False)
        self.questrade_pull_button.setText("Pulling...")
        self._set_status(f"Pulling Questrade trades for the last {days} day(s)...")

    def _on_import_finished(self, _summaries: list, message: str) -> None:
        self._set_import_controls_enabled(True)
        self.questrade_pull_button.setText("Pull New Trades")
        self._refresh_questrade_status()
        self._populate_filter_options()
        self._reload()
        self._set_status(message)

    def _on_import_failed(self, message: str) -> None:
        self._set_import_controls_enabled(True)
        self.questrade_pull_button.setText("Pull New Trades")
        self._refresh_questrade_status()
        QMessageBox.warning(self, "Questrade Import Failed", message)
        self._set_status("Questrade import failed.")

    def _apply_filters(self) -> None:
        self.proxy.set_filters(
            status=self.status_input.currentText(),
            direction=self.direction_input.currentText(),
            search_text=self.search_input.text(),
        )

    def _update_kpis(self, trades: list[JournalTrade]) -> None:
        summary = journal_feed.analytics_summary(trades).get("overall", {}) if trades else {}
        self.trades_tile.set_value(str(summary.get("trades", 0)))
        win_rate = summary.get("win_rate")
        self.winrate_tile.set_value("-" if win_rate is None else f"{win_rate * 100:.0f}%")
        net = summary.get("net_pnl")
        self.pnl_tile.set_value("-" if not trades else f"{net:,.0f}")
        profit_factor = summary.get("profit_factor")
        self.pf_tile.set_value("-" if profit_factor is None else f"{profit_factor:.2f}")

    def _on_selection_changed(self, selected: QItemSelection, _deselected: QItemSelection) -> None:
        indexes = selected.indexes()
        if not indexes:
            return
        row = self.model.row_at(self.proxy.mapToSource(indexes[0]).row())
        if row is not None:
            self.detail.set_trade(row)

    def _on_annotation_saved(
        self,
        trade_id: str,
        setup_tags: str,
        notes: str,
        review_outcome: str,
        decision_reason: str,
    ) -> None:
        try:
            journal_feed.save_annotation(trade_id, setup_tags=setup_tags, notes=notes)
            if review_outcome != "Not reviewed" or decision_reason.strip():
                journal_feed.record_trade_review(
                    trade_id,
                    review_outcome=review_outcome,
                    decision_reason=decision_reason,
                    setup_tags=setup_tags,
                    notes=notes,
                )
        except Exception as exc:
            QMessageBox.warning(self, "Save Failed", str(exc))
            return
        self._set_status("Saved trade annotation.")
        self._reload_keeping_selection(trade_id)

    def _reload_keeping_selection(self, trade_id: str) -> None:
        self._reload()
        for proxy_row in range(self.proxy.rowCount()):
            source_index = self.proxy.mapToSource(self.proxy.index(proxy_row, 0))
            row = self.model.row_at(source_index.row())
            if row is not None and row.trade_id == trade_id:
                self.table.selectRow(proxy_row)
                break

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.statusChanged.emit(f"Journal: {message}")


class JournalDetailPanel(QFrame):
    annotationSaved = Signal(str, str, str, str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("InfoDrawer")
        self.setMinimumWidth(320)
        self._trade: JournalTrade | None = None

        self.analytics_view = QTextEdit()
        self.analytics_view.setReadOnly(True)
        self.analytics_view.setPlaceholderText("Portfolio analytics will appear here.")

        self.detail_view = QWidget()
        self._build_detail_view()

        self.stack = QStackedWidget()
        self.stack.addWidget(self.analytics_view)
        self.stack.addWidget(self.detail_view)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(SectionHeader("Detail", "Trade context and annotations."))
        layout.addWidget(self.stack, 1)

    def _build_detail_view(self) -> None:
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        back_button = QPushButton("Back to Analytics")
        back_button.clicked.connect(lambda: self.stack.setCurrentWidget(self.analytics_view))

        self.tags_input = QLineEdit()
        self.tags_input.setPlaceholderText("Setup tags (semicolon separated)")
        self.review_input = QComboBox()
        self.review_input.addItems(
            [
                "Not reviewed",
                "Followed plan",
                "Good trade / bad outcome",
                "Late or chased entry",
                "Poor stop discipline",
                "Poor exit discipline",
                "Oversized",
                "Setup was invalid",
                "Other lesson",
            ]
        )
        self.decision_reason_input = QLineEdit()
        self.decision_reason_input.setPlaceholderText("Why did you take this trade?")
        self.notes_input = QPlainTextEdit()
        self.notes_input.setPlaceholderText("Notes / review")
        save_button = QPushButton("Save Annotation")
        save_button.setObjectName("PrimaryButton")
        save_button.clicked.connect(self._save)

        layout = QVBoxLayout(self.detail_view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(back_button)
        layout.addWidget(self.summary_label)
        tags_label = QLabel("Setup Tags")
        tags_label.setObjectName("MutedLabel")
        layout.addWidget(tags_label)
        layout.addWidget(self.tags_input)
        review_label = QLabel("Structured Review")
        review_label.setObjectName("MutedLabel")
        layout.addWidget(review_label)
        layout.addWidget(self.review_input)
        layout.addWidget(self.decision_reason_input)
        notes_label = QLabel("Notes")
        notes_label.setObjectName("MutedLabel")
        layout.addWidget(notes_label)
        layout.addWidget(self.notes_input, 1)
        layout.addWidget(save_button)

    def show_analytics(self, text: str) -> None:
        self.analytics_view.setPlainText(text)
        if self._trade is None:
            self.stack.setCurrentWidget(self.analytics_view)

    def set_trade(self, trade: JournalTrade) -> None:
        self._trade = trade
        self.summary_label.setText(_trade_summary(trade))
        self.tags_input.setText(trade.tags)
        self.notes_input.setPlainText(trade.notes)
        review = journal_feed.latest_trade_review(trade.trade_id) or {}
        payload = review.get("payload") if isinstance(review.get("payload"), dict) else {}
        outcome = str(payload.get("review_outcome") or "Not reviewed")
        index = self.review_input.findText(outcome)
        self.review_input.setCurrentIndex(index if index >= 0 else 0)
        self.decision_reason_input.setText(str(review.get("reason") or ""))
        self.stack.setCurrentWidget(self.detail_view)

    def _save(self) -> None:
        if self._trade is None:
            return
        self.annotationSaved.emit(
            self._trade.trade_id,
            self.tags_input.text(),
            self.notes_input.toPlainText(),
            self.review_input.currentText(),
            self.decision_reason_input.text(),
        )


def _trade_summary(trade: JournalTrade) -> str:
    lines = [
        f"<b>{trade.symbol}</b> &nbsp; {trade.direction} &nbsp; {trade.status.title()}",
        f"Date: {trade.trade_date}",
        f"Quantity: {_num(trade.quantity)}",
        f"Entry: {_num(trade.entry_price)} &nbsp; Exit: {_num(trade.exit_price)}",
        f"Net P&L: {_num(trade.net_pnl)} &nbsp; Costs: {_num(trade.fees)}",
        f"Account: {trade.broker} {trade.account}".strip(),
    ]
    regime = " / ".join(
        str(trade.raw.get(key))
        for key in ("mid_term_regime", "short_term_regime", "intraday_regime")
        if trade.raw.get(key)
    )
    if regime:
        lines.append(f"Regime: {regime}")
    return "<br>".join(line for line in lines if line.strip())


def _num(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)
