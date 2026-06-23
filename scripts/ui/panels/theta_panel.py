from __future__ import annotations

from PySide6.QtCore import QFileSystemWatcher, QItemSelection, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)

from ui.models.theta import ThetaRow
from ui.models.theta_table_model import ROW_ROLE, ThetaFilterProxyModel, ThetaTableModel
from ui.services.theta_feed import THETA_REPORT_FILE, format_theta_symbols, load_theta_report_text, load_theta_rows
from ui.widgets.data_table import DataTable
from ui.widgets.empty_state import EmptyState
from ui.widgets.section_header import SectionHeader


class ThetaPanel(QFrame):
    statusChanged = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.model = ThetaTableModel()
        self.proxy = ThetaFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)

        self.table = DataTable()
        self.table.setModel(self.proxy)
        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)

        self.raw_details = QTextEdit()
        self.raw_details.setReadOnly(True)
        self.raw_details.setPlaceholderText("Select a theta row to see parsed details, or refresh after a scan.")

        self.min_score = QSpinBox()
        self.min_score.setRange(0, 200)
        self.min_score.setSpecialValueText("Any score")
        self.min_score.valueChanged.connect(self._apply_filters)
        self.min_supports = QSpinBox()
        self.min_supports.setRange(0, 20)
        self.min_supports.setSpecialValueText("Any supports")
        self.min_supports.valueChanged.connect(self._apply_filters)
        self.max_dte = QSpinBox()
        self.max_dte.setRange(0, 365)
        self.max_dte.setSpecialValueText("Any DTE")
        self.max_dte.valueChanged.connect(self._apply_filters)
        self.play_type = QComboBox()
        self.play_type.addItem("All Plays", "ALL")
        self.play_type.addItem("Sold Puts", "sold_put")
        self.play_type.addItem("PCS", "pcs")
        self.play_type.currentIndexChanged.connect(self._apply_filters)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Symbol")
        self.search.textChanged.connect(self._apply_filters)

        self.status_label = QLabel("")
        self.status_label.setObjectName("MutedLabel")
        self.empty_state = EmptyState("No theta plays yet", "Run a Master AVWAP scan, then refresh this tab.")

        self._build_layout()
        self._configure_watcher()
        self.refresh()

    def _build_layout(self) -> None:
        header = SectionHeader(
            "Theta Plays",
            "Sold put and PCS candidates from the Master AVWAP theta report.",
        )
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh)
        copy_button = QPushButton("Copy Symbols")
        copy_button.clicked.connect(self.copy_symbols)
        raw_button = QPushButton("Copy Raw Report")
        raw_button.clicked.connect(self.copy_raw_report)
        header.add_action(refresh_button)
        header.add_action(copy_button)
        header.add_action(raw_button)

        filters = QHBoxLayout()
        filters.setContentsMargins(0, 0, 0, 0)
        filters.setSpacing(8)
        filters.addWidget(self.search)
        filters.addWidget(self.play_type)
        filters.addWidget(QLabel("Min score"))
        filters.addWidget(self.min_score)
        filters.addWidget(QLabel("Min supports"))
        filters.addWidget(self.min_supports)
        filters.addWidget(QLabel("Max DTE"))
        filters.addWidget(self.max_dte)
        filters.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(header)
        layout.addLayout(filters)
        layout.addWidget(self.status_label)
        layout.addWidget(self.table, 3)
        layout.addWidget(self.raw_details, 1)

    def _configure_watcher(self) -> None:
        self.watcher = QFileSystemWatcher(self)
        if THETA_REPORT_FILE.exists():
            self.watcher.addPath(str(THETA_REPORT_FILE))
        self.watcher.fileChanged.connect(lambda _path: self.refresh())

    def refresh(self) -> None:
        rows = load_theta_rows()
        self.model.set_rows(rows)
        self._apply_filters()
        self.table.setVisible(bool(rows))
        if rows:
            self.table.sortByColumn(2, Qt.SortOrder.DescendingOrder)
            self.table.fit_columns()
        self.status_label.setText(_status_text(rows))
        self.statusChanged.emit(self.status_label.text())
        if THETA_REPORT_FILE.exists() and str(THETA_REPORT_FILE) not in self.watcher.files():
            self.watcher.addPath(str(THETA_REPORT_FILE))

    def copy_symbols(self) -> None:
        rows = self.filtered_rows()
        text = format_theta_symbols(rows)
        QApplication.clipboard().setText(text)
        self.status_label.setText(f"Copied {len([item for item in text.split(',') if item.strip()])} theta symbol(s).")
        self.statusChanged.emit(self.status_label.text())

    def copy_raw_report(self) -> None:
        QApplication.clipboard().setText(load_theta_report_text())
        self.status_label.setText("Copied raw theta report.")
        self.statusChanged.emit(self.status_label.text())

    def filtered_rows(self) -> list[ThetaRow]:
        rows: list[ThetaRow] = []
        for proxy_row in range(self.proxy.rowCount()):
            source_index = self.proxy.mapToSource(self.proxy.index(proxy_row, 0))
            row = self.model.data(source_index, ROW_ROLE)
            if isinstance(row, ThetaRow):
                rows.append(row)
        return rows

    def _apply_filters(self) -> None:
        self.proxy.set_filters(
            min_score=self.min_score.value() or None,
            min_supports=self.min_supports.value() or None,
            max_dte=self.max_dte.value() or None,
            play_type=str(self.play_type.currentData() or "ALL"),
            search_text=self.search.text(),
        )

    def _on_selection_changed(self, selected: QItemSelection, _deselected: QItemSelection) -> None:
        indexes = selected.indexes()
        if not indexes:
            return
        source_index = self.proxy.mapToSource(indexes[0])
        row = self.model.row_at(source_index.row())
        if row is None:
            return
        self.raw_details.setPlainText(_format_row_details(row))


def _status_text(rows: list[ThetaRow]) -> str:
    sold_puts = sum(1 for row in rows if row.play_type != "pcs")
    pcs = sum(1 for row in rows if row.play_type == "pcs")
    return f"{THETA_REPORT_FILE.name} | {len(rows)} candidate(s) | Sold puts: {sold_puts} | PCS: {pcs}"


def _format_row_details(row: ThetaRow) -> str:
    lines = [
        f"{row.symbol} - {row.play_label}",
        f"Score: {row.score}",
        f"Supports: {row.support_count}",
        f"Close: {row.close}",
        f"Strike: {row.recommended_strike}",
        f"Credit: {row.recommended_credit}",
        f"Expiration: {row.recommended_expiration}",
        f"Earnings DTE: {row.next_earnings_days if row.next_earnings_days is not None else row.next_earnings_label}",
        f"Strike band: {row.primary_strike_band}",
        "",
        "Raw parsed fields:",
    ]
    for key, value in sorted(row.raw.items()):
        lines.append(f"{key}: {value}")
    return "\n".join(lines)
