"""Fees: what trading cost, per account and per currency (§9 step 13).

Trade commissions and cash-side fees are shown **side by side and never added**.
The first is already inside each trade's net P&L; the second is not. A single
"total costs" column summing both would double-count the commissions, which is
the sort of number that looks authoritative and quietly overstates the year's
expenses on a tax return.
"""

from __future__ import annotations

import datetime as _dt

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from journal_tax_report import describe_tax_report
from ui.services import journal_feed

ACTIVITY_FILTERS = ("All", "FEE", "INTEREST", "DIVIDEND", "FX", "OTHER")


class FeesTab(QFrame):
    statusChanged = Signal(str)

    def __init__(self, header, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._header = header

        self.totals_table = QTableWidget(0, 7)
        self.totals_table.setHorizontalHeaderLabels(
            ["Broker", "Account", "Currency", "Commission", "Trade fees", "Cash fees", "Dividends"]
        )
        self.totals_table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.activity_filter = QComboBox()
        self.activity_filter.addItems(ACTIVITY_FILTERS)
        self.activity_filter.currentTextChanged.connect(self.reload)

        self.cash_table = QTableWidget(0, 6)
        self.cash_table.setHorizontalHeaderLabels(
            ["Date", "Broker", "Account", "Type", "Symbol", "Amount"]
        )
        self.cash_table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.export_button = QPushButton("Export fees CSV")
        self.export_button.clicked.connect(self._export)

        # The tax number, and the only one in the journal that is NOT
        # recomputed: it adds up what the broker itself said each fill did to
        # cash. Trader decision 2026-08-28.
        self.tax_year = QComboBox()
        self.tax_year.addItem("All years", None)
        current = _dt.date.today().year
        for offset in range(0, 5):
            year = current - offset
            self.tax_year.addItem(str(year), year)
        self.tax_button = QPushButton("Realised P&L for tax...")
        self.tax_button.setToolTip(
            "Realised profit and loss added up from the broker's own amounts, "
            "never recomputed. Writes a CSV listing every closed position and "
            "every position it could not count, with the reason."
        )
        self.tax_button.clicked.connect(self._build_tax_report)
        self.tax_output = QLabel("")
        self.tax_output.setWordWrap(True)
        self.tax_output.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.note = QLabel(
            "Commissions and trade fees are already inside each trade's net P&L. "
            "Cash fees and dividends are not - they are shown here and never added to the "
            "columns on their left."
        )
        self.note.setWordWrap(True)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Cash activity"))
        controls.addWidget(self.activity_filter)
        controls.addStretch(1)
        controls.addWidget(self.export_button)

        tax_row = QHBoxLayout()
        tax_row.addWidget(QLabel("Tax year"))
        tax_row.addWidget(self.tax_year)
        tax_row.addWidget(self.tax_button)
        tax_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Totals"))
        layout.addWidget(self.totals_table, 2)
        layout.addWidget(self.note)
        layout.addLayout(controls)
        layout.addWidget(self.cash_table, 3)
        layout.addLayout(tax_row)
        layout.addWidget(self.tax_output)

    def reload(self, *_args) -> None:
        query = self._header.query()
        try:
            totals = journal_feed.fee_totals(**query)
        except Exception as exc:  # noqa: BLE001
            totals = []
            self.statusChanged.emit(f"fees unavailable: {exc}")
        self.totals_table.setRowCount(len(totals))
        for row, record in enumerate(totals):
            values = [
                record.get("broker"),
                record.get("account"),
                record.get("currency"),
                f"{record.get('commission', 0.0):,.2f}",
                f"{record.get('fees', 0.0):,.2f}",
                f"{record.get('cash_fees', 0.0):,.2f}",
                f"{record.get('dividends', 0.0):,.2f}",
            ]
            for column, text in enumerate(values):
                self.totals_table.setItem(row, column, QTableWidgetItem(str(text or "")))

        activity = self.activity_filter.currentText()
        rows = journal_feed.cash_transactions(
            date_from=query.get("date_from"),
            date_to=query.get("date_to"),
            accounts_filter=query.get("accounts_filter"),
            activity_type="" if activity == "All" else activity,
        )
        self.cash_table.setRowCount(len(rows))
        for row, record in enumerate(rows):
            values = [
                record.get("txn_date"),
                record.get("broker"),
                record.get("account_number"),
                record.get("activity_type"),
                record.get("symbol"),
                f"{float(record.get('amount') or 0.0):,.2f}",
            ]
            for column, text in enumerate(values):
                self.cash_table.setItem(row, column, QTableWidgetItem(str(text or "")))

    def _export(self) -> None:
        try:
            path = journal_feed.export_fees_csv(**self._header.query())
        except Exception as exc:  # noqa: BLE001
            self.statusChanged.emit(f"fee export failed: {exc}")
            return
        self.statusChanged.emit(f"exported {path}")

    def _build_tax_report(self) -> None:
        query = self._header.query()
        try:
            report = journal_feed.tax_report(
                self.tax_year.currentData(), accounts_filter=query.get("accounts_filter")
            )
            path = journal_feed.export_tax_csv(report)
        except Exception as exc:  # noqa: BLE001
            self.tax_output.setText(f"tax report unavailable: {exc}")
            self.statusChanged.emit(f"tax report failed: {exc}")
            return
        self.tax_output.setText(
            describe_tax_report(report, report.get("cross_check")) + f"\nWrote {path}"
        )
        self.statusChanged.emit(f"tax report written to {path}")
