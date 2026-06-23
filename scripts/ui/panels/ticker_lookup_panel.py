from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
)

from ui.services.ticker_lookup_feed import (
    TickerLookupService,
    format_earnings_rows,
    format_headlines,
    lookup_summary,
)
from ui.widgets.section_header import SectionHeader


_EARNINGS_COLUMNS = ("Scope", "Date", "Ticker", "Company", "Time", "Risk")


class TickerLookupPanel(QFrame):
    statusChanged = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.service = TickerLookupService(self)
        self.service.started.connect(self._on_started)
        self.service.finished.connect(self._on_finished)
        self.service.failed.connect(self._on_failed)

        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Ticker (e.g. NVDA)")
        self.ticker_input.setMaximumWidth(160)
        self.ticker_input.returnPressed.connect(self.run_lookup)

        self.days_input = QSpinBox()
        self.days_input.setRange(7, 180)
        self.days_input.setValue(10)
        self.days_input.setPrefix("Days ")

        self.lookup_button = QPushButton("Lookup")
        self.lookup_button.setObjectName("PrimaryButton")
        self.lookup_button.clicked.connect(self.run_lookup)

        self.summary_label = QLabel("Enter a ticker and run lookup.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setObjectName("MutedLabel")

        self.overview_view = _mono_text("Ticker Lookup is ready. Enter a symbol such as NVDA, AAPL, or AMD.")
        self.earnings_table = self._build_earnings_table()
        self.news_view = _mono_text("")
        self.raw_view = _mono_text("")

        self.tabs = QTabWidget()
        self.tabs.addTab(self.overview_view, "Overview")
        self.tabs.addTab(self.earnings_table, "Earnings / Events")
        self.tabs.addTab(self.news_view, "News")
        self.tabs.addTab(self.raw_view, "Report")

        self.status_label = QLabel("")
        self.status_label.setObjectName("MutedLabel")

        self._build_layout()

    def _build_layout(self) -> None:
        header = SectionHeader(
            "Ticker Lookup",
            "Single-symbol earnings, peer events, filings, and industry news (secondary research).",
        )

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)
        toolbar.addWidget(QLabel("Ticker"))
        toolbar.addWidget(self.ticker_input)
        toolbar.addWidget(self.days_input)
        toolbar.addWidget(self.lookup_button)
        toolbar.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(header)
        layout.addLayout(toolbar)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.status_label)

    def _build_earnings_table(self) -> QTableWidget:
        table = QTableWidget(0, len(_EARNINGS_COLUMNS))
        table.setHorizontalHeaderLabels(_EARNINGS_COLUMNS)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def run_lookup(self) -> None:
        symbol = self.ticker_input.text().strip().upper()
        if not symbol:
            self._set_status("Enter a ticker symbol before running lookup.")
            return
        self.service.lookup(symbol, self.days_input.value())

    def _on_started(self, symbol: str) -> None:
        self.lookup_button.setEnabled(False)
        self.summary_label.setText(f"Looking up {symbol}…")
        self._set_status(f"Running lookup for {symbol}…")

    def _on_finished(self, payload: dict[str, Any]) -> None:
        self.lookup_button.setEnabled(True)
        self.summary_label.setText(lookup_summary(payload))
        self.overview_view.setPlainText(str(payload.get("markdown") or "No report text returned."))
        self.raw_view.setPlainText(str(payload.get("markdown") or ""))
        self.news_view.setPlainText(format_headlines(payload))
        self._fill_earnings(format_earnings_rows(payload))
        ticker = str(payload.get("ticker") or "").upper()
        self._set_status(f"Lookup complete for {ticker}.")

    def _on_failed(self, message: str) -> None:
        self.lookup_button.setEnabled(True)
        summary = message.splitlines()[0] if message else "Lookup failed."
        self.summary_label.setText(f"Lookup failed: {summary}")
        self._set_status(f"Lookup failed: {summary}")

    def _fill_earnings(self, rows: list[dict[str, str]]) -> None:
        self.earnings_table.setRowCount(len(rows))
        keys = ("scope", "date", "ticker", "company", "time", "importance")
        for row_index, row in enumerate(rows):
            for col_index, key in enumerate(keys):
                self.earnings_table.setItem(row_index, col_index, QTableWidgetItem(row.get(key, "")))
        self.earnings_table.resizeColumnsToContents()

    def shutdown(self) -> None:
        self.service.shutdown()

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.statusChanged.emit(f"Ticker Lookup: {message}")


def _mono_text(placeholder: str) -> QTextEdit:
    view = QTextEdit()
    view.setReadOnly(True)
    view.setPlaceholderText(placeholder)
    if placeholder:
        view.setPlainText(placeholder)
    font = QFont("Cascadia Mono")
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPointSizeF(9.5)
    view.setFont(font)
    view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
    return view
