"""Calendar: a month grid and a year heatmap of daily P&L (§9 step 12).

``calendar_pnl_by_day`` has existed in ``journal_analytics`` since long before
R7 and nothing in the Qt desk ever called it. This is that function finally
getting a surface, plus the click-through that makes it useful: a day here
filters the Trades tab, so "what happened on the 12th" is two clicks rather than
a search.
"""

from __future__ import annotations

import calendar as calendar_module
from datetime import date

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.services import journal_feed

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


class CalendarTab(QFrame):
    daySelected = Signal(str)
    statusChanged = Signal(str)

    def __init__(self, header, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._header = header
        self._by_day: dict[str, float] = {}
        today = date.today()

        self.year_input = QComboBox()
        self.year_input.addItems([str(year) for year in range(today.year - 4, today.year + 1)])
        self.year_input.setCurrentText(str(today.year))
        self.year_input.currentTextChanged.connect(self._render)
        self.month_input = QComboBox()
        self.month_input.addItems([calendar_module.month_name[m] for m in range(1, 13)])
        self.month_input.setCurrentIndex(today.month - 1)
        self.month_input.currentIndexChanged.connect(self._render)

        self.grid = QTableWidget(6, 7)
        self.grid.setHorizontalHeaderLabels(list(WEEKDAYS))
        self.grid.setEditTriggers(QTableWidget.NoEditTriggers)
        self.grid.verticalHeader().setVisible(False)
        self.grid.cellClicked.connect(self._on_cell_clicked)

        self.heatmap = QTableWidget(12, 31)
        self.heatmap.setVerticalHeaderLabels([calendar_module.month_abbr[m] for m in range(1, 13)])
        self.heatmap.setHorizontalHeaderLabels([str(day) for day in range(1, 32)])
        self.heatmap.setEditTriggers(QTableWidget.NoEditTriggers)
        self.heatmap.cellClicked.connect(self._on_heat_clicked)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Month"))
        controls.addWidget(self.month_input)
        controls.addWidget(self.year_input)
        controls.addStretch(1)
        self.summary = QLabel("")
        controls.addWidget(self.summary)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self.grid, 3)
        layout.addWidget(QLabel("Year"))
        layout.addWidget(self.heatmap, 2)

    def reload(self) -> None:
        try:
            self._by_day = journal_feed.calendar_pnl_by_day(**self._header.query())
        except Exception as exc:  # noqa: BLE001
            self._by_day = {}
            self.statusChanged.emit(f"calendar unavailable: {exc}")
        self._render()

    def _render(self, *_args) -> None:
        year = int(self.year_input.currentText() or date.today().year)
        month = self.month_input.currentIndex() + 1
        self.grid.clearContents()
        weeks = calendar_module.Calendar(firstweekday=0).monthdayscalendar(year, month)
        self.grid.setRowCount(max(len(weeks), 1))
        month_total = 0.0
        for row, week in enumerate(weeks):
            for column, day in enumerate(week):
                if not day:
                    continue
                key = date(year, month, day).isoformat()
                value = self._by_day.get(key)
                text = f"{day}\n{value:,.0f}" if value is not None else str(day)
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, key)
                if value is not None:
                    month_total += value
                    item.setForeground(Qt.darkGreen if value > 0 else Qt.red)
                self.grid.setItem(row, column, item)

        self.heatmap.clearContents()
        for key, value in self._by_day.items():
            try:
                when = date.fromisoformat(key)
            except ValueError:
                continue
            if when.year != year:
                continue
            item = QTableWidgetItem(f"{value:,.0f}")
            item.setData(Qt.UserRole, key)
            item.setForeground(Qt.darkGreen if value > 0 else Qt.red)
            self.heatmap.setItem(when.month - 1, when.day - 1, item)

        days = sum(1 for key in self._by_day if key.startswith(f"{year:04d}-{month:02d}"))
        self.summary.setText(f"{days} trading day(s), {month_total:,.2f} this month")

    def _on_cell_clicked(self, row: int, column: int) -> None:
        item = self.grid.item(row, column)
        if item is not None and item.data(Qt.UserRole):
            self.daySelected.emit(str(item.data(Qt.UserRole)))

    def _on_heat_clicked(self, row: int, column: int) -> None:
        item = self.heatmap.item(row, column)
        if item is not None and item.data(Qt.UserRole):
            self.daySelected.emit(str(item.data(Qt.UserRole)))
