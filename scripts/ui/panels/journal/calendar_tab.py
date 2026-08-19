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

try:  # pragma: no cover - the desk has pyqtgraph; a headless box may not.
    import numpy as np
    import pyqtgraph as pg

    PYQTGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover
    np = None
    pg = None
    PYQTGRAPH_AVAILABLE = False

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def year_heatmap_matrix(by_day: dict, year: int) -> tuple[list[list[float | None]], float]:
    """A 12x31 grid of daily P&L for ``year``, and the scale to colour it by.

    ``None`` marks a day with no trading, which must NOT be drawn as zero: a
    flat day and a day the trader did not trade are different facts, and a
    heatmap that paints them the same colour invents a hundred break-even
    sessions a year.

    The returned scale is the largest ABSOLUTE value, so the colour map can be
    centred on zero. Scaling a diverging map to the raw min/max makes a good
    year look mediocre and a bad one look catastrophic, purely from where the
    extremes happen to fall.
    """
    matrix: list[list[float | None]] = [[None] * 31 for _ in range(12)]
    scale = 0.0
    for key, value in (by_day or {}).items():
        try:
            when = date.fromisoformat(str(key))
            amount = float(value)
        except (TypeError, ValueError):
            continue
        if when.year != int(year):
            continue
        matrix[when.month - 1][when.day - 1] = amount
        scale = max(scale, abs(amount))
    return matrix, scale


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

        # R7's deferred pyqtgraph year heatmap, built 2026-08-18. The table
        # below keeps every number; this is the shape of the year, which is the
        # thing a table cannot show. Days with no trading stay transparent -
        # they are not break-even days.
        self.heat_plot = pg.PlotWidget(title="Year") if PYQTGRAPH_AVAILABLE else None
        self.heat_image = pg.ImageItem() if PYQTGRAPH_AVAILABLE else None
        if PYQTGRAPH_AVAILABLE:
            self.heat_plot.addItem(self.heat_image)
            self.heat_plot.setMouseEnabled(x=False, y=False)
            self.heat_plot.getAxis("left").setTicks(
                [[(index + 0.5, calendar_module.month_abbr[index + 1]) for index in range(12)]]
            )
            self.heat_plot.getAxis("bottom").setTicks(
                [[(day - 0.5, str(day)) for day in range(1, 32, 2)]]
            )
            self.heat_plot.invertY(True)
            self.heat_image.mouseClickEvent = self._on_heat_image_clicked
        self.heat_note = QLabel("")
        self.heat_note.setWordWrap(True)

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
        if self.heat_plot is not None:
            layout.addWidget(self.heat_plot, 2)
        layout.addWidget(self.heat_note)
        layout.addWidget(self.heatmap, 2)

    def reload(self) -> None:
        try:
            self._by_day = journal_feed.calendar_pnl_by_day(
                currency_mode=self._header.currency_mode, **self._header.query()
            )
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

        self._render_heat_image(year)

        days = sum(1 for key in self._by_day if key.startswith(f"{year:04d}-{month:02d}"))
        self.summary.setText(f"{days} trading day(s), {month_total:,.2f} this month")

    def _render_heat_image(self, year: int) -> None:
        matrix, scale = year_heatmap_matrix(self._by_day, year)
        traded = sum(1 for row in matrix for value in row if value is not None)
        if not PYQTGRAPH_AVAILABLE:
            self.heat_note.setText(
                "pyqtgraph is not installed; the grid below carries the same numbers."
            )
            return
        if not traded:
            self.heat_image.clear()
            self.heat_note.setText(f"No trading days recorded in {year}.")
            return
        # NaN for "no trading", so the image leaves those cells transparent
        # instead of colouring them like a flat day.
        data = np.full((12, 31), np.nan, dtype=float)
        for month_index, row in enumerate(matrix):
            for day_index, value in enumerate(row):
                if value is not None:
                    data[month_index][day_index] = value
        # Diverging map centred on zero: red loss, white flat, green gain.
        colormap = pg.ColorMap(
            [0.0, 0.5, 1.0],
            [(178, 34, 34, 255), (245, 245, 245, 255), (34, 139, 34, 255)],
        )
        self.heat_image.setImage(data.T, autoLevels=False, levels=(-scale, scale))
        self.heat_image.setLookupTable(colormap.getLookupTable(0.0, 1.0, 256))
        self.heat_note.setText(
            f"{traded} trading day(s) in {year}; colour is centred on zero and scaled to "
            f"the largest single day ({scale:,.0f}). Days with no trading are blank, not "
            "break-even."
        )

    def _on_heat_image_clicked(self, event) -> None:  # pragma: no cover - GUI path
        try:
            position = event.pos()
            day = int(position.x()) + 1
            month = int(position.y()) + 1
            key = date(int(self.year_input.currentText()), month, day).isoformat()
        except (ValueError, AttributeError):
            return
        if key in self._by_day:
            self.daySelected.emit(key)

    def _on_cell_clicked(self, row: int, column: int) -> None:
        item = self.grid.item(row, column)
        if item is not None and item.data(Qt.UserRole):
            self.daySelected.emit(str(item.data(Qt.UserRole)))

    def _on_heat_clicked(self, row: int, column: int) -> None:
        item = self.heatmap.item(row, column)
        if item is not None and item.data(Qt.UserRole):
            self.daySelected.emit(str(item.data(Qt.UserRole)))
