"""Research > Results "Looking back" (WISHLIST P2-9): one population's pick curve.

Renders the service's `looking_back` reading and computes nothing. Swing picks
and M5 alerts are two separate curves; the page shows the one for the chosen
horizon.
"""

from __future__ import annotations

from typing import Any, Mapping

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

import looking_back
from ui import theme
from ui.widgets.section_header import SectionHeader

try:  # pragma: no cover - the desk has pyqtgraph; a headless box may not.
    import pyqtgraph as pg

    PYQTGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover
    pg = None
    PYQTGRAPH_AVAILABLE = False

#: Which curve each Results horizon shows.
HORIZON_POPULATION = {"swing": looking_back.SWING, "day": looking_back.M5}
POPULATION_TITLES = dict(looking_back.POPULATIONS)

POINT_COLUMNS = ("Session", "Day R", "Day n", "Cumulative R", "Cumulative n")

NO_READING = "no looking-back reading yet - the working-lately build writes one"

#: How many dates the curve's bottom axis names.
AXIS_TICKS = 6


class LookingBackView(QFrame):
    """The pick equity curve for one population, with n."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self._population = looking_back.SWING
        self._curve: dict[str, Any] = {}

        self.header = SectionHeader(
            "Looking back",
            "Cumulative R of every graded pick, by the session it was made, with n. "
            "Swing picks and M5 alerts are separate curves and are never pooled.",
        )
        self.title_label = QLabel("")
        self.title_label.setObjectName("SectionTitle")
        self.line_label = QLabel(NO_READING)
        self.line_label.setObjectName("MutedLabel")
        self.line_label.setWordWrap(True)
        self.line_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        if PYQTGRAPH_AVAILABLE:
            self.plot = pg.PlotWidget(background=theme.color("bg_panel"))
            self.plot.setMinimumHeight(theme.px(200))
            self.plot.setMaximumHeight(theme.px(260))
            self.plot.showGrid(x=False, y=True, alpha=0.2)
            self.plot.setLabel("left", "R")
        else:
            self.plot = None
        self.points_table = QTableWidget(0, len(POINT_COLUMNS))
        self.points_table.setHorizontalHeaderLabels(list(POINT_COLUMNS))
        self.points_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.points_table.setMaximumHeight(theme.px(220))
        # The table carries the same numbers; it is shown only without a plot.
        self.points_table.setVisible(self.plot is None)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        layout.addWidget(self.header)
        layout.addWidget(self.title_label)
        layout.addWidget(self.line_label)
        if self.plot is not None:
            layout.addWidget(self.plot)
        layout.addWidget(self.points_table)

    # -- rendering ---------------------------------------------------------

    def population(self) -> str:
        return self._population

    def curve_points(self) -> list[dict[str, Any]]:
        """The points on screen, for parity checks."""
        return list(self._curve.get("points") or ())

    def set_reading(self, payload: Mapping[str, Any] | None, horizon: str) -> None:
        """Draw the curve for `horizon` ("swing" or "day") from one reading."""
        population = HORIZON_POPULATION.get(str(horizon), looking_back.SWING)
        self._population = population
        self.title_label.setText(POPULATION_TITLES.get(population, population))
        curves = (payload or {}).get("curves") if isinstance(payload, Mapping) else None
        curve = (curves or {}).get(population) if isinstance(curves, Mapping) else None
        self._curve = dict(curve) if isinstance(curve, Mapping) else {}
        if not self._curve:
            self.line_label.setText(NO_READING)
        else:
            as_of = str((payload or {}).get("as_of") or "")
            suffix = f" · as of {as_of}" if as_of else ""
            self.line_label.setText(looking_back.curve_line(self._curve) + suffix)
        self._draw()

    def _draw(self) -> None:
        points = self.curve_points()
        if self.plot is not None:
            self.plot.clear()
            if points:
                xs = list(range(len(points)))
                self.plot.plot(
                    xs,
                    [float(point["cum_r"]) for point in points],
                    pen=pg.mkPen(theme.color("accent"), width=2),
                )
                self.plot.addLine(y=0, pen=pg.mkPen(theme.color("border"), width=1))
                step = max(1, len(points) // AXIS_TICKS)
                ticks = [(index, points[index]["session"]) for index in range(0, len(points), step)]
                self.plot.getAxis("bottom").setTicks([ticks])
        self.points_table.setRowCount(len(points))
        for row, point in enumerate(points):
            values = (
                str(point.get("session") or ""),
                f"{float(point.get('day_r') or 0):+.2f}",
                str(point.get("day_n") or 0),
                f"{float(point.get('cum_r') or 0):+.2f}",
                str(point.get("cum_n") or 0),
            )
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if column:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.points_table.setItem(row, column, item)
