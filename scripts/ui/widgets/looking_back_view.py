"""Research > Results "Looking back" (WISHLIST P2-9): one population's pick curve
and the hold-out columns.

Renders the service's `looking_back` reading and computes nothing. Swing picks
and M5 alerts are two separate curves; the page shows the one for the chosen
horizon, and under it each grade and working-lately cell over the last window
beside the window before it.
"""

from __future__ import annotations

from typing import Any, Mapping

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
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

#: Which curve and which hold-out block each Results horizon shows.
HORIZON_POPULATION = {"swing": looking_back.SWING, "day": looking_back.M5}
HORIZON_HOLDOUT = {"swing": "swing", "day": "day"}
POPULATION_TITLES = dict(looking_back.POPULATIONS)

POINT_COLUMNS = ("Session", "Day R", "Day n", "Cumulative R", "Cumulative n")
HOLDOUT_COLUMNS = ("Measure", "Cell", "Last window", "Prior window")

KIND_LABELS = {
    "swing_trade_r": "Win rate (trade R)",
    "swing_favorable": "Favorable move",
    "daytrade_held_run": "Held x ran",
}

NO_READING = "no looking-back reading yet - the working-lately build writes one"
NO_HOLDOUT = "no hold-out reading yet"

#: How many dates the curve's bottom axis names.
AXIS_TICKS = 6


class LookingBackView(QFrame):
    """The pick equity curve for one population, with n, and its hold-out table."""

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

        self.holdout_label = QLabel(NO_HOLDOUT)
        self.holdout_label.setObjectName("MutedLabel")
        self.holdout_label.setWordWrap(True)
        self.holdout_table = QTableWidget(0, len(HOLDOUT_COLUMNS))
        self.holdout_table.setHorizontalHeaderLabels(list(HOLDOUT_COLUMNS))
        self.holdout_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.holdout_table.setMinimumHeight(theme.px(140))
        self.holdout_table.setMaximumHeight(theme.px(260))

        # Curve on the left, hold-out on the right: the desk is wide and the
        # shortlist above keeps its height.
        curve_column = QVBoxLayout()
        curve_column.setSpacing(4)
        curve_column.addWidget(self.title_label)
        curve_column.addWidget(self.line_label)
        if self.plot is not None:
            curve_column.addWidget(self.plot)
        curve_column.addWidget(self.points_table)
        holdout_column = QVBoxLayout()
        holdout_column.setSpacing(4)
        holdout_column.addWidget(self.holdout_label)
        holdout_column.addWidget(self.holdout_table)
        columns = QHBoxLayout()
        columns.setSpacing(12)
        columns.addLayout(curve_column, 1)
        columns.addLayout(holdout_column, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        layout.addWidget(self.header)
        layout.addLayout(columns)

    # -- reading -----------------------------------------------------------

    def population(self) -> str:
        return self._population

    def curve_points(self) -> list[dict[str, Any]]:
        """The points on screen, for parity checks."""
        return list(self._curve.get("points") or ())

    def holdout_rows(self) -> list[tuple[str, ...]]:
        """The hold-out table's text, row by row, for parity checks."""
        out = []
        for row in range(self.holdout_table.rowCount()):
            out.append(
                tuple(
                    (self.holdout_table.item(row, column) or QTableWidgetItem("")).text()
                    for column in range(self.holdout_table.columnCount())
                )
            )
        return out

    def set_reading(self, payload: Mapping[str, Any] | None, horizon: str) -> None:
        """Draw the curve and hold-out for `horizon` ("swing" or "day")."""
        payload = payload if isinstance(payload, Mapping) else {}
        population = HORIZON_POPULATION.get(str(horizon), looking_back.SWING)
        self._population = population
        self.title_label.setText(POPULATION_TITLES.get(population, population))
        curves = payload.get("curves")
        curve = curves.get(population) if isinstance(curves, Mapping) else None
        self._curve = dict(curve) if isinstance(curve, Mapping) else {}
        if not self._curve:
            self.line_label.setText(NO_READING)
        else:
            as_of = str(payload.get("as_of") or "")
            suffix = f" · as of {as_of}" if as_of else ""
            self.line_label.setText(looking_back.curve_line(self._curve) + suffix)
        self._draw()
        holdout = payload.get("holdout")
        block = holdout.get(HORIZON_HOLDOUT.get(str(horizon), "swing")) if isinstance(holdout, Mapping) else None
        self._fill_holdout(block if isinstance(block, Mapping) else {})

    # -- rendering ---------------------------------------------------------

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

    def _fill_holdout(self, block: Mapping[str, Any]) -> None:
        rows: list[tuple[str, str, str, str]] = []
        for row in block.get("grades") or ():
            rows.append(
                (
                    "Grade",
                    str(row.get("key") or "").replace("|", " "),
                    str(row.get("recent_text") or ""),
                    str(row.get("prior_text") or ""),
                )
            )
        for row in block.get("working_lately") or ():
            kind = str(row.get("kind") or "")
            rows.append(
                (
                    KIND_LABELS.get(kind, kind),
                    _cell_name(row),
                    str(row.get("recent_text") or ""),
                    str(row.get("prior_text") or ""),
                )
            )
        self.holdout_label.setText(_windows_text(block) if rows else NO_HOLDOUT)
        self.holdout_table.setRowCount(len(rows))
        for index, values in enumerate(rows):
            for column, text in enumerate(values):
                self.holdout_table.setItem(index, column, QTableWidgetItem(text))
        self.holdout_table.resizeColumnsToContents()


def _cell_name(row: Mapping[str, Any]) -> str:
    """SIDE bucket family, with a non-live namespace named: one population per row."""
    parts = [str(row.get(key) or "") for key in ("side", "bucket", "family")]
    name = " ".join(part for part in parts if part)
    namespace = str(row.get("namespace") or "live")
    return name if namespace == "live" else f"{name} [{namespace}]"


def _span(windows: Any) -> tuple[str, str]:
    windows = windows if isinstance(windows, Mapping) else {}
    recent = list(windows.get("recent") or ("?", "?"))
    prior = list(windows.get("prior") or ("?", "?"))
    return f"{recent[0]} to {recent[1]}", f"{prior[0]} to {prior[1]}"


def _windows_text(block: Mapping[str, Any]) -> str:
    """Which dates each column covers. Under the floor, a cell says n<30."""
    recent, prior = _span(block.get("windows"))
    text = (
        f"Hold-out: the same statistic over the last window ({recent}) and the "
        f"window before it ({prior}). A cell under its floor says n<30."
    )
    if block.get("favorable_windows"):
        fav_recent, fav_prior = _span(block.get("favorable_windows"))
        text += f" Favorable move: {fav_recent} vs {fav_prior}."
    return text
