"""Inventory of every armed alert, with health and a cancel button.

Nothing in the application listed armed watches before this. Session watches
were visible only as lit buttons on whichever chart happened to be on screen,
and persistent D1 level alerts - which survive restarts and keep watching
symbols that are not being scanned - had no visual representation and no
cancel path at all. The only way to undo one was to hand-edit
d1_level_watches.json.
"""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from chart_watch import D1_LEVEL_KINDS, WATCH_KINDS
from ui import theme

COLUMNS = ("Symbol", "Kind", "Level", "Armed", "Age", "Health", "")

HEALTH_OK = "ok"
HEALTH_NO_BARS = "no M5 bars"
HEALTH_STALE = "stale"


def watch_health(kind: str, has_m5_bars: bool, armed_at: datetime, now: datetime) -> str:
    """Why an armed watch might never fire.

    Session watches evaluate against cached M5 bars and die at midnight, so a
    symbol with no cached bars cannot progress and a watch armed on an earlier
    day is already dead. Persistent level alerts have neither constraint - they
    also read the daily store - so they are always reported healthy.
    """
    if kind in D1_LEVEL_KINDS:
        return HEALTH_OK
    if armed_at.date() != now.date():
        return HEALTH_STALE
    if not has_m5_bars:
        return HEALTH_NO_BARS
    return HEALTH_OK


def format_age(armed_at: datetime, now: datetime) -> str:
    seconds = max(0, int((now - armed_at).total_seconds()))
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes = seconds // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


class ArmedWatchList(QFrame):
    """Table of armed session watches and persistent level alerts."""

    disarmWatchRequested = Signal(str, str)  # symbol, kind
    disarmLevelRequested = Signal(str, str, float)  # symbol, direction, level
    symbolActivated = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.cellClicked.connect(self._on_cell_clicked)

        self.empty_label = QLabel(
            "Nothing armed. Arm a watch or a price level from any chart and it "
            "shows up here until it fires."
        )
        self.empty_label.setObjectName("MutedLabel")
        self.empty_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self.empty_label)
        layout.addWidget(self.table, 1)
        self._rows: list[tuple] = []
        self.set_watches([], [], has_m5_bars=lambda _symbol: True)

    def set_watches(self, watches, levels, *, has_m5_bars, now: datetime | None = None) -> None:
        """Render armed session watches and persistent level alerts together."""
        moment = now or datetime.now()
        self._rows = []
        for watch in watches or []:
            self._rows.append(
                (
                    watch.symbol,
                    WATCH_KINDS.get(watch.kind, watch.kind),
                    f"{watch.baseline:.2f}" if watch.baseline is not None else "—",
                    watch.armed_at.strftime("%H:%M"),
                    format_age(watch.armed_at, moment),
                    watch_health(watch.kind, bool(has_m5_bars(watch.symbol)), watch.armed_at, moment),
                    ("watch", watch.symbol, watch.kind, 0.0),
                )
            )
        for watch in levels or []:
            kind = f"d1_level_{watch.direction}"
            self._rows.append(
                (
                    watch.symbol,
                    D1_LEVEL_KINDS.get(kind, kind),
                    f"{watch.level:.2f}",
                    watch.armed_at.strftime("%m/%d"),
                    format_age(watch.armed_at, moment),
                    watch_health(kind, True, watch.armed_at, moment),
                    ("level", watch.symbol, watch.direction, float(watch.level)),
                )
            )

        self.table.setRowCount(len(self._rows))
        for index, row in enumerate(self._rows):
            for column in range(len(COLUMNS) - 1):
                item = QTableWidgetItem(str(row[column]))
                if column == 5 and row[column] != HEALTH_OK:
                    item.setForeground(QColor(theme.color("caution")))
                    item.setToolTip(
                        "A session watch needs cached M5 bars to evaluate, and "
                        "never survives into the next session."
                    )
                self.table.setItem(index, column, item)
            disarm = QTableWidgetItem("✕")
            disarm.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            disarm.setToolTip("Disarm")
            self.table.setItem(index, len(COLUMNS) - 1, disarm)

        self.empty_label.setVisible(not self._rows)
        self.table.setVisible(bool(self._rows))

    def _on_cell_clicked(self, row: int, column: int) -> None:
        if not 0 <= row < len(self._rows):
            return
        handle = self._rows[row][-1]
        if column == len(COLUMNS) - 1:
            if handle[0] == "watch":
                self.disarmWatchRequested.emit(handle[1], handle[2])
            else:
                self.disarmLevelRequested.emit(handle[1], handle[2], handle[3])
            return
        if column == 0:
            self.symbolActivated.emit(handle[1])
