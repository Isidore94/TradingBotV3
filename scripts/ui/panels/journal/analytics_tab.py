"""Analytics: the equity curve, the per-group breakdowns, and walk-away (§9 step 12).

Two things here are corrections rather than features.

The **honest total**: when the selection mixes currencies and anything is
unconverted, this tab shows the reason instead of a number. Adding a USD win to
a CAD loss produces 60 of nothing, and quietly dropping the unconverted rows
produces a total that looks right and is not.

The **honest curve**: a trade that cannot be converted is skipped and counted,
never absorbed as a flat step. A curve that swallows a real position as zero is
a lie in the shape a chart makes easy to believe.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.services import journal_feed

try:  # pragma: no cover - the desk has pyqtgraph; a headless box may not.
    import pyqtgraph as pg

    PYQTGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover
    pg = None
    PYQTGRAPH_AVAILABLE = False


class _WalkawayWorker(QThread):
    """Walk-away replays daily bars, so it never runs on the GUI thread."""

    finished_with = Signal(dict)
    failed = Signal(str)

    def __init__(self, since, until, parent=None) -> None:
        super().__init__(parent)
        self._since = since
        self._until = until

    def run(self) -> None:  # pragma: no cover - exercised on the desk
        try:
            self.finished_with.emit(journal_feed.walkaway_summary(self._since, self._until))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class AnalyticsTab(QFrame):
    statusChanged = Signal(str)

    def __init__(self, header, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._header = header
        self._worker: _WalkawayWorker | None = None

        self.headline = QLabel("")
        self.headline.setWordWrap(True)
        self.currency_note = QLabel("")
        self.currency_note.setObjectName("CurrencyNote")
        self.currency_note.setWordWrap(True)

        self.curve = pg.PlotWidget(title="Cumulative P&L") if PYQTGRAPH_AVAILABLE else QLabel(
            "pyqtgraph is not installed; the table below carries the same numbers."
        )
        self.curve_table = QTableWidget(0, 2)
        self.curve_table.setHorizontalHeaderLabels(["Date", "Cumulative"])
        self.curve_table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.groups_table = QTableWidget(0, 6)
        self.groups_table.setHorizontalHeaderLabels(
            ["Group", "Bucket", "Trades", "Win rate", "Profit factor", "Net"]
        )
        self.groups_table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.walkaway_button = QPushButton("Run walk-away for this range")
        self.walkaway_button.clicked.connect(self._run_walkaway)
        self.walkaway_output = QLabel("Walk-away has not been run for this range yet.")
        self.walkaway_output.setWordWrap(True)

        self.export_button = QPushButton("Export trades CSV")
        self.export_button.clicked.connect(self._export)

        buttons = QHBoxLayout()
        buttons.addWidget(self.walkaway_button)
        buttons.addWidget(self.export_button)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.headline)
        layout.addWidget(self.currency_note)
        layout.addWidget(self.curve, 3)
        layout.addWidget(self.curve_table, 1)
        layout.addWidget(QLabel("By group"))
        layout.addWidget(self.groups_table, 2)
        layout.addLayout(buttons)
        layout.addWidget(self.walkaway_output)

    def reload(self) -> None:
        try:
            trades = journal_feed.load_trades(**self._header.query())
        except Exception as exc:  # noqa: BLE001
            self.statusChanged.emit(f"analytics unavailable: {exc}")
            return
        mode = self._header.currency_mode
        summary = journal_feed.analytics_summary(trades, mode)

        overall = summary.get("overall") or {}
        net = overall.get("net_pnl")
        win_rate = overall.get("win_rate")
        self.headline.setText(
            f"{overall.get('trades', 0)} trades, {overall.get('closed', 0)} closed"
            + (f", win rate {win_rate:.0%}" if win_rate is not None else "")
            + (f", net {net:,.2f}" if net is not None else ", net not shown")
        )
        # The refusal, rendered. `pnl_note` carries the reason the total is
        # missing, and a missing total with a reason beats a wrong one.
        note = str(summary.get("pnl_note") or "")
        unconvertible = journal_feed.unconvertible_count(trades, mode)
        if unconvertible:
            note = (note + " " if note else "") + (
                f"{unconvertible} closed trade(s) have no booked {mode} rate and are not in the curve."
            )
        self.currency_note.setText(note)
        self.currency_note.setVisible(bool(note))

        points = journal_feed.equity_curve(trades, mode)
        if PYQTGRAPH_AVAILABLE:
            self.curve.clear()
            if points:
                self.curve.plot(list(range(len(points))), [value for _day, value in points])
        self.curve_table.setRowCount(len(points))
        for row, (day, value) in enumerate(points):
            self.curve_table.setItem(row, 0, QTableWidgetItem(day))
            self.curve_table.setItem(row, 1, QTableWidgetItem(f"{value:,.2f}"))

        groups = summary.get("groups") or {}
        rows = [
            (group_name, stats)
            for group_name, buckets in groups.items()
            for stats in (buckets or [])
        ]
        self.groups_table.setRowCount(len(rows))
        for row, (group_name, stats) in enumerate(rows):
            values = [
                group_name,
                stats.get("label", ""),
                stats.get("trades", 0),
                f"{stats['win_rate']:.0%}" if stats.get("win_rate") is not None else "-",
                f"{stats['profit_factor']:.2f}" if stats.get("profit_factor") is not None else "-",
                f"{stats['net_pnl']:,.2f}" if stats.get("net_pnl") is not None else "-",
            ]
            for column, text in enumerate(values):
                self.groups_table.setItem(row, column, QTableWidgetItem(str(text)))

    def _run_walkaway(self) -> None:  # pragma: no cover - worker path
        if self._worker is not None and self._worker.isRunning():
            return
        since, until = self._header.date_bounds()
        self.walkaway_button.setEnabled(False)
        self.walkaway_output.setText("Running walk-away...")
        self._worker = _WalkawayWorker(since, until, self)
        self._worker.finished_with.connect(self._on_walkaway_done)
        self._worker.failed.connect(self._on_walkaway_failed)
        self._worker.start()

    def _on_walkaway_done(self, result: dict) -> None:  # pragma: no cover
        self.walkaway_button.setEnabled(True)
        self.walkaway_output.setText(journal_feed.render_walkaway_summary(result))

    def _on_walkaway_failed(self, message: str) -> None:  # pragma: no cover
        self.walkaway_button.setEnabled(True)
        self.walkaway_output.setText(f"Walk-away failed: {message}")
        self.statusChanged.emit(f"walk-away failed: {message}")

    def _export(self) -> None:
        try:
            path = journal_feed.export_trades_csv()
        except Exception as exc:  # noqa: BLE001
            self.statusChanged.emit(f"export failed: {exc}")
            return
        self.statusChanged.emit(f"exported {path}")

    def shutdown(self) -> None:
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.wait(2000)
