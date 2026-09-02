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

import csv
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Signal
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

from ui.services import journal_feed

try:  # pragma: no cover - the desk has pyqtgraph; a headless box may not.
    import pyqtgraph as pg

    PYQTGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover
    pg = None
    PYQTGRAPH_AVAILABLE = False


#: Buckets with fewer closed trades than this are drawn, but labeled thin. The
#: number is not a threshold anything is decided on - it is the point at which
#: a bar chart starts to look like evidence when it is not.
THIN_SAMPLE_TRADES = 5

#: How many buckets one chart shows. Beyond this the axis is unreadable, and
#: what is dropped is SAID rather than silently trimmed.
GROUP_CHART_MAX_BARS = 12


def group_breakdown_rows(summary: dict, group_name: str) -> list[dict]:
    """The chosen group's buckets, sorted the way the table already sorts."""
    groups = summary.get("groups") or {}
    return list(groups.get(group_name) or [])


def group_chart_series(rows: list[dict]) -> tuple[list[str], list[float], int]:
    """(labels with honest n, net values, dropped count) for one breakdown.

    A bucket whose net is ``None`` is EXCLUDED, never plotted as zero: None
    here means "mixed currencies with unconverted rows", and a zero bar would
    read as "this setup broke even" - a claim the data refuses to make. The
    count of what was excluded is returned so the caller can say it.
    """
    plottable = [row for row in rows if row.get("net_pnl") is not None]
    dropped = len(rows) - len(plottable)
    shown = plottable[:GROUP_CHART_MAX_BARS]
    dropped += max(0, len(plottable) - len(shown))
    labels = []
    values = []
    for row in shown:
        closed = int(row.get("closed", 0) or 0)
        label = f"{row.get('label', '')} (n={closed})"
        if closed < THIN_SAMPLE_TRADES:
            label += " thin"
        labels.append(label)
        values.append(float(row.get("net_pnl") or 0.0))
    return labels, values, dropped


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

        # R7's deferred per-group charts, built 2026-08-18. The table below
        # already carries every number; what was missing is the SHAPE, and the
        # n count beside each bar is what stops a two-trade setup from looking
        # like a finding.
        self.group_picker = QComboBox()
        self.group_chart = (
            pg.PlotWidget(title="Net by group")
            if PYQTGRAPH_AVAILABLE
            else QLabel("pyqtgraph is not installed; the table below carries the same numbers.")
        )
        self.group_note = QLabel("")
        self.group_note.setWordWrap(True)
        self.group_picker.currentTextChanged.connect(lambda _text: self._draw_group_chart())
        self.group_csv_button = QPushButton("Export this breakdown CSV")
        self.group_csv_button.clicked.connect(self._export_group_csv)
        self._summary: dict = {}

        self.walkaway_button = QPushButton("Run walk-away for this range")
        self.walkaway_button.clicked.connect(self._run_walkaway)
        self.walkaway_output = QLabel("Walk-away has not been run for this range yet.")
        self.walkaway_output.setWordWrap(True)

        self.export_button = QPushButton("Export trades CSV")
        self.export_button.clicked.connect(self._export)

        buttons = QHBoxLayout()
        buttons.addWidget(self.walkaway_button)
        buttons.addWidget(self.export_button)
        buttons.addWidget(self.group_csv_button)
        buttons.addStretch(1)

        picker_row = QHBoxLayout()
        picker_row.addWidget(QLabel("Chart group"))
        picker_row.addWidget(self.group_picker)
        picker_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.headline)
        layout.addWidget(self.currency_note)
        layout.addWidget(self.curve, 3)
        layout.addWidget(self.curve_table, 1)
        layout.addWidget(QLabel("By group"))
        layout.addLayout(picker_row)
        layout.addWidget(self.group_chart, 2)
        layout.addWidget(self.group_note)
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

        self._summary = summary
        groups = summary.get("groups") or {}
        self._sync_group_picker(groups)
        self._draw_group_chart()
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

    def _sync_group_picker(self, groups: dict) -> None:
        names = list(groups.keys())
        existing = [self.group_picker.itemText(i) for i in range(self.group_picker.count())]
        if names == existing:
            return
        current = self.group_picker.currentText()
        self.group_picker.blockSignals(True)
        self.group_picker.clear()
        self.group_picker.addItems(names)
        if current in names:
            self.group_picker.setCurrentText(current)
        self.group_picker.blockSignals(False)

    def _draw_group_chart(self) -> None:
        group_name = self.group_picker.currentText()
        rows = group_breakdown_rows(self._summary, group_name)
        # R1: `group_notes` was written and nothing read it. The note is the
        # whole point of the coverage check - a bar chart of five tagged trades
        # beside a full one, at the same width, with nothing saying which is
        # which. PREPENDED to the group's own label, as the packet asked.
        coverage_note = str((self._summary.get("group_notes") or {}).get(group_name) or "")
        labels, values, dropped = group_chart_series(rows)
        if PYQTGRAPH_AVAILABLE:
            self.group_chart.clear()
            if values:
                bars = pg.BarGraphItem(x=list(range(len(values))), height=values, width=0.6)
                self.group_chart.addItem(bars)
                self.group_chart.getAxis("bottom").setTicks([list(enumerate(labels))])
            self.group_chart.setTitle(f"Net by {group_name}" if group_name else "Net by group")
        parts = []
        if coverage_note:
            parts.append(coverage_note)
        if not rows:
            parts.append("No trades in this range for that grouping.")
        else:
            thin = sum(1 for row in rows if int(row.get("closed", 0) or 0) < THIN_SAMPLE_TRADES)
            parts.append(
                f"{len(labels)} bucket(s) charted; n is closed trades. "
                f"{thin} bucket(s) have fewer than {THIN_SAMPLE_TRADES} closed trades "
                "and are labeled thin."
            )
        if dropped:
            # Said out loud: a chart that silently drops buckets reads as
            # "that was all of them".
            parts.append(
                f"{dropped} bucket(s) not charted (no convertible total, or beyond the "
                f"{GROUP_CHART_MAX_BARS}-bar cap). They are still in the table below."
            )
        if group_name in (self._summary.get("nonexclusive_groups") or []):
            parts.append(
                "One trade can carry several tags here, so these buckets overlap and "
                "do not sum to the headline."
            )
        if group_name in (self._summary.get("provisional_groups") or []):
            # P6a. Said on the chart itself rather than only in the group's name:
            # this is the one breakdown on the page whose buckets nobody has
            # agreed to yet, and a bar is a bar.
            parts.append(
                "These tags were applied for you and are still waiting for review - "
                "confirm or correct them in the Trades tab, where they are marked "
                "provisional. They are never counted under \"my setups\"."
            )
        self.group_note.setText(" ".join(parts))

    def _export_group_csv(self) -> None:
        """The charted breakdown, exactly as shown, as a CSV beside it."""
        group_name = self.group_picker.currentText()
        rows = group_breakdown_rows(self._summary, group_name)
        if not rows:
            self.statusChanged.emit("nothing to export for that grouping")
            return
        try:
            from project_paths import JOURNAL_EXPORT_DIR

            slug = "".join(ch if ch.isalnum() else "_" for ch in group_name) or "group"
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = Path(JOURNAL_EXPORT_DIR) / f"journal_by_{slug}_{stamp}.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            columns = ["label", "trades", "closed", "win_rate", "profit_factor", "net_pnl"]
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow({column: row.get(column, "") for column in columns})
        except Exception as exc:  # noqa: BLE001
            self.statusChanged.emit(f"breakdown export failed: {exc}")
            return
        self.statusChanged.emit(f"exported {target}")

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
