"""Research -> Long lab: the shadow long-rule replay, per rule x regime x horizon. Display only.

Constructing the panel reads nothing. "Load last report" reads the saved JSON on
a worker; "Run lab" replays the cached daily bars on a worker and saves the
report there. This panel only formats what comes back. Nothing here feeds a
score, alert, grade or list.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ui.read_worker import ReadWorker, join_worker
from ui.widgets.section_header import SectionHeader

AXIS_CHOICES = (
    ("spy_trend", "SPY vs rising 20-day"),
    ("structural", "Your structural regime"),
    ("month", "Month"),
)
COLUMNS = (
    ("rule", "Rule"),
    ("regime", "Regime"),
    ("horizon", "Sessions"),
    ("n", "n"),
    ("win_raw", "Win raw"),
    ("wilson_lb", "Win low bound"),
    ("win_vs_spy", "Beat SPY"),
    ("mean_raw", "Mean"),
    ("median_raw", "Median"),
    ("mean_vs_spy", "Mean vs SPY"),
    ("median_vs_spy", "Median vs SPY"),
    ("mfe_atr", "MFE ATR"),
    ("mae_atr", "MAE ATR"),
    ("limit_fill_rate", "Limit fill"),
    ("best_exit", "Best exit"),
)
SWEEP_COLUMNS = (
    ("knob", "Knob"),
    ("bucket", "Bucket"),
    ("regime", "Regime"),
    ("n", "n"),
    ("win_vs_spy", "Beat SPY"),
    ("mean_vs_spy", "Mean vs SPY"),
    ("mean_raw", "Mean"),
)
PERCENT_KEYS = {"win_raw", "wilson_lb", "win_vs_spy", "limit_fill_rate"}
RETURN_KEYS = {"mean_raw", "median_raw", "mean_vs_spy", "median_vs_spy"}
SWEEP_HORIZON_TEXT = "10 sessions"
NO_REPORT_TEXT = "No long-lab report yet. Press Run lab (it reads the cached daily bars)."
CAVEAT_TEXT = (
    "SHADOW STUDY - nothing here changes an alert, score, grade or list. Each flag uses only "
    "bars up to its close; entry is the next open. Numbers are per regime, never pooled. "
    "Win low bound = 95% Wilson. MFE/MAE are medians in the flag day's ATR. Limit = 0.25 ATR "
    "under the flag close, live 3 sessions. Best exit (mean ATR, n >= 30): +1 ATR take with a "
    "1 ATR stop, a 10-session time stop, or a 1 ATR trail. The universe is today's cached "
    "names, so delisted names are missing (survivorship bias). The sweep buckets the loose "
    "leader-pullback set at 10 sessions."
)


class LongLabPanel(QFrame):
    """Two buttons, an axis picker, the rule table and the sweep table. The worker owns IO."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        layout = QVBoxLayout(self)
        layout.addWidget(SectionHeader("Long lab (shadow replay of long rules)"))
        controls = QHBoxLayout()
        self.load_button = QPushButton("Load last report")
        self.load_button.clicked.connect(self.load)
        controls.addWidget(self.load_button)
        self.run_button = QPushButton("Run lab")
        self.run_button.clicked.connect(self.run)
        controls.addWidget(self.run_button)
        self.axis_combo = QComboBox()
        for key, label in AXIS_CHOICES:
            self.axis_combo.addItem(label, key)
        self.axis_combo.currentIndexChanged.connect(self._render)
        controls.addWidget(self.axis_combo)
        self.status_label = QLabel("Press Load last report or Run lab.")
        self.status_label.setWordWrap(True)
        controls.addWidget(self.status_label, stretch=1)
        layout.addLayout(controls)
        self.table = self._table(COLUMNS)
        layout.addWidget(self.table, stretch=3)
        self.sweep_label = QLabel("")
        self.sweep_label.setWordWrap(True)
        layout.addWidget(self.sweep_label)
        self.sweep_table = self._table(SWEEP_COLUMNS)
        layout.addWidget(self.sweep_table, stretch=2)
        caveat = QLabel(CAVEAT_TEXT)
        caveat.setWordWrap(True)
        caveat.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(caveat)
        self._report: dict | None = None
        self._worker: ReadWorker | None = None

    @staticmethod
    def _table(columns) -> QTableWidget:
        table = QTableWidget(0, len(columns))
        table.setHorizontalHeaderLabels([label for _key, label in columns])
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        return table

    # ------------------------------------------------------------ worker
    def load(self) -> None:
        self._start(self._read_saved, "Reading the last report...")

    def run(self) -> None:
        self._start(self._run_and_save, "Running the lab (about a minute)...")

    def _start(self, fn, text: str) -> None:
        """Single-flight: one worker at a time."""
        if self._worker is not None and self._worker.isRunning():
            self.status_label.setText("Still running...")
            return
        self.load_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.status_label.setText(text)
        worker = ReadWorker(fn, self)
        worker.finished_with.connect(self._on_report)
        worker.failed.connect(self._on_failed)
        self._worker = worker
        worker.start()

    @staticmethod
    def _read_saved():
        from research_warehouse import long_lab

        return long_lab.read_report()

    @staticmethod
    def _run_and_save():
        from research_warehouse import long_lab

        report = long_lab.lab_from_live_inputs()
        long_lab.write_report(report)
        return report

    def _on_report(self, report) -> None:
        self._done()
        self.show_report(report)

    def _on_failed(self, message: str) -> None:
        self._done()
        self.status_label.setText(f"Long lab failed: {message}. The rows below are the last good report.")

    def _done(self) -> None:
        self.load_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self._worker = None

    # ------------------------------------------------------------ display
    def show_report(self, report) -> None:
        """Keeps and formats a report. Pure display."""
        if not isinstance(report, dict) or report.get("error"):
            self.status_label.setText(NO_REPORT_TEXT if not isinstance(report, dict)
                                      else f"Long lab: {report.get('error')}.")
            return
        self._report = report
        self._render()

    def _render(self, *_args) -> None:
        report = self._report
        if report is None:
            return
        axis = self.axis_combo.currentData()
        labels = {rule.get("key"): rule.get("label") or rule.get("key") for rule in report.get("rules") or ()}
        rows = [cell for cell in report.get("cells") or () if cell.get("axis") == axis]
        self._fill(self.table, COLUMNS, rows, labels)
        sweep_axis = axis if axis in ("spy_trend", "structural") else "spy_trend"
        sweep = [cell for cell in report.get("sweep") or () if cell.get("axis") == sweep_axis]
        self._fill(self.sweep_table, SWEEP_COLUMNS, sweep, labels)
        spreads = sorted(
            (item for item in report.get("knob_spread") or () if item.get("axis") == sweep_axis),
            key=lambda item: -float(item.get("spread_vs_spy") or 0.0),
        )
        spread_text = "; ".join(
            f"{item['regime']}: {item['knob']} {float(item['spread_vs_spy']) * 100:.1f} pts"
            for item in spreads[:6]
        ) or "no knob has two buckets with n >= 30"
        self.sweep_label.setText(
            f"Leader pullback sweep ({SWEEP_HORIZON_TEXT}). Which knob matters (spread of mean vs "
            f"SPY across buckets): {spread_text}."
        )
        universe = report.get("universe") or {}
        self.status_label.setText(
            f"{report.get('sessions') or 0} sessions ({report.get('first_session') or '-'} to "
            f"{report.get('last_session') or '-'}), {universe.get('symbols_in_window', '-')} names. "
            f"Generated {report.get('generated_at') or '-'}."
        )

    @staticmethod
    def _fill(table: QTableWidget, columns, rows, labels) -> None:
        table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            for column, (key, _label) in enumerate(columns):
                value = labels.get(row.get(key), row.get(key)) if key == "rule" else row.get(key)
                text = format_cell(key, value)
                if key == "n" and row.get("thin"):
                    text += " (thin)"
                table.setItem(index, column, QTableWidgetItem(text))

    def row_count(self) -> int:
        return self.table.rowCount()

    def shutdown(self) -> None:
        join_worker(self._worker)
        self._worker = None



def format_cell(key: str, value) -> str:
    if value is None:
        return "-"
    if key in PERCENT_KEYS:
        return f"{float(value) * 100:.0f}%"
    if key in RETURN_KEYS:
        return f"{float(value) * 100:+.1f}%"
    if isinstance(value, float):
        return f"{value:+.2f}"
    return str(value)
