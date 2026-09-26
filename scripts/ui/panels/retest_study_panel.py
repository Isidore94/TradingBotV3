"""Research -> Retest entry: the S8 study's two EVs per family. Display only.

Constructing the panel reads nothing. "Run study" computes the report on a
worker (the candidates CSV and the lake's M5 bars) and this panel only
formats what comes back. Nothing here feeds a score, alert or gate.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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

COLUMNS = (
    ("source", "Source"),
    ("family", "Family"),
    ("side", "Side"),
    ("n", "n"),
    ("flag_ev_r", "Flag close EV R"),
    ("flag_win_share", "Flag win %"),
    ("retest_n_filled", "Retest fills"),
    ("retest_no_fill_share", "No-fill %"),
    ("retest_ev_r_per_fill", "Retest EV R / fill"),
    ("retest_ev_r_per_alert", "Retest EV R / alert"),
)
PERCENT_KEYS = {"flag_win_share", "retest_no_fill_share"}
DISABLED_TEXT = "Research warehouse is not configured, so there are no cached M5 bars to study."
CAVEAT_TEXT = (
    "SHADOW STUDY - nothing here changes an alert, score or grade. Flag close = enter at "
    "the alert bar's close. Retest = a limit at the level +/- 0.25 M5 ATR, live for the 6 "
    "bars after the alert; no touch = no trade. Both share one stop (level -/+ 0.5 ATR) "
    "and the 1:1 bracket (+1R before -1R, a bar touching both is a loss, else the "
    "session-close R). 'Per alert' counts a no-fill as 0R, so it compares with the flag "
    "EV. Only bars after the alert are used."
)


class RetestStudyPanel(QFrame):
    """Run button, status line, one table. The worker owns every read."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        layout = QVBoxLayout(self)
        layout.addWidget(SectionHeader("Retest entry vs flag close (shadow study)"))
        controls = QHBoxLayout()
        self.run_button = QPushButton("Run study")
        self.run_button.clicked.connect(self.refresh)
        controls.addWidget(self.run_button)
        self.status_label = QLabel("Press Run study. It reads the last 45 days of alerts.")
        self.status_label.setWordWrap(True)
        controls.addWidget(self.status_label, stretch=1)
        layout.addLayout(controls)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([label for _key, label in COLUMNS])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.table, stretch=1)
        caveat = QLabel(CAVEAT_TEXT)
        caveat.setWordWrap(True)
        caveat.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(caveat)
        self._worker: ReadWorker | None = None

    def refresh(self) -> None:
        """Run the study once, off the Qt thread. Single-flight."""
        if self._worker is not None and self._worker.isRunning():
            self.status_label.setText("Still running...")
            return
        self.run_button.setEnabled(False)
        self.status_label.setText("Running the study...")
        worker = ReadWorker(self._compute, self)
        worker.finished_with.connect(self._on_report)
        worker.failed.connect(self._on_failed)
        self._worker = worker
        worker.start()

    @staticmethod
    def _compute():
        """Everything that reads files. Runs on the worker."""
        from research_warehouse import retest_entry
        from research_warehouse.store import ResearchStore

        store = ResearchStore.open()
        if store is None:
            return None
        return retest_entry.study_from_live_inputs(store)

    def _on_report(self, report) -> None:
        self.run_button.setEnabled(True)
        self._worker = None
        self.show_report(report)

    def _on_failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self._worker = None
        self.status_label.setText(f"Study failed: {message}. The rows below are the last good run.")

    def show_report(self, report) -> None:
        """Formats a study report. Pure display."""
        if not isinstance(report, dict):
            self.status_label.setText(DISABLED_TEXT)
            return
        rows = list(report.get("families") or [])
        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            for column, (key, _label) in enumerate(COLUMNS):
                self.table.setItem(index, column, QTableWidgetItem(format_cell(key, row.get(key))))
        skipped = report.get("skipped") or {}
        skipped_text = ", ".join(f"{name} {count}" for name, count in skipped.items()) or "none"
        self.status_label.setText(
            f"{sum(int(r.get('n') or 0) for r in rows)} alerts, {len(rows)} families, "
            f"{report.get('sessions') or 0} sessions ({report.get('first_session') or '-'} to "
            f"{report.get('last_session') or '-'}). Skipped (unknown): {skipped_text}."
        )

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
    if isinstance(value, float):
        return f"{value:+.2f}"
    return str(value)
