from __future__ import annotations

import csv
import threading
import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
)
from PySide6.QtCore import Qt

from move_forensics import (
    FORENSICS_AI_DIGEST_JSON,
    FORENSICS_BASELINE_CSV,
    FORENSICS_MOVERS_CSV,
    FORENSICS_PATTERNS_CSV,
    FORENSICS_REPORT_TXT,
    run_move_forensics,
)
from ui.models.tracker_table_model import ROW_ROLE, TrackerSortProxyModel, TrackerTableModel
from ui.widgets.data_table import DataTable
from ui.widgets.research_explanation_view import ResearchExplanationView
from ui.widgets.section_header import SectionHeader

PATTERN_COLUMNS = (
    ("side", "Side"),
    ("kind", "Kind"),
    ("pattern", "Conditions"),
    ("lift", "Lift"),
    ("movers_with", "Moves"),
    ("mover_rate", "% of Moves"),
    ("baseline_rate", "% Ordinary"),
    ("avg_move_atr", "Avg Move ATR"),
    ("novel", "Not In Playbook"),
)
PERCENT_KEYS = {"mover_rate", "baseline_rate"}
NUMERIC_KEYS = {"lift", "movers_with", "mover_rate", "baseline_rate", "avg_move_atr"}


class MoveForensicsPanel(QFrame):
    """Research tab: outcome-first critical thinking, on command.

    Runs the move-forensics study (find every big clean move long AND short,
    snapshot the tracked conditions at each move's start, mine single and
    pair patterns for lift vs ordinary days) in a background thread, then
    shows the pattern leaderboard and the report. The movers/baseline CSVs +
    AI digest JSON it writes are the database for a Claude deep-dive.
    """

    statusChanged = Signal(str)
    _runFinished = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self._run_thread: threading.Thread | None = None

        self.days_input = QSpinBox()
        self.days_input.setRange(40, 500)
        self.days_input.setValue(150)
        self.days_input.setSuffix(" sessions")

        self.move_atr_input = QDoubleSpinBox()
        self.move_atr_input.setRange(1.5, 8.0)
        self.move_atr_input.setSingleStep(0.5)
        self.move_atr_input.setValue(3.0)
        self.move_atr_input.setSuffix(" ATR min move")

        self.horizon_input = QSpinBox()
        self.horizon_input.setRange(3, 20)
        self.horizon_input.setValue(10)
        self.horizon_input.setSuffix(" d horizon")

        self.max_symbols_input = QSpinBox()
        self.max_symbols_input.setRange(0, 5000)
        self.max_symbols_input.setValue(0)
        self.max_symbols_input.setSpecialValueText("all symbols")
        self.max_symbols_input.setToolTip("Cap the symbol count for a faster exploratory pass (0 = full store).")

        self.run_button = QPushButton("Run Move Forensics")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.clicked.connect(self.start_run)

        self.status_label = QLabel(
            "Finds every clean big move (long + short), then mines which tracked conditions - alone "
            "and in pairs - preceded them. Full store takes several minutes; cap symbols for a quick pass."
        )
        self.status_label.setObjectName("MutedLabel")
        self.status_label.setWordWrap(True)

        self.model = TrackerTableModel(
            PATTERN_COLUMNS,
            percent_keys=PERCENT_KEYS,
            signed_keys=set(),
            numeric_keys=NUMERIC_KEYS,
            tooltip_keys={"pattern"},
        )
        proxy = TrackerSortProxyModel(self)
        proxy.setSourceModel(self.model)
        self.table = DataTable()
        self.table.setModel(proxy)
        self.table.setShowGrid(False)
        self.explanation_view = ResearchExplanationView(self)
        self.table.clicked.connect(self._show_row_explanation)

        self.report_view = QTextBrowser()
        self.report_view.setLineWrapMode(QTextBrowser.LineWrapMode.NoWrap)
        self.report_view.setFontFamily("Consolas")

        self._runFinished.connect(self._on_run_finished)
        self._build_layout()
        self.reload_from_disk()

    def _build_layout(self) -> None:
        header = SectionHeader(
            "Move Forensics",
            "Outcome-first: every stock that made a good move, and the condition combos that preceded "
            "them - including ones the playbook does not scan for yet.",
        )
        header.add_action(self.run_button)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)
        for widget in (self.days_input, self.move_atr_input, self.horizon_input, self.max_symbols_input):
            controls.addWidget(widget)
        controls.addStretch(1)

        results_splitter = QSplitter(Qt.Orientation.Vertical)
        results_splitter.addWidget(self.table)
        results_splitter.addWidget(self.report_view)
        results_splitter.setStretchFactor(0, 3)
        results_splitter.setStretchFactor(1, 2)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(results_splitter)
        splitter.addWidget(self.explanation_view)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(header)
        layout.addLayout(controls)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.status_label)

    # ------------------------------------------------------------------
    def reload_from_disk(self) -> None:
        rows = _load_csv_rows(FORENSICS_PATTERNS_CSV)
        if rows:
            self.model.set_rows(rows[:400])
            self.table.fit_columns()
        if Path(FORENSICS_REPORT_TXT).exists():
            self.report_view.setPlainText(Path(FORENSICS_REPORT_TXT).read_text(encoding="utf-8"))
        if rows:
            self._set_status(
                f"Loaded last run from disk ({len(rows)} pattern rows). AI digest: {FORENSICS_AI_DIGEST_JSON}"
            )

    def start_run(self) -> None:
        if self._run_thread is not None and self._run_thread.is_alive():
            return
        self.run_button.setEnabled(False)
        self._set_status("Move forensics running... (loading bars, finding moves, mining patterns)")
        params = {
            "days": int(self.days_input.value()),
            "min_move_atr": float(self.move_atr_input.value()),
            "horizon": int(self.horizon_input.value()),
            "max_symbols": int(self.max_symbols_input.value()) or None,
        }
        self._run_thread = threading.Thread(
            target=self._run_worker, kwargs=params, name="move-forensics", daemon=True
        )
        self._run_thread.start()

    def _run_worker(self, **params) -> None:
        try:
            result = run_move_forensics(
                progress=lambda message: self._runFinished.emit(f"PROGRESS:{message}"),
                **params,
            )
            longs = result["digest"]["mover_counts"].get("LONG", 0)
            shorts = result["digest"]["mover_counts"].get("SHORT", 0)
            message = (
                f"Done: {longs} long / {shorts} short moves across {result['symbols_processed']} symbols. "
                f"Exports written for AI analysis: {FORENSICS_MOVERS_CSV.name}, "
                f"{FORENSICS_BASELINE_CSV.name}, {FORENSICS_AI_DIGEST_JSON.name}."
            )
        except Exception as exc:  # surfaced in the status label, never crashes the UI
            traceback.print_exc()
            message = f"Move forensics failed: {exc}"
        self._runFinished.emit(message)

    def _on_run_finished(self, message: str) -> None:
        if message.startswith("PROGRESS:"):
            self._set_status(message.split(":", 1)[1])
            return
        self.run_button.setEnabled(True)
        self.reload_from_disk()
        self._set_status(message)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.statusChanged.emit(f"Move forensics: {message}")

    def _show_row_explanation(self, index) -> None:
        row = index.data(ROW_ROLE)
        if isinstance(row, dict):
            self.explanation_view.show_row("move_forensics", row)


def _load_csv_rows(path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
