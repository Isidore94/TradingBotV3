"""Research -> Setup keys (WISHLIST P1-4 / 4c): the permutation report, read-only.

Family -> ranked facet keys with lift, n, sessions and the hold-out result, and
"no key found" as a first-class row. Swing and day trades are separate
populations and are never shown pooled. The report file is read on a worker
thread only when the trader presses Refresh; the Qt thread only fills the table.

Shadow only: nothing here ranks, filters or alerts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

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

POPULATIONS = (("swing", "Swing"), ("m5", "Day trades (M5)"))

COLUMNS = (
    ("family", "Family"),
    ("rank", "#"),
    ("key", "Key"),
    ("depth", "Depth"),
    ("lift_pp", "Lift (pp)"),
    ("win_rate", "Win %"),
    ("wilson_lb", "Low bound"),
    ("mean_r", "Mean R"),
    ("n", "n"),
    ("sessions", "Sessions"),
    ("holdout", "Hold-out (last 20 sessions)"),
)

NO_REPORT_TEXT = (
    "No setup-keys report yet. Run the backfill and the search "
    "(scripts/setup_permutation_backfill.py, then setup_permutation_search.py --out {path})."
)
CAVEAT_TEXT = (
    "SHADOW ONLY: nothing here ranks, filters or alerts. A key is reported only when it beats the "
    "family baseline in selection (n >= 30 over 10+ sessions) AND on the last 20 sessions, which "
    "selection never saw. 'No key found' is an answer, not a gap."
)
_VERDICT_TEXT = {"no_key_found": "no key found", "too_little_data": "too little data"}


def _pct(value: Any) -> str:
    return "-" if value is None else f"{float(value) * 100:.0f}%"


def _num(value: Any, fmt: str) -> str:
    return "-" if value is None else format(float(value), fmt)


def horizons_in(report: Mapping[str, Any] | None, population: str) -> list[str]:
    pops = (report or {}).get("populations") or {}
    return sorted(((pops.get(population) or {}).get("horizons") or {}), key=lambda text: int(text))


def report_rows(report: Mapping[str, Any] | None, population: str, horizon: str) -> list[dict[str, str]]:
    """Display rows for one population and horizon. Pure: no Qt, no I/O."""
    block = (((report or {}).get("populations") or {}).get(population) or {}).get("horizons", {}).get(horizon) or {}
    rows: list[dict[str, str]] = []
    for name, family in sorted((block.get("families") or {}).items()):
        base = family.get("baseline") or {}
        keys = family.get("keys") or []
        if not keys:
            rows.append({
                "family": name, "rank": "-",
                "key": _VERDICT_TEXT.get(str(family.get("verdict")), "no key found"),
                "depth": "-", "lift_pp": "-", "win_rate": _pct(base.get("win_rate")),
                "wilson_lb": _pct(base.get("wilson_lb")), "mean_r": _num(base.get("mean_r"), "+.2f"),
                "n": str(base.get("n", 0)), "sessions": str(base.get("sessions", 0)),
                "holdout": f"baseline {_pct((family.get('holdout_baseline') or {}).get('win_rate'))}",
            })
            continue
        for key in keys:
            selection = key.get("selection") or {}
            hold = key.get("holdout") or {}
            rows.append({
                "family": name,
                "rank": str(key.get("rank", "")),
                "key": str(key.get("label") or ""),
                "depth": str(key.get("depth", "")),
                "lift_pp": _num(key.get("lift_pp"), "+.1f"),
                "win_rate": _pct(selection.get("win_rate")),
                "wilson_lb": _pct(selection.get("wilson_lb")),
                "mean_r": _num(selection.get("mean_r"), "+.2f"),
                "n": str(selection.get("n", "")),
                "sessions": str(selection.get("sessions", "")),
                "holdout": (
                    f"passed: {_pct(hold.get('win_rate'))} on n={hold.get('n', 0)} "
                    f"vs {_pct((family.get('holdout_baseline') or {}).get('win_rate'))} baseline"
                ),
            })
    return rows


def read_report(path: Path) -> dict[str, Any] | None:
    """The report, or None when there is none yet. Runs on the worker."""
    target = Path(path)
    if not target.is_file():
        return None
    payload = json.loads(target.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


class SetupKeysPanel(QFrame):
    """Read-only view of `permutation_report.json`. Refresh is always explicit."""

    def __init__(self, report_path: Path | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        if report_path is None:
            import project_paths

            report_path = project_paths.SETUP_PERMUTATION_REPORT_FILE
        self.report_path = Path(report_path)
        self._report: dict[str, Any] | None = None
        self._worker: ReadWorker | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(SectionHeader("Setup keys (shadow-only evidence)"))
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Population"))
        self.population_input = QComboBox()
        for key, label in POPULATIONS:
            self.population_input.addItem(label, key)
        controls.addWidget(self.population_input)
        controls.addWidget(QLabel("Horizon"))
        self.horizon_input = QComboBox()
        controls.addWidget(self.horizon_input)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        controls.addWidget(self.refresh_button)
        self.status_label = QLabel("Press Refresh to read the setup-keys report.")
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

        self.population_input.currentIndexChanged.connect(self._on_population)
        self.horizon_input.currentIndexChanged.connect(lambda _index: self._render())

    def refresh(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self.status_label.setText("Still reading the report...")
            return
        self.refresh_button.setEnabled(False)
        self.status_label.setText("Reading the report...")
        path = self.report_path
        worker = ReadWorker(lambda: read_report(path), self)
        worker.finished_with.connect(self._on_read)
        worker.failed.connect(self._on_failed)
        self._worker = worker
        worker.start()

    def _on_read(self, report) -> None:
        self.refresh_button.setEnabled(True)
        self._worker = None
        if report is None:
            self.status_label.setText(NO_REPORT_TEXT.format(path=self.report_path))
            return
        self._report = report
        self._on_population()
        self.status_label.setText(
            f"Report {report.get('generated_at', '?')} ({report.get('search_version', '?')}, "
            f"{report.get('permutation_rule_version', '?')})."
        )

    def _on_failed(self, message: str) -> None:
        self.refresh_button.setEnabled(True)
        self._worker = None
        self.status_label.setText(f"Read failed: {message}. The rows below are the last good read.")

    def _on_population(self, *_args) -> None:
        population = str(self.population_input.currentData() or "swing")
        current = self.horizon_input.currentData()
        self.horizon_input.blockSignals(True)
        self.horizon_input.clear()
        for horizon in horizons_in(self._report, population):
            label = "first 30 minutes" if horizon == "0" else f"{horizon} session(s)"
            self.horizon_input.addItem(label, horizon)
        index = self.horizon_input.findData(current)
        self.horizon_input.setCurrentIndex(index if index >= 0 else 0)
        self.horizon_input.blockSignals(False)
        self._render()

    def _render(self) -> None:
        population = str(self.population_input.currentData() or "swing")
        horizon = str(self.horizon_input.currentData() or "")
        rows = report_rows(self._report, population, horizon)
        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            for column, (key, _label) in enumerate(COLUMNS):
                self.table.setItem(index, column, QTableWidgetItem(row.get(key, "")))

    def row_count(self) -> int:
        return self.table.rowCount()

    def shutdown(self) -> None:
        join_worker(self._worker)
        self._worker = None
