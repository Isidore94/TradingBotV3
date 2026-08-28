"""Minimal research-warehouse readout (plan Phase 7, sec 17).

The whole Phase-7 UI deliverable: raw canned-query results for the two slice
setups - counts, mean R, checkpoint values - and nothing else. No shrinkage, no
intervals, no evidence tiers beyond the EXPLORATORY label, no ranking. The
Section 16.3 output contract and the Current Edge dashboard arrive with
milestone M-E and are deliberately absent here.

Two rules shape this widget:

* **Nothing reads the lake on the render path** (sec 20). Constructing the
  panel touches no files; a read happens only when the trader presses Refresh.
* **It is inert when the warehouse is disabled.** With no ``research_store_dir``
  configured the panel says so and stays empty, exactly like every other
  warehouse entry point.
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
    ("canonical_setup_id", "Setup"),
    ("side", "Side"),
    ("recipe_id", "Recipe"),
    ("n_episodes", "Episodes"),
    ("n_occurrences", "Occurrences"),
    ("n_matured", "Matured"),
    ("n_open", "Open"),
    ("n_no_trigger", "No trigger"),
    ("mean_net_r", "Mean net R"),
    ("mean_r_at_s18", "R @ s18"),
    ("mean_r_at_60m", "R @ 60m"),
    ("mean_mfe_r", "Mean MFE R"),
    ("mean_mae_r", "Mean MAE R"),
)

DISABLED_TEXT = (
    "Research warehouse is not configured. Set a research store directory in "
    "Settings (or TRADINGBOTV3_RESEARCH_DIR) to enable this readout."
)
EMPTY_TEXT = "No slice outcomes recorded yet. Run the warehouse build job first."
CAVEAT_TEXT = (
    "EXPLORATORY: raw counts only - no shrinkage, no intervals, no ranking, and nothing here "
    "affects a score or an alert. 'Episodes' is the sample size; rows and occurrences are not."
)


class WarehouseReadoutPanel(QFrame):
    """Read-only table over the research lake. Refresh is always explicit."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        layout = QVBoxLayout(self)
        layout.addWidget(SectionHeader("Research Warehouse (shadow-only evidence)"))

        controls = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        controls.addWidget(self.refresh_button)
        self.status_label = QLabel("Press Refresh to read the lake.")
        self.status_label.setWordWrap(True)
        controls.addWidget(self.status_label, stretch=1)
        layout.addLayout(controls)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([label for _key, label in COLUMNS])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.table, stretch=1)

        self._worker: ReadWorker | None = None

        self.caveat_label = QLabel(CAVEAT_TEXT)
        self.caveat_label.setWordWrap(True)
        self.caveat_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.caveat_label)

    # -- data ---------------------------------------------------------------
    def refresh(self) -> None:
        """Read the lake once, on demand, OFF the Qt thread. Single-flight.

        G-P1.5. This ran `ResearchStore.open()` and `queries.slice_readout()`
        inline. The lake lives on the DAS, and that share is known to drop and
        re-establish; an SMB read against a dropped share does not fail fast,
        it blocks until it times out - and every one of those seconds was a
        frozen desk with nothing on screen to explain it. This is the only
        panel in the audit whose read leaves the machine.
        """
        if self._worker is not None and self._worker.isRunning():
            self.status_label.setText("Still reading the lake...")
            return
        self.refresh_button.setEnabled(False)
        self.status_label.setText("Reading the lake...")
        worker = ReadWorker(self._read_lake, self)
        worker.finished_with.connect(self._on_read)
        worker.failed.connect(self._on_read_failed)
        self._worker = worker
        worker.start()

    @staticmethod
    def _read_lake():
        """Everything that touches the share. Runs on the worker.

        Returns the snapshot, or a ``(reason, message)`` pair for the two
        outcomes that are not failures: a lake that is switched off, and one
        that is misconfigured. Those are messages, not crashes.
        """
        from research_warehouse import queries
        from research_warehouse.store import ResearchStore

        try:
            store = ResearchStore.open()
        except Exception as exc:  # a misconfigured lake is a message
            return ("unavailable", f"Research warehouse unavailable: {exc}")
        if store is None:
            return ("disabled", DISABLED_TEXT)
        return queries.slice_readout(store)

    def _on_read(self, payload: object) -> None:  # pragma: no cover - signal seam
        self.refresh_button.setEnabled(True)
        self._worker = None
        if isinstance(payload, tuple) and len(payload) == 2:
            # Switched off or misconfigured: say so, and keep whatever is shown.
            self.status_label.setText(str(payload[1]))
            return
        # A SUCCESSFUL read is authoritative even when it found nothing - that
        # is a real answer about the lake, unlike a failure.
        self._set_rows(payload.rows)
        if payload.rows:
            self.status_label.setText(
                f"{len(payload.rows)} row(s) from {payload.files} sealed file(s), "
                f"manifest position {payload.manifest_seq}."
            )
        else:
            self.status_label.setText(EMPTY_TEXT)

    def _on_read_failed(self, message: str) -> None:  # pragma: no cover - signal seam
        """State the failure and keep every row already on screen.

        Every failure path here used to call `_set_rows([])`. An unreadable
        lake is not an empty lake, and blanking the table replaces "here is
        what we last saw" with a silent claim that there is nothing there.
        """
        self.refresh_button.setEnabled(True)
        self._worker = None
        self.status_label.setText(
            f"Read failed: {message}. The rows below are the last good read, "
            "not the lake as it is now."
        )

    def shutdown(self) -> None:
        # Bounded: this reader waits on the DAS, which is precisely the read
        # that can take minutes when the share is unwell.
        join_worker(self._worker)
        self._worker = None

    def _set_rows(self, rows) -> None:
        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            for column, (key, _label) in enumerate(COLUMNS):
                self.table.setItem(index, column, QTableWidgetItem(_format(row.get(key))))

    def row_count(self) -> int:
        return self.table.rowCount()


def _format(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:+.2f}"
    return str(value)
