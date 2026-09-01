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

COLUMNS = (
    ("canonical_setup_id", "Setup"),
    ("side", "Side"),
    ("recipe_id", "Recipe"),
    ("n_episodes", "Episodes"),
    ("n_occurrences", "Occurrences"),
    # `slice_readout` has always computed these three and this panel dropped
    # them. Symbols and sessions are what say whether a large count is really
    # one name on one day; truncated is a data shortfall that the query counts
    # deliberately so it stays visible rather than being silently dropped.
    ("n_symbols", "Symbols"),
    ("n_sessions", "Sessions"),
    ("n_matured", "Matured"),
    ("n_open", "Open"),
    ("n_truncated", "Truncated"),
    ("n_no_trigger", "No trigger"),
    ("mean_net_r", "Mean net R"),
    ("mean_r_at_s18", "R @ s18"),
    ("mean_r_at_60m", "R @ 60m"),
    ("mean_mfe_r", "Mean MFE R"),
    ("mean_mae_r", "Mean MAE R"),
    # True when every capture mode behind the row is an as-observed one, i.e.
    # nothing was reconstructed. A reader comparing two rows needs to know
    # which of them rests on rebuilt inputs.
    ("as_observed_only", "As observed"),
)

#: The family filter's "everything" entry. `slice_readout(setups=None)` reads
#: every family in the lake; the pinned Phase-6 slice stays the default so the
#: panel opens on the same two rows it always has.
ALL_FAMILIES = "__all__"
SLICE_ONLY = "__slice__"

DISABLED_TEXT = (
    "Research warehouse is not configured. Set a research store directory in "
    "Settings (or TRADINGBOTV3_RESEARCH_DIR) to enable this readout."
)
EMPTY_TEXT = (
    "No outcomes recorded for the selected families yet. Run the warehouse build "
    "job, or switch the family filter to 'All families' - the pinned slice is two "
    "setups and the lake holds more."
)
CAVEAT_TEXT = (
    "EXPLORATORY: raw counts only - no shrinkage, no intervals, no ranking, and nothing here "
    "affects a score or an alert. 'Episodes' is the sample size; rows and occurrences are not. "
    "'Symbols' and 'Sessions' beside them are what say whether a large count is really one name "
    "on one day, and 'As observed' is false when any input behind the row was reconstructed."
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
        # The family filter. It changes WHICH FAMILIES THE NEXT READ ASKS FOR,
        # so selecting one does not touch the lake - `refresh` stays the only
        # thing that does, and it stays explicit (sec 20).
        controls.addWidget(QLabel("Family"))
        self.family_input = QComboBox()
        self.family_input.addItem("Slice setups (pinned)", SLICE_ONLY)
        self.family_input.addItem("All families", ALL_FAMILIES)
        self.family_input.setToolTip(
            "Which families the next Refresh reads. 'Slice setups' is the pinned "
            "Phase-6 vertical slice; 'All families' reads every family present in "
            "the lake. Choosing one does not read anything - press Refresh."
        )
        controls.addWidget(self.family_input)
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
        selection = self.selected_families()
        worker = ReadWorker(lambda: self._read_lake(selection), self)
        worker.finished_with.connect(self._on_read)
        worker.failed.connect(self._on_read_failed)
        self._worker = worker
        worker.start()

    def selected_families(self) -> str:
        """The combo's current choice, read on the Qt thread before the worker
        starts - a worker must never touch a widget."""
        return str(self.family_input.currentData() or SLICE_ONLY)

    @staticmethod
    def _read_lake(selection: str = SLICE_ONLY):
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
        if selection == ALL_FAMILIES:
            return queries.slice_readout(store, setups=None)
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
            families = len({str(row.get("canonical_setup_id") or "") for row in payload.rows})
            self.status_label.setText(
                f"{len(payload.rows)} row(s) across {families} family(ies) from "
                f"{payload.files} sealed file(s), manifest position {payload.manifest_seq}."
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
