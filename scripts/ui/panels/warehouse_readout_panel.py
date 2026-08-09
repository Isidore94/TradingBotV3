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

        self.caveat_label = QLabel(CAVEAT_TEXT)
        self.caveat_label.setWordWrap(True)
        self.caveat_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.caveat_label)

    # -- data ---------------------------------------------------------------
    def refresh(self) -> None:
        """Read the lake once, on demand. Never called from a paint path."""
        try:
            from research_warehouse import queries
            from research_warehouse.store import ResearchStore

            store = ResearchStore.open()
        except Exception as exc:  # a misconfigured lake is a message, not a crash
            self._set_rows([])
            self.status_label.setText(f"Research warehouse unavailable: {exc}")
            return
        if store is None:
            self._set_rows([])
            self.status_label.setText(DISABLED_TEXT)
            return
        try:
            snapshot = queries.slice_readout(store)
        except Exception as exc:
            self._set_rows([])
            self.status_label.setText(f"Read failed: {exc}")
            return
        self._set_rows(snapshot.rows)
        if snapshot.rows:
            self.status_label.setText(
                f"{len(snapshot.rows)} row(s) from {snapshot.files} sealed file(s), "
                f"manifest position {snapshot.manifest_seq}."
            )
        else:
            self.status_label.setText(EMPTY_TEXT)

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
