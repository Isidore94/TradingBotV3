from __future__ import annotations

from collections.abc import Mapping

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


_COLUMNS = (
    ("Scope", "entity_type"),
    ("Name", "label"),
    ("Integrity", "score"),
    ("State", "state"),
    ("Break pressure", "pressure"),
    ("Confidence", "confidence"),
    ("Tests", "test_count"),
    ("Symbols", "symbol_count"),
)


class TechnicalIntegrityDialog(QDialog):
    """Searchable read-only view of the advisory score hierarchy."""

    def __init__(self, snapshot: Mapping | None, parent=None) -> None:
        super().__init__(parent)
        self.snapshot = snapshot if isinstance(snapshot, Mapping) else {}
        self.setWindowTitle("Technical Integrity — Research Advisory")
        self.resize(940, 620)

        title = QLabel("Technical Integrity: are levels earning respect today?")
        title.setObjectName("SectionTitle")
        explanation = QLabel(
            "1 = technical levels are breaking easily; 10 = levels are repeatedly holding. "
            "Break pressure shows the direction of clean failures. Scores cover BounceBot-scanned "
            "symbols only and never change alerts, watchlists, or setup rankings."
        )
        explanation.setWordWrap(True)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter stock, industry, sector, state, or pressure...")
        self.search.textChanged.connect(self._apply_filter)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels([label for label, _key in _COLUMNS])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(False)
        self._populate()
        self.table.setSortingEnabled(True)
        self.table.sortItems(2, Qt.SortOrder.AscendingOrder)
        self.table.horizontalHeader().setStretchLastSection(True)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        controls = QHBoxLayout()
        controls.addStretch(1)
        controls.addWidget(close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(explanation)
        layout.addWidget(self.search)
        layout.addWidget(self.table, 1)
        layout.addLayout(controls)

    def _populate(self) -> None:
        entities = self.snapshot.get("entities") or []
        for row in entities:
            if not isinstance(row, Mapping):
                continue
            row_index = self.table.rowCount()
            self.table.insertRow(row_index)
            for column, (_label, key) in enumerate(_COLUMNS):
                value = row.get(key)
                item = QTableWidgetItem()
                if key in {"score", "test_count", "symbol_count"}:
                    number = float(value or 0.0)
                    item.setData(Qt.ItemDataRole.EditRole, number)
                    if key == "score":
                        item.setText(f"{number:.1f}")
                else:
                    item.setText(str(value or ""))
                self.table.setItem(row_index, column, item)

    def _apply_filter(self, text: str) -> None:
        needle = str(text or "").strip().casefold()
        for row in range(self.table.rowCount()):
            searchable = " ".join(
                self.table.item(row, column).text()
                for column in range(self.table.columnCount())
                if self.table.item(row, column) is not None
            ).casefold()
            self.table.setRowHidden(row, bool(needle and needle not in searchable))
