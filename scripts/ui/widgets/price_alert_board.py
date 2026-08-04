"""Compact Focus-tab editor for the shared price-alert store."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ui.services.focus_service import FocusService
from ui.services.price_alert_service import PriceAlertService

_COLUMNS = ("Symbol", "Cross up", "Cross down", "▲ armed", "▼ armed", "Last trigger")
_SOFT_SYMBOL_LIMIT = 25


class PriceAlertBoard(QFrame):
    """A simple two-level price board; satellites deliberately stay read-only."""

    def __init__(
        self,
        service: PriceAlertService,
        focus_service: FocusService,
        *,
        read_only: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.service = service
        self.focus_service = focus_service
        self.read_only = bool(read_only)

        title = QLabel("Phone Price Alerts")
        title.setObjectName("SectionTitle")
        hint = QLabel(
            "One optional cross-up and cross-down per ticker. Each side fires once, "
            "pushes urgently to your phone, then stays off until you re-arm it."
        )
        hint.setObjectName("MutedLabel")
        hint.setWordWrap(True)

        self.symbol_input = QComboBox()
        self.symbol_input.setEditable(True)
        self.symbol_input.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.symbol_input.setMinimumWidth(110)
        self.symbol_input.lineEdit().setPlaceholderText("Ticker")
        self.above_input = QLineEdit()
        self.above_input.setPlaceholderText("Cross up")
        self.below_input = QLineEdit()
        self.below_input.setPlaceholderText("Cross down")
        self.save_button = QPushButton("Add / Update")
        self.remove_button = QPushButton("Remove selected")
        self.rearm_button = QPushButton("Re-arm selected")
        self.save_button.clicked.connect(self._save_input)
        self.remove_button.clicked.connect(self._remove_selected)
        self.rearm_button.clicked.connect(self._rearm_selected)
        self.symbol_input.lineEdit().returnPressed.connect(self._save_input)
        self.above_input.returnPressed.connect(self._save_input)
        self.below_input.returnPressed.connect(self._save_input)

        entry_row = QHBoxLayout()
        entry_row.setContentsMargins(0, 0, 0, 0)
        entry_row.setSpacing(6)
        entry_row.addWidget(self.symbol_input)
        entry_row.addWidget(self.above_input)
        entry_row.addWidget(self.below_input)
        entry_row.addWidget(self.save_button)
        entry_row.addWidget(self.remove_button)
        entry_row.addWidget(self.rearm_button)
        entry_row.addStretch(1)

        self.status_label = QLabel("")
        self.status_label.setObjectName("MutedLabel")
        self.status_label.setWordWrap(True)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(len(_COLUMNS) - 1, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._prefill_selected)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addLayout(entry_row)
        layout.addWidget(self.status_label)
        layout.addWidget(self.table, 1)

        if self.read_only:
            for widget in (
                self.symbol_input,
                self.above_input,
                self.below_input,
                self.save_button,
                self.remove_button,
                self.rearm_button,
            ):
                widget.setEnabled(False)

        self.service.entriesChanged.connect(self.refresh)
        self.service.alertTriggered.connect(lambda _payload: self.refresh())
        self.focus_service.focusChanged.connect(self._refresh_symbol_choices)
        self._refresh_symbol_choices()
        self.refresh()

    def _refresh_symbol_choices(self) -> None:
        current = self.symbol_input.currentText().strip()
        focus = self.focus_service.all_focus()
        names = sorted(set((focus.get("long") or []) + (focus.get("short") or [])))
        self.symbol_input.blockSignals(True)
        self.symbol_input.clear()
        self.symbol_input.addItems(names)
        self.symbol_input.setEditText(current)
        self.symbol_input.blockSignals(False)

    def refresh(self) -> None:
        entries = self.service.entries()
        self.table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            history = entry.get("history") or []
            last = ""
            if history:
                event = history[-1]
                last = (
                    f"{event.get('date', '')} {event.get('at', '')} "
                    f"{event.get('side', '')} {event.get('level', '')} @ {event.get('last', '')}"
                ).strip()
            values = (
                str(entry.get("symbol") or ""),
                _level_text(entry.get("above")),
                _level_text(entry.get("below")),
                "Yes" if entry.get("armed_above") else "No",
                "Yes" if entry.get("armed_below") else "No",
                last,
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))

        armed = sum(
            1 for entry in entries if entry.get("armed_above") or entry.get("armed_below")
        )
        if self.read_only:
            text = "Read-only on this satellite. Edit and re-arm price alerts on the main desk."
        elif armed > _SOFT_SYMBOL_LIMIT:
            text = (
                f"{armed} armed tickers — above the {_SOFT_SYMBOL_LIMIT}-ticker soft limit; "
                "minute quote checks may take longer. Nothing was removed."
            )
        else:
            text = f"{len(entries)} ticker(s), {armed} armed. Phone pushes originate on this main desk."
        self.status_label.setText(text)

    def _save_input(self) -> None:
        if self.read_only:
            return
        symbol = self.symbol_input.currentText().strip().upper()
        above = _parse_level(self.above_input.text())
        below = _parse_level(self.below_input.text())
        if not symbol:
            self.status_label.setText("Enter a ticker.")
            self.symbol_input.setFocus()
            return
        if self.above_input.text().strip() and above is None:
            self.status_label.setText("Cross-up must be a positive number.")
            self.above_input.setFocus()
            return
        if self.below_input.text().strip() and below is None:
            self.status_label.setText("Cross-down must be a positive number.")
            self.below_input.setFocus()
            return
        if above is None and below is None:
            self.status_label.setText("Enter at least one valid price level.")
            self.above_input.setFocus()
            return

        entries = self.service.entries()
        existing = next((entry for entry in entries if entry.get("symbol") == symbol), None)
        if existing is None:
            entries.append(
                {
                    "symbol": symbol,
                    "above": above,
                    "below": below,
                    "armed_above": above is not None,
                    "armed_below": below is not None,
                    "note": "",
                    "history": [],
                }
            )
        else:
            for side, level in (("above", above), ("below", below)):
                old_level = existing.get(side)
                existing[side] = level
                if level is None:
                    existing[f"armed_{side}"] = False
                elif old_level != level:
                    existing[f"armed_{side}"] = True
        if self.service.save_entries(entries):
            self.symbol_input.setEditText("")
            self.above_input.clear()
            self.below_input.clear()

    def _selected_symbols(self) -> set[str]:
        return {
            str(self.table.item(index.row(), 0).text()).strip().upper()
            for index in self.table.selectionModel().selectedRows()
            if self.table.item(index.row(), 0) is not None
        }

    def _remove_selected(self) -> None:
        if self.read_only:
            return
        selected = self._selected_symbols()
        if selected:
            self.service.save_entries(
                [entry for entry in self.service.entries() if entry.get("symbol") not in selected]
            )

    def _rearm_selected(self) -> None:
        if self.read_only:
            return
        selected = self._selected_symbols()
        if not selected:
            return
        entries = self.service.entries()
        for entry in entries:
            if entry.get("symbol") in selected:
                entry["armed_above"] = entry.get("above") is not None
                entry["armed_below"] = entry.get("below") is not None
        self.service.save_entries(entries)

    def _prefill_selected(self) -> None:
        if self.read_only:
            return
        rows = self.table.selectionModel().selectedRows()
        if len(rows) != 1:
            return
        row = rows[0].row()
        self.symbol_input.setEditText(self.table.item(row, 0).text())
        self.above_input.setText(self.table.item(row, 1).text())
        self.below_input.setText(self.table.item(row, 2).text())


def _parse_level(text: str) -> float | None:
    cleaned = str(text or "").replace("$", "").replace(",", "").strip()
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if value > 0 else None


def _level_text(value: Any) -> str:
    return "" if value is None else f"{float(value):.2f}"
