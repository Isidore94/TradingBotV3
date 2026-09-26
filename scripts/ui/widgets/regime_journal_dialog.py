"""The trader's regime journal on the Mentor (S16 item 1): format only.

The dialog shows the lane it is handed (read on a worker) and emits
``segmentConfirmed`` when the trader clicks. It writes nothing itself; the card
files the segment off the Qt thread. A past-regime prefill is written only after
its Confirm is clicked.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping

from PySide6.QtCore import QDate, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import structural_regime as regimes


def _qdate(text: str) -> QDate:
    try:
        value = date.fromisoformat(str(text)[:10])
    except ValueError:
        value = date.today()
    return QDate(value.year, value.month, value.day)


def _iso(widget: QDateEdit) -> str:
    return widget.date().toString("yyyy-MM-dd")


def current_line(lane: Mapping[str, Any] | None) -> str:
    """One line for the regime in force, or the unknown / loading line."""
    if not isinstance(lane, Mapping) or not lane.get("loaded"):
        return "Reading your regime journal..."
    current = lane.get("current")
    if not isinstance(current, Mapping) or not current:
        return "Regime: unknown. You have not typed one yet."
    note = str(current.get("structure_note") or "").strip()
    text = (
        f"Regime: {regimes.label(current.get('regime'))}, day {current.get('day_count')} "
        f"(since {str(current.get('start_date'))[:10]})."
    )
    return f"{text} {note}" if note else text


class RegimeJournalDialog(QDialog):
    """Say the regime changed, and confirm the three past regimes one click each."""

    #: dict(start_date, regime, structure_note[, prefill]) - the trader clicked.
    segmentConfirmed = Signal(dict)

    def __init__(self, parent=None, *, today: date | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("RegimeJournalDialog")
        self.setWindowTitle("Market regime")
        self.setModal(False)
        self._today = today or date.today()
        self._lane: Mapping[str, Any] | None = None
        self._prefill_rows: dict[str, dict[str, Any]] = {}

        self.current_label = QLabel(current_line(None), self)
        self.current_label.setWordWrap(True)

        change = QGroupBox("The regime changed", self)
        grid = QGridLayout(change)
        self.regime_combo = QComboBox(change)
        self.regime_combo.addItem("-", "")
        for name in regimes.VOCABULARY:
            self.regime_combo.addItem(regimes.label(name), name)
        self.start_edit = QDateEdit(_qdate(self._today.isoformat()), change)
        self.start_edit.setCalendarPopup(True)
        self.start_edit.setDisplayFormat("yyyy-MM-dd")
        self.note_edit = QLineEdit(change)
        self.note_edit.setPlaceholderText("Structure, e.g. weekly HH, daily LH/LL channel")
        self.save_button = QPushButton("Save regime", change)
        self.save_button.clicked.connect(self._save_change)
        grid.addWidget(QLabel("Regime", change), 0, 0)
        grid.addWidget(self.regime_combo, 0, 1)
        grid.addWidget(QLabel("Started", change), 1, 0)
        grid.addWidget(self.start_edit, 1, 1)
        grid.addWidget(QLabel("Note", change), 2, 0)
        grid.addWidget(self.note_edit, 2, 1)
        grid.addWidget(self.save_button, 3, 1)

        self.past_box = QGroupBox("Past regimes you told me - confirm each one", self)
        self._past_layout = QVBoxLayout(self.past_box)
        self.past_box.setVisible(False)

        self.status_label = QLabel("", self)
        self.status_label.setObjectName("MutedLabel")
        self.status_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.current_label)
        layout.addWidget(change)
        layout.addWidget(self.past_box)
        layout.addWidget(self.status_label)

    # -- the lane ------------------------------------------------------------
    def set_lane(self, lane: Mapping[str, Any] | None) -> None:
        """Redraw from a lane already read on a worker."""
        self._lane = lane
        self.current_label.setText(current_line(lane))
        wanted = {}
        if isinstance(lane, Mapping) and lane.get("loaded"):
            wanted = {str(item.get("key")): item for item in lane.get("prefills") or () if isinstance(item, Mapping)}
        for key in [key for key in self._prefill_rows if key not in wanted]:
            row = self._prefill_rows.pop(key)["widget"]
            self._past_layout.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        for key, item in wanted.items():
            if key not in self._prefill_rows:
                self._add_prefill_row(key, item)
        self.past_box.setVisible(bool(self._prefill_rows))

    def _add_prefill_row(self, key: str, item: Mapping[str, Any]) -> None:
        row = QWidget(self.past_box)
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        text = QLabel(f"{item.get('label')}: {regimes.label(item.get('regime'))}", row)
        start = QDateEdit(_qdate(str(item.get("start_date"))), row)
        start.setCalendarPopup(True)
        start.setDisplayFormat("yyyy-MM-dd")
        note = QLineEdit(str(item.get("structure_note") or ""), row)
        confirm = QPushButton("Confirm", row)
        confirm.clicked.connect(lambda _checked=False, name=key: self._confirm_prefill(name))
        line.addWidget(text)
        line.addWidget(start)
        line.addWidget(note, 1)
        line.addWidget(confirm)
        self._past_layout.addWidget(row)
        self._prefill_rows[key] = {
            "widget": row, "item": dict(item), "start": start, "note": note, "confirm": confirm,
        }

    # -- the trader's clicks -----------------------------------------------
    def prefill_confirm_button(self, key: str):
        entry = self._prefill_rows.get(str(key))
        return entry["confirm"] if entry else None

    def prefill_start_edit(self, key: str):
        entry = self._prefill_rows.get(str(key))
        return entry["start"] if entry else None

    def _confirm_prefill(self, key: str) -> None:
        entry = self._prefill_rows.get(key)
        if entry is None:
            return
        entry["confirm"].setEnabled(False)
        self.status_label.setText("Saving...")
        self.segmentConfirmed.emit(
            {
                "start_date": _iso(entry["start"]),
                "regime": str(entry["item"].get("regime") or ""),
                "structure_note": entry["note"].text().strip(),
                "prefill": key,
            }
        )

    def _save_change(self) -> None:
        regime = str(self.regime_combo.currentData() or "")
        if not regime:
            self.status_label.setText("Pick a regime first.")
            return
        self.save_button.setEnabled(False)
        self.status_label.setText("Saving...")
        self.segmentConfirmed.emit(
            {
                "start_date": _iso(self.start_edit),
                "regime": regime,
                "structure_note": self.note_edit.text().strip(),
            }
        )

    def write_finished(self, ok: bool, message: str, segment: Mapping[str, Any] | None = None) -> None:
        """The card's worker is done: say so, and re-arm the button that was clicked."""
        self.status_label.setText(message)
        self.save_button.setEnabled(True)
        if ok and segment is not None and not segment.get("prefill"):
            self.regime_combo.setCurrentIndex(0)
            self.note_edit.clear()
        key = str((segment or {}).get("prefill") or "")
        if key in self._prefill_rows and not ok:
            self._prefill_rows[key]["confirm"].setEnabled(True)
