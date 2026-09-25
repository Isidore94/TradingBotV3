"""Confirm suggested setups: the Sunday bulk confirm on the Trades tab (P8 P2 B).

A non-modal list of every needs_review / provisional trade with the Mentor's
suggestion and why. Loading and confirming run on a worker thread
(`journal_bulk_confirm`); a failed journal write shows a warning.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from swallowed import note_swallowed

COLUMNS = ("Confirm", "Date", "Symbol", "Side", "P&L", "Setup", "Why")
COL_CHECK, COL_DATE, COL_SYMBOL, COL_SIDE, COL_PNL, COL_SETUP, COL_WHY = range(len(COLUMNS))


def _default_store():
    from ui.services import journal_feed

    return journal_feed._store()


def _load(store_factory: Callable[[], Any]) -> dict[str, Any]:
    import journal_bulk_confirm as bulk
    import trade_mentor_trade_check as check

    store = store_factory()
    return {"rows": bulk.rows_to_review(store), "vocabulary": tuple(check.setup_vocabulary())}


def _confirm(store_factory: Callable[[], Any], choices: list) -> dict[str, Any]:
    import journal_bulk_confirm as bulk

    return bulk.confirm(store_factory(), choices)


def _pnl_text(row) -> str:
    if row.net_pnl is None:
        return "unknown"
    return f"{row.net_pnl:+,.2f} {row.currency}".strip()


class BulkConfirmDialog(QDialog):
    """Tick the suggested setups that are right, fix the rest, press Confirm."""

    #: Emitted on the Qt thread after a successful confirm (the result dict).
    confirmedSetups = Signal(dict)
    _loaded = Signal(object)
    _done = Signal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        store_factory: Callable[[], Any] | None = None,
        threaded: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Confirm suggested setups")
        self.setModal(False)
        self._store_factory = store_factory or _default_store
        self._threaded = bool(threaded)
        self._rows: list = []
        self._loaded.connect(self._on_loaded)
        self._done.connect(self._on_done)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.status_label = QLabel("loading...")
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.setEnabled(False)
        self.confirm_button.setToolTip(
            "Confirms the ticked rows with the setup shown. Unticked rows are left alone."
        )
        self.confirm_button.clicked.connect(self.confirm)
        buttons = QHBoxLayout()
        buttons.addWidget(self.status_label, 1)
        buttons.addWidget(self.confirm_button)
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("Trades whose setup is a machine suggestion. Tick the right ones, fix the setup, press Confirm.")
        )
        layout.addWidget(self.table)
        layout.addLayout(buttons)
        self.resize(900, 600)

    # -- running on a worker ---------------------------------------------------

    def _run(self, work: Callable[[], Any], signal) -> None:
        def body() -> None:
            try:
                result = work()
            except Exception as exc:  # noqa: BLE001 - handed to the Qt thread, shown there
                result = exc
            try:
                signal.emit(result)
            except RuntimeError as exc:  # the dialog was closed while it ran
                note_swallowed("bulk confirm finished after the dialog closed", exc, quiet=True)

        if self._threaded:
            threading.Thread(target=body, name="journal-bulk-confirm", daemon=True).start()
        else:
            body()

    def load(self) -> None:
        self.confirm_button.setEnabled(False)
        self.status_label.setText("loading...")
        factory = self._store_factory
        self._run(lambda: _load(factory), self._loaded)

    def confirm(self) -> None:
        choices = [
            (row, self.setup_box(index).currentData() or "")
            for index, row in enumerate(self._rows)
            if self.table.item(index, COL_CHECK).checkState() == Qt.Checked
        ]
        if not choices:
            self.status_label.setText("nothing ticked")
            return
        self.confirm_button.setEnabled(False)
        self.status_label.setText("confirming...")
        factory = self._store_factory
        self._run(lambda: _confirm(factory, choices), self._done)

    # -- back on the Qt thread ------------------------------------------------

    def _on_loaded(self, result) -> None:
        if isinstance(result, Exception):
            self.status_label.setText(f"could not load trades: {result}")
            return
        self._rows = list(result.get("rows") or ())
        vocabulary = tuple(result.get("vocabulary") or ())
        self.table.setRowCount(len(self._rows))
        for index, row in enumerate(self._rows):
            tick = QTableWidgetItem("")
            tick.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            tick.setCheckState(Qt.Checked if row.checked_by_default else Qt.Unchecked)
            self.table.setItem(index, COL_CHECK, tick)
            for column, text in (
                (COL_DATE, row.trade_date),
                (COL_SYMBOL, row.symbol),
                (COL_SIDE, row.direction),
                (COL_PNL, _pnl_text(row)),
                (COL_WHY, row.evidence or ("no suggestion" if not row.suggestion else "no evidence stored")),
            ):
                self.table.setItem(index, column, QTableWidgetItem(text))
            self.table.setCellWidget(index, COL_SETUP, self._setup_combo(row, vocabulary))
        self.table.resizeColumnsToContents()
        self.status_label.setText(f"{len(self._rows)} trade(s) to review")
        self.confirm_button.setEnabled(bool(self._rows))

    def _setup_combo(self, row, vocabulary) -> QComboBox:
        combo = QComboBox()
        if not row.suggestion:
            combo.addItem("- pick a setup -", "")
        names = [row.suggestion] if row.suggestion else []
        names += [name for name in vocabulary if name not in names]
        for name in names:
            combo.addItem(name, name)
        combo.setCurrentIndex(0)
        return combo

    def setup_box(self, index: int) -> QComboBox:
        return self.table.cellWidget(index, COL_SETUP)

    def _on_done(self, result) -> None:
        self.confirm_button.setEnabled(bool(self._rows))
        if isinstance(result, Exception):
            # A journal write fails LOUDLY.
            self.status_label.setText(f"Confirm FAILED: {result}")
            QMessageBox.warning(self, "Confirm failed", f"The journal write failed: {result}")
            return
        import journal_bulk_confirm as bulk

        done = set(result.get("confirmed_ids") or ())
        for index in reversed(range(len(self._rows))):
            if self._rows[index].trade_id in done:
                self.table.removeRow(index)
                del self._rows[index]
        self.confirm_button.setEnabled(bool(self._rows))
        self.status_label.setText(bulk.summary_text(result))
        self.confirmedSetups.emit(dict(result))
