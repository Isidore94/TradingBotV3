"""Sunday ritual card (B3) on Weekend Prep's Tag week page.

One card: setup tags waiting (the button opens the bulk confirm, which lists
every suggestion before anything is written), the plan review, and exit early /
held losers per setup family. `load()` reads on a `ReadWorker`
(`sunday_ritual.read_card`); the Qt thread only sets the texts that
`sunday_ritual.format_card` built.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from swallowed import note_swallowed

CARD_OBJECT_NAME = "SundayRitualCard"


class SundayRitualCard(QFrame):
    """Tags, plan and exits per family, formatted from one worker read."""

    #: Emitted after the bulk confirm wrote tags, so the host page can re-read.
    tagsConfirmed = Signal(dict)

    def __init__(
        self,
        parent=None,
        *,
        reader: Callable[[], Any] | None = None,
        store_factory: Callable[[], Any] | None = None,
        threaded: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(CARD_OBJECT_NAME)
        self.setFrameShape(QFrame.StyledPanel)
        self._reader = reader
        self._store_factory = store_factory
        self._threaded = bool(threaded)
        self._worker = None
        self._again = False
        self._dialog = None

        title = QLabel("Sunday ritual")
        title.setObjectName("SectionSubtitle")
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.tags_label = QLabel("Setup tags: press Refresh to read.")
        self.tags_label.setWordWrap(True)
        self.confirm_button = QPushButton("Review and confirm setups...")
        self.confirm_button.setToolTip(
            "Opens the list of suggested setups first. Nothing is confirmed until "
            "you tick rows and press Confirm there."
        )
        self.confirm_button.clicked.connect(self.open_bulk_confirm)
        self.plan_label = QLabel("")
        self.plan_label.setWordWrap(True)
        self.families_label = QLabel("")
        self.families_label.setWordWrap(True)

        tags_row = QHBoxLayout()
        tags_row.addWidget(self.tags_label, 1)
        tags_row.addWidget(self.confirm_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        layout.addWidget(title)
        layout.addWidget(self.note)
        layout.addLayout(tags_row)
        layout.addWidget(self.plan_label)
        layout.addWidget(self.families_label)

    # -- reading, on a worker --------------------------------------------------

    def _read(self):
        if self._reader is not None:
            return self._reader()
        import sunday_ritual

        return sunday_ritual.read_card(self._store_factory)

    def load(self) -> None:
        """Start one read; a read already running is followed by one more."""
        if self._worker is not None and self._worker.isRunning():
            self._again = True
            return
        self.note.setText("Reading the journal and the plan...")
        if not self._threaded:
            try:
                self.apply(self._read())
            except Exception as exc:  # noqa: BLE001 - stated on the card
                self._failed(str(exc))
            return
        try:
            from ui.read_worker import ReadWorker

            worker = ReadWorker(self._read, self)
            worker.finished_with.connect(self.apply)
            worker.failed.connect(self._failed)
            worker.finished.connect(self._worker_done)
            self._worker = worker
            worker.start()
        except Exception:  # noqa: BLE001 - the card never costs the page
            logging.debug("Sunday ritual read could not start.", exc_info=True)

    def _worker_done(self) -> None:
        if self._again:
            self._again = False
            self.load()

    def _failed(self, message: str) -> None:
        # Last good text stays; only the note says the read failed.
        self.note.setText(f"The Sunday card could not be read: {message}")

    def shutdown(self) -> None:
        try:
            from ui.read_worker import join_worker

            join_worker(self._worker)
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            note_swallowed("sunday ritual worker join failed at shutdown", exc, quiet=True)

    # -- back on the Qt thread: formatting only ----------------------------------

    def apply(self, payload: object) -> None:
        import sunday_ritual

        texts = sunday_ritual.format_card(payload if isinstance(payload, dict) else {})
        self.note.setText("")
        self.tags_label.setText(texts["tags"])
        self.plan_label.setText(texts["plan"])
        self.families_label.setText(texts["families"])

    # -- the bulk confirm: the trader's click, a preview, then their Confirm ------

    def open_bulk_confirm(self) -> None:
        """Open (or raise) the non-modal bulk confirm; it loads on a worker."""
        from ui.panels.journal.bulk_confirm_dialog import BulkConfirmDialog

        dialog = self._dialog
        if dialog is None:
            dialog = BulkConfirmDialog(
                self, store_factory=self._store_factory, threaded=self._threaded
            )
            dialog.confirmedSetups.connect(self._on_confirmed)
            self._dialog = dialog
        dialog.load()
        dialog.show()
        dialog.raise_()

    def _on_confirmed(self, result: dict) -> None:
        self.tagsConfirmed.emit(dict(result))
        self.load()
