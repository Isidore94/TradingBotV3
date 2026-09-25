"""The trader's trading plan, shown read-only (P1-7 7a).

Day Review and Weekend Prep each hold one. `refresh()` reads the plan on a
`ReadWorker` (which also creates the template when the file is missing and
snapshots a changed plan); the Qt thread only sets text. The trader edits the
file in their own editor; this view never writes it.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from PySide6.QtWidgets import QLabel, QPlainTextEdit, QVBoxLayout, QWidget

PLAN_OBJECT_NAME = "TradingPlanView"
LOADING_TEXT = "Reading your trading plan..."


class TradingPlanView(QWidget):
    def __init__(self, parent=None, *, loader: Callable[[], Any] | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(PLAN_OBJECT_NAME)
        self._loader = loader
        self._worker = None
        self._again = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.title = QLabel("My trading plan", self)
        self.note = QLabel("", self)
        self.note.setWordWrap(True)
        self.text = QPlainTextEdit(self)
        self.text.setReadOnly(True)
        self.text.setPlaceholderText(LOADING_TEXT)
        self.text.setMinimumHeight(160)
        layout.addWidget(self.title)
        layout.addWidget(self.note)
        layout.addWidget(self.text)

    def refresh(self) -> None:
        """Start one read; a read already running is followed by one more."""
        if self._worker is not None and self._worker.isRunning():
            self._again = True
            return
        try:
            from ui.read_worker import ReadWorker

            worker = ReadWorker(self._read, self)
            worker.finished_with.connect(self._apply)
            worker.failed.connect(self._failed)
            worker.finished.connect(self._worker_done)
            self._worker = worker
            worker.start()
        except Exception:  # noqa: BLE001 - the plan view never costs the page
            logging.debug("Trading plan read could not start.", exc_info=True)

    def shutdown(self) -> None:
        try:
            from ui.read_worker import join_worker

            join_worker(self._worker)
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass

    def _read(self):
        if self._loader is not None:
            return self._loader()
        import trading_plan

        return trading_plan.read_plan(create=True, snapshot=True)

    def _worker_done(self) -> None:
        if self._again:
            self._again = False
            self.refresh()

    def _failed(self, message: str) -> None:
        # Last good text stays; only the note says the read failed.
        self.note.setText(f"The plan could not be read: {message}")

    def _apply(self, payload: object) -> None:
        result = dict(payload) if isinstance(payload, dict) else {}
        error = str(result.get("error") or "")
        if error:
            self.note.setText(error)
            return
        path = str(result.get("path") or "")
        parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
        missing = list(parsed.get("missing") or ())
        parts = [f"Edit it in any editor: {path}" if path else "Edit it in any editor."]
        if result.get("created"):
            parts.append("A blank plan was made for you.")
        if missing:
            parts.append("Missing headings: " + ", ".join(missing) + ".")
        self.note.setText(" ".join(parts))
        self.text.setPlainText(str(result.get("text") or ""))


__all__ = ["LOADING_TEXT", "PLAN_OBJECT_NAME", "TradingPlanView"]
