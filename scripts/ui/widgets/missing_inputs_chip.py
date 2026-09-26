"""The status-bar "Inputs: N trades missing stop/setup" chip (P8 B4, goal 10).

Reads `journal_missing_inputs.load_chip` on a `ReadWorker`; the Qt thread only
formats. No timer of its own: the host asks for a refresh when the journal
changes (the Trades tab reload, a Mentor save, a Mentor prompt), coalesced by
`SignalCoalescer`. Hidden when no trade in the window misses a stop or setup.
One click asks the host to open the Mentor on the oldest such trade.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QPushButton
from swallowed import note_swallowed


def _default_loader() -> dict[str, Any]:
    import journal_missing_inputs as missing
    from journal_store import JournalStore

    return missing.load_chip(JournalStore())


class MissingInputsChip(QPushButton):
    #: (TradeQuestion) - open the Mentor on this trade.
    openTradeRequested = Signal(object)

    def __init__(self, parent=None, *, loader: Callable[[], Any] | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("MissingInputsChip")
        self.setFlat(True)
        self._loader = loader or _default_loader
        self._result: dict[str, Any] | None = None
        self._worker = None
        self._again = False
        from ui.timer_utils import SignalCoalescer

        self._coalescer = SignalCoalescer(self.refresh, parent=self)
        self.clicked.connect(self._on_click)
        self.setVisible(False)

    # -- public -------------------------------------------------------------
    def request_refresh(self, *_args) -> None:
        """A journal-changed signal: one coalesced re-read per burst."""
        self._coalescer.request()

    def refresh(self) -> None:
        """Start one read on a worker; a read already running is followed by one more."""
        if self._worker is not None and self._worker.isRunning():
            self._again = True
            return
        try:
            from ui.read_worker import ReadWorker

            worker = ReadWorker(self._loader, self)
            worker.finished_with.connect(self.apply)
            worker.failed.connect(self._failed)
            worker.finished.connect(self._worker_done)
            self._worker = worker
            worker.start()
        except Exception:  # noqa: BLE001 - a chip never costs the desk
            logging.debug("Missing-inputs chip read could not start.", exc_info=True)

    def question(self) -> Any:
        """The Mentor question for the oldest trade missing a stop or setup."""
        return (self._result or {}).get("question")

    def apply(self, payload: object) -> None:
        """Show the worker's answer (Qt thread: formatting only)."""
        import journal_missing_inputs as missing

        result = dict(payload) if isinstance(payload, dict) else None
        self._result = result
        text = missing.chip_text(result)
        self.setText(text)
        oldest = missing.oldest_chip_row(result or {})
        counts = (result or {}).get("counts") or {}
        tip = ""
        if oldest:
            tip = (
                f"Trades opened since {(result or {}).get('since')}: "
                f"{int(counts.get('stop') or 0)} without a stop, "
                f"{int(counts.get('setup') or 0)} without a confirmed setup. "
                f"Click to answer the oldest: {oldest['symbol']} "
                f"{str(oldest['opened_at'])[:10]} ({', '.join(oldest['missing'])})."
            )
        self.setToolTip(tip)
        self.setVisible(bool(text))

    def shutdown(self) -> None:
        self._coalescer.cancel()
        worker = self._worker
        if worker is None:
            return
        try:
            from ui.read_worker import join_worker

            join_worker(worker)
        except Exception as exc:  # noqa: BLE001
            note_swallowed("missing-inputs chip worker join failed at shutdown", exc, quiet=True)

    # -- internals ----------------------------------------------------------
    def _on_click(self) -> None:
        question = self.question()
        if question is not None:
            self.openTradeRequested.emit(question)

    def _worker_done(self) -> None:
        if self._again:
            self._again = False
            self.refresh()

    def _failed(self, message: str) -> None:
        logging.debug("Missing-inputs chip read failed: %s", message)
