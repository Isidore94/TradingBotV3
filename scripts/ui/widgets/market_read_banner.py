"""Market Prep's morning read on three axes (SPY state, breadth, internals), P2-8.

The read is built on a `ReadWorker` (local files only: the D1 label, the
breadth row and the session tape); the Qt thread only sets the text.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from PySide6.QtWidgets import QLabel

#: What the banner says before its first read lands or when nothing was read.
NO_READ_TEXT = "Market read: not recorded yet."


class MarketReadBanner(QLabel):
    def __init__(self, parent=None, *, loader: Callable[[], Any] | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("MutedLabel")
        self.setWordWrap(True)
        self._loader = loader
        self._worker = None
        self._again = False
        self.setText(NO_READ_TEXT)

    def refresh(self) -> None:
        """Start one read on a worker; a read already running is followed by one more."""
        if self._worker is not None and self._worker.isRunning():
            self._again = True
            return
        try:
            from ui.read_worker import ReadWorker

            worker = ReadWorker(self._read, self)
            worker.finished_with.connect(self.apply)
            worker.failed.connect(lambda message: logging.debug("Market read failed: %s", message))
            worker.finished.connect(self._worker_done)
            self._worker = worker
            worker.start()
        except Exception:  # noqa: BLE001 - a banner never costs the page
            logging.debug("Market read could not start.", exc_info=True)

    def shutdown(self) -> None:
        if self._worker is None:
            return
        try:
            from ui.read_worker import join_worker

            join_worker(self._worker)
        except Exception:  # noqa: BLE001
            pass

    def _read(self):
        if self._loader is not None:
            return self._loader()
        import market_axes

        return market_axes.latest_morning_read()

    def _worker_done(self) -> None:
        if self._again:
            self._again = False
            self.refresh()

    def apply(self, payload: object) -> None:
        """Show the read's one line; the tooltip lists each axis."""
        read = payload if isinstance(payload, dict) else {}
        line = str(read.get("line") or "").strip()
        self.setText(line or NO_READ_TEXT)
        tips = [str(axis.get("text") or "") for axis in read.get("axes") or () if isinstance(axis, dict)]
        tips.append("Internals are left out when they were not recorded. Graded each evening in Day Review.")
        self.setToolTip("\n".join(tip for tip in tips if tip))


__all__ = ["NO_READ_TEXT", "MarketReadBanner"]
