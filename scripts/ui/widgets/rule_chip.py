"""Today's rule (Day Recap coach), always in front of the trader.

`RuleChip` is the small "Rule: ..." label for the status bar; `banner=True`
makes the prep page's "Today's rule: ... (streak N)" line. Each reads the rule
on a `ReadWorker` at `start()` and again when the market date changes; the Qt
thread only formats. No rule means the widget is hidden.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QLabel

#: How often the Qt thread compares the market date (no file read).
DATE_CHECK_MS = 60_000


class RuleChip(QLabel):
    ruleLoaded = Signal(object)

    def __init__(
        self,
        parent=None,
        *,
        banner: bool = False,
        loader: Callable[[], Any] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("RuleBanner" if banner else "RuleChip")
        self._banner = bool(banner)
        self._loader = loader
        self._clock = clock or (lambda: datetime.now().astimezone())
        self._info: dict[str, Any] | None = None
        self._worker = None
        self._again = False
        self._date = ""
        self._timer: QTimer | None = None
        self.setVisible(False)
        if banner:
            self.setWordWrap(True)
            self.setStyleSheet("font-weight: 600; color: #e3b341; padding: 4px 2px;")
        else:
            self.setStyleSheet("color: #e3b341; font-weight: 600; padding: 0 6px;")

    # -- public -------------------------------------------------------------
    def info(self) -> dict[str, Any] | None:
        return dict(self._info) if isinstance(self._info, dict) else None

    def start(self) -> None:
        """Read now, then re-read whenever the market date changes."""
        self._date = self._market_date()
        self.refresh()
        if self._timer is None:
            self._timer = QTimer(self)
            self._timer.setInterval(DATE_CHECK_MS)
            self._timer.timeout.connect(self._check_date)
            self._timer.start()

    def refresh(self) -> None:
        """Start one read on a worker; a read already running is followed by one more."""
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
        except Exception:  # noqa: BLE001 - a chip never costs the desk
            logging.debug("Rule chip read could not start.", exc_info=True)

    def shutdown(self) -> None:
        if self._timer is not None:
            self._timer.stop()
        worker = self._worker
        if worker is None:
            return
        try:
            from ui.read_worker import join_worker

            join_worker(worker)
        except Exception:  # noqa: BLE001
            pass

    # -- internals ----------------------------------------------------------
    def _market_date(self) -> str:
        try:
            import recap_rule_loop

            return recap_rule_loop.market_date(self._clock()).isoformat()
        except Exception:  # noqa: BLE001
            return ""

    def _check_date(self) -> None:
        today = self._market_date()
        if today and today != self._date:
            self._date = today
            self.refresh()

    def _read(self):
        if self._loader is not None:
            return self._loader()
        import recap_rule_loop

        return recap_rule_loop.load_today_rule(self._clock())

    def _worker_done(self) -> None:
        if self._again:
            self._again = False
            self.refresh()

    def _failed(self, message: str) -> None:
        logging.debug("Rule chip read failed: %s", message)

    def _apply(self, payload: object) -> None:
        import recap_rule_loop

        info = dict(payload) if isinstance(payload, dict) else None
        self._info = info
        if self._banner:
            text = recap_rule_loop.prep_line(info)
        else:
            text = recap_rule_loop.chip_text(info)
        self.setText(text)
        self.setToolTip(recap_rule_loop.chip_tooltip(info))
        self.setVisible(bool(text))
        self.ruleLoaded.emit(info)
