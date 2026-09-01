"""The journal rebuild, off the Qt thread and single-flight.

`JournalStore.rebuild_trades` re-derives every trade from the raw executions and
then re-runs both auto-tag lanes. The second half is the expensive one:
`AutoTagger.load_context_rows` parses the scanner's output files, and
`master_avwap_setup_tracker.json` measured **1.08 GB** on 2026-08-31, beside a
73 MB CSV. All of it ran synchronously behind the Corrections dialog's OK
button and behind "Add execution..." - so accepting a correction froze the desk
for as long as a gigabyte takes to parse.

Nothing about what the rebuild COMPUTES moves here. It is the same call, made
from a worker thread, with three rules the journal already lives by:

* **Single-flight.** A second request while one is running is refused and says
  so, rather than queueing a second gigabyte parse behind the first.
* **A journal write fails LOUDLY.** The worker never swallows: the outcome
  carries `ok` and the reason, and the host shows it. That is the one place the
  evidence-store rule is inverted on purpose (CLAUDE.md, "Evidence, journal and
  statistics").
* **Results land on the GUI thread**, through a signal, because everything the
  host does next - re-enabling buttons, reloading the table - is Qt work.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal


class _RebuildWorker(QThread):
    """One `rebuild_trades()` call. Never raises into Qt."""

    done = Signal(dict)

    def __init__(self, reason: str = "", token: str = "", parent=None) -> None:
        super().__init__(parent)
        self._reason = str(reason or "")
        self._token = str(token or "")

    def run(self) -> None:  # pragma: no cover - exercised through its seam
        from ui.services import journal_feed

        try:
            trades = journal_feed.rebuild_trades()
        except Exception as exc:  # noqa: BLE001
            logging.warning("Journal rebuild failed: %s", exc)
            self.done.emit(
                {
                    "ok": False,
                    "trades": 0,
                    "reason": str(exc),
                    "context": self._reason,
                    "token": self._token,
                }
            )
            return
        self.done.emit(
            {
                "ok": True,
                "trades": int(trades or 0),
                "reason": "",
                "context": self._reason,
                "token": self._token,
            }
        )


class JournalRebuildService(QObject):
    """Owns the one in-flight journal rebuild."""

    #: Emitted when a rebuild actually starts (not when one is refused).
    started = Signal(str)
    #: {"ok": bool, "trades": int, "reason": str, "context": str}
    finished = Signal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._worker: _RebuildWorker | None = None
        # Both journal tabs listen to the one shared service, so a result has
        # to say WHOSE request it answers. Without it, a rebuild the Trades tab
        # asked for would also make the Health tab reload and post a status.
        self._tokens = itertools.count(1)

    def is_running(self) -> bool:
        worker = self._worker
        return worker is not None and worker.isRunning()

    def request(self, reason: str = "", *, blocking: bool = False) -> str:
        """Start a rebuild and return its token. "" when one is already running.

        ``blocking`` runs it inline - for tests and for a headless caller with
        no event loop to deliver the signal into.
        """
        if self.is_running():
            return ""
        token = f"rebuild-{next(self._tokens)}"
        if blocking:
            from ui.services import journal_feed

            self.started.emit(str(reason or ""))
            try:
                trades = journal_feed.rebuild_trades()
            except Exception as exc:  # noqa: BLE001
                self._on_done(
                    {
                        "ok": False,
                        "trades": 0,
                        "reason": str(exc),
                        "context": str(reason or ""),
                        "token": token,
                    }
                )
                return token
            self._on_done(
                {
                    "ok": True,
                    "trades": int(trades or 0),
                    "reason": "",
                    "context": str(reason or ""),
                    "token": token,
                }
            )
            return token
        worker = _RebuildWorker(reason, token, self)
        worker.done.connect(self._on_done)
        worker.finished.connect(self._release_worker)
        self._worker = worker
        self.started.emit(str(reason or ""))
        worker.start()
        return token

    def _on_done(self, result: dict[str, Any]) -> None:
        self.finished.emit(dict(result))

    def _release_worker(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.deleteLater()

    def shutdown(self, msecs: int = 30_000) -> None:
        """Wait for an in-flight rebuild. Generous, because abandoning one
        mid-write is the thing this must never do."""
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.wait(msecs)


_SHARED: JournalRebuildService | None = None


def shared_rebuild_service() -> JournalRebuildService:
    """The process's one rebuild service - which is what makes single-flight
    mean anything across the two tabs that ask for a rebuild."""
    global _SHARED
    if _SHARED is None:
        _SHARED = JournalRebuildService()
    return _SHARED
