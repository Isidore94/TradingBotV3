"""The one writer both Market Journal surfaces go through — R10.H.

Two surfaces, one store. The Desk "Journal" tab writes a note while the tape is
moving; the left-nav "Market Journal" page writes the sit-down review and reads
everything back. Neither owns the store — this does — so an entry means the
same thing whichever one produced it, and there is exactly one place where a
write can go wrong.

Reads and writes are cheap (a small JSONL through the month-segmented ledger),
but "cheap" is not "free" and this is a Qt process: `entries_for` is called from
a worker in both hosts, never from a paint path (ground rule 9).
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Iterable, Mapping

from PySide6.QtCore import QObject, QThread, Signal


class _CaptureWorker(QThread):
    """Builds and stores one chart capture off the GUI thread.

    The bars are already trimmed and copied by the caller, so this thread
    touches nothing another thread owns. It writes two files and emits what
    happened; it never raises into Qt.
    """

    done = Signal(dict)

    def __init__(self, payload: dict, parent=None) -> None:
        super().__init__(parent)
        self._payload = payload

    def run(self) -> None:  # pragma: no cover - exercised through its seam
        import market_journal_capture

        try:
            capture = market_journal_capture.build_capture(**self._payload)
            result = market_journal_capture.record_capture(capture)
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "reason": str(exc)}
        result.setdefault("entry_id", str(self._payload.get("entry_id") or ""))
        self.done.emit(result)


class MarketJournalService(QObject):
    """Owns `market_journal.jsonl`. One writer, per ground rule 8."""

    #: Emitted after a successful write, so both surfaces refresh from the
    #: store rather than from each other.
    entryWritten = Signal(dict)
    statusChanged = Signal(str)
    #: Emitted after a chart capture is stored (or refused), so a page showing
    #: an entry can start drawing the tape it was written against.
    chartCaptured = Signal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._ledger = None
        self._capture_workers: list[_CaptureWorker] = []

    # -- store ------------------------------------------------------------
    def _stream(self):
        if self._ledger is None:
            from evidence_ledger import EvidenceLedger
            from market_journal import SCHEMA_MARKET_JOURNAL_ENTRY, STREAM

            self._ledger = EvidenceLedger(
                stream=STREAM, schema=SCHEMA_MARKET_JOURNAL_ENTRY
            )
        return self._ledger

    # -- writing ----------------------------------------------------------
    def write_entry(
        self,
        *,
        text: str,
        session_date: str,
        timeframe: str = "",
        symbols: Iterable[str] = (),
        origin: str = "",
        now: datetime | None = None,
        supersedes: str = "",
    ) -> dict[str, Any]:
        """Write one entry. Returns the row, or a refusal that says why.

        A refusal is returned rather than raised: both hosts show it in a
        status line, and an exception here would turn "you typed nothing" into
        a traceback.
        """
        import market_journal

        entry = market_journal.build_entry(
            text=text,
            session_date=session_date,
            timeframe=timeframe or market_journal.TIMEFRAME_M5,
            symbols=symbols,
            origin=origin or market_journal.ORIGIN_DESK_TAB,
            now=now,
            supersedes=supersedes,
        )
        ok, reason = market_journal.is_publishable(entry)
        if not ok:
            self.statusChanged.emit(reason)
            return {"ok": False, "reason": reason}
        try:
            row = self._stream().append(entry)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Market journal entry not written: %s", exc)
            self.statusChanged.emit(f"entry NOT saved: {exc}")
            # Said plainly. A capture that did not reach disk must never look
            # like one that did - the trader would believe the record holds a
            # thought it does not.
            return {"ok": False, "reason": str(exc)}
        self.statusChanged.emit(
            "entry saved for "
            + str(row.get("session_date") or "")
            + (" (written after the session)" if row.get("written_after_the_session") else "")
        )
        self.entryWritten.emit(dict(row))
        return {"ok": True, "entry": row}

    # -- reading ----------------------------------------------------------
    def entries_for(self, session_date: str = "") -> list[dict[str, Any]]:
        """Current entries, superseded ones resolved away. Worker-thread call."""
        import market_journal

        try:
            result = self._stream().read()
        except Exception:
            logging.debug("Market journal unreadable.", exc_info=True)
            return []
        rows = market_journal.resolve_entries(result.rows)
        if session_date:
            rows = [row for row in rows if str(row.get("session_date") or "") == session_date]
        return rows

    # -- chart captures ---------------------------------------------------
    def capture_charts(
        self,
        *,
        entry_id: str,
        symbol: str = "",
        m5_bars=None,
        d1_bars=None,
        benchmark_m5=None,
        benchmark_d1=None,
        reason: str = "",
        note: str = "",
    ) -> bool:
        """Store what the charts looked like, for an entry already on disk.

        Called AFTER the entry is written, never before: a note must never wait
        on a chart, and an entry with no capture is honestly chartless while an
        entry that was never saved is a lost thought.

        The bars are trimmed and copied HERE, on the caller's thread, because
        they come from caches other threads keep writing to; the digest and the
        two file writes go to a worker (ground rule 9).
        """
        import market_journal_capture as capture_mod

        entry_id = str(entry_id or "").strip()
        if not entry_id:
            return False
        payload = {
            "entry_id": entry_id,
            "symbol": str(symbol or "").strip().upper(),
            "reason": reason or capture_mod.REASON_ENTRY,
            "note": str(note or ""),
            "m5_bars": capture_mod.trim_bars(m5_bars, capture_mod.M5_BAR_LIMIT),
            "d1_bars": capture_mod.trim_bars(d1_bars, capture_mod.D1_BAR_LIMIT),
            "benchmark_m5": capture_mod.trim_bars(benchmark_m5, capture_mod.M5_BAR_LIMIT),
            "benchmark_d1": capture_mod.trim_bars(benchmark_d1, capture_mod.D1_BAR_LIMIT),
        }
        worker = _CaptureWorker(payload, self)
        worker.done.connect(self._on_capture_done)
        worker.finished.connect(lambda w=worker: self._release_capture_worker(w))
        self._capture_workers.append(worker)
        worker.start()
        return True

    def _on_capture_done(self, result: dict) -> None:
        if not result.get("ok"):
            # Said, but never as an error the trader has to act on: the note
            # itself is safe on disk and this is the picture beside it.
            logging.info("Journal chart capture skipped: %s", result.get("reason"))
        self.chartCaptured.emit(dict(result))

    def _release_capture_worker(self, worker) -> None:
        try:
            self._capture_workers.remove(worker)
        except ValueError:
            pass
        worker.deleteLater()

    def chart_capture(self, entry_id: str) -> dict[str, Any] | None:
        """The stored bars for one entry. Worker-thread call - it reads a file."""
        import market_journal_capture

        return market_journal_capture.load_capture(entry_id)

    def chart_digests(self) -> dict[str, dict[str, Any]]:
        """Every capture's short text digest, keyed by entry id."""
        import market_journal_capture

        return market_journal_capture.digests_by_entry()

    def wait_for_captures(self, msecs: int = 4000) -> None:
        """Let in-flight captures finish. Called on shutdown, never in a loop."""
        for worker in list(self._capture_workers):
            try:
                worker.wait(int(msecs))
            except RuntimeError:
                pass

    def sessions_with_entries(self) -> list[str]:
        return sorted({str(row.get("session_date") or "") for row in self.entries_for() if row.get("session_date")})

    def regime_timeline(self, *, limit: int = 60) -> dict[str, Any]:
        """R10.G's shifts plus the auto-vs-manual agreement rate.

        Read-only over a store this service does not own; a missing store is a
        quieter page, never an error.
        """
        import market_journal

        try:
            from evidence_ledger import EvidenceLedger
            from market_context_ledger import SCHEMA_MARKET_REGIME_SHIFT, STREAM_REGIME

            rows = list(
                EvidenceLedger(
                    stream=STREAM_REGIME, schema=SCHEMA_MARKET_REGIME_SHIFT
                ).read().rows
            )
        except Exception:
            logging.debug("Regime shift stream unreadable.", exc_info=True)
            rows = []
        rows.sort(key=lambda row: str(row.get("event_at") or ""))
        return {
            "shifts": rows[-int(limit):],
            "agreement": market_journal.agreement_rate(rows),
        }

    def day_context(self, session_date: str) -> dict[str, Any]:
        """R10.G's machine-side row for one session, if it exists.

        Absence is reported as absence. A session the desk never measured has
        no row, and inventing one here would defeat the point of never
        fabricating it there.
        """
        try:
            from evidence_ledger import EvidenceLedger
            from market_context_ledger import SCHEMA_DAILY_MARKET_CONTEXT, STREAM_CONTEXT

            rows = [
                row
                for row in EvidenceLedger(
                    stream=STREAM_CONTEXT, schema=SCHEMA_DAILY_MARKET_CONTEXT
                ).read().rows
                if str(row.get("session_date") or "") == session_date
            ]
        except Exception:
            logging.debug("Daily context stream unreadable.", exc_info=True)
            return {"measured": False, "reason": "the daily context store could not be read"}
        if not rows:
            return {
                "measured": False,
                "reason": f"no daily context row exists for {session_date}; the desk did not measure it",
            }
        return {"measured": True, "row": rows[-1]}


_SHARED: MarketJournalService | None = None


def shared_journal_service() -> MarketJournalService:
    """The process's one journal service.

    Ground rule 8 says one component owns each mutable shared export, and the
    docstring at the top of this file has claimed "one writer" since R10.H -
    but the Desk tab built its own instance, so there were two. Writes still
    landed in the same file (the ledger append is atomic per line), and what
    was actually lost was the SIGNAL: a note typed on the desk never told the
    left-nav page to refresh. One instance, created on first use from the GUI
    thread, fixes both the claim and the symptom.
    """
    global _SHARED
    if _SHARED is None:
        _SHARED = MarketJournalService()
    return _SHARED
