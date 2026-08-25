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

from PySide6.QtCore import QObject, Signal


class MarketJournalService(QObject):
    """Owns `market_journal.jsonl`. One writer, per ground rule 8."""

    #: Emitted after a successful write, so both surfaces refresh from the
    #: store rather than from each other.
    entryWritten = Signal(dict)
    statusChanged = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._ledger = None

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
