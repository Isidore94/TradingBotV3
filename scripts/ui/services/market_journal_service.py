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
from datetime import datetime
from typing import Any, Iterable

from PySide6.QtCore import QObject, QThread, Signal

def _forecast_origin() -> str:
    import market_journal

    return market_journal.ORIGIN_EXTERNAL_FORECAST


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
        mentor: Any = None,
        reaffirms: str = "",
    ) -> dict[str, Any]:
        """Write one entry. Returns the row, or a refusal that says why.

        A refusal is returned rather than raised: both hosts show it in a
        status line, and an exception here would turn "you typed nothing" into
        a traceback.

        `mentor` / `reaffirms` are WISHLIST 10J's Trade Mentor fields, passed
        straight through to `build_entry`. The card writes its RAW text through
        HERE and nowhere else: there is one owner of this store (ground rule 8),
        and a prompt-answering surface with its own writer would be a second one.

        Phase 0.31 passes the subject session explicitly. The ledger preserves
        it in `session_date` and records the write day separately.
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
            mentor=mentor,
            reaffirms=reaffirms,
        )
        ok, reason = market_journal.is_publishable(entry)
        if not ok:
            self.statusChanged.emit(reason)
            return {"ok": False, "reason": reason}
        try:
            # The SAME moment the entry was built from. Without this the ledger
            # stamped `event_at` and its own `session_date` from `datetime.now()`
            # while `created_at` said something else, so an entry written with an
            # explicit `now` was filed under one date and stamped with another -
            # and a reader narrowing the ledger by session would miss it
            # entirely. No production caller passed `now` before WISHLIST 10J,
            # which is why it never showed; the Trade Mentor's injected clock is
            # what found it.
            row = self._stream().append(
                entry,
                now=now,
                subject_session_date=session_date,
            )
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

    def entries_about(self, session_date: str) -> list[dict[str, Any]]:
        """Entries about one session, including legacy pre-Phase-0.31 rows.

        TJ-1 item 2: the desk's OWN rows are dropped here, which is the ONE
        filter every trader-facing read of this store inherits (`daily_story`
        and `theses_for` both select through this method). A Trade Mentor answer
        is NOT one of them - the trader wrote every word of it and the desk only
        chose the moment. Nothing is deleted: `is_machine_entry` is asked at
        read time off `origin`, and the rows stay in the append-only ledger.
        """
        import market_journal

        wanted = str(session_date or "").strip()
        if not wanted:
            return []
        rows = [
            row
            for row in self.entries_for()
            if market_journal.session_of_entry(row) == wanted
            and not market_journal.is_machine_entry(row)
        ]
        rows.sort(key=lambda row: str(row.get("created_at") or ""))
        return rows

    def daily_story(self, session_date: str, *, index_bars=None):
        """WS-10D item 1: the session's story. Worker-thread call - it reads files.

        The measured part comes from the DURABLE daily bars already on disk;
        nothing here fetches, and a benchmark with no cached file is reported
        unmeasured rather than filled in.
        """
        import market_story

        entries = self.entries_about(session_date)
        try:
            digests = self.chart_digests()
        except Exception:  # noqa: BLE001 - a missing capture store is a quieter story
            digests = {}
        context = self.day_context(session_date)
        row = context.get("row") if context.get("measured") else None
        bars = index_bars if index_bars is not None else self._index_bars(market_story.BENCHMARKS)
        return market_story.build_daily_story(
            session_date,
            entries=entries,
            captures=digests,
            context_row=row,
            index_bars=bars,
        )

    def _index_bars(self, symbols) -> dict[str, list[dict[str, Any]]]:
        """Completed daily OHLC for the benchmarks. ONE reader, shared.

        `market_story_rollups.load_index_bars` is the same reader the overnight
        rollup uses, so the desk's Story pane and the weekly pack measure the
        same bars. It lives there because `market_story` is pure by contract
        and the nightly slot must not import Qt.
        """
        try:
            from market_story_rollups import load_index_bars

            return load_index_bars(symbols)
        except Exception:  # noqa: BLE001 - no bars is an unmeasured cell, never a broken story
            logging.debug("Benchmark daily bars unreadable.", exc_info=True)
            return {}

    # -- theses -----------------------------------------------------------
    def theses_for(self, session_date: str = "") -> list[dict[str, Any]]:
        """WS-10D item 2: one row per entry of that session, trader edits on top.

        A stored row WINS over a fresh extraction: once the trader has written
        their own interpretation, the machine's reading is history and must not
        quietly replace it on the next refresh. Nothing is written here - a
        read that writes is how a store grows rows nobody asked for.
        """
        import market_thesis

        entries = self.entries_about(session_date) if session_date else self.entries_for()
        try:
            stored = market_thesis.current_theses(market_thesis.read_rows())
        except Exception:  # noqa: BLE001
            logging.debug("Market theses unreadable.", exc_info=True)
            stored = []
        by_entry = {str(row.get("entry_id") or ""): row for row in stored}
        out: list[dict[str, Any]] = []
        for entry in entries:
            entry_id = str(entry.get("entry_id") or "")
            if str(entry.get("origin") or "") == _forecast_origin():
                continue
            existing = by_entry.get(entry_id)
            if existing is not None:
                out.append(dict(existing))
                continue
            out.append(market_thesis.draft_row(market_thesis.extract_thesis(entry)))
        return out

    def save_interpretation(
        self, *, entry_id: str, supersedes: str, text: str
    ) -> dict[str, Any]:
        """The trader's own reading, as a superseding row. Never an edit.

        The draft is recorded first when it has never been written, so the row
        this one SUPERSEDES is really on disk: a superseding row naming
        nothing would hide the machine's reading instead of correcting it.
        """
        import market_thesis

        body = str(text or "").strip()
        if not body:
            return {"ok": False, "reason": "an empty interpretation is not a correction"}
        try:
            known = {
                str(row.get("thesis_id") or "") for row in market_thesis.read_rows()
            }
            if supersedes and supersedes not in known:
                entry = next(
                    (
                        row
                        for row in self.entries_for()
                        if str(row.get("entry_id") or "") == str(entry_id)
                    ),
                    None,
                )
                if entry is not None:
                    market_thesis.record_draft(market_thesis.extract_thesis(entry))
            row = market_thesis.record_interpretation(
                entry_id=str(entry_id), supersedes=str(supersedes), text=body
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("Thesis interpretation not saved: %s", exc)
            self.statusChanged.emit(f"interpretation NOT saved: {exc}")
            return {"ok": False, "reason": str(exc)}
        self.statusChanged.emit("interpretation saved; the original draft is untouched")
        return {"ok": True, "row": row}

    # -- the imported daily forecast (TJ-1 item 5; WISHLIST 10K before it) --
    def import_daily_forecast(
        self,
        *,
        text: str,
        target_session: str = "",
        source_model: str = "",
        created_at_claimed: str = "",
        target_week: str = "",
        scenarios: Iterable[str] = (),
        links: Iterable[str] = (),
        now: datetime | None = None,
        theses_path=None,
    ) -> dict[str, Any]:
        """Paste someone else's brief in, whole, FILED AGAINST ONE SESSION.

        The trader's scheduled ChatGPT prompt produces one brief a day ("that's
        where I paste the output from my scheduled chatgpt prompt", 2026-09-17),
        so the question the store has to answer is "which DAY is this about" -
        not which week. `target_session` is that answer and it comes from the
        caller (the brief's own first heading, or the page's session), never
        from the paste moment: a brief pasted at 21:00 for today is about today.

        Two writes and one order: the ENTRY first (the text is the thing worth
        keeping), then the sidecar that records where it came from. A sidecar
        failure is reported and never costs the import - the evidence store may
        not cost the event it records.

        A SECOND paste for the same session supersedes the first, so the page's
        External forecast block shows one brief rather than two. Both rows stay
        on disk; superseding is a read-side decision (`resolve_entries`).

        `created_at_claimed` stays `unknown` when nobody supplied it. Filling it
        from `imported_at` would turn "a forecast written at some unknown time"
        into "a forecast written at the moment it was pasted", which is a claim
        about what was knowable when, and it would be false.
        """
        import market_journal
        import market_thesis

        session = str(target_session or "").strip() or market_journal.session_date_for(now)
        written = self.write_entry(
            text=text,
            session_date=session,
            timeframe=market_journal.TIMEFRAME_D1,
            origin=market_journal.ORIGIN_EXTERNAL_FORECAST,
            now=now,
            supersedes=self._current_forecast_entry_id(session),
        )
        if not written.get("ok"):
            return written
        entry = written["entry"]
        try:
            sidecar = market_thesis.record_forecast(
                entry_id=str(entry.get("entry_id") or ""),
                text=str(entry.get("text") or ""),
                source_model=source_model,
                created_at_claimed=created_at_claimed,
                target_week=target_week,
                target_session=session,
                scenarios=scenarios,
                links=links,
                session_date=str(session or entry.get("session_date") or ""),
                path=theses_path,
                now=now,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("Forecast sidecar not written: %s", exc)
            self.statusChanged.emit(f"forecast saved; its source row was NOT: {exc}")
            return {"ok": True, "entry": entry, "forecast": None, "sidecar_error": str(exc)}
        self.statusChanged.emit(
            f"daily forecast imported as outside commentary for {session}"
        )
        return {"ok": True, "entry": entry, "forecast": sidecar}

    def _current_forecast_entry_id(self, session_date: str) -> str:
        """The forecast entry this session already has, if any. Read-only.

        Quiet on failure: an unreadable store means "no earlier forecast", which
        writes a new independent row - the wrong answer is a second row on the
        page, never a lost paste.
        """
        import market_journal

        try:
            rows = [
                row
                for row in self.entries_about(session_date)
                if str(row.get("origin") or "") == market_journal.ORIGIN_EXTERNAL_FORECAST
            ]
        except Exception:  # noqa: BLE001
            logging.debug("Earlier forecasts unreadable.", exc_info=True)
            return ""
        if not rows:
            return ""
        return str(rows[-1].get("entry_id") or "")

    def import_weekly_forecast(
        self,
        *,
        text: str,
        source_model: str = "",
        created_at_claimed: str = "",
        target_week: str = "",
        scenarios: Iterable[str] = (),
        links: Iterable[str] = (),
        session_date: str = "",
        now: datetime | None = None,
        theses_path=None,
    ) -> dict[str, Any]:
        """DEPRECATED (TJ-1 item 5): use `import_daily_forecast`.

        The weekly name survives one release as a thin alias so a caller written
        before the rename is not broken by it; the next packet deletes it. It
        files the text against `session_date` exactly as the daily method files
        it against `target_session`, and it still passes `target_week` through
        for the rows that have one.
        """
        return self.import_daily_forecast(
            text=text,
            target_session=session_date,
            source_model=source_model,
            created_at_claimed=created_at_claimed,
            target_week=target_week,
            scenarios=scenarios,
            links=links,
            now=now,
            theses_path=theses_path,
        )

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
