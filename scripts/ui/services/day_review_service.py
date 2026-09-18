"""One read for the whole Day Review page — TJ-1 item 3.

Trader, 2026-09-17: *"there's just too much shit in these tabs and it's laggy as
all hell. this should be a simple 'what worked what didn't and what was your
process'."* Decision 0021 answer 12: the lag is on OPENING the tab and CLICKING
an entry.

So the page reads ONCE, on one worker, through :meth:`DayReviewService.read_day`,
and paints only from what that call hands back. Every section is a key of one
payload; a section cannot start a read of its own, because it has nothing to
read with.

Three rules this file keeps:

* **One writer.** The trader's words go through `shared_journal_service()`, the
  process's one Market Journal writer (ground rule 8). `write_entry` and
  `import_daily_forecast` here are forwards, not second writers.
* **Uncertainty is reported, never hidden.** A store that cannot be read costs
  its own section and nothing else, and the payload's `error` names it. A blank
  section with no reason reads as "nothing happened", which is a claim.
* **The index, when there is one.** The big outcome stores come from the
  per-session index (`day_review_index`); a session opened without one is
  streamed, and the post-close build leaves one behind for next time.

No Qt: this is plain Python called from a `QThread`, and there is one thing it
deliberately CANNOT do. The desk's M5 cache accessor
(`alert_center.journal_chart_bars`) mutates the Alert Center's bar cache and arms
a `QTimer.singleShot`; armed from a worker with no event loop it never fires,
which latched `_d1_prefetch_flush_armed` True and killed D1 prefetch for the
session (reviewer, 2026-09-17). So this service holds no bars reader and asks for
none: `read_day` takes `spy_m5_bars` as an INPUT, read by the Qt-thread slot that
starts the worker.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Mapping

_log = logging.getLogger(__name__)

#: What one read hands the page. Declared here so the page and this service
#: cannot drift about what a payload is.
PAYLOAD_KEYS: tuple[str, ...] = (
    "session_date",
    "provisional",
    "story",
    "theses",
    "entries",
    "rejected_that_worked",
    "trades",
    "forecast",
    "spy_m5_bars",
)

#: The benchmark whose tape the page draws. One name, the desk's own. The PAGE
#: reads its bars (Qt thread only); this constant is what it reads them for.
BENCHMARK_SYMBOL = "SPY"

#: How many prior sessions the walk-away read looks back over. The page offers
#: no control for it (the Daily Recap's 1/2/3 picker is gone with the page); the
#: reader's own default is the answer.
DEFAULT_LOOKBACK_SESSIONS = 3


def empty_payload(session_date: str = "") -> dict[str, Any]:
    """A payload with every key present and nothing in it.

    The shape of a first paint, a failed read and a quiet session are the same
    shape on purpose: the page has one `render` and no special cases.
    """
    return {
        "session_date": str(session_date or "")[:10],
        "provisional": False,
        "story": None,
        "theses": [],
        "entries": [],
        "rejected_that_worked": (),
        "trades": [],
        "forecast": {},
        "spy_m5_bars": [],
    }


class DayReviewService:
    """Reads one day for the Day Review page. Writes only through the journal."""

    def __init__(self, journal_service: Any = None) -> None:
        self._journal = journal_service

    # -- seams the host wires ---------------------------------------------
    @property
    def journal(self):
        """The process's one Market Journal service, created on first use."""
        if self._journal is None:
            from ui.services.market_journal_service import shared_journal_service

            self._journal = shared_journal_service()
        return self._journal

    # -- the one read ------------------------------------------------------
    def read_day(
        self,
        session_date: str,
        *,
        lookback_sessions: int = DEFAULT_LOOKBACK_SESSIONS,
        now: datetime | None = None,
        spy_m5_bars: Any = None,
    ) -> dict[str, Any]:
        """Everything the page shows for one session, in one mapping.

        Worker-thread call: it opens the journal ledger, the trade journal and
        the per-session index (or the stores behind it). Each in its own guard,
        so one unreadable store costs one section.

        `spy_m5_bars` are HANDED IN, never read here. The desk's accessor for
        them (`alert_center.journal_chart_bars`) mutates the Alert Center's bar
        cache and arms a `QTimer.singleShot`; armed from a worker that has no
        event loop it never fires, which latched `_d1_prefetch_flush_armed` True
        and killed D1 prefetch for the session (reviewer, 2026-09-17). So the
        Qt-thread slot that starts the read is what calls it - this service has
        no reader and cannot acquire one.
        """
        session = str(session_date or "")[:10]
        payload = empty_payload(session)
        moment = now or datetime.now()
        problems: list[str] = []

        entries: list[dict[str, Any]] = []
        try:
            entries = list(self.journal.entries_about(session))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"the journal entries could not be read: {exc}")
            _log.debug("Day Review entries unreadable.", exc_info=True)
        payload["entries"] = entries
        payload["forecast"] = self._forecast(entries)

        try:
            payload["story"] = self.journal.daily_story(session)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"the story facts could not be built: {exc}")
            _log.debug("Day Review story unreadable.", exc_info=True)

        try:
            payload["theses"] = list(self.journal.theses_for(session))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"the open theses could not be read: {exc}")
            _log.debug("Day Review theses unreadable.", exc_info=True)

        payload["provisional"] = self._provisional(session, moment)
        try:
            recap = self._read_recap(session, lookback_sessions, moment)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"the walk-away tables could not be read: {exc}")
            _log.debug("Day Review walk-away unreadable.", exc_info=True)
        else:
            payload["rejected_that_worked"] = tuple(
                getattr(recap.rejected_that_worked, "rows", ()) or ()
            )
            payload["provisional"] = bool(getattr(recap, "provisional", False))

        try:
            payload["trades"] = self._trades(session)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"the day's trades could not be read: {exc}")
            _log.debug("Day Review trades unreadable.", exc_info=True)

        payload["spy_m5_bars"] = [
            dict(bar) for bar in (spy_m5_bars or ()) if isinstance(bar, Mapping)
        ]
        if problems:
            payload["error"] = " · ".join(problems)
        return payload

    # -- the index ---------------------------------------------------------
    def build_index_for(
        self,
        session_date: str,
        *,
        lookback_sessions: int = DEFAULT_LOOKBACK_SESSIONS,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Build and store the per-session index. The ONE named seam for it.

        Called by the page's post-close tick (once for the session that just
        closed) and by `read_day` for a session opened without one. A failure is
        logged and answered with `None`: the index is derived and rebuildable,
        and the page reads the stores directly when it is absent.
        """
        session = str(session_date or "")[:10]
        if not session:
            return None
        try:
            import daily_recap_reader
            import day_review_index

            sources = daily_recap_reader.RecapSources()
            index = day_review_index.build_index(
                session, lookback_sessions=lookback_sessions, sources=sources, now=now
            )
            # The write refuses a session that has not closed and skips a file
            # whose content has not changed; both are 22-30 MB decisions.
            day_review_index.write_index(index, now=now)
            return index
        except Exception:  # noqa: BLE001 - a cache never costs the page
            _log.debug("The Day Review index could not be built.", exc_info=True)
            return None

    def _read_recap(self, session: str, lookback_sessions: int, now: datetime):
        """`read_session`, through the index when there is a usable one."""
        import daily_recap_reader
        import day_review_index

        index: Mapping[str, Any] | None = None
        sources = daily_recap_reader.RecapSources()
        try:
            stored = day_review_index.read_index(session)
            if stored is not None:
                # The files' own verdict, computed ONCE per open (it stats the
                # four stores and may read an appended tail): an index whose
                # stores were REWRITTEN describes files that are no longer there,
                # while an APPEND outside this index's scope leaves it valid.
                verdict, stamp = day_review_index.stamp_verdict(stored, sources=sources)
                if verdict != "rebuild" and not day_review_index.is_stale(
                    stored, now=now
                ):
                    index = stored
                    if verdict == "moved":
                        # Grew outside the scope: record the new stamp beside the
                        # body so the next open compares sizes instead of reading
                        # the same tail again. Hundreds of bytes, not 22 MB.
                        day_review_index.refresh_stamp(stored, stamp=stamp)
        except Exception:  # noqa: BLE001
            _log.debug("The stored Day Review index was unreadable.", exc_info=True)
            index = None
        if index is None:
            index = self.build_index_for(
                session, lookback_sessions=lookback_sessions, now=now
            )
        return daily_recap_reader.read_session(
            session,
            lookback_sessions=lookback_sessions,
            now=now,
            index=index,
        )

    # -- the pieces --------------------------------------------------------
    def _forecast(self, entries) -> dict[str, Any]:
        """The session's pasted brief, or `{}`. Outside commentary, labelled.

        The newest surviving `external_forecast` entry: a second paste supersedes
        the first, so `resolve_entries` has already left one.
        """
        import market_journal

        rows = [
            row
            for row in entries or ()
            if str(row.get("origin") or "") == market_journal.ORIGIN_EXTERNAL_FORECAST
        ]
        if not rows:
            return {}
        row = rows[-1]
        forecast: dict[str, Any] = {
            "entry_id": str(row.get("entry_id") or ""),
            "text": str(row.get("text") or ""),
            "created_at": str(row.get("created_at") or ""),
            "source_model": "",
        }
        try:
            import market_thesis

            sidecar = [
                sheet
                for sheet in market_thesis.read_rows()
                if str(sheet.get("kind") or "") == market_thesis.KIND_FORECAST
                and str(sheet.get("entry_id") or "") == forecast["entry_id"]
            ]
            if sidecar:
                forecast["source_model"] = str(sidecar[-1].get("source_model") or "")
                forecast["target_session"] = str(sidecar[-1].get("target_session") or "")
        except Exception:  # noqa: BLE001 - a missing sidecar is a quieter block
            _log.debug("The forecast sidecar was unreadable.", exc_info=True)
        try:
            import forecast_brief

            forecast["brief"] = forecast_brief.parse(forecast["text"])
        except Exception:  # noqa: BLE001 - the TEXT is the record; the view is not
            _log.debug("The forecast could not be parsed.", exc_info=True)
        return forecast

    def _trades(self, session: str) -> list[dict[str, Any]]:
        """The day's trades, read-only, from the shared trade journal."""
        if not session:
            return []
        from ui.services.journal_feed import trades_on

        return list(trades_on(session))

    @staticmethod
    def _provisional(session: str, now: datetime) -> bool:
        """Has this session NOT closed yet? Unknown reads as provisional.

        A provisional session's numbers cannot be compared with a closed one's,
        so the label is the only thing standing between the two readings.
        """
        if not session:
            return False
        try:
            import market_calendar

            return date.fromisoformat(session) > market_calendar.last_completed_session(now)
        except Exception:  # noqa: BLE001
            return True

    # -- writing (forwards, never a second writer) -------------------------
    def write_entry(self, **kwargs) -> dict[str, Any]:
        """One journal entry, through the process's one writer."""
        return self.journal.write_entry(**kwargs)

    def import_daily_forecast(self, **kwargs) -> dict[str, Any]:
        """One pasted brief, through the process's one writer."""
        return self.journal.import_daily_forecast(**kwargs)


__all__ = [
    "BENCHMARK_SYMBOL",
    "DEFAULT_LOOKBACK_SESSIONS",
    "DayReviewService",
    "PAYLOAD_KEYS",
    "empty_payload",
]
