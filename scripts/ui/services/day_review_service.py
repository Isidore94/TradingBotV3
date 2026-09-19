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
from datetime import date, datetime, timedelta
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
    "walkaway",
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

#: How many symbols ONE Day Review open may read daily bars for (TJ-11).
#: Measured 2026-09-19: `chart_snapshot.load_d1_bars` costs ~4.6 ms and ~310 KB
#: per symbol, and a live session's scan holds ~1,100 distinct names - reading
#: every one of them would spend ~5 s of the worker and leave ~340 MB in the
#: reader's process-wide cache. The caps are SIZE rules and never rankings: the
#: trader's OWN decisions are read first and in full, then a bounded slice of
#: the earlier calls and a bounded slice of the untouched names, both in name
#: order. A name past a cap is `unmeasured` and SAID to be - every skill cell
#: carries `measured` beside `n` - and is never assumed into a rate.
DAILY_BAR_SYMBOL_CAP = 400
EARLIER_SYMBOL_CAP = 150
UNTOUCHED_SYMBOL_CAP = 150

#: How many daily bars a walk-away measurement can possibly need: ATR(14) at the
#: decision plus the twenty sessions the "lately" window reaches back over plus
#: the five-session horizon. Kept from each read so one open does not hold a
#: symbol's whole history.
DAILY_BAR_TAIL = 60


def _stamped_dates_for(session: str) -> tuple[str, ...]:
    """`session` plus the non-session dates that belong to it.

    The weekend between Friday's close and Monday's open, or a holiday Monday:
    a decision stamped on one of them was made FOR this session
    (`market_calendar.decision_session`). Existing rows are never rewritten, so
    the reader asks for their dates too.
    """
    try:
        import market_calendar

        day = date.fromisoformat(str(session)[:10])
        previous = market_calendar.previous_session(day)
    except Exception:  # noqa: BLE001 - an unanswerable calendar adds no dates
        return (str(session)[:10],)
    out = [day.isoformat()]
    cursor = previous + timedelta(days=1)
    while cursor < day:
        out.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return tuple(out)


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
        "walkaway": None,
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
        recap = None
        try:
            recap = self._read_recap(session, lookback_sessions, moment)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"the walk-away tables could not be read: {exc}")
            _log.debug("Day Review walk-away unreadable.", exc_info=True)
        else:
            rejected = getattr(recap, "rejected_that_worked", None)
            payload["rejected_that_worked"] = tuple(getattr(rejected, "rows", ()) or ())
            payload["provisional"] = bool(getattr(recap, "provisional", False))

        try:
            payload["trades"] = self._trades(session)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"the day's trades could not be read: {exc}")
            _log.debug("Day Review trades unreadable.", exc_info=True)

        # TJ-2B is another projection of the SAME worker payload.  It opens no
        # live desk store and the page never starts a second read for a table.
        try:
            import claimed_picks
            import daily_recap_reader
            import evidence_stats
            import walkaway_day

            recap_sources = daily_recap_reader.RecapSources()
            annotations = daily_recap_reader._read_jsonl("annotations", recap_sources.annotations, "created_at")
            feedback = daily_recap_reader._read_jsonl("pick_feedback", recap_sources.pick_feedback, "ts")
            favorites = daily_recap_reader._read_jsonl("swing_favorites", recap_sources.swing_favorites, "event_at")
            events = daily_recap_reader._read_jsonl("review_events", recap_sources.review_events, "ts")
            def _decisions_for(target: str) -> list[dict[str, Any]]:
                """Every verdict that BELONGS to `target`, mapped forward.

                `daily_recap_reader._decisions` filters `session_date` by exact
                match, and the desk used to stamp an after-close call with New
                York's next calendar date - so Friday evening's 18 D1 calls
                carry a Saturday and would never reach Monday's page. The rows
                are never rewritten: this asks the reader for the non-session
                dates that map onto `target` as well as for `target` itself.
                """
                rows: list[dict[str, Any]] = []
                for stamped in _stamped_dates_for(target):
                    for decision in daily_recap_reader._decisions(
                        stamped, annotations, feedback, favorites, events
                    ):
                        rows.append({
                            "session_date": target,
                            "symbol": decision.symbol, "side": decision.side,
                            "category": decision.category, "verdict": decision.verdict,
                            "source": decision.source, "timeframe": decision.timeframe,
                            "stamp": getattr(decision.observed_at, "isoformat", lambda: "")(),
                            "capture_id": decision.capture_id,
                            # TJ-11 item 5 counts reasons in the table's own
                            # sentence; without this the page would need a
                            # second read of the same store to name one.
                            "reason": decision.reason,
                            # The session this decision BELONGS to, decided here
                            # because this loop is what asked the exact-match
                            # reader for `stamped`. `session_date` on the stored
                            # row is untouched and still means what it always
                            # meant to every other reader on the desk.
                            "decision_session": target,
                        })
                return rows

            decisions = _decisions_for(session)
            earlier_decisions: list[dict[str, Any]] = []
            for earlier in walkaway_day.earlier_sessions(
                session, count=max(0, evidence_stats.LATELY_SESSIONS - 1)
            ):
                earlier_decisions.extend(_decisions_for(earlier))
            claims = claimed_picks.load_rows(recap_sources.claimed_picks)
            preference = daily_recap_reader._read_csv("preference_report", recap_sources.preference_report, "generated_at").rows
            outcomes = daily_recap_reader._read_csv("session_horizon_outcomes", recap_sources.session_horizon_outcomes, "scan_date").rows
            # A preference can match a later trade, so this is intentionally the
            # journal's full read, not the visible session-only trades table.
            from journal_store import JournalStore
            all_trades = list(JournalStore().list_trades())
            stored = {}
            try:
                import day_review_bars
                stored = day_review_bars.read_session_bars(session) or {}
                # A later matched trade is measured against its EXIT session,
                # never against the decision day's tape. Reads are durable and
                # stay on this worker; missing past tapes are backfilled by the
                # page's existing worker door on the next open.
                for trade in all_trades:
                    if str(trade.get("status") or "").lower() != "closed":
                        continue
                    exit_day = str(trade.get("last_closing_leg_at") or trade.get("closed_at") or "")[:10]
                    if exit_day and exit_day != session:
                        exit_bars = day_review_bars.read_session_bars(exit_day)
                        if exit_bars is not None:
                            stored[exit_day] = exit_bars
            except Exception:  # noqa: BLE001
                _log.debug("Walk-away bars unreadable.", exc_info=True)
            # The scan's own rows are the base-rate population (TJ-11 item 4),
            # and they are the SAME read the claim rows already use: the
            # horizon-outcomes store, filtered to this session. The 1.1 GB
            # tracker is never opened by a page.
            lately = set(
                walkaway_day.earlier_sessions(
                    session, count=max(0, evidence_stats.LATELY_SESSIONS - 1)
                )
            ) | {session}
            scan_rows = [
                row for row in outcomes
                if str(row.get("scan_date") or "")[:10] in lately
            ]
            daily = self._daily_bars_for(
                session, decisions, earlier_decisions, claims, scan_rows
            )
            payload["walkaway"] = walkaway_day.build(
                session,
                {
                    "decisions": decisions,
                    "preference": preference,
                    "outcomes": outcomes,
                    "scan_rows": scan_rows,
                    "earlier_decisions": earlier_decisions,
                },
                stored,
                trades=all_trades,
                claims=claims,
                now=moment,
                daily_bars=daily,
            )
            payload["walkaway_backfill_sessions"] = tuple(
                exit_day for trade in all_trades
                if str(trade.get("status") or "").lower() == "closed"
                and (exit_day := str(trade.get("last_closing_leg_at") or trade.get("closed_at") or "")[:10])
                and exit_day != session and exit_day not in stored
                and day_review_bars.session_is_backfillable(exit_day, now=moment)
            )
        except Exception as exc:  # noqa: BLE001
            problems.append(f"the instant walk-away tables could not be built: {exc}")
            _log.debug("Day Review instant walk-away unreadable.", exc_info=True)

        payload["spy_m5_bars"] = [
            dict(bar) for bar in (spy_m5_bars or ()) if isinstance(bar, Mapping)
        ]
        # A closed day's tape is durable and is safe to read from this worker.
        # Today's hand-off remains the Alert Center cache supplied by the Qt slot.
        try:
            import day_review_bars

            if day_review_bars.session_is_closed(session, now=moment):
                stored_bars = day_review_bars.read_session_bars(session)
                if stored_bars is not None:
                    payload["spy_m5_bars"] = [dict(bar) for bar in stored_bars.get(BENCHMARK_SYMBOL, ())]
        except Exception:  # noqa: BLE001 - a missing chart file never costs the read
            _log.debug("Day Review session bars were unreadable.", exc_info=True)
        if problems:
            payload["error"] = " · ".join(problems)
        return payload

    # -- the daily ruler ---------------------------------------------------
    @staticmethod
    def _daily_bars_for(
        session: str,
        decisions,
        earlier_decisions,
        claims,
        scan_rows,
    ) -> dict[str, list[dict[str, Any]]]:
        """Daily bars for the walk-away rulers, read ONCE per symbol.

        `chart_snapshot.load_d1_bars` is the symbol-level DAILY reader the desk
        already uses off the Qt thread: the durable parquet store, memoized on
        the file's mtime, no network and no IB. It is NOT
        `ui/journal_chart_bars.py`, which mutates the Alert Center's bar cache
        and arms a `QTimer.singleShot` that never fires from a worker.

        Two bounds, both stated rather than hidden:

        * at most `DAILY_BAR_SYMBOL_CAP` symbols, in the priority order named on
          that constant - a name past the cap is `unmeasured`, never assumed;
        * only the last `DAILY_BAR_TAIL` bars are KEPT, and a symbol this read
          put into the reader's process-wide cache is dropped from it again, so
          one Day Review open cannot leave hundreds of megabytes of history
          behind. A symbol the desk had already cached is left exactly as it was.
        """
        wanted: list[str] = []
        seen: set[str] = set()

        def _add(symbol: Any, *, budget: list[int] | None = None) -> None:
            name = str(symbol or "").strip().upper()
            if not name or name in seen:
                return
            if budget is not None:
                if budget[0] <= 0:
                    return
                budget[0] -= 1
            seen.add(name)
            wanted.append(name)

        # 1. What the trader did this session, in full: these are the rows the
        #    page is about, and none of them may go unmeasured for a budget.
        for row in decisions or ():
            _add(row.get("symbol"))
        for claim in claims or ():
            if str(claim.get("session_date") or "")[:10] == session:
                _add(claim.get("symbol"))
        # 2. The earlier calls, then 3. the untouched names of this session's
        #    scan - both in NAME order, both bounded, neither chosen by result.
        earlier_budget = [EARLIER_SYMBOL_CAP]
        for row in sorted(
            earlier_decisions or (), key=lambda item: str(item.get("symbol") or "").upper()
        ):
            _add(row.get("symbol"), budget=earlier_budget)
        untouched_budget = [UNTOUCHED_SYMBOL_CAP]
        # The distinct NAMES, sorted - not the rows: the lately window holds
        # ~90,000 horizon rows carrying a few thousand names between them.
        for name in sorted({str(row.get("symbol") or "").upper() for row in scan_rows or ()}):
            _add(name, budget=untouched_budget)

        bars: dict[str, list[dict[str, Any]]] = {}
        try:
            import chart_snapshot
        except Exception:  # noqa: BLE001 - no daily store is `unmeasured`, not an error
            _log.debug("The durable daily store is unavailable.", exc_info=True)
            return bars
        cache = getattr(chart_snapshot, "_daily_bars_cache", None)
        for symbol in wanted[:DAILY_BAR_SYMBOL_CAP]:
            # Decided PER SYMBOL, immediately before the read: a snapshot taken
            # before the loop could not tell an entry this read inserted from
            # one another thread put there while the loop ran, and evicting
            # someone else's cache entry is not this read's business.
            held_before = isinstance(cache, dict) and symbol in cache
            try:
                rows = chart_snapshot.load_d1_bars(symbol) or []
            except Exception:  # noqa: BLE001 - one unreadable name costs one name
                _log.debug("Daily bars unreadable for %s.", symbol, exc_info=True)
                continue
            bars[symbol] = list(rows[-DAILY_BAR_TAIL:])
            if isinstance(cache, dict) and not held_before:
                cache.pop(symbol, None)
        return bars

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
            day_review_index.write_index(index, now=now)
            return index
        except Exception:  # noqa: BLE001 - a cache never costs the page
            _log.debug("The Day Review index could not be built.", exc_info=True)
            return None

    def build_session_bars_for(self, session_date: str, **_kwargs) -> Any:
        """Fetch and persist a closed session's M5 tape after its index exists."""
        import daily_recap_reader
        import day_review_bars

        session = str(session_date or "")[:10]
        if not session or not day_review_bars.session_is_closed(session):
            return None
        names = day_review_bars.decided_symbols(session, daily_recap_reader.RecapSources())
        bars = day_review_bars.fetch_session_bars(names, session)
        return day_review_bars.write_session_bars(session, bars)

    def backfill_session_bars_for(self, session_date: str, **_kwargs) -> Any:
        """The past-session recovery seam; never called for the live session."""
        return self.build_session_bars_for(session_date)

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
