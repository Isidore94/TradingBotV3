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
    "trade_reviews",
    "forecast",
    "spy_m5_bars",
    # TJ-3: where the trader's own words sit on each tape, resolved to a bar
    # index ON THE WORKER. The page draws them and builds none of them.
    "spy_markers",
    "name_charts",
    # How many of those marks the tape could not carry, counted on the worker so
    # the page can SAY it without counting anything on the Qt thread.
    "spy_marker_placements",
    # TJ-10: what the trader SAID the market would do, and what it did. One row
    # per read with its measured verdict, plus the three congruence lines. Both
    # are built ON THIS WORKER and both are written on EVERY path - a payload
    # that omitted them on a failure would make the page's own key check lie.
    "reads",
    "congruence",
    # TJ-4: the two VERIFIED files the night wrote - one day story for this
    # session, and the ONE rolling D1 view. Both are READ here, on the worker,
    # in a few kilobytes. Building either on the desk would be a 14 GB model
    # load in front of the trader, which is what the night window exists for.
    "day_story",
    "story_freshness",
    "d1_view",
    # TJ-12: the six-line report card that HEADS the page. Built LAST, on this
    # worker, from what the payload ended up with - it is another projection of
    # the ONE read, never a second one, and the page formats it without ever
    # calling `day_report_card.build` itself.
    "report_card",
    # TJ-6: the night's ideas for this session, with whatever the trader has
    # already decided about each one. READ here, on the worker, in a few
    # kilobytes - the card never opens a store of its own, and a suggestion is
    # all any of these rows will ever be.
    "ideas",
    # TJ-7: what the trader said about THEMSELVES this session, built on this
    # worker from the entries the payload already holds and formatted into one
    # line here. The page prints it and reads nothing of its own - no builder,
    # no model, no second pass over the journal on the Qt thread.
    "mood",
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
    """`session` plus the non-session dates that belong to it - the ones AFTER.

    The weekend between Friday's close and Monday's open, or a holiday Monday:
    a decision stamped on one of them JUDGED this session's scan and this
    session's close (`market_calendar.decision_session`, reversed by TJ-11F on
    the trader's word, 2026-09-19 - *"a veto on friday night ... should not be
    considered monday since we have new information then"*). Friday owns its
    Saturday and Sunday; Monday owns only itself. Existing rows are never
    rewritten, so the reader asks for their dates too.
    """
    try:
        import market_calendar

        day = date.fromisoformat(str(session)[:10])
        following = market_calendar.next_session(day)
    except Exception:  # noqa: BLE001 - an unanswerable calendar adds no dates
        return (str(session)[:10],)
    out = [day.isoformat()]
    cursor = day + timedelta(days=1)
    while cursor < following:
        out.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return tuple(out)


def _looks_like_a_date(key: Any) -> bool:
    """Is this mapping key a session DATE rather than a symbol?

    `read_day`'s bars mapping carries both, and a name is never spelled
    `2026-09-18`.
    """
    text = str(key or "")
    if len(text) != 10:
        return False
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


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
        "trade_reviews": [],
        "forecast": {},
        "spy_m5_bars": [],
        "spy_markers": (),
        "name_charts": {},
        "spy_marker_placements": {},
        "reads": (),
        "congruence": (),
        "day_story": None,
        "story_freshness": {"state": "missing", "reason": "no saved facts yet"},
        "d1_view": None,
        "report_card": {},
        "ideas": [],
        # TJ-7. Empty until the trader clicks something; the `line` is what the
        # page prints, and it says "no mood recorded yet" rather than nothing.
        "mood": {},
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
        core_unread: list[str] = []
        # Bound HERE so the marker build at the end of this method is safe when
        # the walk-away block below - which is what fills them - raised before it
        # reached them. Both are plain locals of this method; that block binds
        # them exactly as it always did (TJ-3).
        decisions: list[dict[str, Any]] = []
        claims: list[dict[str, Any]] = []
        stored: dict[str, Any] = {}

        entries: list[dict[str, Any]] = []
        try:
            entries = list(self.journal.entries_about(session))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"the journal entries could not be read: {exc}")
            core_unread.append("journal entries")
            _log.debug("Day Review entries unreadable.", exc_info=True)
        payload["entries"] = entries
        payload["forecast"] = self._forecast(entries)
        # TJ-7, on this worker and from the entries already in hand: the
        # session's moods and the ONE line the page prints. A mood is REPORTED -
        # nothing here ranks, scores or acts on one - and a section that cannot
        # be built costs the line, never the day.
        #
        # The ids match the PACK's ids for the same rows, because both come
        # from `day_review_pack.mood_section` and its one `mood_source_id`
        # seam, which derives an id from the ROW and from nothing else
        # (reviewer advisory 6; pinned by
        # `test_tj7_day_review_mood_line.py::test_the_payloads_mood_ids_are_the_packs_own_ids`).
        # The written pack is deliberately NOT read here: a mood typed in the
        # evening arrives AFTER the post-close pack was written, and reading
        # the pack would show the trader a stale line on the very save live
        # gate #151 asks them to make.
        try:
            import day_review_pack

            section = day_review_pack.mood_section(entries)
            payload["mood"] = {
                **section,
                "line": day_review_pack.mood_statement({"mood": section}),
            }
        except Exception as exc:  # noqa: BLE001
            problems.append(f"the session's mood could not be read: {exc}")
            _log.debug("Day Review mood unreadable.", exc_info=True)

        try:
            payload["story"] = self.journal.daily_story(session)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"the story facts could not be built: {exc}")
            core_unread.append("day facts")
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
            core_unread.append("walk-away results")
            _log.debug("Day Review walk-away unreadable.", exc_info=True)
        else:
            rejected = getattr(recap, "rejected_that_worked", None)
            payload["rejected_that_worked"] = tuple(getattr(rejected, "rows", ()) or ())
            payload["provisional"] = bool(getattr(recap, "provisional", False))

        try:
            payload["trades"] = self._trades(session)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"the day's trades could not be read: {exc}")
            core_unread.append("trades")
            _log.debug("Day Review trades unreadable.", exc_info=True)
        # TJ-9E, on this worker and from ONE read of the append-only table: the
        # trader's own words about each exit, and - only where they CONFIRMED
        # it - the three fields behind it. Both keys are PRESENT and EMPTY on
        # every trade row, because a page that has to tell "no note" from "this
        # build did not look" is reading two different absences as one. A
        # provisional draft is deliberately NOT here: it is the machine's
        # reading and the Day Review page shows what the trader recorded.
        #
        # The ONE read is kept as a local and handed to the report card below,
        # so the line that counts the notes counts the ones this payload really
        # opened - one read for the page and the card, never two.
        exit_notes = self._attach_exit_notes(session, payload["trades"], problems)
        if payload["trades"] and exit_notes is None:
            core_unread.append("exit notes")
        if payload["trades"]:
            try:
                from ui.services import journal_feed

                payload["trade_reviews"] = journal_feed.trade_reviews_on(
                    session, payload["trades"], exit_notes
                )
            except Exception as exc:  # noqa: BLE001
                problems.append(f"the trade answers could not be read: {exc}")
                core_unread.append("trade answers")
                payload["trade_reviews"] = [
                    {"trade_id": str(row.get("trade_id") or ""), "status": "unread"}
                    for row in payload["trades"]
                ]

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
                """Every verdict that JUDGED `target`, mapped back onto it.

                `daily_recap_reader._decisions` filters `session_date` by exact
                match, and the desk stamps an after-close call with New York's
                next calendar date - so Friday evening's 18 D1 calls carry a
                Saturday and would never reach Friday's own page. The rows are
                never rewritten: this asks the reader for the non-session dates
                that map onto `target` as well as for `target` itself.
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
                            # The session this decision JUDGED, decided here
                            # because this loop is what asked the exact-match
                            # reader for `stamped`. `session_date` on the stored
                            # row is untouched and still means what it always
                            # meant to every other reader on the desk. The rule
                            # marker travels with the value (TJ-11F): a stored
                            # session with no marker came from the forward rule
                            # and every reader recomputes it instead.
                            "decision_session": target,
                            "decision_session_rule": walkaway_day.DECISION_SESSION_RULE,
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
        # TJ-3, and LAST because it is built against what the payload ENDED UP
        # with: the Qt-thread hand-off for a live session, the durable file for a
        # closed one. Pure arithmetic over rows already read - it opens no store,
        # so a page that draws markers still reads the day exactly once.
        try:
            import day_review_markers

            payload["spy_markers"] = day_review_markers.benchmark_markers(
                payload["spy_m5_bars"],
                entries=entries,
                trades=payload["trades"],
            )
            payload["spy_marker_placements"] = day_review_markers.placement_counts(
                payload["spy_markers"]
            )
            # `stored` holds TWO shapes: this session's SYMBOL -> bars, and a
            # later exit session's DATE -> {symbol: bars} (the walk-away read
            # adds those). Only the first shape is a tape, so a date key can
            # never be read as a name.
            payload["name_charts"] = day_review_markers.name_charts(
                {
                    name: rows for name, rows in stored.items()
                    if isinstance(rows, list) and not _looks_like_a_date(name)
                },
                decisions=decisions,
                trades=payload["trades"],
                claims=[
                    claim for claim in claims
                    if str(claim.get("session_date") or "")[:10] == session
                ],
            )
        except Exception:  # noqa: BLE001 - a marker never costs the day
            _log.debug("Day Review markers could not be built.", exc_info=True)
        # TJ-10, and last for the same reason the markers are: it is measured
        # against what the payload ENDED UP with. Pure arithmetic over rows this
        # read already opened, on this worker - the page formats and computes
        # nothing. A grading failure costs the verdicts and nothing else, and
        # both keys are already present from `empty_payload`, so every path -
        # including this one's failure path - hands the page the same shape.
        try:
            reads, congruence, _grades = self._graded_reads(
                session,
                entries=entries,
                decisions=decisions,
                claims=claims,
                trades=payload["trades"],
                spy_m5_bars=payload["spy_m5_bars"],
                tapes=stored,
                now=moment,
            )
            payload["reads"] = reads
            payload["congruence"] = congruence
        except Exception:  # noqa: BLE001 - a verdict never costs the day
            core_unread.append("market reads")
            _log.debug("Day Review reads could not be graded.", exc_info=True)
        # TJ-4: the night's two verified files, READ on this worker. The page
        # calls no model, no grader and no pack builder - it formats what the
        # night already wrote. A story stamped for another session is not shown:
        # a page that fell back to "the newest story" would print Thursday's
        # reading over Friday's tape.
        payload["day_story"] = self._day_story(session)
        payload["d1_view"] = self._d1_view()
        # TJ-12, and LAST of all because it is a projection of what the payload
        # ENDED UP with. Two file reads belong to this worker and to nowhere
        # else: `prediction_ledger.your_reads` (the tally, as INTEGERS, so the
        # page re-counts nothing) and the AI-job ledger's bounded TAIL. A failed
        # card costs the card and nothing else - the key is already present from
        # `empty_payload`, so every path hands the page the same shape.
        try:
            payload["report_card"] = self._report_card(
                session,
                payload=payload,
                decisions=decisions,
                claims=claims,
                exit_notes=exit_notes,
                now=moment,
            )
        except Exception:  # noqa: BLE001 - a card never costs the day
            core_unread.append("report card")
            _log.debug("The Day Review report card could not be built.", exc_info=True)
        payload["pack_sources_unread"] = tuple(core_unread)
        payload["story_freshness"] = self._story_freshness(session, payload, moment)
        if payload["story_freshness"]["state"] != "current":
            payload["day_story"] = None
        # TJ-6: the night's suggestions, in their own guard - one unreadable
        # store costs one section. A dismissed idea is already gone by the time
        # the rows arrive here; nothing on this page ever writes one.
        try:
            from ai_jobs import improvement_ideas

            payload["ideas"] = [dict(row) for row in improvement_ideas.ideas_for_session(session)]
        except Exception as exc:  # noqa: BLE001
            problems.append(f"the desk's ideas could not be read: {exc}")
            _log.debug("Day Review ideas unreadable.", exc_info=True)
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

    # -- the night's story (read only) -------------------------------------
    @staticmethod
    def _day_story(session: str) -> dict[str, Any] | None:
        """The verified narration for THIS session, or `None`. Reads a file."""
        if not session:
            return None
        try:
            from ai_jobs import day_review_narration

            stored = day_review_narration.read_narration(session)
        except Exception:  # noqa: BLE001 - an unreadable story is a quieter page
            _log.debug("The Day Review story was unreadable.", exc_info=True)
            return None
        if not isinstance(stored, Mapping):
            return None
        if str(stored.get("session_date") or "")[:10] != session:
            return None
        return dict(stored)

    def _story_freshness(
        self, session: str, payload: Mapping[str, Any], now: datetime
    ) -> dict[str, str]:
        """Compare current worker facts, the saved pack, and its verified story."""
        import day_review_pack

        if payload.get("pack_sources_unread"):
            return {"state": "unread", "reason": ", ".join(payload["pack_sources_unread"])}
        try:
            current = self._compose_pack(session, payload, now=now, strict=True)
        except Exception as exc:  # noqa: BLE001
            return {"state": "unread", "reason": f"current facts: {exc}"}
        saved = day_review_pack.read_pack(session)
        if not isinstance(saved, Mapping):
            return {"state": "missing", "reason": "no saved day facts yet"}
        if not saved.get("inputs_hash"):
            return {"state": "unread", "reason": "saved facts have no content stamp"}
        if saved.get("inputs_hash") != current.get("inputs_hash"):
            return {"state": "stale", "reason": "day facts changed since the saved story"}
        story = payload.get("day_story")
        if not isinstance(story, Mapping):
            return {"state": "missing", "reason": "day facts are current; story is waiting"}
        if story.get("inputs_hash") != saved.get("inputs_hash"):
            return {"state": "stale", "reason": "story was written for older facts"}
        return {"state": "current", "reason": "story matches current facts"}

    def _compose_pack(
        self, session: str, data: Mapping[str, Any], *, now: datetime | None = None,
        strict: bool = False,
    ) -> dict[str, Any]:
        """The one pure pack composition used by page freshness and night write."""
        import day_report_card
        import day_review_pack

        reviews = {
            str(row.get("trade_id") or ""): dict(row)
            for row in data.get("trade_reviews") or () if isinstance(row, Mapping)
        }
        trades = [
            {**dict(row), "trade_review": reviews.get(str(row.get("trade_id") or ""), {})}
            for row in data.get("trades") or () if isinstance(row, Mapping)
        ]
        entries = list(data.get("entries") or ())
        environment = self._regime_shifts(session, strict=True) if strict else self._regime_shifts(session)
        d1_label = self._d1_label_for(session, strict=True) if strict else self._d1_label_for(session)
        internals = (
            self._internals_marks(session, entries, strict=True)
            if strict else self._internals_marks(session, entries)
        )
        return day_review_pack.build_pack(
            session,
            entries=entries,
            forecast=data.get("forecast") or {},
            story=data.get("story"),
            environment=environment,
            d1_label=d1_label,
            internals=internals,
            walkaway=data.get("walkaway"),
            reads=data.get("reads") or (),
            congruence=data.get("congruence") or (),
            trades=trades,
            report_card=day_report_card.pack_card(data.get("report_card")),
            now=now,
        )

    @staticmethod
    def _d1_view() -> dict[str, Any] | None:
        """The ONE rolling D1 view, or `None`. Not keyed to a session."""
        try:
            from ai_jobs import day_review_narration

            stored = day_review_narration.read_d1_view()
        except Exception:  # noqa: BLE001
            _log.debug("The rolling D1 view was unreadable.", exc_info=True)
            return None
        return dict(stored) if isinstance(stored, Mapping) else None

    # -- the report card (TJ-12) -------------------------------------------
    @staticmethod
    def _ledger_path():
        """The AI-job ledger, WITHOUT creating the store. ``None`` when absent.

        `ai_jobs.ledger.ledger_path()` defaults to `create=True`, which makes
        the folder; a reader whose honest answer may be "night status unknown"
        must not be the thing that creates the store it is asking about.
        """
        try:
            import ai_jobs.ledger as ledger

            return ledger.ledger_path(create=False)
        except Exception:  # noqa: BLE001 - no store configured is `unknown`
            _log.debug("The AI job ledger path is unresolvable.", exc_info=True)
            return None

    @staticmethod
    def _fills_current_to() -> str:
        """The last session the desk has VERIFIED fill coverage for, or ``""``.

        `trade_mentor_trade_check.fills_current_to` is the ONE owner of that
        question (TJ-9). An absence is NOT a date, so it comes back empty and
        the card says the desk has no verified coverage rather than printing
        today.
        """
        try:
            import trade_mentor_trade_check as check
            from journal_store import JournalStore

            answer = check.fills_current_to(JournalStore())
        except Exception:  # noqa: BLE001 - an unreadable ledger names no date
            _log.debug("Fill coverage unreadable.", exc_info=True)
            return ""
        return answer.isoformat() if answer is not None else ""

    def _report_card(
        self,
        session: str,
        *,
        payload: Mapping[str, Any],
        decisions,
        claims,
        now: datetime,
        exit_notes: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The six lines, built on THIS worker from what the payload holds.

        Everything the card counts is already in the payload; the only stores
        opened here are the read ledger (through its owner, so the counts travel
        as integers) and the AI-job ledger's bounded tail. The page never calls
        `day_report_card.build`.
        """
        import day_report_card
        import prediction_ledger

        tally = None
        try:
            tally = prediction_ledger.your_reads(session)
        except Exception:  # noqa: BLE001 - no graded reads is a SAID absence
            _log.debug("The read tally was unreadable.", exc_info=True)

        # `trade_origin` reads a lane row's own stamp key, and the recap's
        # decision rows carry theirs as `stamp`. Renaming it HERE keeps that
        # module's key list the one authority on when a statement was made.
        decision_lane = [
            {**row, "created_at": row.get("stamp") or row.get("created_at")}
            for row in (decisions or ())
            if isinstance(row, Mapping)
        ]
        loaded = {"decisions": decision_lane, "claims": list(claims or ())}
        story = payload.get("day_story")
        card = day_report_card.build(
            {
                "session": session,
                "walkaway": payload.get("walkaway"),
                "your_reads": tally,
                "congruence": payload.get("congruence") or (),
                "trades": payload.get("trades") or [],
                # WHICH lanes this read opened, declared rather than inferred:
                # an unread lane and an empty one look identical to
                # `trade_origin.planned_state`, and the Process line has to say
                # which doors it could not open. The list is
                # `day_report_card.DESK_ORIGIN_LANES_READ` - ONE constant, so
                # the desk's Mentor lane and this worker cannot disagree.
                "origin_lanes": {
                    name: loaded.get(name, ())
                    if name in day_report_card.DESK_ORIGIN_LANES_READ
                    else ()
                    for name in day_report_card.ORIGIN_LANES
                },
                "origin_lanes_read": day_report_card.DESK_ORIGIN_LANES_READ,
                # TJ-9E, and the SAME read `_attach_exit_notes` made a few
                # lines above - one read for the page and the card. `None` is
                # "nobody opened them", which the line says out loud instead of
                # printing a zero (review 1 blocker 3: `build` never passed
                # this, so the live page said "nobody opened the exit notes" on
                # the payload that had just opened every one of them).
                "exit_notes": exit_notes,
                "freshness": {
                    "session": session,
                    "story_written_at": (
                        str((story or {}).get("generated_at") or "")
                        if isinstance(story, Mapping)
                        else ""
                    ),
                    "fills_current_to": self._fills_current_to(),
                    "reads_graded_through": session if payload.get("reads") else "",
                    "ledger_path": self._ledger_path(),
                },
            }
        )
        return {"session": card.session, "lines": [dict(line) for line in card.lines]}

    # -- the day pack ------------------------------------------------------
    def build_pack_for(
        self,
        session_date: str,
        *,
        payload: Mapping[str, Any] | None = None,
        now: datetime | None = None,
        strict: bool = False,
        root=None,
        **_kwargs,
    ) -> dict[str, Any] | None:
        """Build and store the session's day pack. The ONE named seam for it.

        Called by the page's post-close tick (`_IndexBuildWorker`, on the worker
        thread) AFTER `build_reads_for`, because the pack's `reads` section IS
        what the grader just wrote. The nightly slot reads the file this leaves
        behind and never builds one of its own.

        A failure costs the pack and nothing else: it is derived and
        rebuildable, and the night says "no story yet" rather than narrating a
        half-built day.
        """
        import day_review_pack

        session = str(session_date or "")[:10]
        if not session:
            return None
        try:
            data = (
                dict(payload)
                if isinstance(payload, Mapping)
                else self.read_day(session, now=now)
            )
        except Exception:  # noqa: BLE001 - an unreadable day builds no pack
            _log.debug("The day pack's inputs were unreadable.", exc_info=True)
            return None
        if strict and data.get("pack_sources_unread"):
            return None
        try:
            pack = self._compose_pack(session, data, now=now, strict=strict)
            saved = day_review_pack.read_pack(session, root=root)
            if isinstance(saved, Mapping) and saved.get("inputs_hash") == pack.get("inputs_hash"):
                return dict(saved)
            day_review_pack.write_pack(pack, root=root)
        except Exception:  # noqa: BLE001 - a pack never costs the page
            _log.debug("The day pack could not be built.", exc_info=True)
            return None
        return pack

    @staticmethod
    def _regime_shifts(session: str, *, strict: bool = False) -> list[dict[str, Any]]:
        """The session's own regime-shift rows, oldest first. Read-only."""
        try:
            from evidence_ledger import EvidenceLedger
            from market_context_ledger import SCHEMA_MARKET_REGIME_SHIFT, STREAM_REGIME

            rows = [
                dict(row)
                for row in EvidenceLedger(
                    stream=STREAM_REGIME, schema=SCHEMA_MARKET_REGIME_SHIFT
                ).read().rows
                if str(row.get("session_date") or "")[:10] == session
            ]
        except Exception:  # noqa: BLE001 - strict refresh keeps the prior pack
            if strict:
                raise
            _log.debug("The regime-shift stream was unreadable.", exc_info=True)
            return []
        rows.sort(key=lambda row: str(row.get("event_at") or ""))
        return rows

    @staticmethod
    def _d1_label_for(session: str, *, strict: bool = False) -> str:
        """The desk's own D1 label for the session, or "" (nobody labelled it)."""
        try:
            import d1_environment_store

            label = d1_environment_store.label_for_session(session, BENCHMARK_SYMBOL)
        except Exception:  # noqa: BLE001
            if strict:
                raise
            _log.debug("The desk's D1 label was unreadable.", exc_info=True)
            return ""
        return "" if str(label or "") in ("", "unknown") else str(label)

    @staticmethod
    def _internals_marks(session: str, entries, *, strict: bool = False) -> list[dict[str, Any]]:
        """The open, each Mentor hour and the close - TJ-14A's v2 context each.

        Built through `trade_mentor_context`'s ONE builder from the durable
        tape, so the pack cannot carry a shape the live card never wrote. The
        bars are loaded ONCE per session and the builder cuts them to each
        stamp; a session with no tape has its facts `unmeasured`, never guessed.
        """
        if not session:
            return []
        try:
            import market_calendar
            import trade_mentor_context

            day = date.fromisoformat(session)
            close = market_calendar.session_close(day)
            opening = close.replace(hour=9, minute=30, second=0, microsecond=0)
            moments: list[tuple[str, datetime]] = [("open", opening)]
            for entry in entries or ():
                if not isinstance(entry, Mapping):
                    continue
                mentor = entry.get("mentor")
                if not isinstance(mentor, Mapping) or not mentor:
                    continue
                stamp = str(
                    mentor.get("responded_at") or entry.get("created_at") or ""
                )
                try:
                    moment = datetime.fromisoformat(stamp)
                except ValueError:
                    continue
                if moment.tzinfo is None:
                    continue
                moments.append(("mentor", moment))
            moments.append(("close", close))
            bars = trade_mentor_context.internals_bars_at(session, close)
            marks: list[dict[str, Any]] = []
            for kind, moment in sorted(moments, key=lambda item: item[1]):
                marks.append({
                    "kind": kind,
                    "at": moment.isoformat(),
                    "context": trade_mentor_context.internals_at(session, moment, bars),
                })
            return marks
        except Exception:  # noqa: BLE001 - an unreadable tape costs the internals
            if strict:
                raise
            _log.debug("The day pack's internals could not be built.", exc_info=True)
            return []

    def build_session_bars_for(self, session_date: str, **_kwargs) -> Any:
        """Fetch and persist a closed session's M5 tape after its index exists."""
        import daily_recap_reader
        import day_review_bars

        session = str(session_date or "")[:10]
        if not session or not day_review_bars.session_is_closed(session):
            return None
        if _kwargs.get("reuse_existing"):
            existing = day_review_bars.read_session_bars(session)
            if existing:
                return existing
        names = day_review_bars.decided_symbols(session, daily_recap_reader.RecapSources())
        bars = day_review_bars.fetch_session_bars(names, session)
        return day_review_bars.write_session_bars(session, bars)

    # -- the read grader ---------------------------------------------------
    def build_reads_for(self, session_date: str, **kwargs) -> list[dict[str, Any]] | None:
        """Grade the session's reads and APPEND them to the ledger. One seam.

        The named seam the post-close tick calls (`_IndexBuildWorker`, on the
        worker thread) and the only place a grade is WRITTEN. `read_day` builds
        the same rows to show them and writes nothing: a page open is a read.

        Append-only and idempotent: a read whose current stored verdict already
        says what this pass measured is not written again, and a verdict that
        MOVED is a new row naming the old one (`supersedes`). A failure here
        costs the grades and nothing else.
        """
        import market_read_grades as grader

        session = str(session_date or "")[:10]
        if not session:
            return []
        now = kwargs.get("now") or datetime.now()
        strict = bool(kwargs.get("strict"))
        entries: list[dict[str, Any]] = []
        try:
            entries = list(self.journal.entries_about(session))
        except Exception:  # noqa: BLE001 - an unreadable ledger grades nothing
            _log.debug("The journal could not be read for grading.", exc_info=True)
            return None if strict else []
        decisions, claims = self._decisions_and_claims(session)
        trades: list[dict[str, Any]] = []
        try:
            trades = self._trades(session)
        except Exception:  # noqa: BLE001
            _log.debug("The day's trades were unreadable for grading.", exc_info=True)
        spy_bars: list[dict[str, Any]] = []
        tapes: dict[str, Any] = {}
        try:
            import day_review_bars

            tapes = day_review_bars.read_session_bars(session) or {}
            spy_bars = list(tapes.get(BENCHMARK_SYMBOL) or ())
        except Exception:  # noqa: BLE001 - a missing tape is `unmeasured`
            _log.debug("The session tape was unreadable for grading.", exc_info=True)
        _reads, _lines, grades = self._graded_reads(
            session,
            entries=entries,
            decisions=decisions,
            claims=claims,
            trades=trades,
            spy_m5_bars=spy_bars,
            tapes=tapes,
            now=now,
        )
        stored = grader.current_grades(grader.read_grades(session))
        by_read = {str(row.get("read_id") or ""): row for row in stored}
        fresh: list[dict[str, Any]] = []
        for grade in self._storable_grades(grades):
            previous = by_read.get(str(grade.get("read_id") or ""))
            if previous is None:
                fresh.append(grade)
                continue
            was = str(previous.get("verdict") or "")
            if was == str(grade.get("verdict") or ""):
                continue
            # The same rule the nightly re-grade keeps: a verdict may only move
            # UP, so an absent store can never retire a correct `pending` row.
            if grader.verdict_rank(str(grade.get("verdict") or "")) <= grader.verdict_rank(was):
                continue
            fresh.append({**grade, "supersedes": str(previous.get("grade_id") or "")})
        if fresh:
            grader.append_grades(session, fresh)
        return fresh

    @staticmethod
    def _decisions_and_claims(session: str):
        """The day's verdicts and claims, read the way `read_day` reads them."""
        import claimed_picks
        import daily_recap_reader

        rows: list[dict[str, Any]] = []
        claims: list[dict[str, Any]] = []
        try:
            sources = daily_recap_reader.RecapSources()
            annotations = daily_recap_reader._read_jsonl(
                "annotations", sources.annotations, "created_at"
            )
            feedback = daily_recap_reader._read_jsonl(
                "pick_feedback", sources.pick_feedback, "ts"
            )
            favorites = daily_recap_reader._read_jsonl(
                "swing_favorites", sources.swing_favorites, "event_at"
            )
            events = daily_recap_reader._read_jsonl(
                "review_events", sources.review_events, "ts"
            )
            for stamped in _stamped_dates_for(session):
                for decision in daily_recap_reader._decisions(
                    stamped, annotations, feedback, favorites, events
                ):
                    rows.append({
                        "session_date": session, "symbol": decision.symbol,
                        "side": decision.side, "category": decision.category,
                        "verdict": decision.verdict, "source": decision.source,
                        "timeframe": decision.timeframe,
                        "capture_id": decision.capture_id,
                        "decision_session": session,
                    })
            claims = list(claimed_picks.load_rows(sources.claimed_picks))
        except Exception:  # noqa: BLE001 - an unreadable store names no side
            _log.debug("The day's decisions were unreadable.", exc_info=True)
        return rows, claims

    def _graded_reads(
        self,
        session: str,
        *,
        entries,
        decisions,
        claims,
        trades,
        spy_m5_bars,
        tapes,
        now: datetime,
    ):
        """`(reads, congruence, grades)` for one session. Worker-thread only.

        One row per read with the verdict the BARS give it, each gradable grade
        carrying the point-in-time context snapshot TJ-16 item 1 requires, and
        the three congruence lines beside them. Nothing here writes.
        """
        import market_read_grades as grader

        rows = grader.read_rows(entries, session=session)
        by_entry = {
            str(entry.get("entry_id") or ""): entry for entry in entries or ()
        }
        tape_of: dict[str, list[dict[str, Any]]] = {}
        daily_of: dict[str, list[dict[str, Any]]] = {}

        def tape_for(symbol: str) -> list[dict[str, Any]]:
            if symbol not in tape_of:
                if symbol == BENCHMARK_SYMBOL:
                    tape_of[symbol] = list(spy_m5_bars or ())
                else:
                    rows_for = (tapes or {}).get(symbol)
                    tape_of[symbol] = list(rows_for or ())
            return tape_of[symbol]

        def daily_for(symbol: str) -> list[dict[str, Any]]:
            """The benchmark's daily history, through the ONE shared loader.

            `market_read_grades.daily_bars_for_symbol` is what the nightly
            re-grade uses too. A page and a night that read different stores
            grade different markets - which is exactly what happened before the
            fix round (reviewer, 2026-09-20).
            """
            if symbol not in daily_of:
                try:
                    daily_of[symbol] = list(grader.daily_bars_for_symbol(symbol) or ())
                except Exception:  # noqa: BLE001 - `unmeasured`, never an error
                    _log.debug("Daily bars unreadable for %s.", symbol, exc_info=True)
                    daily_of[symbol] = []
            return daily_of[symbol]

        labels = self._d1_labels()
        prior_grades = self._prior_grades(session)
        internals_bars = self._internals_bars(session, rows)
        # The ONE read per timeframe a congruence line may be compared with: a
        # CLICK outranks an extraction, and contradictory extracted stances name
        # themselves instead of one of them being picked (`select_read`).
        latest_d1, d1_note = grader.select_read(rows, timeframe="D1")
        latest_m5, m5_note = grader.select_read(rows, timeframe="M5")

        reads: list[dict[str, Any]] = []
        grades: list[dict[str, Any]] = []
        for row in rows:
            symbol = str(row.get("benchmark") or BENCHMARK_SYMBOL)
            daily = daily_for(symbol)
            context: dict[str, Any] = {}
            context_gap = ""
            try:
                context = grader.context_for(
                    by_entry.get(str(row.get("entry_id") or "")) or {},
                    row=row,
                    bars=internals_bars,
                    spy_m5_bars=tape_for(BENCHMARK_SYMBOL),
                    prior_daily_bar=self._prior_daily_bar(daily, session),
                    d1_labels=labels,
                    prior_grades=prior_grades,
                    latest_d1_click=latest_d1 if latest_d1 is not row else None,
                    # TJ-7: the session's own entries, so the snapshot can carry
                    # the mood the read was made KNOWING. `context_for` takes
                    # only the one recorded AT OR BEFORE the stamp - the mood
                    # clicked at the close is not context for an 07:02 read.
                    mood_entries=entries or (),
                )
            except Exception as exc:  # noqa: BLE001
                # A CLICK whose snapshot cannot be built is NOT stored this pass
                # (`_storable_grades` drops it) and the row says why, so the next
                # pass tries again. Degrading it to a named absence would make a
                # stated call permanently context-less (lead, 2026-09-20).
                context_gap = f"context_unbuildable: {type(exc).__name__}: {exc}"
                _log.debug("A read's context could not be built.", exc_info=True)
            grade = grader.grade_read(
                row,
                m5_bars=tape_for(symbol),
                daily_bars=daily,
                atr=grader.daily_atr(daily, through=session),
                now=now,
                context=context,
            )
            if context_gap:
                grade["grader_gap"] = context_gap
            grades.append(grade)
            entry = by_entry.get(str(row.get("entry_id") or "")) or {}
            mentor = entry.get("mentor") if isinstance(entry.get("mentor"), Mapping) else {}
            reads.append({
                **{key: value for key, value in row.items() if key != "context"},
                "verdict": grade["verdict"],
                "move_atr": grade["move_atr"],
                "checkpoints": grade["checkpoints"],
                "flat_band_rule": grade["flat_band_rule"],
                "grader_gap": grade["grader_gap"],
                # What the trader SAW, kept visibly apart from what they
                # EXPECTED (TJ-14A's rule, and the packet's item 7).
                "observation": str((mentor or {}).get("observation") or ""),
            })
        label = ""
        try:
            import d1_environment_store

            label = d1_environment_store.label_for_session(session, BENCHMARK_SYMBOL)
        except Exception:  # noqa: BLE001 - an unread label is a missing side
            _log.debug("The desk's D1 label was unreadable.", exc_info=True)
        lines = grader.congruence_lines(
            session=session,
            d1_read=latest_d1,
            # "unknown" is "nobody labelled it", which is a MISSING side and
            # never a label with no direction.
            d1_label="" if label in ("", "unknown") else label,
            decisions=decisions or (),
            claims=claims or (),
            trades=trades or (),
            d1_note=d1_note,
            # Lead decision 6's M5 half: a rest-of-day read belongs with the
            # session's M5 likes, never with its D1 ones.
            m5_read=latest_m5,
            m5_note=m5_note,
        )
        return reads, lines, grades

    @staticmethod
    def _storable_grades(grades) -> list[dict[str, Any]]:
        """The grades this pass may WRITE.

        A clicked grade whose context could not be built is held back rather
        than stored with a named absence: the ledger refuses it anyway
        (`ContextMissingError`), and a row written once can never be given a
        snapshot afterwards. The next pass builds it again.
        """
        import market_read_grades as grader

        keep: list[dict[str, Any]] = []
        for grade in grades or ():
            clicked = str(grade.get("source") or "") == grader.SOURCE_CLICK
            gap = str(grade.get("grader_gap") or "")
            if clicked and gap.startswith("context_unbuildable"):
                _log.debug(
                    "A clicked grade was held back: %s", gap
                )
                continue
            keep.append(grade)
        return keep

    @staticmethod
    def _prior_daily_bar(daily_bars, session: str):
        """The last daily bar BEFORE this session - the gap's other half."""
        day = str(session or "")[:10]
        prior = None
        for bar in daily_bars or ():
            stamp = str(bar.get("dt") or bar.get("date") or "")[:10]
            if stamp and stamp < day:
                prior = bar
        return prior

    @staticmethod
    def _d1_labels() -> dict[str, str]:
        """Every session the desk has labelled, for the point-in-time read.

        Measured 2026-09-19: `d1_environment.jsonl` holds 15 rows over five
        sessions and none for 2026-09-18, so this honestly answers `unmeasured`
        most days rather than reaching for the nearest label.
        """
        try:
            import d1_environment_store

            return dict(d1_environment_store.labels_by_session(benchmark=BENCHMARK_SYMBOL))
        except Exception:  # noqa: BLE001
            _log.debug("The D1 environment labels were unreadable.", exc_info=True)
            return {}

    @staticmethod
    def _prior_grades(session: str) -> list[dict[str, Any]]:
        """This session's and the previous session's grades, for "what did you
        know at the stamp?". Two small files, never the whole ledger."""
        import market_read_grades as grader

        days = [str(session or "")[:10]]
        try:
            import market_calendar

            days.insert(0, market_calendar.previous_session(date.fromisoformat(days[0])).isoformat())
        except Exception:  # noqa: BLE001 - one session is still an answer
            _log.debug("The previous session could not be read.", exc_info=True)
        rows: list[dict[str, Any]] = []
        for day in days:
            try:
                rows.extend(grader.current_grades(grader.read_grades(day)))
            except Exception:  # noqa: BLE001
                _log.debug("The read ledger was unreadable.", exc_info=True)
        for row in rows:
            read = row.get("read")
            if isinstance(read, Mapping) and not row.get("stamp"):
                row["stamp"] = read.get("stamp")
        return rows

    @staticmethod
    def _internals_bars(session: str, rows) -> dict[str, Any]:
        """The bars `trade_mentor_context.internals_at` rebuilds a block from.

        Read ONCE per session, not once per read: the loader hands over the
        session's whole tape and the daily history, and the ONE builder cuts it
        to each read's own stamp. A session with no read reads nothing at all.
        """
        if not rows:
            return {}
        try:
            import trade_mentor_context

            stamp = rows[0].get("stamp")
            moment = datetime.fromisoformat(str(stamp)) if stamp else None
            return trade_mentor_context.internals_bars_at(session, moment) or {}
        except Exception:  # noqa: BLE001 - no bars is `unmeasured`, not an error
            _log.debug("The internals bars were unreadable.", exc_info=True)
            return {}

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
                verdict, _stamp = day_review_index.stamp_verdict(stored, sources=sources)
                if verdict != "rebuild" and not day_review_index.is_stale(
                    stored, now=now
                ):
                    index = stored
                    # A page open is read-only. The deterministic night/post-close
                    # writer may refresh a moved stamp; this read uses the still
                    # valid body without writing a cache file.
        except Exception:  # noqa: BLE001
            _log.debug("The stored Day Review index was unreadable.", exc_info=True)
            index = None
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
    def _attach_exit_notes(
        session: str, trades: list[dict[str, Any]], problems: list[str]
    ) -> dict[str, Any] | None:
        """Put TJ-9E's two keys on every trade row. ONE read for the whole day.

        Returns the notes mapping the report card then COUNTS, or ``None`` when
        no read was made - which is not the same thing as an empty mapping and
        must not be reported as one.

        **It reads nothing when the day's trades are empty, and that is not an
        optimisation.** `journal_feed._store()` caches a module-global store for
        the life of the process, so whoever calls it FIRST decides which store
        the whole run uses and on which thread its `initialize_schema()`
        migration runs. The trades came from that same store a few lines above;
        if they did not come back, this read must not be the thing that opens
        it. Review 1 blocker 1 is what happened without the guard: three
        Day-Review test files stub the trades read, this read cached a FAKE
        store, and every later `JournalPanel` in the pytest process died on
        `db_path` - 32 errors that are green on base. A day with no trade rows
        has no exit to explain either way.

        Both keys go on EVERY row, including when the read failed: a reader
        must never have to tell an absent key from an empty one. A failure
        costs the notes and says so - never the day's trades.
        """
        rows = [row for row in (trades or ()) if isinstance(row, dict)]
        for row in rows:
            row["exit_note"] = ""
            row["exit_fields"] = {}
        if not rows:
            # Said, never guessed: the report card's line reads `unmeasured`
            # rather than "0 of 0" for a day nobody opened the journal for.
            problems.append(
                "the day's exit notes were not read: this payload opened no trades"
            )
            return None
        notes: dict[str, Any] = {}
        try:
            from ui.services import journal_feed

            notes = dict(journal_feed.exit_notes_on(session))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"the day's exit notes could not be read: {exc}")
            _log.debug("Day Review exit notes unreadable.", exc_info=True)
            return None
        for row in rows:
            found = notes.get(str(row.get("trade_id") or "")) or {}
            row["exit_note"] = str(found.get("raw_text") or "")
            row["exit_fields"] = dict(found.get("exit_fields") or {})
        return notes

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
