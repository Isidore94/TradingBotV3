"""The Trade Mentor's host logic, shared by the desk and the Trade Mentor app.

These methods were ``MainWindow``'s; they moved here whole so the desk (flag
``mentor_app_enabled`` off) and the app (flag on) run the same code. A host
supplies ``trade_mentor_service``, ``_mentor_review()`` (the surface holding
``mentor_card``: ``show_mentor_slot``, ``hide_mentor_card``, ``show_econ_brief``,
``update_econ_brief``) and ``_econ_morning_has_started()``; the rest is here.
"""

from __future__ import annotations

import logging
from datetime import date as _date


def _review(host):
    """The host's card surface; a duck-typed desk host (tests, old callers) has the Alert review."""
    finder = getattr(host, "_mentor_review", None)
    return finder() if finder is not None else host.trading_panel.alert_center.chart_review


def _morning_started(host) -> bool:
    finder = getattr(host, "_econ_morning_has_started", None)
    return finder() if finder is not None else host.econ_reminder_service.morning_has_started()


class MentorHostMixin:
    """Builds and shows the Trade Mentor card: slot -> state -> questions -> writers."""

    _journal_importer = None
    _journal_retry_date = ""

    def _mentor_review(self):
        """The surface that holds ``mentor_card`` (the desk's Alert review, or the app's dock)."""
        raise NotImplementedError

    def _econ_morning_has_started(self) -> bool:
        raise NotImplementedError

    def _sync_trade_mentor_label(self) -> None:
        """The desk's Settings line; a host without one does nothing."""

    def _mentor_rule_lane(self, store, session: str, trades) -> list:
        """The rule-reflection lane: the session's closed trades the rule can be checked on.

        The rule comes from the status-bar chip's worker and the size baseline
        from `_rule_size_baselines`, keyed by the card's session and filled by
        its own worker: no journal read here. An uncached session asks that
        worker and the size check says nothing until the baseline lands.
        """
        try:
            import recap_rule_loop as loop

            chip = getattr(self, "rule_chip", None)
            rule = chip.info() if chip is not None else None
            if not rule or str(rule.get("tag") or "") not in loop.CHECKABLE_TAGS:
                return []
            median = None
            if rule.get("tag") == "size_down_in_chop":
                day = str(session)[:10]
                cached = getattr(self, "_rule_size_baselines", None) or {}
                if day in cached:
                    median = cached[day]
                else:
                    self._request_rule_size_baseline(day)
            return loop.reflection_rows(
                rule,
                trades,
                session=session,
                size_median=median,
                regime_timeline=list(getattr(self, "_regime_timeline", ()) or ()),
            )
        except Exception:  # noqa: BLE001 - a lane never costs the card
            logging.debug("Mentor rule lane unreadable.", exc_info=True)
            return []

    def _on_econ_view(self, view: dict) -> None:
        """Show today's news & econ once per session; later views redraw in place."""
        try:
            review = _review(self)
            session = str(view.get("session") or "")
            mentor = self.trade_mentor_service
            if mentor.econ_brief_shown(session):
                review.update_econ_brief(view)
                return
            if not mentor.enabled() or mentor.is_paused():
                return
            # Nobody is at the desk to read it, or it is before 05:00 PT; the
            # next refresh (or mode flip) tries again.
            if self._auto_mode_now() in ("AWAY", "EVENING"):
                return
            if not _morning_started(self):
                return
            review.show_econ_brief(view)
            mentor.mark_econ_brief_shown(session)
        except Exception:  # noqa: BLE001 - the brief never costs the desk
            logging.debug("Econ block could not be shown.", exc_info=True)

    def _previous_mentor_read(self, session: str):
        """The last read the Trade Mentor filed for this session, if any.

        Shown beside the new prompt so "Read unchanged" has something to name.
        One bounded read of a small JSONL, at most once an hour - not a paint
        path, and never in the 60-second poll (the service emits, this runs).

        The answer is the latest read of EACH timeframe, `{"M5": row, "D1":
        row}`, not the latest row full stop. "Read unchanged" reaffirms PER
        TIMEFRAME, and `rows[-1]` is the D1 row on every day the 08:00 card was
        answered - which is how a D1-timeframe entry came to be filed at 09:00
        carrying a rest-of-day call, a row true of neither timeframe.
        """
        try:
            from ui.services.market_journal_service import shared_journal_service

            rows = [
                row
                for row in shared_journal_service().entries_for(session)
                if str(row.get("origin") or "") == "trade_mentor"
            ]
        except Exception:  # noqa: BLE001 - a missing previous read is not an error
            logging.debug("Previous mentor read unreadable.", exc_info=True)
            return None
        latest: dict[str, dict] = {}
        for row in rows:
            timeframe = str(row.get("timeframe") or "").strip().upper()
            if timeframe in ("M5", "D1") and str(row.get("text") or "").strip():
                latest[timeframe] = row
        return latest or None

    def _show_trade_mentor_prompt(self, slot) -> None:
        """Show a due prompt in its reusable popup, with its question."""
        review = _review(self)
        try:
            review.show_mentor_slot(slot, previous=self._previous_mentor_read(str(slot.session)))
        except Exception:  # noqa: BLE001 - a prompt never costs the desk
            logging.debug("Trade Mentor prompt could not be shown.", exc_info=True)
            return
        # The Settings line says when the NEXT one is, so it moves every time a
        # prompt lands rather than telling the trader what was true at startup.
        self._sync_trade_mentor_label()
        # The 09:00 second section, and the RIDE. Everything from here down is
        # inside one guard on purpose: a trade check that cannot be built must
        # never cost the prompt above it, which is the read the trader is
        # actually being interrupted for. The kind test used to sit OUTSIDE it
        # and to read the constant off the wrong module, so every prompt raised
        # `AttributeError` in a Qt slot and no trade section ever appeared.
        try:
            import trade_mentor_trade_check as check
            from journal_store import JournalStore
            from trade_mentor_schedule import KIND_M5_TRADES

            card = review.mentor_card
            store = JournalStore()
            # The TASK is built FIRST and the pulls come after it (TJ-14B review,
            # item B): a raise anywhere in the pull path used to lose the forced
            # trade section that is built further down, which is the one thing on
            # this card the trader is not allowed to skip.
            task = check.build_task(store, slot.scheduled_at.date())
            # TJ-14B item 4: ONE card starts AT MOST ONE import. Everything
            # inside is guarded and returns rather than raises.
            self._mentor_journal_pull(slot, task)
            # TJ-14B item 3: the few questions the desk cannot work out on its
            # own, at most three, on EVERY card - the trade section below has
            # its own ride rule and its own early returns.
            self._show_mentor_questions(slot, store=store)
            # S16: the regime journal is read on a worker for this and the next card.
            card.refresh_regime_lane(store)

            is_check_slot = str(getattr(slot, "kind", "")) == KIND_M5_TRADES
            # EVERY delivered slot of the session hands the card the FRESH
            # task; the card MERGES it (`set_trade_check`), keeping the exact
            # widgets of a row the trader may already have touched, adding a
            # trade it does not hold yet, dropping one that is answered, and
            # rewriting the heading every time.
            #
            # The host used to return early whenever the card held ANSWER
            # WIDGETS, which was the only protection those half-set combos had.
            # Once the 09:00 not-ready card started drawing today's own fills
            # (TJ-14B) that early return fired on every later slot of the day:
            # the reviewed session's trades were never asked about at all, and
            # the card went on printing `journal not ready` and a freshness
            # date that was no longer true. The protection now lives in the
            # merge, which is where it can protect the widgets WITHOUT also
            # freezing the words above them.
            carrying = str(card.trade_check_session() or "") == str(slot.session)
            if not is_check_slot and not carrying and not self._trade_check_is_owed(check, slot, task):
                return

            card.set_trade_check(task, store=store, auto_mode=self._auto_mode_now())
        except Exception:  # noqa: BLE001 - the read still stands without it
            logging.debug("Trade Mentor trade check could not be built.", exc_info=True)

    def _trade_check_is_owed(self, check, slot, task=None) -> bool:
        """Does this ORDINARY slot have to carry the trade check?

        TJ-9 item 2: *"AWAY still prompts nothing; the first DESK slot after it
        carries the section."* The section used to be built only for the
        `m5_trades` kind, and only the 09:00 slot has that kind - so a 09:00
        that was away, idle, locked, skipped or expired took the whole day's
        questions with it, and a trader who sat down at 11:00 was asked nothing
        at all.

        It is owed when the reviewed session still has an unlabelled trade, or
        an EXIT nobody has explained (TJ-9E), or when its broker statement has
        not landed (item 6's line has to ride too). A session whose trades are
        all answered brings nothing back. AWAY needs no test here: the service
        records the absence and never emits `promptDue`, so an ordinary slot in
        AWAY does not reach this.

        The exit count is asked SEPARATELY and not folded into the unlabelled
        one, because they are two questions: a swing whose four entry fields
        were answered the morning after it opened is not unlabelled and can
        still have an exit nobody explained. Review 1 blocker 4 is what one
        number costs - a trader who was away at 09:00, or who dismissed the
        card, was never asked about that exit at all, and the next morning the
        reviewed session has moved on.
        """
        try:
            # P8 P2: a trade with no stop rides on every card, so it is asked
            # before the day's budgeted questions.
            if task is not None and check.stop_owed(task):
                return True
            reviewed = check.previous_exchange_session(slot.scheduled_at.date())
            # Only what may still be ASKED: a trade asked once (`MENTOR_ASKED`,
            # 2026-09-23) keeps its blanks and never brings the section back.
            if self.trade_mentor_service.unlabelled_trades(reviewed, askable=True) > 0:
                return True
            if self.trade_mentor_service.unexplained_exits(reviewed, askable=True) > 0:
                return True
            from journal_store import JournalStore

            return check.fills_current_to(JournalStore()) != _date.fromisoformat(reviewed)
        except Exception:  # noqa: BLE001 - an unreadable journal asks nothing extra
            logging.debug("Trade check ride undecidable.", exc_info=True)
            return False

    def _show_mentor_questions(self, slot, store=None) -> None:
        """Put this card's budgeted questions on it (TJ-14B item 3).

        Everything `mentor_questions.pending` needs arrives already loaded - the
        registry is PURE and a trigger that opened a store would be a second
        opinion about it, on whatever thread the card happened to be built on.
        The reads here are bounded to the two sessions a question can be about.

        `carried` is kept on the window so the next card asks the remainder
        first: a question over budget is counted and carried, never dropped.
        """
        try:
            import mentor_questions

            card = _review(self).mentor_card
            if store is None:
                from journal_store import JournalStore

                store = JournalStore()
            state = self._mentor_question_state(slot, store)
            result = mentor_questions.pending(state, slot)
            self._mentor_carried = tuple(result.carried)
            card.set_questions(result, store=store, service=self.trade_mentor_service)
        except Exception:  # noqa: BLE001 - a question never costs the prompt
            logging.debug("Trade Mentor questions could not be built.", exc_info=True)

    def _mentor_question_state(self, slot, store) -> dict:
        """Every lane `mentor_questions.pending` reads, loaded once, here."""
        import trade_mentor_trade_check as check

        session = str(getattr(slot, "session", "") or "")
        reviewed = check.previous_exchange_session(slot.scheduled_at.date())
        trades: list = []
        for day in (session, reviewed):
            try:
                trades.extend(store.list_trades(trade_date=day))
            except Exception:  # noqa: BLE001 - an unreadable day asks nothing
                logging.debug("Mentor trade lane unreadable for %s.", day, exc_info=True)
        plan_open, plan_closed = self._mentor_plan_lanes(slot.scheduled_at)
        regime_lane = self._mentor_regime_lane()
        return {
            "session": session,
            "now": slot.scheduled_at,
            "auto_mode": self._auto_mode_now(),
            "trades": trades,
            "open_positions": [
                row for row in trades if str(row.get("status") or "").upper() != "CLOSED"
            ],
            "likes": self._mentor_like_lane(trades, (session, reviewed)),
            # `grader_gap` is still DORMANT (plan.md §12.5 names TJ-10's small
            # follow-up), so its lane is still named and still empty.
            "grader_gaps": (),
            # TJ-12 woke `trade_origin` and `open_position_check`, so these are
            # really read now. An EMPTY lane is not neutral here: `planned_state`
            # answers `unplanned` when nothing was said, and an unread store
            # looks exactly like nothing said - the trader would be asked where
            # every trade came from.
            **self._mentor_origin_lanes((session, reviewed)),
            "ai_question": self._mentor_ai_question(),
            # TJ-9E, and it is a LANE like every other one here: the registry
            # is pure, so a trigger that opened a store would be a second
            # opinion about it on whatever thread the card was built on. The
            # kind shipped AWAKE with nothing feeding this key, so lead
            # decision 7's budget clause could never fire (review 1 advisory
            # 2). The window ends at the CARD's own session, never the reviewed
            # one, because a draft is offered on its own clock.
            #
            # MEASURED, on the Qt thread at card-show time, over a 201-trade
            # scratch journal carrying 20 drafts: **about 4 ms warm** for the
            # lane, 1.4-1.8 ms on a small journal, and **12.7 ms on the FIRST
            # call** of the process, where the imports and the first statement
            # are paid. ONE pack read: two `opportunity_events` queries cover
            # the whole five-session window, and a session nobody wrote a note
            # in costs no file read at all. Review 2 measured 14.7 ms against
            # the older shape, which read five pack files whatever the journal
            # said. `_mentor_annotation_lane` above already reads a file here,
            # so this is the same class of cost and not a new one; it stays on
            # this thread this round by decision.
            "exit_drafts": self._mentor_exit_drafts(store, session),
            # Day Recap coach: closed trades checked against today's rule.
            "rule_reflections": self._mentor_rule_lane(store, session, trades),
            # P1-7 7b: the night's open challenges to the trading plan.
            "plan_challenges": plan_open,
            # S16: the regime journal lane, already read on the card's worker.
            "structural_regime": regime_lane,
            "answered": {
                **self._mentor_answered(
                    store,
                    (session, reviewed),
                    trade_ids=[str(row.get("trade_id") or "") for row in trades],
                ),
                **plan_closed,
                **dict((regime_lane or {}).get("answered") or {}),
            },
            "retired": self.trade_mentor_service.retired_subjects(),
            "carried": getattr(self, "_mentor_carried", ()),
        }

    def _mentor_regime_lane(self):
        """The regime lane the card's worker last read (S16), or ``None``. No read here."""
        try:
            return _review(self).mentor_card.regime_lane()
        except Exception:  # noqa: BLE001 - a lane never costs the card
            logging.debug("Mentor regime lane unavailable.", exc_info=True)
            return None

    def _on_regime_lane_changed(self, first_load: bool) -> None:
        """The first regime read after a card was shown: put its question on that card."""
        if not first_load:
            return
        card = _review(self).mentor_card
        slot, store = card.shown_slot(), card.regime_store()
        # Only a scheduled card handed a store; a hand-opened card gets no questions.
        if slot is not None and store is not None:
            self._show_mentor_questions(slot, store=store)

    @staticmethod
    def _mentor_plan_lanes(now) -> tuple[list, dict]:
        """Open plan challenges, and the closed ones as `answered` keys (P1-7 7b).

        Two small append-only files; never raises - a lane never costs the card.
        """
        try:
            import plan_challenges

            moment = now if getattr(now, "tzinfo", None) is not None else None
            return (
                plan_challenges.open_challenges(moment),
                plan_challenges.closed_keys(moment),
            )
        except Exception:  # noqa: BLE001 - a lane never costs the card
            logging.debug("Mentor plan-challenge lane unreadable.", exc_info=True)
            return [], {}

    @staticmethod
    def _mentor_exit_drafts(store, session: str) -> list:
        """The night's exit readings the trader has NOT signed off yet (TJ-9E).

        The lane behind `exit_draft_review`, and the reason the Confirm click
        exists at all: a draft is offered on its OWN clock
        (`check.EXIT_DRAFT_OFFER_SESSIONS`, walked on the exchange calendar),
        never on the session the card happens to be reviewing. The trade exits
        Monday, the note is typed on TUESDAY's card, TUESDAY NIGHT drafts it,
        and Wednesday's card reviews Tuesday - where Monday's trade is not a
        row at all.

        `session` is the CARD's own session, so the window is the trader's last
        five sessions ending today. The rule and the reads live in
        `trade_mentor_trade_check`; this is the seam that hands them to a pure
        registry, because a trigger that opened a store would be a second
        opinion about it on whatever thread the card was built on.

        Never raises: a lane never costs the card.
        """
        try:
            import trade_mentor_trade_check as check

            return list(check.waiting_exit_drafts(store, str(session or "")[:10]))
        except Exception:  # noqa: BLE001 - a lane never costs the card
            logging.debug("Mentor exit-draft lane unreadable.", exc_info=True)
            return []

    @staticmethod
    def _mentor_annotation_lane(days) -> list:
        """The sessions' like/claim annotations, bounded to those sessions.

        ONE reader for two lanes: the quick-like follow-up and TJ-12's
        planned-vs-unplanned question both ask this log the same bounded
        question, and two walks of an append-only file on the Qt thread is the
        shape every other log-walking read grew a stall out of.
        """
        from pathlib import Path

        from project_paths import TRADER_ANNOTATIONS_FILE
        from ui.annotations.store import EVENT_LIKE_CLAIM, load_annotations

        rows: list[dict] = []
        for day in [str(value)[:10] for value in days if str(value or "").strip()]:
            rows.extend(
                load_annotations(
                    Path(TRADER_ANNOTATIONS_FILE),
                    session_date=day,
                    event_types=(EVENT_LIKE_CLAIM,),
                )
            )
        return rows

    @classmethod
    def _mentor_origin_lanes(cls, days) -> dict:
        """The lanes `trade_origin.planned_state` reads, loaded once, bounded.

        Each lane is guarded on its own: an unreadable store asks MORE questions
        (nothing was said about that name, as far as the desk can tell) and
        never takes the Mentor card down.

        Which lanes are really read is `day_report_card.DESK_ORIGIN_LANES_READ`
        and is declared THERE, once, because the Day Review worker builds the
        same lanes and the card has to say which doors were opened. An unread
        lane is indistinguishable from "nothing was said" to
        `trade_origin.planned_state`, so the card names it rather than printing
        a bare `unplanned` (reviewer, 2026-09-20: 30 of the trader's 33 trades
        since 2026-08-20 read `unplanned` for exactly this reason).

        `focus_adds` and `armed` are named and EMPTY until **TJ-12F**: neither
        store has a public reader that hands back a row with the stamp key
        `trade_origin` reads, and inventing one is that packet's work. A trade
        planned only through a Focus add or an armed alert is therefore asked
        once - `CADENCE_ONCE` - and the trader's own answer is what the Process
        line then reads.
        """
        import day_report_card

        wanted = [str(value)[:10] for value in days if str(value or "").strip()]
        decisions: list[dict] = []
        try:
            decisions = list(cls._mentor_annotation_lane(wanted))
        except Exception:  # noqa: BLE001 - an unreadable log says nothing
            logging.debug("Mentor decision lane unreadable.", exc_info=True)
        claims: list[dict] = []
        try:
            import claimed_picks

            claims = [
                row
                for row in claimed_picks.load_rows()
                if str(row.get("session_date") or "")[:10] in wanted
            ]
        except Exception:  # noqa: BLE001 - an unreadable store says nothing
            logging.debug("Mentor claim lane unreadable.", exc_info=True)
        loaded = {"decisions": decisions, "claims": claims}
        return {
            name: loaded.get(name, ())
            if name in day_report_card.DESK_ORIGIN_LANES_READ
            else ()
            for name in day_report_card.ORIGIN_LANES
        }

    @classmethod
    def _mentor_like_lane(cls, trades, days) -> list:
        """The sessions' QUICK likes, each told whether it was then traded.

        The join is by name and SIDE against the same two sessions' trades - a
        LONG like says nothing about a SHORT entry - and it is done here rather
        than in the registry so the trigger stays pure.

        **Bounded to the sessions a question can be about** (TJ-14B review, item
        E): the store's own `session_date` filter is asked once per session
        rather than the whole append-only log being walked on the Qt thread
        every prompt - 3.5 ms today and unbounded, which is the shape that grew
        every other log-walking read into a stall.
        """
        wanted = [str(day)[:10] for day in days if str(day or "").strip()]
        rows: list[dict] = []
        try:
            rows = list(cls._mentor_annotation_lane(wanted))
        except Exception:  # noqa: BLE001 - a missing log asks nothing
            logging.debug("Like lane unreadable.", exc_info=True)
            return []
        traded: set[tuple[str, str]] = set()
        for trade in trades:
            symbol = str(trade.get("symbol") or "").strip().upper()
            side = str(trade.get("direction") or "").strip().upper()
            if symbol and side:
                traded.add((symbol, side))
        lane: list[dict] = []
        for row in rows:
            symbol = str(row.get("symbol") or "").strip().upper()
            side = str(row.get("side") or "").strip().upper()
            side = "LONG" if side.startswith("LONG") else "SHORT" if side.startswith("SHORT") else ""
            enriched = dict(row)
            if symbol and side and (symbol, side) in traded:
                enriched["matched_trade_id"] = f"{symbol}:{side}"
            lane.append(enriched)
        return lane

    @staticmethod
    def _mentor_ai_question() -> dict:
        """Last night's coaching question and its click options, or `{}`.

        ONE walk of the narrations folder - the card's own legacy line reads the
        same newest file, and it is hidden when this becomes a click.
        """
        try:
            import json
            from pathlib import Path

            from project_paths import MARKET_STORY_NARRATIONS_DIR

            for path in reversed(sorted(Path(MARKET_STORY_NARRATIONS_DIR).glob("*.json"))):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                narration = payload.get("narration") if isinstance(payload, dict) else None
                if not isinstance(narration, dict):
                    continue
                question = str(narration.get("mentor_question") or "").strip()
                if not question:
                    continue
                options = narration.get("mentor_question_options") or ()
                return {
                    "question": question,
                    "options": tuple(str(item) for item in options if str(item or "").strip()),
                }
        except Exception:  # noqa: BLE001 - a degraded night asks nothing
            logging.debug("Overnight Mentor question unreadable.", exc_info=True)
        return {}

    @staticmethod
    def _mentor_answered(store, days, trade_ids=()) -> dict:
        """Which questions already have an answer, so none is asked twice.

        Two stores, because two kinds file their answers as the trader's own
        dated Market Journal row (`day_close`, `ai_question`) and the rest as
        append-only annotation rows. Both reads are bounded to the sessions a
        question can be about.

        **An answer about a TRADE is found by the trade, never only by the day
        it was given on** (trader 2026-09-21: *"dont ask for it again just store
        that info"*). A trade's `trade_date` MOVES - an open position answered
        on Monday closes on Thursday and lands back on Thursday's card - and a
        read keyed on the answer's own date could not see Monday's row, so a
        `once` question came back. Read FIRST, so a same-day row still wins
        the stamp a daily kind compares.
        """
        answered: dict[str, dict] = {}
        import mentor_questions

        def _keep(rows) -> None:
            for row in rows:
                payload = row.get("payload") or {}
                kind = str(payload.get("mentor_question_kind") or "")
                subject_id = str(payload.get("subject_id") or "")
                if kind and subject_id:
                    answered[f"{kind}:{subject_id}"] = {
                        "answered_at": str(row.get("occurred_at") or "")[:10]
                    }

        for trade_id in dict.fromkeys(str(value or "").strip() for value in trade_ids):
            if not trade_id:
                continue
            try:
                _keep(
                    store.list_opportunity_events(
                        trade_id=trade_id,
                        event_type=mentor_questions.EVENT_MENTOR_ANSWER,
                        limit=1000,
                    )
                )
            except Exception:  # noqa: BLE001
                logging.debug("Mentor answers unreadable for %s.", trade_id, exc_info=True)

        for day in days:
            if not str(day or "").strip():
                continue
            try:
                rows = store.list_opportunity_events(
                    event_type=mentor_questions.EVENT_MENTOR_ANSWER,
                    trade_date=str(day)[:10],
                    limit=1000,
                )
            except Exception:  # noqa: BLE001
                logging.debug("Mentor answers unreadable for %s.", day, exc_info=True)
                continue
            _keep(rows)
        try:
            from ui.services.market_journal_service import shared_journal_service

            for day in days:
                if not str(day or "").strip():
                    continue
                for row in shared_journal_service().entries_for(str(day)[:10]):
                    payload = (row.get("mentor") or {}).get("mentor_question") or {}
                    kind = str(payload.get("mentor_question_kind") or "")
                    subject_id = str(payload.get("subject_id") or "")
                    if kind and subject_id:
                        answered[f"{kind}:{subject_id}"] = {
                            "answered_at": str(payload.get("answered_at") or "")[:10]
                            or str(day)[:10]
                        }
        except Exception:  # noqa: BLE001 - a missing journal answers nothing
            logging.debug("Mentor journal answers unreadable.", exc_info=True)
        return answered

    def _auto_mode_now(self) -> str:
        """The Auto Pilot mode, or `""` when it cannot be read.

        An unreadable mode is NOT an absence: the safe direction for a pull is
        the same as for a prompt, and the Mentor service has already decided
        the trader is present by the time this runs.
        """
        try:
            from autopilot_core import read_auto_pilot_mode

            return str(read_auto_pilot_mode() or "").upper()
        except Exception:  # noqa: BLE001 - an unreadable mode blocks nothing
            logging.debug("Auto mode unreadable for the pre-card pull.", exc_info=True)
            return ""

    def _mentor_journal_pull(self, slot, task) -> dict:
        """The ONE owner of the desk's day-time Questrade attempts (TJ-14B).

        **One card starts AT MOST ONE import** (review blocker 1). Two calls in
        one synchronous slot cannot both start: `JournalImportService.running`
        refuses the second, and the one that lost was the THREE-day morning
        catch-up, marked spent and never fired again that morning - so a Monday
        whose Friday-night import had failed never reached back to Friday.

        So they are ordered, never stacked:

        * when the morning catch-up is OWED - the reviewed session's statement
          has not landed and nothing has been retried today - it goes FIRST and
          the pre-card pull is skipped on this card. Its three days cover today
          too, and it does not spend the pre-card cap;
        * otherwise the pre-card pull runs, and only on one of the day's three
          RESERVED cards (`mentor_questions.pull_slot_ids`: the 09:00 card, the
          middle card, and the last one).

        The tally is persisted beside the Mentor's slot state, so a restart
        cannot spend the day's attempts twice; a corrupt one is read as empty
        and rewritten clean. The pull runs on `JournalImportService`'s own
        `QThread`, the desk's single caller of the Questrade refresh chain, and
        nothing here refreshes a token. **The card never waits for it**: a fill
        a late pull lands is asked about on the NEXT card, and a pull that
        raises, refuses or is busy costs the card nothing.
        """
        try:
            import mentor_questions
            import trade_mentor_trade_check as check

            service = self.trade_mentor_service
            tally = service.pull_tally()
            day = str(getattr(slot, "session", "") or "")
            auto_mode = self._auto_mode_now()
            last_retry = self._journal_retry_date or str(tally.get("last_retry") or "")
            if not getattr(task, "journal_ready", False) and str(last_retry)[:10] != day[:10]:
                outcome = check.morning_import_retry(
                    self._journal_import_service(),
                    task,
                    today=day,
                    # The once-a-morning date is kept in the PERSISTED tally
                    # too, so a restart before 10:00 no longer allows a second
                    # morning pull (TJ-9's advisory).
                    last_retry=last_retry,
                    tally=tally,
                    auto_mode=auto_mode,
                )
                self._journal_retry_date = str(outcome.get("last_retry") or "")
                persisted = dict(outcome.get("tally") or tally)
                persisted["last_retry"] = self._journal_retry_date
                service.set_pull_tally(persisted)
                if outcome.get("retried"):
                    logging.info(
                        "Trade Mentor: retrying the Questrade import before the "
                        "%s card (fills current to %s).",
                        getattr(slot, "slot_id", ""),
                        getattr(task, "fills_current_to", "") or "nothing yet",
                    )
                return outcome
            outcome = mentor_questions.pre_card_pull(
                self._journal_import_service(),
                today=day,
                tally=tally,
                auto_mode=auto_mode,
                slot=slot,
            )
            persisted = dict(outcome.get("tally") or {})
            if last_retry:
                persisted["last_retry"] = last_retry
            service.set_pull_tally(persisted)
            if outcome.get("pulled"):
                logging.info(
                    "Trade Mentor: pulling the Questrade journal before the %s card.",
                    getattr(slot, "slot_id", ""),
                )
            return outcome
        except Exception:  # noqa: BLE001 - a pull never costs the card
            logging.debug("Mentor journal pull failed.", exc_info=True)
            return {}

    def _journal_import_service(self):
        """The desk's ONE journal import owner, built on first need.

        `JournalImportService` owns its own `QThread` and is the single caller
        of the Questrade refresh chain; the Mentor's morning retry calls it and
        never refreshes a token itself.
        """
        service = getattr(self, "_journal_importer", None)
        if service is None:
            from ui.services.journal_import_service import JournalImportService

            service = JournalImportService(self)
            self._journal_importer = service
        return service
