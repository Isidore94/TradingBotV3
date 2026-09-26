"""The M5 Strength Board host half of the Alert Center: attach it, auto-adopt its rows into M5 Focus.

Moved verbatim out of `AlertCenterPanel`, which inherits this mixin.
"""

from __future__ import annotations

import logging

from ui.models.bounce import SYMBOL_RE


class StrengthBoardAdoptionMixin:
    """`AlertCenterPanel` methods that host the Strength Board and adopt its rows."""

    def attach_strength_board(self, service, focus_service=None) -> None:
        """Host the M5 Strength Board at the foot of the Strength page.

        `MainWindow` builds and owns the one `StrengthBoardService`; this
        panel is given it. Called once at startup - a second call would add a
        second board to the page, but nothing does.

        Deliberately NOT here: any refresh, timer, thread or fetch. The
        service's single-flight owner and its 15-minute clock are unchanged by
        the move, and the board is still batched yfinance over
        `universe_all.txt` with **zero IB traffic**. The only thing this panel
        adds is a parent and the chart route for a row click - the same route
        every other board on this panel takes.
        """
        from ui.panels.strength_board_panel import StrengthBoardPanel

        adopting_service = (
            self.focus_service if focus_service is None else focus_service
        )
        board = StrengthBoardPanel(service=service, focus_service=adopting_service)
        board.symbolActivated.connect(self._chart_strength_board_symbol)
        self.strength_board = board
        # T1.4 (trader, 2026-09-04): *"I want all shorts and longs on the RS/RW
        # board TC2000 to bne auto added to the M5 focus picks."* The refresh
        # signal, then once for the board already in hand - a desk that starts
        # mid-session must not wait fifteen minutes for its first placement.
        self._strength_focus_service = adopting_service
        try:
            service.boardChanged.connect(self._auto_adopt_strength_board)
        except Exception:
            logging.warning(
                "Strength board auto-Focus could not be connected.", exc_info=True
            )
        else:
            try:
                self._auto_adopt_strength_board(service.board())
            except Exception:
                logging.warning(
                    "Strength board auto-Focus failed on the attached board.",
                    exc_info=True,
                )

        # The alert column's floor is 360 px and the tab stack already claims
        # 170 of it. The board asks for 270 (two side tables, each with a
        # heading row and an "Add all" button), and a widget's minimum reaches
        # the splitter, so hosting it bare would raise the floor the charts
        # are sized against - the one thing this move must not do. The page
        # is a scroll area, so the board's minimum stops there: at a normal
        # column width nothing scrolls sideways, and a trader who drags the
        # column narrower gets a scrollbar instead of narrower charts.
        self.strength_page.attach_strength_board(board)

    #: The `symbol` on the ONE review-event row the board's auto-join writes.
    #: `record_review_event` refuses a row with no symbol, and this event is
    #: about the BOARD rather than about any one name - the names are in the
    #: detail. An underscore makes it unrepresentable as a ticker under this
    #: repo's own grammar (`ui.models.bounce.SYMBOL_RE`), so no symbol-keyed
    #: join can ever match it, and no scanner alert had to be invented for it.
    STRENGTH_BOARD_EVENT_SYMBOL = "M5_STRENGTH_BOARD"

    #: The `Scan` column's vocabulary (packet WS-10B, WISHLIST 10B). The
    #: trader reads the TC2000 board beside their own screen and the one
    #: question it could not answer was "this name is on the list - why is it
    #: not being scanned?". Every verdict here is written at the moment this
    #: method decides, from the numbers it decided on, and rendered verbatim by
    #: `strength_board_panel`; a refusal carries the ADOPTION GATE's own words
    #: rather than a paraphrase, because a second wording of one rule is a
    #: second rule.
    ADOPTION_ADOPTED = "adopted"
    ADOPTION_ALREADY = "already_in_focus"
    ADOPTION_STAGED = "staged (AWAY)"
    ADOPTION_NOT_TODAY = "not today"
    ADOPTION_DECLINED = "declined today"

    def _auto_adopt_strength_board(self, board: dict) -> None:
        """Place the TC2000 board's parity rows on M5 Focus (packet T1.4).

        Trader, 2026-09-04: *"I want all shorts and longs on the RS/RW board
        TC2000 to bne auto added to the M5 focus picks."*

        This is the MACHINE placing a name, so it is modelled on the
        regime-pause auto-join and not on the board's own "Add" button:

        * only rows with an EMPTY ``failed_floors`` are considered - those are
          the rows the TC2000 parity list shows, and a greyed near-miss is a
          name that missed one of the trader's own filters;
        * the ONE adoption gate is re-run on the row's own numbers (the board
          is up to fifteen minutes old), and UNKNOWN fails as it always does;
        * a symbol the trader said "Not today" to this session is skipped, so
          the next refresh cannot undo their answer - and so is a symbol they
          took OFF a focus side today through ANY door (``store.declined_today``,
          fix round 1). ``_ignored_symbols`` only ever holds names the "Not
          today" verb parked; the Focus-review walkthrough, the Focus list's own
          remove button, the cross-focus toggle and the Master AVWAP panel all
          remove without parking, and every one of them was being undone by the
          next fifteen-minute refresh. The record is kept in the STORE, so a
          fifth removal door counts for free;
        * the write goes through the STORE plus an auto-pick marker, NEVER
          ``FocusService.add``, which would forge a trader "like" into
          ``pick_feedback.jsonl``. An existing entry is COUNTED, never
          re-marked: absence of a marker is what makes a pick the trader's.

        It NEVER removes. A name that drops off the board stays on Focus - the
        ten-session fade and "Not today" own removal.

        **WS-10B (2026-09-12) changed two things and neither is the gate.**

        1. **AWAY STAGES.** DESK adopting and AWAY doing nothing at all was
           half the auto-mode matrix: AWAY is meant to STAGE and let the return
           to the desk adopt. The eligible rows now go into the EXISTING
           auto-populate queue (`autopilot_core.stage_auto_populate_candidates`,
           one owner, one lock, one file) rather than into Focus, so
           `_poll_auto_pick_queue`'s drain - which re-measures on the flip back
           to DESK before it adopts anything - is still the only door an
           unattended pick can come through. Nothing new polls: this rides
           `boardChanged` exactly as the adoption already did. EVENING and OFF
           still do nothing.
        2. **EVERY ROW SAYS WHY.** The verdicts above are collected per side
           and pushed to the board's own table, where they render as the last
           column, `Scan`. Computed HERE, at the moment of the decision, and
           never recomputed by the view - a display that re-ran the gate on
           slightly older numbers would disagree with the machine about the
           trader's own list. One INFO line per refresh carries the same
           counts to `trading_bot.log`.

        **Scanner inclusion and Focus adoption stay DISTINCT contracts.** A row
        the gate refuses is not scanned as a consolation prize; if the trader
        ever wants every board row in the scan set regardless, that is a
        separate selection and a separate decision.

        Cheap enough for the Qt thread by construction: a walk over the board's
        own rows, a pure-Python gate per row, the store writes the store
        already batches, and - only when AWAY - one JSON read/write of the
        staging queue, the same file this panel's drain already reads on its
        own poll. No fetch, no re-measurement, no timer.
        """
        if not isinstance(board, dict):
            return
        try:
            mode = self._auto_mode_now()
        except Exception:
            return
        rows_by_side: dict[str, list[dict]] = {
            side: [row for row in (board.get(side) or []) if isinstance(row, dict)]
            for side in ("long", "short")
        }
        total_rows = sum(len(rows) for rows in rows_by_side.values())
        verdicts: dict[str, dict[str, str]] = {"long": {}, "short": {}}

        if mode not in ("DESK", "AWAY"):
            # EVENING and OFF do not adopt or stage from this board.
            # The board still says so, per row: "nothing happened" was the
            # answer this packet exists to replace.
            for side, rows in rows_by_side.items():
                for row in rows:
                    symbol = str(row.get("symbol") or "").strip().upper()
                    if symbol:
                        verdicts[side][symbol] = f"mode {mode}"
            self._publish_strength_board_adoption(verdicts, total_rows)
            return

        focus_service = (
            getattr(self, "_strength_focus_service", None) or self.focus_service
        )
        store = getattr(focus_service, "store", None)
        if store is None:
            # Nothing was decided, so nothing is claimed: the column stays
            # blank rather than reading "not adopted", which would name a
            # refusal that never happened.
            return
        import focus_adoption_gate

        as_of = str(board.get("as_of") or "")
        declined_today = getattr(store, "declined_today", None)
        wanted_by_side: dict[str, list[str]] = {"long": [], "short": []}
        strengths: dict[tuple[str, str], object] = {}
        refused: list[str] = []

        # ------------------------------------------------------ phase 1: judge
        # Both sides judged before anything is written, so the verdict a row
        # carries is the verdict that decided it - not a second opinion formed
        # after the store moved underneath it.
        for side in ("long", "short"):
            for row in rows_by_side[side]:
                symbol = str(row.get("symbol") or "").strip().upper()
                if not symbol or not SYMBOL_RE.fullmatch(symbol):
                    continue
                missed = [str(text) for text in (row.get("failed_floors") or ())]
                if missed:
                    # Greyed near-miss: it is not on the trader's TC2000 list.
                    verdicts[side][symbol] = "not adopted: floor " + "; ".join(missed)
                    continue
                if symbol in self._ignored_symbols:
                    verdicts[side][symbol] = self.ADOPTION_NOT_TODAY
                    refused.append(f"{symbol} (you said not today)")
                    continue
                if callable(declined_today):
                    try:
                        taken_off = bool(declined_today(symbol, side, "m5"))
                    except Exception:
                        taken_off = False
                    if taken_off:
                        verdicts[side][symbol] = self.ADOPTION_DECLINED
                        refused.append(f"{symbol} (you took it off today)")
                        continue
                passes, reason = focus_adoption_gate.passes_focus_adoption_gate(
                    side,
                    row.get("last"),
                    row.get("prev_high"),
                    row.get("prev_low"),
                    row.get("session_vwap"),
                )
                if not passes:
                    # Named, not counted - the same way `_add_all` names one.
                    verdicts[side][symbol] = f"not adopted: {reason}"
                    refused.append(f"{symbol} ({reason})")
                    continue
                strengths[(side, symbol)] = row.get("strength")
                wanted_by_side[side].append(symbol)

        # -------------------------------------------------------- phase 2: act
        adopted: list[str] = []
        staged: list[str] = []
        side_counts = {"long": 0, "short": 0}
        already_auto = 0
        already_trader_owned = 0

        if mode == "AWAY":
            self._stage_strength_board_picks(
                store, wanted_by_side, strengths, verdicts, staged
            )
        else:
            for side in ("long", "short"):
                wanted = wanted_by_side[side]
                if not wanted:
                    continue
                # ONE write per side, not one per name. `add_many` rewrites the
                # focus file, the membership file and the pick clocks once for
                # the batch; sixty names through `add` measured 781 ms on the Qt
                # thread and the board's row count is not bounded by anything
                # this panel controls. The MARKER stays per name - it carries
                # that row's own strength, and `mark_auto_adopted` is a dict
                # write plus a save.
                try:
                    added = list(store.add_many(wanted, side, "m5"))
                except Exception:
                    logging.warning(
                        "Strength board could not place %s on M5 Focus.",
                        ", ".join(wanted),
                        exc_info=True,
                    )
                    for symbol in wanted:
                        verdicts[side][symbol] = "not adopted: add failed"
                        refused.append(f"{symbol} (add failed)")
                    continue
                marker_writer = getattr(store, "mark_auto_adopted", None)
                for symbol in added:
                    if callable(marker_writer):
                        try:
                            marker_writer(
                                symbol,
                                side,
                                "m5",
                                staged_at=as_of,
                                reason=(
                                    f"M5 Strength Board (TC2000) {side} row, "
                                    f"strength {strengths.get((side, symbol))}"
                                ),
                            )
                        except Exception:
                            # An evidence store never costs the event it
                            # records: a lost marker reads as the trader's own
                            # name, which is the safe direction (packet R2).
                            logging.warning(
                                "Strength board could not mark %s as auto-adopted.",
                                symbol,
                                exc_info=True,
                            )
                    verdicts[side][symbol] = self.ADOPTION_ADOPTED
                    adopted.append(symbol)
                    side_counts[side] += 1
                # Everything asked for and not added was already there. Which
                # KIND of "already there" is the interesting number - the
                # regime-pause auto-join distinguishes them the same way - and
                # it is a COUNT only: no marker is ever written over a name the
                # trader typed.
                reader = getattr(store, "is_auto_adopted", None)
                for symbol in wanted:
                    if symbol in added:
                        continue
                    verdicts[side][symbol] = self.ADOPTION_ALREADY
                    ours = False
                    if callable(reader):
                        try:
                            ours = bool(reader(symbol, side, "m5"))
                        except Exception:
                            ours = False
                    if ours:
                        already_auto += 1
                    else:
                        already_trader_owned += 1

        self._publish_strength_board_adoption(verdicts, total_rows)
        if not adopted and not staged and not refused:
            return
        self._record_review_event(
            "strength_board_auto_focus",
            symbol=self.STRENGTH_BOARD_EVENT_SYMBOL,
            detail={
                "mode": mode,
                "side_counts": side_counts,
                "adopted": adopted,
                "staged": staged,
                "refused": refused,
                "already_auto": already_auto,
                "already_trader_owned": already_trader_owned,
                "as_of": as_of,
            },
        )
        if adopted:
            self.statusChanged.emit(
                f"★ {side_counts['long']} long, {side_counts['short']} short "
                "added to M5 Focus from the TC2000 board."
            )
        elif staged:
            self.statusChanged.emit(
                f"{len(staged)} TC2000 board name(s) staged for your return "
                "(AWAY stages, it never adopts)."
            )

    def _stage_strength_board_picks(
        self,
        store,
        wanted_by_side: dict[str, list[str]],
        strengths: dict[tuple[str, str], object],
        verdicts: dict[str, dict[str, str]],
        staged: list[str],
    ) -> None:
        """AWAY: put the eligible rows in the EXISTING staging queue (WS-10B).

        The trader is not at the desk, so nothing may be adopted - but a board
        that discovers a name at 11:00 and forgets it by the time they sit down
        is the discovery thrown away. `stage_auto_populate_candidates` is the
        one owner of that queue: it holds the lock, it applies the per-side
        cap, it refuses a name already on a watchlist or already decided today,
        and the desk's own drain re-measures every queued pick on the flip back
        before adopting it. Passing the STORE's shared paths rather than the
        module defaults keeps a test store inside its sandbox.

        `gate_bar_end` is deliberately left as the staging function writes it
        (empty, with no profile to read one from). An empty measured-bar stamp
        REFUSES at adoption (`pending_pick_gate_ok`), so a board pick can only
        be adopted after the flip's re-verification has measured it against the
        live tape - which is exactly the guarantee AWAY staging is for.
        """
        pending_path = getattr(self, "_auto_pick_pending_path", None)
        wanted = [
            (side, symbol)
            for side in ("long", "short")
            for symbol in wanted_by_side[side]
        ]
        if not wanted:
            return
        if pending_path is None:
            for side, symbol in wanted:
                verdicts[side][symbol] = "not adopted: no staging queue on this desk"
            return
        try:
            from autopilot_core import (
                load_auto_populate_pending_picks,
                stage_auto_populate_candidates,
            )

            candidates: dict[str, list[dict]] = {"longs": [], "shorts": []}
            for side, symbol in wanted:
                try:
                    score = float(strengths.get((side, symbol)))
                except (TypeError, ValueError):
                    score = 0.0
                candidates[f"{side}s"].append(
                    {
                        "symbol": symbol,
                        "reason": f"M5 Strength Board (TC2000) {side} row",
                        "score": score,
                    }
                )
            stage_auto_populate_candidates(
                candidates,
                "strength_board",
                pending_path=pending_path,
                longs_path=store.shared_watchlist_path("long", "m5"),
                shorts_path=store.shared_watchlist_path("short", "m5"),
            )
            queued = load_auto_populate_pending_picks(pending_path)
        except Exception:
            logging.warning(
                "Strength board could not stage %d TC2000 name(s) for the return "
                "to the desk.",
                len(wanted),
                exc_info=True,
            )
            for side, symbol in wanted:
                verdicts[side][symbol] = "not adopted: staging failed"
            return
        for side, symbol in wanted:
            if symbol in (queued.get("pending", {}).get(side) or {}):
                # Staged now, or staged by an earlier refresh - either way the
                # name is waiting in the queue the drain reads.
                verdicts[side][symbol] = self.ADOPTION_STAGED
                staged.append(symbol)
                continue
            try:
                on_focus = symbol in store.focus_symbols(side, "m5")
            except Exception:
                on_focus = False
            verdicts[side][symbol] = (
                self.ADOPTION_ALREADY
                if on_focus
                else "not adopted: the staging queue is full or this name was "
                "already decided today"
            )

    def _publish_strength_board_adoption(
        self, verdicts: dict[str, dict[str, str]], total_rows: int
    ) -> None:
        """The `Scan` column and the one log line per refresh (WS-10B items 2-3).

        Presentation and evidence only: it writes no store, reaches no
        detector, and a failure here can never cost an adoption - the decisions
        are already made and persisted by the time this runs.
        """
        panel = getattr(self, "strength_board", None)
        setter = getattr(panel, "set_adoption", None)
        if callable(setter):
            try:
                setter(verdicts)
            except Exception:
                logging.warning(
                    "Strength board could not show its Scan column.", exc_info=True
                )
        if not total_rows:
            return
        counts: dict[str, int] = {}
        adopted = staged = other = 0
        for side_map in verdicts.values():
            for text in side_map.values():
                if text == self.ADOPTION_ADOPTED:
                    adopted += 1
                    continue
                if text == self.ADOPTION_STAGED:
                    staged += 1
                    continue
                other += 1
                reason = text[len("not adopted: "):] if text.startswith("not adopted: ") else text
                counts[reason] = counts.get(reason, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        reasons = "; ".join(f"{reason} x{count}" for reason, count in ranked[:6])
        if len(ranked) > 6:
            reasons += "; ..."
        logging.info(
            "Strength board: %d rows, %d adopted, %d staged, %d not adopted "
            "(reasons: %s)",
            total_rows,
            adopted,
            staged,
            other,
            reasons or "none",
        )
