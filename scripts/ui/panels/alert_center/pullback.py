"""The Pullback alert of the Alert Center: M15/M30 SMA legs, the tick poll, fires and the auto-arm sweep.

Moved verbatim out of `AlertCenterPanel`, which inherits this mixin.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from chart_watch import (
    ChartWatchTrigger,
    arm_chart_watch,
    evaluate_h1_bars,
    h1_bounce_message,
    PERSISTENT_WATCH_KINDS,
    PULLBACK_KIND,
    PULLBACK_TRIGGERS,
    TRIGGER_H1_EMA15_BOUNCE,
    TRIGGER_SMA_RETEST,
)
from swallowed import note_swallowed
from ui.panels.alert_center.gates import intraday_last_bucket_end


class PullbackWatchMixin:
    """`AlertCenterPanel` methods that evaluate, fire and auto-arm the Pullback alerts."""

    #: The Pullback alert's two SMA timeframes and the SMA each one is read
    #: with, in the trader's own words (2026-09-15): *"on the M15 we use the
    #: 150 moving average on the M30 we use the 75"*.
    PULLBACK_TIMEFRAMES: tuple[tuple[int, int], ...] = ((15, 150), (30, 75))

    #: The three triggers `indicators.pullback_sma_reclaim` owns. The fourth,
    #: `h1_ema15_bounce`, is the WISHLIST 10C rule and keeps its own path.
    PULLBACK_SMA_TRIGGERS = tuple(
        name for name in PULLBACK_TRIGGERS if name != TRIGGER_H1_EMA15_BOUNCE
    )

    def _intraday_history_cache(self, interval_minutes: int):
        """One `IntradayHistoryCache` per interval, built on first use.

        The desk has no cached M15 or M30 series at all - the M5 window is
        BounceBot's and the durable intraday store has never been written on
        this desk - so unlike the H1 leg these two have no primary to fall
        back FROM. The cache is therefore the only source, and it refreshes
        once per completed bucket on its OWN worker thread: nothing here ever
        fetches on the Qt thread, and zero IB traffic (the group RS/RW tape
        precedent). None when the module cannot be built at all, and then the
        timeframe simply reports "not measured" for ever rather than raising
        inside a 60-second poll.
        """
        caches = getattr(self, "_intraday_history", None)
        if caches is None:
            caches = {}
            self._intraday_history = caches
        minutes = int(interval_minutes)
        cache = caches.get(minutes)
        if cache is None:
            try:
                from intraday_history import IntradayHistoryCache

                cache = IntradayHistoryCache(minutes)
            except Exception:  # pragma: no cover - yfinance/env unavailable
                logging.debug(
                    "Intraday history unavailable (%sm)", minutes, exc_info=True
                )
                cache = False
            caches[minutes] = cache
        return cache or None

    @staticmethod
    def _pullback_warmup_bars(sma_length: int) -> int:
        try:
            from indicators.pullback_sma_reclaim import warmup_bars

            return int(warmup_bars(sma_length))
        except Exception:  # pragma: no cover - the module is first-party
            return int(sma_length) + 10

    def _pullback_triggers(self, watch) -> tuple[str, ...]:
        """What THIS watch waits for. A row with no list waits for all four.

        A watch stored before PCT-1 carries only `h1_ema15_bounce` (see
        `chart_watch.chart_watch_from_dict`), so the SMA legs are never run
        for it and its health cell never mentions a timeframe it was not
        armed for.
        """
        return tuple(getattr(watch, "triggers", ()) or PULLBACK_TRIGGERS)

    @staticmethod
    def _pullback_timeframe_scope(watch) -> frozenset[str]:
        """The legs this arm may evaluate; empty remains the legacy all-leg arm."""
        return frozenset(
            str(item).strip().upper()
            for item in (getattr(watch, "timeframes", ()) or ())
            if str(item).strip()
        )

    def _pullback_uses_h1(self, watch) -> bool:
        scope = self._pullback_timeframe_scope(watch)
        return not scope or "H1" in scope

    def _pullback_sma_timeframes(self, watch) -> tuple[tuple[int, int], ...]:
        scope = self._pullback_timeframe_scope(watch)
        return tuple(
            (minutes, length)
            for minutes, length in self.PULLBACK_TIMEFRAMES
            if not scope or f"M{minutes}" in scope
        )

    def _pullback_bars_for_watch(self, watch, interval_minutes: int, *, now=None):
        """This watch's M15 or M30 series, and an ASK for the next refresh.

        Reads only what is already in memory and returns at once; the request
        is what keeps the series current, and the cache itself refuses more
        than one fetch per completed bucket. The first cycle after arming
        simply reports "not measured" until the answer lands.
        """
        cache = self._intraday_history_cache(interval_minutes)
        if cache is None:
            return []
        bars = cache.bars_for(watch.symbol)
        try:
            cache.request(watch.symbol, now=now or datetime.now())
        except Exception:  # pragma: no cover - a refresh never costs the poll
            logging.debug(
                "Intraday history request failed for %s", watch.symbol, exc_info=True
            )
        return bars

    def _pullback_timeframe_note(self, watch, interval_minutes: int, sma_length: int) -> str:
        """One timeframe's half of the health cell. READS, never fetches.

        The poll is what asks for a refresh (`_pullback_bars_for_watch`); a
        health cell that fetched would put the network on the Qt thread's
        critical path for a cosmetic string.
        """
        label = f"M{int(interval_minutes)}"
        needed = self._pullback_warmup_bars(sma_length)
        cache = self._intraday_history_cache(interval_minutes)
        if cache is None:
            return f"not measured (no {label} history)"
        try:
            have = len(cache.bars_for(watch.symbol))
            unavailable = bool(cache.unavailable(watch.symbol))
            stale = bool(cache.last_refresh_failed(watch.symbol))
        except Exception:  # pragma: no cover - a note never costs a row
            return ""
        if have >= needed:
            if stale:
                return f"{label} from yfinance (stale - last refresh failed)"
            return f"{label} from yfinance"
        if unavailable:
            return f"not measured ({have} of {needed} {label} bars, yfinance unavailable)"
        return f"not measured ({have} of {needed} {label} bars)"

    def _armed_watch_note(self, watch) -> str:
        """Why an armed watch is not answering yet, in the inventory's health cell.

        Today this is the H1 retester's warm-up: the rule needs
        `WARMUP_BARS` completed H1 bars and the desk's cached M5 window
        supplies fewer, so the honest state is `not measured (N of 45 H1
        bars)` rather than `ok`. An armed surface whose job is to say "these
        are the exact conditions I am waiting on" must not report a watch as
        healthy when it cannot evaluate at all.

        Since the RV-H1-HISTORY repair (2026-09-13) there is a third honest
        state: bars that are GOOD but STOPPED. A refresh that fails after a
        success keeps the bars it already has - they are still the best answer
        - and the cell says so, because an ageing verdict must not read exactly
        like a live one. `unavailable` keeps its own meaning: nothing was ever
        fetched at all.

        PCT-1 widened the cell to ONE STATE PER TIMEFRAME, joined with `;` -
        `H1 from cache; not measured (0 of 160 M15 bars); M30 from yfinance` -
        because one button now waits for four phenomena on three series and a
        single verdict would hide which of them can actually answer. Each H1
        state is the exact string it has always been; only the joining is new.
        A watch stored before the rename waits for the H1 leg alone and its
        cell still says exactly what it always did.
        """
        if str(getattr(watch, "kind", "") or "") not in PERSISTENT_WATCH_KINDS:
            return ""
        triggers = self._pullback_triggers(watch)
        parts: list[str] = []
        if TRIGGER_H1_EMA15_BOUNCE in triggers and self._pullback_uses_h1(watch):
            parts.append(self._h1_watch_note(watch))
        if any(name in triggers for name in self.PULLBACK_SMA_TRIGGERS):
            for interval_minutes, sma_length in self._pullback_sma_timeframes(watch):
                parts.append(
                    self._pullback_timeframe_note(watch, interval_minutes, sma_length)
                )
        return "; ".join(part for part in parts if part)

    def _poll_pullback_watches(self, now: datetime | None = None) -> None:
        """The armed Pullback alerts, evaluated once per tick (PCT-1).

        Was `_poll_h1_bounce_watches` (WISHLIST 10C). Same timer, same expiry
        call first, and the H1 leg below is the code that shipped - what is
        new is the auto-arm sweep in front of it and the three SMA triggers
        after it.

        Rides the D1 EVENT poll rather than the 30 s chart-watch one because a
        Pullback alert is a multi-day arm: it needs that poll's trading-day
        expiry pass, and re-reading its bars four times an hour would answer
        the same question four times - a completed H1 bar arrives once an hour
        and a completed M15 bar four times, and this returns the same verdict
        until one does.

        **Cost on the Qt thread, after the review** (blocker 3, 2026-09-15 -
        the first tick against the live stores armed 95 watches and spent
        1.92 s here, then ~0.7 s every minute after):

        * a timeframe is judged only when a NEW completed bucket has closed
          for it since this watch was last judged on it. A completed bar is
          the only thing that can change any of these answers, so a tick with
          warm caches and no new bucket does nothing at all;
        * the SMA/LRSI/ATR passes run OFF this thread, on the chart-data
          worker pool, over bars the cache hands out under its own lock; the
          fires come back through a QUEUED signal and the Qt thread only
          records, pushes and draws;
        * the H1 leg stays here because it reads BounceBot's M5 cache, which
          is not thread-safe - but it is bucket-gated like the rest and paced
          at `PULLBACK_H1_BATCH_LIMIT` watches a tick, oldest-waiting first.
          Pacing only: nothing is withheld and no watch is skipped, it is
          simply judged on the next tick instead.

        Nothing is fetched HERE either: the M15/M30 refresh is a request the
        cache answers on its own worker, batched, and it refuses more than one
        fetch per completed bucket.

        `None` from a rule is NOT MEASURED - too little history, or bars that
        stopped arriving - and a watch in that state simply waits. Uncertainty
        never deletes.

        **A new arm never fires on an old bounce** (repair RV-H1-ARM-TIME,
        2026-09-13, and the same fence inside `pullback_sma_reclaim`):
        `evaluate_h1_bars` returns `pre_arm` - neither fired nor
        `invalidated` - for a confirmation or invalidation whose event bar had
        already closed when the watch was armed, so this loop leaves that watch
        armed and writes no row, sends no push and draws no alert, exactly as
        it does for `awaiting_reclaim`.
        """
        moment = now or datetime.now()

        def _key(watch) -> tuple:
            return (watch.symbol, watch.kind, watch.watch_id, watch.armed_at)

        # The expiry pass runs over DECLINED rows too (review advisory 4): a
        # remembered row is still a ten-trading-day arm and must not outlive
        # one just because the trader turned it off.
        persistent = [
            watch
            for watch in self._chart_watches
            if watch.kind in PERSISTENT_WATCH_KINDS
        ]
        if persistent:
            kept, expired = self._expire_armed_watches(
                "chart_watches", persistent, now=moment
            )
            if expired:
                gone = {_key(watch) for watch in persistent} - {
                    _key(watch) for watch in kept
                }
                self._chart_watches = [
                    watch for watch in self._chart_watches if _key(watch) not in gone
                ]
                self._save_chart_watches()
                self._refresh_review_armed_kinds()
                self.armedWatchesChanged.emit()

        # Trader direction 2026-09-17: Pullback alerts are MANUAL only.  Clear
        # the retired automatic rows once so old claim/Focus pre-arms cannot
        # survive a restart, and never recreate them from the sweep.
        automatic = [watch for watch in self._chart_watches if self._is_auto_pullback_watch(watch)]
        if automatic:
            automatic_keys = {_key(watch) for watch in automatic}
            self._chart_watches = [
                watch for watch in self._chart_watches if _key(watch) not in automatic_keys
            ]
            self._save_chart_watches()
            self._refresh_review_armed_kinds()
            self.armedWatchesChanged.emit()
        armed = [
            watch
            for watch in self._chart_watches
            if watch.kind in PERSISTENT_WATCH_KINDS
            and not bool(getattr(watch, "declined", False))
        ]
        if not armed:
            return

        # The SMA legs first, off this thread; their fires come back through
        # `pullbackFiresReady`. Nothing below waits for them.
        self._dispatch_pullback_sma_evaluation(armed, moment)

        finished: set[tuple] = set()
        triggered: list[ChartWatchTrigger] = []
        for watch in self._h1_watches_due(armed, moment):
            try:
                h1_bars, _source = self._h1_bars_for_watch(watch, now=moment)
                result = evaluate_h1_bars(watch, h1_bars, now=moment)
            except Exception:
                logging.debug(
                    "H1 retester evaluation failed for %s", watch.symbol, exc_info=True
                )
                continue
            if result is None:
                continue
            if result.fired:
                finished.add(_key(watch))
                triggered.append(
                    ChartWatchTrigger(
                        watch=watch,
                        price=float(result.confirm_close or 0.0),
                        bar_dt=result.confirm_bar_dt or moment,
                        message=h1_bounce_message(watch, result),
                        resolved_side=result.side,
                        details={
                            "watch_id": watch.watch_id,
                            "reason": watch.reason,
                            "trigger": TRIGGER_H1_EMA15_BOUNCE,
                            "timeframe": "H1",
                            "rule_version": result.rule_version,
                            "bar_dt": (
                                result.confirm_bar_dt.isoformat()
                                if result.confirm_bar_dt is not None else ""
                            ),
                            "close": result.confirm_close,
                            "touch_bar_dt": (
                                result.touch_bar_dt.isoformat()
                                if result.touch_bar_dt is not None
                                else ""
                            ),
                            "confirm_bar_dt": (
                                result.confirm_bar_dt.isoformat()
                                if result.confirm_bar_dt is not None
                                else ""
                            ),
                            "ema": result.ema,
                            "atr": result.atr,
                            "distance_atr": result.distance_atr,
                            "skipped_bars": result.skipped_bars,
                            # One event per watch recording ALL the measured
                            # reasons - the packet's own rule.
                            "reasons": list(result.reasons),
                        },
                    )
                )
            elif result.reason == "invalidated":
                # The level did not hold. The arm is over and the trader is
                # told, but this is not an event worth a phone buzz: nothing
                # to do about it, and the chart says the same thing.
                finished.add(_key(watch))
                self._record_review_event(
                    "watch_invalidated",
                    symbol=watch.symbol,
                    side=watch.side,
                    detail={
                        "kind": watch.kind,
                        "watch_id": watch.watch_id,
                        "reasons": list(result.reasons),
                        "rule_version": result.rule_version,
                    },
                )
                self.statusChanged.emit(
                    f"{watch.symbol}: Pullback alert disarmed - price closed "
                    "through the H1 15-EMA against the setup."
                )

        if finished:
            self._chart_watches = [
                watch for watch in self._chart_watches if _key(watch) not in finished
            ]
            self._save_chart_watches()
            self._refresh_review_armed_kinds()
            self.armedWatchesChanged.emit()
        self._record_pullback_fires(triggered, moment)

    def _record_pullback_fires(self, triggered, moment: datetime) -> None:
        """One review row, one push and one feed row per fire. Qt thread only.

        The single place a fire becomes visible, whichever leg produced it, so
        the order the phone and the display see never depends on which rule
        spoke: the phone FIRST (a broken display path must never be able to
        suppress the buzz the trader armed this for), then the row.
        """
        for hit in triggered or ():
            measured = dict(hit.details or {})
            detail = {
                "kind": hit.watch.kind,
                "watch_id": hit.watch.watch_id,
                "message": str(hit.message or ""),
                "reasons": list(measured.get("reasons") or ()),
            }
            # What PCT-1 added, when the fire carries it: the trigger that
            # spoke, the series it spoke on, the rule sheet that decided and
            # the trader's "ideally it was below 50" label.
            for key in (
                "trigger", "timeframe", "rule_version", "lrsi_from_below_50",
                # Measured event provenance only.  Raw bar tapes never enter
                # review evidence through this seam.
                "bar_dt", "sma", "close", "lrsi", "atr",
                "cross_timeframe", "cross_bar_dt", "cross_lrsi", "sma_bar_dt",
                "touch_bar_dt", "confirm_bar_dt", "ema", "distance_atr",
                "skipped_bars",
                # The v2 dip gate's evidence (2026-09-23).
                "dip_path", "dip_bar_dt", "bear_flip_bar_dt", "d1_atr",
            ):
                if key in measured:
                    detail[key] = measured[key]
            self._record_review_event(
                "watch_fired",
                symbol=hit.watch.symbol,
                side=str(getattr(hit, "resolved_side", "") or hit.watch.side).upper(),
                detail=detail,
            )
            self._push_armed_watch(hit)
            self.add_alert(self._chart_watch_alert(hit, moment))

    #: The source text an auto-armed Pullback alert carries, per source. The
    #: absence of one of these is how the sweep knows a watch is the trader's
    #: own click and leaves it alone - the same "absence of a marker means the
    #: trader owns it" rule Focus provenance holds.
    PULLBACK_AUTO_SOURCES = {
        "claim": "auto: claimed pick",
        "focus": "auto: swing Focus",
    }

    def _is_auto_pullback_watch(self, watch) -> bool:
        return (
            str(getattr(watch, "kind", "") or "") == PULLBACK_KIND
            and str(getattr(watch, "source_text", "") or "")
            in set(self.PULLBACK_AUTO_SOURCES.values())
        )

    #: How many watches the Qt-thread H1 leg judges per tick. Pacing only,
    #: the `AUTO_ADOPT_BATCH_LIMIT` precedent: nothing is withheld and no
    #: watch is skipped, one simply waits for the next 60-second tick. A
    #: completed H1 bar arrives once an hour, so the whole armed set is
    #: covered many times over before the next one can change any answer.
    PULLBACK_H1_BATCH_LIMIT = 12

    def _pullback_judged_marks(self) -> dict:
        marks = getattr(self, "_pullback_judged", None)
        if marks is None:
            marks = {}
            self._pullback_judged = marks
        return marks

    def _pullback_request_marks(self) -> dict:
        marks = getattr(self, "_pullback_requested", None)
        if marks is None:
            marks = {}
            self._pullback_requested = marks
        return marks

    def _request_pullback_cache(self, watch, interval_minutes: int, cache, moment) -> None:
        """Ask a cache once per completed bucket, including simple doubles."""
        try:
            end = intraday_last_bucket_end(moment, interval_minutes)
        except Exception:
            end = moment.replace(minute=0, second=0, microsecond=0)
        key = (str(getattr(watch, "watch_id", "") or watch.symbol), int(interval_minutes))
        if self._pullback_request_marks().get(key) == end:
            return
        self._pullback_request_marks()[key] = end
        try:
            cache.request(watch.symbol, now=moment)
        except Exception:  # pragma: no cover - never costs the poll
            logging.debug("Intraday history request failed for %s", watch.symbol, exc_info=True)

    @staticmethod
    def _pullback_cache_token(cache, symbol: str):
        """Cheap cache generation, with a small compatibility fallback."""
        reader = getattr(cache, "data_token", None)
        if callable(reader):
            try:
                return reader(symbol)
            except Exception as exc:
                note_swallowed("pullback cache token reader failed", exc, quiet=True)
        # Old test doubles have no generation reader.  They are tiny; the
        # fallback preserves their delivery semantics without touching real
        # cache snapshots on every desk poll.
        try:
            bars = cache.bars_for(symbol)
            return (len(bars), str((bars[-1] if bars else {}).get("dt", "")))
        except Exception:
            return None

    @classmethod
    def _pullback_cache_snapshot(cls, cache, symbol: str):
        reader = getattr(cache, "snapshot_for", None)
        if callable(reader):
            try:
                return reader(symbol)
            except Exception as exc:
                note_swallowed("pullback cache snapshot reader failed", exc, quiet=True)
        bars = cache.bars_for(symbol) if cache is not None else []
        return bars, cls._pullback_cache_token(cache, symbol) if cache is not None else None

    def _pullback_due(self, watch, interval_minutes: int, moment: datetime, token=None):
        """(is this timeframe worth judging, the bucket end that made it so).

        A completed bar is the only thing that can change any of these
        answers, so a watch judged on the bucket that is still the latest one
        is not judged again (review blocker 3b). A watch never judged is
        always due - the first tick after arming has to answer.
        """
        try:
            end = intraday_last_bucket_end(moment, interval_minutes)
        except Exception:  # pragma: no cover - a calendar refusal judges anyway
            return True, None
        if end is None:
            return True, None
        key = (str(getattr(watch, "watch_id", "") or watch.symbol), int(interval_minutes))
        previous = self._pullback_judged_marks().get(key)
        if isinstance(previous, tuple) and len(previous) == 2:
            previous_end, previous_token = previous
        else:  # pre-AR-1 in-memory mark
            previous_end, previous_token = previous, object()
        return (previous_end != end or previous_token != token), end

    def _mark_pullback_judged(self, watch, interval_minutes: int, end, token=None) -> None:
        if end is None:
            return
        key = (str(getattr(watch, "watch_id", "") or watch.symbol), int(interval_minutes))
        self._pullback_judged_marks()[key] = (end, token)

    @staticmethod
    def pullback_fire_key(trigger: str, timeframe: str) -> str:
        """The `ChartWatch.fired` key: trigger AND timeframe (review advisory 1).

        `sma_reclaim_lrsi` can fire on the M15 and the M30 of one watch, and a
        map keyed on the trigger alone kept only whichever wrote last - so a
        desk restart re-announced the other one.
        """
        return f"{trigger}@{timeframe}"

    def _pullback_should_push(self, watch, trigger: str) -> bool:
        """Does THIS fire earn a phone buzz? (review advisory 5, lead decision)

        The reviewer measured ~85 fires a session at 108 armed watches, 70 %
        of them `sma_retest` - a phone that buzzes eighty-five times is a
        phone the trader turns off, and then the two triggers that matter are
        lost with the rest. So a watch the DESK armed pushes the two entry
        triggers and writes the retest as a feed row and a review row without
        a push; a watch the TRADER armed by hand pushes all three, because
        they asked for that exact name by pressing the button. The trader may
        overrule this split; nothing is withheld either way - every fire is on
        the feed and in the evidence.
        """
        if not self._is_auto_pullback_watch(watch):
            return True
        return str(trigger or "") != TRIGGER_SMA_RETEST

    def _dispatch_pullback_sma_evaluation(self, armed, moment: datetime) -> bool:
        """Queue the M15/M30 legs that have something new to say. OFF-thread.

        Everything expensive - the SMA, the ATR and the LRSI over ~600 M15
        bars per watch - happens on a worker; this builds the job list, asks
        the caches to refresh, and returns. One job at a time: a second tick
        while one is running dispatches nothing and marks nothing, so the
        bucket it would have judged is simply judged on the next tick.
        """
        if getattr(self, "_pullback_eval_busy", False):
            return False
        self._pullback_judged_marks()  # creates the judged-marks dict on first use
        jobs: list[dict] = []
        for watch in armed:
            triggers = self._pullback_triggers(watch)
            timeframes = self._pullback_sma_timeframes(watch)
            if not timeframes or not any(name in triggers for name in self.PULLBACK_SMA_TRIGGERS):
                continue
            identity = str(getattr(watch, "watch_id", "") or watch.symbol)
            caches = {}
            # M30's frozen rule may read M15 as a companion reversal.  That
            # makes M15 available to M30 without making it an M15 job for a
            # narrow veto arm.
            cache_intervals = {minutes for minutes, _length in timeframes}
            if any(minutes == 30 for minutes, _length in timeframes):
                cache_intervals.add(15)
            for interval_minutes in cache_intervals:
                cache = self._intraday_history_cache(interval_minutes)
                if cache is None:
                    continue
                caches[interval_minutes] = cache
            tokens = {
                interval_minutes: self._pullback_cache_token(cache, watch.symbol)
                for interval_minutes, cache in caches.items()
            }
            due = []
            for interval_minutes, sma_length in timeframes:
                # M30's hold can be released by a newly delivered M15 cross,
                # even while the M30 snapshot has not changed.
                token = tokens.get(interval_minutes)
                if interval_minutes == 30:
                    token = (token, tokens.get(15))
                wanted, end = self._pullback_due(
                    watch, interval_minutes, moment, token
                )
                if wanted:
                    # The worker takes its own atomic cache snapshot and
                    # stamps that generation onto its result.  The dispatch
                    # packet carries only the scheduling fact, so a caller
                    # cannot mistake a pre-worker token for evaluated data.
                    due.append((interval_minutes, sma_length, end))
            if not due:
                continue
            for interval_minutes, cache in caches.items():
                if cache is not None:
                    self._request_pullback_cache(watch, interval_minutes, cache, moment)
            jobs.append(
                {
                    "watch": watch,
                    "identity": identity,
                    "triggers": triggers,
                    "due": due,
                    "tokens": tokens,
                    "caches": caches,
                    # The dip gate's daily ATR(20) source: the chart
                    # service's memoized D1 dicts, read-only on the worker.
                    "d1_bars": self._pullback_d1_bars(watch.symbol),
                    "marks": dict(getattr(watch, "fired", None) or {}),
                    "states": {
                        key: value
                        for key, value in self._pullback_episode_states().items()
                        if key[0] == identity
                    },
                }
            )
        if not jobs:
            return False
        self._pullback_eval_busy = True
        threading.Thread(
            target=self._run_pullback_sma_evaluation,
            args=(jobs, moment),
            name="pullback-eval",
            daemon=True,
        ).start()
        return True

    def _pullback_d1_bars(self, symbol: str) -> list:
        """Daily bars for the Pullback dip gate (cache only, never IB).

        The same memoized list `_d1_bars_for` hands every D1 poll; empty
        when the chart service has not cached the name yet, and then the
        gate's touch path is unknown and only a close break-and-reclaim
        dip can answer.
        """
        try:
            return list(self._d1_bars_for(symbol) or [])
        except Exception:  # pragma: no cover - a missing daily tape is unknown
            return []

    def _pullback_episode_states(self) -> dict:
        states = getattr(self, "_pullback_episodes", None)
        if states is None:
            states = {}
            self._pullback_episodes = states
        return states

    def _run_pullback_sma_evaluation(self, jobs, moment: datetime) -> None:
        """The worker. Pure computation over bars the caches hand out.

        Touches no widget and no panel state: it reads each cache under that
        cache's own lock, runs the frozen rule, and emits what it found. The
        Qt thread does the recording, the pushing and the drawing.
        """
        results: list[dict] = []
        try:
            for job in jobs:
                try:
                    results.append(self._evaluate_one_pullback_job(job, moment))
                except Exception:
                    logging.debug(
                        "Pullback SMA evaluation failed for %s",
                        getattr(job.get("watch"), "symbol", "?"),
                        exc_info=True,
                    )
        finally:
            try:
                self.pullbackFiresReady.emit((results, moment))
            except Exception:  # pragma: no cover - a dead panel loses the fires
                logging.debug("Pullback fires could not be delivered", exc_info=True)
            self._pullback_eval_busy = False

    def _evaluate_one_pullback_job(self, job, moment: datetime) -> dict:
        """One watch's due timeframes. Returns fires, new states and marks.

        One episode clock per (watch, timeframe, side) is carried between
        polls so a trigger speaks once per episode; the `fired` map on the
        watch is the PERSISTED backstop, so a desk restart does not
        re-announce a move the trader was already told about. A watch with no
        side of its own is asked both ways, exactly as the H1 leg is.
        """
        from indicators import pullback_sma_reclaim as rule

        watch = job["watch"]
        identity = job["identity"]
        triggers = job["triggers"]
        caches = job["caches"]
        marks = dict(job["marks"])
        states = dict(job["states"])
        sides = (
            (watch.side,) if watch.side in ("LONG", "SHORT") else ("LONG", "SHORT")
        )
        snapshots = {
            interval_minutes: self._pullback_cache_snapshot(
                caches[interval_minutes], watch.symbol
            ) if interval_minutes in caches else ([], None)
            for interval_minutes, _sma_length in self.PULLBACK_TIMEFRAMES
        }
        series = {interval_minutes: snapshot[0] for interval_minutes, snapshot in snapshots.items()}
        tokens = {interval_minutes: snapshot[1] for interval_minutes, snapshot in snapshots.items()}
        actual_due = []
        for due_row in job["due"]:
            interval_minutes, sma_length, end = due_row[:3]
            token = tokens.get(interval_minutes)
            if interval_minutes == 30:
                token = (token, tokens.get(15))
            actual_due.append((interval_minutes, sma_length, end, token))
        fires: list[dict] = []
        for due_row in actual_due:
            interval_minutes, sma_length, _end = due_row[:3]
            for side in sides:
                state_key = (identity, interval_minutes, side)
                # The trader's M30 leg may be answered by an M15 reversal, so
                # the companion series rides along - it never moves the SMA.
                companion_minutes = (
                    15 if interval_minutes == rule.RECLAIM_THEN_LRSI_MINUTES else None
                )
                result = rule.evaluate(
                    series.get(interval_minutes) or [],
                    side=side.lower(),
                    sma_length=sma_length,
                    bar_minutes=interval_minutes,
                    armed_at=watch.armed_at,
                    now=moment,
                    episode_state=states.get(state_key),
                    companion_bars=(
                        series.get(companion_minutes) if companion_minutes else None
                    ),
                    companion_minutes=companion_minutes,
                    daily_bars=job.get("d1_bars") or None,
                )
                if result is None:
                    continue  # NOT MEASURED - the watch simply waits
                states[state_key] = result.episode_state
                for fire in result.fired:
                    if fire.trigger not in triggers:
                        continue
                    stamp = fire.bar_dt.isoformat()
                    mark_key = self.pullback_fire_key(fire.trigger, fire.timeframe)
                    if self._pullback_mark_covers(marks.get(mark_key), stamp):
                        continue  # already announced, before this restart
                    marks[mark_key] = stamp
                    fires.append(
                        {
                            "side": side,
                            "price": float(fire.close),
                            "bar_dt": fire.bar_dt,
                            "message": (
                                f"{watch.symbol} {side}: Pullback - {fire.message}"
                            ),
                            "details": {
                                "watch_id": watch.watch_id,
                                "reason": watch.reason,
                                "trigger": fire.trigger,
                                "timeframe": fire.timeframe,
                                "rule_version": result.rule_version,
                                "lrsi_from_below_50": bool(fire.lrsi_from_below_50),
                                "bar_dt": stamp,
                                "sma": fire.sma,
                                "close": fire.close,
                                "lrsi": fire.lrsi,
                                "atr": fire.atr,
                                "cross_timeframe": fire.cross_timeframe,
                                "cross_bar_dt": (
                                    fire.cross_bar_dt.isoformat()
                                    if fire.cross_bar_dt is not None else ""
                                ),
                                "cross_lrsi": fire.cross_lrsi,
                                "sma_bar_dt": (
                                    fire.sma_bar_dt.isoformat()
                                    if fire.sma_bar_dt is not None else ""
                                ),
                                "dip_path": fire.dip_path or "",
                                "dip_bar_dt": (
                                    fire.dip_bar_dt.isoformat()
                                    if fire.dip_bar_dt is not None else ""
                                ),
                                "bear_flip_bar_dt": (
                                    fire.bear_flip_bar_dt.isoformat()
                                    if fire.bear_flip_bar_dt is not None else ""
                                ),
                                "d1_atr": fire.d1_atr,
                            },
                        }
                    )
        return {
            "identity": identity,
            "watch_id": str(getattr(watch, "watch_id", "") or ""),
            "symbol": watch.symbol,
            "fires": fires,
            "states": states,
            "marks": marks,
            "tokens": tokens,
            "due": actual_due,
        }

    @staticmethod
    def _pullback_mark_covers(mark, stamp: str) -> bool:
        """True when a persisted event is this event or a later old stamp."""
        if not mark:
            return False
        try:
            prior = datetime.fromisoformat(str(mark))
            event = datetime.fromisoformat(str(stamp))
            from market_session import get_market_local_timezone

            market_tz, _name = get_market_local_timezone()
            if prior.tzinfo is None:
                prior = prior.replace(tzinfo=market_tz)
            else:
                prior = prior.astimezone(market_tz)
            if event.tzinfo is None:
                event = event.replace(tzinfo=market_tz)
            else:
                event = event.astimezone(market_tz)
            return prior >= event
        except (TypeError, ValueError):
            return str(mark) == stamp

    def _on_pullback_fires(self, payload) -> None:
        """The worker's answer, back on the Qt thread. Records, pushes, draws.

        A watch disarmed, declined or retired while the worker ran is dropped
        here: its fire is about an arm that no longer exists.
        """
        self._pullback_eval_busy = False
        try:
            results, moment = payload
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return
        if not results:
            return
        states = self._pullback_episode_states()
        live = {
            str(getattr(watch, "watch_id", "") or watch.symbol): watch
            for watch in self._chart_watches
            if watch.kind == PULLBACK_KIND
            and not bool(getattr(watch, "declined", False))
        }
        updated: dict[str, dict] = {}
        triggered: list[ChartWatchTrigger] = []
        for result in results:
            identity = str(result.get("identity") or "")
            watch = live.get(identity)
            if watch is None:
                continue  # disarmed while the worker ran
            result_tokens = dict(result.get("tokens") or {})
            current_tokens = {
                minutes: self._pullback_cache_token(cache, watch.symbol)
                for minutes, cache in (
                    (minutes, self._intraday_history_cache(minutes))
                    for minutes, _length in self.PULLBACK_TIMEFRAMES
                )
            }
            if any(
                due_row[3] != (
                    (current_tokens.get(minutes), current_tokens.get(15))
                    if minutes == 30 else current_tokens.get(minutes)
                )
                for due_row in result.get("due") or ()
                for minutes in (due_row[0],)
                if len(due_row) > 3
            ):
                continue  # data changed after the worker snapshot; next tick owns it
            states.update(result.get("states") or {})
            for due_row in result.get("due") or ():
                interval_minutes, _length, end = due_row[:3]
                token = due_row[3] if len(due_row) > 3 else result_tokens.get(interval_minutes)
                # The captured token is the only mark valid for this answer.
                self._mark_pullback_judged(watch, interval_minutes, end, token)
            fires = result.get("fires") or []
            if not fires:
                continue
            updated[identity] = dict(result.get("marks") or {})
            for fire in fires:
                details = dict(fire.get("details") or {})
                details["push"] = self._pullback_should_push(
                    watch, str(details.get("trigger") or "")
                )
                triggered.append(
                    ChartWatchTrigger(
                        watch=watch,
                        price=float(fire.get("price") or 0.0),
                        bar_dt=fire.get("bar_dt") or moment,
                        message=str(fire.get("message") or ""),
                        resolved_side=str(fire.get("side") or ""),
                        details=details,
                    )
                )
        if updated:
            # A Pullback alert is a STANDING arm: a trigger firing records the
            # bar it fired on and the watch stays armed for the next one, up
            # to expiry or a disarm. Only the H1 leg retires the watch,
            # because a completed retest is the pattern finishing.
            self._chart_watches = [
                replace(
                    watch,
                    fired=updated[
                        str(getattr(watch, "watch_id", "") or watch.symbol)
                    ],
                )
                if str(getattr(watch, "watch_id", "") or watch.symbol) in updated
                and watch.kind == PULLBACK_KIND
                else watch
                for watch in self._chart_watches
            ]
            self._save_chart_watches()
        self._record_pullback_fires(triggered, moment)

    def _active_d1_claims(self, path: Path, moment: datetime):
        """Today's active claims, from an mtime-keyed cache. None if unreadable.

        The sweep runs every 60 seconds on the Qt thread and the claims file
        only changes when the trader claims or drops something, so re-parsing
        it every minute is a file read and a JSON pass bought for nothing
        (PCT-1 review advisory 3). Keyed on (mtime, size) AND the session date,
        because the fade is a session clock - the same key
        `_active_claim_keys` already uses beside it.
        """
        try:
            stat = path.stat()
            stamp: object = (stat.st_mtime_ns, stat.st_size, moment.date())
        except OSError:
            # No file yet is "no claims", not "unreadable": a desk that has
            # never claimed anything must still be able to sweep.
            return []
        cached = getattr(self, "_auto_claim_cache", None)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        try:
            import claimed_picks

            rows = [
                dict(row)
                for row in claimed_picks.active_claims(path, as_of=moment.date())
            ]
        except Exception:  # noqa: BLE001 - an unreadable store arms nothing
            logging.debug("Pullback auto-arm: claims unreadable", exc_info=True)
            return None
        self._auto_claim_cache = (stamp, rows)
        return rows

    def _auto_pullback_sources(self, moment: datetime):
        """`(symbol, side) -> source text` for every pick that arms itself.

        The trader's answer to "which names does this watch?" (2026-09-15) was
        *"Chart arm + my picks"*: every ACTIVE claimed D1 pick and every swing
        Focus name. A claim outranks a Focus entry on the same name because it
        is the more specific statement. ``None`` when a store could not be
        read at all - and then the sweep does nothing, because uncertainty
        never retires a watch.
        """
        wanted: dict[tuple[str, str], str] = {}
        path = getattr(self, "_claimed_picks_path", None)
        if path is not None:
            rows = self._active_d1_claims(Path(path), moment)
            if rows is None:
                return None
            for row in rows:
                if str(row.get("horizon") or "").strip().lower() != "d1":
                    continue  # an M5 claim is not a multi-day thesis
                symbol = str(row.get("symbol") or "").strip().upper()
                side = str(row.get("side") or "").strip().upper()
                if symbol and side in ("LONG", "SHORT"):
                    wanted[(symbol, side)] = self.PULLBACK_AUTO_SOURCES["claim"]
        service = getattr(self, "focus_service", None)
        if service is not None:
            try:
                swing = service.all_focus(category="swing") or {}
            except Exception:  # noqa: BLE001 - as above
                logging.debug("Pullback auto-arm: Focus unreadable", exc_info=True)
                return None
            for side_key, symbols in swing.items():
                side = str(side_key or "").strip().upper()
                if side not in ("LONG", "SHORT"):
                    continue
                for name in symbols or ():
                    symbol = str(name or "").strip().upper()
                    if symbol:
                        wanted.setdefault(
                            (symbol, side), self.PULLBACK_AUTO_SOURCES["focus"]
                        )
        return wanted

    def _sweep_auto_pullback_watches(self, moment: datetime) -> bool:
        """Arm the trader's own picks, retire the ones whose pick is gone.

        Runs inside the armed poll, BEFORE evaluation, in every Auto mode -
        these are the trader's own picks and they ride the armed-alert
        exception that pushes in DESK, AWAY, EVENING and OFF alike
        (`docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md`). Returns True when the
        armed set changed.

        Four rules the tests pin:

        * one watch per (symbol, side), never a second on the next tick;
        * a watch the trader DISARMED is remembered as `declined` and is not
          armed again while that claim or pick lives;
        * the sweep owns only what it armed - a watch with no `auto:` source
          is the trader's own click and is never retired by it;
        * **it writes `auto_arm_watch`, never `arm_watch`** (review blocker 2).
          `review_learning.TAKE_ACTIONS` holds `arm_watch` because a trader
          pressing that button IS a take; this sweep armed 95 watches on its
          first tick against the live stores, and scoring those as takes would
          have said the trader took ninety-five setups in a minute.

        **One save, one event batch, one emit** (review blocker 3a). Arming 95
        watches through the public button - a file write, a kernel-locked
        append and a widget rebuild EACH - is most of what that first tick
        cost on the Qt thread.
        """
        wanted = self._auto_pullback_sources(moment)
        if wanted is None:
            return False
        # Existence is per SYMBOL, not per (symbol, side): the arm surface
        # itself is per (symbol, kind), so a second side of the same name
        # cannot be armed and asking every minute would only emit "already
        # armed" sixty times an hour. A DECLINED row counts as existing -
        # that is the whole point of remembering it.
        existing = {
            watch.symbol
            for watch in self._chart_watches
            if watch.kind == PULLBACK_KIND
        }
        events: list[dict] = []
        retired = [
            watch
            for watch in self._chart_watches
            if self._is_auto_pullback_watch(watch)
            and (watch.symbol, watch.side) not in wanted
        ]
        if retired:
            gone = {(watch.symbol, watch.side) for watch in retired}
            self._chart_watches = [
                watch
                for watch in self._chart_watches
                if not (
                    self._is_auto_pullback_watch(watch)
                    and (watch.symbol, watch.side) in gone
                )
            ]
            for watch in retired:
                existing.discard(watch.symbol)
                events.append(
                    {
                        "action": "watch_retired_source_gone",
                        "symbol": watch.symbol,
                        "side": watch.side,
                        "detail": {
                            "kind": watch.kind,
                            "watch_id": watch.watch_id,
                            "source_text": watch.source_text,
                            "declined": bool(getattr(watch, "declined", False)),
                            "auto": True,
                        },
                    }
                )
        armed_now: list = []
        for (symbol, side), source_text in sorted(wanted.items()):
            if symbol in existing:
                continue
            watch = arm_chart_watch(
                PULLBACK_KIND,
                symbol,
                side,
                (),  # a pullback watch takes no M5 baseline
                now=moment,
                source_text=source_text,
            )
            armed_now.append(watch)
            existing.add(symbol)
            events.append(
                {
                    "action": "auto_arm_watch",
                    "symbol": symbol,
                    "side": side,
                    "detail": {
                        "kind": PULLBACK_KIND,
                        "watch_id": watch.watch_id,
                        "source_text": source_text,
                        "auto": True,
                    },
                }
            )
        if armed_now:
            self._chart_watches = list(self._chart_watches) + armed_now
        if not events:
            return False
        self._save_chart_watches()
        self._record_review_events(events)
        self._refresh_review_armed_kinds()
        self.armedWatchesChanged.emit()
        return True

