"""The H1 leg of the Pullback alert: its history cache, warm-up notes and per-tick due list.

Moved verbatim out of `AlertCenterPanel`, which inherits this mixin.
"""

from __future__ import annotations

import logging
from datetime import datetime

from chart_watch import h1_bars_for_watch, H1_SOURCE_YFINANCE, TRIGGER_H1_EMA15_BOUNCE
from intraday_history import H1_MINUTES as H1_INTERVAL_MINUTES


class H1RetesterMixin:
    """`AlertCenterPanel` methods for the Pullback alert's H1 EMA15 retest leg."""

    def _h1_history_cache(self):
        """The H1 fallback cache, built on first use. None if unavailable.

        Lead decision 2026-09-13 (the trader may overrule): the desk's cached
        M5 window aggregates to ~35 completed H1 bars against a 45-bar warm-up,
        so an armed watch would never fire. The missing history is fetched for
        ARMED SYMBOLS ONLY, through yfinance, on the cache's own worker thread -
        the group RS/RW tape precedent: zero IB traffic, no engine change. The
        desk's own cache stays primary and a full one never touches the network.
        """
        cache = getattr(self, "_h1_history", None)
        if cache is None:
            try:
                from h1_history import H1HistoryCache

                cache = H1HistoryCache()
            except Exception:  # pragma: no cover - yfinance/env unavailable
                logging.debug("H1 history fallback unavailable", exc_info=True)
                cache = False
            self._h1_history = cache
        return cache or None

    def _h1_bars_for_watch(self, watch, *, now: datetime | None = None):
        """(the H1 series this watch is judged on, which source it came from).

        Reads only what is already in memory. When the cached window is short
        of the warm-up it ASKS the fallback for a refresh and returns whatever
        it has right now - the first cycle after arming simply reports "not
        measured", and the answer lands before the next completed bucket.

        **The need is measured on the PRIMARY series, never on the chosen one**
        (repair RV-H1-HISTORY, review blocker B1, 2026-09-13). Asking only when
        the CHOSEN series was short meant that once the fallback held 45 bars
        nothing ever asked again, while the desk's own window stayed at ~35 for
        ever - so the watch was judged on ageing bars until the rule's 24 h
        staleness answered "not measured" for good. `h1_bars_for_watch` hands
        back the yfinance series only when the primary is short of the warm-up,
        so that source IS the short answer; otherwise `bars` is the primary and
        its own length is the test. No second aggregation pass.
        """
        m5_bars = self._m5_bars_for(watch.symbol, sessions=self.H1_WATCH_M5_SESSIONS)
        cache = self._h1_history_cache()
        fallback = cache.bars_for(watch.symbol) if cache is not None else None
        bars, source = h1_bars_for_watch(m5_bars, fallback_h1_bars=fallback)
        if cache is not None:
            primary_is_short = (
                source == H1_SOURCE_YFINANCE or len(bars) < self._h1_warmup_bars()
            )
            if primary_is_short:
                self._request_pullback_cache(
                    watch, H1_INTERVAL_MINUTES, cache, now or datetime.now()
                )
        return bars, source

    @staticmethod
    def _h1_warmup_bars() -> int:
        try:
            from indicators.h1_ema_bounce import WARMUP_BARS

            return int(WARMUP_BARS)
        except Exception:  # pragma: no cover - the module is first-party
            return 45

    def _h1_watch_note(self, watch) -> str:
        """The H1 leg's half of the health cell - the WS-10C string, unchanged."""
        have, needed, source = self._h1_warmup_counts(watch)
        if have is None:
            return ""
        if have >= needed:
            # It can answer - and the trader can see WHICH history answered it.
            if source != H1_SOURCE_YFINANCE:
                return "H1 from cache"
            if self._h1_refresh_failed(watch.symbol):
                return "H1 from yfinance (stale - last refresh failed)"
            return "H1 from yfinance"
        cache = self._h1_history_cache()
        if cache is not None and cache.unavailable(watch.symbol):
            return f"not measured ({have} of {needed} H1 bars, yfinance unavailable)"
        return f"not measured ({have} of {needed} H1 bars)"

    def _h1_refresh_failed(self, symbol) -> bool:
        """True when the fallback's LAST refresh for this symbol failed.

        A note never costs its caller, and a cache stand-in without the reader
        simply answers "not stale" rather than raising in the health cell.
        """
        cache = self._h1_history_cache()
        reader = getattr(cache, "last_refresh_failed", None)
        if reader is None:
            return False
        try:
            return bool(reader(symbol))
        except Exception:  # pragma: no cover - a health cell never raises
            return False

    def _h1_warmup_counts(self, watch) -> tuple[int | None, int, str]:
        """(H1 bars available, bars the rule needs, source). One O(bars) pass."""
        try:
            bars, source = self._h1_bars_for_watch(watch)
        except Exception:  # pragma: no cover - a note never costs the caller
            return (None, 0, "")
        return (len(bars), self._h1_warmup_bars(), source)

    def _h1_warmup_note(self, watch) -> str:
        """What the H1 rule can see for this watch RIGHT NOW, measured.

        The rule needs `WARMUP_BARS` completed H1 bars before it will answer at
        all, and the desk's cached M5 window is five sessions (SN2), which
        aggregates to about 35. Saying so at the click - from the bars actually
        in hand, never from a remembered number - is the difference between a
        watch that is waiting and a watch the trader thinks is watching.
        """
        have, needed, source = self._h1_warmup_counts(watch)
        if have is None:
            return "It evaluates on every completed H1 bar."
        if have >= needed:
            where = "from yfinance" if source == H1_SOURCE_YFINANCE else "cached"
            return f"{have} completed H1 bars {where} - it evaluates on every new one."
        return (
            f"Only {have} completed H1 bars are available and the rule needs "
            f"{needed}; the missing history is being fetched, so it reports "
            "NOT MEASURED until it lands."
        )

    #: How many sessions of cached M5 bars an H1 retester asks for. The desk's
    #: own window is five (SN2) and `_m5_bars_for` never fetches, so this is a
    #: ceiling rather than a request: ask for what the rule's warm-up wants and
    #: take what is cached.
    H1_WATCH_M5_SESSIONS = 10

    def _h1_watches_due(self, armed, moment: datetime) -> list:
        """Which watches get their H1 leg read on THIS tick.

        Gated on a new completed H1 bucket and then paced, oldest-waiting
        first so nothing starves. The mark is written here rather than after
        the evaluation because an evaluation that raises has still asked the
        question this bucket poses.
        """
        due = []
        for watch in armed:
            if (
                TRIGGER_H1_EMA15_BOUNCE not in self._pullback_triggers(watch)
                or not self._pullback_uses_h1(watch)
            ):
                continue
            cache = self._h1_history_cache()
            token = self._pullback_cache_token(cache, watch.symbol) if cache is not None else None
            wanted, end = self._pullback_due(watch, H1_INTERVAL_MINUTES, moment, token)
            if not wanted:
                continue
            if self._m5_unknown(watch.symbol, sessions=self.H1_WATCH_M5_SESSIONS):
                continue  # not marked judged: it is judged once its bars land
            due.append((watch, end, token))
        due.sort(key=lambda pair: str(getattr(pair[0], "symbol", "")))
        taken = due[: max(1, int(self.PULLBACK_H1_BATCH_LIMIT))]
        for watch, end, token in taken:
            self._mark_pullback_judged(watch, H1_INTERVAL_MINUTES, end, token)
        return [watch for watch, _end, _token in taken]

