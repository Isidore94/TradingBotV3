"""Keeps the review chart's M5 bars current while an alert waits in the queue.

The problem this exists for: ``bot.latest_bars`` is only rewritten when the
scan loop reaches a symbol, and a full scan runs ~28 minutes. An alert the
trader opens twenty minutes after it fired therefore rebuilds its chart from
the bars of its last scan and, because the rebuilt series is byte-identical,
the chart's own repaint guard correctly decides nothing changed. Refreshing
harder against the same cache changes nothing - only a refetch does.

Three rules shape this module:

* **Display-only.** Fetched bars are held here and handed to the chart. They
  are never written back into ``latest_bars``, which is a detector input and
  a warehouse capture source; feeding it from a chart view would change what
  the champions see (plan.md sec 5).
* **Bounded.** IB allows roughly 60 historical requests per 10 minutes and
  the champion scan needs that budget. A per-symbol cooldown plus a small
  refresh set keeps this to a few requests a minute, so a 60-deep alert queue
  cannot starve the scanner.
* **One owner.** A single service instance owns the cache, the cooldowns and
  the worker thread (plan.md sec 5). The Alert Center drives it from the 30s
  tick it already runs rather than starting a second timer.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta

from PySide6.QtCore import QObject, Signal

# Three M5 bars. The trader's own threshold: below this the chart is close
# enough to now, above it the candles are visibly behind the tape.
STALE_AFTER_BARS = 3
M5_BAR_MINUTES = 5
STALE_AFTER = timedelta(minutes=STALE_AFTER_BARS * M5_BAR_MINUTES)

# A symbol refetched this recently is left alone even if it still looks old:
# a name that has stopped trading would otherwise be refetched every tick
# forever, spending the IB budget on bars that cannot move.
REFRESH_COOLDOWN = timedelta(minutes=5)

# How far past the displayed alert to pre-warm. Small on purpose - see the
# budget rule above.
DEFAULT_LOOKAHEAD = 3


def _last_dt(bars) -> datetime | None:
    """The last bar's timestamp, or None if these bars carry none."""
    if not bars:
        return None
    last = bars[-1].get("dt")
    return last if isinstance(last, datetime) else None


def bars_age(bars, *, now=None) -> timedelta | None:
    """How far behind ``now`` the last bar is, or None if there are no bars."""
    if not bars:
        return None
    last = bars[-1].get("dt")
    if not isinstance(last, datetime):
        return None
    moment = now or datetime.now(tz=last.tzinfo)
    if last.tzinfo is None and moment.tzinfo is not None:
        moment = moment.replace(tzinfo=None)
    elif last.tzinfo is not None and moment.tzinfo is None:
        moment = moment.replace(tzinfo=last.tzinfo)
    # The stamp is the bar's START, so a just-closed bar is already 5 minutes
    # old by its timestamp and would read as stale without this.
    return max(timedelta(0), moment - last - timedelta(minutes=M5_BAR_MINUTES))


def bars_are_stale(bars, *, now=None) -> bool:
    """Whether these bars are far enough behind to be worth a refetch.

    Empty bars are NOT stale: a symbol outside the scan set has nothing to
    refresh from a cache, and the chart already says so in its own words.
    """
    age = bars_age(bars, now=now)
    return age is not None and age >= STALE_AFTER


class ChartBarRefreshService(QObject):
    """Display-only M5 refetches for the alert the trader is about to reach."""

    barsRefreshed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._bars: dict[str, list[dict]] = {}
        self._fetched_at: dict[str, datetime] = {}
        self._attempted_at: dict[str, datetime] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    def bars_for(self, symbol: str) -> list[dict]:
        """The freshest display bars held for ``symbol``, or []."""
        with self._lock:
            return list(self._bars.get(str(symbol or "").strip().upper()) or [])

    def fetched_at(self, symbol: str) -> datetime | None:
        with self._lock:
            return self._fetched_at.get(str(symbol or "").strip().upper())

    def best_bars(self, symbol: str, cached) -> list[dict]:
        """Whichever of the bot cache and our refetch reaches further forward.

        Never returns the older series: a refetch that came back short (a
        halted name, a partial provider answer) must not replace a longer
        cached one just because it is newer.
        """
        mine = self.bars_for(symbol)
        if not mine:
            return list(cached or [])
        if not cached:
            return mine
        # Compared by last-bar timestamp, NOT by age against the wall clock:
        # age is measured from "now", so two series that are both behind clamp
        # to the same value and the fresher one loses the tie.
        mine_last = _last_dt(mine)
        cached_last = _last_dt(cached)
        if mine_last is None:
            return list(cached)
        if cached_last is None or mine_last > cached_last:
            # ...unless the fresh answer came back truncated. Three current
            # candles are not worth the history the trader was reading.
            return mine if len(mine) >= len(cached) else list(cached)
        return list(cached)

    # ------------------------------------------------------------------
    def refresh_if_stale(self, symbols, cached_for, bot, *, now=None) -> list[str]:
        """Queue refetches for whichever of ``symbols`` is behind.

        ``cached_for(symbol)`` supplies the bars the chart would otherwise
        draw, so the decision is made against what the trader would actually
        see. Returns the symbols queued, for tests and logging.
        """
        if bot is None:
            return []
        now = now or datetime.now()
        wanted: list[str] = []
        for raw in symbols or ():
            symbol = str(raw or "").strip().upper()
            if not symbol or symbol in wanted:
                continue
            if not self._cooldown_expired(symbol, now):
                continue
            try:
                cached = cached_for(symbol)
            except Exception:
                cached = []
            # An empty cache means the symbol left the scan set. Refetching it
            # IS the only way to chart it at all, so it counts as stale.
            if cached and not bars_are_stale(self.best_bars(symbol, cached), now=now):
                continue
            wanted.append(symbol)
        if not wanted:
            return []
        with self._lock:
            for symbol in wanted:
                self._attempted_at[symbol] = now
            if self._thread is not None and self._thread.is_alive():
                # One worker at a time: the fetch is blocking and serialised by
                # IB anyway, so a second thread would only deepen the queue.
                return []
            self._thread = threading.Thread(
                target=self._worker,
                args=(list(wanted), bot),
                name="chart-m5-refresh",
                daemon=True,
            )
            self._thread.start()
        return wanted

    def _cooldown_expired(self, symbol: str, now: datetime) -> bool:
        with self._lock:
            attempted = self._attempted_at.get(symbol)
        return attempted is None or (now - attempted) >= REFRESH_COOLDOWN

    def _worker(self, symbols: list[str], bot) -> None:
        for symbol in symbols:
            try:
                bars = bot.fetch_m5_chart_bars(symbol)
            except Exception:
                logging.debug("Chart M5 refresh failed for %s.", symbol, exc_info=True)
                continue
            if not bars:
                continue
            with self._lock:
                self._bars[symbol] = bars
                self._fetched_at[symbol] = datetime.now()
            try:
                self.barsRefreshed.emit(symbol)
            except RuntimeError:
                return  # the panel went away mid-fetch

    def shutdown(self) -> None:
        """Drop the cache. The worker is a daemon and is left to finish."""
        with self._lock:
            self._bars.clear()
            self._fetched_at.clear()
            self._attempted_at.clear()


_SHARED: ChartBarRefreshService | None = None


def shared_refresh_service() -> ChartBarRefreshService:
    """The one refresh service, mirroring ``chart_data_service.shared_service``.

    A module singleton rather than an injected object because every chart that
    can show an alert needs to READ the refreshed bars, while only the Alert
    Center drives the refresh. One owner of the cache and the cooldowns is the
    part plan.md sec 5 cares about.
    """
    global _SHARED
    if _SHARED is None:
        _SHARED = ChartBarRefreshService()
    return _SHARED


def reset_shared_refresh_service() -> None:
    """Test seam: drop the singleton so cases cannot leak cache into each other."""
    global _SHARED
    if _SHARED is not None:
        _SHARED.shutdown()
    _SHARED = None
