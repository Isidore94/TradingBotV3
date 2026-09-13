"""H1 bar history for an ARMED watch, when the desk's own cache is too short.

Why this exists (lead decision 2026-09-13, WISHLIST 10C; the trader may
overrule). The H1 retester rule needs 45 completed hourly bars before it will
answer. The desk's cached M5 window is five regular-hours sessions - one
`"5 D"` IB fetch, kept and extended by SN2 - which aggregates to about 35. Two
sources were checked and neither closes the gap: WS-CH's "Load older" is the
SAME `m5_chart_bars` read with a ceiling of ten sessions on the ASK, and the
durable H1 store `project_paths.MASTER_AVWAP_INTRADAY_BARS_DIR` has never been
written on this desk. A watch that cannot fire for a whole test week gives the
trader nothing to judge, so the missing history is fetched.

The shape follows the group RS/RW tape precedent exactly: **its own clock, one
outbound `yfinance` call, zero IB traffic, and no change to any engine.**

Five rules this module holds:

* **Armed symbols only, and only when short.** Nothing here runs for a symbol
  the trader has not armed a watch on, and nothing runs while the M5 cache can
  answer on its own. The cache stays PRIMARY.
* **At most one fetch per completed H1 bar per symbol.** A new bar is the only
  thing that can change the answer, so a 60-second poll asking sixty times an
  hour would be fifty-nine wasted round trips. A fetch already in flight is
  never duplicated.
* **Never on the Qt thread.** `request` starts a one-shot daemon thread and
  returns immediately; `bars_for` reads the finished result out of memory.
  A poll therefore never blocks on the network, and the first cycle after
  arming simply reports "not measured" until the answer lands.
* **Completed bars only.** The forming hour is dropped through the one rule
  (`completed_bars.is_completed_bar`), and timestamps are CONVERTED to the
  desk's market-local clock with `astimezone`, never stripped - that is the
  same fault N1 recorded on the capture sidecars.
* **A failure is a refusal, not a value.** An empty or unreadable download
  leaves the cache untouched and is reported as unavailable; nothing here ever
  invents a bar, and nothing it fails to do can cost the watch.

In memory only: no file, no store, no evidence row. Restarting the desk simply
fetches again.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Callable

H1_MINUTES = 60
H1_SPAN = timedelta(minutes=H1_MINUTES)

#: How much history to ask for. `1mo` of hourly bars is ~22 sessions (~150
#: completed bars) - comfortably past the 45-bar warm-up with room for
#: holidays, and small enough to stay one quick request.
DEFAULT_PERIOD = "1mo"
DEFAULT_INTERVAL = "60m"

#: Source labels the armed inventory prints, so the trader can see which
#: history a verdict was measured on.
SOURCE_CACHE = "cache"
SOURCE_YFINANCE = "yfinance"


def _download(symbol: str, *, period: str, interval: str):
    """One outbound yfinance call. Imported lazily, like every other caller."""
    import yfinance as yf

    return yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        prepost=False,  # regular hours, the same session the rule is written for
        threads=False,
    )


def _market_local(stamp: Any) -> datetime | None:
    """A bar stamp on the desk's market-local clock, naive.

    yfinance returns an exchange-tz-aware index. The chart-watch store's whole
    convention is naive market-local, so the offset is CONVERTED through
    `astimezone` and only then dropped - stripping it would move every bar by
    the exchange offset and silently mis-time the touch.
    """
    if stamp is None:
        return None
    try:
        moment = stamp.to_pydatetime()  # pandas Timestamp
    except AttributeError:
        moment = stamp
    if not isinstance(moment, datetime):
        return None
    if moment.tzinfo is None:
        return moment
    try:
        from market_session import get_market_local_timezone

        local_tz, _ = get_market_local_timezone()
    except Exception:  # pragma: no cover - settings unavailable
        return moment.astimezone().replace(tzinfo=None)
    return moment.astimezone(local_tz).replace(tzinfo=None)


def frame_to_h1_bars(frame, *, now: datetime) -> list[dict[str, Any]]:
    """Completed H1 bar dicts from a yfinance frame, oldest first.

    Shaped exactly like `BounceBot.m5_chart_bars` output so the rule cannot
    tell the two sources apart, and filtered through the ONE completed-bar rule
    so the hour still printing never reaches it.
    """
    if frame is None:
        return []
    try:
        if frame.empty:
            return []
    except AttributeError:
        return []
    columns = getattr(frame, "columns", None)
    if columns is not None and getattr(columns, "nlevels", 1) > 1:
        # A single-ticker download can still come back column-grouped.
        try:
            frame = frame.droplevel(-1, axis=1)
        except Exception:
            try:
                frame = frame.droplevel(0, axis=1)
            except Exception:
                return []
    from completed_bars import is_completed_bar

    bars: list[dict[str, Any]] = []
    for stamp, row in frame.iterrows():
        moment = _market_local(stamp)
        if moment is None:
            continue
        try:
            bar = {
                "dt": moment,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
        if any(value != value for value in (bar["open"], bar["high"], bar["low"], bar["close"])):
            continue  # NaN is missing data, never a price
        if not is_completed_bar(bar, H1_MINUTES, now=now):
            continue
        bars.append(bar)
    bars.sort(key=lambda bar: bar["dt"])
    return bars


class H1HistoryCache:
    """Per-symbol H1 bars, refreshed at most once per completed hour."""

    def __init__(
        self,
        *,
        downloader: Callable[..., Any] | None = None,
        period: str = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL,
    ) -> None:
        self._downloader = downloader or _download
        self._period = period
        self._interval = interval
        self._lock = threading.Lock()
        self._bars: dict[str, list[dict[str, Any]]] = {}
        #: symbol -> the completed hour the last ATTEMPT was made for, so a
        #: failure is not retried sixty times before the next bar prints.
        self._attempted_hour: dict[str, datetime] = {}
        self._in_flight: set[str] = set()
        self._failed: set[str] = set()

    # -- reads (Qt thread) ---------------------------------------------
    def bars_for(self, symbol: str) -> list[dict[str, Any]]:
        """Whatever has already been fetched. Never blocks, never fetches."""
        key = str(symbol or "").strip().upper()
        with self._lock:
            return list(self._bars.get(key) or ())

    def unavailable(self, symbol: str) -> bool:
        """True when the last attempt for this symbol failed and none succeeded."""
        key = str(symbol or "").strip().upper()
        with self._lock:
            return key in self._failed and not self._bars.get(key)

    # -- the fetch (worker thread) -------------------------------------
    def request(self, symbol: str, *, now: datetime | None = None) -> bool:
        """Ask for this symbol's H1 history. Returns True if a fetch started.

        Refused - quietly and cheaply - when one is already in flight or when
        this completed hour has already been attempted.
        """
        key = str(symbol or "").strip().upper()
        if not key:
            return False
        moment = now or datetime.now()
        hour = moment.replace(minute=0, second=0, microsecond=0)
        with self._lock:
            if key in self._in_flight:
                return False
            if self._attempted_hour.get(key) == hour:
                return False
            self._attempted_hour[key] = hour
            self._in_flight.add(key)
        thread = threading.Thread(
            target=self._fetch,
            args=(key, moment),
            name=f"h1-history-{key}",
            daemon=True,
        )
        thread.start()
        return True

    def fetch_now(self, symbol: str, *, now: datetime | None = None) -> list[dict[str, Any]]:
        """The same fetch, inline. For tests and for a headless caller."""
        key = str(symbol or "").strip().upper()
        self._fetch(key, now or datetime.now())
        return self.bars_for(key)

    def _fetch(self, symbol: str, moment: datetime) -> None:
        bars: list[dict[str, Any]] = []
        try:
            frame = self._downloader(
                symbol, period=self._period, interval=self._interval
            )
            bars = frame_to_h1_bars(frame, now=moment)
        except Exception:
            # A download that fails is unavailability, not an empty tape.
            logging.debug("H1 history fetch failed for %s", symbol, exc_info=True)
            bars = []
        with self._lock:
            self._in_flight.discard(symbol)
            if bars:
                self._bars[symbol] = bars
                self._failed.discard(symbol)
            else:
                self._failed.add(symbol)
