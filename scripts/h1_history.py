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
* **At most one fetch per completed H1 bar per symbol, and an H1 bar here is
  the SESSION-aligned one** (repair RV-H1-HISTORY, 2026-09-13). A new bar is
  the only thing that can change the answer, so a 60-second poll asking sixty
  times an hour would be fifty-nine wasted round trips - and the bar that can
  change the answer is the primary series' bucket (06:30, 07:30 ... 12:30
  market-local, the last one 30 minutes long and closed at the bell), never the
  wall-clock hour. Keying the refusal on the clock hour both refetched twice
  inside one bucket and refetched every hour all evening, when no bucket can
  complete at all. A fetch already in flight is never duplicated.
* **Never on the Qt thread.** `request` starts a one-shot daemon thread and
  returns immediately; `bars_for` reads the finished result out of memory.
  A poll therefore never blocks on the network, and the first cycle after
  arming simply reports "not measured" until the answer lands.
* **Completed bars only, on the SESSION rule.** The forming bucket is dropped
  through the one rule (`completed_bars.is_completed_bar`), but the span it is
  measured over is the bucket's own - 60 minutes, or the walk to the session
  close for the short 12:30 bucket - so the fetched series and the desk's own
  aggregation admit the same bar at the same moment. Timestamps are CONVERTED
  to the desk's market-local clock with `astimezone`, never stripped - that is
  the same fault N1 recorded on the capture sidecars.
* **A failure is a refusal, not a value.** An empty or unreadable download
  leaves the cache untouched and is reported as unavailable; nothing here ever
  invents a bar, and nothing it fails to do can cost the watch. A failure
  AFTER a success keeps the bars it already has - they are still the best
  answer available - and says so through `last_refresh_failed`, so the armed
  inventory can tell a live verdict from an ageing one.

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


def _session_bounds(reference: datetime) -> tuple[datetime, datetime] | None:
    """(open, close) market-local naive for `reference`'s session, or None.

    The SAME helpers the primary series uses
    (`indicators.h1_ema_bounce._session_open` / `_session_close`), so the two
    agree on where a bucket starts and when it ends. A settings or zone failure
    answers None and every caller falls back to the clock hour rather than
    inventing a session.
    """
    try:
        from market_session import (
            get_market_session_close_naive,
            get_market_session_open_naive,
        )

        opened = get_market_session_open_naive(reference=reference)
        closed = get_market_session_close_naive(reference=reference)
    except Exception:  # pragma: no cover - settings unavailable
        return None
    if opened is None or closed is None or closed <= opened:
        return None
    return opened, closed


def h1_bucket_end(start: datetime) -> datetime:
    """When the session-aligned H1 bucket beginning at `start` is finished.

    `start + 60 min`, except for the day's last bucket, which is the short one
    the bell closes: 12:30 finishes at 13:00 market-local, not 13:30. The
    primary series (`indicators.h1_ema_bounce.closed_h1_bars`) closes that
    bucket the moment the tape reaches the bell; a fetched frame that waited a
    further hour left the two sources one bar apart for an hour every session.
    """
    end = start + H1_SPAN
    bounds = _session_bounds(start)
    if bounds is None:
        return end
    _opened, closed = bounds
    if start < closed < end:
        return closed
    return end


def _last_bucket_start(opened: datetime, closed: datetime) -> datetime:
    """The start of the LAST session-aligned H1 bucket of a session."""
    span = closed - opened
    count = int(span // H1_SPAN)
    if count * H1_SPAN == span:
        count -= 1  # a bucket never starts at the bell
    return opened + max(0, count) * H1_SPAN


def last_completed_h1_bucket(moment: datetime) -> datetime | None:
    """The start of the most recent session-aligned H1 bucket to have CLOSED.

    This is the cadence key: it changes exactly when a new answer can exist and
    at no other time. Inside a session it walks the open-relative buckets; at or
    after the bell it is the short closing bucket; before the session's first
    bucket closes it is still the previous session's last one, which is why the
    60-second armed poll asks for nothing all evening. None when the session
    cannot be resolved at all - the caller then falls back to the clock hour.
    """
    if moment.tzinfo is not None:
        # The session bounds are naive market-local, so an aware caller is
        # CONVERTED onto that clock - never stripped (N1).
        moment = _market_local(moment) or moment.replace(tzinfo=None)
    reference = moment
    for _ in range(8):  # a long weekend plus holidays, then give up
        bounds = _session_bounds(reference)
        if bounds is None:
            return None
        opened, closed = bounds
        if moment >= closed:
            return _last_bucket_start(opened, closed)
        if moment >= opened + H1_SPAN:
            elapsed = int((moment - opened) // H1_SPAN)
            return opened + (elapsed - 1) * H1_SPAN
        # Nothing has closed in this session yet: the last completed bucket is
        # still the previous session's.
        reference = (opened - timedelta(days=1)).replace(
            hour=12, minute=0, second=0, microsecond=0
        )
    return None


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
        # The ONE completed-bar rule, over the bucket's OWN span: the day's
        # short closing bucket finishes at the bell, and the primary series
        # admits it there.
        span_minutes = int((h1_bucket_end(moment) - moment).total_seconds() // 60)
        if not is_completed_bar(bar, span_minutes, now=now):
            continue
        bars.append(bar)
    bars.sort(key=lambda bar: bar["dt"])
    return bars


class H1HistoryCache:
    """Per-symbol H1 bars, refreshed at most once per completed H1 BUCKET."""

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
        #: symbol -> the completed session-aligned BUCKET the last ATTEMPT was
        #: made for, so a failure is not retried sixty times before the next
        #: bar prints and the evening poll asks for nothing at all.
        self._attempted_bucket: dict[str, datetime] = {}
        self._in_flight: set[str] = set()
        self._failed: set[str] = set()

    # -- reads (Qt thread) ---------------------------------------------
    def bars_for(self, symbol: str) -> list[dict[str, Any]]:
        """Whatever has already been fetched. Never blocks, never fetches."""
        key = str(symbol or "").strip().upper()
        with self._lock:
            return list(self._bars.get(key) or ())

    def unavailable(self, symbol: str) -> bool:
        """True when the last attempt for this symbol failed and none succeeded.

        Unchanged by the RV repair: this is "nothing was EVER fetched", which
        is a different state from "the bars are good but stopped updating".
        """
        key = str(symbol or "").strip().upper()
        with self._lock:
            return key in self._failed and not self._bars.get(key)

    def last_refresh_failed(self, symbol: str) -> bool:
        """True when the most recent attempt failed, retained bars or not.

        The armed inventory reads this to say `stale - last refresh failed`
        beside bars it is still judging the watch on. Held bars are never
        thrown away for a failure - a failure is a refusal, not a value - but
        an ageing verdict must not look like a live one.
        """
        key = str(symbol or "").strip().upper()
        with self._lock:
            return key in self._failed

    # -- the fetch (worker thread) -------------------------------------
    def request(self, symbol: str, *, now: datetime | None = None) -> bool:
        """Ask for this symbol's H1 history. Returns True if a fetch started.

        Refused - quietly and cheaply - when one is already in flight or when
        the last COMPLETED SESSION-ALIGNED BUCKET has already been attempted.
        A new bucket is the only thing that can change the answer, so two asks
        inside one bucket are one question, and after the bell there is no new
        question until the next session's first bucket closes.
        """
        key = str(symbol or "").strip().upper()
        if not key:
            return False
        moment = now or datetime.now()
        bucket = last_completed_h1_bucket(moment)
        if bucket is None:  # the session is unreadable - the clock hour then
            bucket = moment.replace(minute=0, second=0, microsecond=0)
        with self._lock:
            if key in self._in_flight:
                return False
            if self._attempted_bucket.get(key) == bucket:
                return False
            self._attempted_bucket[key] = bucket
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
                # Whatever was fetched before STAYS - it is still the best
                # answer available - and the failure is recorded so the armed
                # inventory can say the bars stopped updating.
                self._failed.add(symbol)
