"""Intraday bar history for an ARMED watch, when the desk's own cache is short.

Was `h1_history.py` (WISHLIST 10C, repair RV-H1-HISTORY). PCT-1 generalised it
by `interval_minutes` so the Pullback alert's M15 and M30 legs ride the SAME
cadence rule as the H1 one instead of growing a second copy of it;
`h1_history.H1HistoryCache` is now this class fixed at 60 minutes and every
`test_rv_h1_*` test passes unchanged.

Why it exists (lead decision 2026-09-13; the trader may overrule). The H1
retester rule needs 45 completed hourly bars before it will answer. The desk's
cached M5 window is five regular-hours sessions - one `"5 D"` IB fetch, kept
and extended by SN2 - which aggregates to about 35, and there is no cached M15
or M30 series at all. Two sources were checked and neither closes the gap:
WS-CH's "Load older" is the SAME `m5_chart_bars` read with a ceiling of ten
sessions on the ASK, and the durable H1 store
`project_paths.MASTER_AVWAP_INTRADAY_BARS_DIR` has never been written on this
desk. A watch that cannot fire for a whole test week gives the trader nothing
to judge, so the missing history is fetched.

The shape follows the group RS/RW tape precedent exactly: **its own clock, one
outbound `yfinance` call, zero IB traffic, and no change to any engine.**

Six rules this module holds:

* **Armed symbols only, and only when short.** Nothing here runs for a symbol
  the trader has not armed a watch on, and nothing runs while the M5 cache can
  answer on its own. The cache stays PRIMARY.
* **At most one fetch per completed BUCKET per symbol, and a bucket is the
  SESSION-aligned one** (repair RV-H1-HISTORY, 2026-09-13). A new bar is the
  only thing that can change the answer, so a 60-second poll asking sixty
  times an hour would be fifty-nine wasted round trips - and the bar that can
  change the answer is the primary series' bucket (06:30, 06:45 ... for a
  quarter-hour cache; 06:30, 07:30 ... 12:30 for the hourly one, whose last
  bucket is 30 minutes long and closes at the bell), never the wall-clock
  hour. Keying the refusal on the clock hour both refetched twice inside one
  bucket and refetched every hour all evening, when no bucket can complete at
  all. A fetch already in flight is never duplicated.
* **A bucket the bell cuts short is still a completed bucket.** The rule is
  "once per completed SESSION bucket, including the closing one, and never
  outside the session's own buckets" - so a poll after the close still fetches
  the session's last bar exactly once, and then asks for nothing until the
  next session's first bucket closes. PCT-1 briefly added an after-the-bell
  refusal here and its review caught the cost: it dropped the closing bar of
  every session (the 12:45-13:00 M15 bar was never fetched) while the health
  cell still read `from yfinance`. Removed 2026-09-15.
* **One worker, batched downloads, never on the Qt thread.** `request` only
  ENQUEUES and returns immediately; the cache's single daemon worker drains
  the queue in chunks of at most `DOWNLOAD_CHUNK_SYMBOLS` (50) and issues ONE
  multi-ticker download per chunk - the group RS/RW tape's precedent - then
  exits when the queue is empty. `bars_for` reads the finished result out of
  memory. A poll therefore never blocks on the network, and the first cycle
  after arming simply reports "not measured" until the answer lands. Before
  the batching (PCT-1 review) one tick that armed 95 watches issued 285
  single-ticker downloads on 286 threads.
* **Completed bars only, on the SESSION rule.** The forming bucket is dropped
  through the one rule (`completed_bars.is_completed_bar`), but the span it is
  measured over is the bucket's own - the interval, or the walk to the session
  close for a bucket the bell cuts short - so the fetched series and the
  desk's own aggregation admit the same bar at the same moment. Timestamps are
  CONVERTED to the desk's market-local clock with `astimezone`, never stripped
  - that is the same fault N1 recorded on the capture sidecars.
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
import time
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any, Callable

#: The hourly cache's own numbers, kept under their old names because
#: `chart_watch` and four shipped test files read them.
H1_MINUTES = 60
H1_SPAN = timedelta(minutes=H1_MINUTES)

DEFAULT_INTERVAL_MINUTES = H1_MINUTES

#: How much history to ask for. `1mo` is ~22 sessions: ~150 completed hourly
#: bars, ~600 M15 ones - comfortably past every warm-up this desk reads, with
#: room for holidays, and small enough to stay one quick request.
DEFAULT_PERIOD = "1mo"
DEFAULT_INTERVAL = "60m"

#: Source labels the armed inventory prints, so the trader can see which
#: history a verdict was measured on.
SOURCE_CACHE = "cache"
SOURCE_YFINANCE = "yfinance"


def interval_label(interval_minutes: int) -> str:
    """The yfinance interval string for this bucket size (`15m`, `30m`, `60m`)."""
    return f"{max(1, int(interval_minutes))}m"


#: How many symbols ride in one download. The group RS/RW tape's precedent:
#: ONE batched `yfinance` call rather than a call per name. 50 keeps the URL
#: and the response inside what the endpoint answers reliably, and 95 armed
#: watches is then two requests rather than ninety-five.
DOWNLOAD_CHUNK_SYMBOLS = 50

#: How long the worker waits before its FIRST take, so one poll's asks become
#: one request. The armed poll enqueues its whole armed set in a single pass
#: that costs microseconds; without this window the worker would take the
#: first symbol alone and batch only the remainder. It is a batching window on
#: a worker thread, never a poll and never on the Qt thread.
QUEUE_COALESCE_SECONDS = 0.10


def _download(symbols, *, period: str, interval: str):
    """One outbound yfinance call for MANY symbols. Returns symbol -> frame.

    Batched since PCT-1's review: a tick that armed 95 watches issued 285
    single-ticker downloads and 286 threads. `yfinance` takes a ticker list
    and answers column-grouped, which is exactly what
    `scripts/group_rrs.py` already relies on for the RS/RW tape. Imported
    lazily, like every other caller.
    """
    import yfinance as yf

    tickers = [str(name or "").strip().upper() for name in symbols]
    tickers = [name for name in tickers if name]
    if not tickers:
        return {}
    frame = yf.download(
        tickers=tickers if len(tickers) > 1 else tickers[0],
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        prepost=False,  # regular hours, the same session the rule is written for
        threads=False,
        group_by="ticker",
    )
    if len(tickers) == 1:
        return {tickers[0]: frame}
    out: dict[str, Any] = {}
    for ticker in tickers:
        try:
            out[ticker] = frame[ticker]
        except Exception:  # noqa: BLE001 - one absent ticker is not a batch failure
            out[ticker] = None
    return out


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


def bucket_end(start: datetime, interval_minutes: int = H1_MINUTES) -> datetime:
    """When the session-aligned bucket beginning at `start` is finished.

    `start + interval`, except for a bucket the bell cuts short, which finishes
    at the close: an hourly 12:30 bucket finishes at 13:00 market-local, not
    13:30. The primary series (`indicators.h1_ema_bounce.closed_h1_bars`)
    closes that bucket the moment the tape reaches the bell; a fetched frame
    that waited a further hour left the two sources one bar apart for an hour
    every session. A 15- or 30-minute grid divides the 390-minute session
    exactly, so nothing is clamped there - the rule is the same one anyway.
    """
    end = start + timedelta(minutes=max(1, int(interval_minutes)))
    bounds = _session_bounds(start)
    if bounds is None:
        return end
    _opened, closed = bounds
    if start < closed < end:
        return closed
    return end


def h1_bucket_end(start: datetime) -> datetime:
    """The hourly case, under the name `chart_watch` already imports."""
    return bucket_end(start, H1_MINUTES)


def _last_bucket_start(
    opened: datetime, closed: datetime, interval_minutes: int
) -> datetime:
    """The start of the LAST session-aligned bucket of a session."""
    span = closed - opened
    step = timedelta(minutes=max(1, int(interval_minutes)))
    count = int(span // step)
    if count * step == span:
        count -= 1  # a bucket never starts at the bell
    return opened + max(0, count) * step


def last_completed_bucket(
    moment: datetime, interval_minutes: int = H1_MINUTES
) -> datetime | None:
    """The start of the most recent session-aligned bucket to have CLOSED.

    This is the cadence key: it changes exactly when a new answer can exist and
    at no other time. Inside a session it walks the open-relative buckets; at or
    after the bell it is the session's last bucket; before the session's first
    bucket closes it is still the previous session's last one, which is why the
    60-second armed poll asks for nothing at the open. None when the session
    cannot be resolved at all - the caller then falls back to the clock hour.
    """
    step = timedelta(minutes=max(1, int(interval_minutes)))
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
            return _last_bucket_start(opened, closed, interval_minutes)
        if moment >= opened + step:
            elapsed = int((moment - opened) // step)
            return opened + (elapsed - 1) * step
        # Nothing has closed in this session yet: the last completed bucket is
        # still the previous session's.
        reference = (opened - timedelta(days=1)).replace(
            hour=12, minute=0, second=0, microsecond=0
        )
    return None


def last_completed_h1_bucket(moment: datetime) -> datetime | None:
    """The hourly case, under the name four shipped test files read."""
    return last_completed_bucket(moment, H1_MINUTES)


def frame_to_bars(
    frame, *, now: datetime, interval_minutes: int = H1_MINUTES
) -> list[dict[str, Any]]:
    """Completed bar dicts from a yfinance frame, oldest first.

    Shaped exactly like `BounceBot.m5_chart_bars` output so a rule cannot tell
    the two sources apart, and filtered through the ONE completed-bar rule so
    the bucket still printing never reaches it.
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
        # The ONE completed-bar rule, over the bucket's OWN span: a bucket the
        # bell cuts short finishes at the bell, and the primary series admits
        # it there.
        span_minutes = int(
            (bucket_end(moment, interval_minutes) - moment).total_seconds() // 60
        )
        if not is_completed_bar(bar, span_minutes, now=now):
            continue
        bars.append(bar)
    bars.sort(key=lambda bar: bar["dt"])
    return bars


def frame_to_h1_bars(frame, *, now: datetime) -> list[dict[str, Any]]:
    """The hourly case, under the name the shipped tests read."""
    return frame_to_bars(frame, now=now, interval_minutes=H1_MINUTES)


class IntradayHistoryCache:
    """Per-symbol intraday bars, refreshed at most once per completed BUCKET.

    ``interval_minutes`` is positional with a default of 60 so the hourly
    subclass and every existing caller keep working; one INSTANCE serves one
    interval, because the bucket cadence and the fetched series are both that
    interval's.

    **One worker, batched downloads** (PCT-1 review, 2026-09-15). `request`
    only ENQUEUES; the cache's single worker drains the queue in chunks of at
    most `DOWNLOAD_CHUNK_SYMBOLS` and issues ONE multi-ticker download per
    chunk. Before that, a poll that armed 95 watches produced 285 downloads
    and 286 threads. A per-symbol failure inside a batch marks only that
    symbol; the worker exits when the queue is empty rather than idling for
    days, so a desk that never arms anything carries no thread at all.
    """

    def __init__(
        self,
        interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
        *,
        downloader: Callable[..., Any] | None = None,
        period: str = DEFAULT_PERIOD,
        interval: str | None = None,
        batch_window_seconds: float = QUEUE_COALESCE_SECONDS,
    ) -> None:
        self.interval_minutes = max(1, int(interval_minutes))
        self._downloader = downloader or _download
        self._period = period
        self._interval = interval or interval_label(self.interval_minutes)
        self._batch_window = max(0.0, float(batch_window_seconds))
        self._lock = threading.Lock()
        self._bars: dict[str, list[dict[str, Any]]] = {}
        #: symbol -> the completed session-aligned BUCKET the last ATTEMPT was
        #: made for, so a failure is not retried sixty times before the next
        #: bar prints and the evening poll asks for nothing at all.
        self._attempted_bucket: dict[str, datetime] = {}
        #: symbol -> the bucket it is QUEUED for, waiting on the worker.
        self._pending: dict[str, datetime] = {}
        self._in_flight: set[str] = set()
        self._failed: set[str] = set()
        self._worker: threading.Thread | None = None

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

    def last_completed_bucket_end(self, now: datetime | None = None) -> datetime | None:
        """When this interval's most recently CLOSED bucket finished.

        The one thing that can change this cache's answer, exposed so a caller
        can ask "is there anything new to judge?" before spending a pass over
        the bars. None when the session cannot be resolved.
        """
        moment = now or datetime.now()
        bucket = last_completed_bucket(moment, self.interval_minutes)
        if bucket is None:
            return None
        return bucket_end(bucket, self.interval_minutes)

    # -- the fetch (worker thread) -------------------------------------
    def request(self, symbol: str, *, now: datetime | None = None) -> bool:
        """Queue this symbol's history. True when it was newly enqueued.

        Refused - quietly and cheaply - when a fetch for it is already queued
        or in flight, or when the last COMPLETED SESSION-ALIGNED BUCKET has
        already been attempted. A new bucket is the only thing that can change
        the answer, so two asks inside one bucket are one question; a bucket
        the bell cuts short IS a completed bucket, so the last bar of the
        session is fetched like every other one.

        Nothing is downloaded here: the cache's single worker takes the queue
        in batches (`DOWNLOAD_CHUNK_SYMBOLS` per request), so ninety-five
        armed watches are two round trips on one thread rather than
        ninety-five on ninety-five.
        """
        key = str(symbol or "").strip().upper()
        if not key:
            return False
        moment = now or datetime.now()
        bucket = last_completed_bucket(moment, self.interval_minutes)
        if bucket is None:  # the session is unreadable - the clock hour then
            bucket = moment.replace(minute=0, second=0, microsecond=0)
        with self._lock:
            if key in self._in_flight:
                return False
            if self._attempted_bucket.get(key) == bucket:
                return False
            # A symbol already QUEUED whose next bucket closes before the
            # worker reaches it is not a second question: the queue holds one
            # entry per symbol and the fetch about to run answers the newer
            # bucket too, so the ask is accepted and the entry is updated.
            self._attempted_bucket[key] = bucket
            self._pending[key] = bucket
            self._start_worker_locked(moment)
        return True

    def _start_worker_locked(self, moment: datetime) -> None:
        """Start the ONE drain worker, if it is not already running.

        Called with the lock held. The worker exits when the queue is empty
        and clears this slot under the same lock, so "already running" and
        "about to exit" can never both be true for one enqueue. The hourly
        cache keeps the `h1-history-` thread-name prefix the shipped RV-H1
        tests wait on.
        """
        if self._worker is not None and self._worker.is_alive():
            return
        label = "h1" if self.interval_minutes == H1_MINUTES else "intraday"
        worker = threading.Thread(
            target=self._drain,
            args=(moment,),
            name=f"{label}-history-{self.interval_minutes}m",
            daemon=True,
        )
        self._worker = worker
        worker.start()

    def _drain(self, moment: datetime) -> None:
        """Take the queue in batches until it is empty, then stop existing."""
        if self._batch_window:
            # One poll's asks are one request: the caller enqueues its whole
            # armed set in a pass that costs microseconds, and taking the
            # first symbol the instant it lands would split that into a batch
            # of one and a batch of the rest.
            time.sleep(self._batch_window)
        while True:
            with self._lock:
                if not self._pending:
                    self._worker = None
                    return
                batch = sorted(self._pending)[:DOWNLOAD_CHUNK_SYMBOLS]
                for key in batch:
                    self._pending.pop(key, None)
                self._in_flight.update(batch)
            try:
                self._fetch_batch(batch, moment)
            finally:
                with self._lock:
                    self._in_flight.difference_update(batch)

    def fetch_now(self, symbol: str, *, now: datetime | None = None) -> list[dict[str, Any]]:
        """The same fetch, inline, for ONE symbol. Tests and headless callers."""
        key = str(symbol or "").strip().upper()
        self._fetch_batch([key], now or datetime.now())
        return self.bars_for(key)

    def _bars_by_symbol(self, payload, symbols, moment: datetime) -> dict[str, list]:
        """Downloader output -> symbol -> completed bars.

        A batched downloader answers with a mapping. A single-symbol caller -
        `fetch_now`, and every fake in the shipped tests - may answer with the
        frame itself, which is also what `yfinance` does for one ticker, so
        that shape is accepted for a batch of one rather than made an error.
        """
        frames: dict[str, Any] = {}
        if isinstance(payload, Mapping):
            frames = {
                str(key or "").strip().upper(): value
                for key, value in payload.items()
            }
        elif len(symbols) == 1:
            frames = {symbols[0]: payload}
        out: dict[str, list] = {}
        for key in symbols:
            frame = frames.get(key)
            if frame is None:
                continue
            try:
                bars = frame_to_bars(
                    frame, now=moment, interval_minutes=self.interval_minutes
                )
            except Exception:  # noqa: BLE001 - one bad frame is not a batch failure
                logging.debug(
                    "Intraday history frame unreadable for %s (%sm)",
                    key,
                    self.interval_minutes,
                    exc_info=True,
                )
                continue
            if bars:
                out[key] = bars
        return out

    def _fetch_batch(self, symbols, moment: datetime) -> None:
        """ONE download for these symbols. A failure is a refusal, not a value."""
        keys = [str(name or "").strip().upper() for name in symbols]
        keys = [key for key in keys if key]
        if not keys:
            return
        try:
            payload = self._downloader(
                keys, period=self._period, interval=self._interval
            )
            fetched = self._bars_by_symbol(payload, keys, moment)
        except Exception:
            # A download that fails is unavailability, not an empty tape - and
            # it fails for the whole batch, so every symbol in it is marked.
            logging.debug(
                "Intraday history fetch failed for %s (%sm)",
                ",".join(keys[:5]),
                self.interval_minutes,
                exc_info=True,
            )
            fetched = {}
        with self._lock:
            for key in keys:
                bars = fetched.get(key)
                if bars:
                    self._bars[key] = bars
                    self._failed.discard(key)
                else:
                    # Whatever was fetched before STAYS - it is still the best
                    # answer available - and the failure is recorded so the
                    # armed inventory can say the bars stopped updating. One
                    # absent symbol inside a good batch marks only itself.
                    self._failed.add(key)
