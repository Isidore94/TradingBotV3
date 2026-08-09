from __future__ import annotations

"""Builds chart snapshots off the GUI thread and delivers them by signal.

Part C rule C3: the UI thread does zero I/O. The chart used to call
``chart_snapshot.build_d1_snapshot`` inline on every symbol switch, which
reads the durable daily store out of the Drive-backed home folder and parses
parquet - a network read on the paint path.

Here the same functions run on a worker, fed by :mod:`ui.services.bar_cache`,
and the result arrives as ``snapshotReady``. The indicator math is untouched:
``build_d1_snapshot`` already exposes a ``loader`` seam, so this changes only
WHERE the bars come from and WHICH THREAD does the work - never what is
computed from them (plan.md sec 5).

Ordering: a fast sequence of symbol switches can complete out of order, so
each request carries a sequence number and a task whose number is no longer
the newest for its symbol drops its result instead of painting over a newer
chart.
"""

import atexit
import itertools
import logging
import threading
from collections import OrderedDict
from typing import Any, Iterable, Mapping, Sequence

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from ui.services.bar_cache import BarSeries, D1BarStore, shared_store

_log = logging.getLogger(__name__)

#: Two workers: one serving the chart the trader is looking at, one draining
#: the prefetch queue behind it. More would just contend for the same disk.
DEFAULT_MAX_THREADS = 2
#: Built snapshots kept for instant repaint on revisit. Small next to the bar
#: cache itself - these are only the ~90 shown bars plus overlay series.
_LAST_SNAPSHOT_CAP = 60


class _SnapshotTask(QRunnable):
    """One symbol's D1+M5 snapshot build, on a pool thread."""

    def __init__(
        self,
        service: "ChartDataService",
        request_id: int,
        symbol: str,
        m5_bars: Sequence[Mapping[str, Any]],
        sessions: int | None,
    ) -> None:
        super().__init__()
        self._service = service
        self._request_id = request_id
        self._symbol = symbol
        self._m5_bars = list(m5_bars or [])
        self._sessions = sessions

    def run(self) -> None:  # noqa: D401 (Qt override)
        service = self._service
        if service._closing:
            return
        try:
            d1, m5, meta = service.build_snapshots(
                self._symbol, self._m5_bars, self._sessions
            )
        except Exception:
            _log.warning(
                "Chart snapshot build failed for %s.", self._symbol, exc_info=True
            )
            service._finish(self._request_id, self._symbol, None, None, {})
            return
        service._finish(self._request_id, self._symbol, d1, m5, meta)


class _PrefetchTask(QRunnable):
    def __init__(self, service: "ChartDataService", symbols: Sequence[str]) -> None:
        super().__init__()
        self._service = service
        self._symbols = list(symbols)

    def run(self) -> None:  # noqa: D401 (Qt override)
        if self._service._closing:
            return
        try:
            warmed = self._service.store.prefetch(self._symbols)
        except Exception:
            _log.debug("Chart prefetch failed.", exc_info=True)
            return
        self._service._prefetch_done(warmed, len(self._symbols))


class ChartDataService(QObject):
    """Async front door to the chart's data. One per desk is plenty."""

    #: (symbol, d1_snapshot, m5_snapshot, meta). The first three are plain
    #: dicts in chart_snapshot's existing shape, so renderers need no new
    #: contract; ``meta`` carries what the worker learned on the way -
    #: freshness probes and which cache tier answered.
    snapshotReady = Signal(str, object, object, object)
    #: (symbol) - the build raised; the host should say so, not show blanks.
    snapshotFailed = Signal(str)
    #: (warmed, requested) once a prefetch batch drains.
    prefetchFinished = Signal(int, int)

    def __init__(
        self,
        *,
        store: D1BarStore | None = None,
        parent: QObject | None = None,
        max_threads: int = DEFAULT_MAX_THREADS,
    ) -> None:
        super().__init__(parent)
        self.store = store if store is not None else shared_store()
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max(1, int(max_threads)))
        self._counter = itertools.count(1)
        self._lock = threading.Lock()
        self._newest: dict[str, int] = {}
        self._last: "OrderedDict[str, tuple[dict, dict]]" = OrderedDict()
        #: Set by shutdown(). Queued tasks check it and return without
        #: touching anything - see the note on shutdown() for why.
        self._closing = False

    # -- requests --------------------------------------------------------
    def request(
        self,
        symbol: str,
        m5_bars: Sequence[Mapping[str, Any]] = (),
        *,
        sessions: int | None = None,
    ) -> int:
        """Queue a snapshot build. Returns immediately; never blocks.

        ``m5_bars`` must already be in hand - the bot's ``m5_chart_bars`` is
        an in-memory read the caller does on the GUI thread, so this service
        never reaches into the bot from a worker.
        """
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return 0
        request_id = next(self._counter)
        with self._lock:
            self._newest[symbol] = request_id
        self._pool.start(_SnapshotTask(self, request_id, symbol, m5_bars, sessions))
        return request_id

    def prefetch(self, symbols: Iterable[str]) -> int:
        """Warm the bar cache for a watchlist off-thread (D4).

        Symbols already resident are skipped inside the store, so calling
        this on every workspace open is cheap after the first time.
        """
        wanted = []
        seen = set()
        for symbol in symbols or ():
            symbol = str(symbol or "").strip().upper()
            if symbol and symbol not in seen:
                seen.add(symbol)
                wanted.append(symbol)
        if not wanted:
            return 0
        self._pool.start(_PrefetchTask(self, wanted))
        return len(wanted)

    # -- the work itself (worker threads) --------------------------------
    def build_snapshots(
        self,
        symbol: str,
        m5_bars: Sequence[Mapping[str, Any]],
        sessions: int | None = None,
    ) -> tuple[dict, dict, dict]:
        """The blocking build. Public so tests can exercise it directly.

        Returns (d1, m5, meta). ``meta`` carries the freshness probes, which
        belong here rather than on the GUI thread: they resolve the market
        session, and that reads local_settings.json for the configured
        timezone on every call.
        """
        from ui.services import safe_import

        # First, and under the shared lock: this worker, the D1 backfill
        # thread and the forming-candle thread can all be the first importer
        # of master_avwap_lib, and racing on it yields a partially
        # initialized package.
        safe_import.warm()
        import chart_snapshot

        tier = "shared"

        def loader(name: str) -> list[dict]:
            nonlocal tier
            series = self.store.load(name)
            if series is None:
                return []
            tier = series.source
            return series.as_bar_dicts()

        kwargs: dict[str, Any] = {"loader": loader, "intraday_bars": list(m5_bars or [])}
        if sessions is not None:
            kwargs["sessions"] = sessions
        d1 = chart_snapshot.build_d1_snapshot(symbol, **kwargs)
        m5 = chart_snapshot.build_m5_snapshot(symbol, list(m5_bars or []))

        meta: dict[str, Any] = {"source": tier, "stale_store": False, "want_forming": False}
        try:
            bars = d1.get("bars") or []
            if chart_snapshot.d1_store_is_stale(bars):
                meta["stale_store"] = True
            elif not m5_bars and bars and chart_snapshot.session_has_opened():
                # Store is current through the last close but today's candle
                # is simply absent, and nothing else fetches it for a symbol
                # outside the scan set.
                meta["want_forming"] = True
        except Exception:
            _log.debug("D1 freshness probe failed for %s.", symbol, exc_info=True)
        return d1, m5, meta

    def cached_series(self, symbol: str) -> BarSeries | None:
        """Memory-only peek, safe on the GUI thread (for skeleton decisions)."""
        return self.store.cached(symbol)

    # -- completion ------------------------------------------------------
    def _finish(
        self,
        request_id: int,
        symbol: str,
        d1: dict | None,
        m5: dict | None,
        meta: dict,
    ) -> None:
        with self._lock:
            newest = self._newest.get(symbol)
        if newest is not None and request_id != newest:
            # The trader has already moved on; painting this would rewind them.
            return
        if d1 is not None and m5 is not None:
            self._remember(symbol, d1, m5)
        try:
            if d1 is None or m5 is None:
                self.snapshotFailed.emit(symbol)
            else:
                self.snapshotReady.emit(symbol, d1, m5, dict(meta or {}))
        except RuntimeError:
            pass  # the service was torn down while the task ran

    def _remember(self, symbol: str, d1: dict, m5: dict) -> None:
        """Keep the last built snapshot so a revisit paints before the reload.

        This is what makes C2's cold budget reachable: navigating back to a
        symbol repaints from memory immediately, and the fresh build lands on
        top when the worker finishes.
        """
        with self._lock:
            self._last[symbol] = (d1, m5)
            self._last.move_to_end(symbol)
            while len(self._last) > _LAST_SNAPSHOT_CAP:
                self._last.popitem(last=False)

    def last_snapshot(self, symbol: str) -> tuple[dict, dict] | None:
        """The most recent build for ``symbol``. Memory only; GUI-thread safe."""
        symbol = str(symbol or "").strip().upper()
        with self._lock:
            found = self._last.get(symbol)
            if found is not None:
                self._last.move_to_end(symbol)
            return found

    def _prefetch_done(self, warmed: int, requested: int) -> None:
        try:
            self.prefetchFinished.emit(int(warmed), int(requested))
        except RuntimeError:
            pass

    # -- lifecycle / tests ------------------------------------------------
    def wait_for_idle(self, timeout_ms: int = 5000) -> bool:
        """Block until queued builds finish. Tests and shutdown only."""
        return bool(self._pool.waitForDone(int(timeout_ms)))

    def shutdown(self) -> None:
        """Drop queued work and wait for the running tasks to finish.

        This MUST happen before the interpreter starts finalizing. A worker
        part-way through ``build_snapshots`` re-enters modules through
        function-level imports; if ``sys.modules`` is being torn down
        underneath it, that is an access violation, not an exception - the
        symptom was a segfault at the end of an otherwise green test run.
        """
        self._closing = True
        self._pool.clear()
        self._pool.waitForDone(5000)


_SERVICE: ChartDataService | None = None
_SERVICE_LOCK = threading.Lock()
_ATEXIT_REGISTERED = False


def shared_service() -> ChartDataService:
    """Process-wide service, created on first use from the GUI thread."""
    global _SERVICE, _ATEXIT_REGISTERED
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = ChartDataService()
        if not _ATEXIT_REGISTERED:
            # Registered here rather than at import so a process that never
            # charts anything pays nothing, and so tests that build their own
            # services are unaffected.
            atexit.register(_shutdown_shared_service)
            _ATEXIT_REGISTERED = True
        return _SERVICE


def _shutdown_shared_service() -> None:
    service = _SERVICE
    if service is not None:
        try:
            service.shutdown()
        except Exception:
            pass


def reset_shared_service() -> None:
    """Test hook: drop the process-wide service."""
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is not None:
            _SERVICE.shutdown()
        _SERVICE = None
