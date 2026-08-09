from __future__ import annotations

"""In-memory numpy bar cache for the chart path, with a local disk mirror.

Part C rules C3 and D4. Three things this fixes about the old chart path:

1. It read the durable daily store - which lives in the Drive-backed home
   folder - synchronously on the GUI thread. Every navigation paid it, and a
   cloud-synced read can stall for seconds.
2. Nothing remembered a symbol between clicks beyond a per-file mtime cache
   of already-materialized dicts, so revisiting a name re-parsed parquet.
3. Nothing warmed anything ahead of time, so the first click on every symbol
   was always the slow one.

The store here keeps per-symbol numpy OHLCV arrays under an LRU, mirrors what
it reads into ``%LOCALAPPDATA%`` feather so a cold start warms from local
disk instead of Drive, and appends live bars in place.

THREADING CONTRACT - the reason this module exists:

* :meth:`D1BarStore.cached` is non-blocking and safe to call on the GUI
  thread. It answers only from memory.
* :meth:`D1BarStore.load` does I/O and MUST NOT be called on the GUI thread.
* Everything is guarded by one lock; worker threads may call it concurrently.

This module computes no indicators and makes no provider calls. It supplies
bars in exactly the shape ``chart_snapshot``'s ``loader`` seam already
expects, so the AVWAP/sigma math it feeds is untouched (plan.md sec 5).
"""

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

_OHLCV = ("open", "high", "low", "close", "volume")

#: How many symbols stay resident. 500 symbols x ~490 daily bars x 6 float64
#: columns is roughly 12MB - nothing on a 32GB desk, and it comfortably spans
#: a full scan list plus a session of ad-hoc lookups.
DEFAULT_MAX_SYMBOLS = 500

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BarSeries:
    """Per-symbol OHLCV as parallel numpy arrays, plus where it came from."""

    symbol: str
    dt: np.ndarray  # datetime64[ns]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    #: "memory" | "local" | "shared" - which tier answered. Surfaced so a
    #: slow first paint can be explained rather than guessed at.
    source: str = "shared"

    def __len__(self) -> int:
        return int(self.dt.shape[0])

    def as_bar_dicts(self) -> list[dict[str, Any]]:
        """The bar-dict list ``chart_snapshot``'s loader contract expects.

        Materialized on a worker thread, never on the GUI thread. Costs about
        2ms for a 490-bar symbol, which is why the arrays - not these dicts -
        are what the cache holds.
        """
        stamps = self.dt.astype("datetime64[us]").astype(object)
        return [
            {
                "dt": stamp,
                "open": float(o),
                "high": float(h),
                "low": float(low),
                "close": float(c),
                "volume": float(v),
            }
            for stamp, o, h, low, c, v in zip(
                stamps, self.open, self.high, self.low, self.close, self.volume
            )
        ]

    def with_appended(self, bar: Mapping[str, Any]) -> "BarSeries":
        """This series plus one bar, or with its last bar replaced.

        A same-stamp bar replaces the tail rather than duplicating it, which
        is what a still-forming session bar does as it updates.
        """
        stamp = np.datetime64(bar["dt"], "ns")
        replace = len(self) > 0 and self.dt[-1] == stamp
        cut = slice(0, len(self) - 1) if replace else slice(None)

        def extended(values: np.ndarray, key: str) -> np.ndarray:
            return np.append(values[cut], float(bar.get(key) or 0.0))

        return BarSeries(
            symbol=self.symbol,
            dt=np.append(self.dt[cut], stamp),
            open=extended(self.open, "open"),
            high=extended(self.high, "high"),
            low=extended(self.low, "low"),
            close=extended(self.close, "close"),
            volume=extended(self.volume, "volume"),
            source=self.source,
        )

    @classmethod
    def from_frame(cls, symbol: str, frame, *, source: str = "shared") -> "BarSeries":
        import pandas as pd

        stamps = pd.to_datetime(frame["datetime"]).to_numpy(dtype="datetime64[ns]")
        columns = {}
        for name in _OHLCV:
            if name in frame.columns:
                columns[name] = frame[name].to_numpy(dtype="float64")
            else:
                # Only volume is ever legitimately absent; a store missing a
                # price column is a broken file, and zeros would draw a lie.
                if name != "volume":
                    raise KeyError(f"{symbol}: daily store has no '{name}' column")
                columns[name] = np.zeros(stamps.shape[0], dtype="float64")
        return cls(symbol=symbol, dt=stamps, source=source, **columns)

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "datetime": self.dt,
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "volume": self.volume,
            }
        )


class D1BarStore:
    """LRU of :class:`BarSeries`, backed by a local mirror then the store."""

    def __init__(
        self,
        *,
        max_symbols: int = DEFAULT_MAX_SYMBOLS,
        cache_dir: Path | str | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._series: "OrderedDict[str, BarSeries]" = OrderedDict()
        #: Durable-store mtime (ns) each disk-loaded series was read from.
        #: load() re-stats the store on every memory hit and reloads when the
        #: file has changed - without this, a series read before the scanner
        #: publishes a new session would be served (yesterday's chart)
        #: until process restart. Symbols put() directly have no entry and
        #: are served as-is: memory is authoritative for them.
        self._source_mtime_ns: dict[str, int] = {}
        self._max_symbols = max(1, int(max_symbols))
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._hits = 0
        self._misses = 0

    # -- paths ---------------------------------------------------------
    @property
    def cache_dir(self) -> Path:
        if self._cache_dir is None:
            from project_paths import CHART_BAR_CACHE_DIR

            self._cache_dir = Path(CHART_BAR_CACHE_DIR)
        return self._cache_dir

    def _shared_path(self, symbol: str):
        """(stem, path) of the durable Drive-backed store for ``symbol``."""
        import chart_snapshot

        candidates = chart_snapshot._daily_store_candidates(symbol)
        for stem, path in candidates:
            try:
                if path.exists():
                    return stem, path
            except OSError:
                continue
        return candidates[0] if candidates else ("", None)

    def _mirror_path(self, stem: str, source_mtime_ns: int) -> Path:
        # The source mtime is IN the filename, so a mirror can never be
        # served for a store file that has since changed - no sidecar index
        # to keep in sync, and staleness is structurally impossible.
        return self.cache_dir / f"{stem}-{source_mtime_ns}.feather"

    # -- GUI-thread-safe reads ------------------------------------------
    def cached(self, symbol: str) -> BarSeries | None:
        """Memory-only lookup. Non-blocking; safe on the GUI thread."""
        symbol = _normalize(symbol)
        with self._lock:
            series = self._series.get(symbol)
            if series is not None:
                self._series.move_to_end(symbol)
            return series

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "symbols": len(self._series),
                "hits": self._hits,
                "misses": self._misses,
            }

    # -- worker-thread reads --------------------------------------------
    def load(self, symbol: str) -> BarSeries | None:
        """Resolve ``symbol`` from memory, the local mirror, or the store.

        BLOCKING - worker threads only. Returns None when the symbol has no
        daily store at all (out-of-universe), which callers must render as
        "no data", never as an empty chart that looks like a flat market.

        A memory hit is only served after re-checking the durable store's
        mtime against the one the resident series was loaded from. The stat
        is cheap and off the GUI thread; without it, the cache would keep
        serving the bars it read before the scanner published a new session,
        for as long as the process lives.
        """
        symbol = _normalize(symbol)
        hit = self.cached(symbol)
        if hit is not None and self._memory_hit_is_fresh(symbol):
            with self._lock:
                self._hits += 1
            return hit
        with self._lock:
            self._misses += 1

        # Before any of the function-level imports below. Two pool threads
        # (a snapshot build and a prefetch batch) reaching an un-imported
        # pandas at the same moment faults inside the import machinery.
        from ui.services import safe_import

        safe_import.warm()

        stem, shared = self._shared_path(symbol)
        if not stem:
            return None

        source_mtime_ns = 0
        if shared is not None:
            try:
                source_mtime_ns = shared.stat().st_mtime_ns
            except OSError:
                source_mtime_ns = 0

        series = None
        if source_mtime_ns:
            series = self._read_mirror(symbol, stem, source_mtime_ns)
        if series is None:
            series = self._read_shared(symbol, stem)
            if series is not None and source_mtime_ns:
                self._write_mirror(series, stem, source_mtime_ns)
        if series is None:
            # Drive unreachable or the store file vanished: an older mirror is
            # stale but honest, and beats a blank chart.
            series = self._read_newest_mirror(symbol, stem)
        if series is not None:
            self.put(series)
            with self._lock:
                # After put(), which clears it: this entry came from disk, so
                # record which store version it represents (0 = unknown, which
                # keeps it re-checked on every subsequent load).
                self._source_mtime_ns[symbol] = source_mtime_ns
        return series

    def _memory_hit_is_fresh(self, symbol: str) -> bool:
        """Whether the resident series still matches the durable store.

        Direct put() entries (live streaming, tests) have no recorded mtime
        and are always fresh: memory is authoritative for them. A stat
        failure keeps the resident series - stale but honest beats a blank
        chart, exactly like the mirror fallback below.
        """
        with self._lock:
            recorded = self._source_mtime_ns.get(symbol)
        if recorded is None:
            return True
        _, shared = self._shared_path(symbol)
        if shared is None:
            return True
        try:
            return shared.stat().st_mtime_ns == recorded
        except OSError:
            return True

    def prefetch(self, symbols: Iterable[str]) -> int:
        """Warm the cache for ``symbols``. BLOCKING - worker threads only."""
        from ui.services import safe_import

        safe_import.warm()
        warmed = 0
        for symbol in symbols:
            symbol = _normalize(symbol)
            if not symbol or self.cached(symbol) is not None:
                continue
            try:
                if self.load(symbol) is not None:
                    warmed += 1
            except Exception:
                _log.debug("Prefetch failed for %s.", symbol, exc_info=True)
        return warmed

    # -- mutation --------------------------------------------------------
    def put(self, series: BarSeries) -> None:
        with self._lock:
            self._series[series.symbol] = series
            self._series.move_to_end(series.symbol)
            # A direct put is authoritative until a disk load records anew.
            self._source_mtime_ns.pop(series.symbol, None)
            while len(self._series) > self._max_symbols:
                evicted, _ = self._series.popitem(last=False)
                self._source_mtime_ns.pop(evicted, None)

    def append_live_bar(self, symbol: str, bar: Mapping[str, Any]) -> BarSeries | None:
        """Fold one completed/forming bar into a resident series, in place.

        A symbol that is not resident is left alone: pulling it in here would
        turn a streaming update into a blocking store read.
        """
        symbol = _normalize(symbol)
        if not bar or bar.get("dt") is None:
            return None
        with self._lock:
            current = self._series.get(symbol)
            if current is None:
                return None
            updated = current.with_appended(bar)
            self._series[symbol] = updated
            self._series.move_to_end(symbol)
            return updated

    def invalidate(self, symbol: str) -> None:
        with self._lock:
            symbol = _normalize(symbol)
            self._series.pop(symbol, None)
            self._source_mtime_ns.pop(symbol, None)

    def clear(self) -> None:
        with self._lock:
            self._series.clear()
            self._source_mtime_ns.clear()

    # -- tiers -----------------------------------------------------------
    def _read_mirror(self, symbol: str, stem: str, source_mtime_ns: int) -> BarSeries | None:
        path = self._mirror_path(stem, source_mtime_ns)
        try:
            if not path.exists():
                return None
            import pandas as pd

            return BarSeries.from_frame(symbol, pd.read_feather(path), source="local")
        except Exception:
            # A corrupt mirror must never be fatal: drop it and fall through
            # to the authoritative store.
            _log.debug("Chart bar mirror unreadable: %s", path, exc_info=True)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def _read_newest_mirror(self, symbol: str, stem: str) -> BarSeries | None:
        try:
            mirrors = sorted(self.cache_dir.glob(f"{stem}-*.feather"))
        except OSError:
            return None
        for path in reversed(mirrors):
            try:
                import pandas as pd

                return BarSeries.from_frame(symbol, pd.read_feather(path), source="local")
            except Exception:
                continue
        return None

    def _read_shared(self, symbol: str, stem: str) -> BarSeries | None:
        from setup_playbook_study import _load_daily_frame

        try:
            frame = _load_daily_frame(stem)
        except Exception:
            _log.debug("Daily store read failed for %s.", symbol, exc_info=True)
            return None
        if frame is None or getattr(frame, "empty", True):
            return None
        try:
            return BarSeries.from_frame(symbol, frame, source="shared")
        except (KeyError, ValueError, TypeError):
            _log.warning("Daily store for %s is malformed.", symbol, exc_info=True)
            return None

    def _write_mirror(self, series: BarSeries, stem: str, source_mtime_ns: int) -> None:
        path = self._mirror_path(stem, source_mtime_ns)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Write beside, then replace: a torn feather file read by the next
            # launch would look like a corrupt store. The temp name carries
            # the thread id because a snapshot build and a prefetch batch can
            # be resolving the SAME symbol at the same moment, and a shared
            # temp path would have them writing over each other.
            tmp = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
            series.to_frame().to_feather(tmp)
            tmp.replace(path)
        except Exception:
            _log.debug("Could not mirror %s to local cache.", series.symbol, exc_info=True)
            return
        self._prune_mirrors(stem, keep=path)

    def _prune_mirrors(self, stem: str, *, keep: Path) -> None:
        """Drop this symbol's superseded mirrors (one file per store mtime)."""
        try:
            for path in self.cache_dir.glob(f"{stem}-*.feather"):
                if path != keep:
                    path.unlink(missing_ok=True)
        except OSError:
            pass


def _normalize(symbol: str) -> str:
    return str(symbol or "").strip().upper()


#: One store per process. The chart data service owns it; nothing else should
#: hold a second one, or the two LRUs would each pay the cold reads.
_STORE: D1BarStore | None = None
_STORE_LOCK = threading.Lock()


def shared_store() -> D1BarStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = D1BarStore()
        return _STORE


def reset_shared_store() -> None:
    """Test hook: drop the process-wide store."""
    global _STORE
    with _STORE_LOCK:
        _STORE = None
