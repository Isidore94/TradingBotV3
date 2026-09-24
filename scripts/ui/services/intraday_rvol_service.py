"""Intraday relative volume for the chart review header.

The reading is ``rvol.session_rvol``: today's completed 5-minute volume so far
against the same time of day over the prior 15 sessions (the trader's TC2000
rule). The desk's own M5 cache holds only five sessions, too few for the
baseline, so this makes one single-ticker yfinance download (1mo, 5m, regular
hours) on a daemon thread - never on the Qt thread, zero IB traffic, at most
once per ``REFRESH`` per symbol. In memory only.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable

from PySide6.QtCore import QObject, Signal

#: How often a symbol's reading may be re-fetched (one new M5 bar = 5 min).
REFRESH = timedelta(minutes=2)
PERIOD = "1mo"
INTERVAL = "5m"


@dataclass(frozen=True)
class RvolReading:
    symbol: str
    session_date: date
    session_rvol: float | None
    last_bar_rvol: float | None
    last_bar_at: datetime | None
    prior_sessions: int
    fetched_at: datetime


def _now() -> datetime:
    """Market-local naive now, the clock the bars are on."""
    try:
        from market_session import get_market_local_now

        return get_market_local_now().replace(tzinfo=None)
    except Exception:  # pragma: no cover - settings unavailable
        return datetime.now()


def frame_to_volume_bars(frame, *, now: datetime) -> list[dict[str, Any]]:
    """Completed {dt, volume} bars from a yfinance 5m frame, oldest first."""
    if frame is None:
        return []
    try:
        if frame.empty:
            return []
    except AttributeError:
        return []
    columns = getattr(frame, "columns", None)
    if columns is not None and getattr(columns, "nlevels", 1) > 1:
        # A single-ticker download can still come back column-grouped; keep
        # whichever level holds the field names.
        for level in range(columns.nlevels):
            names = columns.get_level_values(level)
            if "Volume" in names:
                frame = frame.set_axis(names, axis=1)
                break
        else:
            return []
    from completed_bars import is_completed_bar
    from intraday_history import _market_local

    bars: list[dict[str, Any]] = []
    for stamp, row in frame.iterrows():
        moment = _market_local(stamp)
        if moment is None:
            continue
        try:
            volume = float(row["Volume"])
        except (KeyError, TypeError, ValueError):
            continue
        if volume != volume or volume < 0:
            continue  # NaN or negative volume is missing, never a reading
        bar = {"dt": moment, "volume": volume}
        if is_completed_bar(bar, 5, now=now):
            bars.append(bar)
    bars.sort(key=lambda bar: bar["dt"])
    return bars


def reading_from_bars(
    symbol: str, bars: list[dict[str, Any]], *, fetched_at: datetime
) -> RvolReading | None:
    """Session and last-bar rvol for the newest session in ``bars``."""
    from rvol import RVOL_BASELINE_SESSIONS, bar_rvol, session_rvol, split_sessions

    dated = [bar for bar in bars if hasattr(bar.get("dt"), "date")]
    if not dated:
        return None
    sessions = split_sessions((bar["dt"].date(), bar["volume"]) for bar in dated)
    today, prior = sessions[-1], sessions[:-1]
    return RvolReading(
        symbol=symbol,
        session_date=dated[-1]["dt"].date(),
        session_rvol=session_rvol(today, prior),
        last_bar_rvol=bar_rvol(today, prior),
        last_bar_at=dated[-1]["dt"],
        prior_sessions=min(len(prior), RVOL_BASELINE_SESSIONS),
        fetched_at=fetched_at,
    )


def _download(symbol: str):
    from intraday_history import _download as batched_download

    return batched_download([symbol], period=PERIOD, interval=INTERVAL).get(symbol)


class IntradayRvolService(QObject):
    """Per-symbol rvol readings, fetched one symbol at a time off the Qt thread.

    ``request`` never blocks: it starts the single worker, or, when the worker
    is busy, remembers only the LATEST symbol asked for (flipping through a
    review queue must not queue a download per name).
    """

    #: (symbol) - a new reading (or a failed attempt) landed for this symbol.
    readingReady = Signal(str)

    def __init__(
        self,
        parent=None,
        *,
        downloader: Callable[[str], Any] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(parent)
        self._downloader = downloader
        self._clock = clock or _now
        self._lock = threading.Lock()
        self._readings: dict[str, RvolReading] = {}
        self._attempted: dict[str, datetime] = {}
        self._worker: threading.Thread | None = None
        self._next: str | None = None

    def reading(self, symbol: str) -> RvolReading | None:
        with self._lock:
            return self._readings.get(str(symbol or "").strip().upper())

    def request(self, symbol: str) -> bool:
        """Fetch ``symbol`` unless it was tried within ``REFRESH``. True = queued."""
        key = str(symbol or "").strip().upper()
        if not key:
            return False
        with self._lock:
            attempted = self._attempted.get(key)
            if attempted is not None and self._clock() - attempted < REFRESH:
                return False
            if self._worker is not None and self._worker.is_alive():
                self._next = key
                return True
            self._attempted[key] = self._clock()
            self._worker = threading.Thread(
                target=self._run, args=(key,), name=f"rvol-{key}", daemon=True
            )
            self._worker.start()
        return True

    def wait_idle(self, timeout: float = 5.0) -> None:
        """Test helper: block until the worker (and any queued symbol) is done."""
        deadline = datetime.now() + timedelta(seconds=timeout)
        while datetime.now() < deadline:
            with self._lock:
                worker = self._worker
            if worker is None or not worker.is_alive():
                return
            worker.join(0.05)

    def _run(self, key: str) -> None:
        while key:
            self._fetch_one(key)
            with self._lock:
                key, self._next = self._next, None
                if key:
                    self._attempted[key] = self._clock()

    def _fetch_one(self, key: str) -> None:
        try:
            now = self._clock()
            download = self._downloader or _download
            bars = frame_to_volume_bars(download(key), now=now)
            reading = reading_from_bars(key, bars, fetched_at=now)
            if reading is not None:
                with self._lock:
                    self._readings[key] = reading
        except Exception:
            logging.warning("Intraday rvol fetch failed for %s.", key, exc_info=True)
        try:
            self.readingReady.emit(key)
        except RuntimeError:
            pass  # service torn down while the fetch ran


_SHARED: IntradayRvolService | None = None


def shared_rvol_service() -> IntradayRvolService:
    """The one desk-wide service, so every chart pane shares one cache."""
    global _SHARED
    if _SHARED is None:
        _SHARED = IntradayRvolService()
    return _SHARED
