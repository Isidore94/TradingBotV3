"""Owner of the M5 strength board's data (plan.md Phase 0.5, packet R2 Part B).

One single-flight owner, one timer, one last-good snapshot - the Industry Board
pattern the spec points at. Everything heavy happens on a worker thread; this
object only orchestrates and publishes.

Transport is a batched yfinance 5m download over `universe_all.txt`, reusing
`autopilot_core.fetch_intraday_profiles`' batching. **Zero IB traffic**, so the
locked pacing budget in `docs/ULTIMATE_SETUP_DATABASE_PLAN.md` sec 5.2-5.3 is
untouched and does not need re-litigating.

Measured on this desk 2026-08-15: 27.6 s for all 1,506 symbols at `period=5d`
(see that plan's sec 10), which is why the default refresh is 15 minutes.

Board output is decision support only: no alerts, no watchlist writes, no
influence on any champion path. The only thing it can change is what the trader
chooses to add to Focus, and that add goes through packet R2 Part A's adoption
gate like every other one.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Callable

from PySide6.QtCore import QObject, QTimer, Signal

import autopilot_core as core
import strength_scan
from ui.timer_utils import start_staggered, stop_staggered

#: Refresh cadence, in minutes. Settings-tunable; the default comes from the
#: 27.6 s measurement rather than a guess.
STRENGTH_BOARD_REFRESH_SETTING = "strength_board_refresh_minutes"
STRENGTH_BOARD_REFRESH_MINUTES = 15
#: Percentile kept per side.
STRENGTH_BOARD_FRACTION_SETTING = "strength_board_top_fraction"
_TICK_INTERVAL_MS = 30_000


class StrengthBoardService(QObject):
    """Fetches, scores and publishes the M5 strength board."""

    boardChanged = Signal(dict)
    statusChanged = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._running = False
        self._board: dict[str, Any] = {"long": [], "short": [], "offered": 0, "measured": 0}
        self._last_success: datetime | None = None
        self._last_error = ""
        self._last_attempt: datetime | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)
        start_staggered(self._timer, 41_000)

    # ------------------------------------------------------------------ reads
    @property
    def running(self) -> bool:
        return self._running

    def board(self) -> dict[str, Any]:
        return dict(self._board)

    def status_text(self) -> str:
        if self._running:
            return "Strength board: refreshing..."
        if self._last_success is None:
            return "Strength board: never refreshed" + (
                f" (last attempt failed: {self._last_error})" if self._last_error else ""
            )
        stamp = self._last_success.strftime("%H:%M:%S")
        counts = (
            f"{len(self._board.get('long') or [])} long / "
            f"{len(self._board.get('short') or [])} short"
        )
        measured = f"{self._board.get('measured', 0)} of {self._board.get('offered', 0)} measurable"
        # A failed refresh keeps the last good board, so the age has to be
        # visible - a stale board that looks current is worse than none.
        suffix = f" · last refresh FAILED: {self._last_error}" if self._last_error else ""
        return f"Strength board {stamp}: {counts} ({measured}){suffix}"

    # ----------------------------------------------------------------- control
    def refresh_now(self) -> bool:
        """Manual refresh. Never gated on quiet hours - the trader asked."""
        return self._start(manual=True)

    def shutdown(self) -> None:
        stop_staggered(self._timer)

    # ------------------------------------------------------------------ timer
    def _tick(self) -> None:
        try:
            now = datetime.now()
            if not self._due(now):
                return
            self._start(manual=False)
        except Exception:
            logging.exception("Strength board tick failed")

    def _due(self, now: datetime) -> bool:
        if self._running:
            return False
        # Quiet hours (packet R1): the board is automatic work, so it runs on
        # the session window and stops overnight like everything else. The
        # manual button bypasses this.
        try:
            allowed, _reason = core.auto_scanning_due(now)
        except Exception:
            allowed = True  # fail open, as everywhere else
        if not allowed:
            return False
        if self._last_attempt is None:
            return True
        elapsed = (now - self._last_attempt).total_seconds() / 60.0
        return elapsed >= self._refresh_minutes()

    def _refresh_minutes(self) -> float:
        try:
            from project_paths import get_local_setting

            value = float(get_local_setting(
                STRENGTH_BOARD_REFRESH_SETTING, STRENGTH_BOARD_REFRESH_MINUTES
            ))
        except Exception:
            return float(STRENGTH_BOARD_REFRESH_MINUTES)
        return value if value > 0 else float(STRENGTH_BOARD_REFRESH_MINUTES)

    def _fraction(self) -> float:
        try:
            from project_paths import get_local_setting

            value = float(get_local_setting(
                STRENGTH_BOARD_FRACTION_SETTING, strength_scan.STRENGTH_TOP_FRACTION
            ))
        except Exception:
            return strength_scan.STRENGTH_TOP_FRACTION
        return value if 0.0 < value <= 1.0 else strength_scan.STRENGTH_TOP_FRACTION

    # ------------------------------------------------------------------- work
    def _start(self, *, manual: bool) -> bool:
        """Single flight. A refresh already in progress wins; the caller is
        told rather than silently ignored."""
        if self._running:
            self.statusChanged.emit("Strength board refresh already running.")
            return False
        self._running = True
        self._last_attempt = datetime.now()
        self.statusChanged.emit(
            f"Strength board refreshing ({'manual' if manual else 'scheduled'})..."
        )
        threading.Thread(
            target=self._worker, name="strength-board", daemon=True
        ).start()
        return True

    def _worker(self) -> None:
        try:
            board = build_board(fraction=self._fraction())
            self._board = board
            self._last_success = datetime.now()
            self._last_error = ""
            self.boardChanged.emit(dict(board))
        except Exception as exc:
            # The last good board survives a failed refresh (plan.md sec 5: a
            # failed publish never destroys the last verified one). The error
            # rides in the status line so a stale board cannot look current.
            self._last_error = str(exc) or exc.__class__.__name__
            logging.exception("Strength board refresh failed")
        finally:
            self._running = False
            self.statusChanged.emit(self.status_text())


#: A year of daily bars: the 200 SMA needs 200 closes and a year holds about
#: 252. Daily rows are a fraction of the 5m payload, so this second batched
#: download is cheap next to the one it rides beside - and it is the only way to
#: measure a D1 floor without an IB request, which this service does not make.
DAILY_FETCH_PERIOD = "1y"


def board_universe() -> list[str]:
    """`universe_all.txt` PLUS every name on the trader's own watchlists.

    Decision 0016 answer 9 and packet V1: *"scan universe_all.txt PLUS every
    symbol in the trader's watchlists (longs/shorts/swing lists) so nothing the
    trader follows is missing."* The universe is built to a liquidity and market
    cap specification; a name the trader is watching for their own reasons may
    not clear it, and the board it never appears on is the one they are reading.

    Order is preserved and duplicates dropped, so the batching sees each symbol
    once. A watchlist that cannot be read contributes nothing and never raises -
    a missing longs.txt must not empty the board.
    """
    from watchlist_utils import read_watchlist_symbols

    from project_paths import UNIVERSE_ALL_FILE, get_master_avwap_watchlist_paths

    seen: dict[str, None] = {}
    # The four the Master AVWAP scanner already reads, through its own accessor
    # rather than a second list of constants that could drift from it.
    for path in (UNIVERSE_ALL_FILE, *get_master_avwap_watchlist_paths()):
        try:
            names = read_watchlist_symbols(path)
        except Exception:
            continue
        for name in names:
            text = str(name or "").strip().upper()
            if text:
                seen.setdefault(text, None)
    return list(seen)


def _daily_closes(
    pool: list[str],
    downloader,
    *,
    chunk_size: int,
) -> dict[str, list[float]]:
    """Batched daily closes for the D1 SMA floors. Zero IB traffic, as ever.

    A chunk that fails contributes nothing and is logged: the symbols in it lose
    their SMA floors and are shown greyed with "cannot measure the D1 200 SMA",
    which is the honest outcome. Failing the whole board because one chunk timed
    out would be worse than a partly greyed one.
    """
    closes: dict[str, list[float]] = {}
    for start in range(0, len(pool), chunk_size):
        chunk = pool[start : start + chunk_size]
        try:
            data = downloader(chunk, period=DAILY_FETCH_PERIOD, interval="1d")
        except Exception as exc:
            logging.warning(
                "Strength board daily chunk %s..%s failed: %s", chunk[0], chunk[-1], exc
            )
            continue
        for symbol in chunk:
            try:
                frame = data[symbol] if len(chunk) > 1 else data
            except Exception:
                continue
            values = [
                row.get("close")
                for row in core._frame_rows(frame)
                if row.get("close") is not None
            ]
            if values:
                closes[symbol] = [float(value) for value in values]
    return closes


def build_board(
    *,
    fraction: float = strength_scan.STRENGTH_TOP_FRACTION,
    rvol_fraction: float = strength_scan.RVOL_TOP_FRACTION,
    session_volume_fraction: float = strength_scan.SESSION_VOLUME_TOP_FRACTION,
    symbols: list[str] | None = None,
    downloader: Callable[..., Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fetch the universe's 5m bars and turn them into a board.

    Plain function rather than a method so the whole pipeline is testable
    without Qt, a timer or a thread.
    """
    pool = symbols if symbols is not None else board_universe()
    pool = [str(item or "").strip().upper() for item in pool]
    pool = [item for item in pool if item]
    if not pool:
        return {"long": [], "short": [], "offered": 0, "measured": 0,
                "long_filtered_out": 0, "short_filtered_out": 0}

    downloader = downloader or core._default_downloader
    moment = now or datetime.now()
    chunk_size = max(1, int(core.AUTOPILOT_OPEN_SCAN_CHUNK_SIZE))
    bars_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for start in range(0, len(pool), chunk_size):
        chunk = pool[start : start + chunk_size]
        try:
            data = downloader(chunk, period=strength_scan.STRENGTH_FETCH_PERIOD, interval="5m")
        except Exception as exc:
            logging.warning(
                "Strength board chunk %s..%s failed: %s", chunk[0], chunk[-1], exc
            )
            continue
        for symbol in chunk:
            try:
                frame = data[symbol] if len(chunk) > 1 else data
            except Exception:
                continue
            rows = core._frame_rows(frame)
            completed = _completed_bars(rows, moment)
            if len(completed) >= strength_scan.STRENGTH_ATR_PERIOD + 1:
                bars_by_symbol[symbol] = completed

    # The D1 floors, over the symbols that actually reached the board - not the
    # whole pool. A name with no measurable M5 strength is not on the board, so
    # fetching its year of daily bars would be a download nobody reads.
    daily = _daily_closes(sorted(bars_by_symbol), downloader, chunk_size=chunk_size)

    board = strength_scan.build_strength_board(
        bars_by_symbol,
        fraction=fraction,
        daily_closes_by_symbol=daily,
        rvol_fraction=rvol_fraction,
        session_volume_fraction=session_volume_fraction,
    )
    board["offered"] = len(pool)
    board["daily_measured"] = len(daily)
    board["as_of"] = moment.isoformat(timespec="seconds")
    return board


def _completed_bars(rows: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    """Drop the forming bar (plan.md sec 5: a forming bar is a preview).

    A board that ranked on a forming bar would reshuffle every few seconds
    against moves that had not finished happening.
    """
    from datetime import timedelta

    from market_session import normalize_market_local_datetime

    moment = normalize_market_local_datetime(now)
    completed: list[dict[str, Any]] = []
    for row in rows:
        stamp = row.get("dt")
        if not isinstance(stamp, datetime):
            continue
        local_start = normalize_market_local_datetime(stamp)
        if local_start + timedelta(minutes=5) > moment:
            continue
        completed.append(row)
    return completed
