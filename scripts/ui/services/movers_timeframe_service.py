"""Owner of the Movers M30 and Daily boards: one QObject, one single-shot timer,
one worker thread at a time (trader 2026-09-29: "we dont want too much burden on
the program").

Schedule (New York time; the trader's clock is Pacific):
- M30 once per trading day at 12:00 (09:00 PT). A desk started later that day
  with no M30 board for today runs it once; the board always reads the bars
  completed by 12:00, so a late run shows the same board.
- Daily once per trading day at 16:15, after the close. On start, a saved Daily
  board older than the last completed session runs once, outside 09:30-16:15.
- Nothing starts in the night AI window, 22:00-06:00 Pacific; a due scan waits
  until 06:00.

Each scan downloads batched Yahoo bars (M30: 30m over 1mo; Daily: 1d over 1y)
through `autopilot_core._default_downloader` (the guarded `yahoo_download` path)
for `liquid_universe()` + the bot scan set + Focus names + today's M5 board
names, builds the board (`movers_timeframe`), logs its picks and resolves due
outcomes (`movers_timeframe_outcomes`, the Daily scan only), attaches the
summaries, saves the board atomically and emits it. A failed scan keeps and
re-shows the last good board with the error. Nothing heavy runs on the Qt thread.

The Daily Dip boxes measure from a date the trader picks (`set_d1_since`); a new
date rebuilds the Daily board from this session's cached daily bars off the Qt
thread without logging picks, or runs a normal Daily scan when none are cached.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

from PySide6.QtCore import QObject, QTimer, Signal

import movers_timeframe as mtf
import movers_timeframe_outcomes as mto
from ui.services import movers_service

NY_TZ = mtf.NY_TZ
PT_TZ = ZoneInfo("America/Los_Angeles")
#: The night AI window on the trader's clock: nothing starts inside it.
NIGHT_START = dt_time(22, 0)
NIGHT_END = dt_time(6, 0)
#: Market hours for the Daily catch-up: a stale Daily board waits past these.
RTH_START = dt_time(9, 30)
#: First check after the desk starts, and the longest sleep between checks.
STARTUP_DELAY_MS = 60_000
MAX_SLEEP_MS = 30 * 60 * 1000
#: Yahoo windows.
M30_PERIOD = "1mo"
M30_INTERVAL = "30m"
D1_PERIOD = "1y"
D1_INTERVAL = "1d"
#: Market caps asked for listed names without one, per scan.
CAP_FETCH_MAX = 60
#: Scans per timeframe per target session; after that the last good board stays
#: (Yahoo may lack the day's bar at 16:15; no retry every 30 min until night).
MAX_TRIES_PER_SESSION = 3


# ------------------------------------------------------------------ clock
def _aware(now: datetime) -> datetime:
    return now if now.tzinfo is not None else now.astimezone()


def in_night_window(now: datetime) -> bool:
    """22:00-06:00 Pacific: the night AI owns the machine."""
    clock = _aware(now).astimezone(PT_TZ).time()
    return clock >= NIGHT_START or clock < NIGHT_END


def _is_session(day: date) -> bool:
    try:
        from market_calendar import is_session

        return bool(is_session(day))
    except Exception:
        return day.weekday() < 5


def m30_due(now: datetime, board_session: str | None) -> bool:
    """A trading day at or after 12:00 New York with no M30 board for it yet."""
    if in_night_window(now):
        return False
    ny = _aware(now).astimezone(NY_TZ)
    return (_is_session(ny.date()) and ny.time() >= mtf.M30_SCAN_TIME
            and board_session != ny.date().isoformat())


def m30_expected_session(now: datetime) -> date | None:
    """The session the M30 board should be for: today from 12:00 New York, else the
    last session before today."""
    ny = _aware(now).astimezone(NY_TZ)
    if _is_session(ny.date()) and ny.time() >= mtf.M30_SCAN_TIME:
        return ny.date()
    cursor = ny.date() - timedelta(days=1)
    for _ in range(30):
        if _is_session(cursor):
            return cursor
        cursor -= timedelta(days=1)
    return None


def d1_target_session(now: datetime) -> date | None:
    """The latest session whose Daily scan time (close + 15 min) has passed."""
    ny = _aware(now).astimezone(NY_TZ)
    cursor = ny.date()
    for _ in range(30):
        if _is_session(cursor) and datetime.combine(cursor, mtf.D1_SCAN_TIME, NY_TZ) <= ny:
            return cursor
        cursor -= timedelta(days=1)
    return None


def d1_due(now: datetime, board_session: str | None) -> bool:
    """The Daily board is older than the target session, outside market hours and
    outside the night window."""
    if in_night_window(now):
        return False
    ny = _aware(now).astimezone(NY_TZ)
    if _is_session(ny.date()) and RTH_START <= ny.time() < mtf.D1_SCAN_TIME:
        return False
    target = d1_target_session(now)
    return target is not None and (board_session or "") < target.isoformat()


def next_check_ms(now: datetime) -> int:
    """Milliseconds to the next 12:00 / 16:15 New York or 06:00 Pacific, capped at 30 min."""
    moment = _aware(now)
    ny = moment.astimezone(NY_TZ)
    pt = moment.astimezone(PT_TZ)
    candidates = []
    for offset in (0, 1):
        day = ny.date() + timedelta(days=offset)
        for clock in (mtf.M30_SCAN_TIME, mtf.D1_SCAN_TIME):
            candidates.append(datetime.combine(day, clock, NY_TZ))
        candidates.append(datetime.combine(pt.date() + timedelta(days=offset), NIGHT_END, PT_TZ))
    ahead = [c for c in candidates if c > moment]
    wait = (min(ahead) - moment).total_seconds() * 1000 if ahead else MAX_SLEEP_MS
    return int(max(1000, min(MAX_SLEEP_MS, wait + 1000)))


# ------------------------------------------------------------------ files
def load_board(path: Path | None, tf: str) -> dict[str, Any]:
    """A saved board ({} when missing, unreadable or another timeframe's)."""
    if path is None:
        return {}
    try:
        board = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return board if isinstance(board, dict) and board.get("tf") == tf else {}


def save_board(path: Path | None, board: Mapping[str, Any]) -> bool:
    """Atomic write; a failure keeps the previous file and returns False."""
    if path is None:
        return True
    try:
        from diagnostics.artifact_io import atomic_write_json

        atomic_write_json(Path(path), dict(board), fsync=False)
        return True
    except Exception as exc:
        logging.warning("Movers %s board save failed: %s", board.get("tf"), exc)
        return False


def _upper_names(names: Iterable[Any]) -> list[str]:
    return [s for s in dict.fromkeys(str(n or "").strip().upper() for n in names or ()) if s]


class MoversTimeframeService(QObject):
    """Fetches, ranks, logs and publishes the M30 and Daily Movers boards."""

    timeframeBoardChanged = Signal(str, dict)
    statusChanged = Signal(str)
    #: A worker ended (queued to the Qt thread): apply a date that waited for it.
    _workerDone = Signal()

    def __init__(
        self,
        parent=None,
        *,
        downloader: Callable[..., Any] | None = None,
        universe_provider: Callable[[], list[str]] | None = None,
        bot_provider: Callable[[], Any] | None = None,
        focus_provider: Callable[[], Mapping[str, Mapping[str, Iterable[str]]]] | None = None,
        m5_board_provider: Callable[[], Mapping[str, Any]] | None = None,
        clock: Callable[[], datetime] | None = None,
        autostart: bool = True,
        industry_provider: Callable[[], Mapping[str, str]] | None = None,
        earnings_provider: Callable[[date], Iterable[str]] | None = None,
        cap_provider: Callable[[list[str]], Mapping[str, float | None]] | None = None,
        picks_path: Path | None = None,
        board_paths: Mapping[str, Path | None] | None = None,
    ) -> None:
        super().__init__(parent)
        live = downloader is None
        # An injected downloader (a test) gets no live store and no network default.
        if live:
            from project_paths import (
                MOVERS_D1_BOARD_FILE,
                MOVERS_M30_BOARD_FILE,
                MOVERS_TIMEFRAME_PICKS_FILE,
            )

            picks_path = picks_path or MOVERS_TIMEFRAME_PICKS_FILE
            board_paths = board_paths or {mtf.TF_M30: MOVERS_M30_BOARD_FILE,
                                          mtf.TF_D1: MOVERS_D1_BOARD_FILE}
        self._downloader = downloader
        self._universe_provider = universe_provider or (
            movers_service.liquid_universe if live else list)
        self._bot_provider = bot_provider
        self._focus_provider = focus_provider
        self._m5_board_provider = m5_board_provider
        self._clock = clock or (lambda: datetime.now().astimezone())
        self._industry_provider = industry_provider or (
            movers_service.default_industry_map if live else dict)
        self._earnings_provider = earnings_provider or (
            movers_service.default_earnings_names if live else (lambda _day: ()))
        self._cap_provider = cap_provider if cap_provider is not None else (
            movers_service.default_market_caps if live else None)
        self._picks_path = picks_path
        self._board_paths = dict(board_paths or {})
        self._caps: dict[str, float] = {}
        self._daily_cache: dict[str, Any] = {"session": "", "bars": {}}
        self._errors: dict[str, str] = {}
        # (tf, target session) -> scans started for it.
        self._tries: dict[tuple[str, str], int] = {}
        self._running = False
        self._stopped = False
        # The Daily Dip boxes' start date (None: 20 sessions back) and whether a
        # date change waits for the running worker.
        self._d1_since: date | None = None
        self._since_pending = False
        self._workerDone.connect(self._after_worker)
        # Last good boards, shown right after a restart (two small JSON reads).
        self._boards: dict[str, dict[str, Any]] = {
            tf: load_board(self._board_paths.get(tf), tf) for tf in mtf.TIMEFRAMES
        }
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._tick)
        if autostart:
            self._timer.start(STARTUP_DELAY_MS)

    # ------------------------------------------------------------ reads
    @property
    def running(self) -> bool:
        return self._running

    def boards(self) -> dict[str, dict[str, Any]]:
        """{tf: last good board with its freshness stamped} (copies)."""
        now = self._now()
        return {tf: self._stamped(tf, board, now) for tf, board in self._boards.items() if board}

    def status_text(self) -> str:
        parts = []
        for tf in mtf.TIMEFRAMES:
            board = self._boards.get(tf) or {}
            label = mtf.TF_LABELS[tf]
            target = self._target(tf, self._now())
            behind = board and str(board.get("session") or "") < target
            if behind and self._tries.get((tf, target), 0) >= MAX_TRIES_PER_SESSION:
                parts.append(f"{label}: no {target} bars after {MAX_TRIES_PER_SESSION} tries, "
                             f"showing {board.get('session') or '?'}")
            elif self._errors.get(tf):
                parts.append(f"{label} scan FAILED: {self._errors[tf]}")
            elif board:
                parts.append(f"{label} {board.get('session') or '?'}")
            else:
                parts.append(f"{label} not scanned yet")
        return "Movers " + (" · ".join(parts)) + (" · scanning..." if self._running else "")

    # ------------------------------------------------------------ control
    @property
    def d1_since(self) -> date | None:
        return self._d1_since

    def set_d1_since(self, day: date | None, *, rebuild: bool = True) -> None:
        """The Daily Dip boxes' start date. `rebuild=False` only stores it (the
        restored date on start); otherwise the Daily board is rebuilt, after the
        running worker when one runs."""
        self._d1_since = day
        if not rebuild or self._stopped:
            return
        if self._running:
            self._since_pending = True
            return
        self._apply_d1_since()

    def _apply_d1_since(self) -> None:
        target = d1_target_session(self._now())
        cache = self._daily_cache
        if (target is not None and cache.get("session") == target.isoformat()
                and (cache.get("bars") or {}).get("SPY") is not None):
            self._running = True
            self._spawn(self._rebuild_worker)
        else:
            self._start([mtf.TF_D1])

    def _after_worker(self) -> None:
        if self._since_pending and not self._running:
            self._since_pending = False
            self._apply_d1_since()

    def _spawn(self, target: Callable[[], None]) -> None:
        threading.Thread(target=target, name="movers-timeframe", daemon=True).start()

    def shutdown(self) -> None:
        self._stopped = True
        self._timer.stop()

    def refresh_now(self, tfs: Iterable[str] = mtf.TIMEFRAMES) -> bool:
        """Manual scan of the given timeframes, any hour. False while one runs."""
        return self._start([tf for tf in tfs if tf in mtf.TIMEFRAMES])

    def _now(self) -> datetime:
        try:
            return _aware(self._clock())
        except Exception:
            return datetime.now().astimezone()

    def _arm(self) -> None:
        if self._stopped:
            return
        try:
            delay = next_check_ms(self._now())
        except Exception:
            delay = MAX_SLEEP_MS
        self._timer.start(delay)

    def due(self, now: datetime | None = None) -> list[str]:
        """The timeframes whose scan is due now."""
        now = now or self._now()
        out = []
        if (m30_due(now, (self._boards.get(mtf.TF_M30) or {}).get("session"))
                and not self._tries_spent(mtf.TF_M30, now)):
            out.append(mtf.TF_M30)
        if (d1_due(now, (self._boards.get(mtf.TF_D1) or {}).get("session"))
                and not self._tries_spent(mtf.TF_D1, now)):
            out.append(mtf.TF_D1)
        return out

    def _target(self, tf: str, now: datetime) -> str:
        day = m30_expected_session(now) if tf == mtf.TF_M30 else d1_target_session(now)
        return day.isoformat() if day else ""

    def _tries_spent(self, tf: str, now: datetime) -> bool:
        return self._tries.get((tf, self._target(tf, now)), 0) >= MAX_TRIES_PER_SESSION

    def _tick(self) -> None:
        try:
            if not self._running:
                due = self.due()
                if due:
                    self._start(due)
                else:
                    # Re-stamp freshness (a board goes stale overnight).
                    for tf, board in self.boards().items():
                        self.timeframeBoardChanged.emit(tf, board)
        except Exception:
            logging.exception("Movers timeframe tick failed")
        finally:
            self._arm()

    def _snapshot(self) -> dict[str, list[str]]:
        """Focus and M5-board names, read on the Qt thread (in-memory reads)."""
        focus: list[str] = []
        if self._focus_provider is not None:
            try:
                for sides in (self._focus_provider() or {}).values():
                    for names in (sides or {}).values():
                        focus.extend(names or ())
            except Exception:
                logging.debug("Movers timeframe: focus names unavailable", exc_info=True)
        m5: list[str] = []
        if self._m5_board_provider is not None:
            try:
                board = self._m5_board_provider() or {}
                for key in ("pop", "dip", "rip", "swing"):
                    for side in ("long", "short"):
                        m5.extend(r.get("symbol") for r in ((board.get(key) or {}).get(side)) or [])
            except Exception:
                logging.debug("Movers timeframe: M5 board names unavailable", exc_info=True)
        return {"focus": _upper_names(focus), "m5": _upper_names(m5)}

    def _start(self, tfs: list[str]) -> bool:
        if self._running or not tfs:
            return False
        self._running = True
        snapshot = self._snapshot()
        self._spawn(lambda: self._worker(list(tfs), snapshot))
        return True

    # ------------------------------------------------------------ worker
    def _worker(self, tfs: list[str], snapshot: dict[str, list[str]]) -> None:
        try:
            for tf in tfs:
                key = (tf, self._target(tf, self._now()))
                self._tries[key] = self._tries.get(key, 0) + 1
                try:
                    self._scan(tf, snapshot)
                    self._errors.pop(tf, None)
                except Exception as exc:
                    self._errors[tf] = str(exc) or exc.__class__.__name__
                    logging.exception("Movers %s scan failed", tf)
                    board = self._boards.get(tf) or {}
                    if board:
                        # The last good board stays up, saying the new scan failed.
                        self.timeframeBoardChanged.emit(tf, self._stamped(tf, board, self._now()))
        finally:
            self._running = False
            self.statusChanged.emit(self.status_text())
            self._workerDone.emit()

    def _rebuild_worker(self) -> None:
        """The Daily board again from the cached daily bars for a new start date.
        Logs no picks and resolves no outcomes (only the scheduled scan does)."""
        try:
            now = self._now()
            bars = dict(self._daily_cache.get("bars") or {})
            spy = bars.pop("SPY")
            records: list[dict[str, Any]] = []
            if self._picks_path is not None:
                records = mto.load_records(Path(self._picks_path))
            board = self._build(mtf.TF_D1, bars, spy, now, None)
            board["summaries"] = mto.summaries(records, mtf.TF_D1)
            self._publish(mtf.TF_D1, board)
        except Exception:
            logging.exception("Movers Daily rebuild for a new date failed")
        finally:
            self._running = False
            self.statusChanged.emit(self.status_text())
            self._workerDone.emit()

    def _stamped(self, tf: str, board: Mapping[str, Any], now: datetime) -> dict[str, Any]:
        out = dict(board)
        expected = (m30_expected_session(now) if tf == mtf.TF_M30 else d1_target_session(now))
        out["stale"] = bool(expected is not None and str(board.get("session") or "")
                            < expected.isoformat())
        out["last_error"] = self._errors.get(tf, "")
        return out

    def _downloader_or_default(self):
        if self._downloader is not None:
            return self._downloader
        import autopilot_core as core

        return core._default_downloader

    def _universe(self, snapshot: Mapping[str, list[str]]) -> list[str]:
        try:
            names = list(self._universe_provider() or [])
        except Exception:
            logging.warning("Movers timeframe: universe unavailable", exc_info=True)
            names = []
        bot = None
        if self._bot_provider is not None:
            try:
                bot = self._bot_provider()
            except Exception:
                bot = None
        return _upper_names(["SPY", *names, *movers_service.bot_universe(bot),
                             *snapshot.get("focus", []), *snapshot.get("m5", [])])

    def _daily_bars(self, symbols: list[str], now: datetime) -> dict[str, list]:
        """Daily bars for `symbols`, kept from this session's earlier fetch when possible."""
        target = d1_target_session(now)
        key = target.isoformat() if target else ""
        if self._daily_cache.get("session") != key:
            self._daily_cache = {"session": key, "bars": {}}
        cached = self._daily_cache["bars"]
        need = [s for s in symbols if s not in cached]
        if need:
            cached.update(movers_service.fetch_yahoo_bars(
                need, downloader=self._downloader_or_default(), period=D1_PERIOD,
                interval=D1_INTERVAL))
        return {s: cached[s] for s in symbols if s in cached}

    def _scan(self, tf: str, snapshot: Mapping[str, list[str]]) -> None:
        now = self._now()
        ny = now.astimezone(NY_TZ)
        universe = self._universe(snapshot)
        records: list[dict[str, Any]] = []
        if self._picks_path is not None:
            records = mto.load_records(Path(self._picks_path))
        if tf == mtf.TF_M30:
            # The board reads the bars completed by 12:00 New York, whenever it runs.
            noon = datetime.combine(ny.date(), mtf.M30_SCAN_TIME, NY_TZ)
            scan_now = min(now, noon) if ny.time() >= mtf.M30_SCAN_TIME else now
            bars = movers_service.fetch_yahoo_bars(
                universe, downloader=self._downloader_or_default(), period=M30_PERIOD,
                interval=M30_INTERVAL)
            daily = self._daily_bars(universe, now)
        else:
            scan_now = now
            pending = [r.get("symbol") for r in records if r.get("kind") == "pick"]
            universe = _upper_names([*universe, *pending])
            bars = self._daily_bars(universe, now)
            daily = None
        if not bars.get("SPY"):
            raise RuntimeError("no SPY bars from Yahoo")
        spy = bars.pop("SPY")
        board = self._build(tf, bars, spy, scan_now, daily)
        new_rows = mto.new_picks(board, records)
        if tf == mtf.TF_D1:
            normalised = {s: mtf.normalize_tf_bars(mtf.TF_D1, b, now=now)
                          for s, b in {**bars, "SPY": spy}.items()}
            new_rows += mto.resolve([*records, *new_rows], normalised,
                                    resolved_at=now.isoformat(timespec="seconds"))
        if self._picks_path is not None and new_rows:
            mto.append_records(Path(self._picks_path), new_rows)
        board["summaries"] = mto.summaries([*records, *new_rows], tf)
        self._publish(tf, board)

    def _build(self, tf: str, bars: Mapping[str, Any], spy: Any, scan_now: datetime,
               daily: Mapping[str, Any] | None) -> dict[str, Any]:
        """One board (worker thread); market caps for newly listed names, then again."""
        ny_day = scan_now.astimezone(NY_TZ).date()
        try:
            industry = dict(self._industry_provider() or {})
        except Exception:
            industry = {}
        try:
            earnings = set(self._earnings_provider(ny_day) or ())
        except Exception:
            earnings = set()

        def build() -> dict[str, Any]:
            return mtf.build_timeframe_board(
                tf, bars, spy, now=scan_now, daily_bars=daily,
                fundamentals={s: {"market_cap_m": self._caps.get(s)} for s in bars},
                industry=industry, earnings=earnings,
                since=self._d1_since if tf == mtf.TF_D1 else None)

        board = build()
        if self._fetch_caps(mtf.listed_symbols(board)):
            board = build()
        return board

    def _publish(self, tf: str, board: dict[str, Any]) -> None:
        self._boards[tf] = board
        save_board(self._board_paths.get(tf), board)
        self.timeframeBoardChanged.emit(tf, self._stamped(tf, board, self._now()))

    def _fetch_caps(self, symbols: list[str]) -> bool:
        """Market caps for listed names without one. True when any arrived."""
        need = [s for s in symbols if s not in self._caps][:CAP_FETCH_MAX]
        if self._cap_provider is None or not need:
            return False
        try:
            caps = dict(self._cap_provider(need) or {})
        except Exception:
            logging.warning("Movers timeframe: market caps unavailable", exc_info=True)
            return False
        added = False
        for symbol in need:
            if caps.get(symbol):
                self._caps[symbol] = float(caps[symbol])
                added = True
        return added
