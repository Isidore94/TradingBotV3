"""Owner of the Movers board's data: one QObject, one timer, one worker at a time.

Each tick (once per 5-minute bar, 20 s after the boundary, regular hours only):
1. reads cached M5 bars from the live bot (`m5_chart_bars`, cache only, never an
   IB request; on the process proxy that is one RPC per symbol, spaced out, and
   always off the Qt thread) for the bot's scan set, SPY and the Focus names;
2. when the bot's scan set is under `BOT_UNIVERSE_MIN` names, also downloads
   the liquid universe's 5m bars from yfinance in batches, on its own 5-minute
   cadence; otherwise gap-fills from yfinance only the names whose bot series
   is stale or missing (capped at GAP_FILL_MAX, most liquid first);
3. fetches a ~20-session 5m history once per symbol per day for the RVOL
   baseline (batched yfinance, kept in memory);
4. builds the board with `movers_scan.build_movers_board`, adds group tags
   (shared classification cache), ER tags (local earnings calendar, read once
   per session) and list persistence, and emits it;
5. feeds the Dip-strong outcome tracker and appends its rows to
   `MOVERS_DIP_OUTCOMES_FILE` (a failed write loses the rows, never the board).

Zero IB traffic. Display only: no alerts, no watchlist or Focus writes. A failed
tick keeps the last good board and says so in the status line.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, time as dt_time, timedelta, tzinfo
from typing import Any, Callable, Iterable, Mapping

from PySide6.QtCore import QObject, QTimer, Signal

import movers_outcomes
import movers_scan


def default_industry_map() -> dict[str, str]:
    """symbol -> industry from the shared classification cache (worker thread)."""
    from industry_scanner import load_symbol_classifications

    return {
        symbol: str(row.get("industry") or "")
        for symbol, row in load_symbol_classifications().items()
        if row.get("industry")
    }


def default_earnings_names(today: date) -> set[str]:
    """Names reporting today or after the previous session's close (local calendar file)."""
    from earnings_history import get_events_in_window
    from market_calendar import previous_session

    previous = previous_session(today)
    return movers_scan.earnings_symbols(
        get_events_in_window(previous, today), today=today, previous=previous
    )


#: One tick per 5-minute bar, this many seconds after the boundary (completed
#: bars only change then; the bot's cache needs a moment to take the new bar).
BAR_SECONDS = 300
TICK_GRACE_SECONDS = 20
#: Below this many bot-scanned names, the yfinance universe sweep is added.
BOT_UNIVERSE_MIN = 300
#: yfinance intraday sweep cadence (minutes); settings-tunable.
MOVERS_YAHOO_MINUTES_SETTING = "movers_yahoo_refresh_minutes"
MOVERS_YAHOO_MINUTES = 5
#: Window for today's bars (prior close + ATR history) and for the baseline.
YAHOO_TODAY_PERIOD = "2d"
YAHOO_BASELINE_PERIOD = "1mo"
#: IB historical TRADES volume is in round lots; yfinance is in shares.
IB_VOLUME_LOT_SIZE = 100
#: Gap-fill: Yahoo 5m bars for stale/missing names only, at most this many per
#: tick (most liquid first), over this short window.
GAP_FILL_MAX = 600
GAP_FILL_PERIOD = "5d"
#: A name Yahoo returned nothing for is skipped for this many ticks.
GAP_EMPTY_BACKOFF_TICKS = 3
#: A symbol whose baseline download returned nothing is retried after this long.
BASELINE_RETRY_MINUTES = 15
#: Gap between two proxy RPCs, so Qt-thread RPCs are not starved of its lock.
RPC_GAP_SECONDS = 0.01
#: Regular session in New York, with a few minutes for the last bar to close.
_RTH_START = dt_time(9, 30)
_RTH_END = dt_time(16, 5)


def in_regular_hours(now: datetime) -> bool:
    """True on a weekday between 09:30 and 16:05 New York."""
    moment = now if now.tzinfo is not None else now.astimezone()
    ny = moment.astimezone(movers_scan.NY_TZ)
    return ny.weekday() < 5 and _RTH_START <= ny.time() <= _RTH_END


def next_tick_delay_ms(now: datetime) -> int:
    """Milliseconds until the next bar boundary + TICK_GRACE_SECONDS."""
    moment = now if now.tzinfo is not None else now.astimezone()
    into_bar = (moment.minute % 5) * 60 + moment.second + moment.microsecond / 1e6
    wait = TICK_GRACE_SECONDS - into_bar
    if wait <= 0:
        wait += BAR_SECONDS
    return int(round(wait * 1000))


def _market_local_tz() -> tzinfo:
    try:
        from market_session import get_market_local_timezone

        return get_market_local_timezone()[0]
    except Exception:
        return datetime.now().astimezone().tzinfo


# ------------------------------------------------------------------ bot reads
def bot_universe(bot) -> list[str]:
    """The bot's scan set (one call; an RPC on the proxy). Empty on failure."""
    if bot is None:
        return []
    try:
        names = bot.get_scan_symbol_set()
    except Exception as exc:
        logging.info("Movers: bot scan set unavailable: %s", exc)
        return []
    return sorted({str(n or "").strip().upper() for n in names or () if str(n or "").strip()})


def read_bot_bars(bot, symbols: Iterable[str], *, rpc_gap: float = RPC_GAP_SECONDS) -> dict[str, list]:
    """Cached M5 bars per symbol, volume converted from round lots to shares."""
    out: dict[str, list] = {}
    if bot is None:
        return out
    proxy = bool(getattr(bot, "is_process_proxy", False))
    for symbol in symbols:
        try:
            bars = list(bot.m5_chart_bars(symbol, max_sessions=2) or [])
        except Exception:
            bars = []
        if proxy and rpc_gap > 0:
            time.sleep(rpc_gap)
        if not bars:
            continue
        converted = []
        for bar in bars:
            row = dict(bar)
            volume = row.get("volume")
            row["volume"] = None if volume is None else float(volume) * IB_VOLUME_LOT_SIZE
            converted.append(row)
        out[symbol] = converted
    return out


# ------------------------------------------------------------------ yfinance
def fetch_yahoo_bars(
    symbols: Iterable[str], *, downloader, period: str, chunk_size: int | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Batched 5m download. A failed chunk contributes nothing."""
    import autopilot_core as core

    pool = [s for s in dict.fromkeys(str(x or "").strip().upper() for x in symbols) if s]
    size = max(1, int(chunk_size or core.AUTOPILOT_OPEN_SCAN_CHUNK_SIZE))
    out: dict[str, list[dict[str, Any]]] = {}
    for start in range(0, len(pool), size):
        chunk = pool[start : start + size]
        try:
            data = downloader(chunk, period=period, interval="5m")
        except Exception as exc:
            logging.warning("Movers chunk %s..%s failed: %s", chunk[0], chunk[-1], exc)
            continue
        for symbol in chunk:
            try:
                frame = data[symbol] if len(chunk) > 1 else data
            except Exception:
                continue
            rows = core._frame_rows(frame)
            if rows:
                out[symbol] = rows
    return out


def liquid_universe() -> list[str]:
    """The Strength Board's universe: universe_all.txt plus the four watchlists."""
    from ui.services.strength_board_service import board_universe

    return board_universe()


def choose_freshest(
    bot_bars: Mapping[str, list],
    yahoo_bars: Mapping[str, list],
    *,
    now: datetime,
    local_tz: tzinfo,
) -> dict[str, list[dict[str, Any]]]:
    """Per symbol, the series whose last completed bar is newest; a tie goes to Yahoo."""
    out: dict[str, list[dict[str, Any]]] = {}
    for symbol in set(bot_bars) | set(yahoo_bars):
        bot = movers_scan.normalize_bars(bot_bars.get(symbol) or (), now=now, local_tz=local_tz)
        yahoo = movers_scan.normalize_bars(
            yahoo_bars.get(symbol) or (), now=now, local_tz=local_tz
        )
        if bot and (not yahoo or bot[-1]["dt"] > yahoo[-1]["dt"]):
            out[symbol] = bot
        elif yahoo:
            out[symbol] = yahoo
    return out


class MoversService(QObject):
    """Fetches, ranks and publishes the Movers board."""

    moversChanged = Signal(dict)
    statusChanged = Signal(str)

    def __init__(
        self,
        parent=None,
        *,
        bot_provider: Callable[[], Any] | None = None,
        focus_provider: Callable[[], Mapping[str, Iterable[str]]] | None = None,
        downloader: Callable[..., Any] | None = None,
        universe_provider: Callable[[], list[str]] | None = None,
        clock: Callable[[], datetime] | None = None,
        autostart: bool = True,
        industry_provider: Callable[[], Mapping[str, str]] | None = None,
        earnings_provider: Callable[[date], Iterable[str]] | None = None,
        outcomes_path=None,
    ) -> None:
        super().__init__(parent)
        self._industry_provider = industry_provider or default_industry_map
        self._earnings_provider = earnings_provider or default_earnings_names
        if outcomes_path is None:
            from project_paths import MOVERS_DIP_OUTCOMES_FILE

            outcomes_path = MOVERS_DIP_OUTCOMES_FILE
        self._outcomes_path = outcomes_path
        self._tracker = movers_outcomes.DipOutcomeTracker()
        self._industry: dict[str, str] = {}
        self._earnings: set[str] = set()
        self._tags_day: date | None = None
        self._persistence: dict[str, Any] = {}
        self._gap_at: datetime | None = None  # last gap-fill that returned bars
        self._gap_count = 0
        self._gap_skip_until: dict[str, int] = {}
        self._tick_no = 0
        self._tracker_restored = False
        self._bot_provider = bot_provider
        self._focus_provider = focus_provider
        self._downloader = downloader
        self._universe_provider = universe_provider or liquid_universe
        self._clock = clock or (lambda: datetime.now().astimezone())
        self._running = False
        self._board: dict[str, Any] = {}
        self._last_success: datetime | None = None
        self._last_error = ""
        self._yahoo_bars: dict[str, list] = {}
        self._yahoo_at: datetime | None = None
        self._baselines: dict[str, dict[int, float] | None] = {}
        self._baseline_day: date | None = None
        self._baseline_tried: dict[str, datetime] = {}
        self.bot_universe_size: int | None = None
        self._stopped = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._tick)
        if autostart:
            self._arm()

    # ------------------------------------------------------------ wiring
    def set_bot_provider(self, provider: Callable[[], Any] | None) -> None:
        self._bot_provider = provider

    def set_focus_provider(self, provider) -> None:
        self._focus_provider = provider

    # ------------------------------------------------------------ reads
    @property
    def running(self) -> bool:
        return self._running

    def board(self) -> dict[str, Any]:
        return dict(self._board)

    def status_text(self) -> str:
        if self._running:
            return "Movers: refreshing..."
        if self._last_success is None:
            return "Movers: no board yet" + (
                f" (last attempt failed: {self._last_error})" if self._last_error else ""
            )
        suffix = f" · last refresh FAILED: {self._last_error}" if self._last_error else ""
        return f"Movers {self._last_success.strftime('%H:%M:%S')}{suffix}"

    # ------------------------------------------------------------ control
    def refresh_now(self) -> bool:
        """Manual refresh, any hour."""
        return self._start()

    def shutdown(self) -> None:
        self._stopped = True
        self._timer.stop()

    def _arm(self) -> None:
        """Schedule the next tick at the next bar boundary plus the grace."""
        if self._stopped:
            return
        try:
            delay = next_tick_delay_ms(self._clock())
        except Exception:
            delay = BAR_SECONDS * 1000
        self._timer.start(delay)

    def _tick(self) -> None:
        try:
            if not self._running and in_regular_hours(self._clock()):
                self._start()
        except Exception:
            logging.exception("Movers tick failed")
        finally:
            self._arm()

    def _focus_snapshot(self) -> dict[str, list[str]]:
        """Focus names by side. Read on the Qt thread: an in-memory store read."""
        if self._focus_provider is None:
            return {"long": [], "short": []}
        try:
            by_category = self._focus_provider() or {}
        except Exception:
            return {"long": [], "short": []}
        out: dict[str, list[str]] = {"long": [], "short": []}
        for sides in by_category.values():
            for side, names in (sides or {}).items():
                key = str(side or "").lower()
                if key in out:
                    for name in names or ():
                        text = str(name or "").strip().upper()
                        if text and text not in out[key]:
                            out[key].append(text)
        return out

    def _start(self) -> bool:
        if self._running:
            return False
        self._running = True
        focus = self._focus_snapshot()
        threading.Thread(
            target=self._worker, args=(focus,), name="movers-board", daemon=True
        ).start()
        return True

    # ------------------------------------------------------------ worker
    def _worker(self, focus: dict[str, list[str]]) -> None:
        try:
            self._run_once(focus)
            self._last_error = ""
        except Exception as exc:
            self._last_error = str(exc) or exc.__class__.__name__
            logging.exception("Movers refresh failed")
        finally:
            self._running = False
            self.statusChanged.emit(self.status_text())

    def _run_once(self, focus: dict[str, list[str]]) -> None:
        self._tick_no += 1
        now = self._clock()
        if now.tzinfo is None:
            now = now.astimezone()
        local_tz = _market_local_tz()
        today = now.astimezone(movers_scan.NY_TZ).date()
        if self._baseline_day != today:
            self._baselines = {}
            self._baseline_tried = {}
            self._baseline_day = today

        bot = self._bot_provider() if self._bot_provider is not None else None
        universe = bot_universe(bot)
        self.bot_universe_size = len(universe) if bot is not None else None
        focus_names = [*focus.get("long", []), *focus.get("short", [])]
        wanted = list(dict.fromkeys(["SPY", *universe, *focus_names]))
        bot_bars = read_bot_bars(bot, wanted)

        downloader = self._downloader
        if downloader is None:
            import autopilot_core as core

            downloader = core._default_downloader
        if len(universe) < BOT_UNIVERSE_MIN and self._yahoo_due(now):
            try:
                pool = list(dict.fromkeys(["SPY", *self._universe_provider(), *wanted]))
            except Exception:
                pool = wanted
            fetched = fetch_yahoo_bars(pool, downloader=downloader, period=YAHOO_TODAY_PERIOD)
            if fetched:
                # Only a sweep that returned bars moves the clock; a failed one retries.
                self._yahoo_bars = fetched
                self._yahoo_at = now

        else:
            self._gap_fill(wanted, bot_bars, now=now, local_tz=local_tz, downloader=downloader)

        self._refresh_daily_tags(today)
        series = choose_freshest(bot_bars, self._yahoo_bars, now=now, local_tz=local_tz)
        spy = series.pop("SPY", [])
        self._publish(series, spy, now, focus, local_tz, final=False)

        retry = timedelta(minutes=BASELINE_RETRY_MINUTES)
        missing = [
            s for s in series
            if s not in self._baselines
            and (s not in self._baseline_tried or now - self._baseline_tried[s] >= retry)
        ]
        if missing:
            history = fetch_yahoo_bars(
                missing, downloader=downloader, period=YAHOO_BASELINE_PERIOD
            )
            for symbol in missing:
                self._baseline_tried[symbol] = now
                if symbol in history:
                    self._baselines[symbol] = movers_scan.build_rvol_baseline(
                        history[symbol], before=today, local_tz=local_tz
                    )
        self._publish(series, spy, now, focus, local_tz, final=True)

    def _publish(self, series, spy, now, focus, local_tz, *, final: bool) -> None:
        """Build and emit. Only the tick's FINAL publish advances persistence
        and the outcome log, so a tick that publishes twice counts once."""
        board = movers_scan.build_movers_board(
            series, spy, now=now, baselines=self._baselines,
            focus_by_side=focus, local_tz=local_tz, earnings=self._earnings,
        )
        movers_scan.apply_group_tags(board, self._industry)
        session = now.astimezone(movers_scan.NY_TZ).date()
        memory = movers_scan.apply_persistence(
            board, self._persistence if final else dict(self._persistence), session=session
        )
        if final:
            self._persistence = memory
        board["bot_universe"] = self.bot_universe_size
        board["yahoo_universe"] = len(self._yahoo_bars)
        board["gap_filled"] = self._gap_count
        board["gap_filled_at"] = self._gap_at.isoformat(timespec="seconds") if self._gap_at else ""
        self._board = board
        self._last_success = datetime.now()
        self.moversChanged.emit(dict(board))
        if final:
            if not self._tracker_restored:
                # Once per desk start: today's logged flags rebuild pending
                # episodes, so a restart never re-flags a name (worker thread).
                self._tracker_restored = True
                try:
                    self._tracker.restore(
                        movers_outcomes.load_records(self._outcomes_path),
                        session=session, now=now,
                    )
                except Exception:
                    logging.warning("Movers outcome log could not be read back", exc_info=True)
            try:
                records = self._tracker.observe(board, series, spy, now=now)
            except Exception:
                logging.warning("Movers outcome tracker failed", exc_info=True)
                records = []
            if records:
                movers_outcomes.append_records(self._outcomes_path, records)

    # ------------------------------------------------------------ daily tags
    def _refresh_daily_tags(self, today: date) -> None:
        """Industry map and earnings names, read once per session on the worker."""
        if self._tags_day == today:
            return
        try:
            self._industry = dict(self._industry_provider() or {})
        except Exception:
            logging.warning("Movers: industry map unavailable", exc_info=True)
            self._industry = {}
        try:
            self._earnings = set(self._earnings_provider(today) or ())
        except Exception:
            logging.warning("Movers: earnings calendar unavailable", exc_info=True)
            self._earnings = set()
        self._tags_day = today

    # ------------------------------------------------------------ gap fill
    def _gap_fill(self, wanted, bot_bars, *, now, local_tz, downloader) -> None:
        """Yahoo 5m bars for names whose bot series is stale or missing (and whose
        cached Yahoo series is not fresh either), most liquid first, capped."""
        cutoff = movers_scan.freshness_cutoff(now)
        need: list[str] = []
        for symbol in wanted:
            fresh = False
            for source in (bot_bars.get(symbol), self._yahoo_bars.get(symbol)):
                bars = movers_scan.normalize_bars(source or (), now=now, local_tz=local_tz)
                if bars and bars[-1]["dt"] >= cutoff:
                    fresh = True
                    break
            if not fresh and self._gap_skip_until.get(symbol, 0) < self._tick_no:
                need.append(symbol)
        if not need:
            self._gap_count = 0
            return
        if len(need) > GAP_FILL_MAX:
            need = sorted(need, key=lambda s: (-self._liquidity(s, bot_bars), s))[:GAP_FILL_MAX]
        else:
            need = sorted(need, key=lambda s: (-self._liquidity(s, bot_bars), s))
        fetched = fetch_yahoo_bars(need, downloader=downloader, period=GAP_FILL_PERIOD)
        self._gap_count = len(fetched)
        for symbol in need:
            if symbol not in fetched:
                # Yahoo had nothing for it: skip it for the next few ticks.
                self._gap_skip_until[symbol] = self._tick_no + GAP_EMPTY_BACKOFF_TICKS
        if fetched:
            # Only a fetch that returned bars moves the clock; failures keep the last bars.
            self._yahoo_bars.update(fetched)
            self._gap_at = now

    def _liquidity(self, symbol: str, bot_bars) -> float:
        """Price x volume of the latest session we hold for the name; 0 when unknown."""
        for source in (bot_bars.get(symbol), self._yahoo_bars.get(symbol)):
            rows = [r for r in (source or []) if isinstance(r, Mapping)]
            if not rows:
                continue
            stamp = rows[-1].get("dt")
            day = stamp.date() if hasattr(stamp, "date") else None
            volume = sum(float(r.get("volume") or 0.0) for r in rows
                         if hasattr(r.get("dt"), "date") and r["dt"].date() == day)
            try:
                return float(rows[-1].get("close") or 0.0) * volume
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    def _yahoo_due(self, now: datetime) -> bool:
        if self._yahoo_at is None:
            return True
        try:
            from project_paths import get_local_setting

            minutes = float(get_local_setting(MOVERS_YAHOO_MINUTES_SETTING, MOVERS_YAHOO_MINUTES))
        except Exception:
            minutes = float(MOVERS_YAHOO_MINUTES)
        if minutes <= 0:
            minutes = float(MOVERS_YAHOO_MINUTES)
        return now - self._yahoo_at >= timedelta(minutes=minutes)
