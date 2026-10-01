"""P18 E: the desk's one writer of M5 bars on disk, for the Trade Mentor app (never the bot, never IB).

Every 60 s in the session a desk ``QTimer`` hands one pass to this object's own worker thread. A pass asks the
bounce proxy ``current_bot().m5_chart_bars(sym, 2)`` (cache only) for up to ``BATCH`` names of the watch list,
round robin, ``RPC_GAP_SECONDS`` apart so the proxy's lock is not starved; a call slower than ``SLOW_CALL_MS``
halves the next batch. It appends the COMPLETED bars it has not written yet (never the forming one) to
``M5_BARS_DIR/<ET date>.jsonl`` and rewrites ``M5_LATEST_FILE`` (last completed bar per symbol + the session
VWAP over the full regular session) atomically. A failed pass or write is logged and never raises into the desk.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time as _time
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
BAR = timedelta(minutes=5)
TICK_MS = 60_000
BATCH = 50
MIN_BATCH = 10
SLOW_CALL_MS = 200.0
RPC_GAP_SECONDS = 0.01
#: The publisher runs 09:25-16:10 ET on weekdays (a few minutes either side of the regular session).
SESSION_START, SESSION_END = time(9, 25), time(16, 10)
RTH_OPEN, RTH_CLOSE = time(9, 30), time(16, 0)
SCHEMA_LATEST = "m5_latest_v1"
SOURCE = "desk_publisher"
INDEX_SYMBOLS = ("SPY", "QQQ", "IWM")

_log = logging.getLogger(__name__)


def in_session(now: datetime) -> bool:
    local = now.astimezone(ET)
    return local.weekday() < 5 and SESSION_START <= local.time() <= SESSION_END


def _float(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def bar_start(raw: Mapping[str, Any], market_tz: Any) -> datetime | None:
    """The bar's start, tz-aware; a naive time is market-local (the bot's bars are)."""
    value = raw.get("dt") or raw.get("interval_start") or raw.get("start")
    if isinstance(value, datetime):
        moment = value
    else:
        try:
            moment = datetime.fromisoformat(str(value or "").strip())
        except ValueError:
            return None
    return moment if moment.tzinfo else moment.replace(tzinfo=market_tz)


def completed_bars(raw_bars: Iterable[Mapping[str, Any]], now: datetime, market_tz: Any) -> list[dict[str, Any]]:
    """Completed bars only (start + 5 min <= now), oldest first, one per start, tz-aware ET starts."""
    out: dict[datetime, dict[str, Any]] = {}
    for raw in raw_bars or ():
        start = bar_start(raw, market_tz)
        values = [_float(raw.get(key)) for key in ("open", "high", "low", "close")]
        if start is None or None in values or raw.get("is_complete") is False or start + BAR > now:
            continue
        out[start] = {"start": start.astimezone(ET).isoformat(), "open": values[0], "high": values[1],
                      "low": values[2], "close": values[3], "volume": _float(raw.get("volume")) or 0.0}
    return [out[key] for key in sorted(out)]


def session_vwap(bars: list[Mapping[str, Any]], day: str) -> tuple[float | None, bool, int]:
    """(typical-price VWAP over the day's regular-session bars, complete-from-09:30-with-no-gap, bar count)."""
    session = []
    for bar in bars:
        start = datetime.fromisoformat(str(bar["start"])).astimezone(ET)
        if start.date().isoformat() == day and RTH_OPEN <= start.time() < RTH_CLOSE:
            session.append((start, bar))
    if not session:
        return None, False, 0
    volume = sum(float(bar["volume"]) for _s, bar in session)
    complete = session[0][0].time() == RTH_OPEN and all(
        later[0] - earlier[0] == BAR for earlier, later in zip(session, session[1:], strict=False))
    if volume <= 0:
        return None, complete, len(session)
    value = sum((float(b["high"]) + float(b["low"]) + float(b["close"])) / 3 * float(b["volume"])
                for _s, b in session) / volume
    return value, complete, len(session)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_latest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def publish(pulled: Mapping[str, Iterable[Mapping[str, Any]]], now: datetime, *, directory: Path, latest: Path,
            market_tz: Any, written: dict[str, str]) -> dict[str, Any]:
    """Append each symbol's new completed bars and rewrite the latest file. ``written`` = last start per symbol
    (updated in place). Returns ``{appended, symbols}``."""
    moment = now if now.tzinfo else now.astimezone()
    pulled_utc = moment.astimezone(timezone.utc).isoformat(timespec="seconds")
    state = read_latest(latest)
    symbols: dict[str, Any] = dict(state.get("symbols") or {})
    by_day: dict[str, list[str]] = {}
    for symbol, raw in pulled.items():
        bars = completed_bars(raw, moment, market_tz)
        if not bars:
            continue
        last_written = written.get(symbol, "")
        for bar in bars:
            if bar["start"] > last_written:
                day = bar["start"][:10]
                by_day.setdefault(day, []).append(json.dumps(
                    {"symbol": symbol, **bar, "source": SOURCE, "pulled_utc": pulled_utc}, sort_keys=True))
        last = bars[-1]
        vwap, complete, count = session_vwap(bars, last["start"][:10])
        symbols[symbol] = {"bar": last, "session_vwap": vwap, "vwap_complete": complete, "session_bars": count,
                           "pulled_utc": pulled_utc}
    appended = 0
    for day, lines in sorted(by_day.items()):
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / f"{day}.jsonl", "a", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines) + "\n")
        appended += len(lines)
    for symbol, row in symbols.items():
        written[symbol] = max(written.get(symbol, ""), str(row["bar"]["start"]))
    atomic_write_json(latest, {"schema": SCHEMA_LATEST, "written_utc": pulled_utc, "symbols": symbols})
    return {"appended": appended, "symbols": len(pulled)}


def seed_written(latest: Path) -> dict[str, str]:
    """After a restart: the last start already on disk per symbol, so no bar is written twice."""
    return {str(sym): str((row.get("bar") or {}).get("start") or "")
            for sym, row in (read_latest(latest).get("symbols") or {}).items() if isinstance(row, Mapping)}


def live_watch_symbols() -> list[str]:
    """The desk's names: SPY/QQQ/IWM, then longs/shorts and the Focus lists (files read on the worker)."""
    from project_paths import FOCUS_LONGS_FILE, FOCUS_SHORTS_FILE, LONGS_FILE, SHORTS_FILE

    names: list[str] = list(INDEX_SYMBOLS)
    for path in (LONGS_FILE, SHORTS_FILE, FOCUS_LONGS_FILE, FOCUS_SHORTS_FILE):
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            continue
        for word in text.replace(",", "\n").split():
            sym = word.strip().upper().lstrip("$")
            if sym and sym.replace(".", "").replace("-", "").isalnum() and sym not in names:
                names.append(sym)
    return names


class M5BarPublisher:
    """One owner of ``M5_BARS_DIR``: a pass at a time on its own thread; the desk's timer only asks for one."""

    def __init__(self, bot: Callable[[], Any], *, symbols: Callable[[], list[str]] = live_watch_symbols,
                 directory: Path | None = None, latest: Path | None = None, market_tz: Any = None,
                 now: Callable[[], datetime] | None = None, gap_seconds: float = RPC_GAP_SECONDS) -> None:
        if directory is None or latest is None:
            from project_paths import M5_BARS_DIR, M5_LATEST_FILE

            directory = Path(directory or M5_BARS_DIR)
            latest = Path(latest or M5_LATEST_FILE)
        self._bot, self._symbols = bot, symbols
        self.directory, self.latest = Path(directory), Path(latest)
        self._market_tz = market_tz
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._gap = float(gap_seconds)
        self._written: dict[str, str] | None = None
        self._cursor = 0
        self.batch = BATCH
        self._busy = threading.Lock()
        self.last_result: dict[str, Any] = {}
        self.last_call_ms: float = 0.0

    def _tz(self) -> Any:
        if self._market_tz is None:
            import market_session

            self._market_tz = market_session.get_market_local_timezone()[0]
        return self._market_tz

    def tick(self) -> bool:
        """Qt thread: start one pass on a worker unless one is running or the session is closed."""
        if not in_session(self._now()) or not self._busy.acquire(blocking=False):
            return False
        thread = threading.Thread(target=self._run_locked, name="m5-bar-publisher", daemon=True)
        thread.start()
        return True

    def _run_locked(self) -> None:
        try:
            self.run_once()
        finally:
            self._busy.release()

    def run_once(self) -> dict[str, Any]:
        """One pass (worker thread). Never raises."""
        try:
            bot = self._bot()
            if bot is None:
                self.last_result = {"skipped": "no bot"}
                return self.last_result
            names = list(self._symbols())
            if not names:
                self.last_result = {"skipped": "no symbols"}
                return self.last_result
            if self._written is None:
                self._written = seed_written(self.latest)
            start = self._cursor % len(names)
            chunk = (names[start:] + names[:start])[:self.batch]
            self._cursor = start + len(chunk)
            pulled: dict[str, Any] = {}
            slowest = 0.0
            for symbol in chunk:
                began = _time.perf_counter()
                try:
                    pulled[symbol] = bot.m5_chart_bars(symbol, max_sessions=2) or []
                except Exception:  # noqa: BLE001 - one name failing never stops the pass
                    _log.debug("m5 publisher: %s not read", symbol, exc_info=True)
                slowest = max(slowest, (_time.perf_counter() - began) * 1000)
                if self._gap:
                    _time.sleep(self._gap)
            self.last_call_ms = slowest
            # A slow proxy gets smaller batches; a quick one grows back to BATCH.
            self.batch = max(MIN_BATCH, self.batch // 2) if slowest > SLOW_CALL_MS else min(BATCH, self.batch * 2)
            result = publish(pulled, self._now(), directory=self.directory, latest=self.latest,
                             market_tz=self._tz(), written=self._written)
            self.last_result = {**result, "slowest_ms": round(slowest, 1), "batch": len(chunk)}
        except Exception:  # noqa: BLE001 - a failed publish never raises into the desk; the last good file stays
            _log.warning("m5 publisher: pass failed", exc_info=True)
            self.last_result = {"failed": True}
        return self.last_result


def start_desk_timer(parent: Any, publisher: M5BarPublisher) -> Any:
    """The desk's one 60 s ``QTimer`` for the publisher (Qt thread; the pass runs on the publisher's thread)."""
    from PySide6.QtCore import QTimer

    timer = QTimer(parent)
    timer.setInterval(TICK_MS)
    timer.timeout.connect(publisher.tick)
    timer.start()
    return timer
