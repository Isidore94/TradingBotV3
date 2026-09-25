"""One bounded background reader for Trade Mentor market context."""

from __future__ import annotations

import logging
import threading
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping

from PySide6.QtCore import QObject, QTimer, Signal

from trade_mentor_context import SYMBOLS, build_context, unavailable_context
from swallowed import note_swallowed

_MAX_HOUR_CACHE = 4
_MAX_D1_SESSION_CACHE = 2

class TradeMentorContextService(QObject):
    """Fetch one all-symbol snapshot only when a card is shown.

    There is no polling timer.  The hour cache makes a second manual read cheap;
    its original capture time stays intact so an old observation is never dressed
    up as a current one.
    """

    contextReady = Signal(str, object)
    contextUnavailable = Signal(str, object)

    def __init__(
        self,
        parent=None,
        *,
        loader: Callable[..., Mapping[str, Any]] | None = None,
        clock: Callable[[], datetime] | None = None,
        timeout_seconds: float = 4,
        cache_loader: Callable[..., Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__(parent)
        self._loader = loader or _yahoo_loader
        self._cache_loader = cache_loader
        self._clock = clock or _market_clock
        self._timeout_seconds = max(0.1, float(timeout_seconds))
        self._worker: threading.Thread | None = None
        self._request_id = ""
        self._request_hour: tuple[str, int] | None = None
        self._hour_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._d1_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self._d1_attempted_sessions: set[str] = set()
        self._failed_hours: set[tuple[str, int]] = set()
        self._closed = False

    def request_context(self, request_id: str, *, now: datetime | None = None) -> bool:
        """Start one request, or replay its hour's immutable snapshot."""
        if self._closed or self._worker is not None:
            return False
        moment = now or self._clock()
        if not isinstance(moment, datetime) or moment.tzinfo is None:
            return False
        key = _hour_key(moment)
        self._prune_caches()
        cached = self._hour_cache.get(key)
        if cached is not None:
            self._queue_delivery("ready", str(request_id), _context_at_read_time(cached, moment))
            return True
        if key in self._failed_hours:
            return False
        self._request_id = str(request_id)
        self._request_hour = key
        worker = threading.Thread(
            target=self._run_worker, args=(moment,), name="trade-mentor-context", daemon=True
        )
        self._worker = worker
        worker.start()
        return True

    def shutdown(self, timeout_ms: int = 250) -> None:
        """Stop accepting work and wait only a bounded time for a read in flight."""
        self._closed = True
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(max(0, int(timeout_ms)) / 1000.0)
        # A slow daemon worker cannot block process exit.  Once shutdown marks
        # this service closed its late result is discarded without a Qt call.

    def _run_worker(self, moment: datetime) -> None:
        try:
            context = self._build(moment)
            no_measurements = all(
                row.get("m5_status") == "unavailable"
                and row.get("d1_status") == "unavailable"
                for row in context.get("readings", ())
            )
            payload: dict[str, Any] = (
                {"ok": False, "reason": "all context sources unavailable"}
                if no_measurements
                else {"ok": True, "context": context}
            )
        except Exception as exc:  # noqa: BLE001 - no context must cost a read
            payload = {"ok": False, "reason": str(exc) or "context loader failed"}
        self._on_done(payload)

    def _prune_caches(self) -> None:
        """Keep one small read window, never a day-long context archive."""
        for key in sorted(self._hour_cache)[:-_MAX_HOUR_CACHE]:
            self._hour_cache.pop(key, None)
        for key in sorted(self._failed_hours)[:-_MAX_HOUR_CACHE]:
            self._failed_hours.discard(key)
        for key in sorted(self._d1_cache)[:-_MAX_D1_SESSION_CACHE]:
            self._d1_cache.pop(key, None)
        for key in sorted(self._d1_attempted_sessions)[:-_MAX_D1_SESSION_CACHE]:
            self._d1_attempted_sessions.discard(key)

    def _queue_delivery(self, kind: str, request_id: str, context: dict[str, Any]) -> None:
        """Deliver on the next GUI turn when called on Qt's GUI thread."""
        def emit() -> None:
            if self._closed:
                return
            if kind == "ready":
                self.contextReady.emit(request_id, context)
            else:
                self.contextUnavailable.emit(request_id, context)

        if threading.current_thread() is threading.main_thread():
            QTimer.singleShot(1, emit)
        else:
            # Qt automatically queues this signal to every QObject receiver
            # owned by the GUI thread.  The service itself touches no widgets.
            emit()

    def _build(self, moment: datetime) -> dict[str, Any]:
        cached: dict[str, Mapping[str, Any]] = {"m5": {}, "d1": {}}
        session_key = _completed_session_key(moment)
        saved_d1 = self._d1_cache.get(session_key, {}) if session_key else {}
        cached["d1"] = dict(saved_d1)
        if self._cache_loader is not None:
            for timeframe in ("m5", "d1"):
                try:
                    candidate = self._cache_loader(
                        timeframe, SYMBOLS, now=moment, timeout_seconds=self._timeout_seconds
                    )
                    if isinstance(candidate, Mapping):
                        normalized = _normalize_bars(candidate, timeframe)
                        if timeframe == "d1" and saved_d1:
                            # Each saved D1 symbol has already passed the
                            # completed-session check. A local cache can fill
                            # a gap but cannot replace a valid saved symbol.
                            cached[timeframe].update(
                                {
                                    symbol: bars
                                    for symbol, bars in normalized.items()
                                    if symbol not in saved_d1
                                }
                            )
                        else:
                            cached[timeframe] = normalized
                except Exception:
                    logging.debug("Trade Mentor %s cache unreadable.", timeframe, exc_info=True)
        initial = build_context(
            now=moment, m5_bars=cached["m5"], d1_bars=cached["d1"],
            sources={"m5": "cached", "d1": "cached"},
        )
        needed = {
            timeframe: tuple(
                row["symbol"]
                for row in initial["readings"]
                if row[f"{timeframe}_status"] != "measured"
            )
            for timeframe in ("m5", "d1")
        }
        if session_key and session_key in self._d1_attempted_sessions:
            needed["d1"] = ()
        final = {name: dict(value) for name, value in cached.items()}
        sources: dict[str, str] = {}
        failures: dict[str, str] = {}
        for timeframe in ("m5", "d1"):
            names = needed[timeframe]
            if names:
                if timeframe == "d1" and session_key:
                    # An empty or partial Yahoo response is still this
                    # completed session's one bounded refresh attempt.
                    self._d1_attempted_sessions.add(session_key)
                try:
                    fetched = self._loader(
                        timeframe, names, now=moment, timeout_seconds=self._timeout_seconds
                    )
                except Exception as exc:  # noqa: BLE001 - a leg may fail alone
                    failures[timeframe] = str(exc) or f"{timeframe} loader failed"
                    sources[timeframe] = "cached" if cached[timeframe] else "unavailable"
                    continue
                fresh = _normalize_bars(fetched, timeframe)
                final[timeframe].update(fresh)
                sources[timeframe] = "mixed:cached,yahoo" if cached[timeframe] else "yahoo"
            else:
                sources[timeframe] = "cached"
        context = build_context(
            now=moment, m5_bars=final["m5"], d1_bars=final["d1"], sources=sources
        )
        for timeframe, failure in failures.items():
            for row in context["readings"]:
                status = row[f"{timeframe}_status"]
                if status == "unavailable":
                    row[f"{timeframe}_reason"] = failure
        if session_key:
            valid_d1 = {
                row["symbol"]: final["d1"][row["symbol"]]
                for row in context["readings"]
                if row["d1_status"] == "measured" and row["symbol"] in final["d1"]
            }
            if valid_d1:
                self._d1_cache[session_key] = valid_d1
        self._prune_caches()
        return context

    def _on_done(self, payload: object) -> None:
        self._worker = None
        request_id = self._request_id
        request_hour = self._request_hour or _hour_key(self._clock())
        self._request_id = ""
        self._request_hour = None
        if self._closed:
            return
        moment = self._clock()
        key = request_hour
        if not isinstance(payload, Mapping) or not payload.get("ok"):
            self._failed_hours.add(key)
            reason = str(payload.get("reason") if isinstance(payload, Mapping) else "context loader failed")
            self._queue_delivery("unavailable", request_id, unavailable_context(now=moment, reason=reason))
            return
        context = payload.get("context")
        if not isinstance(context, Mapping):
            self._failed_hours.add(key)
            self._queue_delivery("unavailable", request_id, unavailable_context(now=moment, reason="context loader returned no snapshot"))
            return
        captured = _captured_hour(context, moment)
        self._hour_cache[captured] = dict(context)
        self._queue_delivery("ready", request_id, _context_at_read_time(context, moment))


def _market_clock() -> datetime:
    from market_calendar import MARKET_TZ

    return datetime.now(MARKET_TZ)


def _hour_key(moment: datetime) -> tuple[str, int]:
    from market_calendar import MARKET_TZ

    market = moment.astimezone(MARKET_TZ)
    return market.date().isoformat(), market.hour


def _completed_session_key(moment: datetime) -> str:
    """D1 cache key: the latest session that actually closed, including half days."""
    try:
        from market_calendar import MARKET_TZ, is_session, previous_session
        from market_early_close import session_close

        market = moment.astimezone(MARKET_TZ)
        day = market.date()
        if is_session(day) and session_close(day) <= market:
            return day.isoformat()
        return previous_session(day).isoformat()
    except Exception:
        return ""


def _captured_hour(context: Mapping[str, Any], fallback: datetime) -> tuple[str, int]:
    try:
        return _hour_key(datetime.fromisoformat(str(context.get("captured_at") or "")))
    except ValueError:
        return _hour_key(fallback)


def _context_at_read_time(context: Mapping[str, Any], now: datetime) -> dict[str, Any]:
    result = deepcopy(dict(context))
    try:
        captured = datetime.fromisoformat(str(result.get("captured_at") or ""))
        age = now - captured.astimezone(now.tzinfo)
    except (TypeError, ValueError):
        return result
    if age > timedelta(minutes=15):
        result["availability"] = "stale"
        result["reason"] = f"context snapshot is {int(age.total_seconds() // 60)} minutes old"
    return result


def _normalize_bars(payload: Mapping[str, Any], timeframe: str) -> dict[str, list[dict[str, Any]]]:
    """Copy cache/download rows and attach known market time to naive M5 bars."""
    from market_calendar import MARKET_TZ

    normalized: dict[str, list[dict[str, Any]]] = {}
    for symbol in SYMBOLS:
        source = payload.get(symbol) or payload.get(symbol.lower()) or []
        rows: list[dict[str, Any]] = []
        for item in source if isinstance(source, (list, tuple)) else ():
            if isinstance(item, Mapping):
                row = dict(item)
            else:
                row = {name: getattr(item, name, None) for name in ("dt", "timestamp", "date", "open", "high", "low", "close", "volume")}
            stamp = row.get("dt") or row.get("timestamp") or row.get("date")
            if timeframe == "m5" and isinstance(stamp, datetime) and stamp.tzinfo is None:
                row["dt"] = stamp.replace(tzinfo=MARKET_TZ)
            rows.append(row)
        if rows:
            normalized[symbol] = rows
    return normalized


def _yahoo_loader(timeframe: str, symbols: tuple[str, ...], *, now: datetime, timeout_seconds: float) -> dict[str, list[dict[str, Any]]]:
    """One Yahoo batch for the names whose local cache could not answer."""
    if not symbols:
        return {}
    from autopilot_core import _frame_rows
    from yahoo_download import download

    kwargs = {
        "tickers": " ".join(symbols),
        "period": "1d" if timeframe == "m5" else "3mo",
        "interval": "5m" if timeframe == "m5" else "1d",
        "group_by": "ticker", "auto_adjust": False, "progress": False,
        "threads": False, "timeout": timeout_seconds,
    }
    if timeframe == "m5":
        kwargs["prepost"] = False
    data = download(**kwargs)
    result: dict[str, list[dict[str, Any]]] = {}
    for symbol in symbols:
        frame = _symbol_frame(data, symbol, len(symbols))
        rows = _frame_rows(frame)
        if rows:
            result[symbol] = rows
    return result


def _symbol_frame(data: Any, symbol: str, count: int) -> Any:
    try:
        columns = data.columns
        if getattr(columns, "nlevels", 1) > 1:
            for level in range(columns.nlevels):
                if symbol in columns.get_level_values(level):
                    return data.xs(symbol, axis=1, level=level)
    except Exception as swallowed_exc:
        note_swallowed("multi-index frame has no column level for the symbol", swallowed_exc, quiet=True)
    if count == 1:
        return data
    try:
        return data[symbol]
    except Exception as exc:
        note_swallowed("frame has no column for the symbol", exc, quiet=True)
    return None
