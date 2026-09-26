"""S7 shadow setups sidecar: the four shadow engines over the bot's cached M5 bars.

SHADOW ONLY. The engines (`m5_signal_engines.shadow_setup_events`) run over the
champion's in-memory bar cache (`latest_bars`, cache only, never IB) and each new
event is appended to `project_paths.M5_SHADOW_SETUPS_FILE`. Nothing here makes an
alert, a score, a Show row or a phone line, and nothing live reads the sidecar.
Graduation only through the `docs/SETUPS_TEST.md` ladder.

The (b) VWAP-reclaim engine asks for the market environment at the reclaim bar:
the champion's own auto read (`_auto_market_regime_stats`) over SPY's cached
bars up to that bar, so it is point-in-time. No SPY bars is no event.

Owner: `ShadowSetupsCapture` is the only writer (append-only jsonl, one line per
event, de-duplicated by event_id). `submit` never raises and never blocks: a
latest-wins mailbox and one daemon worker. A failed write loses the events.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, tzinfo
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from completed_bars import bar_time, completed_m5_bars
from m5_signal_engines import ShadowSetupEvent, shadow_setup_events
from swallowed import note_swallowed

SCHEMA = "m5_shadow_setup_v1"
#: Set to "0" to switch the sidecar off (a bad day).
ENABLED_ENV = "TRADINGBOTV3_M5_SHADOW_SETUPS"
M5_KEY_SUFFIX = "|5 D|5 mins"
BENCHMARK = "SPY"


def enabled() -> bool:
    return os.environ.get(ENABLED_ENV, "1").strip() != "0"


def sidecar_path() -> Path:
    import project_paths

    return Path(project_paths.M5_SHADOW_SETUPS_FILE)


def local_tz() -> tzinfo:
    """The zone the bot's naive bar stamps are written in."""
    from market_session import get_market_local_timezone

    return get_market_local_timezone()[0]


def _chart_dict(bar: Any) -> Any:
    """A cached bar as a plain dict (the bot caches `IbBar` objects; the engines read keys)."""
    if isinstance(bar, Mapping):
        return bar
    return {name: getattr(bar, name, None) for name in ("dt", "open", "high", "low", "close", "volume")}


def m5_series(latest_bars: Mapping[Any, Any]) -> dict[str, list]:
    """{symbol: M5 bars as dicts} from the bot's ``|5 D|5 mins`` cache keys only.

    The plain symbol key can hold any bar size, so it is never read as M5.
    """
    out: dict[str, list] = {}
    for key, bars in dict(latest_bars or {}).items():
        text = str(key)
        if text.endswith(M5_KEY_SUFFIX) and bars:
            out[text[: -len(M5_KEY_SUFFIX)].strip().upper()] = [_chart_dict(bar) for bar in bars]
    return out


def _as_ib(bar: Any) -> SimpleNamespace | None:
    get = bar.get if isinstance(bar, Mapping) else lambda name, default=None: getattr(bar, name, default)
    try:
        return SimpleNamespace(
            dt=bar_time(bar), open=float(get("open")), high=float(get("high")), low=float(get("low")),
            close=float(get("close")), volume=float(get("volume") or 0.0),
        )
    except (TypeError, ValueError):
        return None


def spy_environment_reader(spy_bars: list, *, now: datetime, tz: tzinfo) -> Callable[[datetime], str | None]:
    """environment_at(bar start) from the champion's auto regime read over SPY's bars to that bar.

    The SPY session is the bar's market-local date and the reference close is the
    last close before it, as the champion reads it. Unknown is None.
    """
    from bounce_bot_lib.legacy import _auto_market_regime_stats

    bars = [item for item in (_as_ib(bar) for bar in completed_m5_bars(spy_bars or [], now=now)) if item]
    bars = [bar for bar in bars if bar.dt is not None]

    def environment_at(start: datetime) -> str | None:
        local = start.astimezone(tz).replace(tzinfo=None) if start.tzinfo is not None else start
        day = local.date()
        today = [bar for bar in bars if _naive(bar.dt, tz).date() == day and _naive(bar.dt, tz) <= local]
        prior = [bar for bar in bars if _naive(bar.dt, tz).date() < day]
        if not today or not prior:
            return None
        try:
            reading = _auto_market_regime_stats(today, prior[-1].close)
        except Exception as exc:  # noqa: BLE001 - an unreadable environment is unknown
            note_swallowed("S7 SPY environment read failed", exc, quiet=True)
            return None
        return str(reading["env_key"]) if reading else None

    return environment_at


def _naive(stamp: datetime, tz: tzinfo) -> datetime:
    return stamp.astimezone(tz).replace(tzinfo=None) if stamp.tzinfo is not None else stamp


def record_of(event: ShadowSetupEvent, *, observed_at: datetime) -> dict[str, Any]:
    """The sidecar line for one event: what the S3 bracket needs, plus the engine's details."""
    return {
        "schema": SCHEMA,
        "shadow_only": True,
        "event_id": event.event_id,
        "engine": event.engine,
        "symbol": event.symbol,
        "side": event.side,
        "session": event.bar_time.date().isoformat(),
        "bar_time": event.bar_time.isoformat(),
        "bar_close": event.bar_close.isoformat(),
        "level": event.level,
        "entry": event.entry,
        "stop": event.stop,
        "risk_per_share": round(event.risk_per_share, 6),
        "details": dict(event.details),
        "observed_at": observed_at.isoformat(),
    }


def records_for(
    bars_by_symbol: Mapping[str, list],
    *,
    now: datetime,
    tz: tzinfo,
    environment_at: Callable[[datetime], str | None] | None,
) -> list[dict[str, Any]]:
    """Every shadow event on each symbol's latest session, as sidecar records."""
    out: list[dict[str, Any]] = []
    for symbol in sorted(bars_by_symbol):
        try:
            events = shadow_setup_events(
                bars_by_symbol[symbol], symbol=symbol, now=now, tz=tz, environment_at=environment_at
            )
        except Exception as exc:  # noqa: BLE001 - one bad series costs that symbol, never the pass
            note_swallowed(f"S7 shadow engines failed for {symbol}", exc, quiet=True)
            continue
        out.extend(record_of(event, observed_at=now) for event in events)
    return out


def read_event_ids(path: Path) -> set[str]:
    """Every event_id already in the sidecar (damaged lines skipped)."""
    ids: set[str] = set()
    try:
        with open(path, "rb") as handle:
            for raw in handle:
                try:
                    event_id = json.loads(raw).get("event_id")
                except (ValueError, AttributeError):
                    continue
                if event_id:
                    ids.add(str(event_id))
    except FileNotFoundError:
        pass
    return ids


def append_records(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


class ShadowSetupsCapture:
    """The GUI-owned shadow-setups pass: latest-wins mailbox, one daemon worker, one writer."""

    CLOSE_TIMEOUT = 2.0

    def __init__(self, *, path: Path | None = None, tz: tzinfo | None = None,
                 clock: Callable[[tzinfo], datetime] | None = None):
        self._path = path
        self._tz = tz
        self._clock = clock or (lambda zone: datetime.now(zone))
        self._seen: set[str] | None = None
        self._pending: Any = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._closing = threading.Event()
        self._worker: threading.Thread | None = None
        self.last_error = ""
        self.passes = 0
        self.rows_written = 0

    # -- the GUI-thread half: memory only -----------------------------------
    def submit(self, bot) -> bool:
        """Hand the bot's bar cache to the worker. Never raises, never blocks."""
        if bot is None or not enabled():
            return False
        try:
            if bool(getattr(bot, "is_process_proxy", False)):
                snapshot = bot  # the worker reads the proxy's cache over RPC
            else:
                cache = getattr(bot, "latest_bars", None)
                if not cache:
                    return False
                snapshot = dict(cache)  # on the owning thread, before anything iterates
            with self._lock:
                self._pending = snapshot
                self._idle.clear()
                self._closing.clear()
                if self._worker is None or not self._worker.is_alive():
                    self._worker = threading.Thread(target=self._run, name="m5-shadow-setups", daemon=True)
                    self._worker.start()
            self._wake.set()
            return True
        except Exception as exc:  # noqa: BLE001 - a shadow pass never costs the desk
            note_swallowed("S7 shadow setups submit failed", exc, quiet=True)
            return False

    def close(self, timeout: float | None = None) -> None:
        self._closing.set()
        self._wake.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(self.CLOSE_TIMEOUT if timeout is None else float(timeout))

    def wait_idle(self, timeout: float = 5.0) -> bool:
        return self._idle.wait(timeout)

    # -- the worker-thread half ----------------------------------------------
    def _run(self) -> None:
        while not self._closing.is_set():
            with self._lock:
                pending, self._pending = self._pending, None
            if pending is None:
                self._idle.set()
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            try:
                self.run_pass(pending)
            except Exception as exc:  # noqa: BLE001 - never fatal; the events are lost
                self._log_once(f"{type(exc).__name__}: {exc}")
        self._idle.set()

    def run_pass(self, snapshot: Any) -> int:
        """One pass over a cache snapshot (dict, or the proxy to read). Returns rows written."""
        if not isinstance(snapshot, dict):
            snapshot = snapshot.latest_bars
            snapshot = dict(snapshot) if isinstance(snapshot, dict) else {}
        series = m5_series(snapshot)
        if not series:
            return 0
        tz = self._tz or local_tz()
        now = self._clock(tz)
        environment_at = None
        if series.get(BENCHMARK):
            environment_at = spy_environment_reader(series[BENCHMARK], now=now, tz=tz)
        records = records_for(series, now=now, tz=tz, environment_at=environment_at)
        path = self._path or sidecar_path()
        if self._seen is None:
            self._seen = read_event_ids(path)
        fresh = [record for record in records if record["event_id"] not in self._seen]
        self.passes += 1
        if not fresh:
            return 0
        append_records(fresh, path)
        self._seen.update(record["event_id"] for record in fresh)
        self.rows_written += len(fresh)
        self.last_error = ""
        return len(fresh)

    def _log_once(self, reason: str) -> None:
        if reason != self.last_error:
            self.last_error = reason
            logging.warning("S7 shadow setups pass failed; the desk is unaffected: %s", reason)


__all__ = [
    "SCHEMA",
    "ShadowSetupsCapture",
    "append_records",
    "m5_series",
    "read_event_ids",
    "record_of",
    "records_for",
    "spy_environment_reader",
]
