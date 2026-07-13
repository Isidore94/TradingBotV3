"""Shadow bridge: run the new SPY market-state engine beside the legacy pause
detector (plan.md sec 16 / Phase 5.13 champion-challenger).

The legacy `_detect_spy_pause_start()` stays the champion - nothing here may
change live behavior. Each evaluation converts the bot's cached SPY 5-minute
bars, runs the pure MarketStateEngine, and appends a JSONL shadow record ONLY
when the engine state changes or its agreement with the legacy detector
flips. The log gives the promotion evidence the plan requires before the
engine replaces the one-red-candle pause rule.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path

from market_state import (
    ENGINE_VERSION,
    M5Bar,
    MarketStateConfig,
    MarketStateEngine,
    MarketState,
    MarketStateSnapshot,
)
from market_session import get_market_local_timezone, normalize_market_local_datetime

_BAR_MINUTES = 5
SHADOW_SCHEMA = "spy_state_shadow_v2"
STATUS_SCHEMA = "spy_state_shadow_status_v1"

# Engine states the legacy detector would call "paused".
_PAUSE_LIKE_STATES = {
    MarketState.COUNTERMOVE_ARMED,
    MarketState.COUNTERMOVE_ACTIVE,
    MarketState.STABILIZING,
}

_lock = threading.Lock()
_last_written: dict[str, str] = {}
_coverage: dict = {}


def shadow_log_path() -> Path:
    try:
        from project_paths import get_diagnostics_dir

        return get_diagnostics_dir() / "spy_state_shadow.jsonl"
    except Exception:
        return Path.home() / ".tradingbotv3" / "spy_state_shadow.jsonl"


def shadow_status_path() -> Path:
    return shadow_log_path().with_name("spy_state_shadow_status.json")


def _config_hash(config: MarketStateConfig) -> str:
    payload = {
        name: str(value) if isinstance(value, timedelta) else value
        for name, value in vars(config).items()
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _empty_coverage(session_date: str, config_hash: str) -> dict:
    return {
        "schema": STATUS_SCHEMA,
        "engine_version": ENGINE_VERSION,
        "config_hash": config_hash,
        "machine": socket.gethostname(),
        "session_date": session_date,
        "evaluations": 0,
        "usable_evaluations": 0,
        "skipped_missing_input": 0,
        "rows_written": 0,
        "errors": 0,
        "last_evaluation_at": "",
        "last_complete_bar_at": "",
        "last_error": "",
        "last_error_at": "",
    }


def _record_coverage(
    *,
    evaluated_at: datetime,
    config_hash: str,
    snapshot: MarketStateSnapshot | None = None,
    row_written: bool = False,
    missing_input: bool = False,
    error: Exception | None = None,
) -> None:
    global _coverage
    session_date = evaluated_at.date().isoformat()
    with _lock:
        if _coverage.get("session_date") != session_date or _coverage.get("config_hash") != config_hash:
            path = shadow_status_path()
            try:
                loaded = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            except (OSError, json.JSONDecodeError):
                loaded = {}
            _coverage = (
                loaded
                if loaded.get("session_date") == session_date and loaded.get("config_hash") == config_hash
                else _empty_coverage(session_date, config_hash)
            )
        _coverage["evaluations"] = int(_coverage.get("evaluations", 0)) + 1
        _coverage["last_evaluation_at"] = evaluated_at.isoformat(timespec="seconds")
        if snapshot is not None:
            _coverage["usable_evaluations"] = int(_coverage.get("usable_evaluations", 0)) + 1
            _coverage["last_complete_bar_at"] = (
                snapshot.ts.isoformat(timespec="seconds") if snapshot.ts is not None else ""
            )
        if missing_input:
            _coverage["skipped_missing_input"] = int(_coverage.get("skipped_missing_input", 0)) + 1
        if row_written:
            _coverage["rows_written"] = int(_coverage.get("rows_written", 0)) + 1
        if error is not None:
            _coverage["errors"] = int(_coverage.get("errors", 0)) + 1
            _coverage["last_error"] = str(error)[:500]
            _coverage["last_error_at"] = evaluated_at.isoformat(timespec="seconds")
        payload = dict(_coverage)
        path = shadow_status_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(tmp, path)
        except OSError:
            logging.warning("SPY shadow coverage status write failed.", exc_info=True)


def _rotate_legacy_shadow_if_needed(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    try:
        first = next(
            (line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()),
            "",
        )
        payload = json.loads(first) if first else {}
    except (OSError, json.JSONDecodeError):
        payload = {}
    if payload.get("schema") == SHADOW_SCHEMA:
        return
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    archive = path.with_name(f"{path.stem}.legacy-{stamp}{path.suffix}")
    counter = 1
    while archive.exists():
        archive = path.with_name(f"{path.stem}.legacy-{stamp}-{counter}{path.suffix}")
        counter += 1
    os.replace(path, archive)
    logging.info("Archived legacy SPY shadow evidence to %s", archive)


def m5_bars_from_bot_bars(bot_bars, *, now: datetime | None = None) -> list[M5Bar]:
    """Convert the bot's cached SPY bars; the last bar is marked incomplete
    while it can still be forming so the engine never acts on a partial bar."""
    local_timezone, _ = get_market_local_timezone()
    moment = normalize_market_local_datetime(now, local_timezone=local_timezone)
    bars: list[M5Bar] = []
    total = len(bot_bars)
    for index, bar in enumerate(bot_bars):
        raw_start = getattr(bar, "dt", None)
        if raw_start is None:
            continue
        start = normalize_market_local_datetime(raw_start, local_timezone=local_timezone)
        is_last = index == total - 1
        complete = True
        if is_last:
            complete = (moment - start) >= timedelta(minutes=_BAR_MINUTES)
        bars.append(
            M5Bar(
                ts=start + timedelta(minutes=_BAR_MINUTES),
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(getattr(bar, "volume", 0.0) or 0.0),
                complete=complete,
            )
        )
    return bars


def evaluate_spy_shadow_state(
    bot_bars,
    prev_close,
    *,
    now: datetime | None = None,
    config: MarketStateConfig | None = None,
) -> MarketStateSnapshot | None:
    """Fresh engine pass over today's cached SPY bars; None when unusable."""
    if not bot_bars or not prev_close:
        return None
    bars = m5_bars_from_bot_bars(bot_bars, now=now)
    if not bars:
        return None
    engine = MarketStateEngine(float(prev_close), config=config)
    snapshot = None
    for bar in bars:
        snapshot = engine.on_bar(bar)
    return snapshot


def record_spy_shadow(
    bot_bars,
    prev_close,
    *,
    legacy_pause_start=None,
    side: str = "",
    now: datetime | None = None,
    config: MarketStateConfig | None = None,
) -> dict | None:
    """Champion/challenger observation; appends to the shadow log only on an
    engine state change or an agreement flip. Never raises."""
    moment = normalize_market_local_datetime(now)
    _, timezone_name = get_market_local_timezone()
    active_config = config or MarketStateConfig()
    config_hash = _config_hash(active_config)
    try:
        snapshot = evaluate_spy_shadow_state(
            bot_bars,
            prev_close,
            now=moment,
            config=active_config,
        )
        if snapshot is None:
            _record_coverage(
                evaluated_at=moment,
                config_hash=config_hash,
                missing_input=True,
            )
            return None
        engine_paused = snapshot.state in _PAUSE_LIKE_STATES
        legacy_paused = legacy_pause_start is not None
        row = {
            "schema": SHADOW_SCHEMA,
            "ts": moment.isoformat(timespec="seconds"),
            "evaluated_at": moment.isoformat(timespec="seconds"),
            "bar_ts": snapshot.ts.isoformat(timespec="seconds") if snapshot.ts is not None else "",
            "session_date": moment.date().isoformat(),
            "timezone": timezone_name,
            "machine": socket.gethostname(),
            "engine_version": ENGINE_VERSION,
            "config_hash": config_hash,
            "observation_id": (
                f"SPY|{moment.date().isoformat()}|{snapshot.state.value}|"
                f"{snapshot.ts.isoformat(timespec='seconds') if snapshot.ts is not None else 'none'}"
            ),
            "state": snapshot.state.value,
            "side_sign": snapshot.side_sign,
            "trend_score": round(snapshot.trend_score, 3),
            "day_return_pct": round(snapshot.day_return_pct, 4),
            "vwap": round(snapshot.vwap, 4) if snapshot.vwap is not None else None,
            "m5_atr": round(snapshot.m5_atr, 4) if snapshot.m5_atr is not None else None,
            "depth_atr": round(snapshot.countermove_depth_atr, 3),
            "stale": snapshot.stale,
            "legacy_side": str(side or ""),
            "legacy_pause_start": (
                normalize_market_local_datetime(legacy_pause_start).isoformat(timespec="seconds")
                if hasattr(legacy_pause_start, "isoformat")
                else str(legacy_pause_start or "")
            ),
            "legacy_paused": legacy_paused,
            "engine_paused": engine_paused,
            "agree": engine_paused == legacy_paused,
            "input_bar_count": len(bot_bars or []),
            "prior_close": float(prev_close),
        }
        fingerprint = f"{row['state']}|{row['legacy_paused']}|{row['engine_paused']}"
        duplicate = False
        with _lock:
            if _last_written.get("fingerprint") == fingerprint:
                duplicate = True
            else:
                _last_written["fingerprint"] = fingerprint
        if duplicate:
            _record_coverage(
                evaluated_at=moment,
                config_hash=config_hash,
                snapshot=snapshot,
            )
            return row  # nothing new to persist
        path = shadow_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        _rotate_legacy_shadow_if_needed(path)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        _record_coverage(
            evaluated_at=moment,
            config_hash=config_hash,
            snapshot=snapshot,
            row_written=True,
        )
        return row
    except Exception as exc:
        _record_coverage(
            evaluated_at=moment,
            config_hash=config_hash,
            error=exc,
        )
        logging.warning("SPY shadow-state recording failed (live behavior unaffected).", exc_info=True)
        return None


def reset_shadow_dedupe() -> None:
    """Test hook: forget the last written fingerprint."""
    global _coverage
    with _lock:
        _last_written.clear()
        _coverage = {}
