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

import json
import logging
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

_BAR_MINUTES = 5

# Engine states the legacy detector would call "paused".
_PAUSE_LIKE_STATES = {
    MarketState.COUNTERMOVE_ARMED,
    MarketState.COUNTERMOVE_ACTIVE,
    MarketState.STABILIZING,
}

_lock = threading.Lock()
_last_written: dict[str, str] = {}


def shadow_log_path() -> Path:
    try:
        from project_paths import CACHE_DIR

        return Path(CACHE_DIR).parent / "diagnostics" / "spy_state_shadow.jsonl"
    except Exception:
        return Path.home() / ".tradingbotv3" / "spy_state_shadow.jsonl"


def m5_bars_from_bot_bars(bot_bars, *, now: datetime | None = None) -> list[M5Bar]:
    """Convert the bot's cached SPY bars; the last bar is marked incomplete
    while it can still be forming so the engine never acts on a partial bar."""
    moment = now or datetime.now()
    bars: list[M5Bar] = []
    total = len(bot_bars)
    for index, bar in enumerate(bot_bars):
        start = getattr(bar, "dt", None)
        if start is None:
            continue
        is_last = index == total - 1
        complete = True
        if is_last:
            try:
                complete = (moment - start) >= timedelta(minutes=_BAR_MINUTES)
            except TypeError:
                complete = True  # tz-mismatched clocks: keep legacy leniency
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
) -> dict | None:
    """Champion/challenger observation; appends to the shadow log only on an
    engine state change or an agreement flip. Never raises."""
    try:
        snapshot = evaluate_spy_shadow_state(bot_bars, prev_close, now=now)
        if snapshot is None:
            return None
        engine_paused = snapshot.state in _PAUSE_LIKE_STATES
        legacy_paused = legacy_pause_start is not None
        row = {
            "ts": (now or datetime.now()).isoformat(timespec="seconds"),
            "engine_version": ENGINE_VERSION,
            "state": snapshot.state.value,
            "side_sign": snapshot.side_sign,
            "trend_score": round(snapshot.trend_score, 3),
            "depth_atr": round(snapshot.countermove_depth_atr, 3),
            "stale": snapshot.stale,
            "legacy_side": str(side or ""),
            "legacy_paused": legacy_paused,
            "engine_paused": engine_paused,
            "agree": engine_paused == legacy_paused,
        }
        fingerprint = f"{row['state']}|{row['legacy_paused']}|{row['engine_paused']}"
        with _lock:
            if _last_written.get("fingerprint") == fingerprint:
                return row  # nothing new to persist
            _last_written["fingerprint"] = fingerprint
        path = shadow_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        return row
    except Exception:
        logging.debug("SPY shadow-state recording failed (live behavior unaffected).", exc_info=True)
        return None


def reset_shadow_dedupe() -> None:
    """Test hook: forget the last written fingerprint."""
    with _lock:
        _last_written.clear()
