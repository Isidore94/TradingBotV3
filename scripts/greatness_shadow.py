"""Greatness Monitor shadow runner (plan.md sections 7.3 and 9, shadow-first).

Runs the confirmation engine beside the existing one-wick D1 trigger alerts:
each time the bot evaluates a D1 watchlist symbol, the same intraday bars
feed a persistent DevelopmentCandidate. Stages/attempts survive restarts via
an atomic JSON store; every typed transition appends to a JSONL log. Live
alerts are untouched - this accumulates the evidence for replacing the
single-cross rule with staged confirmation.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path

from greatness_monitor import (
    DevelopmentCandidate,
    GreatnessEngine,
    candidate_from_d1_trigger_levels,
)
from market_state import M5Bar

_BAR_MINUTES = 5
_lock = threading.Lock()
_board: "GreatnessBoard | None" = None


def _diag_dir() -> Path:
    try:
        from project_paths import CACHE_DIR

        return Path(CACHE_DIR).parent / "diagnostics"
    except Exception:
        return Path.home() / ".tradingbotv3"


class GreatnessBoard:
    """Manages persistent candidates and feeds them completed bars once."""

    def __init__(self, store_path: Path | None = None, events_path: Path | None = None) -> None:
        self.store_path = store_path or (_diag_dir() / "greatness_candidates.json")
        self.events_path = events_path or (_diag_dir() / "greatness_shadow.jsonl")
        self.engine = GreatnessEngine()
        self.candidates: dict[str, DevelopmentCandidate] = {}
        self.last_bar_ts: dict[str, str] = {}
        self._load()

    # ------------------------------------------------------------------
    def update(self, symbol: str, side: str, trigger_levels: list[dict], bars: list[M5Bar], *, session_date: str) -> list:
        key = f"{symbol.upper()}|{session_date}"
        candidate = self.candidates.get(key)
        if candidate is None:
            candidate = candidate_from_d1_trigger_levels(
                symbol, side, trigger_levels, session_date=session_date
            )
            if candidate is None:
                return []
            self.candidates[key] = candidate
        last_seen = self.last_bar_ts.get(key, "")
        events = []
        for bar in bars:
            stamp = bar.ts.isoformat(timespec="seconds")
            if stamp <= last_seen or not bar.complete:
                continue
            events.extend(self.engine.on_bar(candidate, bar))
            last_seen = stamp
        self.last_bar_ts[key] = last_seen
        if events:
            self._append_events(events)
        self._save()
        return events

    # ------------------------------------------------------------------
    def _append_events(self, events) -> None:
        try:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as handle:
                for e in events:
                    handle.write(
                        json.dumps(
                            {
                                "ts": e.ts.isoformat(timespec="seconds"),
                                "symbol": e.symbol,
                                "event": e.event.value,
                                "step": e.step_label,
                                "price": e.price,
                                "attempt": e.attempt,
                                "stage": e.stage.value,
                            }
                        )
                        + "\n"
                    )
        except OSError:
            logging.debug("Greatness shadow event append failed.", exc_info=True)

    def _save(self) -> None:
        try:
            payload = {
                "candidates": {k: c.to_dict() for k, c in self.candidates.items()},
                "last_bar_ts": self.last_bar_ts,
            }
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(self.store_path.parent), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(tmp, self.store_path)
        except OSError:
            logging.debug("Greatness store save failed.", exc_info=True)

    def _load(self) -> None:
        try:
            if not self.store_path.exists():
                return
            payload = json.loads(self.store_path.read_text(encoding="utf-8"))
            self.candidates = {
                k: DevelopmentCandidate.from_dict(v)
                for k, v in (payload.get("candidates") or {}).items()
            }
            self.last_bar_ts = dict(payload.get("last_bar_ts") or {})
        except (OSError, json.JSONDecodeError, ValueError, KeyError):
            logging.debug("Greatness store load failed; starting fresh.", exc_info=True)


def shadow_board() -> GreatnessBoard:
    global _board
    with _lock:
        if _board is None:
            _board = GreatnessBoard()
        return _board


def _bars_from_frame(today_df, *, now: datetime | None = None) -> list[M5Bar]:
    moment = now or datetime.now()
    bars: list[M5Bar] = []
    rows = today_df.dropna(subset=["datetime"]).sort_values("datetime")
    total = len(rows)
    for index, (_, row) in enumerate(rows.iterrows()):
        start = row["datetime"].to_pydatetime()
        complete = True
        if index == total - 1:
            try:
                complete = (moment - start) >= timedelta(minutes=_BAR_MINUTES)
            except TypeError:
                complete = True
        try:
            bars.append(
                M5Bar(
                    ts=start + timedelta(minutes=_BAR_MINUTES),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0.0) or 0.0),
                    complete=complete,
                )
            )
        except (TypeError, ValueError, KeyError):
            continue
    return bars


def record_d1_shadow(bot, symbol: str, today_df, *, now: datetime | None = None) -> list:
    """Fail-safe hook called from the live D1 trigger path. Never raises."""
    try:
        symbol = str(symbol or "").strip().upper()
        watch_entry = (
            getattr(bot, "master_avwap_d1_upgrade_alerts", {}).get(symbol)
            or getattr(bot, "master_avwap_d1_watchlist", {}).get(symbol)
            or {}
        )
        trigger_levels = watch_entry.get("trigger_levels") or []
        if not trigger_levels or today_df is None or getattr(today_df, "empty", True):
            return []
        working = today_df
        if "datetime" not in working.columns:
            return []
        side = str(watch_entry.get("side") or "LONG")
        session_date = (now or datetime.now()).date().isoformat()
        return shadow_board().update(
            symbol,
            side,
            trigger_levels,
            _bars_from_frame(working, now=now),
            session_date=session_date,
        )
    except Exception:
        logging.debug("Greatness D1 shadow failed (live alerts unaffected).", exc_info=True)
        return []
