from __future__ import annotations

import json
import logging
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from project_paths import (
    AUTOPILOT_LOG_FILE,
    AUTOPILOT_REPORT_FILE,
    AUTOPILOT_STATE_FILE,
    LONGS_FILE,
    SHORTS_FILE,
)
from watchlist_utils import read_watchlist_symbols

import autopilot_core as core
from ui.services.scan_service import ScanService


_TICK_INTERVAL_MS = 30_000
_REPORT_HEARTBEAT_MINUTES = 10
_MAX_LOG_LINES = 400
_MAX_REPORT_ALERTS = 15
_MAX_REPORT_LOG_LINES = 30


class AutopilotService(QObject):
    """Unattended mini-PC mode: schedules swing scans, self-builds the
    BounceBot watchlists at the open, folds near-HOD names in on regime
    pauses, and keeps the shared-Drive away report fresh. All heavy work runs
    off the GUI thread; this object only orchestrates."""

    logMessage = Signal(str)
    enabledChanged = Signal(bool)
    statusChanged = Signal(dict)

    def __init__(self, bounce_service, parent=None) -> None:
        super().__init__(parent)
        self._bounce_service = bounce_service
        self._scan_service = ScanService(self)
        self._scan_service.finished.connect(self._on_scan_finished)
        self._scan_service.failed.connect(self._on_scan_failed)

        self._log_lines: deque[str] = deque(maxlen=_MAX_LOG_LINES)
        self._alerts_today: deque[str] = deque(maxlen=60)
        self._alerts_date = datetime.now().date().isoformat()
        self._state = self._load_state()
        self._enabled = bool(self._state.get("enabled"))
        self._active_scan_slot: str | None = None
        self._building_watchlists = False
        self._hod_check_running = False
        self._reconnect_running = False
        self._last_report_write: datetime | None = None
        self._last_ib_status: str | None = None
        self._weekend_logged_date: str | None = None

        if bounce_service is not None:
            bounce_service.alertReceived.connect(self._on_alert)
            bounce_service.connectionChanged.connect(self._on_connection_changed)

        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        if self._enabled:
            self._log("Auto Pilot resuming from saved state (was ON at last shutdown).")
            self._ensure_bot_running()

    # ------------------------------------------------------------------
    # Public control surface
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._enabled:
            return
        self._enabled = enabled
        self._state["enabled"] = enabled
        self._save_state()
        if enabled:
            self._log("AUTO PILOT ON - scheduling swing scans, self-building watchlists, writing the away report.")
            self._ensure_bot_running()
            self._tick()
        else:
            self._log("AUTO PILOT OFF - automation paused (BounceBot keeps running; stop it from the desk if needed).")
        self.enabledChanged.emit(enabled)
        self._write_report()

    def force_reconnect(self) -> None:
        if self._reconnect_running:
            self._log("Reconnect already in progress.")
            return
        bot = self._current_bot()
        if bot is None:
            self._log("No BounceBot instance yet - starting it now.")
            self._ensure_bot_running()
            return
        self._reconnect_running = True
        self._log("Manual IB reconnect requested...")

        def worker() -> None:
            try:
                ok = bool(bot.ensure_connected(timeout=20))
                self._log("IB reconnected." if ok else "IB reconnect failed - will keep retrying automatically.")
            except Exception as exc:
                self._log(f"IB reconnect error: {exc}")
            finally:
                self._reconnect_running = False

        threading.Thread(target=worker, name="autopilot-reconnect", daemon=True).start()

    def run_swing_scan_now(self) -> None:
        now = datetime.now()
        slots = core.get_autopilot_swing_slots(now)
        slot = now.strftime("%H:%M")
        update = core.slot_writes_setup_tracker(slot, reference=now) if slots else False
        self._start_swing_scan(slot_label=f"manual {slot}", update_setup_tracker=update, mark_slots=[])

    def rebuild_watchlists_now(self) -> None:
        self._start_watchlist_build(manual=True)

    def write_report_now(self) -> None:
        self._write_report()
        self._log(f"Away report written to {AUTOPILOT_REPORT_FILE}")

    def status_snapshot(self) -> dict[str, Any]:
        now = datetime.now()
        slots = core.get_autopilot_swing_slots(now)
        done = set(self._state.get("slots_done", []))
        next_slot = next((slot for slot in slots if slot not in done), None)
        longs, shorts = self._read_watchlists()
        return {
            "enabled": self._enabled,
            "ib_status": self._ib_status_text(),
            "regime": self._regime_text(),
            "slots": slots,
            "slots_done": sorted(done),
            "next_slot": next_slot,
            "watchlist_built_at": self._state.get("watchlist_built_at") or "",
            "longs_count": len(longs),
            "shorts_count": len(shorts),
            "scan_running": self._scan_service.running,
            "report_path": str(AUTOPILOT_REPORT_FILE),
        }

    def shutdown(self) -> None:
        self._timer.stop()
        self._save_state()
        self._scan_service.shutdown()

    # ------------------------------------------------------------------
    # Tick loop
    # ------------------------------------------------------------------
    @Slot()
    def _tick(self) -> None:
        try:
            self._roll_day_state()
            if not self._enabled:
                return
            now = datetime.now()
            if now.weekday() >= 5:
                today = now.date().isoformat()
                if self._weekend_logged_date != today:
                    self._weekend_logged_date = today
                    self._log("Weekend - Auto Pilot idle until the next session.")
                return
            self._ensure_bot_running()
            self._maybe_build_watchlists(now)
            self._maybe_run_swing_slot(now)
            self._maybe_add_near_extreme_names(now)
            self._maybe_heartbeat_report(now)
            self.statusChanged.emit(self.status_snapshot())
        except Exception:
            logging.exception("Auto Pilot tick failed")

    def _roll_day_state(self) -> None:
        today = datetime.now().date().isoformat()
        if self._state.get("date") != today:
            self._state = {
                "date": today,
                "enabled": self._enabled,
                "slots_done": [],
                "watchlist_built_at": None,
                "hod_last_check": None,
                "hod_added": [],
            }
            self._save_state()
        if self._alerts_date != today:
            self._alerts_date = today
            self._alerts_today.clear()

    def _ensure_bot_running(self) -> None:
        service = self._bounce_service
        if service is None:
            return
        if not service.running:
            self._log("Starting BounceBot (IB connect + intraday scanning).")
            service.start()
        if not service.scanning_enabled:
            service.set_scanning_enabled(True)

    # ------------------------------------------------------------------
    # Watchlist self-build (open scan)
    # ------------------------------------------------------------------
    def _maybe_build_watchlists(self, now: datetime) -> None:
        if self._building_watchlists or self._state.get("watchlist_built_at"):
            return
        since_open = core.minutes_since_open(now)
        if since_open < core.AUTOPILOT_WATCHLIST_BUILD_AFTER_OPEN_MINUTES:
            return
        if since_open > core.AUTOPILOT_WATCHLIST_BUILD_DEADLINE_MINUTES:
            return
        self._start_watchlist_build(manual=False)

    def _start_watchlist_build(self, *, manual: bool) -> None:
        if self._building_watchlists:
            self._log("Watchlist build already running.")
            return
        self._building_watchlists = True
        origin = "manual" if manual else "scheduled"
        self._log(f"Building today's longs.txt / shorts.txt from the open scan ({origin}, yfinance batch)...")

        def worker() -> None:
            try:
                pool = core.load_universe_pool()
                if not pool:
                    self._log("Universe files are empty/missing - keeping the existing watchlists. Run the Universe builder.")
                    return
                moves = core.fetch_open_scan_moves(pool, log=self._log)
                if not moves:
                    self._log("Open scan returned no data - keeping the existing watchlists.")
                    return
                spy_move = moves.get("SPY")
                built = core.build_watchlists_from_moves(moves, spy_move)
                longs = built["longs"]
                shorts = built["shorts"]
                if not longs and not shorts:
                    self._log(f"Open scan found no gap/RS movers across {built['scanned']} names - watchlists unchanged.")
                    return
                core.write_bouncebot_watchlists(longs, shorts)
                self._state["watchlist_built_at"] = datetime.now().strftime("%H:%M:%S")
                self._save_state()
                spy_text = ""
                if spy_move and spy_move.get("early_move_pct") is not None:
                    spy_text = f" (SPY early move {float(spy_move['early_move_pct']):+.2f}%)"
                self._log(
                    f"Watchlists built from {built['scanned']} names{spy_text}: "
                    f"{len(longs)} longs [{', '.join(longs[:10])}{'...' if len(longs) > 10 else ''}], "
                    f"{len(shorts)} shorts [{', '.join(shorts[:10])}{'...' if len(shorts) > 10 else ''}]."
                )
                self._write_report()
            except Exception as exc:
                self._log(f"Watchlist build failed: {exc}")
                logging.exception("Auto Pilot watchlist build failed")
            finally:
                self._building_watchlists = False

        threading.Thread(target=worker, name="autopilot-watchlists", daemon=True).start()

    # ------------------------------------------------------------------
    # Swing scan schedule
    # ------------------------------------------------------------------
    def _maybe_run_swing_slot(self, now: datetime) -> None:
        if self._scan_service.running:
            return
        slots = core.get_autopilot_swing_slots(now)
        done = set(self._state.get("slots_done", []))
        due = [
            slot
            for slot in slots
            if slot not in done and datetime.combine(now.date(), _parse_slot(slot)) <= now
        ]
        if not due:
            return
        slot = due[-1]
        if len(due) > 1:
            self._log(f"Catching up: {len(due)} swing slots due; running {slot} and marking {', '.join(due[:-1])} skipped.")
        update = core.slot_writes_setup_tracker(slot, reference=now)
        self._start_swing_scan(slot_label=slot, update_setup_tracker=update, mark_slots=due)

    def _start_swing_scan(self, *, slot_label: str, update_setup_tracker: bool, mark_slots: list[str]) -> None:
        if self._scan_service.running:
            self._log("A swing scan is already running.")
            return
        self._active_scan_slot = slot_label
        self._pending_slot_marks = list(mark_slots)
        tracker_text = "WITH setup-tracker write" if update_setup_tracker else "no tracker write"
        started = self._scan_service.run_autopilot_scan(
            update_setup_tracker=update_setup_tracker,
            label=f"Auto Pilot swing scan ({slot_label}, {tracker_text})",
        )
        if started:
            self._log(f"Swing scan started for slot {slot_label} ({tracker_text}).")
        else:
            self._active_scan_slot = None
            self._pending_slot_marks = []

    @Slot(dict, list, str)
    def _on_scan_finished(self, run_result: dict, rows: list, stamp: str) -> None:
        slot = self._active_scan_slot or "?"
        self._mark_slots_done()
        self._log(f"Swing scan for slot {slot} finished at {stamp} ({len(rows)} setup rows).")
        self._active_scan_slot = None
        self._write_report()

    @Slot(str)
    def _on_scan_failed(self, message: str) -> None:
        slot = self._active_scan_slot or "?"
        self._mark_slots_done()  # do not retry-loop a broken slot all hour
        first_line = str(message or "").strip().splitlines()[0] if str(message or "").strip() else "unknown error"
        self._log(f"Swing scan for slot {slot} FAILED: {first_line}")
        self._active_scan_slot = None
        self._write_report()

    def _mark_slots_done(self) -> None:
        marks = getattr(self, "_pending_slot_marks", [])
        if not marks:
            return
        done = set(self._state.get("slots_done", []))
        done.update(marks)
        self._state["slots_done"] = sorted(done)
        self._pending_slot_marks = []
        self._save_state()

    # ------------------------------------------------------------------
    # Near-HOD/LOD adds on regime pauses
    # ------------------------------------------------------------------
    def _maybe_add_near_extreme_names(self, now: datetime) -> None:
        if self._hod_check_running:
            return
        last_check = self._state.get("hod_last_check")
        if last_check:
            try:
                last_dt = datetime.strptime(f"{self._state.get('date')} {last_check}", "%Y-%m-%d %H:%M:%S")
                if (now - last_dt).total_seconds() < core.AUTOPILOT_HOD_CHECK_COOLDOWN_MINUTES * 60:
                    return
            except ValueError:
                pass
        bot = self._current_bot()
        if bot is None:
            return
        try:
            regime = str(bot.get_market_environment() or "")
            spy_today, _prev = bot._spy_session_bars()
        except Exception:
            return
        if len(spy_today) < 6:
            return
        last_bar = spy_today[-1]
        side = None
        if regime.startswith("bullish") and last_bar.close < last_bar.open:
            side = "long"
        elif regime.startswith("bearish") and last_bar.close > last_bar.open:
            side = "short"
        if side is None:
            return

        self._hod_check_running = True
        self._state["hod_last_check"] = now.strftime("%H:%M:%S")
        self._save_state()
        extreme = "HOD" if side == "long" else "LOD"
        self._log(f"{regime} tape pausing - checking swing-scanner {side}s near their {extreme}...")

        def worker() -> None:
            try:
                symbols = self._top_swing_symbols(side)
                if not symbols:
                    self._log(f"No swing-scanner {side} rows available for the {extreme} check.")
                    return
                snapshot = core.fetch_day_snapshot(symbols, log=self._log)
                matches = core.near_extreme_candidates(snapshot, side)
                target = Path(LONGS_FILE) if side == "long" else Path(SHORTS_FILE)
                added = core.append_watchlist_symbols(target, matches)
                if added:
                    already = sorted(set(self._state.get("hod_added", [])) | set(added))
                    self._state["hod_added"] = already
                    self._save_state()
                    self._log(f"Added near-{extreme} names to {target.name}: {', '.join(added)}.")
                    self._write_report()
                else:
                    self._log(f"No new names within {core.AUTOPILOT_HOD_PROXIMITY_PCT:.1f}% of their {extreme}.")
            except Exception as exc:
                self._log(f"Near-{extreme} check failed: {exc}")
                logging.exception("Auto Pilot near-extreme check failed")
            finally:
                self._hod_check_running = False

        threading.Thread(target=worker, name="autopilot-hod", daemon=True).start()

    def _top_swing_symbols(self, side: str) -> list[str]:
        rows = self._load_swing_rows()
        wanted = "LONG" if side == "long" else "SHORT"
        scored = [
            row
            for row in rows
            if str(getattr(row, "side", "")).strip().upper() == wanted
        ]
        scored.sort(
            key=lambda row: (
                getattr(row, "expected_r", None) is None,
                -(getattr(row, "expected_r", None) or 0.0),
                -(getattr(row, "score", None) or 0.0),
            )
        )
        current_longs, current_shorts = self._read_watchlists()
        existing = set(current_longs if side == "long" else current_shorts)
        symbols = []
        for row in scored:
            symbol = str(getattr(row, "symbol", "")).strip().upper()
            if symbol and symbol not in existing:
                symbols.append(symbol)
            if len(symbols) >= core.AUTOPILOT_HOD_TOP_ROWS:
                break
        return symbols

    @staticmethod
    def _load_swing_rows() -> list:
        try:
            from ui.services.data_feed import load_latest_setup_rows

            return load_latest_setup_rows()
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Away report
    # ------------------------------------------------------------------
    def _maybe_heartbeat_report(self, now: datetime) -> None:
        if self._last_report_write is None or (
            (now - self._last_report_write).total_seconds() >= _REPORT_HEARTBEAT_MINUTES * 60
        ):
            self._write_report()

    def _write_report(self) -> None:
        try:
            longs, shorts = self._read_watchlists()
            snapshot = self.status_snapshot()
            picks = []
            for row in self._load_swing_rows()[:60]:
                expected = getattr(row, "expected_r", None)
                picks.append(
                    {
                        "symbol": getattr(row, "symbol", ""),
                        "side": getattr(row, "side", ""),
                        "bucket": getattr(row, "bucket_label", "") or getattr(row, "bucket", ""),
                        "expected_r": expected,
                    }
                )
            picks = [pick for pick in picks if pick["symbol"]][:10]
            payload = {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "enabled": self._enabled,
                "ib_status": snapshot["ib_status"],
                "regime": snapshot["regime"],
                "longs": longs,
                "shorts": shorts,
                "swing_picks": picks,
                "alerts": list(self._alerts_today)[-_MAX_REPORT_ALERTS:][::-1],
                "slots_done": snapshot["slots_done"],
                "next_slot": snapshot["next_slot"],
                "log_lines": list(self._log_lines)[-_MAX_REPORT_LOG_LINES:][::-1],
            }
            core.write_away_report(payload)
            self._last_report_write = datetime.now()
        except Exception:
            logging.exception("Auto Pilot report write failed")

    # ------------------------------------------------------------------
    # Bot plumbing
    # ------------------------------------------------------------------
    def _current_bot(self):
        service = self._bounce_service
        if service is None:
            return None
        try:
            return service.current_bot()
        except Exception:
            return None

    def _ib_status_text(self) -> str:
        bot = self._current_bot()
        if bot is None:
            return "bot not running"
        connected = bool(getattr(bot, "connection_status", False))
        if not connected:
            return "DISCONNECTED - waiting to reconnect"
        pacing = 0.0
        try:
            pacing = float(bot.pacing_delay_remaining())
        except Exception:
            pacing = 0.0
        if pacing > 0:
            return f"connected (pacing backoff {pacing:.0f}s)"
        return "connected"

    def _regime_text(self) -> str:
        bot = self._current_bot()
        if bot is None:
            return "unknown"
        try:
            return str(bot.get_market_environment())
        except Exception:
            return "unknown"

    def _read_watchlists(self) -> tuple[list[str], list[str]]:
        try:
            longs = list(read_watchlist_symbols(Path(LONGS_FILE)))
        except Exception:
            longs = []
        try:
            shorts = list(read_watchlist_symbols(Path(SHORTS_FILE)))
        except Exception:
            shorts = []
        return longs, shorts

    @Slot(object)
    def _on_alert(self, alert) -> None:
        text = str(getattr(alert, "raw_text", "") or "").strip()
        if not text or "candle has closed" in text.lower():
            return
        stamp = getattr(alert, "time_text", "") or datetime.now().strftime("%H:%M:%S")
        self._alerts_today.append(f"{stamp} {text}")

    @Slot(str)
    def _on_connection_changed(self, message: str) -> None:
        message = str(message or "")
        if message == self._last_ib_status:
            return
        previous = self._last_ib_status
        self._last_ib_status = message
        if not self._enabled:
            return
        if "disconnected" in message.lower() or "retrying" in message.lower():
            self._log(f"{message} - Auto Pilot will wait and auto-reconnect (log back in via Moonlight or hit Reconnect).")
        elif previous is not None:
            self._log(message)

    # ------------------------------------------------------------------
    # Logging & state
    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        self._log_lines.append(line)
        logging.info("AutoPilot: %s", message)
        try:
            AUTOPILOT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with AUTOPILOT_LOG_FILE.open("a", encoding="utf-8") as handle:
                handle.write(f"{datetime.now():%Y-%m-%d} {line}\n")
        except Exception:
            pass
        self.logMessage.emit(line)

    def log_lines(self) -> list[str]:
        return list(self._log_lines)

    def _load_state(self) -> dict[str, Any]:
        today = datetime.now().date().isoformat()
        try:
            payload = json.loads(Path(AUTOPILOT_STATE_FILE).read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if not isinstance(payload, dict) or payload.get("date") != today:
            return {
                "date": today,
                "enabled": bool(payload.get("enabled")) if isinstance(payload, dict) else False,
                "slots_done": [],
                "watchlist_built_at": None,
                "hod_last_check": None,
                "hod_added": [],
            }
        payload.setdefault("slots_done", [])
        payload.setdefault("hod_added", [])
        return payload

    def _save_state(self) -> None:
        try:
            path = Path(AUTOPILOT_STATE_FILE)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        except Exception:
            logging.exception("Auto Pilot state save failed")


def _parse_slot(slot: str):
    from datetime import time as dt_time

    hours, minutes = str(slot).strip().split(":", 1)
    return dt_time(int(hours), int(minutes))
