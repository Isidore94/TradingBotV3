from __future__ import annotations

import json
import logging
import os
import socket
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from project_paths import (
    AUTO_LONGS_FILE,
    AUTO_SHORTS_FILE,
    AUTOPILOT_LOG_FILE,
    AUTOPILOT_PICKS_FILE,
    AUTOPILOT_REPORT_FILE,
    AUTOPILOT_SCORECARD_FILE,
    AUTOPILOT_STATE_FILE,
    INDUSTRY_BOARD_STATE_FILE,
    INDUSTRY_INTRADAY_RS_STATE_FILE,
    INTRADAY_BOUNCE_CANDIDATES_FILE,
    INTRADAY_BOUNCE_OUTCOMES_FILE,
    LONGS_FILE,
    SHORTS_FILE,
)
from market_session import is_within_regular_market_session
from watchlist_utils import read_watchlist_symbols

import autopilot_core as core
from ui.services.scan_service import ScanService, active_scan_label


_TICK_INTERVAL_MS = 30_000
_HOURLY_REPORT_RETRY_MINUTES = 5
_MAX_LOG_LINES = 400
_MAX_REPORT_ALERTS = 15
_MAX_REPORT_LOG_LINES = 30


# Truthful Auto Mode semantics (plan.md sec 14.3 / Packet A):
# OFF     - no automatic user-facing list mutations, scans, or alerts.
#           Optional shadow research (suggestion scans that write only the
#           bot-owned autolongs/autoshorts lists) continues ONLY while the
#           "collect research while Auto is off" setting is enabled.
# DESK    - full automation; desktop notifications are the primary surface.
# AWAY    - identical trading decisions; only report cadence/notification
#           presentation may differ. Never different strategy logic.
AUTO_MODE_OFF = "OFF"
AUTO_PROFILE_DESK = "DESK"
AUTO_PROFILE_AWAY = "AWAY"
SHADOW_RESEARCH_SETTING = "autopilot_shadow_research"


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
        try:
            from job_ledger import get_default_ledger

            self._job_ledger = get_default_ledger()
            for stale in self._job_ledger.mark_stale_running():
                logging.warning("Job did not survive restart: %s", stale.key)
        except Exception:
            logging.exception("Job ledger unavailable; scheduling falls back to state file only.")
            self._job_ledger = None
        self._enabled = bool(self._state.get("enabled"))
        self._profile = str(self._state.get("profile") or AUTO_PROFILE_DESK)
        if self._profile not in (AUTO_PROFILE_DESK, AUTO_PROFILE_AWAY):
            self._profile = AUTO_PROFILE_DESK
        self._active_scan_slot: str | None = None
        self._waiting_scan_slot: str | None = None
        self._building_watchlists = False
        self._hod_check_running = False
        self._reconnect_running = False
        self._universe_rebuild_running = False
        self._universe_last_attempt: datetime | None = None
        self._wrapup_running = False
        self._scorecard_line = ""
        self._last_report_write: datetime | None = None
        self._last_report_attempt: datetime | None = None
        self._last_report_error = ""
        self._last_hourly_report_attempt_slot = ""
        self._last_hourly_report_attempt_at: datetime | None = None
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
            # Sporadic use: never trust that yesterday's after-close routine ran.
            self._ensure_universe_fresh("activation")
            self._tick()
        else:
            # A manual OFF blocks the daily auto-arm for the rest of the day -
            # the trader's hand always wins over the 07:00 self-arm.
            self._state["auto_armed_date"] = datetime.now().date().isoformat()
            self._save_state()
            self._log("AUTO PILOT OFF - automation paused for today (BounceBot keeps running; stop it from the desk if needed).")
        self.enabledChanged.emit(enabled)
        self._write_report()

    @property
    def auto_mode(self) -> str:
        """OFF, or the active profile (DESK/AWAY) while enabled."""
        return self._profile if self._enabled else AUTO_MODE_OFF

    @property
    def profile(self) -> str:
        return self._profile

    def set_profile(self, profile: str) -> None:
        """Desk/Away are presentation profiles - never strategy changes."""
        profile = str(profile or "").strip().upper()
        if profile not in (AUTO_PROFILE_DESK, AUTO_PROFILE_AWAY) or profile == self._profile:
            return
        self._profile = profile
        self._state["profile"] = profile
        self._save_state()
        self._log(f"Auto profile -> {profile} (same decisions; presentation/cadence only).")
        self._write_report()

    def _shadow_research_allowed(self) -> bool:
        """OFF-mode suggestion scans may run only with explicit consent."""
        try:
            from project_paths import get_local_setting

            return bool(get_local_setting(SHADOW_RESEARCH_SETTING, True))
        except Exception:
            return True

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
        publish = self._write_report()
        if publish.get("ok"):
            self._log(f"Away report verified at {AUTOPILOT_REPORT_FILE}")
        else:
            self._log(f"Away report NOT updated: {publish.get('error') or 'unknown failure'}")

    def status_snapshot(self) -> dict[str, Any]:
        now = datetime.now()
        slots = core.get_autopilot_swing_slots(now)
        done = set(self._state.get("slots_done", []))
        if now.weekday() >= 5:
            # Weekend: never advertise a weekday slot as the "next update" -
            # a Saturday report claiming 07:30 reads as broken automation.
            next_slot = "next session"
        else:
            in_flight = {
                str(slot)
                for slot in (self._active_scan_slot, self._waiting_scan_slot)
                if slot and str(slot) in slots
            }
            next_slot = next(
                (slot for slot in slots if slot not in done and slot not in in_flight),
                None,
            )
        longs, shorts = self._read_watchlists()
        return {
            "enabled": self._enabled,
            "auto_mode": self.auto_mode,
            "ib_status": self._ib_status_text(),
            "regime": self._regime_text(),
            "slots": slots,
            "slots_done": sorted(done),
            "next_slot": next_slot,
            "watchlist_built_at": self._state.get("watchlist_built_at") or "",
            "longs_count": len(longs),
            "shorts_count": len(shorts),
            "auto_longs_count": len(self._read_auto_watchlist(AUTO_LONGS_FILE)),
            "auto_shorts_count": len(self._read_auto_watchlist(AUTO_SHORTS_FILE)),
            "scan_running": self._scan_service.running,
            "report_path": str(AUTOPILOT_REPORT_FILE),
            "report_last_attempt": (
                getattr(self, "_last_report_attempt", None).isoformat(timespec="seconds")
                if getattr(self, "_last_report_attempt", None)
                else ""
            ),
            "report_last_verified": (
                getattr(self, "_last_report_write", None).isoformat(timespec="seconds")
                if getattr(self, "_last_report_write", None)
                else ""
            ),
            "report_error": getattr(self, "_last_report_error", ""),
            "universe_line": self._universe_line(now),
            "industry_line": self._industry_line(),
            "universe_rebuilding": self._universe_rebuild_running,
            "wrapup_done_at": self._state.get("wrapup_done_at") or "",
            "wrapup_running": self._wrapup_running,
        }

    def _universe_line(self, now: datetime | None = None) -> str:
        now = now or datetime.now()
        built_at = core.universe_built_at()
        if self._universe_rebuild_running:
            return "Universe: rebuilding now..."
        if built_at is None:
            return "Universe: MISSING - run the Universe builder."
        state = "stale" if core.universe_is_stale(now, built_at) else "fresh"
        return f"Universe: {state} (built {built_at:%Y-%m-%d %H:%M})"

    @staticmethod
    def _industry_line() -> str:
        def read_payload(path: Path) -> dict:
            try:
                value = json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
            return value if isinstance(value, dict) else {}

        return core.format_industry_snapshot_line(
            read_payload(INDUSTRY_BOARD_STATE_FILE),
            read_payload(INDUSTRY_INTRADAY_RS_STATE_FILE),
        )

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
            now = datetime.now()
            if now.weekday() >= 5:
                today = now.date().isoformat()
                if self._enabled and self._weekend_logged_date != today:
                    self._weekend_logged_date = today
                    self._log("Weekend - Auto Pilot idle until the next session.")
                return

            # Hands-off default: Auto Pilot arms itself once per weekday at
            # 07:00 (or immediately when the GUI launches later than that).
            # One arm per day, so switching it OFF by hand sticks all day.
            self._maybe_auto_arm(now)

            # Always-on duties while the GUI is open, Auto Pilot ON or OFF:
            # near-HOD pause alerts and the daily pick scorecard measure the
            # trader's normal days too (alerts only - no file writes when OFF).
            self._maybe_clear_stale_auto_lists(now)
            self._maybe_add_near_extreme_names(now)
            self._maybe_score_picks_daily(now)
            if not self._enabled:
                self._maybe_suggest_watchlists(now)
                return

            self._ensure_bot_running()
            self._ensure_universe_fresh("tick")
            self._maybe_build_watchlists(now)
            self._maybe_run_swing_slot(now)
            self._maybe_run_wrapup(now)
            self._maybe_hourly_away_report(now)
            core.write_heartbeat(
                current_job=self._active_scan_slot or active_scan_label(),
                next_job=str(self.status_snapshot().get("next_slot") or ""),
                last_success=self._last_report_write.isoformat(timespec="seconds") if self._last_report_write else "",
            )
            self.statusChanged.emit(self.status_snapshot())
        except Exception:
            logging.exception("Auto Pilot tick failed")

    def _roll_day_state(self) -> None:
        today = datetime.now().date().isoformat()
        if self._state.get("date") != today:
            self._state = {
                "date": today,
                "enabled": self._enabled,
                "profile": self._profile,
                "slots_done": [],
                "hourly_report_slot": None,
                "watchlist_built_at": None,
                "suggested_at": None,
                "hod_last_check": None,
                "hod_added": [],
                "wrapup_done_at": None,
                "picks_scored_at": None,
                # What Auto Pilot itself wrote survives the day roll - it is
                # how tomorrow's build tells its own picks from the trader's.
                "autopilot_written": self._state.get("autopilot_written") or {"longs": [], "shorts": []},
            }
            self._scorecard_line = ""
            self._save_state()
        if self._alerts_date != today:
            self._alerts_date = today
            self._alerts_today.clear()

    def _maybe_auto_arm(self, now: datetime) -> None:
        from project_paths import get_local_setting

        try:
            auto_arm_enabled = bool(get_local_setting("qt_autopilot_auto_arm", True))
        except Exception:
            auto_arm_enabled = True
        if not core.autopilot_auto_arm_due(
            now,
            enabled=self._enabled,
            armed_date=self._state.get("auto_armed_date"),
            auto_arm_enabled=auto_arm_enabled,
        ):
            return
        self._state["auto_armed_date"] = now.date().isoformat()
        self._save_state()
        self._log(
            f"{core.AUTOPILOT_AUTO_ARM_HOUR:02d}:00 auto-arm: Auto Pilot ON for the day "
            "(flip it OFF to stay manual today; disable auto-arm on the Auto Pilot page)."
        )
        self.set_enabled(True)

    def _maybe_clear_stale_auto_lists(self, now: datetime) -> None:
        """Empty autolongs/autoshorts once per new session so BounceBot never
        chases yesterday's bot picks. mtime-guarded: if any machine already
        wrote them today (shared Drive), they are today's picks - keep them."""
        today = now.date()
        if getattr(self, "_auto_lists_cleared_date", None) == today:
            return
        self._auto_lists_cleared_date = today
        try:
            written_at = core.universe_built_at((Path(AUTO_LONGS_FILE), Path(AUTO_SHORTS_FILE)))
            if written_at is not None and written_at.date() == today:
                return
            core.write_auto_watchlists([], [])
            self._log("New session - cleared autolongs.txt / autoshorts.txt for today's open scan.")
        except Exception:
            logging.exception("Auto watchlist day-roll clear failed")

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
    # Universe freshness (sporadic activation self-heals a stale universe)
    # ------------------------------------------------------------------
    def _ensure_universe_fresh(self, reason: str, *, force: bool = False) -> None:
        if self._universe_rebuild_running:
            return
        if not force and not self._enabled:
            return
        now = datetime.now()
        if not force:
            if not core.universe_is_stale(now):
                return
            if (
                self._universe_last_attempt is not None
                and (now - self._universe_last_attempt).total_seconds()
                < core.AUTOPILOT_UNIVERSE_RETRY_MINUTES * 60
            ):
                return
        self._universe_rebuild_running = True
        self._universe_last_attempt = now
        built_at = core.universe_built_at()
        built_text = built_at.strftime("%Y-%m-%d %H:%M") if built_at else "never"
        self._log(f"Universe is stale (built {built_text}) - rebuilding ({reason}, yfinance only)...")

        def worker() -> None:
            try:
                outcome = core.rebuild_universe_if_stale(force=True, log=self._log)
                if outcome == "rebuilt":
                    self._write_report()
                elif outcome == "busy":
                    self._log("Universe rebuild already running elsewhere (launch self-heal?) - skipping.")
                elif outcome == "failed":
                    self._log(f"Universe rebuild failed - retrying in ~{core.AUTOPILOT_UNIVERSE_RETRY_MINUTES}m.")
            finally:
                self._universe_rebuild_running = False

        threading.Thread(target=worker, name="autopilot-universe", daemon=True).start()

    def rebuild_universe_now(self) -> None:
        if self._universe_rebuild_running:
            self._log("Universe rebuild already running.")
            return
        self._ensure_universe_fresh("manual", force=True)

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
        # The build only makes sense off a fresh pool - wait for the rebuild.
        if self._universe_rebuild_running or core.universe_is_stale(now):
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
                # Holiday / stale-feed guard: if SPY's freshest bars are not
                # from today, there is no session to scan - don't build lists
                # out of the previous session's tape.
                spy_session = (spy_move or {}).get("session_date")
                if spy_session is None or spy_session != datetime.now().date():
                    self._state["watchlist_built_at"] = "skipped (no fresh session - holiday?)"
                    self._save_state()
                    self._log("No fresh SPY session in the open-scan data (market holiday?) - watchlists unchanged.")
                    return
                built = core.build_watchlists_from_moves(moves, spy_move)
                longs = built["longs"]
                shorts = built["shorts"]
                if not longs and not shorts:
                    self._log(f"Open scan found no gap/RS movers across {built['scanned']} names - watchlists unchanged.")
                    return

                # Keep the trader's hand-added names: replace only what Auto
                # Pilot itself wrote last time.
                written = self._state.get("autopilot_written") or {}
                current_longs, current_shorts = self._read_watchlists()
                merged_longs = core.merge_autopilot_watchlist(longs, current_longs, written.get("longs", []))
                merged_shorts = core.merge_autopilot_watchlist(shorts, current_shorts, written.get("shorts", []))
                core.write_bouncebot_watchlists(merged_longs["symbols"], merged_shorts["symbols"])
                # The raw bot picks also land in autolongs/autoshorts.txt so
                # they build a separately-attributable outcome history.
                core.write_auto_watchlists(longs, shorts)
                self._state["autopilot_written"] = {"longs": list(longs), "shorts": list(shorts)}
                self._state["watchlist_built_at"] = datetime.now().strftime("%H:%M:%S")
                self._save_state()

                self._append_pick_rows(
                    [
                        {"side": "long", "symbol": symbol, "source": "open_scan", "why": built["long_reasons"].get(symbol, "")}
                        for symbol in longs
                    ]
                    + [
                        {"side": "short", "symbol": symbol, "source": "open_scan", "why": built["short_reasons"].get(symbol, "")}
                        for symbol in shorts
                    ],
                    moves,
                )

                spy_text = ""
                if spy_move and spy_move.get("early_move_pct") is not None:
                    spy_text = f" (SPY early move {float(spy_move['early_move_pct']):+.2f}%)"
                kept = merged_longs["manual_kept"] + merged_shorts["manual_kept"]
                kept_text = f"; kept your names: {', '.join(kept)}" if kept else ""
                self._log(
                    f"Watchlists built from {built['scanned']} names{spy_text}: "
                    f"{len(longs)} longs [{', '.join(longs[:10])}{'...' if len(longs) > 10 else ''}], "
                    f"{len(shorts)} shorts [{', '.join(shorts[:10])}{'...' if len(shorts) > 10 else ''}]{kept_text}."
                )
                self._write_report()
            except Exception as exc:
                self._log(f"Watchlist build failed: {exc}")
                logging.exception("Auto Pilot watchlist build failed")
            finally:
                self._building_watchlists = False

        threading.Thread(target=worker, name="autopilot-watchlists", daemon=True).start()

    def _maybe_suggest_watchlists(self, now: datetime) -> None:
        """Auto Pilot OFF: run the open scan anyway and *suggest* the picks.

        No file writes - one alert plus pick rows (source=suggestion) so the
        engine keeps accruing evidence on the trader's manual days.
        """
        if self._building_watchlists:
            return
        if not self._shadow_research_allowed():
            return  # strict OFF: no bot-owned list writes without consent
        if self._state.get("watchlist_built_at") or self._state.get("suggested_at"):
            return
        since_open = core.minutes_since_open(now)
        if since_open < core.AUTOPILOT_WATCHLIST_BUILD_AFTER_OPEN_MINUTES:
            return
        if since_open > core.AUTOPILOT_WATCHLIST_BUILD_DEADLINE_MINUTES:
            return
        if core.universe_is_stale(now):
            return  # the launch self-heal is presumably still running
        self._building_watchlists = True
        self._log("Open scan (suggestion mode - Auto Pilot OFF, watchlists untouched)...")

        def worker() -> None:
            try:
                pool = core.load_universe_pool()
                if not pool:
                    self._state["suggested_at"] = "skipped (no universe)"
                    self._save_state()
                    return
                moves = core.fetch_open_scan_moves(pool, log=self._log)
                spy_move = (moves or {}).get("SPY")
                spy_session = (spy_move or {}).get("session_date")
                if spy_session is None or spy_session != datetime.now().date():
                    self._state["suggested_at"] = "skipped (no fresh session)"
                    self._save_state()
                    return
                built = core.build_watchlists_from_moves(moves, spy_move)
                message = core.format_suggestion_message(built)
                self._state["suggested_at"] = datetime.now().strftime("%H:%M:%S")
                self._save_state()
                # The bot's picks get their own tracked watchlists even in
                # suggestion mode: BounceBot scans autolongs/autoshorts.txt
                # like the trader's lists, so this data accrues every day.
                core.write_auto_watchlists(built["longs"], built["shorts"])
                if not message:
                    self._log(f"Open scan found no gap/RS movers across {built['scanned']} names.")
                    return
                self._emit_info_alert(message, "blue")
                self._log(
                    f"{message} | written to autolongs.txt ({len(built['longs'])}) / "
                    f"autoshorts.txt ({len(built['shorts'])}) - BounceBot tracks them separately."
                )
                self._append_pick_rows(
                    [
                        {"side": "long", "symbol": symbol, "source": "suggestion", "why": built["long_reasons"].get(symbol, "")}
                        for symbol in built["longs"]
                    ]
                    + [
                        {"side": "short", "symbol": symbol, "source": "suggestion", "why": built["short_reasons"].get(symbol, "")}
                        for symbol in built["shorts"]
                    ],
                    moves,
                )
            except Exception as exc:
                self._log(f"Suggestion scan failed: {exc}")
                logging.exception("Auto Pilot suggestion scan failed")
            finally:
                self._building_watchlists = False

        threading.Thread(target=worker, name="autopilot-suggest", daemon=True).start()

    def _emit_info_alert(self, message: str, color: str = "blue") -> None:
        """Push an informational line into the normal alert stream/center."""
        service = self._bounce_service
        if service is None:
            return
        try:
            from ui.models.bounce import BounceAlert

            service.alertReceived.emit(BounceAlert.from_callback(message, color))
        except Exception:
            logging.exception("Auto Pilot info alert emit failed")

    def _append_pick_rows(self, picks: list[dict], moves: dict | None = None) -> None:
        """Evidence trail: every auto pick with its gap/RS numbers."""
        if not picks:
            return
        try:
            import csv

            AUTOPILOT_PICKS_FILE.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = ["date", "logged_at", "symbol", "side", "source", "gap_pct", "excess_pct", "why"]
            write_header = not AUTOPILOT_PICKS_FILE.exists() or AUTOPILOT_PICKS_FILE.stat().st_size == 0
            spy_early = 0.0
            if moves and moves.get("SPY", {}).get("early_move_pct") is not None:
                spy_early = float(moves["SPY"]["early_move_pct"])
            with AUTOPILOT_PICKS_FILE.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                now = datetime.now()
                for pick in picks:
                    move = (moves or {}).get(pick.get("symbol", ""), {})
                    gap = move.get("gap_pct")
                    early = move.get("early_move_pct")
                    writer.writerow(
                        {
                            "date": now.date().isoformat(),
                            "logged_at": now.strftime("%H:%M:%S"),
                            "symbol": pick.get("symbol", ""),
                            "side": pick.get("side", ""),
                            "source": pick.get("source", ""),
                            "gap_pct": f"{float(gap):.2f}" if gap is not None else "",
                            "excess_pct": f"{float(early) - spy_early:.2f}" if early is not None else "",
                            "why": pick.get("why", ""),
                        }
                    )
        except Exception:
            logging.exception("Auto Pilot pick logging failed")

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
        ledger = getattr(self, "_job_ledger", None)
        if ledger is not None:
            try:
                from job_ledger import job_key

                key = job_key(now.date().isoformat(), "swing_scan", slot, "shared-v1")
                if ledger.is_done(key):
                    done.update(due)
                    self._state["slots_done"] = sorted(done)
                    self._save_state()
                    self._log(
                        f"Swing slot {slot} already completed in the job ledger; "
                        "reconciled local scheduler state without rescanning."
                    )
                    return
            except Exception:
                logging.exception("Could not reconcile swing slot with the job ledger.")
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
            slot_label=slot_label,
        )
        if started:
            self._waiting_scan_slot = None
            self._log(f"Swing scan started for slot {slot_label} ({tracker_text}).")
        else:
            rejection = self._scan_service.last_rejection_reason
            if rejection == "scheduled slot already completed":
                self._mark_slots_done()
                self._log(f"Swing scan for slot {slot_label} was already completed; no duplicate scan launched.")
                self._active_scan_slot = None
                self._waiting_scan_slot = None
                return
            self._active_scan_slot = None
            self._pending_slot_marks = []
            holder = active_scan_label() or "another Master AVWAP scan"
            if self._waiting_scan_slot != slot_label:
                self._log(f"Swing scan for slot {slot_label} is waiting; {holder} is already running.")
                self._waiting_scan_slot = slot_label

    @Slot(dict, list, str)
    def _on_scan_finished(self, run_result: dict, rows: list, stamp: str) -> None:
        slot = self._active_scan_slot or "?"
        self._mark_slots_done()
        self._log(f"Swing scan for slot {slot} finished at {stamp} ({len(rows)} setup rows).")
        self._active_scan_slot = None
        self._waiting_scan_slot = None
        self._write_report()
        self._maybe_run_wrapup(datetime.now())

    @Slot(str)
    def _on_scan_failed(self, message: str) -> None:
        slot = self._active_scan_slot or "?"
        self._mark_slots_done()  # do not retry-loop a broken slot all hour
        detail = str(message or "").strip()
        first_line = detail.splitlines()[0] if detail else "unknown error"
        self._log(f"Swing scan for slot {slot} FAILED: {first_line}")
        # The feed keeps one line for the phone report, but the subprocess
        # stderr/traceback lives in the remaining lines - keep it findable.
        if detail and detail != first_line:
            logging.error("Auto Pilot swing scan for slot %s failed:\n%s", slot, detail)
        self._active_scan_slot = None
        self._waiting_scan_slot = None
        self._write_report()
        self._maybe_run_wrapup(datetime.now())

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
        if not self._enabled and not self._shadow_research_allowed():
            return  # strict OFF: no automatic checks or alerts at all
        # Live-session only (stale after-hours bars would fake a "pause").
        try:
            if not is_within_regular_market_session():
                return
        except Exception:
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
                if not matches:
                    self._log(f"No new names within {core.AUTOPILOT_HOD_PROXIMITY_PCT:.1f}% of their {extreme}.")
                    return
                # Always surface the find in the alert stream - at the desk
                # this is the whole feature; away, it is the audit trail.
                self._emit_info_alert(
                    f"NEAR-{extreme} PAUSE WATCH ({regime}): swing {side}s holding "
                    f"{'highs' if side == 'long' else 'lows'} while SPY pauses: {', '.join(matches)}",
                    "green" if side == "long" else "red",
                )
                if not self._enabled:
                    if not self._shadow_research_allowed():
                        self._log(
                            f"Near-{extreme} watch (Auto OFF, shadow research disabled): "
                            "surfaced as an alert only; no lists touched."
                        )
                        return
                    core.add_candidate_registry_memberships(
                        "near_extreme",
                        side,
                        matches,
                        lease_minutes=90,
                    )
                    auto_target = Path(AUTO_LONGS_FILE) if side == "long" else Path(AUTO_SHORTS_FILE)
                    auto_added = core.append_watchlist_symbols(auto_target, matches)
                    self._append_pick_rows(
                        [{"side": side, "symbol": symbol, "source": "suggestion", "why": f"near {extreme}"} for symbol in auto_added or matches]
                    )
                    self._log(
                        f"Near-{extreme} watch (Auto Pilot OFF): added to {auto_target.name}: "
                        f"{', '.join(auto_added) if auto_added else '(already tracked)'}."
                    )
                    return
                core.add_candidate_registry_memberships(
                    "near_extreme",
                    side,
                    matches,
                    lease_minutes=90,
                )
                target = Path(LONGS_FILE) if side == "long" else Path(SHORTS_FILE)
                added = core.append_watchlist_symbols(target, matches)
                if added:
                    already = sorted(set(self._state.get("hod_added", [])) | set(added))
                    self._state["hod_added"] = already
                    written = self._state.get("autopilot_written") or {"longs": [], "shorts": []}
                    side_key = "longs" if side == "long" else "shorts"
                    written[side_key] = sorted(set(written.get(side_key, [])) | set(added))
                    self._state["autopilot_written"] = written
                    self._save_state()
                    self._append_pick_rows(
                        [{"side": side, "symbol": symbol, "source": "hod_add", "why": f"near {extreme}"} for symbol in added]
                    )
                    self._log(f"Added near-{extreme} names to {target.name}: {', '.join(added)}.")
                    self._write_report()
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
    def _load_swing_feed() -> dict[str, Any]:
        try:
            from ui.services.data_feed import load_latest_setup_rows_with_meta

            return load_latest_setup_rows_with_meta()
        except Exception:
            return {"rows": [], "data_date": None, "source": "none", "is_stale": True}

    @staticmethod
    def _load_swing_rows() -> list:
        return list(AutopilotService._load_swing_feed().get("rows") or [])

    # ------------------------------------------------------------------
    # After-close wrap-up: universe rebuild + learning refresh + scorecard
    # ------------------------------------------------------------------
    def _maybe_run_wrapup(self, now: datetime) -> None:
        if self._wrapup_running or not self._enabled:
            return
        if not core.after_close_wrapup_due(
            now,
            self._state.get("slots_done", []),
            bool(self._state.get("wrapup_done_at")),
            self._scan_service.running,
        ):
            return
        self._start_wrapup()

    def _start_wrapup(self) -> None:
        self._wrapup_running = True
        self._log("After-close wrap-up: rebuilding universe, refreshing day-trade learning, scoring today's picks...")

        def worker() -> None:
            try:
                # 1) Fresh universe with today's closes -> tomorrow's open scan
                #    is instantly ready (post-close means it is stale by rule).
                self._ensure_universe_fresh("after-close")

                # 2) Day-trade learning loop: performance rows + report + the
                #    alert-time learning state the tiers/mutes read.
                try:
                    from bounce_bot_lib.learning import refresh_bounce_learning_state

                    state = refresh_bounce_learning_state()
                    segments = (state or {}).get("segments") or {}
                    segment_count = sum(len(v) for v in segments.values())
                    self._log(f"Day-trade learning refreshed ({segment_count} measured segments).")
                except Exception as exc:
                    self._log(f"Learning refresh failed: {exc}")
                    logging.exception("Auto Pilot learning refresh failed")

                # 3) Scorecard: did the self-built lists produce anything?
                #    (idempotent - the always-on tick path may have run it)
                try:
                    self._maybe_score_picks_daily(datetime.now())
                except Exception as exc:
                    self._log(f"Pick scorecard failed: {exc}")
                    logging.exception("Auto Pilot pick scorecard failed")

                self._state["wrapup_done_at"] = datetime.now().strftime("%H:%M:%S")
                self._save_state()
                self._write_report()
                self._log("After-close wrap-up complete.")
            finally:
                self._wrapup_running = False

        threading.Thread(target=worker, name="autopilot-wrapup", daemon=True).start()

    def _maybe_score_picks_daily(self, now: datetime) -> None:
        """Once per day after the close: snapshot the trader's manual watchlist
        names as picks, then score every pick group (bot / suggested / yours)
        against the day-trade candidate + outcome logs."""
        if self._state.get("picks_scored_at"):
            return
        last_close = core.last_completed_session_close(now)
        if last_close is None or last_close.date() != now.date():
            return  # today's session has not closed yet
        self._state["picks_scored_at"] = now.strftime("%H:%M:%S")
        self._save_state()
        try:
            self._snapshot_manual_picks(now)
            lines = self._score_todays_picks()
            if lines:
                self._scorecard_line = " | ".join(lines)
                for line in lines:
                    self._log(line)
        except Exception:
            logging.exception("Auto Pilot daily pick scoring failed")

    def _snapshot_manual_picks(self, now: datetime) -> None:
        """Log the trader's own watchlist names (source=manual) so the daily
        scorecard compares the bot's picks against the human's."""
        import csv

        today = now.date().isoformat()
        logged_pairs: set[tuple[str, str]] = set()
        try:
            with AUTOPILOT_PICKS_FILE.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("date") == today:
                        logged_pairs.add((str(row.get("symbol") or "").upper(), str(row.get("side") or "").lower()))
        except OSError:
            pass

        written = self._state.get("autopilot_written") or {}
        longs, shorts = self._read_watchlists()
        rows = []
        for side, symbols, written_key in (("long", longs, "longs"), ("short", shorts, "shorts")):
            auto_written = {str(item).upper() for item in written.get(written_key, [])}
            for symbol in symbols:
                symbol = str(symbol).strip().upper()
                if symbol and symbol not in auto_written and (symbol, side) not in logged_pairs:
                    rows.append({"side": side, "symbol": symbol, "source": "manual", "why": "trader watchlist"})
        if rows:
            self._append_pick_rows(rows)
            self._log(f"Snapshotted {len(rows)} of your watchlist names for the daily scorecard.")

    def _score_todays_picks(self) -> list[str]:
        import csv

        today = datetime.now().date().isoformat()
        picks: list[dict] = []
        try:
            with AUTOPILOT_PICKS_FILE.open("r", encoding="utf-8", newline="") as handle:
                picks = [row for row in csv.DictReader(handle) if row.get("date") == today]
        except OSError:
            pass
        if not picks:
            return ["Picks scorecard: nothing logged today."]

        def _rows(path: Path) -> list[dict]:
            try:
                with Path(path).open("r", encoding="utf-8", newline="") as handle:
                    return list(csv.DictReader(handle))
            except OSError:
                return []

        candidates = [row for row in _rows(INTRADAY_BOUNCE_CANDIDATES_FILE) if row.get("trade_date") == today]
        candidate_ids = {str(row.get("event_id") or "") for row in candidates}
        outcomes = [row for row in _rows(INTRADAY_BOUNCE_OUTCOMES_FILE) if str(row.get("event_id") or "") in candidate_ids]

        lines: list[str] = []
        try:
            AUTOPILOT_SCORECARD_FILE.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = [
                "date", "source_group", "picks", "longs", "shorts",
                "alerted", "alerted_symbols", "avg_close_r", "avg_mfe_r",
            ]
            write_header = not AUTOPILOT_SCORECARD_FILE.exists() or AUTOPILOT_SCORECARD_FILE.stat().st_size == 0
            with AUTOPILOT_SCORECARD_FILE.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                for group, group_picks in sorted(core.group_picks_by_source(picks).items()):
                    scorecard = core.score_autopilot_picks(group_picks, candidates, outcomes)
                    label = core.PICK_GROUP_LABELS.get(group, group)
                    lines.append(core.format_scorecard_line(scorecard, label=label))
                    writer.writerow(
                        {
                            "date": today,
                            "source_group": group,
                            "picks": scorecard["picks"],
                            "longs": scorecard["longs"],
                            "shorts": scorecard["shorts"],
                            "alerted": scorecard["alerted"],
                            "alerted_symbols": ";".join(scorecard["alerted_symbols"]),
                            "avg_close_r": f"{scorecard['avg_close_r']:.3f}" if scorecard["avg_close_r"] is not None else "",
                            "avg_mfe_r": f"{scorecard['avg_mfe_r']:.3f}" if scorecard["avg_mfe_r"] is not None else "",
                        }
                    )
        except Exception:
            logging.exception("Auto Pilot scorecard write failed")
        return lines

    # ------------------------------------------------------------------
    # Away report
    # ------------------------------------------------------------------
    def _maybe_hourly_away_report(self, now: datetime) -> None:
        """Publish once per local clock-hour in Away mode, starting at 07:00."""
        if self._profile != AUTO_PROFILE_AWAY:
            return
        slot = core.hourly_away_report_slot_due(
            now,
            last_completed_slot=self._state.get("hourly_report_slot"),
        )
        if slot is None:
            return
        last_attempt_slot = getattr(self, "_last_hourly_report_attempt_slot", "")
        last_attempt_at = getattr(self, "_last_hourly_report_attempt_at", None)
        if (
            last_attempt_slot == slot
            and last_attempt_at is not None
            and (now - last_attempt_at).total_seconds() < _HOURLY_REPORT_RETRY_MINUTES * 60
        ):
            return
        self._last_hourly_report_attempt_slot = slot
        self._last_hourly_report_attempt_at = now
        publish = self._write_report()
        if publish.get("ok"):
            self._state["hourly_report_slot"] = slot
            self._save_state()
            self._log(f"Hourly Away swing report verified for {slot.split('|', 1)[1]}.")

    def _write_report(self) -> dict[str, Any]:
        try:
            longs, shorts = self._read_watchlists()
            snapshot = self.status_snapshot()
            swing_feed = self._load_swing_feed()
            swing_data_date = str(swing_feed.get("data_date") or "")
            current_session_data = swing_data_date == datetime.now().date().isoformat()
            swing_rows = list(swing_feed.get("rows") or []) if current_session_data else []
            picks = []
            for row in swing_rows[:60]:
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
                "auto_mode": self.auto_mode,
                "ib_status": snapshot["ib_status"],
                "regime": snapshot["regime"],
                "longs": longs,
                "shorts": shorts,
                "swing_picks": picks,
                "swing_data_current": current_session_data,
                "swing_data_line": (
                    f"Swing data: current session {swing_data_date} ({swing_feed.get('source') or 'unknown'})"
                    if current_session_data
                    else (
                        f"Swing data: awaiting today's first completed scan; prior data is {swing_data_date}."
                        if swing_data_date
                        else "Swing data: awaiting today's first completed scan."
                    )
                ),
                "alerts": list(self._alerts_today)[-_MAX_REPORT_ALERTS:][::-1],
                "slots_done": snapshot["slots_done"],
                "next_slot": snapshot["next_slot"],
                "log_lines": list(self._log_lines)[-_MAX_REPORT_LOG_LINES:][::-1],
                "universe_line": snapshot.get("universe_line", ""),
                "industry_line": snapshot.get("industry_line", ""),
                "scorecard_line": self._scorecard_line,
                "auto_longs": self._read_auto_watchlist(AUTO_LONGS_FILE),
                "auto_shorts": self._read_auto_watchlist(AUTO_SHORTS_FILE),
                "runtime_line": f"Runtime: {socket.gethostname()} pid={os.getpid()}",
            }
            try:
                from operations_audit import build_operations_audit

                payload.update(core.build_away_operations_lines(build_operations_audit()))
            except Exception as exc:
                payload.update(
                    {
                        "operations_line": "Health: UNKNOWN - operations audit unavailable",
                        "last_scan_line": "Last scan: UNKNOWN",
                        "tracker_line": f"Tracker: UNKNOWN - {exc}",
                    }
                )
            publish = core.publish_away_report(payload)
            self._last_report_attempt = datetime.now()
            if publish.get("ok"):
                # Only a verified publish counts as a fresh phone report
                # (plan.md 23.8: last_attempt is not last_verified_success).
                self._last_report_write = datetime.now()
                self._last_report_error = ""
                if getattr(self, "_report_publish_failing", False):
                    self._report_publish_failing = False
                    self._log("Away report publishing recovered.")
            else:
                self._last_report_error = str(publish.get("error") or "unknown")
                if not getattr(self, "_report_publish_failing", False):
                    self._report_publish_failing = True
                    self._log(
                        f"Away report publish FAILED ({publish.get('error') or 'unknown'}) - "
                        "phone report is stale until this recovers."
                    )
                logging.error("Away report publish failed: %s", publish.get("error"))
            self.statusChanged.emit(self.status_snapshot())
            return publish
        except Exception as exc:
            self._last_report_attempt = datetime.now()
            self._last_report_error = repr(exc)
            logging.exception("Auto Pilot report write failed")
            return {"ok": False, "verified": False, "path": AUTOPILOT_REPORT_FILE, "error": repr(exc)}

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

    @staticmethod
    def _read_auto_watchlist(path) -> list[str]:
        try:
            return list(read_watchlist_symbols(Path(path)))
        except Exception:
            return []

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
            previous = payload if isinstance(payload, dict) else {}
            return {
                "date": today,
                "enabled": bool(previous.get("enabled")),
                "profile": str(previous.get("profile") or AUTO_PROFILE_DESK),
                "slots_done": [],
                "hourly_report_slot": None,
                "watchlist_built_at": None,
                "suggested_at": None,
                "hod_last_check": None,
                "hod_added": [],
                "wrapup_done_at": None,
                "picks_scored_at": None,
                "autopilot_written": previous.get("autopilot_written") or {"longs": [], "shorts": []},
            }
        payload.setdefault("slots_done", [])
        payload.setdefault("profile", AUTO_PROFILE_DESK)
        payload.setdefault("hourly_report_slot", None)
        payload.setdefault("hod_added", [])
        payload.setdefault("wrapup_done_at", None)
        payload.setdefault("suggested_at", None)
        payload.setdefault("picks_scored_at", None)
        payload.setdefault("autopilot_written", {"longs": [], "shorts": []})
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
