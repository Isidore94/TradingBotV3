from __future__ import annotations

import json
import logging
import os
import socket
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

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
import evening_mode
import price_alerts
import push_notify
from ui.services.scan_service import ScanService, active_scan_label
from ui.timer_utils import start_staggered, stop_staggered


_TICK_INTERVAL_MS = 30_000
_HOURLY_REPORT_RETRY_MINUTES = 5
_MAX_LOG_LINES = 400
_MAX_REPORT_ALERTS = 15
# Machine-local kill switch for the swing-picks push, defaulting ON: only the
# machine actually publishing the Away report should be phoning its picks.
PUSH_SWINGS_SETTING = "push_away_swings"
# Hour (desk-local, 0-23) before which the swing push stays quiet. The digest
# still publishes hourly from 07:00; only the phone waits.
PUSH_SWINGS_START_HOUR_SETTING = "push_away_swings_start_hour"
# Machine-local kill switch for the hourly D1 level/event push, defaulting ON.
PUSH_D1_EVENTS_SETTING = "push_away_d1_events"
# D1 events awaiting the next hourly push. Bounded so a runaway alert source
# can never grow this without limit; the oldest pending events fall off first.
_MAX_PENDING_D1_EVENTS = 200


#: path -> ((st_mtime_ns, st_size), parsed value) for status_snapshot's
#: file-backed pieces. status_snapshot ran on the GUI thread from a 5 s panel
#: timer plus twice per 30 s tick, re-reading 2 watchlists, 2 auto-watchlists
#: and 2 state JSONs every call - most of the 10 minutes the 2026-08-31 stall
#: log charged to watchlist_utils.py:33 and 3.9 minutes to
#: project_paths.py:165. An unchanged stamp returns the same parsed value;
#: both stamps are needed because an append inside one filesystem timestamp
#: tick still moves the byte count (the review_events template). Caching only:
#: the snapshot's content is unchanged for unchanged files.
_status_file_memo: dict[str, tuple[tuple[int, int], Any]] = {}


def _memoized_file_read(path: Path, reader):
    """``reader(path)``, memoized on the file's ``(st_mtime_ns, st_size)``.

    An unstatable (missing) file is read through uncached - both readers here
    return a cheap default for it, and a stamp that does not exist cannot be
    a cache key.
    """
    path = Path(path)
    try:
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return reader(path)
    slot = str(path)
    cached = _status_file_memo.get(slot)
    if cached is not None and cached[0] == key:
        return cached[1]
    value = reader(path)
    _status_file_memo[slot] = (key, value)
    return value


def _enter_background_thread_mode() -> None:
    """Drop the CALLING thread to Windows background mode (CPU and I/O).

    The after-close wrap-up legitimately grinds for a long time (universe
    rebuild, learning refresh, the Technical Integrity calibration replay over
    a 100 MB+ event log). At normal priority that pegged a core and lagged the
    whole desk - measured live on 2026-07-30. Background mode
    (THREAD_MODE_BACKGROUND_BEGIN) tells the scheduler AND the I/O manager to
    yield to everything interactive; the work still completes, just politely.
    Falls back to lowest thread priority, and to a no-op off Windows. Never
    raises: a priority tweak must not be able to break the wrap-up itself.
    """
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetCurrentThread()
        thread_mode_background_begin = 0x00010000
        thread_priority_lowest = -2
        if not kernel32.SetThreadPriority(handle, thread_mode_background_begin):
            kernel32.SetThreadPriority(handle, thread_priority_lowest)
    except Exception:
        pass
_MAX_REPORT_LOG_LINES = 30


# Truthful Auto Mode semantics (trader rules 2026-08-14, packet R1; supersedes
# the 2026-07-31 and 2026-08-05 adoption rules for AWAY and EVENING).
#
# The discovery logic is IDENTICAL in every mode. What changes is who is
# present to act on it, and therefore what may self-apply and what may make a
# noise. No mode is ever a different strategy.
#
# OFF     - no automatic user-facing list mutations, scans, or alerts.
#           Optional shadow research (suggestion scans that write only the
#           bot-owned autolongs/autoshorts lists) continues ONLY while the
#           "collect research while Auto is off" setting is enabled.
# DESK    - full automation; the desk is the primary surface. Auto-populate
#           picks stage and are adopted into M5 Focus immediately, for the
#           trader to prune (2026-08-05 directive: culling is quicker than
#           approving one at a time). No phone push except price alerts.
# AWAY    - scans, builds watchlists and writes the hourly digest as always,
#           and it is the only mode that phones the swing picks and D1 events.
#           But nobody is at the desk, so: picks STAGE and are never adopted
#           (a name adopted at 09:00 would alert unwatched all day), and live
#           alerts queue SILENTLY - feed, history and the D1 unread badge all
#           keep filling, only the sound is suppressed. The staged picks drain
#           on the flip back to DESK.
# EVENING - armed the night before a sleep-in morning (trader home at 23:30,
#           at the desk 07:00-07:30). It prepares the morning and then STOPS:
#           the Master AVWAP swing scan runs one slot early (open+30 = 07:00
#           on a normal session), the 07:00/07:15/07:30 strength-persistence
#           checks run, and the morning briefing is written - after which no
#           ordinary hourly slot and no open watchlist self-build runs at all.
#           Picks stage and adopt on the wake-up flip to DESK. Price-level
#           alerts push at wake-the-trader priority, and so does the SPY +/-1%
#           wake alarm, the second deliberate exception to the AWAY-only push
#           rule.
#
# Over all four, quiet hours (autopilot_core.auto_scanning_due) confine every
# AUTOMATIC starter to the session window. Manual buttons are never gated.
AUTO_MODE_OFF = "OFF"
AUTO_PROFILE_DESK = "DESK"
AUTO_PROFILE_AWAY = "AWAY"
AUTO_PROFILE_EVENING = "EVENING"
AUTO_PROFILES = (AUTO_PROFILE_DESK, AUTO_PROFILE_AWAY, AUTO_PROFILE_EVENING)
SHADOW_RESEARCH_SETTING = "autopilot_shadow_research"


class AutopilotService(QObject):
    """Unattended mini-PC mode: schedules swing scans, self-builds the
    BounceBot watchlists at the open, folds near-HOD names in on regime
    pauses, and keeps the away report fresh. All heavy work runs
    off the GUI thread; this object only orchestrates."""

    logMessage = Signal(str)
    enabledChanged = Signal(bool)
    statusChanged = Signal(dict)
    #: (previous, current) whenever `auto_mode` actually changes - OFF/DESK/
    #: AWAY/EVENING. Announced rather than acted on: this service decides
    #: nothing new here, and the one listener (the Market Journal capture in
    #: `ui.app`) only writes evidence. A profile change while Auto is OFF is
    #: NOT a flip and is not emitted, because `auto_mode` did not move.
    autoModeChanged = Signal(str, str)
    _reportFinished = Signal(object, str)

    def __init__(self, bounce_service, parent=None) -> None:
        super().__init__(parent)
        self._bounce_service = bounce_service
        self._scan_service = ScanService(self)
        self._scan_service.finished.connect(self._on_scan_finished)
        self._scan_service.failed.connect(self._on_scan_failed)

        self._log_lines: deque[str] = deque(maxlen=_MAX_LOG_LINES)
        self._alerts_today: deque[str] = deque(maxlen=60)
        self._alerts_date = datetime.now().date().isoformat()
        #: D1 level/event alerts seen since the last hourly D1 push. Cleared on
        #: a sent push, so each push carries only what is new.
        self._d1_events_pending: deque[dict[str, str]] = deque(maxlen=_MAX_PENDING_D1_EVENTS)
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
        if self._profile not in AUTO_PROFILES:
            self._profile = AUTO_PROFILE_DESK
        self._active_scan_slot: str | None = None
        self._waiting_scan_slot: str | None = None
        self._building_watchlists = False
        self._hod_check_running = False
        self._reconnect_running = False
        self._universe_rebuild_running = False
        self._universe_last_attempt: datetime | None = None
        self._wrapup_running = False
        #: Packet Q5: the daily pick scorecard has ONE owned worker. The tick
        #: only decides; the 600 MB of CSV is read on `autopilot-scorecard`.
        self._scorecard_running = False
        self._scorecard_guard = threading.Lock()
        self._evening_prep_running = False
        self._evening_briefing_lines: list[str] = []
        self._scorecard_line = ""
        #: M2.3 - how many of today's outcomes were MEASURED, beside the
        #: scorecard's average R. Filled from the rows the scorecard already
        #: streamed; never its own read.
        self._outcome_coverage_line = ""
        self._last_report_write: datetime | None = None
        self._last_report_attempt: datetime | None = None
        self._last_report_error = ""
        # One writer owns the report publication. GUI-originated requests are
        # single-flight and execute on a background thread; existing scan and
        # wrap-up workers use the same lock when they publish synchronously.
        self._report_build_lock = threading.Lock()
        self._report_async_running = False
        self._report_async_pending = False
        self._report_async_pending_reason = ""
        self._report_shutdown = False
        self._reportFinished.connect(self._on_report_finished)
        self._last_hourly_report_attempt_slot = ""
        self._last_hourly_report_attempt_at: datetime | None = None
        self._last_d1_push_slot = ""
        self._last_ib_status: str | None = None
        self._weekend_logged_date: str | None = None
        self._bot_start_deferred = False
        #: Last observed verdict of the BounceBot scan window; None until the
        #: first tick, so a desk started after hours pauses on that first tick
        #: rather than waiting for the next boundary.
        self._scan_window_open: bool | None = None
        #: Same idea for the quiet-hours window, so the Auto Pilot log records
        #: each crossing once instead of once every 30 seconds.
        self._auto_window_open: bool | None = None
        #: One Evening SPY alarm send at a time. The send is a blocking HTTPS
        #: POST on a worker; without this a hung ntfy would stack one thread
        #: per 30-second tick.
        self._spy_alarm_sending = False

        if bounce_service is not None:
            bounce_service.alertReceived.connect(self._on_alert)
            bounce_service.connectionChanged.connect(self._on_connection_changed)

        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)
        start_staggered(self._timer, 35_000)

        if self._enabled:
            # Quiet hours (packet R1): a desk booted at 21:00 with Auto left ON
            # used to connect BounceBot to IB right here. Nothing starts until
            # the window opens; the tick loop picks it up from there.
            allowed, reason = self._auto_work_due()
            if allowed:
                self._log("Auto Pilot resuming from saved state (was ON at last shutdown).")
                self._ensure_bot_running()
            else:
                self._log(
                    "Auto Pilot is ON from saved state, but nothing starts yet - "
                    f"{reason}. BounceBot connects when the window opens; manual "
                    "scans and rebuilds work now."
                )

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
        previous_mode = self.auto_mode
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
        self._emit_auto_mode_change(previous_mode)
        self._request_report_write()

    @property
    def auto_mode(self) -> str:
        """OFF, or the active profile (DESK/AWAY) while enabled."""
        return self._profile if self._enabled else AUTO_MODE_OFF

    @property
    def profile(self) -> str:
        return self._profile

    def set_profile(self, profile: str) -> None:
        """Desk/Away/Evening are presentation profiles - never strategy changes."""
        profile = str(profile or "").strip().upper()
        if profile not in AUTO_PROFILES or profile == self._profile:
            return
        previous_mode = self.auto_mode
        self._profile = profile
        self._state["profile"] = profile
        self._save_state()
        if profile == AUTO_PROFILE_EVENING:
            self._log(
                "Auto profile -> EVENING (sleep-in mode: same discovery, picks stage "
                "silently, 07:00 early swing scan + morning briefing, price alerts "
                "push to the phone at urgent priority)."
            )
        else:
            self._log(f"Auto profile -> {profile} (same decisions; presentation/cadence only).")
        self._emit_auto_mode_change(previous_mode)
        self._request_report_write()

    def _emit_auto_mode_change(self, previous_mode: str) -> None:
        """Announce a real mode change. Never raises into the caller.

        A listener that fails must not be able to leave the mode half-applied -
        the state and the log line are already written by the time this runs.
        """
        current = self.auto_mode
        if current == previous_mode:
            return
        try:
            self.autoModeChanged.emit(str(previous_mode), str(current))
        except Exception:  # noqa: BLE001
            logging.exception("Auto mode change listeners failed.")

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
            self._ensure_bot_running(force=True)  # trader pressed Reconnect
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

    def _swing_slots(self, now: datetime) -> list[str]:
        """Today's swing slots; Evening mode adds the open+30 early run.

        DESK days use the reduced cadence (S4, 2026-09-03: four scans instead
        of six, the close slot kept) unless ``desk_scan_cadence`` says hourly;
        AWAY and EVENING keep the hourly ladder the phone digest reads.
        """
        cadence = core.desk_scan_cadence() if self._profile == AUTO_PROFILE_DESK else "hourly"
        return core.get_autopilot_swing_slots(
            now,
            include_early_slot=self._enabled and self._profile == AUTO_PROFILE_EVENING,
            cadence=cadence,
        )

    def run_swing_scan_now(self) -> None:
        now = datetime.now()
        slots = self._swing_slots(now)
        slot = now.strftime("%H:%M")
        update = core.slot_writes_setup_tracker(slot, reference=now) if slots else False
        self._start_swing_scan(slot_label=f"manual {slot}", update_setup_tracker=update, mark_slots=[])

    def rebuild_watchlists_now(self) -> None:
        self._start_watchlist_build(manual=True)

    def write_report_now(self) -> None:
        self._request_report_write("manual")

    def status_snapshot(self) -> dict[str, Any]:
        now = datetime.now()
        slots = self._swing_slots(now)
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
        def parse(path: Path) -> dict:
            try:
                value = json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
            return value if isinstance(value, dict) else {}

        def read_payload(path: Path) -> dict:
            payload = _memoized_file_read(Path(path), parse)
            # Copied out of the memo before it is handed to the formatter.
            return dict(payload) if isinstance(payload, dict) else {}

        return core.format_industry_snapshot_line(
            read_payload(INDUSTRY_BOARD_STATE_FILE),
            read_payload(INDUSTRY_INTRADAY_RS_STATE_FILE),
        )

    def shutdown(self) -> None:
        self._report_shutdown = True
        stop_staggered(self._timer)
        self._save_state()
        # Hand the shared writer lease back before the scan service goes away:
        # a clean exit must not leave the other machine locked out for the rest
        # of the lease TTL. Only a lease this process instance owns is dropped.
        try:
            core.release_away_report_lease()
        except Exception:
            logging.debug("Away report lease release failed on shutdown.", exc_info=True)
        self._scan_service.shutdown()

    # ------------------------------------------------------------------
    # Tick loop
    # ------------------------------------------------------------------
    @Slot()
    def _tick(self) -> None:
        try:
            self._roll_day_state()
            now = datetime.now()
            # Before every short-circuit below: the sweep must be stoppable on
            # a Friday evening and while Auto Pilot is OFF, and neither of
            # those paths reaches _ensure_bot_running.
            self._apply_scan_window(now)
            self._apply_quiet_hours(now)
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
            self._maybe_run_evening_prep(now)
            self._maybe_hourly_away_report(now)
            self._maybe_push_d1_events(now)
            self._maybe_push_spy_alarm(now)
            # One snapshot per tick: it reads files, and the heartbeat and the
            # emit want the same moment anyway.
            snapshot = self.status_snapshot()
            core.write_heartbeat(
                current_job=self._active_scan_slot or active_scan_label(),
                next_job=str(snapshot.get("next_slot") or ""),
                last_success=self._last_report_write.isoformat(timespec="seconds") if self._last_report_write else "",
            )
            self.statusChanged.emit(snapshot)
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
                # Explicit rather than merely absent: this is what day-rolls
                # the Evening SPY alarm, so last night's stamp can never
                # suppress this morning's first alarm. The attempt clock and
                # failure count roll with it - yesterday's broken ntfy must not
                # start this morning already backed off.
                "spy_alarm_last_sent": None,
                "spy_alarm_last_attempt": None,
                "spy_alarm_failures": 0,
                # What Auto Pilot itself wrote survives the day roll - it is
                # how tomorrow's build tells its own picks from the trader's.
                "autopilot_written": self._state.get("autopilot_written") or {"longs": [], "shorts": []},
            }
            self._scorecard_line = ""
            self._outcome_coverage_line = ""
            self._save_state()
        if self._alerts_date != today:
            self._alerts_date = today
            self._alerts_today.clear()
            # Yesterday's unsent D1 events are not news; a push naming them at
            # 07:00 would read as this morning's.
            self._d1_events_pending.clear()

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
        wrote them today, they are today's picks - keep them."""
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

    def _ensure_bot_running(self, *, force: bool = False) -> None:
        service = self._bounce_service
        if service is None:
            return
        if not service.running:
            # Quiet hours belong HERE, not only at the __init__ resume. This is
            # the one place automation starts BounceBot and the tick calls it
            # every 30 seconds, so gating the boot alone made the refusal
            # cosmetic: a 21:00 launch logged "nothing starts yet" and then
            # connected to IB half a minute later. `force` is the manual
            # carve-out - the desk's Reconnect button still works at any hour.
            if not force:
                allowed, _reason = self._auto_work_due()
                if not allowed:
                    return
            # BounceService owns the "may a startup begin?" decision: a tick
            # landing while a previous startup worker is still inside its IB
            # connect gets False back instead of a second worker (two
            # run_bot_with_gui calls share one hard-coded client id -> IB
            # Error 326 and two live sessions). Log the deferral once, not
            # once per tick.
            if service.start() is False:
                if not service.running and not self._bot_start_deferred:
                    self._bot_start_deferred = True
                    self._log(
                        "BounceBot start deferred: the previous BounceBot worker has not "
                        "retired. Auto Pilot will retry on the next tick."
                    )
            else:
                self._bot_start_deferred = False
                self._log("Starting BounceBot (IB connect + intraday scanning).")
        # Only the session window may switch the sweep back on. Outside it,
        # _apply_scan_window owns the state, so a deliberate manual resume at
        # 21:00 is not undone by the next 30-second tick.
        if not service.scanning_enabled and self._scanning_allowed_now():
            service.set_scanning_enabled(True)

    def _scanning_allowed_now(self, now: datetime | None = None) -> bool:
        """Fails OPEN: a session lookup this cannot answer must never be the
        reason BounceBot sits out a trading day. Extra overnight sweeps are
        waste; a silent daytime pause would be a missed session."""
        try:
            allowed, _ = core.bouncebot_scanning_due(now or datetime.now())
        except Exception:
            logging.exception("BounceBot scan-window check failed; leaving scanning enabled.")
            return True
        return allowed

    # ------------------------------------------------------------------
    # Quiet hours (packet R1): automatic work only inside the session window
    # ------------------------------------------------------------------
    def _auto_work_due(self, now: datetime | None = None) -> tuple[bool, str]:
        """Quiet-hours verdict plus a reason fit for the Auto Pilot log.

        Fails OPEN for the same reason `_scanning_allowed_now` does: a session
        lookup this cannot answer must never be why the desk sits out a trading
        day. Only automatic starters consult it - every manual button runs at
        any hour.
        """
        moment = now or datetime.now()
        try:
            return core.auto_scanning_due(moment)
        except Exception:
            # `auto_scanning_due` already falls back to the fixed window on a
            # session-lookup failure, so reaching here means the check itself
            # broke. Apply the same fixed window rather than opening the day
            # completely (R2.1): a broken gate must not be able to wake the
            # desk at 21:00, which is the thing it was added to stop.
            #
            # The window and the comparison both come from `core` (R2.2 item 2).
            # Spelled out locally, this branch used `hour < 14` while
            # `auto_scanning_due` used an inclusive datetime endpoint, so the
            # two gates disagreed at exactly 14:00:00.000000 - one boundary, two
            # answers, depending on which caller asked.
            logging.exception("Quiet-hours check failed; using the fixed fallback window.")
            if moment.weekday() >= 5:
                return False, "weekend - quiet hours until the next session"
            start, end = core.auto_quiet_hours_fallback_window(moment)
            inside = core.within_auto_scanning_window(moment, start, end)
            label = core.AUTO_QUIET_HOURS_FALLBACK_LABEL
            if inside:
                return True, f"quiet-hours check failed; inside the {label} fallback window"
            return False, f"quiet-hours check failed; outside the {label} fallback window"

    def _apply_quiet_hours(self, now: datetime) -> None:
        """Log each quiet-hours crossing once, never once per tick.

        Announcement only - the refusals themselves live at each automatic
        starter, so a single missed transition can never leave work running
        that the gate would refuse.
        """
        allowed, reason = self._auto_work_due(now)
        if allowed == self._auto_window_open:
            return
        self._auto_window_open = allowed
        if allowed:
            self._log(f"Automatic work resumed - {reason}.")
        else:
            self._log(
                f"Automatic work paused - {reason}. Manual scans, watchlist "
                "rebuilds and BounceBot resumes still work from the desk."
            )

    def _apply_scan_window(self, now: datetime) -> None:
        """Pause or resume BounceBot's sweep on the session boundary.

        Acts only on a *transition*. Re-asserting the verdict every tick would
        make a manual resume impossible to hold for more than 30 seconds, which
        is the failure this method exists to correct, only inverted.

        Runs before the tick's weekend and Auto-Pilot-OFF short-circuits: a
        sweep still running on Friday evening has to be stopped by something,
        and neither of those paths reaches _ensure_bot_running.
        """
        service = self._bounce_service
        if service is None:
            return
        try:
            allowed, reason = core.bouncebot_scanning_due(now)
        except Exception:
            logging.exception("BounceBot scan-window check failed; leaving scanning as it is.")
            return
        if allowed == self._scan_window_open:
            return
        self._scan_window_open = allowed
        if allowed:
            if not service.running or service.scanning_enabled:
                return
            service.set_scanning_enabled(True)
            self._log(f"BounceBot scanning resumed - {reason}.")
        elif service.scanning_enabled:
            service.set_scanning_enabled(False)
            self._log(
                f"BounceBot scanning paused - {reason}. The IB connection stays up; "
                "resume it by hand from the desk if you need an off-hours sweep."
            )

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
            # Quiet hours: `force` is the manual carve-out (rebuild_universe_now),
            # so the trader's button still rebuilds at 21:00. An automatic heal
            # would otherwise sweep the whole universe through yfinance at any
            # hour of the night.
            allowed, _reason = self._auto_work_due(now)
            if not allowed:
                return
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
        allowed, _reason = self._auto_work_due(now)
        if not allowed:
            return
        since_open = core.minutes_since_open(now)
        if since_open < core.AUTOPILOT_WATCHLIST_BUILD_AFTER_OPEN_MINUTES:
            return
        if since_open > core.AUTOPILOT_WATCHLIST_BUILD_DEADLINE_MINUTES:
            return
        if self._profile == AUTO_PROFILE_EVENING:
            # Evening prepares the morning and then stops (trader rule
            # 2026-08-14). Deliberately NOT recorded as `watchlist_built_at`:
            # a skip marker would survive the wake-up flip to DESK and suppress
            # the build for the rest of the morning, which is the one time the
            # trader does want it.
            today = now.date()
            if getattr(self, "_evening_build_skip_logged_date", None) != today:
                self._evening_build_skip_logged_date = today
                self._log(
                    "Evening mode: skipping the open watchlist self-build - Evening "
                    "runs the early swing slot, the strength checks and the briefing, "
                    "then stops. Flip to DESK to build."
                )
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
                trend_context = core.load_daily_context(list(moves.keys()))
                built = core.build_watchlists_from_moves(moves, spy_move, trend_context=trend_context)
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
                wrote = core.write_bouncebot_watchlists(
                    merged_longs["symbols"], merged_shorts["symbols"]
                )
                # The raw bot picks also land in autolongs/autoshorts.txt so
                # they build a separately-attributable outcome history.
                wrote = core.write_auto_watchlists(longs, shorts) and wrote
                if not wrote:
                    # A read-only secondary must not record picks it did not
                    # write: `autopilot_written` is the merge basis that keeps
                    # the trader's hand-added names, and a false record of it
                    # would let a later merge drop names this machine never
                    # placed (plan.md sec 5).
                    self._log(
                        "Shared watchlists not updated: this machine is not the "
                        "designated writer for the home folder."
                    )
                    return
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
                trend_context = core.load_daily_context(list(moves.keys()))
                built = core.build_watchlists_from_moves(moves, spy_move, trend_context=trend_context)
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
        allowed, _reason = self._auto_work_due(now)
        if not allowed:
            self._resolve_slots_after_window(now)
            return
        slots = self._swing_slots(now)
        done = set(self._state.get("slots_done", []))
        due = [
            slot
            for slot in slots
            if slot not in done and datetime.combine(now.date(), _parse_slot(slot)) <= now
        ]
        if not due:
            return
        due = self._evening_filter_slots(due, now, done)
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

    def _resolve_slots_after_window(self, now: datetime) -> None:
        """Once the window has closed, resolve slots that never ran.

        Same reasoning as Evening's refused slots: `after_close_wrapup_due`
        requires EVERY slot to be done, so slots still pending after the window
        closes - a desk that crashed, or slept through the close as this one did
        for 4h39m on 2026-08-11 - would stay pending forever and silently cancel
        the whole after-close wrap-up for the day.

        Before the window opens nothing is resolved: those slots are still going
        to run.
        """
        if now.weekday() >= 5:
            return
        try:
            _start, end = core.auto_scanning_window(reference=now)
        except Exception:
            logging.exception("Quiet-hours window lookup failed; leaving slots pending.")
            return
        if now <= end:
            return  # before/inside the window - nothing to resolve yet
        slots = self._swing_slots(now)
        done = set(self._state.get("slots_done", []))
        pending = [slot for slot in slots if slot not in done]
        if not pending:
            return
        done.update(pending)
        self._state["slots_done"] = sorted(done)
        self._save_state()
        self._log(
            f"Past the {end.strftime('%H:%M')} automatic-work window with "
            f"{len(pending)} swing slot(s) never run ({', '.join(pending)}) - "
            "marking them resolved so the after-close wrap-up still runs."
        )

    def _evening_filter_slots(
        self, due: list[str], now: datetime, done: set[str]
    ) -> list[str]:
        """Evening runs the open+30 slot only; the rest are resolved, not run.

        Trader rule 2026-08-14: Evening's job is to have the day ready on waking
        and to wake the trader if the market moves - not to scan all day. The
        refused slots are marked DONE rather than left pending on purpose. They
        are not going to run, and `after_close_wrapup_due` requires every slot
        to be done, so leaving them pending would silently cancel the after-close
        wrap-up (universe rebuild, learning refresh, integrity calibration) for
        the whole day.
        """
        if self._profile != AUTO_PROFILE_EVENING:
            return due
        try:
            early = core.autopilot_evening_early_slot(now)
        except Exception:
            # Fail open, as everywhere else here: a session lookup this cannot
            # answer must not be the reason a slot is silently dropped.
            logging.exception("Evening early-slot lookup failed; running slots as scheduled.")
            return due
        refused = [slot for slot in due if slot != early]
        if refused:
            done.update(refused)
            self._state["slots_done"] = sorted(done)
            self._save_state()
            self._log(
                f"Evening mode: swing slot(s) {', '.join(refused)} not run - Evening "
                f"scans the {early} early slot and the strength checks, then stops "
                "for the day. Flip to DESK to resume the hourly schedule."
            )
        return [slot for slot in due if slot == early]

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
        self._request_report_write()
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
        self._request_report_write()
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
            # The wrap-up legitimately does hour-class work (universe rebuild,
            # learning refresh, the calibration replay). Background mode
            # deprioritizes this thread's CPU AND I/O so the trader's machine
            # stays responsive while it grinds; the work still completes.
            _enter_background_thread_mode()
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

                # 3) Point-in-time Technical Integrity calibration. This only
                #    writes a research report; it can never change live config.
                #    The report's own generated_at is a step-level completion
                #    stamp: a restart after a crash later in this chain must
                #    not re-burn the whole multi-config replay (it pegs a core
                #    for as long as the 100MB+ event log takes to chew).
                try:
                    from technical_integrity import (
                        calibration_report_is_current,
                        write_technical_integrity_calibration_report,
                    )

                    if calibration_report_is_current():
                        self._log(
                            "Technical Integrity replay already completed today; skipping."
                        )
                    else:
                        report = write_technical_integrity_calibration_report()
                        self._log(
                            "Technical Integrity replay refreshed "
                            f"({report['event_count']} outcomes / {report['session_count']} sessions)."
                        )
                except Exception as exc:
                    self._log(f"Technical Integrity replay failed: {exc}")
                    logging.exception("Technical Integrity replay failed")

                # 4) Scorecard: did the self-built lists produce anything?
                #    (idempotent - the always-on tick path may have run it).
                #    Already off-thread here, so the body runs INLINE through
                #    the same guard - never a nested worker (packet Q5).
                try:
                    self._score_picks_inline(datetime.now())
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

    #: Packet Q5: how many failed scoring runs a day may see before it stops
    #: trying - a permanently unreadable file must not spin a worker per tick.
    SCORECARD_MAX_ATTEMPTS = 3

    def _scorecard_due(self, now: datetime) -> bool:
        if self._state.get("picks_scored_at") or self._state.get("picks_scoring_failed_at"):
            return False
        last_close = core.last_completed_session_close(now)
        if last_close is None or last_close.date() != now.date():
            return False  # today's session has not closed yet
        return True

    def _maybe_score_picks_daily(self, now: datetime) -> None:
        """Once per day after the close: DECIDE here, read on ONE owned worker.

        Packet Q5. Until 2026-09-04 this ran `_score_todays_picks` on whichever
        thread called it - the 30-second tick is the Qt thread - and that read
        materialised two ~300 MB CSVs: `ui_stalls.jsonl` logged 15,739 ms at
        13:00:44 PT. It also wrote `picks_scored_at` BEFORE scoring, so one
        failure was never retried. Now: a second trigger while the worker runs
        is a no-op, `picks_scored_at` is written only on success, a failure
        keeps the last-good line and counts toward `SCORECARD_MAX_ATTEMPTS`.
        """
        if not self._scorecard_due(now):
            return
        if not self._claim_scorecard():
            return
        threading.Thread(
            target=self._scorecard_worker,
            args=(now,),
            name="autopilot-scorecard",
            daemon=True,
        ).start()

    def _claim_scorecard(self) -> bool:
        """Check-and-set under a lock: the tick and the wrap-up worker both reach it."""
        guard = getattr(self, "_scorecard_guard", None)
        if guard is None:
            guard = self._scorecard_guard = threading.Lock()
        with guard:
            if getattr(self, "_scorecard_running", False):
                return False
            self._scorecard_running = True
            return True

    def _scorecard_worker(self, now: datetime) -> None:
        try:
            self._score_picks_now(now)
        finally:
            self._scorecard_running = False

    def _score_picks_inline(self, now: datetime) -> None:
        """The wrap-up worker's door: already off-thread, same guard, no nesting."""
        if not self._scorecard_due(now) or not self._claim_scorecard():
            return
        try:
            self._score_picks_now(now)
        finally:
            self._scorecard_running = False

    def _score_picks_now(self, now: datetime) -> list[str]:
        """The body: snapshot, score, append - and only THEN mark the day done."""
        today = now.date().isoformat()
        try:
            self._snapshot_manual_picks(now)
            lines = self._score_todays_picks(today)
        except Exception:
            logging.exception("Auto Pilot daily pick scoring failed")
            attempts = self._state.get("scorecard_attempts_today") or {}
            count = int(attempts.get("count") or 0) if attempts.get("date") == today else 0
            count += 1
            self._state["scorecard_attempts_today"] = {"date": today, "count": count}
            if count >= self.SCORECARD_MAX_ATTEMPTS:
                self._state["picks_scoring_failed_at"] = now.strftime("%H:%M:%S")
                self._log(
                    f"Pick scorecard gave up after {count} failed attempts today; "
                    "the last good line stays."
                )
            self._save_state()
            return []
        self._state["picks_scored_at"] = now.strftime("%H:%M:%S")
        self._state.pop("scorecard_attempts_today", None)
        self._save_state()
        if lines:
            self._scorecard_line = " | ".join(lines)
            for line in lines:
                self._log(line)
        return lines

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

    def _score_todays_picks(self, today: str | None = None) -> list[str]:
        """Score today's picks. NEVER on the Qt thread - see `_maybe_score_picks_daily`.

        Packet Q5: the reads are STREAMED through `core.read_scorecard_inputs`
        (today's rows only, never a materialised year), every group is scored
        BEFORE any row is appended so a failure leaves no partial scorecard,
        and a failure RAISES so the caller can retry instead of publishing a
        quietly empty answer. A missing picks file is the one empty answer.
        """
        import csv

        today = today or datetime.now().date().isoformat()
        picks: list[dict] = []
        try:
            with AUTOPILOT_PICKS_FILE.open("r", encoding="utf-8", newline="") as handle:
                picks = [row for row in csv.DictReader(handle) if row.get("date") == today]
        except FileNotFoundError:
            pass
        if not picks:
            return ["Picks scorecard: nothing logged today."]

        candidates, outcomes = core.read_scorecard_inputs(
            INTRADAY_BOUNCE_CANDIDATES_FILE, INTRADAY_BOUNCE_OUTCOMES_FILE, today
        )
        # M2.3, from the rows just streamed - the digest says how many of
        # today's outcomes were measured at all, not only what they averaged.
        self._outcome_coverage_line = core.outcome_coverage_line(outcomes)

        lines: list[str] = []
        rows: list[dict] = []
        for group, group_picks in sorted(core.group_picks_by_source(picks).items()):
            scorecard = core.score_autopilot_picks(group_picks, candidates, outcomes)
            label = core.PICK_GROUP_LABELS.get(group, group)
            lines.append(core.format_scorecard_line(scorecard, label=label))
            rows.append(
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
            writer.writerows(rows)
        return lines

    # ------------------------------------------------------------------
    # Evening mode: strength checks + the morning briefing
    # ------------------------------------------------------------------
    def _maybe_run_evening_prep(self, now: datetime) -> None:
        """EVENING only: take the due 07:00/07:15/07:30 strength check and
        keep the morning briefing current so the desk is ready on arrival."""
        if self._profile != AUTO_PROFILE_EVENING or self._evening_prep_running:
            return
        state = evening_mode.load_evening_state(now)
        recorded = list((state.get("checks") or {}).keys())
        slot = evening_mode.due_strength_check(now, recorded)
        # No new check due: only rebuild when a restart lost the in-memory
        # briefing lines but today's checks already exist on disk.
        if slot is None and (self._evening_briefing_lines or not recorded):
            return
        self._evening_prep_running = True

        def worker() -> None:
            try:
                moment = datetime.now()
                state = evening_mode.load_evening_state(moment)
                if slot is not None:
                    staged = evening_mode.staged_picks_from_pending(
                        core.load_auto_populate_pending_picks(now=moment)
                    )
                    symbols = sorted({pick["symbol"] for pick in staged})
                    snapshot = core.fetch_day_snapshot(symbols, log=self._log) if symbols else {}
                    evening_mode.record_strength_check(state, slot, staged, snapshot, moment)
                    self._log(
                        f"Evening strength check {slot}: observed "
                        f"{len(snapshot)}/{len(symbols)} staged picks."
                    )
                swing_feed = self._load_swing_feed()
                payload = evening_mode.build_evening_briefing(
                    now=moment,
                    regime=self._regime_text(),
                    swing_rows=list(swing_feed.get("rows") or []),
                    swing_data_current=(
                        str(swing_feed.get("data_date") or "") == moment.date().isoformat()
                    ),
                    persistence=evening_mode.assess_pick_persistence(state),
                    overnight_triggers=price_alerts.todays_triggers(moment),
                    checks_done=list((state.get("checks") or {}).keys()),
                )
                text = evening_mode.render_evening_briefing(payload)
                evening_mode.write_evening_briefing_file(text)
                self._evening_briefing_lines = evening_mode.briefing_summary_lines(payload)
                final_slot = evening_mode.EVENING_STRENGTH_CHECK_SLOTS[-1]
                if slot == final_slot and not state.get("announced_at"):
                    # Desk-side announcement only. The phone push this used to
                    # send is retired by the trader's 2026-08-11 rule: AWAY is
                    # the only mode that pushes, and EVENING ends with the
                    # trader walking to this screen anyway. Research-tab price
                    # alerts remain the one always-on phone channel.
                    state["announced_at"] = moment.strftime("%H:%M:%S")
                    summary = " | ".join(self._evening_briefing_lines[:3])
                    self._emit_info_alert(f"MORNING BRIEFING READY - {summary}", "blue")
                    self._log("Morning briefing finalized after the 07:30 strength check.")
                evening_mode.save_evening_state(state)
                self._write_report()
            except Exception as exc:
                self._log(f"Evening briefing update failed: {exc}")
                logging.exception("Evening briefing update failed")
            finally:
                self._evening_prep_running = False

        threading.Thread(target=worker, name="autopilot-evening", daemon=True).start()

    def _push_swing_picks(self, payload: dict, now: datetime | None = None) -> None:
        """Phone the best swings on each VERIFIED Away publish.

        Tied to a verified publish rather than an attempt on purpose: the push
        carries the picks inline, so sending them while the digest on Drive is
        stale would put two different answers in the trader's hand (plan.md
        23.8 - last_attempt is not last_verified_success).

        Normal priority. The urgent channel stays reserved for position price
        levels, which are the only thing allowed to break through Focus; a
        swing list that republishes hourly is not that.

        AWAY only (trader rule 2026-08-11): at the desk the trader is already
        reading these lists on screen, so the phone copy is pure noise, and
        noise here costs the credibility the urgent price alerts depend on.

        Fail-quiet: the report has already published by the time this runs, so
        a push problem is logged and never raised.
        """
        if self.auto_mode != AUTO_PROFILE_AWAY:
            return
        try:
            from project_paths import get_local_setting

            if not get_local_setting(PUSH_SWINGS_SETTING, True):
                return
            start_hour = get_local_setting(
                PUSH_SWINGS_START_HOUR_SETTING, core.AUTOPILOT_SWING_PUSH_START_HOUR
            )
            try:
                start_hour = int(start_hour)
            except (TypeError, ValueError):
                start_hour = core.AUTOPILOT_SWING_PUSH_START_HOUR
            if not core.swing_push_due(now or datetime.now(), start_hour=start_hour):
                return  # digest still published; the phone just stays quiet
            if not push_notify.push_configured():
                return
            built = core.build_swing_push(payload)
            if built is None:
                return  # nothing qualified; silence beats an hourly "none"
            title, message = built
            push_notify.send_push(
                title, message, priority="default", tags="chart_with_upwards_trend"
            )
        except Exception as exc:
            self._log(f"Swing picks push failed: {exc}")

    # ------------------------------------------------------------------
    # Away report
    # ------------------------------------------------------------------
    def _maybe_hourly_away_report(self, now: datetime) -> None:
        """Publish once per local clock-hour in Away/Evening mode, from 07:00.

        Evening gets the same cadence: the trader wakes up to a phone that
        already has the current picture without opening the desk.
        """
        if self._profile not in (AUTO_PROFILE_AWAY, AUTO_PROFILE_EVENING):
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
        self._request_report_write(f"hourly:{slot}")

    def _write_report(self) -> dict[str, Any]:
        with self._report_build_lock:
            return self._write_report_locked()

    def _request_report_write(self, reason: str = "") -> bool:
        """Queue a report publish without doing audit or file I/O on Qt."""

        if self._report_shutdown:
            return False
        if self._report_async_running:
            self._report_async_pending = True
            if reason == "manual" or (
                reason and self._report_async_pending_reason != "manual"
            ):
                self._report_async_pending_reason = str(reason)
            return False
        self._report_async_running = True

        def worker() -> None:
            _enter_background_thread_mode()
            publish = self._write_report()
            if self._report_shutdown:
                return
            try:
                self._reportFinished.emit(publish, str(reason or ""))
            except RuntimeError:
                pass

        threading.Thread(target=worker, name="autopilot-report", daemon=True).start()
        return True

    @Slot(object, str)
    def _on_report_finished(self, publish: object, reason: str) -> None:
        self._report_async_running = False
        result = publish if isinstance(publish, dict) else {}
        if reason == "manual":
            if result.get("ok"):
                self._log(f"Away report verified at {AUTOPILOT_REPORT_FILE}")
            else:
                self._log(
                    f"Away report NOT updated: {result.get('error') or 'unknown failure'}"
                )
        elif reason.startswith("hourly:") and result.get("ok"):
            slot = reason[len("hourly:") :]
            self._state["hourly_report_slot"] = slot
            self._save_state()
            label = slot.split("|", 1)[1] if "|" in slot else slot
            self._log(f"Hourly Away swing report verified for {label}.")

        if self._report_async_pending and not self._report_shutdown:
            pending_reason = self._report_async_pending_reason
            self._report_async_pending = False
            self._report_async_pending_reason = ""
            self._request_report_write(pending_reason)

    def _write_report_locked(self) -> dict[str, Any]:
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
                raw = getattr(row, "raw", None)
                family = str((raw or {}).get("setup_family") or "") if isinstance(raw, dict) else ""
                picks.append(
                    {
                        "symbol": getattr(row, "symbol", ""),
                        "side": getattr(row, "side", ""),
                        "bucket": getattr(row, "bucket_label", "") or getattr(row, "bucket", ""),
                        "expected_r": expected,
                        "family": family,
                        "key_level": str(getattr(row, "key_level", "") or ""),
                    }
                )
            # The roster is built from the FULL feed, not the ten ranked picks:
            # "which names are favorites right now" is a membership question,
            # and answering it from a top-ten slice would silently shorten it.
            roster_rows = [
                {
                    "symbol": getattr(row, "symbol", ""),
                    "side": getattr(row, "side", ""),
                    "bucket": getattr(row, "bucket", "") or getattr(row, "bucket_label", ""),
                }
                for row in swing_rows
            ]
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
                # R4 A11: the tracker's own record per setup family, so the
                # digest can rank ACROSS the buckets by win rate. READ HERE
                # rather than in `render_away_report`, which is a pure renderer;
                # an unreadable file yields {} and the ranking falls back to
                # expected R, which is a weaker order and never a wrong one.
                "swing_family_records": core.swing_family_records(),
                "bucket_roster": core.build_bucket_roster(roster_rows),
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
                "outcome_coverage_line": self._outcome_coverage_line,
                "auto_longs": self._read_auto_watchlist(AUTO_LONGS_FILE),
                "auto_shorts": self._read_auto_watchlist(AUTO_SHORTS_FILE),
                # Since R2, AWAY and EVENING STAGE their picks rather than
                # writing them to longs/shorts.txt. Without this the phone
                # report showed no trace of them at all, and the trader could
                # reasonably read the day-trade lists as "everything the bot
                # found" when several names were sitting in a queue instead.
                "staged_picks": self._staged_pick_summary(),
                "evening_briefing_lines": (
                    list(self._evening_briefing_lines)
                    if self.auto_mode == AUTO_PROFILE_EVENING
                    else []
                ),
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
                self._push_swing_picks(payload)
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
        # Copied out of the memo: callers hand these lists on.
        try:
            return list(_memoized_file_read(Path(path), read_watchlist_symbols))
        except Exception:
            return []

    def _read_watchlists(self) -> tuple[list[str], list[str]]:
        try:
            longs = list(_memoized_file_read(Path(LONGS_FILE), read_watchlist_symbols))
        except Exception:
            longs = []
        try:
            shorts = list(_memoized_file_read(Path(SHORTS_FILE), read_watchlist_symbols))
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

    @Slot(object)
    def record_d1_event(self, event) -> None:
        """Queue one D1 level/event alert for the next hourly phone push.

        Fed by the Alert Center, which is the one component that knows the
        routing rules (which D1 alerts are actionable rather than developing
        research, and which chart watches are D1 rather than M5). This service
        only aggregates: it never classifies an alert itself, so the phone and
        the D1 Focus feed can never disagree about what fired.

        Deduplicated on symbol+label, so a level re-tested three times in an
        hour is one entry. Records in every mode - the AWAY gate belongs on the
        push, not on the collection, or a mode switch mid-session would push a
        hole in the hour it happened to cover.
        """
        if isinstance(event, Mapping):
            symbol = str(event.get("symbol") or "").strip().upper()
            label = str(event.get("label") or "").strip()
            time_text = str(event.get("time_text") or "").strip()
        else:
            symbol = str(getattr(event, "symbol", "") or "").strip().upper()
            label = str(getattr(event, "trigger", "") or "").strip()
            time_text = str(getattr(event, "time_text", "") or "").strip()
        if not symbol:
            return
        label = label.splitlines()[0][:60] if label else ""
        time_text = time_text or datetime.now().strftime("%H:%M:%S")
        for pending in self._d1_events_pending:
            if pending["symbol"] == symbol and pending["label"] == label:
                pending["time_text"] = time_text
                return
        self._d1_events_pending.append(
            {"symbol": symbol, "label": label, "time_text": time_text}
        )

    def _maybe_push_d1_events(self, now: datetime) -> None:
        """AWAY only: phone the D1 level events that fired since the last push.

        Rides the same hourly clock as the Away digest but does NOT depend on
        it publishing: a file-server hiccup must not also cost the trader the
        D1 events, which are the one thing here that exists nowhere else on the
        phone. Silent when nothing fired.
        """
        if self.auto_mode != AUTO_PROFILE_AWAY or not self._d1_events_pending:
            return
        slot = f"{now.date().isoformat()}|{now.hour:02d}"
        if getattr(self, "_last_d1_push_slot", "") == slot:
            return
        try:
            from project_paths import get_local_setting

            if not get_local_setting(PUSH_D1_EVENTS_SETTING, True):
                return
            if not push_notify.push_configured():
                return
            built = core.build_d1_events_push(list(self._d1_events_pending))
            if built is None:
                return
            title, message = built
            result = push_notify.send_push(
                title, message, priority="default", tags="mag"
            )
            # Clear only on a delivered push: an ntfy failure must not swallow
            # the events, and the next hour re-sends them with whatever else
            # fired meanwhile.
            if result.get("ok"):
                self._last_d1_push_slot = slot
                self._d1_events_pending.clear()
        except Exception as exc:
            self._log(f"D1 events push failed: {exc}")

    def _staged_pick_summary(self) -> dict[str, list[str]]:
        """Today's queued picks, per side. Empty when nothing is waiting.

        Read-only and best-effort: the report must publish even when the queue
        cannot be read, because a missing section is a smaller problem than a
        missing report.
        """
        try:
            pending = core.load_auto_populate_pending_picks()
        except Exception:
            logging.debug("Staged picks unavailable for the report.", exc_info=True)
            return {"long": [], "short": []}
        queue = pending.get("pending") or {}
        return {
            side: sorted(str(sym).strip().upper() for sym in (queue.get(side) or {}))
            for side in ("long", "short")
        }

    def _maybe_push_spy_alarm(self, now: datetime) -> None:
        """EVENING only: phone the trader when SPY has moved a full percent.

        The second deliberate exception to the AWAY-only push rule (the first is
        the always-on Research/Focus price alerts). Evening exists because the
        trader worked late and is asleep through the open; a tape that has
        already moved 1% is the thing worth waking up for.

        Repeats every five minutes while the condition holds - the alarm has to
        survive being slept through - and stops the moment the trader flips out
        of EVENING, which is the acknowledgement.
        """
        if self.auto_mode != AUTO_PROFILE_EVENING:
            return
        if self._spy_alarm_sending:
            return  # one send in flight; a slow ntfy must not stack alarms
        try:
            from project_paths import get_local_setting

            if not get_local_setting(core.EVENING_SPY_ALARM_SETTING, True):
                return
            threshold = get_local_setting(
                core.EVENING_SPY_ALARM_PCT_SETTING, core.EVENING_SPY_ALARM_PCT
            )
            try:
                threshold = float(threshold)
            except (TypeError, ValueError):
                threshold = core.EVENING_SPY_ALARM_PCT
            # The alarm belongs to the session, not to the night before it.
            allowed, _reason = self._auto_work_due(now)
            if not allowed:
                return
            if not push_notify.push_configured():
                return
            bot = self._current_bot()
            if bot is None:
                return
            # Champion data path, cached read only: this runs on the GUI thread
            # and must never trigger an IB fetch. No shadow engine is involved.
            spy_today, prev_close = bot._spy_session_bars(cached_only=True)
            if not spy_today or not prev_close:
                return  # missing bars are uncertainty, never confirmation
            # `_spy_session_bars` calls the LAST cached bar's date "today", so
            # overnight it hands back yesterday's session in good faith. The
            # sweep is paused outside the window, so on an Evening morning
            # after a +/-1% day the cache still holds that move and this would
            # wake the trader every five minutes over a tape that already
            # closed. A bar older than today is stale data, not a move.
            last_bar = spy_today[-1]
            stamp = getattr(last_bar, "dt", None)
            bar_date = stamp.date() if hasattr(stamp, "date") else None
            if bar_date != now.date():
                return
            day_pct = (last_bar.close - prev_close) / prev_close * 100.0
            last_sent = self._spy_alarm_last_sent()
            if not core.spy_move_alarm_due(
                day_pct, last_sent, now, threshold_pct=threshold
            ):
                return
            if not self._spy_alarm_attempt_due(now):
                return
            # Everything above is a cheap local read. The SEND is a blocking
            # HTTPS POST with a timeout, and it used to run right here on the
            # GUI thread - so a hung ntfy froze the desk for the request
            # timeout, every tick, in the mode where the trader is asleep and
            # cannot see it. It goes to a worker now, single-flight, so a slow
            # send delays the next attempt instead of stacking sends.
            self._spy_alarm_sending = True
            self._state["spy_alarm_last_attempt"] = now.isoformat(timespec="seconds")
            self._save_state()
            direction = "UP" if day_pct >= 0 else "DOWN"
            title = f"SPY {day_pct:+.2f}% - market is moving"
            message = (
                f"SPY is {direction} {abs(day_pct):.2f}% on the day at "
                f"{now.strftime('%H:%M')}.\n"
                f"Evening wake alarm (threshold ±{threshold:.2f}%). It repeats "
                "every 5 minutes until you flip Auto Pilot out of EVENING."
            )
            threading.Thread(
                target=self._send_spy_alarm,
                args=(title, message, day_pct),
                name="evening-spy-alarm",
                daemon=True,
            ).start()
        except Exception as exc:
            self._spy_alarm_sending = False
            self._log(f"Evening SPY alarm failed: {exc}")

    def _send_spy_alarm(self, title: str, message: str, day_pct: float) -> None:
        """Deliver one alarm off the GUI thread and record what happened.

        Three outcomes, treated differently because they are different:

        - delivered: stamp `spy_alarm_last_sent`, which is what the five-minute
          repeat clock reads, and clear the failure count.
        - rejected: the server answered and said no. Definite, so nothing was
          delivered - but retrying immediately would just fail again, so it
          backs off.
        - ambiguous: a timeout or transport error after the request went out.
          The push may already be on the trader's phone. Retrying immediately
          could wake them twice for one move, so this backs off too, and is
          logged as unknown rather than as a failure.

        Follows the service's existing worker convention (the watchlist build
        does the same): the worker owns the state it writes here, and the tick
        never touches these keys.
        """
        try:
            result = push_notify.send_push(
                title, message, priority="urgent", tags="rotating_light"
            )
            kind = str(result.get("kind") or ("delivered" if result.get("ok") else "rejected"))
            if result.get("ok"):
                self._state["spy_alarm_last_sent"] = datetime.now().isoformat(timespec="seconds")
                self._state["spy_alarm_failures"] = 0
                self._save_state()
                self._log(f"Evening SPY alarm sent: SPY {day_pct:+.2f}% on the day.")
                return
            if kind == "unconfigured":
                # Nothing was transmitted, so this is not a delivery failure and
                # must not push the backoff out - there is simply no phone
                # configured to send to.
                return
            failures = int(self._state.get("spy_alarm_failures") or 0) + 1
            self._state["spy_alarm_failures"] = failures
            self._save_state()
            if kind == "ambiguous":
                self._log(
                    f"Evening SPY alarm outcome UNKNOWN (attempt {failures}) - it may "
                    f"have reached the phone: {result.get('error')}. Backing off rather "
                    "than risking a duplicate wake-up."
                )
            else:
                self._log(
                    f"Evening SPY alarm REJECTED (attempt {failures}): {result.get('error')}"
                )
        except Exception as exc:
            self._log(f"Evening SPY alarm send failed: {exc}")
        finally:
            self._spy_alarm_sending = False

    def _spy_alarm_attempt_due(self, now: datetime) -> bool:
        """Backoff between ATTEMPTS, separate from the five-minute repeat.

        The repeat clock counts delivered alarms; this counts attempts, so a
        broken ntfy cannot turn a 30-second tick into a 30-second retry storm.
        Floor 60s, doubling, capped at one attempt per five minutes - the same
        ceiling as the repeat itself, so a failing alarm never sends faster
        than a working one.
        """
        raw = str(self._state.get("spy_alarm_last_attempt") or "")
        if not raw:
            return True
        try:
            last_attempt = datetime.fromisoformat(raw)
        except ValueError:
            return True
        failures = int(self._state.get("spy_alarm_failures") or 0)
        if failures <= 0:
            return True
        backoff = min(300.0, 60.0 * (2 ** (failures - 1)))
        elapsed = (now - last_attempt).total_seconds()
        return elapsed >= backoff or elapsed < 0

    def _spy_alarm_last_sent(self) -> datetime | None:
        """Last delivered alarm, or None. Day-rolls with the state file, so a
        restart mid-Evening does not re-fire immediately and yesterday's stamp
        can never suppress this morning's alarm."""
        raw = str(self._state.get("spy_alarm_last_sent") or "")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

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
                "spy_alarm_last_sent": None,
                "spy_alarm_last_attempt": None,
                "spy_alarm_failures": 0,
                "autopilot_written": previous.get("autopilot_written") or {"longs": [], "shorts": []},
            }
        payload.setdefault("slots_done", [])
        payload.setdefault("profile", AUTO_PROFILE_DESK)
        payload.setdefault("hourly_report_slot", None)
        payload.setdefault("hod_added", [])
        payload.setdefault("wrapup_done_at", None)
        payload.setdefault("suggested_at", None)
        payload.setdefault("picks_scored_at", None)
        payload.setdefault("spy_alarm_last_sent", None)
        payload.setdefault("spy_alarm_last_attempt", None)
        payload.setdefault("spy_alarm_failures", 0)
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
