"""Quiet hours and the 2026-08-14 auto-mode matrix (plan.md Phase 0.5, packet R1).

Three trader rules are pinned here:

1. **Quiet hours.** Automatic work runs on weekdays inside the session window
   and at no other time. Before this, booting the desk at 21:00 swept the whole
   universe through yfinance, connected BounceBot to IB, and self-armed Auto
   Pilot against a closed tape.
2. **Evening stops after its early block.** Evening prepares the morning - the
   open+30 swing slot, the strength checks, the briefing - and then does
   nothing for the rest of the day.
3. **Away queues, never adopts.** Nobody is at the desk to prune, so picks
   stage instead of self-applying and alerts arrive without a sound.

The window tests are modelled on `test_bouncebot_scan_window.py`, whose gate
this one deliberately mirrors; the push tests on `test_away_push_gating.py`.
"""

import os
import sys

import pytest
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import autopilot_core as core  # noqa: E402
import push_notify  # noqa: E402
from ui.services.autopilot_service import (  # noqa: E402
    AUTO_PROFILE_AWAY,
    AUTO_PROFILE_DESK,
    AUTO_PROFILE_EVENING,
    AutopilotService,
    _MAX_PENDING_D1_EVENTS,
)

PACIFIC = "America/Los_Angeles"
# 2026-07-02 is a Thursday; the regular session is 06:30-13:00 Pacific, so the
# quiet-hours window runs 06:00 (BounceBot's warm-up) to 14:00 (close + 1h).
THURSDAY = datetime(2026, 7, 2)
SATURDAY = datetime(2026, 7, 4)


def _due(moment: datetime) -> bool:
    allowed, _reason = core.auto_scanning_due(
        moment, quiet_hours=True, local_timezone_name=PACIFIC
    )
    return allowed


# ---------------------------------------------------------------------------
# The pure window
# ---------------------------------------------------------------------------


def test_the_window_is_the_session_plus_a_warmup_and_an_hour_after_the_close():
    start, end = core.auto_scanning_window(THURSDAY, local_timezone_name=PACIFIC)
    assert (start.strftime("%H:%M"), end.strftime("%H:%M")) == ("06:00", "14:00")


def test_automatic_work_runs_through_the_session_and_its_edges():
    assert _due(THURSDAY.replace(hour=6, minute=0)), "the warm-up edge is inside"
    assert _due(THURSDAY.replace(hour=9, minute=45)), "mid-session"
    assert _due(THURSDAY.replace(hour=14, minute=0)), "the wind-down edge is inside"


def test_automatic_work_is_refused_outside_the_window():
    assert not _due(THURSDAY.replace(hour=2, minute=0))
    assert not _due(THURSDAY.replace(hour=5, minute=59))
    assert not _due(THURSDAY.replace(hour=14, minute=1))
    # The 21:00 boot this packet exists to stop.
    assert not _due(THURSDAY.replace(hour=21, minute=0))


def test_weekends_are_quiet_all_day():
    assert not _due(SATURDAY.replace(hour=9, minute=45))


def test_the_setting_can_restore_round_the_clock_automatic_work():
    allowed, reason = core.auto_scanning_due(
        THURSDAY.replace(hour=2), quiet_hours=False, local_timezone_name=PACIFIC
    )
    assert allowed and "around the clock" in reason


def test_a_broken_session_lookup_fails_open(monkeypatch):
    """A clock this cannot read must never be the reason the desk sits out a
    trading day. Wasting an overnight rebuild is cheap; missing a session is
    not."""

    def explode(*_args, **_kwargs):
        raise RuntimeError("no session data")

    monkeypatch.setattr(core, "auto_scanning_window", explode)
    allowed, reason = core.auto_scanning_due(
        THURSDAY.replace(hour=2), quiet_hours=True, local_timezone_name=PACIFIC
    )
    assert allowed and "unavailable" in reason


def test_the_quiet_window_contains_the_bouncebot_scan_window():
    """The superset invariant.

    Quiet hours gate the IB connect; `bouncebot_scanning_due` gates the sweep
    that connection exists for. If the quiet window opened later than the scan
    window, a 06:10 desk would refuse the connect while the scan window said
    the sweep could run - a sweep with nothing to run on.
    """
    quiet_start, quiet_end = core.auto_scanning_window(
        THURSDAY, local_timezone_name=PACIFIC
    )
    scan_start, scan_end = core.bouncebot_scan_window(
        THURSDAY, local_timezone_name=PACIFIC
    )
    assert quiet_start <= scan_start and quiet_end >= scan_end


def test_the_early_evening_slot_is_open_plus_thirty():
    assert core.autopilot_evening_early_slot(THURSDAY, local_timezone_name=PACIFIC) == "07:00"


# ---------------------------------------------------------------------------
# Service wiring: every automatic starter honours the gate
# ---------------------------------------------------------------------------


class _Signal:
    def connect(self, _slot):
        pass


class _FakeBounceService:
    def __init__(self):
        self.running = False
        self.scanning_enabled = False
        self.started = False
        self.alertReceived = _Signal()
        self.connectionChanged = _Signal()

    def start(self):
        self.started = True
        self.running = True
        return True

    def set_scanning_enabled(self, enabled):
        self.scanning_enabled = bool(enabled)


class _NeverStartThread:
    def __init__(self, *_args, **_kwargs):
        pass

    def start(self):  # pragma: no cover - defensive
        raise AssertionError("no automatic worker may start inside quiet hours")


def _bare_service(*, profile=AUTO_PROFILE_DESK, enabled=True, state=None):
    """Instance without __init__ side effects (timers, state file, signals)."""
    service = AutopilotService.__new__(AutopilotService)
    service._enabled = enabled
    service._profile = profile
    service._state = dict(state or {})
    service._building_watchlists = False
    service._universe_rebuild_running = False
    service._universe_last_attempt = None
    service._auto_window_open = None
    service._spy_alarm_sending = False
    service._scan_service = None
    service._logged: list[str] = []
    service._log = service._logged.append  # type: ignore[method-assign]
    service._save_state = lambda: None  # type: ignore[method-assign]
    return service


def _pin_window(monkeypatch, allowed: bool, reason: str = "test"):
    monkeypatch.setattr(core, "auto_scanning_due", lambda *_a, **_k: (allowed, reason))


def _boot(monkeypatch, *, allowed: bool):
    """Construct a real service so the __init__ resume branch is exercised."""
    bounce = _FakeBounceService()
    monkeypatch.setattr(
        AutopilotService, "_load_state", lambda self: {"enabled": True, "profile": AUTO_PROFILE_DESK}
    )
    monkeypatch.setattr(AutopilotService, "_save_state", lambda self: None)
    monkeypatch.setattr("job_ledger.get_default_ledger", lambda: None)
    _pin_window(monkeypatch, allowed)
    logged: list[str] = []
    monkeypatch.setattr(AutopilotService, "_log", lambda self, message: logged.append(message))
    service = AutopilotService(bounce)
    service._timer.stop()
    return service, bounce, logged


def test_a_late_boot_with_auto_left_on_starts_nothing(monkeypatch):
    _service, bounce, logged = _boot(monkeypatch, allowed=False)
    assert not bounce.started, "a 21:00 boot must not connect BounceBot to IB"
    assert any("nothing starts yet" in line for line in logged)


def test_a_boot_inside_the_window_still_resumes(monkeypatch):
    _service, bounce, logged = _boot(monkeypatch, allowed=True)
    assert bounce.started
    assert any("resuming from saved state" in line for line in logged)


def test_the_tick_cannot_undo_the_boot_refusal(monkeypatch):
    """The 2026-08-15 review's first blocker.

    Gating only the __init__ resume made the refusal cosmetic: the tick calls
    `_ensure_bot_running` every 30 seconds with no clock check, so a 21:00
    launch logged "nothing starts yet" and connected to IB half a minute
    later. The original boot test never caught it because it stopped the timer
    before a tick could run - so this one runs the tick itself.
    """
    service, bounce, _logged = _boot(monkeypatch, allowed=False)
    # Everything the tick does apart from the bot start is out of scope here;
    # `_ensure_bot_running` is deliberately left real.
    for name in (
        "_roll_day_state", "_apply_scan_window", "_apply_quiet_hours",
        "_maybe_auto_arm", "_maybe_clear_stale_auto_lists",
        "_maybe_add_near_extreme_names", "_maybe_score_picks_daily",
        "_ensure_universe_fresh", "_maybe_build_watchlists",
        "_maybe_run_swing_slot", "_maybe_run_wrapup", "_maybe_run_evening_prep",
        "_maybe_hourly_away_report", "_maybe_push_d1_events",
        "_maybe_push_spy_alarm", "status_snapshot",
    ):
        setattr(service, name, lambda *a, **k: {})
    monkeypatch.setattr(core, "write_heartbeat", lambda **_k: None)
    # A weekday, because the tick short-circuits on weekends and the real
    # clock would otherwise decide whether this test proves anything.
    monkeypatch.setattr(
        "ui.services.autopilot_service.datetime",
        type("D", (datetime,), {"now": staticmethod(lambda: THURSDAY.replace(hour=21))}),
    )

    service._tick()
    assert not bounce.started, "the tick must not connect what the boot refused"

    _pin_window(monkeypatch, True)
    service._tick()
    assert bounce.started, "and it must still connect once the window opens"


def test_the_reconnect_button_starts_the_bot_at_any_hour(monkeypatch):
    """`force` is the manual carve-out: quiet hours never gate the trader."""
    service = _bare_service()
    bounce = _FakeBounceService()
    service._bounce_service = bounce
    service._bot_start_deferred = False
    service._reconnect_running = False
    service._current_bot = lambda: None  # type: ignore[method-assign]
    _pin_window(monkeypatch, False)
    service.force_reconnect()
    assert bounce.started


def test_slots_left_pending_past_the_window_are_resolved(monkeypatch):
    """Otherwise a crash or a long sleep silently cancels the wrap-up.

    `after_close_wrapup_due` requires every slot to be done, so slots still
    pending when the window closes would stay pending forever - and the desk
    slept 4h39m through one session on 2026-08-11.
    """
    service = _bare_service()
    service._scan_service = type("S", (), {"running": False})()
    service._swing_slots = lambda _now: ["07:30", "09:00", "13:00"]  # type: ignore[method-assign]
    service._state["slots_done"] = ["07:30"]
    _pin_window(monkeypatch, False)
    monkeypatch.setattr(
        core, "auto_scanning_window", lambda **_k: (THURSDAY.replace(hour=6), THURSDAY.replace(hour=14))
    )

    # Before the window opens nothing is resolved - those slots still run.
    service._maybe_run_swing_slot(THURSDAY.replace(hour=5))
    assert service._state["slots_done"] == ["07:30"]

    # Past the window they are, so the after-close wrap-up can still fire.
    service._maybe_run_swing_slot(THURSDAY.replace(hour=15))
    assert set(service._state["slots_done"]) == {"07:30", "09:00", "13:00"}
    assert any("after-close wrap-up" in line for line in service._logged)


def test_the_launch_universe_heal_is_gated_and_the_fresh_path_is_untouched(monkeypatch):
    """MainWindow's self-heal fired 2.5 s after launch at any hour."""
    try:
        from PySide6.QtCore import QObject
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:  # pragma: no cover - PySide6 is on the desk
        pytest.skip("PySide6 is not installed")
    QApplication.instance() or QApplication([])
    from ui.app import MainWindow

    started: list[str] = []

    # A real QObject, because the un-gated path parents a QTimer to it.
    class _Window(QObject):
        def __init__(self):
            super().__init__()
            self.universe_status = type(
                "L", (), {"setText": lambda self, _t: None, "setStyleSheet": lambda self, _s: None}
            )()
            self._universe_poll = None

        def _poll_universe_heal(self):  # the timer's slot; never fires here
            pass

    window = _Window()
    monkeypatch.setattr(core, "universe_is_stale", lambda *_a, **_k: True)
    monkeypatch.setattr(
        "threading.Thread",
        lambda *a, **k: started.append(k.get("name") or "thread") or type("T", (), {"start": lambda self: None})(),
    )
    monkeypatch.setattr(core, "auto_scanning_due", lambda *_a, **_k: (False, "quiet"))
    MainWindow._self_heal_universe(window)
    assert started == [], "a 21:00 launch must not sweep the universe"

    monkeypatch.setattr(core, "auto_scanning_due", lambda *_a, **_k: (True, "open"))
    MainWindow._self_heal_universe(window)
    assert started == ["universe-self-heal"]


def test_the_quiet_hours_crossing_is_logged_once_not_every_tick(monkeypatch):
    service = _bare_service()
    _pin_window(monkeypatch, False, "quiet hours - outside the 06:00-14:00 window")
    for _ in range(4):
        service._apply_quiet_hours(THURSDAY.replace(hour=21))
    assert len(service._logged) == 1 and "paused" in service._logged[0]

    _pin_window(monkeypatch, True, "inside the 06:00-14:00 window")
    for _ in range(4):
        service._apply_quiet_hours(THURSDAY.replace(hour=9))
    assert len(service._logged) == 2 and "resumed" in service._logged[1]


def test_the_automatic_universe_heal_is_refused_outside_the_window(monkeypatch):
    service = _bare_service()
    _pin_window(monkeypatch, False)
    monkeypatch.setattr(core, "universe_is_stale", lambda *_a, **_k: True)
    monkeypatch.setattr("threading.Thread", _NeverStartThread)
    service._ensure_universe_fresh("tick")
    assert service._logged == [], "quiet hours must refuse before announcing a rebuild"


def test_a_manual_universe_rebuild_ignores_quiet_hours(monkeypatch):
    """`force` is the manual carve-out: the trader's button works at any hour."""
    service = _bare_service()
    _pin_window(monkeypatch, False)
    monkeypatch.setattr(core, "universe_is_stale", lambda *_a, **_k: True)
    monkeypatch.setattr(core, "universe_built_at", lambda *_a, **_k: None)
    started: list[str] = []

    class _Thread:
        def __init__(self, *_args, **kwargs):
            started.append(kwargs.get("name") or "thread")

        def start(self):
            pass

    monkeypatch.setattr("threading.Thread", _Thread)
    service.rebuild_universe_now()
    assert started == ["autopilot-universe"]


def test_the_open_watchlist_build_is_refused_outside_the_window(monkeypatch):
    service = _bare_service()
    _pin_window(monkeypatch, False)
    monkeypatch.setattr("threading.Thread", _NeverStartThread)
    service._maybe_build_watchlists(THURSDAY.replace(hour=21))
    assert service._logged == []


def test_swing_slots_are_refused_outside_the_window(monkeypatch):
    service = _bare_service()
    service._scan_service = type("S", (), {"running": False})()
    _pin_window(monkeypatch, False)
    called: list[str] = []
    service._start_swing_scan = lambda **kwargs: called.append(kwargs["slot_label"])  # type: ignore[method-assign]
    service._maybe_run_swing_slot(THURSDAY.replace(hour=21))
    assert called == []


# ---------------------------------------------------------------------------
# Evening prepares the morning and then stops
# ---------------------------------------------------------------------------


def _slot_service(monkeypatch, *, profile, slots, done=()):
    service = _bare_service(profile=profile)
    service._scan_service = type("S", (), {"running": False})()
    service._state["slots_done"] = list(done)
    service._job_ledger = None
    service._swing_slots = lambda _now: list(slots)  # type: ignore[method-assign]
    _pin_window(monkeypatch, True)
    monkeypatch.setattr(core, "autopilot_evening_early_slot", lambda *_a, **_k: "07:00")
    monkeypatch.setattr(core, "slot_writes_setup_tracker", lambda *_a, **_k: False)
    started: list[str] = []
    service._start_swing_scan = lambda **kwargs: started.append(kwargs["slot_label"])  # type: ignore[method-assign]
    return service, started


def test_evening_runs_the_early_slot(monkeypatch):
    service, started = _slot_service(
        monkeypatch, profile=AUTO_PROFILE_EVENING, slots=["07:00", "07:30"]
    )
    service._maybe_run_swing_slot(THURSDAY.replace(hour=7, minute=5))
    assert started == ["07:00"]


def test_evening_refuses_every_slot_after_the_early_one(monkeypatch):
    service, started = _slot_service(
        monkeypatch,
        profile=AUTO_PROFILE_EVENING,
        slots=["07:00", "07:30", "09:00", "10:00"],
        done=["07:00"],
    )
    service._maybe_run_swing_slot(THURSDAY.replace(hour=10, minute=5))
    assert started == []
    assert any("not run" in line for line in service._logged)


def test_evening_marks_refused_slots_done_so_the_wrapup_still_runs(monkeypatch):
    """Refused slots are RESOLVED, not left pending.

    `after_close_wrapup_due` requires every slot to be done, so leaving them
    pending would silently cancel the whole after-close wrap-up - the universe
    rebuild, the learning refresh and the integrity calibration - for the day.
    """
    service, _started = _slot_service(
        monkeypatch,
        profile=AUTO_PROFILE_EVENING,
        slots=["07:00", "07:30", "09:00"],
        done=["07:00"],
    )
    service._maybe_run_swing_slot(THURSDAY.replace(hour=9, minute=5))
    assert set(service._state["slots_done"]) == {"07:00", "07:30", "09:00"}


def test_desk_runs_the_ordinary_hourly_slots(monkeypatch):
    service, started = _slot_service(
        monkeypatch, profile=AUTO_PROFILE_DESK, slots=["07:30", "09:00"], done=["07:30"]
    )
    service._maybe_run_swing_slot(THURSDAY.replace(hour=9, minute=5))
    assert started == ["09:00"]


def test_evening_skips_the_open_self_build_without_a_sticky_marker(monkeypatch):
    """The skip must not survive the wake-up flip to DESK.

    Recording it as `watchlist_built_at` would suppress the build for the rest
    of the morning - the one time the trader actually wants it.
    """
    service = _bare_service(profile=AUTO_PROFILE_EVENING)
    _pin_window(monkeypatch, True)
    monkeypatch.setattr(core, "minutes_since_open", lambda *_a, **_k: 45.0)
    monkeypatch.setattr("threading.Thread", _NeverStartThread)
    service._maybe_build_watchlists(THURSDAY.replace(hour=7, minute=15))
    assert not service._state.get("watchlist_built_at")
    assert any("Evening mode" in line for line in service._logged)

    # Logged once a day, not once every 30-second tick.
    service._maybe_build_watchlists(THURSDAY.replace(hour=7, minute=16))
    assert len(service._logged) == 1


# ---------------------------------------------------------------------------
# Evening's SPY wake alarm
# ---------------------------------------------------------------------------


def test_the_alarm_needs_a_real_move():
    assert core.spy_move_alarm_due(1.2, None, THURSDAY)
    assert core.spy_move_alarm_due(-1.2, None, THURSDAY), "a drop wakes the trader too"
    assert core.spy_move_alarm_due(1.0, None, THURSDAY), "the threshold itself counts"
    assert not core.spy_move_alarm_due(0.99, None, THURSDAY)


def test_missing_or_unreadable_data_is_never_an_alarm():
    """Missing data is uncertainty, never confirmation (plan.md sec 5).

    NaN is checked explicitly: `nan < threshold` is False, so without the guard
    a NaN would read as "past the threshold" and phone the trader awake over no
    data at all.
    """
    assert not core.spy_move_alarm_due(None, None, THURSDAY)
    assert not core.spy_move_alarm_due(float("nan"), None, THURSDAY)
    assert not core.spy_move_alarm_due("not a number", None, THURSDAY)


def test_the_alarm_repeats_every_five_minutes_and_no_faster():
    now = THURSDAY.replace(hour=7, minute=0)
    two_minutes_ago = THURSDAY.replace(hour=6, minute=58)
    five_minutes_ago = THURSDAY.replace(hour=6, minute=55)
    assert not core.spy_move_alarm_due(1.5, two_minutes_ago, now)
    assert core.spy_move_alarm_due(1.5, five_minutes_ago, now)
    # A stamp from the future (clock skew) buys silence, never a repeat storm.
    assert not core.spy_move_alarm_due(1.5, THURSDAY.replace(hour=8), now)


class _Sent:
    def __init__(self, ok=True):
        self.ok = ok
        self.calls: list[tuple[str, str, dict]] = []

    def __call__(self, title, message, **kwargs):
        self.calls.append((title, message, kwargs))
        return {"ok": self.ok, "error": "" if self.ok else "ntfy HTTP 500"}


class _Bar:
    def __init__(self, close, dt=None):
        self.close = close
        # Dated by default: the alarm refuses a series whose last bar predates
        # the day it is asked about, so an undated bar is not a valid fixture.
        self.dt = dt if dt is not None else THURSDAY.replace(hour=9, minute=30)


class _Immediate:
    """Run a worker's target inline, so the send is testable without threads.

    The alarm send moved onto a worker in R2.1 (a blocking HTTPS POST must not
    sit on the GUI thread); these tests still want it to have finished by the
    time they assert.
    """

    def __init__(self, *_args, target=None, args=(), **_kwargs):
        self._target = target
        self._args = args

    def start(self):
        if self._target is not None:
            self._target(*self._args)


def _alarm_service(monkeypatch, *, profile=AUTO_PROFILE_EVENING, bars=(_Bar(103.0),), prev=100.0):
    service = _bare_service(profile=profile)
    monkeypatch.setattr("threading.Thread", _Immediate)
    service._d1_events_pending = deque(maxlen=_MAX_PENDING_D1_EVENTS)
    bot = type("Bot", (), {"_spy_session_bars": lambda self, cached_only=False: (list(bars), prev)})()
    service._current_bot = lambda: bot  # type: ignore[method-assign]
    _pin_window(monkeypatch, True)
    monkeypatch.setattr(push_notify, "push_configured", lambda: True)
    monkeypatch.setattr("project_paths.get_local_setting", lambda key, default=None: default)
    return service


def test_the_spy_alarm_is_evening_only(monkeypatch):
    sent = _Sent()
    monkeypatch.setattr(push_notify, "send_push", sent)
    for profile in (AUTO_PROFILE_DESK, AUTO_PROFILE_AWAY):
        _alarm_service(monkeypatch, profile=profile)._maybe_push_spy_alarm(THURSDAY)
    # Auto Pilot OFF is EVENING-with-enabled-False, and must be just as quiet.
    off = _alarm_service(monkeypatch)
    off._enabled = False
    off._maybe_push_spy_alarm(THURSDAY)
    assert sent.calls == []


def test_the_spy_alarm_fires_at_wake_the_trader_priority(monkeypatch):
    sent = _Sent()
    monkeypatch.setattr(push_notify, "send_push", sent)
    service = _alarm_service(monkeypatch)
    service._maybe_push_spy_alarm(THURSDAY.replace(hour=7))
    assert len(sent.calls) == 1
    title, message, kwargs = sent.calls[0]
    assert "+3.00%" in title and "UP 3.00%" in message
    assert kwargs["priority"] == "urgent"
    assert service._state["spy_alarm_last_sent"]


def test_a_quiet_tape_never_wakes_the_trader(monkeypatch):
    sent = _Sent()
    monkeypatch.setattr(push_notify, "send_push", sent)
    _alarm_service(monkeypatch, bars=(_Bar(100.5),))._maybe_push_spy_alarm(THURSDAY)
    assert sent.calls == []


def test_missing_spy_bars_never_wake_the_trader(monkeypatch):
    sent = _Sent()
    monkeypatch.setattr(push_notify, "send_push", sent)
    _alarm_service(monkeypatch, bars=())._maybe_push_spy_alarm(THURSDAY)
    _alarm_service(monkeypatch, prev=None)._maybe_push_spy_alarm(THURSDAY)
    assert sent.calls == []


def test_yesterdays_cached_move_never_wakes_the_trader(monkeypatch):
    """The 2026-08-15 review's second blocker.

    `_spy_session_bars` calls the last cached bar's date "today", and the sweep
    is paused overnight, so on an Evening morning after a big day the cache
    still holds that move. The quiet window opens 30 minutes before the bell,
    so without a date check the trader was woken every five minutes over a tape
    that had already closed.
    """
    sent = _Sent()
    monkeypatch.setattr(push_notify, "send_push", sent)
    yesterday = THURSDAY.replace(hour=12) - timedelta(days=1)
    service = _alarm_service(monkeypatch, bars=(_Bar(103.0, dt=yesterday),))
    service._maybe_push_spy_alarm(THURSDAY.replace(hour=6, minute=5))
    assert sent.calls == [], "a bar from a previous session is stale data, not a move"

    # The same +3% once today's tape actually prints it does wake the trader.
    fresh = _alarm_service(monkeypatch, bars=(_Bar(103.0),))
    fresh._maybe_push_spy_alarm(THURSDAY.replace(hour=7))
    assert len(sent.calls) == 1


def test_the_kill_switch_silences_the_spy_alarm(monkeypatch):
    sent = _Sent()
    monkeypatch.setattr(push_notify, "send_push", sent)
    service = _alarm_service(monkeypatch)
    monkeypatch.setattr(
        "project_paths.get_local_setting",
        lambda key, default=None: False if key == core.EVENING_SPY_ALARM_SETTING else default,
    )
    service._maybe_push_spy_alarm(THURSDAY)
    assert sent.calls == []


def test_a_failed_send_does_not_stamp_the_alarm(monkeypatch):
    """An ntfy failure must not buy five minutes of silence."""
    sent = _Sent(ok=False)
    monkeypatch.setattr(push_notify, "send_push", sent)
    service = _alarm_service(monkeypatch)
    service._maybe_push_spy_alarm(THURSDAY)
    assert len(sent.calls) == 1
    assert not service._state.get("spy_alarm_last_sent")


def test_the_alarm_is_silent_outside_the_session(monkeypatch):
    sent = _Sent()
    monkeypatch.setattr(push_notify, "send_push", sent)
    service = _alarm_service(monkeypatch)
    _pin_window(monkeypatch, False)
    service._maybe_push_spy_alarm(THURSDAY.replace(hour=3))
    assert sent.calls == []


def test_the_alarm_stamp_day_rolls(monkeypatch):
    """Yesterday's stamp must never suppress this morning's first alarm."""
    service = _bare_service(profile=AUTO_PROFILE_EVENING)
    service._alerts_date = "2026-07-01"
    service._alerts_today = deque(maxlen=60)
    service._d1_events_pending = deque(maxlen=_MAX_PENDING_D1_EVENTS)
    service._state = {"date": "2026-07-01", "spy_alarm_last_sent": "2026-07-01T13:05:00"}
    service._scorecard_line = ""
    monkeypatch.setattr(
        "ui.services.autopilot_service.datetime",
        type("D", (datetime,), {"now": staticmethod(lambda: THURSDAY.replace(hour=6, minute=31))}),
    )
    service._roll_day_state()
    assert service._state["spy_alarm_last_sent"] is None
    assert service._spy_alarm_last_sent() is None


# ---------------------------------------------------------------------------
# Away queues quietly and never adopts
# ---------------------------------------------------------------------------


def _panel(monkeypatch, mode):
    try:
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.panels.alert_center_panel import AlertCenterPanel
    except ModuleNotFoundError as exc:  # pragma: no cover - PySide6 is on the desk
        if exc.name == "PySide6":
            pytest.skip("PySide6 is not installed")
        raise
    panel = AlertCenterPanel()
    monkeypatch.setattr("autopilot_core.read_auto_pilot_mode", lambda *_a, **_k: mode)
    panel._auto_mode_cached = None
    return panel


def _stub_reverify(panel):
    """Make the flip-triggered re-measurement synchronous for tests.

    The real one fetches bars on a worker and re-enters the poll when the
    fresh verdicts land; here it just re-enters, so the deferral is exercised
    without threads or a network.
    """
    def immediate():
        panel._reverify_running = False
        panel._poll_auto_pick_pending()

    panel._start_pending_reverify = immediate  # type: ignore[method-assign]


def _bounce_alert():
    from ui.models.bounce import BounceAlert

    return BounceAlert.from_callback("[S-TIER] NVDA: Bounce confirmed (long)", "green")


def test_away_queues_alerts_without_a_sound(monkeypatch):
    panel = _panel(monkeypatch, "AWAY")
    beeps: list[int] = []
    monkeypatch.setattr(
        "ui.panels.alert_center_panel.QApplication.beep", lambda: beeps.append(1)
    )
    panel.sound_input.setChecked(True)
    panel.add_alert(_bounce_alert())
    assert beeps == [], "AWAY must not make a noise at an empty desk"
    assert len(panel._alerts) == 1, "the alert still queues for the trader's return"


def test_evening_queues_alerts_without_a_sound(monkeypatch):
    """Evening means the trader is asleep, so the desk stays quiet too.

    The SPY wake alarm is Evening's deliberate wake channel; a beeping desk in
    an empty room is not one. Spec section 1's matrix has said "queue" for the
    Evening alert cell since 2026-08-14 - the R1 build only implemented AWAY.
    """
    panel = _panel(monkeypatch, "EVENING")
    beeps: list[int] = []
    monkeypatch.setattr(
        "ui.panels.alert_center_panel.QApplication.beep", lambda: beeps.append(1)
    )
    panel.sound_input.setChecked(True)
    panel.add_alert(_bounce_alert())
    assert beeps == []
    assert len(panel._alerts) == 1


def test_the_d1_feed_beep_follows_the_same_rule(monkeypatch):
    """The D1 feed has its own beep site; AWAY has to silence both."""
    from ui.models.bounce import BounceAlert

    upgrade = BounceAlert.from_callback(
        "MASTER_AVWAP_D1_BUCKET_UPGRADE: NVDA (long) Favorite setup upgrade [score=245]",
        "d1_flag_long",
    )
    for mode, expected in (("DESK", [1]), ("AWAY", []), ("EVENING", [])):
        panel = _panel(monkeypatch, mode)
        beeps: list[int] = []
        monkeypatch.setattr(
            "ui.panels.alert_center_panel.QApplication.beep", lambda: beeps.append(1)
        )
        panel.sound_input.setChecked(True)
        panel.add_alert(upgrade)
        assert beeps == expected, f"{mode} D1 beep"
        assert len(panel._d1_alerts) == 1, f"{mode} still queues the D1 alert"


def test_desk_still_beeps(monkeypatch):
    panel = _panel(monkeypatch, "DESK")
    beeps: list[int] = []
    monkeypatch.setattr(
        "ui.panels.alert_center_panel.QApplication.beep", lambda: beeps.append(1)
    )
    panel.sound_input.setChecked(True)
    panel.add_alert(_bounce_alert())
    assert beeps == [1]


def test_an_unreadable_auto_mode_leaves_the_desk_loud(monkeypatch):
    """Fail LOUD: a missing state file must not silence the trader's desk.

    The AWAY rule is the only thing that suppresses sound, so an unreadable
    mode has to resolve to something that is not AWAY. It resolves to OFF, and
    the desk keeps beeping.
    """
    panel = _panel(monkeypatch, "DESK")
    monkeypatch.setattr(
        "autopilot_core.read_auto_pilot_mode",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("state file gone")),
    )
    panel._auto_mode_cached = None
    panel.sound_input.setChecked(True)
    assert panel._auto_mode_now() == "OFF"
    assert panel._alerts_may_sound() is True


@pytest.mark.parametrize("mode", ["AWAY", "EVENING"])
def test_away_and_evening_refuse_to_adopt_staged_picks(monkeypatch, tmp_path, mode):
    panel = _panel(monkeypatch, mode)
    panel._auto_pick_pending_path = tmp_path / "pending.json"
    adopted: list[str] = []
    panel._adopt_auto_pick_into_focus = lambda *a, **k: adopted.append(a[0]) or True  # type: ignore[method-assign]
    monkeypatch.setattr(
        "autopilot_core.load_auto_populate_pending_picks",
        lambda *_a, **_k: {
            "date": "2026-07-02",
            "pending": {
                "long": {
                    "NVDA": {
                        "reason": "PDH break",
                        "score": 1.4,
                        # This test is about the mode gate, so the pick carries
                        # a passing, current verdict from the staging refresh.
                        "gate_state": "open",
                        "gate_checked_at": datetime.now().isoformat(timespec="seconds"),
                        "gate_bar_end": __import__("autopilot_core")
                        .latest_completed_m5_end()
                        .isoformat(),
                    }
                }
            },
        },
    )
    _stub_reverify(panel)
    panel._poll_auto_pick_pending()
    assert adopted == []
    # Nothing was marked seen, so the flip back to DESK still finds the pick.
    assert not panel._auto_picks_enqueued
    monkeypatch.setattr("autopilot_core.read_auto_pilot_mode", lambda *_a, **_k: "DESK")
    panel._auto_mode_cached = None
    panel._poll_auto_pick_pending()
    assert adopted == ["NVDA"]


@pytest.mark.parametrize("mode", ["AWAY", "EVENING"])
def test_the_drain_re_checks_the_gate_and_drops_what_no_longer_qualifies(
    monkeypatch, tmp_path, mode
):
    """R1 left this drain un-revalidated and said so; R2 closes it.

    A pick can sit in the queue for a whole AWAY day, so the flip back to DESK
    adopts only what the staging refresh most recently verified.
    """
    from datetime import datetime, timedelta

    now = datetime(2026, 7, 2, 11, 2)
    fresh = (now - timedelta(minutes=1)).isoformat(timespec="seconds")
    stale = (now - timedelta(hours=3)).isoformat(timespec="seconds")
    current_bar = datetime(2026, 7, 2, 11, 0).isoformat()

    panel = _panel(monkeypatch, mode)
    panel._auto_pick_pending_path = tmp_path / "pending.json"
    adopted: list[str] = []
    panel._adopt_auto_pick_into_focus = lambda *a, **k: adopted.append(a[0]) or True  # type: ignore[method-assign]
    monkeypatch.setattr(
        "autopilot_core.load_auto_populate_pending_picks",
        lambda *_a, **_k: {
            "date": "2026-07-02",
            "pending": {
                "long": {
                    "GOOD": {"reason": "PDH break", "gate_state": "open",
                             "gate_checked_at": fresh, "gate_bar_end": current_bar},
                    "STALE": {"reason": "PDH break", "gate_state": "open",
                              "gate_checked_at": stale, "gate_bar_end": current_bar},
                    "FAILED": {"reason": "PDH break", "gate_state": "closed",
                               "gate_reason": "not above session VWAP",
                               "gate_checked_at": fresh, "gate_bar_end": current_bar},
                    "UNVERIFIED": {"reason": "PDH break"},
                    "OLDBAR": {"reason": "PDH break", "gate_state": "open",
                               "gate_checked_at": fresh,
                               "gate_bar_end": datetime(2026, 7, 2, 10, 15).isoformat()},
                }
            },
        },
    )
    # Freeze the clock the gate check reads by capturing the real function
    # first, then patching the name to call it with a fixed `now`.
    import autopilot_core

    real_gate_ok = autopilot_core.pending_pick_gate_ok
    monkeypatch.setattr(
        "autopilot_core.pending_pick_gate_ok",
        lambda entry, *_a, **_k: real_gate_ok(entry, now),
    )

    _stub_reverify(panel)

    # Away/Evening refuse outright and mark nothing seen.
    panel._poll_auto_pick_pending()
    assert adopted == []
    assert not panel._auto_picks_enqueued

    # The flip to DESK adopts only the pick whose verdict is fresh AND passing.
    monkeypatch.setattr("autopilot_core.read_auto_pilot_mode", lambda *_a, **_k: "DESK")
    panel._auto_mode_cached = None
    panel._poll_auto_pick_pending()
    assert adopted == ["GOOD"], "OLDBAR is refused on its measured bar, not its clock"
    # The refused four were not marked seen, so the next refresh can re-stamp
    # or evict them rather than the desk losing them silently.
    assert {key[2] for key in panel._auto_picks_enqueued} == {"GOOD"}


# ---------------------------------------------------------------------------
# Alarm delivery policy (R2.1 item 3)
# ---------------------------------------------------------------------------


class _Outcome:
    """A send_push stand-in that reports a chosen outcome kind."""

    def __init__(self, kind):
        self.kind = kind
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        if self.kind == "delivered":
            return {"ok": True, "error": "", "kind": "delivered"}
        return {"ok": False, "error": f"simulated {self.kind}", "kind": self.kind}


def test_the_send_happens_off_the_gui_thread(monkeypatch):
    """A blocking HTTPS POST on the tick froze the desk for the request
    timeout, every tick, in the mode where the trader is asleep."""
    sent = _Sent()
    monkeypatch.setattr(push_notify, "send_push", sent)
    service = _alarm_service(monkeypatch)

    spawned: list[str] = []

    class _Recorder(_Immediate):
        def __init__(self, *args, name="", **kwargs):
            spawned.append(name)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("threading.Thread", _Recorder)
    service._maybe_push_spy_alarm(THURSDAY.replace(hour=7))
    assert spawned == ["evening-spy-alarm"]
    assert len(sent.calls) == 1


def test_only_one_alarm_send_is_in_flight_at_a_time(monkeypatch):
    sent = _Sent()
    monkeypatch.setattr(push_notify, "send_push", sent)
    service = _alarm_service(monkeypatch)
    service._spy_alarm_sending = True
    service._maybe_push_spy_alarm(THURSDAY.replace(hour=7))
    assert sent.calls == [], "a hung ntfy must not stack one send per tick"


def test_attempts_and_deliveries_are_recorded_separately(monkeypatch):
    """The five-minute repeat clock reads deliveries; the backoff reads
    attempts. Conflating them lets a broken ntfy either spam or go silent."""
    rejected = _Outcome("rejected")
    monkeypatch.setattr(push_notify, "send_push", rejected)
    service = _alarm_service(monkeypatch)
    service._maybe_push_spy_alarm(THURSDAY.replace(hour=7))

    assert service._state["spy_alarm_last_attempt"], "the attempt was recorded"
    assert not service._state.get("spy_alarm_last_sent"), "nothing was delivered"
    assert service._state["spy_alarm_failures"] == 1


def test_a_failed_attempt_backs_off_and_a_delivery_clears_it(monkeypatch):
    rejected = _Outcome("rejected")
    monkeypatch.setattr(push_notify, "send_push", rejected)
    service = _alarm_service(monkeypatch)
    start = THURSDAY.replace(hour=7)
    service._maybe_push_spy_alarm(start)
    assert rejected.calls == 1

    # 30 seconds later - the tick cadence - is far too soon.
    service._maybe_push_spy_alarm(start + timedelta(seconds=30))
    assert rejected.calls == 1, "the 30-second tick must not become a retry storm"

    # A minute later is the floor, so the next attempt goes.
    service._maybe_push_spy_alarm(start + timedelta(seconds=61))
    assert rejected.calls == 2

    # Now it succeeds: the failure count clears so the next move is not
    # penalised by an outage that has ended.
    delivered = _Outcome("delivered")
    monkeypatch.setattr(push_notify, "send_push", delivered)
    service._maybe_push_spy_alarm(start + timedelta(seconds=200))
    assert service._state["spy_alarm_failures"] == 0
    assert service._state["spy_alarm_last_sent"]


def test_the_backoff_is_capped_at_one_attempt_every_five_minutes(monkeypatch):
    service = _alarm_service(monkeypatch)
    service._state["spy_alarm_last_attempt"] = THURSDAY.replace(hour=7).isoformat()
    for failures, expected_wait in ((1, 60), (2, 120), (3, 240), (4, 300), (9, 300)):
        service._state["spy_alarm_failures"] = failures
        moment = THURSDAY.replace(hour=7) + timedelta(seconds=expected_wait - 1)
        assert service._spy_alarm_attempt_due(moment) is False, failures
        moment = THURSDAY.replace(hour=7) + timedelta(seconds=expected_wait)
        assert service._spy_alarm_attempt_due(moment) is True, failures


def test_an_ambiguous_timeout_is_logged_as_unknown_not_as_a_rejection(monkeypatch):
    """The push may already be on the phone, so an immediate retry could wake
    the trader twice for one move."""
    monkeypatch.setattr(push_notify, "send_push", _Outcome("ambiguous"))
    service = _alarm_service(monkeypatch)
    service._maybe_push_spy_alarm(THURSDAY.replace(hour=7))

    text = " ".join(service._logged)
    assert "UNKNOWN" in text and "duplicate" in text
    assert "REJECTED" not in text
    assert not service._state.get("spy_alarm_last_sent")
    assert service._state["spy_alarm_failures"] == 1


def test_an_unconfigured_phone_is_not_a_delivery_failure(monkeypatch):
    """Nothing was transmitted, so it must not push the backoff out - there is
    simply no phone to send to."""
    monkeypatch.setattr(push_notify, "push_configured", lambda: True)
    monkeypatch.setattr(push_notify, "send_push", _Outcome("unconfigured"))
    service = _alarm_service(monkeypatch)
    service._maybe_push_spy_alarm(THURSDAY.replace(hour=7))
    assert service._state.get("spy_alarm_failures", 0) == 0


def test_send_push_classifies_its_outcomes():
    """The classification the retry policy depends on."""
    import urllib.error

    def rejecting(*_a, **_k):
        raise urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)

    def timing_out(*_a, **_k):
        raise TimeoutError("read timed out")

    config = {"topic": "tradingbotv3-test", "server": "https://ntfy.sh"}
    assert push_notify.send_push("t", "m", config=config, opener=rejecting)["kind"] == "rejected"
    assert push_notify.send_push("t", "m", config=config, opener=timing_out)["kind"] == "ambiguous"
    assert push_notify.send_push("t", "m", config={})["kind"] == "unconfigured"
