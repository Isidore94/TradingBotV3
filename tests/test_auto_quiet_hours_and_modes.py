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
from collections import deque
from datetime import datetime
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
    return bounce, logged


def test_a_late_boot_with_auto_left_on_starts_nothing(monkeypatch):
    bounce, logged = _boot(monkeypatch, allowed=False)
    assert not bounce.started, "a 21:00 boot must not connect BounceBot to IB"
    assert any("nothing starts yet" in line for line in logged)


def test_a_boot_inside_the_window_still_resumes(monkeypatch):
    bounce, logged = _boot(monkeypatch, allowed=True)
    assert bounce.started
    assert any("resuming from saved state" in line for line in logged)


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
    def __init__(self, close):
        self.close = close


def _alarm_service(monkeypatch, *, profile=AUTO_PROFILE_EVENING, bars=(_Bar(103.0),), prev=100.0):
    service = _bare_service(profile=profile)
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
    except ModuleNotFoundError as exc:  # pragma: no cover - PySide6 always present on the desk
        if exc.name == "PySide6":
            return None
        raise
    panel = AlertCenterPanel()
    monkeypatch.setattr("autopilot_core.read_auto_pilot_mode", lambda *_a, **_k: mode)
    panel._auto_mode_cached = None
    return panel


def _bounce_alert():
    from ui.models.bounce import BounceAlert

    return BounceAlert.from_callback("[S-TIER] NVDA: Bounce confirmed (long)", "green")


def test_away_queues_alerts_without_a_sound(monkeypatch):
    panel = _panel(monkeypatch, "AWAY")
    if panel is None:
        return
    beeps: list[int] = []
    monkeypatch.setattr(
        "ui.panels.alert_center_panel.QApplication.beep", lambda: beeps.append(1)
    )
    panel.sound_input.setChecked(True)
    panel.add_alert(_bounce_alert())
    assert beeps == [], "AWAY must not make a noise at an empty desk"
    assert len(panel._alerts) == 1, "the alert still queues for the trader's return"


def test_desk_still_beeps(monkeypatch):
    panel = _panel(monkeypatch, "DESK")
    if panel is None:
        return
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
    if panel is None:
        return
    monkeypatch.setattr(
        "autopilot_core.read_auto_pilot_mode",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("state file gone")),
    )
    panel._auto_mode_cached = None
    panel.sound_input.setChecked(True)
    assert panel._auto_mode_now() == "OFF"
    assert panel._alerts_may_sound() is True


def test_away_refuses_to_adopt_staged_picks(monkeypatch, tmp_path):
    panel = _panel(monkeypatch, "AWAY")
    if panel is None:
        return
    panel._auto_pick_pending_path = tmp_path / "pending.json"
    adopted: list[str] = []
    panel._adopt_auto_pick_into_focus = lambda *a, **k: adopted.append(a[0]) or True  # type: ignore[method-assign]
    monkeypatch.setattr(
        "autopilot_core.load_auto_populate_pending_picks",
        lambda *_a, **_k: {
            "date": "2026-07-02",
            "pending": {"long": {"NVDA": {"reason": "PDH break", "score": 1.4}}},
        },
    )
    panel._poll_auto_pick_pending()
    assert adopted == []
    # Nothing was marked seen, so the flip back to DESK still finds the pick.
    assert not panel._auto_picks_enqueued
    monkeypatch.setattr("autopilot_core.read_auto_pilot_mode", lambda *_a, **_k: "DESK")
    panel._auto_mode_cached = None
    panel._poll_auto_pick_pending()
    assert adopted == ["NVDA"]
