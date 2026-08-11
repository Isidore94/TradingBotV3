"""BounceBot's intraday sweep is confined to the session window.

Before this, Auto Pilot re-enabled scanning on every 30-second tick with no
clock check, so a desk left running swept the watchlists roughly eight times an
hour all night against prices frozen since the close. These tests pin both
halves of the fix: the pure window verdict, and the transition handling that
keeps a deliberate manual resume from being undone one tick later.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import autopilot_core as core  # noqa: E402
from ui.services.autopilot_service import AutopilotService  # noqa: E402

PACIFIC = "America/Los_Angeles"
# 2026-07-02 is a Thursday; the regular session is 06:30-13:00 Pacific, so the
# default margins put the scan window at 06:00-13:30.
THURSDAY = datetime(2026, 7, 2)
SATURDAY = datetime(2026, 7, 4)


def _due(moment: datetime) -> bool:
    allowed, _ = core.bouncebot_scanning_due(
        moment, session_only=True, local_timezone_name=PACIFIC
    )
    return allowed


def test_window_is_the_session_plus_a_warmup_and_a_winddown():
    start, end = core.bouncebot_scan_window(THURSDAY, local_timezone_name=PACIFIC)
    assert (start.strftime("%H:%M"), end.strftime("%H:%M")) == ("06:00", "13:30")


def test_scanning_runs_through_the_session_and_its_edges():
    assert _due(THURSDAY.replace(hour=6, minute=0)), "the warm-up edge is inside"
    assert _due(THURSDAY.replace(hour=9, minute=45)), "mid-session"
    assert _due(THURSDAY.replace(hour=13, minute=30)), "the wind-down edge is inside"


def test_scanning_is_refused_overnight_and_before_the_warmup():
    assert not _due(THURSDAY.replace(hour=2, minute=0)), "the 2am sweep this fixes"
    assert not _due(THURSDAY.replace(hour=5, minute=59))
    assert not _due(THURSDAY.replace(hour=13, minute=31))
    assert not _due(THURSDAY.replace(hour=21, minute=0))


def test_weekends_are_closed_all_day():
    # The tick loop short-circuits on weekends before it reaches the bot, so a
    # sweep still running on Friday evening has nothing else to stop it.
    assert not _due(SATURDAY.replace(hour=9, minute=45))


def test_the_setting_can_restore_round_the_clock_scanning():
    allowed, reason = core.bouncebot_scanning_due(
        THURSDAY.replace(hour=2), session_only=False, local_timezone_name=PACIFIC
    )
    assert allowed and "around the clock" in reason


def test_a_negative_margin_never_eats_into_the_session(monkeypatch):
    monkeypatch.setattr(core, "_scan_margin_minutes", lambda key, default: 0)
    start, end = core.bouncebot_scan_window(THURSDAY, local_timezone_name=PACIFIC)
    assert (start.strftime("%H:%M"), end.strftime("%H:%M")) == ("06:30", "13:00")


def test_an_unreadable_margin_setting_falls_back_to_the_default(monkeypatch):
    def explode(*_args, **_kwargs):
        raise OSError("settings file is gone")

    monkeypatch.setattr("project_paths.get_local_setting", explode)
    assert core._scan_margin_minutes("whatever", 30) == 30


class _FakeBounceService:
    def __init__(self, *, running=True, scanning=True):
        self.running = running
        self.scanning_enabled = scanning
        self.calls: list[bool] = []

    def set_scanning_enabled(self, enabled: bool) -> None:
        self.scanning_enabled = bool(enabled)
        self.calls.append(bool(enabled))


def _service(bounce: _FakeBounceService, *, window_open=None):
    """Instance without __init__ side effects (timers, state file, signals)."""
    service = AutopilotService.__new__(AutopilotService)
    service._bounce_service = bounce
    service._scan_window_open = window_open
    service._logged: list[str] = []
    service._log = service._logged.append  # type: ignore[method-assign]
    return service


def test_the_close_pauses_a_sweep_that_is_still_running():
    bounce = _FakeBounceService()
    service = _service(bounce, window_open=True)

    service._apply_scan_window(THURSDAY.replace(hour=13, minute=31))

    assert bounce.calls == [False]
    assert not bounce.scanning_enabled
    assert "paused" in service._logged[0]


def test_a_desk_started_after_hours_pauses_on_its_first_tick():
    bounce = _FakeBounceService()
    service = _service(bounce, window_open=None)

    service._apply_scan_window(THURSDAY.replace(hour=21, minute=0))

    assert bounce.calls == [False], "no boundary crossing needed to stop it"


def test_a_manual_resume_outside_the_window_survives_the_next_tick():
    bounce = _FakeBounceService()
    service = _service(bounce, window_open=True)
    service._apply_scan_window(THURSDAY.replace(hour=21, minute=0))
    assert bounce.calls == [False]

    bounce.set_scanning_enabled(True)  # the trader hits Start Scanning
    for minute in (1, 2, 30):
        service._apply_scan_window(THURSDAY.replace(hour=22, minute=minute))

    assert bounce.scanning_enabled, "re-asserting the verdict every tick would fight the trader"
    assert bounce.calls == [False, True]


def test_the_open_resumes_scanning_once():
    bounce = _FakeBounceService(scanning=False)
    service = _service(bounce, window_open=False)

    service._apply_scan_window(THURSDAY.replace(hour=6, minute=0))
    service._apply_scan_window(THURSDAY.replace(hour=6, minute=30))

    assert bounce.calls == [True]
    assert "resumed" in service._logged[0]


def test_a_paused_window_is_quiet_when_nothing_was_scanning():
    bounce = _FakeBounceService(scanning=False)
    service = _service(bounce, window_open=None)

    service._apply_scan_window(THURSDAY.replace(hour=21, minute=0))

    assert bounce.calls == [], "no state change, so no log line and no churn"
    assert service._logged == []


def test_a_broken_session_lookup_leaves_scanning_exactly_as_it_was(monkeypatch):
    # Fail open: extra overnight sweeps are waste, but a silent daytime pause
    # would cost a trading session.
    def explode(*_args, **_kwargs):
        raise RuntimeError("calendar unavailable")

    monkeypatch.setattr(core, "bouncebot_scanning_due", explode)
    bounce = _FakeBounceService()
    service = _service(bounce, window_open=True)

    service._apply_scan_window(THURSDAY.replace(hour=21, minute=0))

    assert bounce.calls == []
    assert bounce.scanning_enabled
    assert service._scanning_allowed_now(THURSDAY.replace(hour=21)) is True
