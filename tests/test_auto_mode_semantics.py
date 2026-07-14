"""Packet A items 4-6 (plan.md sec 14.3): truthful OFF/Desk/Away semantics."""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ui.services.autopilot_service import (  # noqa: E402
    AUTO_MODE_OFF,
    AUTO_PROFILE_AWAY,
    AUTO_PROFILE_DESK,
    AutopilotService,
)


def _bare_service(*, enabled: bool, profile: str = AUTO_PROFILE_DESK, shadow: bool = True):
    """Service instance without __init__ side effects (timers, state file)."""
    service = AutopilotService.__new__(AutopilotService)
    service._enabled = enabled
    service._profile = profile
    service._building_watchlists = False
    service._state = {}
    service._shadow_override = shadow
    service._shadow_research_allowed = lambda: service._shadow_override  # type: ignore[method-assign]
    return service


def test_auto_mode_truth_table():
    assert _bare_service(enabled=False).auto_mode == AUTO_MODE_OFF
    assert _bare_service(enabled=False, profile=AUTO_PROFILE_AWAY).auto_mode == AUTO_MODE_OFF
    assert _bare_service(enabled=True).auto_mode == AUTO_PROFILE_DESK
    assert _bare_service(enabled=True, profile=AUTO_PROFILE_AWAY).auto_mode == AUTO_PROFILE_AWAY


def test_strict_off_suggestion_scan_never_starts(monkeypatch):
    from datetime import datetime

    service = _bare_service(enabled=False, shadow=False)
    started = []
    monkeypatch.setattr(
        "threading.Thread",
        lambda *a, **k: started.append(k.get("name") or "thread") or _NeverStartThread(),
    )
    service._maybe_suggest_watchlists(datetime(2026, 7, 10, 7, 5))
    assert started == [], "strict OFF must not launch the suggestion worker"
    assert service._state == {}, "strict OFF must not record suggestion state"


class _NeverStartThread:
    def start(self):  # pragma: no cover - defensive
        raise AssertionError("no worker may start in strict OFF")


def test_shadow_enabled_off_mode_still_respects_time_gates():
    from datetime import datetime

    service = _bare_service(enabled=False, shadow=True)
    # before the open-scan window: the time gate (not the shadow gate) stops it
    service._maybe_suggest_watchlists(datetime(2026, 7, 10, 4, 0))
    assert service._building_watchlists is False


def test_away_profile_changes_labels_not_decisions():
    desk = _bare_service(enabled=True, profile=AUTO_PROFILE_DESK)
    away = _bare_service(enabled=True, profile=AUTO_PROFILE_AWAY)
    # the profile surfaces as a label only; there is no strategy switch to read
    assert desk.auto_mode == "DESK" and away.auto_mode == "AWAY"


def test_only_away_profile_runs_the_hourly_google_drive_report():
    from datetime import datetime

    writes = []
    away = _bare_service(enabled=True, profile=AUTO_PROFILE_AWAY)
    away._state = {"hourly_report_slot": None}
    away._write_report = lambda: writes.append("write") or {"ok": True}
    away._save_state = lambda: None
    away._log = lambda _message: None

    moment = datetime(2026, 7, 10, 7, 0)
    away._maybe_hourly_away_report(moment)
    away._maybe_hourly_away_report(moment)

    assert writes == ["write"]
    assert away._state["hourly_report_slot"] == "2026-07-10|07:00"

    desk = _bare_service(enabled=True, profile=AUTO_PROFILE_DESK)
    desk._write_report = lambda: writes.append("desk") or {"ok": True}
    desk._maybe_hourly_away_report(datetime(2026, 7, 10, 8, 0))
    assert writes == ["write"]


def test_day_roll_preserves_away_profile_and_resets_hourly_report_slot():
    service = _bare_service(enabled=True, profile=AUTO_PROFILE_AWAY)
    service._state = {
        "date": "2000-01-01",
        "hourly_report_slot": "2000-01-01|13:00",
        "autopilot_written": {"longs": ["AAPL"], "shorts": []},
    }
    service._scorecard_line = "old"
    service._alerts_date = "2000-01-01"
    service._alerts_today = []
    service._save_state = lambda: None

    service._roll_day_state()

    assert service._state["profile"] == AUTO_PROFILE_AWAY
    assert service._state["hourly_report_slot"] is None


def test_report_header_renders_mode_labels():
    import autopilot_core as core

    base = {
        "generated_at": "2026-07-10 13:00:00",
        "ib_status": "connected",
        "regime": "bullish_weak",
        "longs": [],
        "shorts": [],
        "swing_picks": [],
        "alerts": [],
        "slots_done": [],
        "next_slot": "10:00",
        "log_lines": [],
        "auto_longs": [],
        "auto_shorts": [],
    }
    off = core.render_away_report({**base, "enabled": False, "auto_mode": "OFF"})
    desk = core.render_away_report({**base, "enabled": True, "auto_mode": "DESK"})
    away = core.render_away_report({**base, "enabled": True, "auto_mode": "AWAY"})
    assert "Mode: OFF" in off
    assert "Mode: AUTO - DESK" in desk
    assert "Mode: AUTO - AWAY" in away
    assert "hourly from 07:00 local" in away


def test_status_snapshot_does_not_advertise_the_active_slot_as_next(monkeypatch):
    from types import SimpleNamespace

    import autopilot_core as core

    service = _bare_service(enabled=True)
    service._state = {"slots_done": ["10:00"]}
    service._active_scan_slot = "11:00"
    service._waiting_scan_slot = None
    service._read_watchlists = lambda: ([], [])
    service._read_auto_watchlist = lambda _path: []
    service._ib_status_text = lambda: "connected"
    service._regime_text = lambda: "bearish_strong"
    service._scan_service = SimpleNamespace(running=True)
    service._universe_line = lambda _now: "Universe: fresh"
    service._universe_rebuild_running = False
    service._wrapup_running = False
    monkeypatch.setattr(core, "get_autopilot_swing_slots", lambda _now: ["10:00", "11:00", "12:00"])

    snapshot = service.status_snapshot()

    assert snapshot["next_slot"] == "12:00"
