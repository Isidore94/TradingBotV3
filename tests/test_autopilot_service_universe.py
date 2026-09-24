"""The scheduled universe heal never overrides the write floor (WISHLIST P0-1).

On 2026-09-23 the 13:02 stale tick called ``rebuild_universe_if_stale(force=True)``;
the flag meant "skip the stale check" to the caller and "skip the write floor" to
the builder, so a 343-name rebuild replaced a 1,455-name universe.
"""

import logging
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
import universe_builder as ub  # noqa: E402
from ui.services.autopilot_service import AUTO_PROFILE_DESK, AutopilotService  # noqa: E402


def _bare_service(state=None):
    service = AutopilotService.__new__(AutopilotService)
    service._enabled = True
    service._profile = AUTO_PROFILE_DESK
    service._state = dict(state or {})
    service._universe_rebuild_running = False
    service._universe_last_attempt = None
    service._logged: list[str] = []
    service._log = service._logged.append  # type: ignore[method-assign]
    service._save_state = lambda: None  # type: ignore[method-assign]
    service._reports: list[bool] = []
    service._write_report = lambda: service._reports.append(True)  # type: ignore[method-assign]
    return service


class _InlineThread:
    def __init__(self, *_args, target=None, **_kwargs):
        self._target = target

    def start(self):
        self._target()


def _arm(monkeypatch, outcome="rebuilt", details_fill=None):
    """Stale universe, open window, inline worker, and a recording rebuild."""
    calls: list[dict] = []

    def fake_rebuild(*_args, **kwargs):
        calls.append(kwargs)
        if details_fill and kwargs.get("details") is not None:
            kwargs["details"].update(details_fill)
        return outcome

    monkeypatch.setattr(core, "auto_scanning_due", lambda *_a, **_k: (True, "open"))
    monkeypatch.setattr(core, "universe_is_stale", lambda *_a, **_k: True)
    monkeypatch.setattr(core, "universe_built_at", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "rebuild_universe_if_stale", fake_rebuild)
    monkeypatch.setattr("ui.services.autopilot_service.threading.Thread", _InlineThread)
    return calls


_REFUSAL = {"refused": True, "produced": 343, "floor": 727, "kept": 1455}


def test_the_stale_tick_never_overrides_the_floor(monkeypatch):
    calls = _arm(monkeypatch)
    _bare_service()._ensure_universe_fresh("tick")
    assert len(calls) == 1
    assert calls[0]["skip_stale_check"] is True
    assert calls[0]["override_floor"] is False


def test_the_after_close_heal_never_overrides_the_floor(monkeypatch):
    calls = _arm(monkeypatch)
    _bare_service()._ensure_universe_fresh("after-close")
    assert calls[0]["override_floor"] is False


def test_the_manual_button_keeps_the_override(monkeypatch):
    calls = _arm(monkeypatch)
    _bare_service().rebuild_universe_now()
    assert calls[0]["skip_stale_check"] is True
    assert calls[0]["override_floor"] is True


def test_a_refusal_is_logged_and_lands_in_the_digest(monkeypatch):
    _arm(monkeypatch, outcome="refused", details_fill=_REFUSAL)
    service = _bare_service()
    service._ensure_universe_fresh("tick")
    assert any("refused" in line and "343" in line and "727" in line for line in service._logged)
    line = service._universe_refusal_line()
    assert line.startswith("universe rebuild refused")
    assert "343 < floor 727, kept 1455" in line
    assert service._reports, "the phone digest is rewritten with the refusal"


def test_a_refusal_retries_on_the_normal_cadence(monkeypatch):
    calls = _arm(monkeypatch, outcome="refused", details_fill=_REFUSAL)
    service = _bare_service()
    service._ensure_universe_fresh("tick")
    service._ensure_universe_fresh("tick")
    assert len(calls) == 1, "a second tick inside the retry window does nothing"
    service._universe_last_attempt = datetime.now().replace(year=2000)
    service._ensure_universe_fresh("tick")
    assert len(calls) == 2


def test_a_good_rebuild_clears_the_refusal_line(monkeypatch):
    _arm(monkeypatch, outcome="rebuilt")
    service = _bare_service(state={"universe_refusal": dict(_REFUSAL, at="2026-09-23 13:02")})
    assert service._universe_refusal_line()
    service._ensure_universe_fresh("tick")
    assert service._universe_refusal_line() == ""


def test_the_refusal_line_sits_in_operations():
    text = core.render_away_report(
        {"universe_refusal_line": "universe rebuild refused 13:02: 343 < floor 727, kept 1455"}
    )
    operations = text.split("== OPERATIONS ==", 1)[1].split("==", 1)[0]
    assert "343 < floor 727, kept 1455" in operations


def test_core_reports_a_refusal_without_a_traceback(caplog):
    def refusing_builder():
        raise ub.UniverseWriteRefused("below floor", produced=343, floor=727, kept=1455)

    logged: list[str] = []
    details: dict = {}
    with caplog.at_level(logging.INFO):
        outcome = core.rebuild_universe_if_stale(
            skip_stale_check=True,
            builder=refusing_builder,
            built_at=None,
            log=logged.append,
            details=details,
        )
    assert outcome == "refused"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "343" in warnings[0].getMessage() and "1455" in warnings[0].getMessage()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert details == {"refused": True, "produced": 343, "floor": 727, "kept": 1455}
    assert logged == [], "the caller owns the activity-log line"


def test_core_still_fails_on_other_errors():
    def broken():
        raise ValueError("network")

    details: dict = {}
    assert core.rebuild_universe_if_stale(skip_stale_check=True, builder=broken, built_at=None, details=details) == "failed"
    assert details == {}
