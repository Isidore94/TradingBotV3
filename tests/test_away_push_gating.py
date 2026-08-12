"""AWAY is the only mode that pushes to the phone (trader rule 2026-08-11).

The Research-tab price alerts are the deliberate exception and are not touched
here - they keep their own always-on path in PriceAlertService.

These tests pin the gate itself and the D1 collector behind the new hourly D1
push: which alerts qualify, that a failed send never silently eats the events,
and that a push carries only what is new since the previous one.
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

from collections import deque  # noqa: E402

import push_notify  # noqa: E402
from ui.services.autopilot_service import (  # noqa: E402
    AUTO_PROFILE_AWAY,
    AUTO_PROFILE_DESK,
    AUTO_PROFILE_EVENING,
    AutopilotService,
    _MAX_PENDING_D1_EVENTS,
)

THURSDAY = datetime(2026, 7, 2, 10, 15)


def _service(*, profile=AUTO_PROFILE_AWAY, enabled=True, events=()):
    """Instance without __init__ side effects (timers, state file, signals)."""
    service = AutopilotService.__new__(AutopilotService)
    service._enabled = enabled
    service._profile = profile
    service._last_d1_push_slot = ""
    service._d1_events_pending = deque(maxlen=_MAX_PENDING_D1_EVENTS)
    service._d1_events_pending.extend(events)
    service._logged: list[str] = []
    service._log = service._logged.append  # type: ignore[method-assign]
    return service


class _Sent:
    """Records pushes instead of sending them."""

    def __init__(self, ok=True):
        self.ok = ok
        self.calls: list[tuple[str, str]] = []

    def __call__(self, title, message, **_kwargs):
        self.calls.append((title, message))
        return {"ok": self.ok, "error": "" if self.ok else "ntfy HTTP 500"}


def _event(symbol="NVDA", label="5d high"):
    return {"symbol": symbol, "label": label, "time_text": "10:14:00"}


# ---------------------------------------------------------------------------
# The AWAY-only gate
# ---------------------------------------------------------------------------


def test_the_swing_push_is_refused_outside_away(monkeypatch):
    sent = _Sent()
    monkeypatch.setattr(push_notify, "send_push", sent)
    payload = {
        "swing_picks": [{"symbol": "NVDA", "side": "LONG", "bucket": "favorite_setup"}],
        "swing_data_current": True,
    }
    for profile in (AUTO_PROFILE_DESK, AUTO_PROFILE_EVENING):
        _service(profile=profile)._push_swing_picks(payload, now=THURSDAY)
    # Auto Pilot OFF is AWAY-with-enabled-False, and must be just as quiet.
    _service(profile=AUTO_PROFILE_AWAY, enabled=False)._push_swing_picks(
        payload, now=THURSDAY
    )
    assert sent.calls == []


def test_the_swing_push_still_fires_in_away(monkeypatch):
    monkeypatch.setattr(push_notify, "push_configured", lambda: True)
    sent = _Sent()
    monkeypatch.setattr(push_notify, "send_push", sent)
    _service()._push_swing_picks(
        {
            "swing_picks": [{"symbol": "NVDA", "side": "LONG", "bucket": "favorite_setup"}],
            "swing_data_current": True,
        },
        now=THURSDAY,
    )
    assert len(sent.calls) == 1 and "NVDA" in sent.calls[0][1]


def test_the_d1_push_is_refused_outside_away(monkeypatch):
    monkeypatch.setattr(push_notify, "push_configured", lambda: True)
    sent = _Sent()
    monkeypatch.setattr(push_notify, "send_push", sent)
    for profile in (AUTO_PROFILE_DESK, AUTO_PROFILE_EVENING):
        _service(profile=profile, events=[_event()])._maybe_push_d1_events(THURSDAY)
    assert sent.calls == []


# ---------------------------------------------------------------------------
# The hourly D1 push
# ---------------------------------------------------------------------------


def test_the_d1_push_sends_once_an_hour_and_clears_what_it_sent(monkeypatch):
    monkeypatch.setattr(push_notify, "push_configured", lambda: True)
    sent = _Sent()
    monkeypatch.setattr(push_notify, "send_push", sent)
    service = _service(events=[_event(), _event("AMD", "D1 break above")])

    service._maybe_push_d1_events(THURSDAY)
    service._maybe_push_d1_events(THURSDAY.replace(minute=45))

    assert len(sent.calls) == 1, "one push per clock hour"
    assert not service._d1_events_pending

    # A new event in the next hour pushes again, carrying only the new one.
    service.record_d1_event(_event("TSLA", "15EMA reject"))
    service._maybe_push_d1_events(THURSDAY.replace(hour=11))
    assert len(sent.calls) == 2
    assert "TSLA" in sent.calls[1][1]
    assert "NVDA" not in sent.calls[1][1], "already sent last hour"


def test_an_hour_with_no_d1_events_stays_silent(monkeypatch):
    monkeypatch.setattr(push_notify, "push_configured", lambda: True)
    sent = _Sent()
    monkeypatch.setattr(push_notify, "send_push", sent)
    _service()._maybe_push_d1_events(THURSDAY)
    assert sent.calls == []


def test_a_failed_send_keeps_the_events_for_the_next_attempt(monkeypatch):
    """An ntfy hiccup must not cost the trader the events themselves."""
    monkeypatch.setattr(push_notify, "push_configured", lambda: True)
    sent = _Sent(ok=False)
    monkeypatch.setattr(push_notify, "send_push", sent)
    service = _service(events=[_event()])

    service._maybe_push_d1_events(THURSDAY)
    assert service._d1_events_pending, "nothing was delivered, so nothing is spent"

    sent.ok = True
    service._maybe_push_d1_events(THURSDAY.replace(minute=45))
    assert len(sent.calls) == 2 and "NVDA" in sent.calls[1][1]


def test_the_kill_switch_stops_the_d1_push(monkeypatch):
    monkeypatch.setattr(push_notify, "push_configured", lambda: True)
    sent = _Sent()
    monkeypatch.setattr(push_notify, "send_push", sent)
    monkeypatch.setattr(
        "project_paths.get_local_setting",
        lambda key, default=None: False if key == "push_away_d1_events" else default,
    )
    _service(events=[_event()])._maybe_push_d1_events(THURSDAY)
    assert sent.calls == []


def test_an_unreadable_setting_never_breaks_the_tick(monkeypatch):
    def explode(*_args, **_kwargs):
        raise OSError("settings file is gone")

    monkeypatch.setattr("project_paths.get_local_setting", explode)
    service = _service(events=[_event()])
    service._maybe_push_d1_events(THURSDAY)  # must not raise
    assert service._logged and "D1 events push failed" in service._logged[0]


# ---------------------------------------------------------------------------
# The collector
# ---------------------------------------------------------------------------


def test_repeat_events_collapse_but_keep_the_latest_time():
    service = _service()
    service.record_d1_event({"symbol": "nvda", "label": "5d high", "time_text": "09:31:00"})
    service.record_d1_event({"symbol": "NVDA", "label": "5d high", "time_text": "10:02:00"})
    service.record_d1_event({"symbol": "NVDA", "label": "AVWAPE bounce", "time_text": "10:05:00"})

    assert [entry["label"] for entry in service._d1_events_pending] == [
        "5d high",
        "AVWAPE bounce",
    ]
    assert service._d1_events_pending[0]["time_text"] == "10:02:00"


def test_events_are_collected_in_every_mode():
    """The gate belongs on the push: collecting only while AWAY would leave a
    hole in the hour the trader switched modes."""
    service = _service(profile=AUTO_PROFILE_DESK)
    service.record_d1_event(_event())
    assert len(service._d1_events_pending) == 1


def test_an_event_without_a_symbol_is_dropped():
    service = _service()
    service.record_d1_event({"symbol": "", "label": "5d high"})
    service.record_d1_event(object())
    assert not service._d1_events_pending


def test_the_pending_queue_is_bounded():
    service = _service()
    for index in range(_MAX_PENDING_D1_EVENTS + 25):
        service.record_d1_event(_event(f"SYM{index}"))
    assert len(service._d1_events_pending) == _MAX_PENDING_D1_EVENTS


# ---------------------------------------------------------------------------
# The Alert Center classifier that feeds the collector
# ---------------------------------------------------------------------------


def _alert(**kwargs):
    from ui.models.bounce import BounceAlert

    base = {"time_text": "10:31:00", "symbol": "NVDA"}
    base.update(kwargs)
    return BounceAlert(**base)


def test_armed_d1_levels_and_events_qualify():
    from ui.panels.alert_center_panel import d1_push_event

    level = d1_push_event(
        _alert(timeframe="D1", payload={"chart_watch_kind": "d1_level_above"})
    )
    assert level == {"symbol": "NVDA", "label": "D1 break above", "time_text": "10:31:00"}
    event = d1_push_event(
        _alert(timeframe="D1", payload={"chart_watch_kind": "new_20d_high"})
    )
    assert event["label"] == "20d high"
    focus = d1_push_event(
        _alert(timeframe="D1", is_d1=True, payload={"focus_d1_kind": "avwape_bounce"})
    )
    assert focus["label"] == "AVWAPE bounce"


def test_scan_d1_focus_events_qualify_with_a_readable_label():
    from ui.panels.alert_center_panel import d1_push_event

    built = d1_push_event(
        _alert(is_d1=True, raw_text="MASTER_AVWAP_D1_BUCKET_UPGRADE: NVDA favorite -> high conviction")
    )
    assert built["label"] == "bucket upgrade"


def test_the_panel_announces_a_d1_event_whatever_feed_it_routes_to(tmp_path):
    """The two D1 alert shapes take different routes through add_alert (D1
    Focus feed vs the tier-gated main feed). Both must reach the phone."""
    from PySide6.QtWidgets import QApplication

    from ui.panels.alert_center_panel import AlertCenterPanel

    QApplication.instance() or QApplication([])
    panel = AlertCenterPanel(
        parked_symbols_path=tmp_path / "parked.json",
        focus_d1_flags_path=tmp_path / "focus_flags.json",
    )
    seen: list[dict] = []
    panel.d1EventRecorded.connect(seen.append)

    panel.add_alert(  # routes to the D1 Focus feed
        _alert(is_d1=True, raw_text="MASTER_AVWAP_D1_TIER_FLIP: NVDA B -> A")
    )
    panel.add_alert(  # an armed D1 level; routes to the main feed
        _alert(
            symbol="AMD",
            side="LONG",
            timeframe="D1",
            raw_text="CHART WATCH AMD (LONG): broke above",
            payload={"chart_watch_kind": "d1_level_above"},
        )
    )
    panel.add_alert(_alert(symbol="TSLA", raw_text="[S-TIER] TSLA: Bounce confirmed"))

    assert [(event["symbol"], event["label"]) for event in seen] == [
        ("NVDA", "tier flip"),
        ("AMD", "D1 break above"),
    ]


def test_m5_and_developing_d1_alerts_never_reach_the_phone():
    from ui.panels.alert_center_panel import d1_push_event

    assert d1_push_event(_alert(payload={"chart_watch_kind": "band_bounce"})) is None
    assert d1_push_event(_alert(raw_text="[S-TIER] NVDA: Bounce confirmed")) is None
    assert (
        d1_push_event(_alert(is_d1=True, raw_text="MASTER_AVWAP_D1_RESEARCH: NVDA watching"))
        is None
    ), "developing research is not an event"
    assert d1_push_event(_alert(symbol="", is_d1=True, raw_text="MASTER_AVWAP_D1_ZONE: ?")) is None
