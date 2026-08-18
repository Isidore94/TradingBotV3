"""Phase 1 emit site: the panel records deliveries without ever costing one.

The guarantee under test is ordering and subordination. By the time capture
runs the alert is already on screen, so a failure in capture must be invisible
to the trader: no exception, no lost alert, no missing feed item. The blast
radius of this packet is meant to be zero, and these assert it.
"""

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6")


def _alert(text, tag="green"):
    from ui.models.bounce import BounceAlert

    return BounceAlert.from_callback(text, tag)


@pytest.fixture
def panel(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from ui.panels import alert_center_panel as module

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(
        "alert_delivery_events.get_diagnostics_dir", lambda: tmp_path
    )
    # No deleteLater(): these tests run without an event loop, so a deferred
    # delete would land at interpreter shutdown instead. The suite's other
    # panel tests keep the widget alive for the same reason.
    return module.AlertCenterPanel(parked_symbols_path=tmp_path / "parked.json")


def _rows(tmp_path):
    from alert_delivery_events import load_delivery_events

    return load_delivery_events(tmp_path / "alert_delivery_events")


def test_a_delivered_alert_is_recorded_once(panel, tmp_path):
    panel.add_alert(_alert("[S-TIER] AAOI: Bounce confirmed (long)"))
    rows = _rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "AAOI"
    assert rows[0]["action"] == "delivered"


def test_a_suppressed_alert_records_nothing(panel, tmp_path):
    """Nothing reached the trader, so nothing may claim it was delivered."""

    panel._ignored_symbols = {"AAOI"}
    panel.add_alert(_alert("[S-TIER] AAOI: Bounce confirmed (long)"))
    assert _rows(tmp_path) == []


def test_loudness_recorded_matches_the_panel_s_own_verdict(panel, tmp_path):
    from ui.panels.alert_center_panel import alert_is_loud

    loud = _alert("[S-TIER] AAOI: Bounce confirmed (long)")
    quiet = _alert("[D-TIER] BBBB: Bounce confirmed (long)")
    panel.add_alert(loud)
    panel.add_alert(quiet)

    recorded = {row["symbol"]: row["loud"] for row in _rows(tmp_path)}
    assert recorded["AAOI"] is alert_is_loud(loud)
    assert recorded.get("BBBB") is alert_is_loud(quiet)


def test_muting_the_feed_changes_sounded_but_not_loud(panel, tmp_path):
    panel.sound_input.setChecked(False)
    panel.add_alert(_alert("[S-TIER] AAOI: Bounce confirmed (long)"))
    row = _rows(tmp_path)[0]
    assert row["loud"] is True
    assert row["sounded"] is False


def test_capture_failure_never_costs_the_alert(panel, tmp_path, monkeypatch):
    """The whole point: recording is subordinate to delivering."""

    from ui.panels import alert_center_panel as module

    def explode(*_args, **_kwargs):
        raise RuntimeError("capture is broken")

    monkeypatch.setattr(module, "record_delivery", explode)

    before = len(panel._alerts)
    panel.add_alert(_alert("[S-TIER] AAOI: Bounce confirmed (long)"))
    assert len(panel._alerts) == before + 1
    assert _rows(tmp_path) == []


def test_capture_failure_does_not_stop_the_next_alert(panel, tmp_path, monkeypatch):
    from ui.panels import alert_center_panel as module

    calls = {"n": 0}
    real = module.record_delivery

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "record_delivery", flaky)
    panel.add_alert(_alert("[S-TIER] AAOI: Bounce confirmed (long)"))
    panel.add_alert(_alert("[S-TIER] NVDA: Bounce confirmed (long)"))

    assert [row["symbol"] for row in _rows(tmp_path)] == ["NVDA"]


def test_every_delivered_row_carries_a_typed_identity(panel, tmp_path):
    panel.add_alert(_alert("[S-TIER] AAOI: Bounce confirmed (long)"))
    panel.add_alert(_alert("[B-TIER] NVDA: Bounce confirmed (short)"))
    rows = _rows(tmp_path)
    ids = {row["alert_event_id"] for row in rows}
    assert len(ids) == len(rows)
    assert all(row["alert_type"] for row in rows)
