"""AR-2B: ordinary D1 scan ideas are held from the visual-review queue by default.

The test drives ``AlertCenterPanel.add_alert`` and the real review-row control.
It deliberately uses a ready D1 scan row, a manual chart, a Focus event and an
armed-chart hit so the queue distinction is exercised at the actual routing seam.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _scan(symbol: str):
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text="09:35:00",
        symbol=symbol,
        side="LONG",
        trigger="(long) zone1 bounce off AVWAPE",
        timeframe="D1",
        tag="d1_flag_long",
        raw_text=f"MASTER_AVWAP_D1_ZONE: {symbol} (long) zone1 bounce",
        is_d1=True,
    )


def _focus_event(symbol: str):
    from ui.models.bounce import BounceAlert
    from ui.panels.alert_center_panel import FOCUS_D1_EVENT_TAG

    return BounceAlert(
        time_text="09:36:00",
        symbol=symbol,
        side="LONG",
        trigger="Focus pullback",
        timeframe="D1",
        tag=FOCUS_D1_EVENT_TAG,
        raw_text=f"FOCUS D1 {symbol}",
        is_d1=True,
    )


def _watch_hit(symbol: str):
    from ui.models.bounce import BounceAlert
    from ui.panels.alert_center_panel import CHART_WATCH_TAG

    return BounceAlert(
        time_text="09:37:00",
        symbol=symbol,
        side="LONG",
        trigger="Pullback fired",
        timeframe="D1",
        tag=CHART_WATCH_TAG,
        raw_text=f"CHART WATCH {symbol} (LONG): Pullback fired",
        is_d1=True,
    )


def _personal_watch_chart(symbol: str):
    from ui.models.bounce import BounceAlert
    from ui.panels.alert_center_panel import CHART_WATCH_TAG

    return BounceAlert(
        time_text="09:37:00",
        symbol=symbol,
        side="LONG",
        trigger="Your pullback fired",
        timeframe="D1",
        tag=CHART_WATCH_TAG,
        raw_text=f"CHART WATCH {symbol} (LONG): Pullback fired",
        is_d1=True,
    )


def _unknown_d1_family(symbol: str):
    """A future D1 family must fail open into the personal review view."""
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text="09:38:00",
        symbol=symbol,
        side="LONG",
        trigger="future D1 condition",
        timeframe="D1",
        tag="future_d1_family",
        raw_text=f"FUTURE_D1_FAMILY: {symbol} (long) condition",
        is_d1=True,
    )


def _panel(tmp_path, monkeypatch):
    import pick_feedback
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_a, **_k: None)
    pick_feedback.clear_reviewed_today_cache()
    panel = AlertCenterPanel(
        ignored_symbols_path=tmp_path / "ignored.json",
        parked_symbols_path=tmp_path / "parked.json",
        chart_watches_path=tmp_path / "chart_watches.json",
        review_events_path=tmp_path / "review_events.jsonl",
    )
    monkeypatch.setattr(panel, "_auto_mode_now", lambda: "DESK")
    monkeypatch.setattr(panel, "_alerts_may_sound", lambda: False)
    monkeypatch.setattr(panel, "_review_movers_only", False, raising=False)
    monkeypatch.setattr(panel.chart_review, "_reviewed_symbols", lambda: set())
    return panel


def _show_all_scan_button(panel) -> QPushButton:
    buttons = [
        button
        for button in panel.findChildren(QPushButton)
        if "show all" in button.text().casefold()
    ]
    assert len(buttons) == 1, [button.text() for button in panel.findChildren(QPushButton)]
    return buttons[0]


def test_ordinary_d1_scan_ideas_stay_held_until_show_all_without_disturbing_the_chart(
    tmp_path, monkeypatch
):
    """The scan remains recorded, but it cannot replace an active personal chart.

    This is deliberately a panel-level test: a classification helper cannot
    prove that a scan is still written to the D1 backing feed, that protected
    alerts still route, or that toggling the displayed set preserves an in-flight
    capture draft.
    """
    panel = _panel(tmp_path, monkeypatch)
    try:
        assert panel.chart_symbol("NVDA", side="LONG", origin="lookup")
        rail = panel.chart_review.capture_rail
        rail.note_input.setText("keep this draft while changing the view")

        scan = _scan("NVDA")
        focus = _focus_event("MSFT")
        watch = _watch_hit("AAPL")
        unknown = _unknown_d1_family("TSLA")
        panel.add_alert(scan)
        panel.add_alert(focus)
        panel.add_alert(watch)
        panel.add_alert(unknown)

        # It is still a real D1 alert in its backing feed, but the ordinary
        # scan family is not a chart candidate in the default My-alerts view.
        assert scan in panel._d1_alerts
        assert panel._current_review_alert is not None
        assert panel._current_review_alert.tag == "manual_chart"
        assert panel._current_review_alert.symbol == "NVDA"
        assert {alert.symbol for alert in panel._review_queue} == {"MSFT", "AAPL", "TSLA"}

        switch = _show_all_scan_button(panel)
        assert "1" in switch.text(), switch.text()
        review_events: list[str] = []
        monkeypatch.setattr(
            panel,
            "_record_review_event",
            lambda action, **_kwargs: review_events.append(action),
        )
        switch.click()
        QApplication.processEvents()

        assert {alert.symbol for alert in panel._review_queue} == {"NVDA", "MSFT", "AAPL", "TSLA"}
        assert panel._current_review_alert.symbol == "NVDA"
        assert panel._current_review_alert.tag == "manual_chart"
        assert rail.note_input.text() == "keep this draft while changing the view"
        assert review_events == [], "a view switch cannot create a verdict or queue event"

        switch.click()
        QApplication.processEvents()
        assert {alert.symbol for alert in panel._review_queue} == {"MSFT", "AAPL", "TSLA"}
        assert panel._current_review_alert.symbol == "NVDA"
        assert rail.note_input.text() == "keep this draft while changing the view"
    finally:
        panel.close()
        panel.deleteLater()


def test_empty_personal_view_keeps_the_scan_switch_visible_and_protects_a_revealed_chart(
    tmp_path, monkeypatch
):
    """A held scan must not make an otherwise empty review pane look broken."""
    panel = _panel(tmp_path, monkeypatch)
    try:
        panel.show()
        QApplication.processEvents()
        scan = _scan("AAPL")
        panel.add_alert(scan)

        switch = _show_all_scan_button(panel)
        assert switch.isVisible()
        assert switch.text() == "Show all (1)"
        assert panel._current_review_alert is None

        switch.click()
        QApplication.processEvents()
        assert switch.text() == "My alerts"
        assert panel._current_review_alert is scan
        rail = panel.chart_review.capture_rail
        rail.note_input.setText("draft survives the view change")

        switch.click()
        QApplication.processEvents()
        assert panel._current_review_alert is scan
        assert rail.note_input.text() == "draft survives the view change"
    finally:
        panel.close()
        panel.deleteLater()


def test_show_all_never_replaces_a_current_personal_watch_or_its_draft(tmp_path, monkeypatch):
    panel = _panel(tmp_path, monkeypatch)
    try:
        personal = _personal_watch_chart("AAPL")
        panel.chart_alert(personal)
        rail = panel.chart_review.capture_rail
        rail.note_input.setText("keep the pullback chart")
        panel.add_alert(_scan("AAPL"))

        switch = _show_all_scan_button(panel)
        switch.click()
        QApplication.processEvents()
        assert panel._current_review_alert is personal
        assert rail.note_input.text() == "keep the pullback chart"
    finally:
        panel.close()
        panel.deleteLater()


def test_vetoed_or_newly_focus_held_scans_do_not_lie_in_the_show_all_count(
    tmp_path, monkeypatch
):
    from ui.widgets.capture_rail import _REASON_ROLE

    panel = _panel(tmp_path, monkeypatch)
    try:
        scan = _scan("AAPL")
        panel.add_alert(scan)
        switch = _show_all_scan_button(panel)
        assert switch.text() == "Show all (1)"
        switch.click()
        QApplication.processEvents()
        for row in range(panel.chart_review.capture_rail.reason_list.count()):
            item = panel.chart_review.capture_rail.reason_list.item(row)
            if item.data(_REASON_ROLE) == "too_extended_from_base":
                panel.chart_review.capture_rail.reason_list.setCurrentItem(item)
                panel.chart_review.capture_rail.reason_list.itemActivated.emit(item)
                break
        else:
            raise AssertionError("missing extended veto reason")
        QApplication.processEvents()
        assert panel._held_d1_scan_review_count() == 0
        assert "AAPL" not in panel._held_d1_scan_reviews
        switch.click()
        assert not any(alert.symbol == "AAPL" for alert in panel._review_queue)

        focus_scan = _scan("MSFT")
        panel.add_alert(focus_scan)
        assert panel._held_d1_scan_review_count() == 1
        monkeypatch.setattr(panel, "_alert_is_focus", lambda alert: alert.symbol == "MSFT")
        panel._on_focus_membership_changed()
        assert "MSFT" not in panel._held_d1_scan_reviews
        assert any(alert.symbol == "MSFT" for alert in panel._review_queue) or (
            panel._current_review_alert is focus_scan
        )
    finally:
        panel.close()
        panel.deleteLater()
