"""AR-2A: reason colours are presentation only, across both alert surfaces."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt

_QT_WIDGETS = pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")
_QT_GUI = pytest.importorskip("PySide6.QtGui", reason="PySide6 not installed")


@pytest.fixture(scope="module", autouse=True)
def qapp():
    app = _QT_WIDGETS.QApplication.instance() or _QT_WIDGETS.QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _dark_theme_after_each_test(qapp):
    """Do not leave a different app-wide stylesheet for a later Qt test."""
    from ui.theme import apply_theme

    apply_theme(qapp, "dark")
    yield
    apply_theme(qapp, "dark")


def _apply_theme(qapp, name: str) -> None:
    from ui.theme import apply_theme

    apply_theme(qapp, name)
    qapp.processEvents()


def _foreground(label) -> str:
    color = label.palette().color(_QT_GUI.QPalette.ColorRole.WindowText)
    return color.name().lower()


def _alert(
    *,
    tag: str = "d1_flag",
    timeframe: str = "D1",
    trigger: str = "D1 setup",
    payload: dict | None = None,
):
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text="09:31:00",
        symbol="NVDA",
        side="LONG",
        trigger=trigger,
        timeframe=timeframe,
        tag=tag,
        raw_text=trigger,
        is_d1=timeframe == "D1",
        payload=payload or {},
    )


def _pullback(source: str):
    return _alert(
        tag="chart_watch",
        trigger=f"{source} pullback",
        payload={"chart_watch_kind": "pullback", "timeframe": source},
    )


@pytest.mark.parametrize("theme_name", ("dark", "light"))
def test_chart_reason_text_repaints_for_each_live_reason_then_resets_muted(
    qapp, monkeypatch, tmp_path, theme_name
):
    """A chart's reason tells price, Focus, D1 and pullback sources apart.

    Drive the chart's public alert path.  It must repolish a changed reason;
    replacing the app stylesheet on every alert is not a valid implementation.
    """
    from ui import theme
    from ui.models.bounce import FOCUS_D1_EVENT_TAG, MANUAL_CHART_TAG, BounceAlert
    from ui.widgets.alert_chart_review import AlertChartReview
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    _apply_theme(qapp, theme_name)
    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_a, **_k: None)
    pane = AlertChartReview(
        annotations_path=tmp_path / "trader_annotations.jsonl",
        # No mentor service is supplied: this test only changes alert display
        # and must not start a mentor or snapshot read.
        mentor_context_service=None,
    )
    pane.show()
    qapp.processEvents()
    stylesheet_before = qapp.styleSheet()

    price = BounceAlert.from_callback("PRICE ALERT: NVDA crossed 100", "red")
    steps = (
        (price, {"in_focus": False}, "short"),
        (_alert(), {"in_focus": True}, "long"),
        (_alert(tag=FOCUS_D1_EVENT_TAG), {"in_focus": False}, "long"),
        (_alert(), {"focus_category": "swing", "in_focus": False}, "chart_blue"),
        (_pullback("M15"), {"in_focus": True}, "chart_yellow"),
        (_pullback("M30"), {"in_focus": True}, "chart_purple"),
        (_pullback("H1"), {"in_focus": True}, "chart_light_blue"),
        (
            _alert(tag=MANUAL_CHART_TAG, timeframe="M5", trigger="Manual look"),
            {"in_focus": True},
            "text_secondary",
        ),
    )
    try:
        for alert, kwargs, color_name in steps:
            pane.set_alert(alert, **kwargs)
            qapp.processEvents()
            assert _foreground(pane.alert_text) == theme.color(color_name, theme_name).lower()
            assert qapp.styleSheet() == stylesheet_before

        pane.clear()
        qapp.processEvents()
        assert _foreground(pane.alert_text) == theme.color("text_secondary", theme_name).lower()
        assert qapp.styleSheet() == stylesheet_before
    finally:
        pane.close()
        pane.deleteLater()


@pytest.mark.parametrize("theme_name", ("dark", "light"))
@pytest.mark.parametrize("tag", ("manual_chart", "auto_pick", "focus_review"))
def test_chart_reason_text_mutes_non_alert_walkthrough_forms(
    qapp, monkeypatch, tmp_path, theme_name, tag
):
    """A look, proposal, or Focus walkthrough never wears a live-alert colour."""
    from ui import theme
    from ui.widgets.alert_chart_review import AlertChartReview
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    _apply_theme(qapp, theme_name)
    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_a, **_k: None)
    pane = AlertChartReview(
        annotations_path=tmp_path / "trader_annotations.jsonl",
        mentor_context_service=None,
    )
    pane.show()
    try:
        pane.set_alert(_alert(tag=tag, timeframe="M5", trigger="Just a look"), in_focus=True)
        qapp.processEvents()
        assert _foreground(pane.alert_text) == theme.color("text_secondary", theme_name).lower()
    finally:
        pane.close()
        pane.deleteLater()


@pytest.mark.parametrize("theme_name", ("dark", "light"))
def test_feed_reason_text_repaints_when_focus_changes_without_rebuild(qapp, theme_name):
    """The in-place Feed Focus update changes its real displayed reason colour."""
    from ui import theme
    from ui.widgets.alert_feed_item import AlertFeedItem

    _apply_theme(qapp, theme_name)
    item = AlertFeedItem(_alert())
    item.show()
    qapp.processEvents()
    stylesheet_before = qapp.styleSheet()
    try:
        assert _foreground(item.trigger_label) == theme.color("chart_blue", theme_name).lower()

        assert item.apply_focus_state("swing") is True
        qapp.processEvents()
        assert _foreground(item.trigger_label) == theme.color("long", theme_name).lower()
        assert qapp.styleSheet() == stylesheet_before

        assert item.apply_focus_state("") is True
        qapp.processEvents()
        assert _foreground(item.trigger_label) == theme.color("chart_blue", theme_name).lower()
        assert qapp.styleSheet() == stylesheet_before
    finally:
        item.close()
        item.deleteLater()


@pytest.mark.parametrize("theme_name", ("dark", "light"))
def test_feed_explicit_live_focus_tag_is_green_without_membership(qapp, theme_name):
    """A Focus D1 flag is green even before the row gets a membership badge."""
    from ui import theme
    from ui.models.bounce import FOCUS_D1_EVENT_TAG
    from ui.widgets.alert_feed_item import AlertFeedItem

    _apply_theme(qapp, theme_name)
    item = AlertFeedItem(_alert(tag=FOCUS_D1_EVENT_TAG))
    item.show()
    qapp.processEvents()
    try:
        assert _foreground(item.trigger_label) == theme.color("long", theme_name).lower()
    finally:
        item.close()
        item.deleteLater()


@pytest.mark.parametrize("theme_name", ("dark", "light"))
@pytest.mark.parametrize(
    ("alert", "color_name"),
    (
        (_pullback("M15"), "chart_yellow"),
        (_pullback("M30"), "chart_purple"),
        (_pullback("H1"), "chart_light_blue"),
    ),
)
def test_feed_pullback_reason_text_keeps_source_colours_over_focus(
    qapp, theme_name, alert, color_name
):
    """Feed source-bar colour has priority over its Focus membership dress."""
    from ui import theme
    from ui.widgets.alert_feed_item import AlertFeedItem

    _apply_theme(qapp, theme_name)
    item = AlertFeedItem(alert, focus_category="swing")
    item.show()
    qapp.processEvents()
    try:
        assert _foreground(item.trigger_label) == theme.color(color_name, theme_name).lower()
    finally:
        item.close()
        item.deleteLater()
