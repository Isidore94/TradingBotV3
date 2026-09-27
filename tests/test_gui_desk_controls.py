"""Compact Desk control placement and Watchlist label clarity."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytestmark = pytest.mark.qt


@pytest.fixture
def desk_gui(tmp_path, monkeypatch):
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    import alert_show_filter
    import longs_market_gate
    import project_paths
    import sector_exclusion
    from ui.panels import alert_center_panel, desk_layout

    settings: dict[str, object] = {
        alert_center_panel.ALERT_SPLIT_KEY: [62, 33, 5],
        alert_center_panel.ALERT_TABS_SPLIT_KEY: [60, 40],
        desk_layout.COMPACT_ALERT_SPLIT_KEY: [80, 20],
    }
    monkeypatch.setattr(
        project_paths,
        "get_local_setting",
        lambda key, default=None: settings.get(key, default),
    )
    monkeypatch.setattr(
        project_paths,
        "save_local_setting",
        lambda key, value: settings.__setitem__(key, value),
    )
    monkeypatch.setattr(
        desk_layout,
        "get_local_setting",
        lambda key, default=None: settings.get(key, default),
    )
    monkeypatch.setattr(
        desk_layout,
        "save_local_setting",
        lambda key, value: settings.__setitem__(key, value),
    )
    monkeypatch.setattr(
        alert_center_panel,
        "get_local_setting",
        lambda key, default=None: settings.get(key, default),
    )
    monkeypatch.setattr(
        alert_center_panel,
        "save_local_settings",
        lambda values: settings.update(values),
    )

    class _Signal:
        def connect(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(
        "ui.services.market_journal_service.shared_journal_service",
        lambda: type("JournalService", (), {"statusChanged": _Signal()})(),
    )
    alert_show_filter.clear_cache()
    sector_exclusion.clear_cache()
    old_gate = longs_market_gate.snapshot()
    longs_market_gate.set_snapshot(
        longs_market_gate.Verdict(
            day="2026-09-27",
            verdict=longs_market_gate.NO,
            reason="market regime needs more room before longs return " * 8,
            since="2026-09-27",
        )
    )
    panel = alert_center_panel.AlertCenterPanel(
        ignored_symbols_path=tmp_path / "ignored.json",
        parked_symbols_path=tmp_path / "parked.json",
        review_events_path=tmp_path / "review_events.jsonl",
    )
    try:
        yield app, panel, settings
    finally:
        panel.set_compact_layout(False)
        panel.close()
        panel.deleteLater()
        app.sendPostedEvents(panel, QEvent.DeferredDelete)
        longs_market_gate.set_snapshot(old_gate)
        alert_show_filter.clear_cache()
        sector_exclusion.clear_cache()


@pytest.mark.parametrize("screen_size", [(1920, 1080), (2560, 1440), (3840, 2160)])
@pytest.mark.parametrize("runner_dips_active", [False, True])
def test_compact_toolbar_fits_and_preserves_filters_and_split(
    desk_gui, screen_size, runner_dips_active
):
    from PySide6.QtCore import Qt
    import alert_show_filter
    from ui.panels.alert_center_panel import ALERT_SPLIT_KEY, ALERT_TABS_SPLIT_KEY

    app, panel, settings = desk_gui
    panel.runner_dips_label.setText("Runner dips: 3 armed (GTLB, DOCU, ANF)")
    panel.runner_dips_label.setVisible(runner_dips_active)
    screen_width, screen_height = screen_size
    # The Alert Center occupies one column of the full-screen Desk.
    panel.resize(round(screen_width * 0.48), screen_height)
    panel.show()
    app.processEvents()

    panel.min_tier_input.setCurrentIndex(min(2, panel.min_tier_input.count() - 1))
    panel.sound_input.setChecked(False)
    panel.hide_sector_input.setChecked(False)
    panel.show_filter_input.setCurrentIndex(
        panel.show_filter_input.findData(alert_show_filter.BEST_NOW)
    )
    panel.first30_input.setChecked(False)
    panel.longs_off_input.setChecked(True)
    before_values = (
        panel.min_tier_input.currentData(),
        panel.sound_input.isChecked(),
        panel.hide_sector_input.isChecked(),
        panel.show_filter_input.currentData(),
        panel.first30_input.isChecked(),
        panel.longs_off_input.isChecked(),
    )
    before_widgets = tuple(panel._control_widgets)
    before_split = panel.splitter.sizes()
    saved_splits = {
        key: list(settings[key])
        for key in (
            ALERT_SPLIT_KEY,
            ALERT_TABS_SPLIT_KEY,
        )
    }

    panel.set_compact_layout(True)
    panel._drawer.expand()
    app.processEvents()
    # A top-level classic window can already have grown for an unwrapped banner.
    # Respect the chart's own minimum; the full-window test covers splitter fit.
    panel.resize(round(screen_width * 0.48), screen_height)
    for _ in range(5):
        app.processEvents()
    column_width = panel.width()
    assert column_width < screen_width
    panel.longs_off_banner.setText(
        "Longs off: " + "market regime needs more room before longs return; " * 24
    )
    panel.longs_off_banner.setVisible(True)
    app.processEvents()

    toolbar = panel._compact_controls_host
    assert panel.width() == column_width
    assert toolbar.objectName() == "DeskAlertControls"
    assert toolbar.isVisible()
    assert panel.tabs.cornerWidget(Qt.Corner.TopRightCorner) is None
    assert panel.rect().contains(toolbar.geometry())
    assert panel.longs_off_banner.wordWrap()
    assert panel._compact_controls_layout.indexOf(panel.longs_off_banner) == 1
    assert panel.longs_off_banner.isVisible()
    assert (
        panel.longs_off_banner.height()
        > panel.longs_off_banner.fontMetrics().lineSpacing()
    )
    for widget in panel._compact_control_widgets:
        assert widget.parent() is toolbar
        if widget is panel.runner_dips_label and not runner_dips_active:
            assert not widget.isVisible()
            continue
        assert toolbar.rect().contains(widget.geometry())
        assert widget.isVisible()

    tab_bar = panel.tabs.tabBar()
    for label in ("Capture", "Journal"):
        index = next(
            i for i in range(panel.tabs.count()) if panel.tabs.tabText(i).startswith(label)
        )
        rect = tab_bar.tabRect(index)
        assert rect.isValid() and tab_bar.rect().contains(rect)

    panel.set_compact_layout(False)
    app.processEvents()
    assert tuple(panel._control_widgets) == before_widgets
    assert all(
        widget.parent() is panel for widget in before_widgets if widget is not None
    )
    assert panel._controls_layout.indexOf(panel.longs_off_banner) >= 0
    assert not panel.longs_off_banner.wordWrap()
    assert panel.runner_dips_label.isVisible() is runner_dips_active
    assert (
        panel.min_tier_input.currentData(),
        panel.sound_input.isChecked(),
        panel.hide_sector_input.isChecked(),
        panel.show_filter_input.currentData(),
        panel.first30_input.isChecked(),
        panel.longs_off_input.isChecked(),
    ) == before_values
    restored_split = panel.splitter.sizes()
    assert len(restored_split) == len(before_split) == 3
    assert sum(restored_split) > 0
    assert restored_split[0] > 0 and restored_split[1] > 0
    assert {key: settings[key] for key in saved_splits} == saved_splits
