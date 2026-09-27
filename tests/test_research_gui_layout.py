"""Research's grouped navigation and full-size Results workspace."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="Research is a Qt panel")

from PySide6.QtCore import QEvent, QPoint, QRect  # noqa: E402
from PySide6.QtGui import QFont, QFontDatabase  # noqa: E402
from PySide6.QtWidgets import QApplication, QScrollArea, QTabWidget  # noqa: E402

import test_g5_research_results as _g5  # noqa: E402

snapshot_payload = _g5.snapshot_payload


@pytest.fixture(autouse=True)
def _application():
    app = QApplication.instance() or QApplication([])
    yield app


def _quiet_results_reads(monkeypatch) -> None:
    """Keep these layout checks away from local stores and background reads."""
    from ui.panels import research_results_panel

    monkeypatch.setattr(
        research_results_panel.ResearchResultsPanel, "refresh", lambda _self: None
    )
    monkeypatch.setattr(
        research_results_panel.ResearchResultsPanel,
        "refresh_report",
        lambda _self: None,
    )


def test_research_destinations_are_grouped_and_keep_tab_routing(monkeypatch):
    _quiet_results_reads(monkeypatch)
    from ui.panels.research_panel import ResearchPanel

    panel = ResearchPanel(None)
    try:
        assert isinstance(panel.tabs, QTabWidget)
        assert panel.tabs.count() == 12
        assert panel.tabs.currentWidget() is panel.results_panel
        assert panel.tabs.tabBar().isHidden()
        assert {
            group: tuple(button.text() for button in buttons)
            for group, buttons in panel.local_nav_groups.items()
        } == {
            "Results": ("Results",),
            "Setups": ("Setup Tracker", "Setup Playbook", "Setup keys"),
            "Studies": (
                "Move Forensics",
                "Day Trade Tracker",
                "Long lab",
                "Retest entry",
            ),
            "Tools": (
                "Master AVWAP Market Prep",
                "Ticker Lookup",
                "Price Alerts",
            ),
            "Data": ("Research Warehouse",),
        }
        assert set(panel.local_nav_buttons) == {
            "Results",
            "Setup Tracker",
            "Setup Playbook",
            "Setup keys",
            "Move Forensics",
            "Day Trade Tracker",
            "Long lab",
            "Retest entry",
            "Master AVWAP Market Prep",
            "Ticker Lookup",
            "Price Alerts",
            "Research Warehouse",
        }

        panel.local_nav_buttons["Setup Tracker"].click()
        assert panel.tabs.currentWidget() is panel.setup_tracker_panel
        assert panel.local_nav_buttons["Setup Tracker"].isChecked()

        # Existing routes used by the Desk and other callers remain valid.
        panel.tabs.setCurrentWidget(panel.results_panel)
        assert panel.local_nav_buttons["Results"].isChecked()
        panel.show_setup_tracker()
        assert panel.tabs.currentWidget() is panel.setup_tracker_panel
        assert panel.local_nav_buttons["Setup Tracker"].isChecked()
    finally:
        panel.shutdown()
        panel.deleteLater()
        QApplication.sendPostedEvents(panel, QEvent.DeferredDelete)
        QApplication.processEvents()


def test_results_keeps_every_section_in_a_scrollable_primary_workspace(monkeypatch):
    _quiet_results_reads(monkeypatch)
    from ui.panels.research_results_panel import ResearchResultsPanel

    app = QApplication.instance() or QApplication([])
    panel = ResearchResultsPanel()
    panel.resize(1280, 760)
    panel.show()
    try:
        app.processEvents()
        assert isinstance(panel.content_scroll_area, QScrollArea)
        assert panel.content_scroll_area.widgetResizable()
        assert panel.content_scroll_area.widget() is panel.content_widget
        assert panel.content_widget.isAncestorOf(panel.shortlist)
        assert panel.content_widget.isAncestorOf(panel.explanation_view)
        assert panel.content_widget.isAncestorOf(panel.looking_back_view)
        assert panel.content_widget.isAncestorOf(panel.review_table)
        assert panel.content_scroll_area.verticalScrollBar().maximum() > 0

        assert panel.reading_guide_button.isCheckable()
        assert panel.reading_guide.isHidden()
        initial_selection = panel.selection()
        panel.reading_guide_button.click()
        assert not panel.reading_guide.isHidden()
        assert panel.selection() == initial_selection
    finally:
        panel.shutdown()
        panel.close()
        panel.deleteLater()
        QApplication.sendPostedEvents(panel, QEvent.DeferredDelete)
        app.processEvents()


def test_populated_results_has_a_themed_4k_layout_screenshot(
    monkeypatch, snapshot_payload, tmp_path
):
    _quiet_results_reads(monkeypatch)
    from ui import theme
    import research_results
    from ui.panels.research_panel import ResearchPanel
    from ui.models.tracker_table_model import ROW_ROLE

    app = QApplication.instance() or QApplication([])
    old_scale = theme.active_scale()
    font_id = -1
    panel_font = None
    panel = None
    try:
        font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
        if font_path.is_file():
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            if font_id >= 0:
                families = QFontDatabase.applicationFontFamilies(font_id)
                if families:
                    panel_font = QFont(families[0], 10)
        stylesheet = theme.build_stylesheet("dark", scale=1.0)

        panel = ResearchPanel(None)
        panel.setStyleSheet(stylesheet)
        if panel_font is not None:
            panel.setFont(panel_font)
        results = panel.results_panel
        results._selection = ("bot", "swing", "recent")
        results._apply_selection_to_buttons()
        view = research_results.build_results_view(
            population="bot",
            horizon="swing",
            window="recent",
            snapshot=snapshot_payload,
        )
        results._render(view)
        results._render_looking_back(None)
        panel.resize(2560, 1440)
        panel.show()
        app.processEvents()

        model = results.shortlist.model()
        assert model is not None and model.rowCount() > 0
        first = model.index(0, 0)
        results.shortlist.setCurrentIndex(first)
        row = first.data(ROW_ROLE)
        assert isinstance(row, dict)
        results._on_row_clicked(first)
        app.processEvents()
        assert results.band_summary_heading.y() > results.detail_splitter.geometry().bottom(), (
            results.band_summary_heading.geometry(), results.detail_splitter.geometry()
        )

        screenshot = tmp_path / "research-results-4k.png"
        assert panel.grab().save(str(screenshot)), screenshot
        assert screenshot.is_file()
        print(f"Research 4K layout screenshot: {screenshot}")

        # Check the most crowded scope inside the full Research shell, with its
        # local navigator taking space beside Results on a 1920px logical screen.
        panel.resize(1920, 1080)
        results._selection = ("mine", "swing", "custom")
        results._apply_selection_to_buttons()
        app.processEvents()
        assert panel.width() == 1920
        viewport = results.content_scroll_area.viewport()
        assert results.content_widget.width() <= viewport.width()
        assert results.band_summary_heading.y() > results.detail_splitter.geometry().bottom()
        date_rect = QRect(
            results.custom_end.mapTo(viewport, QPoint(0, 0)),
            results.custom_end.size(),
        )
        assert results.custom_end.isVisible()
        assert viewport.rect().contains(date_rect), (
            "the final custom-window date control is clipped at 1920 logical "
            f"pixels: viewport={viewport.rect()}, control={date_rect}"
        )

        for width in (2560, 3840):
            panel.resize(width, 1440)
            app.processEvents()
            viewport = results.content_scroll_area.viewport()
            assert results.content_widget.width() <= viewport.width(), (
                f"Results content exceeds its viewport at {width}px: "
                f"content={results.content_widget.width()}, "
                f"viewport={viewport.width()}"
            )
            assert results.band_summary_heading.y() > results.detail_splitter.geometry().bottom()
        results.reading_guide_button.click()
        app.processEvents()
        assert results.band_summary_heading.y() > results.detail_splitter.geometry().bottom()
    finally:
        if panel is not None:
            panel.shutdown()
            panel.close()
            panel.deleteLater()
            app.sendPostedEvents(panel, QEvent.DeferredDelete)
        if font_id >= 0:
            QFontDatabase.removeApplicationFont(font_id)
        theme._ACTIVE_SCALE = old_scale
