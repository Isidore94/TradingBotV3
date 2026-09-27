"""General Settings groups existing controls without changing their save paths."""

import os
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint  # noqa: E402
from PySide6.QtGui import QFontDatabase  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.mark.parametrize("width,height", [(1920, 1080), (2560, 1440), (3840, 2160)])
def test_settings_groups_fit_and_keep_all_controls(tmp_path, width, height):
    from ui import theme
    from ui.panels.settings_panel import SettingsPanel
    from ui.state import UiState

    app = QApplication.instance() or QApplication([])
    old_theme, old_scale = theme.active_theme(), theme.active_scale()
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
    font_id = QFontDatabase.addApplicationFont(str(font_path)) if font_path.is_file() else -1
    panel = None
    try:
        stylesheet = theme.build_stylesheet("dark", scale=1.0)
        theme._ACTIVE_THEME = "dark"
        panel = SettingsPanel(UiState())
        panel.setStyleSheet(stylesheet)
        panel.resize(width, height)
        panel.show()
        app.processEvents()
        viewport = panel.settings_tabs.currentWidget().viewport()
        expected = {
            "Appearance": (panel.theme_input, panel.mode_input, panel.desk_layout_input, panel.explain_input, panel.compact_input, panel.ui_scale_input),
            "Risk and Mentor": (panel.risk_input, panel.trade_mentor_input, panel.trade_mentor_next, panel.trade_mentor_pause),
            "Storage": (panel.data_dir_label, panel.source_label, panel.warm_button),
        }
        assert panel.width() == width
        for title, controls in expected.items():
            for control in controls:
                assert panel.general_groups[title].isAncestorOf(control)
                assert control.isVisible()
                assert viewport.rect().contains(control.mapTo(viewport, QPoint(0, 0)))
                assert viewport.rect().contains(control.mapTo(viewport, control.rect().bottomRight()))
        assert panel.grab().save(str(tmp_path / f"settings-{width}.png"))
    finally:
        if panel is not None:
            panel.close()
            panel.deleteLater()
            app.sendPostedEvents(panel, QEvent.DeferredDelete)
        theme._ACTIVE_THEME, theme._ACTIVE_SCALE = old_theme, old_scale
        if font_id >= 0:
            QFontDatabase.removeApplicationFont(font_id)
