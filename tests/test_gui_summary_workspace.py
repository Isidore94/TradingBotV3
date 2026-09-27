"""The manual AI workspace keeps setup beside a readable result."""

from pathlib import Path
import os
import sys

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint  # noqa: E402
from PySide6.QtGui import QFontDatabase  # noqa: E402
from PySide6.QtWidgets import QApplication, QLineEdit  # noqa: E402


@pytest.mark.parametrize("size", [(1920, 1080), (2560, 1440), (3840, 2160)])
@pytest.mark.parametrize("appearance", ["dark", "light"])
def test_summary_keeps_setup_and_results_in_reach(monkeypatch, tmp_path, size, appearance):
    from ai_credentials import AiCredentialVault, MemoryCredentialBackend
    from ui import theme
    from ui.panels import ai_summary_panel

    app = QApplication.instance() or QApplication([])
    previous_theme, previous_scale = theme.active_theme(), theme.active_scale()
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
    font_id = QFontDatabase.addApplicationFont(str(font_path)) if font_path.is_file() else -1
    monkeypatch.setattr(ai_summary_panel.AiSummaryPanel, "refresh_gates", lambda self: None)
    monkeypatch.setattr(ai_summary_panel, "build_evidence_package", lambda *a, **k: pytest.fail("Layout built evidence"))
    monkeypatch.setattr(ai_summary_panel, "request_ai_summary", lambda *a, **k: pytest.fail("Layout called a model"))
    panel = None
    try:
        stylesheet = theme.build_stylesheet(appearance, scale=1.0)
        theme._ACTIVE_THEME = appearance
        panel = ai_summary_panel.AiSummaryPanel(
            credential_vault=AiCredentialVault(MemoryCredentialBackend(), environ={}),
            output_dir=tmp_path,
        )
        panel.setStyleSheet(stylesheet)
        panel.resize(*size)
        panel.show()
        app.processEvents()
        assert panel.width() == size[0]
        assert panel.tabs.width() > size[0] * .6
        viewport = panel.setup_scroll.viewport()
        assert viewport.rect().contains(panel.generate_button.mapTo(viewport, panel.generate_button.rect().bottomRight()))
        assert panel.key_controls.isHidden()
        panel.key_toggle.click()
        assert panel.key_controls.isVisible()
        assert panel.key_input.echoMode() == QLineEdit.Password
        panel.model_input.setText("unsaved-model-draft")
        panel.key_toggle.click()
        for index in (1, 0):
            panel.tabs.setCurrentIndex(index)
        assert panel.model_input.text() == "unsaved-model-draft"
        assert panel._last_evidence is None
        assert panel._run_thread is None
        assert panel.daily_review_button.isVisible()
        assert panel.rect().contains(panel.daily_review_button.mapTo(panel, QPoint(0, 0)))
        panel.model_input.setText("Select a model")
        app.processEvents()
        assert panel.grab().save(str(tmp_path / f"ai-summary-{appearance}-{size[0]}.png"))
    finally:
        if panel is not None:
            panel.close()
            panel.deleteLater()
            app.sendPostedEvents(panel, QEvent.DeferredDelete)
        theme._ACTIVE_THEME, theme._ACTIVE_SCALE = previous_theme, previous_scale
        if font_id >= 0:
            QFontDatabase.removeApplicationFont(font_id)
