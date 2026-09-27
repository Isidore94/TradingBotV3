"""Health keeps the selected check beside its evidence on a wide desk."""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QFontDatabase  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from test_qt_health_panel import _payload  # noqa: E402


@pytest.mark.parametrize("width,height", [(1920, 1080), (2560, 1440), (3840, 2160)])
def test_health_keeps_full_metadata_and_selected_evidence_reachable(monkeypatch, tmp_path, width, height):
    from ui import theme
    from ui.panels.health_panel import HealthPanel

    app = QApplication.instance() or QApplication([])
    old_theme, old_scale = theme.active_theme(), theme.active_scale()
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
    font_id = QFontDatabase.addApplicationFont(str(font_path)) if font_path.is_file() else -1
    monkeypatch.setattr(HealthPanel, "refresh", lambda self: None)
    panel = HealthPanel(refresh_interval_ms=60_000)
    try:
        theme.apply_theme(panel, "dark", scale=1.0)
        payload = _payload()
        payload["ai_night_lines"] = [f"Job {index}: waiting for its next scheduled window" for index in range(12)]
        panel.set_payload(payload)
        panel.resize(width, height)
        panel.show()
        app.processEvents()
        assert panel.width() == width
        assert panel.workspace_splitter.orientation() == Qt.Horizontal
        assert panel.table.height() > height * .5
        panel.table.selectRow(1)
        selected = panel.table.currentRow()
        evidence = panel.details.toPlainText()
        assert evidence
        panel.audit_help_toggle.click()
        assert panel.audit_help.isVisible()
        for index in (1, 2, 0):
            panel.detail_tabs.setCurrentIndex(index)
        assert panel.table.currentRow() == selected
        assert panel.details.toPlainText() == evidence
        assert "Job 11:" in panel.meta_label.text()
        assert panel.meta_label.wordWrap()
        panel.audit_help_toggle.click()
        app.processEvents()
        assert panel.grab().save(str(tmp_path / f"health-{width}.png"))
    finally:
        panel.shutdown()
        panel.close()
        panel.deleteLater()
        app.sendPostedEvents(panel, QEvent.DeferredDelete)
        theme._ACTIVE_THEME, theme._ACTIVE_SCALE = old_theme, old_scale
        if font_id >= 0:
            QFontDatabase.removeApplicationFont(font_id)
