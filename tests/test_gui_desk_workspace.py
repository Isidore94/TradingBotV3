"""The native Desk stays usable at the full-panel sizes in the GUI guide."""

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.qt

from PySide6.QtCore import QEvent, QThread  # noqa: E402
from PySide6.QtGui import QFontDatabase  # noqa: E402

from test_qt_compact_desk import _app, _pump, _quiet_setup_reads  # noqa: E402
from ui import theme  # noqa: E402


@pytest.fixture(scope="module")
def desk_window():
    from ui import app as app_module
    from ui.app import MainWindow
    from ui.state import UiState

    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
    font_id = QFontDatabase.addApplicationFont(str(font_path)) if font_path.is_file() else -1
    with pytest.MonkeyPatch.context() as patch:
        _quiet_setup_reads(patch)
        patch.setattr(theme, "_ACTIVE_THEME", "dark")
        patch.setattr(theme, "_ACTIVE_SCALE", 1.0)
        # Theme only this window, not widgets retained by unrelated Qt tests.
        patch.setattr(app_module, "apply_theme", lambda *a, **k: None)
        window = MainWindow(UiState(workspace_mode="workspace", desk_layout="compact", ui_scale="1.00"))
        window.setStyleSheet(theme.build_stylesheet("dark", scale=1.0))
        window.show()
        _pump(20)
        window.movers_service.shutdown()
        try:
            yield window
        finally:
            window.close()
            for worker in window.findChildren(QThread):
                assert worker.wait(10000), "Desk visual-test worker did not stop"
            window.deleteLater()
            _app.sendPostedEvents(window, QEvent.DeferredDelete)
            if font_id >= 0:
                QFontDatabase.removeApplicationFont(font_id)


@pytest.mark.parametrize("width,height", [(1920, 1080), (2560, 1440), (3840, 2160)])
def test_native_desk_workspace_at_panel_sizes(desk_window, tmp_path, monkeypatch, width, height):
    from ui.models.bounce import BounceAlert
    from ui.models.setup import SetupRow

    window = desk_window
    try:
        window._select_page(0)
        window.resize(width, height)
        _pump(10)
        assert window.width() == width
        assert window.page_tab_row.isVisible()
        assert window.trading_panel.alert_center.chart_review.isVisible()
        assert window.grab().save(str(tmp_path / f"desk-{width}.png"))
        desk = window.trading_panel
        review = desk.alert_center.chart_review
        monkeypatch.setattr(review.snapshot, "set_symbol", lambda *a, **kw: None)
        review.set_alert(BounceAlert(
            time_text="09:31:00", symbol="DEMO", side="LONG", trigger="Layout test data",
            timeframe="D1", tag="d1_flag", raw_text="Layout test data", is_d1=True, payload={},
        ))
        bars = [
            {"dt": datetime(2026, 8, 1) + timedelta(days=index), "open": 100 + index * .3,
             "high": 101 + index * .3, "low": 99 + index * .3, "close": 100.5 + index * .3,
             "volume": 10000}
            for index in range(40)
        ]
        review.snapshot.d1_chart.set_data(bars, [], timeframe="d1")
        intraday = [dict(bar, dt=datetime(2026, 9, 25, 9, 30) + timedelta(minutes=index * 5))
                    for index, bar in enumerate(bars)]
        review.snapshot.m5_chart.set_data(intraday, [], timeframe="m5")
        desk.master_panel.set_rows([
            SetupRow(symbol="DEMO", side="LONG", score=72.0, bucket="favorite_setup",
                     sector="Technology", industry="Semiconductors", key_level="AVWAPE 101.25",
                     expected_r=1.8)
        ])
        desk.set_setups_visible(True)
        _pump(10)
        assert window.width() == width
        assert desk.master_workspace.tabs.isVisible()
        assert window.grab().save(str(tmp_path / f"desk-setups-{width}.png"))
    finally:
        window.trading_panel.alert_center.chart_review.clear()
        window.trading_panel.set_setups_visible(False)
