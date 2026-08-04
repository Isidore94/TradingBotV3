import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _focus_service(tmp_path):
    from focus_picks import FocusPickStore
    from ui.services.focus_service import FocusService

    return FocusService(
        FocusPickStore(
            focus_longs_path=tmp_path / "focus_longs.txt",
            focus_shorts_path=tmp_path / "focus_shorts.txt",
            longs_path=tmp_path / "longs.txt",
            shorts_path=tmp_path / "shorts.txt",
            membership_path=tmp_path / "focus_pick_membership.json",
        )
    )


def _alert_service(monkeypatch, tmp_path, *, engine_enabled=True):
    import price_alerts
    from ui.services.price_alert_service import PriceAlertService

    path = tmp_path / "price_alerts.json"
    original_load = price_alerts.load_price_alerts
    original_save = price_alerts.save_price_alerts
    monkeypatch.setattr(price_alerts, "load_price_alerts", lambda: original_load(path))
    monkeypatch.setattr(price_alerts, "save_price_alerts", lambda entries: original_save(entries, path))
    return PriceAlertService(engine_enabled=engine_enabled), path


def test_focus_board_round_trip_and_half_filled_side(monkeypatch, tmp_path):
    from ui.widgets.price_alert_board import PriceAlertBoard

    service, _path = _alert_service(monkeypatch, tmp_path)
    board = PriceAlertBoard(service, _focus_service(tmp_path))
    try:
        board.symbol_input.setEditText("nvda")
        board.above_input.setText("150.25")
        board._save_input()

        rows = service.entries()
        assert len(rows) == 1
        assert rows[0]["symbol"] == "NVDA"
        assert rows[0]["above"] == 150.25 and rows[0]["armed_above"] is True
        assert rows[0]["below"] is None and rows[0]["armed_below"] is False
        assert board.table.rowCount() == 1
    finally:
        service.shutdown()
        board.close()


def test_unchanged_fired_level_stays_disarmed_but_changed_level_rearms(monkeypatch, tmp_path):
    from ui.widgets.price_alert_board import PriceAlertBoard

    service, _path = _alert_service(monkeypatch, tmp_path)
    service.save_entries(
        [{"symbol": "SPY", "above": 600, "armed_above": False, "history": []}]
    )
    board = PriceAlertBoard(service, _focus_service(tmp_path))
    try:
        board.symbol_input.setEditText("SPY")
        board.above_input.setText("600")
        board._save_input()
        assert service.entries()[0]["armed_above"] is False

        board.symbol_input.setEditText("SPY")
        board.above_input.setText("601")
        board._save_input()
        assert service.entries()[0]["armed_above"] is True
    finally:
        service.shutdown()
        board.close()


def test_removing_focus_pick_does_not_remove_price_alert(monkeypatch, tmp_path):
    from ui.widgets.price_alert_board import PriceAlertBoard

    focus = _focus_service(tmp_path)
    focus.add("AAPL", "long", category="swing")
    service, _path = _alert_service(monkeypatch, tmp_path)
    board = PriceAlertBoard(service, focus)
    try:
        board.symbol_input.setEditText("AAPL")
        board.below_input.setText("190")
        board._save_input()
        focus.remove("AAPL", "long")

        assert [entry["symbol"] for entry in service.entries()] == ["AAPL"]
    finally:
        service.shutdown()
        board.close()


def test_satellite_board_is_read_only(monkeypatch, tmp_path):
    from ui.widgets.price_alert_board import PriceAlertBoard

    service, _path = _alert_service(monkeypatch, tmp_path, engine_enabled=False)
    board = PriceAlertBoard(service, _focus_service(tmp_path), read_only=True)
    try:
        assert not board.save_button.isEnabled()
        assert "Read-only" in board.status_label.text()
        board.symbol_input.setEditText("TSLA")
        board.above_input.setText("500")
        board._save_input()
        assert service.entries() == []
    finally:
        service.shutdown()
        board.close()
