import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_industry_table_uses_numeric_sorting_for_strength():
    _qapp()
    from PySide6.QtCore import Qt

    from ui.panels.industry_panel import INDUSTRY_COLUMNS, _fill_table, _make_table

    table = _make_table(INDUSTRY_COLUMNS)
    _fill_table(
        table,
        INDUSTRY_COLUMNS,
        [
            {"industry": "Two", "rs_score": "2.0"},
            {"industry": "Ten", "rs_score": "10.0"},
            {"industry": "Negative", "rs_score": "-1.0"},
        ],
    )
    rs_column = next(i for i, (key, _label) in enumerate(INDUSTRY_COLUMNS) if key == "rs_score")
    name_column = next(i for i, (key, _label) in enumerate(INDUSTRY_COLUMNS) if key == "industry")

    table.sortItems(rs_column, Qt.SortOrder.DescendingOrder)
    assert [table.item(row, name_column).text() for row in range(3)] == ["Ten", "Two", "Negative"]
    table.sortItems(rs_column, Qt.SortOrder.AscendingOrder)
    assert [table.item(row, name_column).text() for row in range(3)] == ["Negative", "Two", "Ten"]


def test_sector_etf_click_opens_cached_snapshot(monkeypatch):
    _qapp()
    from ui.panels.industry_panel import (
        IndustryPanel,
        SECTOR_COLUMNS,
        _fill_table,
    )
    import ui.widgets.symbol_snapshot_dialog as snapshot_dialog

    bot = object()

    class _BounceService:
        def current_bot(self):
            return bot

    panel = IndustryPanel()
    panel.service.shutdown()
    panel.set_bounce_service(_BounceService())
    _fill_table(
        panel.sector_table,
        SECTOR_COLUMNS,
        [{"etf": "XLK", "sector": "Technology", "rs_score": "2.5"}],
    )
    calls = []
    monkeypatch.setattr(
        snapshot_dialog,
        "show_symbol_snapshot",
        lambda owner, symbol, **kwargs: calls.append(
            (owner, symbol, kwargs.get("bot"), kwargs.get("side"))
        ),
    )
    etf_column = next(
        index for index, (key, _label) in enumerate(SECTOR_COLUMNS) if key == "etf"
    )
    panel.sector_table.cellClicked.emit(0, etf_column)
    assert calls == [(panel, "XLK", bot, "LONG")]
    panel.sector_table.cellClicked.emit(0, etf_column + 1)
    assert len(calls) == 1


def test_industry_snapshot_freshness_uses_the_older_file(tmp_path):
    from ui.services.industry_board_service import (
        industry_refresh_due,
        inspect_industry_snapshot,
    )

    sector = tmp_path / "sector.csv"
    industry = tmp_path / "industry.csv"
    sector.write_text("sector\n", encoding="utf-8")
    industry.write_text("industry\n", encoding="utf-8")
    now = datetime(2026, 7, 14, 10, 0)
    fresh_time = (now - timedelta(minutes=10)).timestamp()
    stale_time = (now - timedelta(minutes=70)).timestamp()
    os.utime(sector, (fresh_time, fresh_time))
    os.utime(industry, (stale_time, stale_time))

    snapshot = inspect_industry_snapshot(
        sector_path=sector,
        industry_path=industry,
        now=now,
        fresh_for_seconds=3600,
    )

    assert snapshot["state"] == "stale"
    assert snapshot["as_of"] == datetime.fromtimestamp(stale_time)
    assert industry_refresh_due(snapshot)


def test_industry_snapshot_marks_partial_pair_due(tmp_path):
    from ui.services.industry_board_service import (
        industry_refresh_due,
        inspect_industry_snapshot,
    )

    sector = tmp_path / "sector.csv"
    sector.write_text("sector\n", encoding="utf-8")
    snapshot = inspect_industry_snapshot(
        sector_path=sector,
        industry_path=tmp_path / "missing.csv",
    )
    assert snapshot["state"] == "partial"
    assert industry_refresh_due(snapshot)


def test_industry_service_coalesces_overlapping_refreshes(tmp_path):
    app = _qapp()
    from ui.services.industry_board_service import IndustryBoardService

    sector = tmp_path / "sector.csv"
    industry = tmp_path / "industry.csv"
    state = tmp_path / "state.json"
    release = threading.Event()
    calls = []

    def runner(*, write_outputs):
        calls.append(write_outputs)
        release.wait(timeout=2)
        sector.write_text("sector\n", encoding="utf-8")
        industry.write_text("industry\n", encoding="utf-8")
        return {
            "sector_rows": [{"sector": "Technology"}],
            "industry_rows": [{"industry": "Semiconductors"}],
            "symbol_count": 2,
        }

    service = IndustryBoardService(
        scan_runner=runner,
        sector_path=sector,
        industry_path=industry,
        state_path=state,
        startup_delay_ms=60_000,
    )
    assert service.request_refresh(force=True)
    assert not service.request_refresh(force=True)
    release.set()
    deadline = time.monotonic() + 2
    while service.running and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()

    assert calls == [True]
    assert not service.running
    assert state.exists()
    service.shutdown()
