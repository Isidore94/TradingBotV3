"""Desk snappiness packet 1 item 2: bounded column fits, no no-op board reloads.

The 2026-08-31 stall log charged `measure_column_widths` (via
`resizeColumnsToContents`, unbounded) 9.6 minutes in one day with single stalls
of 85 s, and the Industry Board's 60 s check tick emitted `snapshotChanged`
even when `snapshot_id` had not moved - the panel then re-read both CSVs and
rebuilt + re-measured both tables for a board that had not changed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


# ---------------------------------------------------------------------------
# the measurement bound
# ---------------------------------------------------------------------------


def test_column_fit_applies_the_precision_cap():
    """The fit is bounded, and by the RIGHT header: Qt's `sizeHintForColumn`
    walks rows bounded by the VERTICAL header's precision."""
    _qapp()
    from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

    from ui.widgets.data_table import MEASURE_PRECISION_ROWS, measure_column_widths

    table = QTableWidget(1_000, 3)
    for row in range(0, 1_000, 25):
        for column in range(3):
            table.setItem(row, column, QTableWidgetItem(f"row {row} column {column}"))

    widths = measure_column_widths(table)

    assert len(widths) == 3
    assert table.verticalHeader().resizeContentsPrecision() == MEASURE_PRECISION_ROWS
    assert table.horizontalHeader().resizeContentsPrecision() == MEASURE_PRECISION_ROWS


def test_the_shared_width_rule_still_measures_through_the_bounded_fit():
    """`apply_width_rule` funnels through `measure_column_widths`; the bound
    must not change what a small table measures."""
    _qapp()
    from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

    from ui.widgets.data_table import apply_width_rule_to_table_widget

    table = QTableWidget(3, 2)
    for row, (name, value) in enumerate((("alpha", "1"), ("beta_longer_name", "2"), ("gamma", "3"))):
        table.setItem(row, 0, QTableWidgetItem(name))
        table.setItem(row, 1, QTableWidgetItem(value))
    apply_width_rule_to_table_widget(table)
    assert table.horizontalHeader().sectionSize(0) > 0


# ---------------------------------------------------------------------------
# the Industry Board no-op refresh
# ---------------------------------------------------------------------------


def _board_service(tmp_path):
    from ui.services.industry_board_service import IndustryBoardService

    sector = tmp_path / "sector.csv"
    industry = tmp_path / "industry.csv"
    sector.write_text("sector\nTechnology\n", encoding="utf-8")
    industry.write_text("industry\nSemiconductors\n", encoding="utf-8")
    return IndustryBoardService(
        scan_runner=lambda **_kwargs: {},
        sector_path=sector,
        industry_path=industry,
        state_path=tmp_path / "state.json",
        startup_delay_ms=60_000,
    )


def test_the_check_tick_does_not_emit_for_an_unchanged_snapshot(tmp_path):
    _qapp()
    service = _board_service(tmp_path)
    emits: list[dict] = []
    service.snapshotChanged.connect(emits.append)
    try:
        assert service.refresh_if_due() is False  # fresh files: no refresh due
        assert len(emits) == 1, "the first tick carries the first snapshot"
        assert service.refresh_if_due() is False
        assert service.refresh_if_due() is False
        assert len(emits) == 1, "an unchanged snapshot_id must not re-emit"
    finally:
        service.shutdown()


def test_the_check_tick_emits_once_when_the_board_moves(tmp_path):
    _qapp()
    service = _board_service(tmp_path)
    emits: list[dict] = []
    service.snapshotChanged.connect(emits.append)
    try:
        service.refresh_if_due()
        first_id = emits[0]["snapshot_id"]
        sector = tmp_path / "sector.csv"
        sector.write_text("sector\nTechnology\nEnergy\n", encoding="utf-8")
        stat = sector.stat()
        os.utime(sector, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        service.refresh_if_due()
        assert len(emits) == 2, "a moved board emits exactly once"
        assert emits[1]["snapshot_id"] != first_id
    finally:
        service.shutdown()


def test_the_panel_skips_a_reload_for_a_snapshot_it_already_rendered():
    """Belt-and-braces on the panel side: whatever emits, an unchanged id is
    never re-read from disk."""
    _qapp()
    from ui.panels.industry_panel import IndustryPanel

    panel = IndustryPanel()
    panel.service.shutdown()
    try:
        reloads: list[object] = []
        panel.reload_from_disk = lambda snapshot=None: reloads.append(snapshot)
        panel._last_rendered_snapshot_id = "abc123def456"

        panel._on_snapshot_changed({"snapshot_id": "abc123def456"})
        assert reloads == [], "an unchanged snapshot_id must not reload the CSVs"

        panel._on_snapshot_changed({"snapshot_id": "999999999999"})
        assert len(reloads) == 1, "a changed snapshot_id reloads exactly once"
    finally:
        panel.close()
        panel.deleteLater()


# ---------------------------------------------------------------------------
# the two ResizeToContents header modes
# ---------------------------------------------------------------------------


def test_health_detail_tables_fit_once_then_stay_interactive():
    _qapp()
    from PySide6.QtWidgets import QHeaderView

    from ui.panels.health_panel import _fill, _table

    table = _table(("State", "Job", "Detail"), stretch_column=2)
    header = table.horizontalHeader()
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Interactive
    assert header.sectionResizeMode(1) == QHeaderView.ResizeMode.Interactive
    assert header.sectionResizeMode(2) == QHeaderView.ResizeMode.Stretch

    _fill(table, [("COMPLETED", "swing_scan", "run-1")])
    # The fit ran (its precision cap is the witness) and left the modes alone.
    from ui.widgets.data_table import MEASURE_PRECISION_ROWS

    assert table.verticalHeader().resizeContentsPrecision() == MEASURE_PRECISION_ROWS
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Interactive


def test_price_alerts_table_is_interactive_not_resize_to_contents(monkeypatch):
    _qapp()
    from PySide6.QtWidgets import QHeaderView

    from ui.panels import price_alerts_panel as panel_module
    from ui.panels.price_alerts_panel import _COLUMNS, PriceAlertsPanel

    monkeypatch.setattr(panel_module, "save_local_setting", lambda *_a, **_k: None)
    panel = PriceAlertsPanel()
    try:
        header = panel.table.horizontalHeader()
        for column, name in enumerate(_COLUMNS):
            expected = (
                QHeaderView.ResizeMode.Stretch
                if name in ("Note", "Last Trigger")
                else QHeaderView.ResizeMode.Interactive
            )
            assert header.sectionResizeMode(column) == expected, name
    finally:
        panel.close()
        panel.deleteLater()
