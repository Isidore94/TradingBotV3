"""P1-6 6b: the D1 detail plan as setups-table columns, with the stale flag inline.

Seen to fail before the change: `entry_plan` did not exist, the model had no
plan columns, and the compact profile had nothing to hide.
"""

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

import entry_plan  # noqa: E402
import setup_docs  # noqa: E402

LEVELS = {
    "AAA": {
        "vwap": 97.0,
        "bands": {"LOWER_1": 95.0, "UPPER_1": 104.0, "UPPER_2": 110.0, "UPPER_3": 115.0},
        "anchor_date": "2026-08-01",
        "atr20": 2.5,
        "last_close": 100.0,
        "side": "LONG",
    }
}


def _plan(**overrides):
    kwargs = dict(symbol="AAA", side="LONG", setup_family="", setup_tags=(), last_close=100.0,
                  levels_by_symbol=LEVELS)
    kwargs.update(overrides)
    return entry_plan.plan_for_row(**kwargs)


def test_the_plan_projects_the_detail_panes_numbers():
    plan = _plan()
    direct = setup_docs.build_trade_plan(
        side="LONG", setup_family="general", favorite_signals=[], bands=LEVELS["AAA"]["bands"],
        vwap=97.0, atr20=2.5, last_close=100.0,
    )
    assert plan["entry"] == direct["entry_reference"] == 100.0
    assert plan["stop"] == direct["stop_price"] == 95.0
    assert plan["tp1"] == direct["partial_price"] == 110.0
    assert plan["tp1_r"] == pytest.approx(direct["partial_r"]) == pytest.approx(2.0)
    assert plan["stale"] is False
    assert entry_plan.plan_cells(plan, 500) == {
        "plan_entry": "100.00", "plan_stop": "95.00", "plan_tp1": "110.00",
        "plan_r": "+2.0R", "plan_shares": "100",
    }


def test_a_first_band_bounce_uses_the_same_stop_the_pane_does():
    plan = _plan(setup_tags=("BOUNCE_UPPER_1",))
    assert plan["stop"] == 97.0 and plan["stop_label"] == "AVWAPE"


def test_price_past_the_stop_is_stale_inline_and_never_sized():
    plan = _plan(last_close=94.0)
    assert plan["stale"] is True
    cells = entry_plan.plan_cells(plan, 500)
    assert cells["plan_r"] == "stale"
    assert cells["plan_shares"] == ""


def test_an_unknown_stop_is_blank_not_stale():
    levels = {"AAA": {**LEVELS["AAA"], "bands": {"UPPER_2": 110.0}}}
    plan = _plan(levels_by_symbol=levels)
    assert plan["stop"] is None and plan["stale"] is False
    cells = entry_plan.plan_cells(plan, 500)
    assert cells["plan_stop"] == "" and cells["plan_shares"] == "" and cells["plan_r"] == ""


def test_no_level_data_is_no_plan():
    assert _plan(symbol="ZZZ") is None
    assert set(entry_plan.plan_cells(None).values()) == {""}
    assert entry_plan.plan_line(None) == ""


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _rows():
    from ui.models.setup import SetupRow

    return [
        SetupRow(symbol="AAA", side="LONG", bucket="favorite_setup", key_level="$95 AVWAPE",
                 raw={"setup_family": "", "last_close": 100.0}),
        SetupRow(symbol="ZZZ", side="LONG", bucket="favorite_setup", key_level="$10", raw={}),
    ]


def test_the_model_shows_plan_cells_and_the_key_level_tooltip_carries_the_plan(app):
    from PySide6.QtCore import Qt
    from ui.models.setup_table_model import SetupTableModel

    model = SetupTableModel(_rows())
    columns = [key for key, _label in model.COLUMNS]
    for key in model.PLAN_COLUMNS:
        assert key in columns
    model.set_plan_context(LEVELS, 500)

    def cell(row, key, role=Qt.ItemDataRole.DisplayRole):
        return model.data(model.index(row, columns.index(key)), role)

    assert [cell(0, key) for key in model.PLAN_COLUMNS] == ["100.00", "95.00", "110.00", "+2.0R", "100"]
    assert [cell(1, key) for key in model.PLAN_COLUMNS] == ["", "", "", "", ""]
    tooltip = cell(0, "key_level", Qt.ItemDataRole.ToolTipRole)
    assert tooltip.startswith("$95 AVWAPE")
    assert "Plan: entry 100.00 · stop 95.00 (LOWER_1) · TP1 110.00 (+2.0R) · 100 sh" in tooltip
    assert cell(1, "key_level", Qt.ItemDataRole.ToolTipRole) == "$10"
    assert model.has_plans()


def test_compact_hides_the_plan_columns_full_shows_them(app, monkeypatch):
    from ui.panels.master_avwap_panel import MasterAvwapPanel
    from ui.services import ai_state_levels

    monkeypatch.setattr(ai_state_levels, "cached_symbol_levels", lambda: LEVELS)
    monkeypatch.setattr(entry_plan, "risk_per_trade_dollars", lambda: 250.0)
    panel = MasterAvwapPanel()
    try:
        panel.resize(1640, 980)
        panel.set_rows(_rows())
        panel.set_column_profile("compact")
        keys = [key for key, _label in panel.model.COLUMNS]
        for key in panel.model.PLAN_COLUMNS:
            assert panel.table.isColumnHidden(keys.index(key)), f"{key} shown in compact"
        panel.set_column_profile("full")
        for key in panel.model.PLAN_COLUMNS:
            assert not panel.table.isColumnHidden(keys.index(key)), f"{key} hidden in full"
        assert panel.model.risk_dollars() == 250.0
        panel.set_risk_per_trade("")
        assert panel.model.risk_dollars() is None
    finally:
        panel.deleteLater()
