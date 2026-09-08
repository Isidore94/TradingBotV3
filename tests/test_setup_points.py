"""The point system (trader, 2026-09-08): `scripts/setup_points.py`.

Pure parts first, then the one test that matters for a presentation switch:
the setups table shows exactly the same rows with the switch on and off.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import setup_points  # noqa: E402
from setup_points import (  # noqa: E402
    BOUNCE_NAMED,
    BOUNCE_TODAY,
    RS_LEG_CAP,
    SR_CLEAN_PATH,
    SR_FLOOR,
    bounce_part,
    rank_order,
    rs_part,
    score_row,
    setup_part,
    sr_part,
)


@pytest.fixture(scope="module", autouse=True)
def _app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def test_setup_part_uses_the_lower_bound_and_expected_r():
    points, notes = setup_part({"win_rate_lb": 0.52, "win_rate": 1.0, "n": 3}, 0.33)
    assert points == pytest.approx(0.52 * 40 + 0.33 * 10)
    assert notes == []
    # A 100%-on-three family carries a LOW bound; the raw rate never enters.
    high_rate_low_bound, _ = setup_part({"win_rate_lb": 0.29, "win_rate": 1.0, "n": 3}, 0.0)
    steady, _ = setup_part({"win_rate_lb": 0.52, "win_rate": 0.62, "n": 90}, 0.0)
    assert steady > high_rate_low_bound


def test_setup_part_names_an_ungraded_family_and_clamps_expected_r():
    points, notes = setup_part({}, 4.0)
    assert points == pytest.approx(10.0)
    assert any("ungraded" in note for note in notes)
    none_points, none_notes = setup_part(None, None)
    assert none_points == 0.0
    assert any("no expected R" in note for note in none_notes)


def test_sr_part_knocks_levels_ahead_down_and_floors():
    clean, notes = sr_part({}, "LONG")
    assert clean == SR_CLEAN_PATH and notes == []
    crowded, notes = sr_part(
        {
            "hv_level_blocking_count": 5,
            "hv_level_nearby_count": 1,
            "cloud_level_nearby_count": 1,
            "trendline_note": "trendline resistance overhead",
            "hv_level_nearest_distance_atr": 0.05,
            "previous_close": 100.0,
            "atr20": 2.0,
            "ema21": 101.0,  # ahead for a LONG, inside 1 ATR
            "sma_breakout_sma_level": 110.0,  # 5 ATR away: not counted
        },
        "LONG",
    )
    assert crowded == SR_FLOOR
    assert any("5 HV level(s) blocking" in note for note in notes)
    assert any("EMA21 inside" in note for note in notes)
    assert not any("SMA inside" in note for note in notes)


def test_sr_part_reads_ahead_in_the_trades_direction():
    raw = {"previous_close": 100.0, "atr20": 2.0, "ema21": 101.0}
    long_points, _ = sr_part(raw, "LONG")
    short_points, _ = sr_part(raw, "SHORT")
    assert long_points == SR_CLEAN_PATH - 3.0
    assert short_points == SR_CLEAN_PATH  # the EMA is BEHIND a short


def test_rs_part_flips_sign_for_a_short_and_clamps_each_leg():
    raw = {"daily_relative_strength_score": -5.1, "rs_vs_industry": -4.75}
    short_points, short_notes = rs_part(raw, "SHORT", d1_vs_sector=-9.0)
    # -5.1 and -9.0 clamp to the +-5 cap; -4.75 is inside it.
    assert short_points == pytest.approx(2 * RS_LEG_CAP + 4.75)
    assert short_notes == []
    long_points, _ = rs_part(raw, "LONG", d1_vs_sector=-9.0)
    assert long_points == pytest.approx(-(2 * RS_LEG_CAP + 4.75))
    partial, notes = rs_part({"daily_relative_strength_score": 2.0}, "LONG")
    assert partial == pytest.approx(2.0)
    assert sorted(notes) == ["RS vs industry unmeasured", "RS vs sector unmeasured"]
    # The enriched industry reading outranks the scan's own when both exist.
    assert rs_part({"rs_vs_industry": -4.0}, "LONG", d1_vs_industry=1.0)[0] == pytest.approx(1.0)


def test_bounce_part_today_then_named_then_nothing():
    assert bounce_part({"has_bounce_event_today": True})[0] == BOUNCE_TODAY
    assert bounce_part({"favorite_signals": ["BOUNCE_VWAP"]})[0] == BOUNCE_NAMED
    assert bounce_part({"setup_family": "avwap_band_bounce"})[0] == BOUNCE_NAMED
    assert bounce_part({"top_pattern_daily_sma50_bounce": True})[0] == BOUNCE_NAMED
    assert bounce_part({"setup_family": "avwap_breakout"}) == (0.0, ["no recent bounce"])


def test_score_row_sums_the_four_parts_and_tooltip_names_them():
    points = score_row(
        {
            "has_bounce_event_today": True,
            "expected_r": 0.5,
            "daily_relative_strength_score": 3.0,
            "hv_level_blocking_count": 1,
        },
        side="LONG",
        family_record={"win_rate_lb": 0.5},
    )
    assert points.setup == pytest.approx(25.0)
    assert points.sr == pytest.approx(SR_CLEAN_PATH - 4.0)
    assert points.rs == pytest.approx(3.0)
    assert points.bounce == BOUNCE_TODAY
    assert points.total == pytest.approx(25.0 + 6.0 + 3.0 + 15.0)
    assert points.text() == "+49"
    tip = points.tooltip()
    for word in ("setup", "S/R", "RS/RW", "bounce", "1 HV level(s) blocking"):
        assert word in tip


def test_rank_order_reorders_the_ranked_buckets_only_and_keeps_every_row():
    items = [
        ("study", 99.0),  # not a ranked bucket: stays after the ranked rows
        ("favorite_setup", 10.0),
        ("near_favorite_zone", 30.0),
        ("favorite_setup", 30.0),  # ties keep arrival order
        ("", 5.0),
        ("high_conviction", None),  # no total: after every scored ranked row
        ("favorite_setup", -4.0),
    ]
    order = rank_order(items)
    assert sorted(order) == list(range(len(items)))
    assert order == [2, 3, 1, 6, 5, 0, 4]
    assert rank_order([]) == []


def test_rank_enabled_defaults_off(monkeypatch):
    import project_paths

    project_paths.invalidate_local_settings_cache()
    assert setup_points.rank_enabled() is False


def _set_switch(on: bool) -> None:
    import project_paths

    project_paths.save_local_setting(setup_points.SETTING_KEY, bool(on))
    project_paths.invalidate_local_settings_cache()


@pytest.mark.qt
def test_the_switch_reorders_the_setups_table_and_shows_exactly_the_same_rows():
    from ui.models.setup import SetupRow
    from ui.panels.master_avwap_panel import MasterAvwapPanel

    rows = [
        SetupRow(symbol="NVDA", side="LONG", score=90.0, bucket="favorite_setup",
                 raw={"setup_family": "alpha"}),
        SetupRow(symbol="TSLA", side="SHORT", score=80.0, bucket="favorite_setup",
                 raw={"setup_family": "beta", "has_bounce_event_today": True}),
        SetupRow(symbol="AMD", side="LONG", score=70.0, bucket="near_favorite_zone",
                 raw={"setup_family": "alpha", "hv_level_blocking_count": 3}),
        SetupRow(symbol="XYZ", side="LONG", score=99.0, bucket="study",
                 raw={"setup_family": "alpha", "has_bounce_event_today": True}),
    ]

    def _run(switch_on: bool):
        _set_switch(switch_on)
        panel = MasterAvwapPanel(None)
        try:
            assert panel.points_toggle.isChecked() is switch_on
            panel.model.set_family_records({"alpha": {"win_rate_lb": 0.6}, "beta": {"win_rate_lb": 0.6}})
            panel.set_rows(list(rows))
            shown = [row.symbol for row in panel.model.rows()]
            keys = [key for key, _label in panel.model.COLUMNS]
            column = keys.index("points")
            cells = {
                panel.model.rows()[i].symbol: panel.model.data(panel.model.index(i, column))
                for i in range(panel.model.rowCount())
            }
            return shown, cells
        finally:
            panel.deleteLater()

    try:
        off_shown, off_cells = _run(False)
        on_shown, on_cells = _run(True)
    finally:
        _set_switch(False)
    assert off_shown == ["NVDA", "TSLA", "AMD", "XYZ"], off_shown
    # NVDA: alpha 0.6*40 + clean 10 = 34. TSLA: beta 24 + 10 + 15 = 49.
    # AMD: 24 + (10 - 12) = 22. XYZ is a study row: 49 points, still last.
    assert on_shown == ["TSLA", "NVDA", "AMD", "XYZ"], on_shown
    assert off_cells == on_cells
    assert off_cells == {"NVDA": "+34", "TSLA": "+49", "AMD": "+22", "XYZ": "+49"}
    assert sorted(off_shown) == sorted(on_shown)


@pytest.mark.qt
def test_the_switch_lifts_a_higher_point_row_over_a_higher_score_row():
    from ui.models.setup import SetupRow
    from ui.panels.master_avwap_panel import MasterAvwapPanel

    rows = [
        SetupRow(symbol="LOW", side="LONG", score=95.0, bucket="favorite_setup",
                 raw={"setup_family": "alpha", "hv_level_blocking_count": 5}),
        SetupRow(symbol="HIGH", side="LONG", score=50.0, bucket="near_favorite_zone",
                 raw={"setup_family": "alpha", "has_bounce_event_today": True}),
    ]
    _set_switch(True)
    panel = MasterAvwapPanel(None)
    try:
        panel.set_rows(list(rows))
        assert [row.symbol for row in panel.model.rows()] == ["HIGH", "LOW"]
        panel.points_toggle.setChecked(False)
        assert [row.symbol for row in panel.model.rows()] == ["LOW", "HIGH"]
        assert setup_points.rank_enabled() is False
    finally:
        panel.deleteLater()
        _set_switch(False)
