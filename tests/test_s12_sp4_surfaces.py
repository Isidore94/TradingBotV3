"""S12 SP4 on the desk: the shadow column + chip on the Setup Tracker, the Saturday line.

The live Current Picks order must be byte-identical with and without evidence,
even when the SP4 points would order the rows differently.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import points_challenger as pc  # noqa: E402


def _pick(tier, symbol, side, score, family, bucket="favorite_setup"):
    return {"tier": tier, "symbol": symbol, "side": side, "priority_score": score,
            "setup_family": family, "priority_bucket": bucket, "favorite_zone": "", "current_band_zone": "",
            "trend_20d": "", "scan_factor_match_count": "0", "scan_factor_matches": ""}


PICKS = [
    _pick("A", "ZZZ", "LONG", "90", "favorite_zone_watch"),
    _pick("A", "AAA", "SHORT", "40", "avwap_retest_followthrough"),
    _pick("S", "MMM", "LONG", "70", "avwap_breakout"),
    _pick("B", "BBB", "SHORT", "", "avwap_band_bounce"),
]
EVIDENCE = {
    "schema": pc.SCHEMA, "as_of": "2026-09-25", "window": {"sessions": 29},
    "families": {
        # +40 (clamped) for the SHORT, -40 for both LONG families.
        "SHORT|avwap_retest_followthrough": {"n": 400, "sessions": 23, "beat_low_h5": 0.9, "mean_move_atr_h10": 2.0},
        "LONG|favorite_zone_watch": {"n": 900, "sessions": 23, "beat_low_h5": 0.1, "mean_move_atr_h10": -2.0},
        "LONG|avwap_breakout": {"n": 300, "sessions": 23, "beat_low_h5": 0.1, "mean_move_atr_h10": -2.0},
    },
}


def test_live_current_pick_order_is_byte_identical_with_sp4():
    from ui.panels import setup_tracker_panel as module

    before = json.dumps(module._rank_current_picks([dict(p) for p in PICKS]), sort_keys=True)
    shown = module.current_rows_with_sp4([dict(p) for p in PICKS], EVIDENCE)
    stripped = [{k: v for k, v in row.items() if k != "sp4_score"} for row in shown]
    assert json.dumps(stripped, sort_keys=True) == before
    assert [r["symbol"] for r in shown] == ["MMM", "ZZZ", "AAA", "BBB"]
    # SP4 alone would put the SHORT first (80 > 50 > 30): it does not move the live order.
    assert [r["sp4_score"] for r in shown] == [30.0, 50.0, 80.0, ""]
    # Tiers and buckets are untouched.
    assert [(r["tier"], r["priority_bucket"], r["priority_score"]) for r in shown] == [
        (r["tier"], r["priority_bucket"], r["priority_score"]) for r in json.loads(before)
    ]
    # No evidence: SP4 = the live score.
    assert [r["sp4_score"] for r in module.current_rows_with_sp4(PICKS, {})] == [70.0, 90.0, 40.0, ""]


def test_worker_read_carries_the_sp4_rows_chip_and_signature(monkeypatch, tmp_path):
    from ui.panels import setup_tracker_panel as module

    evidence_path = tmp_path / "family_side_evidence.json"
    evidence_path.write_text(json.dumps(EVIDENCE), encoding="utf-8")
    monkeypatch.setattr(module, "_sp4_evidence_path", lambda: evidence_path)
    real = module._load_csv_rows_cached
    monkeypatch.setattr(
        module, "_load_csv_rows_cached",
        lambda path: [dict(p) for p in PICKS] if Path(path) == Path(module.MASTER_AVWAP_TIER_LIST_FILE) else real(path),
    )
    data = module._read_tracker_exports(1)
    assert [r["sp4_score"] for r in data["ranked"]["current"]] == [30.0, 50.0, 80.0, ""]
    assert "avwap_retest_followthrough SHORT +40" in data["sp4_chip"]
    assert data["signatures"]["sp4_evidence"] is not None
    plan = {name: memo for name, _m, _rows, memo in module._table_render_plan(
        data["ranked"], data["signatures"], 1, "")}
    assert data["signatures"]["sp4_evidence"] in plan["current_table"]


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.mark.qt
def test_the_panel_shows_the_sp4_column_after_score_and_the_chip(qapp):
    from ui.panels import setup_tracker_panel as module

    keys = [key for key, _label in module.CURRENT_PICK_COLUMNS]
    assert keys.index("sp4_score") == keys.index("priority_score") + 1
    assert dict(module.CURRENT_PICK_COLUMNS)["sp4_score"] == "SP4 (shadow)"
    panel = module.SetupTrackerPanel()
    try:
        rows = module.current_rows_with_sp4(PICKS, EVIDENCE)
        panel._on_exports_loaded({"signatures": {}, "ranked": {"current": rows}, "raw": {}, "min_closed": 1,
                                  "sp4_chip": pc.chip_text(EVIDENCE)})
        assert panel.sp4_chip_label.text() == pc.chip_text(EVIDENCE)
        assert [r["symbol"] for r in panel.current_model.rows()] == ["MMM", "ZZZ", "AAA", "BBB"]
        panel._on_exports_loaded({"signatures": {}, "ranked": {}, "raw": {}, "min_closed": 1})
        assert panel.sp4_chip_label.text() == pc.chip_text({})
    finally:
        panel.shutdown()
        panel.deleteLater()


@pytest.mark.qt
def test_setup_keys_shows_the_saturday_sp4_line_per_side(qapp, tmp_path):
    from ui.panels import setup_keys_panel as module

    lines = ["SP4 shadow, longs: x.", "SP4 shadow, shorts: y."]
    evidence = tmp_path / "family_side_evidence.json"
    evidence.write_text(json.dumps({"schema": pc.SCHEMA, "saturday_lines": lines}), encoding="utf-8")
    assert module.read_sp4_lines(evidence) == lines
    assert module.read_sp4_lines(tmp_path / "missing.json") == pc.saturday_lines({})
    panel = module.SetupKeysPanel(report_path=tmp_path / "permutation_report.json")
    try:
        panel._on_read((None, None, lines))
        assert panel.sp4_label.text() == "\n".join(lines)
    finally:
        panel.shutdown()
        panel.deleteLater()
