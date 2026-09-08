"""The point system's evidence loop (trader, 2026-09-08): log, grade, correct.

Every write here goes to `tmp_path`; the desk's own files are never touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import setup_points  # noqa: E402
import setup_points_evidence as evidence  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _logged(scan_date, symbol, side, total, **parts):
    row = {"scan_date": scan_date, "symbol": symbol, "side": side, "total": total,
           "setup": 0.0, "sr": 0.0, "rs": 0.0, "bounce": 0.0, "bucket": "favorite_setup"}
    row.update(parts)
    return row


def test_append_log_dedupes_on_scan_date_symbol_side_and_never_raises(tmp_path):
    path = tmp_path / "log.jsonl"
    rows = [_logged("2026-09-08", "NVDA", "LONG", 30.0), _logged("2026-09-08", "nvda", "long", 31.0),
            _logged("", "AMD", "LONG", 5.0)]
    assert evidence.append_log(path, rows) == 1
    assert evidence.append_log(path, rows) == 0
    assert evidence.append_log(path, [_logged("2026-09-09", "NVDA", "LONG", 32.0)]) == 1
    logged = evidence.read_log(path)
    assert [(r["scan_date"], r["symbol"], r["total"]) for r in logged] == [
        ("2026-09-08", "NVDA", 30.0), ("2026-09-09", "NVDA", 32.0)]
    assert all(r["weights_version"] == evidence.WEIGHTS_VERSION and r["logged_at"] for r in logged)
    # An unwritable path loses the rows, never raises.
    assert evidence.append_log(tmp_path / "log.jsonl" / "child", rows) == 0


def test_score_row_log_row_records_raw_parts_and_the_weights_in_force():
    points = setup_points.score_row(
        {"has_bounce_event_today": True, "expected_r": 0.5},
        side="LONG",
        family_record={"win_rate_lb": 0.5},
        weights={"bounce": 0.5, "setup": 1.5},
    )
    assert points.raw_parts == {"setup": 25.0, "sr": 10.0, "rs": 0.0, "bounce": 15.0}
    assert (points.setup, points.bounce, points.total) == (37.5, 7.5, 55.0)
    assert "learned weights: bounce x0.50, setup x1.50" in points.tooltip()
    row = points.log_row(scan_date="2026-09-08", symbol="nvda", side="long", family="alpha", bucket="favorite_setup")
    assert row["symbol"] == "NVDA" and row["side"] == "LONG"
    assert (row["setup"], row["bounce"], row["total"]) == (25.0, 15.0, 55.0)
    assert row["multipliers"] == {"setup": 1.5, "sr": 1.0, "rs": 1.0, "bounce": 0.5}
    # No weights: the parts are the raw parts and the tooltip says nothing about learning.
    plain = setup_points.score_row({"has_bounce_event_today": True}, side="LONG")
    assert plain.weights == {"setup": 1.0, "sr": 1.0, "rs": 1.0, "bounce": 1.0}
    assert "learned" not in plain.tooltip()


def _population(n_per_third: int, top_rate: float, bottom_rate: float):
    """Log rows whose total AND `bounce` part rise with the win rate; `sr` is noise."""
    log, outcomes = [], []
    for index in range(3 * n_per_third):
        third = index // n_per_third  # 0 top, 1 middle, 2 bottom
        total = 90.0 - index
        rate = {0: top_rate, 1: (top_rate + bottom_rate) / 2, 2: bottom_rate}[third]
        win = (index % n_per_third) < round(rate * n_per_third)
        symbol = f"S{index:03d}"
        log.append(_logged("2026-09-08", symbol, "LONG", total, bounce=15.0 if third == 0 else 0.0,
                           sr=float(index % 7)))
        outcomes.append({"scan_date": "2026-09-08", "symbol": symbol, "side": "LONG", "win": str(win)})
    return log, outcomes


def test_grade_reads_terciles_lift_and_per_part_lift():
    log, outcomes = _population(40, 0.70, 0.40)
    result = evidence.grade(log, outcomes, horizon_sessions=5, window=("2026-09-08", "2026-09-08"))
    assert (result.n_logged, result.n_joined) == (120, 120)
    top, middle, bottom = result.terciles
    assert (top.n, middle.n, bottom.n) == (40, 40, 40)
    assert top.win_rate == pytest.approx(0.70) and bottom.win_rate == pytest.approx(0.40)
    assert result.lift == pytest.approx(0.30)
    assert top.lower_bound is not None and top.lower_bound < top.win_rate
    assert result.part_lift["bounce"] > 0.15
    assert result.part_lift["setup"] is None  # every row 0.0: halves are equal, lift is 0 or None
    assert "higher points DID perform better" in result.sentence()
    assert "lift +30 pts" in result.sentence()


def test_grade_says_not_enough_and_never_a_number_below_the_floor():
    log, outcomes = _population(5, 0.9, 0.1)
    result = evidence.grade(log, outcomes, horizon_sessions=5)
    assert "not enough per third yet" in result.sentence()
    empty = evidence.grade(log, [], horizon_sessions=5)
    assert empty.n_joined == 0 and "none graded yet" in empty.sentence()
    # An unmeasured outcome (win blank) joins nothing.
    blank = evidence.grade(log, [{**o, "win": ""} for o in outcomes], horizon_sessions=5)
    assert blank.n_joined == 0


def test_propose_weights_moves_only_a_part_over_the_floor_and_clamps():
    log, outcomes = _population(40, 0.75, 0.35)
    proposal = evidence.propose_weights(evidence.grade(log, outcomes, horizon_sessions=5))
    m = proposal["multipliers"]
    assert m["bounce"] > 1.0 and m["bounce"] <= evidence.WEIGHT_CEILING
    assert m["setup"] == 1.0 and "kept at 1.0" in proposal["reasons"]["setup"]
    assert proposal["n_joined"] == 120 and proposal["weights_version"] == evidence.WEIGHTS_VERSION
    small = evidence.propose_weights(evidence.grade(*_population(5, 0.9, 0.1), horizon_sessions=5))
    assert set(small["multipliers"].values()) == {1.0}
    assert evidence.proposal_multipliers({"multipliers": {"sr": 9.0, "rs": 0.1, "junk": 2}}) == {
        "setup": 1.0, "sr": evidence.WEIGHT_CEILING, "rs": evidence.WEIGHT_FLOOR, "bounce": 1.0}


def test_active_weights_apply_only_when_the_trader_switch_is_on(tmp_path, monkeypatch):
    import project_paths

    weights_file = tmp_path / "weights.json"
    monkeypatch.setattr(project_paths, "SETUP_POINTS_WEIGHTS_FILE", weights_file)
    assert evidence.write_proposal(weights_file, {"multipliers": {"bounce": 1.3}})
    project_paths.invalidate_local_settings_cache()
    assert setup_points.learned_weights_enabled() is False
    assert setup_points.active_weights() == {}
    project_paths.save_local_setting(setup_points.LEARNED_SETTING_KEY, True)
    project_paths.invalidate_local_settings_cache()
    try:
        assert setup_points.active_weights() == {"setup": 1.0, "sr": 1.0, "rs": 1.0, "bounce": 1.3}
        weights_file.unlink()
        assert setup_points.active_weights() == {"setup": 1.0, "sr": 1.0, "rs": 1.0, "bounce": 1.0}
    finally:
        project_paths.save_local_setting(setup_points.LEARNED_SETTING_KEY, False)
        project_paths.invalidate_local_settings_cache()


def test_log_and_grade_writes_the_log_and_the_proposal_and_reads_the_one_outcome_reader(tmp_path):
    outcomes = tmp_path / "tier_outcomes.csv"
    log_rows, outcome_rows = _population(40, 0.7, 0.4)
    import csv

    with open(outcomes, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "observation_id", "scan_date", "horizon_sessions", "symbol", "side", "win", "stale_horizon",
            "outcome_kind", "future_scan_date"])
        writer.writeheader()
        for index, row in enumerate(outcome_rows):
            for horizon in (1, 5):
                writer.writerow({"observation_id": f"o{index}h{horizon}", "scan_date": row["scan_date"],
                                 "horizon_sessions": horizon, "symbol": row["symbol"], "side": row["side"],
                                 "win": row["win"] if horizon == 5 else "False", "stale_horizon": "False",
                                 "outcome_kind": "favorable_direction_scanrow_v1",
                                 "future_scan_date": "2026-09-15"})
    result = evidence.log_and_grade(
        log_rows, log_path=tmp_path / "log.jsonl", weights_path=tmp_path / "w.json", outcomes_path=outcomes)
    assert result.n_joined == 120 and result.lift == pytest.approx(0.30)
    assert len(evidence.read_log(tmp_path / "log.jsonl")) == 120
    proposal = json.loads((tmp_path / "w.json").read_text(encoding="utf-8"))
    assert proposal["multipliers"]["bounce"] > 1.0
    assert proposal["grade"]["sentence"] == result.sentence()


def test_panel_builds_the_evidence_payload_from_ranked_rows_only():
    from ui.models.setup import SetupRow
    from ui.panels.master_avwap_panel import MasterAvwapPanel

    panel = MasterAvwapPanel(None)
    try:
        assert panel._uses_default_feedback_paths is False  # a test panel never writes evidence
        rows = [
            SetupRow(symbol="NVDA", side="LONG", bucket="favorite_setup", last_trade_date="2026-09-08",
                     raw={"setup_family": "alpha", "has_bounce_event_today": True}),
            SetupRow(symbol="AMD", side="LONG", bucket="near_favorite_zone", raw={"setup_family": "alpha"}),
            SetupRow(symbol="XYZ", side="LONG", bucket="study", raw={"setup_family": "alpha"}),
            SetupRow(symbol="", side="LONG", bucket="favorite_setup", raw={}),
        ]
        payload = panel.points_evidence_payload(rows, "2026-09-05")
        assert [(p["symbol"], p["scan_date"], p["bucket"]) for p in payload] == [
            ("NVDA", "2026-09-08", "favorite_setup"), ("AMD", "2026-09-05", "near_favorite_zone")]
        assert payload[0]["bounce"] == 15.0 and payload[0]["family"] == "alpha"
        assert "Learned weights OFF" in panel.points_toggle.toolTip()
        assert panel.points_grade_label.text() == ""
    finally:
        panel.deleteLater()
