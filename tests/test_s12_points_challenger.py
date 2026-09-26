"""S12 points challenger SP4: the frozen adjust, the family evidence, the trial, the night slot.

Hand-built fixture: 30 sessions, SPY flat at 100 (so excess = side return), every
entry close 100 and every atr20 2.0 (so a move in ATR = side return / 2).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import points_challenger as pc  # noqa: E402
from ai_jobs import family_side_evidence as fse  # noqa: E402

DAYS = [f"2026-08-{d:02d}" for d in range(1, 31)]
AS_OF = DAYS[-1]


def _row(symbol, side, family, day_index, horizon, side_return, *, mature=True):
    return {
        "symbol": symbol, "side": side, "scan_date": DAYS[day_index],
        "target_session": DAYS[day_index + horizon], "horizon_sessions": str(horizon),
        "side_return_pct": str(side_return), "entry_close": "100",
        "measured": "True" if mature else "False", "maturity": "mature" if mature else "immature",
        "setup_family": family, "outcome_kind": pc.OUTCOME_KIND,
    }


def fixture_rows(sessions=16):
    """Per session: 6 SHORT `alpha` (5 win +2%, 1 lose -1%) and 6 LONG `beta` (1 win +1%, 5 lose -2%)."""
    rows = []
    for i in range(sessions):
        for k in range(6):
            alpha = 2.0 if k < 5 else -1.0
            beta = 1.0 if k == 0 else -2.0
            for horizon in (5, 10):
                rows.append(_row(f"A{k}", "SHORT", "alpha", i, horizon, alpha))
                rows.append(_row(f"B{k}", "LONG", "beta", i, horizon, beta))
    return rows


SPY = {day: 100.0 for day in DAYS}
ATR = {(f"{p}{k}", day): 2.0 for p in "AB" for k in range(6) for day in DAYS}


# ------------------------------------------------------------------ the adjust
def test_the_adjust_is_the_frozen_formula_with_clamp_and_floors():
    cell = {"n": 100, "sessions": 20, "beat_low_h5": 0.65, "mean_move_atr_h10": 1.0}
    assert pc.family_adjust(cell) == pytest.approx(60 * 0.15 + 20 * 1.0)  # 29.0
    assert pc.family_adjust({**cell, "mean_move_atr_h10": 5.0}) == 40.0
    assert pc.family_adjust({**cell, "beat_low_h5": 0.0, "mean_move_atr_h10": -5.0}) == -40.0
    assert pc.family_adjust({**cell, "n": 79}) == 0.0
    assert pc.family_adjust({**cell, "sessions": 14}) == 0.0
    assert pc.family_adjust({**cell, "mean_move_atr_h10": None}) == 0.0
    assert (pc.BEAT_WEIGHT, pc.BEAT_CENTRE, pc.MOVE_WEIGHT, pc.ADJUST_CLAMP, pc.MIN_N, pc.MIN_SESSIONS) == (
        60.0, 0.50, 20.0, 40.0, 80, 15
    )


def test_sp4_points_is_champion_plus_adjust_and_unknown_without_a_score():
    evidence = {"families": {"SHORT|alpha": {"n": 100, "sessions": 20, "beat_low_h5": 0.65,
                                             "mean_move_atr_h10": 1.0}}}
    assert pc.sp4_points({"priority_score": "50", "side": "short", "setup_family": "Alpha"}, evidence) == 79.0
    assert pc.sp4_points({"priority_score": 50, "side": "LONG", "setup_family": "alpha"}, evidence) == 50.0
    assert pc.sp4_points({"priority_score": "", "side": "SHORT", "setup_family": "alpha"}, evidence) is None
    assert pc.sp4_points({"priority_score": 10, "side": "SHORT", "setup_family": "alpha"}, None) == 10.0


# ---------------------------------------------------------------- the evidence
def test_family_cells_count_tape_beats_moves_and_payoff_by_hand():
    cells, days = pc.build_families(fixture_rows(), SPY, ATR, {}, as_of=AS_OF)
    assert len(days) == 16
    alpha = cells["SHORT|alpha"]
    # 16 sessions x 6 rows at h5: 80 beat SPY (flat), 16 do not.
    assert (alpha["n"], alpha["sessions"], alpha["wins_h5"]) == (96, 16, 80)
    assert alpha["beat_rate_h5"] == pytest.approx(80 / 96, abs=1e-4)
    from swing_headline import wilson_lower_bound

    assert alpha["beat_low_h5"] == pytest.approx(wilson_lower_bound(80, 96), abs=1e-4)
    # mean excess = (80 x 2 - 16 x 1) / 96 = 1.5; move ATR = return / 2.
    assert alpha["mean_excess_pct_h5"] == pytest.approx(1.5)
    assert alpha["mean_move_atr_h10"] == pytest.approx(0.75)
    assert alpha["median_move_atr_h10"] == pytest.approx(1.0)
    # payoff = mean up move 1.0 / mean down move 0.5.
    assert alpha["payoff_h10"] == pytest.approx(2.0)
    expected = round(60 * (alpha["beat_low_h5"] - 0.5) + 20 * 0.75, 1)
    assert alpha["adjust"] == pytest.approx(expected, abs=0.051)
    beta = cells["LONG|beta"]
    assert beta["wins_h5"] == 16 and beta["adjust"] < 0


def test_immature_rows_and_rows_past_as_of_are_unknown_not_losses():
    rows = [_row("A0", "SHORT", "alpha", 0, 5, -3.0, mature=False), _row("A1", "SHORT", "alpha", 0, 5, 2.0)]
    cells, _days = pc.build_families(rows, SPY, ATR, {}, as_of=AS_OF)
    assert cells["SHORT|alpha"]["n"] == 1
    cells, _days = pc.build_families(rows, SPY, ATR, {}, as_of=DAYS[2])
    assert cells["SHORT|alpha"]["n"] == 0


def test_the_window_is_the_trailing_40_scan_sessions():
    rows = [_row("A0", "SHORT", "alpha", i % 25, 5, 1.0) for i in range(25)]
    assert pc.window_days(rows, as_of=AS_OF, sessions=10) == DAYS[15:25]


def test_tracker_avg_total_r_is_setup_weighted_over_buckets():
    rows = [
        {"attribute_key": "setup.setup_family", "side": "SHORT", "value_label": "alpha",
         "setup_count": "300", "avg_total_r": "0.2"},
        {"attribute_key": "setup.setup_family", "side": "SHORT", "value_label": "alpha",
         "setup_count": "100", "avg_total_r": "-0.2"},
        {"attribute_key": "other", "side": "SHORT", "value_label": "alpha",
         "setup_count": "999", "avg_total_r": "9"},
    ]
    out = pc.tracker_r_by_family(rows)
    assert out["SHORT|alpha"]["setups"] == 400
    assert out["SHORT|alpha"]["avg_total_r"] == pytest.approx(0.1)


# ------------------------------------------------------------------- the trial
def _trial_rows(day_index, short_returns):
    return [_row(f"S{k}", "SHORT", "alpha" if k < 4 else "gamma", day_index, 5, ret)
            for k, ret in enumerate(short_returns)]


def test_trial_ranks_each_entry_session_with_that_nights_adjusts():
    # Session 0: 8 shorts. Champion scores favour gamma (S4..S7); the adjust
    # written THAT night lifts alpha (S0..S3) over them. Quartile = 2 rows.
    rows = _trial_rows(0, [3.0, 3.0, 3.0, 3.0, -1.0, -1.0, -1.0, -1.0])
    scores = {(f"S{k}", "SHORT", DAYS[0]): (10.0 if k < 4 else 30.0) + k for k in range(8)}
    tracker = {(f"S{k}", "SHORT", DAYS[0]): (0.5 if k < 4 else -0.5) for k in range(8)}
    history = {DAYS[0]: {"SHORT|alpha": 40.0, "SHORT|gamma": 0.0}, DAYS[1]: {"SHORT|alpha": -40.0}}
    trial = pc.trial_summary(rows, SPY, scores, tracker, history, as_of=AS_OF)
    short = trial["short"]
    assert trial["first_session"] == DAYS[0]
    assert short["entry_sessions"] == 1 and short["matured_sessions"] == 1
    assert short["champion"] == {"n": 2, "excess_pct": -1.0, "r_n": 2, "tracker_r": -0.5}
    assert short["sp4"] == {"n": 2, "excess_pct": 3.0, "r_n": 2, "tracker_r": 0.5}
    assert short["verdict"] == pc.COLLECTING
    assert trial["long"]["entry_sessions"] == 0


def _block(matured, champion_x, sp4_x, champion_r=0.0, sp4_r=0.0):
    return {"matured_sessions": matured,
            "champion": {"excess_pct": champion_x, "tracker_r": champion_r},
            "sp4": {"excess_pct": sp4_x, "tracker_r": sp4_r}}


def test_trial_verdicts_follow_the_fixed_rules():
    assert pc.trial_verdict(_block(20, 0.0, 0.5, 0.0, 0.10), entries_done=20) == pc.SUCCESS
    assert pc.trial_verdict(_block(20, 0.0, 0.5, 0.0, 0.09), entries_done=20) == pc.NO_EDGE
    assert pc.trial_verdict(_block(20, 0.0, 0.49, 0.0, 0.5), entries_done=20) == pc.NO_EDGE
    assert pc.trial_verdict(_block(19, 0.0, 1.0, 0.0, 0.5), entries_done=20) == pc.COLLECTING
    assert pc.trial_verdict(_block(10, 1.0, 0.49), entries_done=12) == pc.STOPPED
    assert pc.trial_verdict(_block(9, 1.0, 0.0), entries_done=12) == pc.COLLECTING
    assert pc.trial_verdict(_block(10, 1.0, 0.5), entries_done=12) == pc.COLLECTING
    assert pc.trial_verdict(_block(20, None, 0.5), entries_done=20) == pc.COLLECTING
    assert (pc.TRIAL_ENTRY_SESSIONS, pc.TRIAL_MATURE_SESSIONS, pc.SUCCESS_EXCESS_PCT, pc.SUCCESS_R,
            pc.STOP_TRAIL_PCT, pc.STOP_AFTER_SESSIONS) == (20, 5, 0.5, 0.10, 0.5, 10)


def test_saturday_lines_one_per_side():
    assert len(pc.saturday_lines({})) == 2
    assert "no entry sessions yet" in pc.saturday_lines({})[0]
    trial = {"first_session": DAYS[0], "short": {
        "entry_sessions": 3, "matured_sessions": 2, "verdict": pc.COLLECTING,
        "champion": {"n": 6, "excess_pct": 0.2, "tracker_r": -0.05},
        "sp4": {"n": 6, "excess_pct": 1.25, "tracker_r": 0.1}}}
    long_line, short_line = pc.saturday_lines(trial)
    assert long_line.startswith("SP4 shadow, longs: no entry sessions yet")
    assert short_line == (
        "SP4 shadow, shorts: 2 of 20 entry sessions measured (3 entered since 2026-08-01). "
        "Top quartile vs SPY: SP4 +1.25% vs champion +0.20% (n 6 / 6); tracker R SP4 +0.10R vs -0.05R. "
        "collecting."
    )


def test_chip_names_the_biggest_adjusts_and_says_live_is_unchanged():
    cells, _ = pc.build_families(fixture_rows(), SPY, ATR, {}, as_of=AS_OF)
    text = pc.chip_text({"families": cells, "as_of": AS_OF, "window": {"sessions": 16}})
    assert "alpha SHORT +" in text and "beta LONG -" in text
    assert "live sort, buckets and alerts are unchanged" in text
    assert "no family evidence yet" in pc.chip_text({})


# ---------------------------------------------------------------- the night slot
def _write_inputs(tmp_path, sessions=16):
    rows = fixture_rows(sessions)
    horizon = tmp_path / "horizon.csv"
    with horizon.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fse.HORIZON_COLUMNS) + ["extra"])
        writer.writeheader()
        writer.writerows(rows)
    features = tmp_path / "features.csv"
    with features.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fse.FEATURE_COLUMNS) + ["other"])
        writer.writeheader()
        for (symbol, day), atr in ATR.items():
            side = "SHORT" if symbol.startswith("A") else "LONG"
            writer.writerow({"run_date": day, "run_timestamp": f"{day}T16:00:00", "last_trade_date": day,
                             "symbol": symbol, "side": side, "atr20": atr, "priority_score": "40"})
    spy = tmp_path / "SPY.csv"
    spy.write_text("date,close\n" + "".join(f"{d},100\n" for d in DAYS), encoding="utf-8")
    board = tmp_path / "board.csv"
    board.write_text("side,priority_bucket,attribute_key,value_label,setup_count,avg_total_r\n"
                     "SHORT,favorite_setup,setup.setup_family,alpha,10,0.3\n", encoding="utf-8")
    return {"horizon_path": horizon, "features_path": features, "spy_path": spy,
            "leaderboard_path": board, "snapshot_path": tmp_path / "no_snapshot.json"}


def test_night_slot_writes_the_evidence_and_carries_the_adjust_history(tmp_path):
    inputs = _write_inputs(tmp_path)
    out = tmp_path / "out" / "family_side_evidence.json"
    prior = {"schema": pc.SCHEMA, "adjust_history": {"2026-07-31": {"SHORT|alpha": 12.0}}}
    out.parent.mkdir()
    out.write_text(json.dumps(prior), encoding="utf-8")
    result = fse.run_family_side_evidence(session_date=AS_OF, out_path=out, **inputs)
    assert result["status"] == "ok", result
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == pc.SCHEMA and payload["as_of"] == AS_OF
    assert payload["window"] == {"first": DAYS[0], "last": DAYS[15], "sessions": 16}
    assert payload["families"]["SHORT|alpha"]["avg_total_r"] == pytest.approx(0.3)
    assert payload["adjust_history"]["2026-07-31"] == {"SHORT|alpha": 12.0}
    assert payload["adjust_history"][AS_OF]["SHORT|alpha"] == payload["families"]["SHORT|alpha"]["adjust"]
    assert len(payload["saturday_lines"]) == 2


def test_night_slot_refuses_to_write_under_15_sessions(tmp_path):
    inputs = _write_inputs(tmp_path, sessions=14)
    out = tmp_path / "family_side_evidence.json"
    out.write_text("LAST GOOD", encoding="utf-8")
    result = fse.run_family_side_evidence(session_date=AS_OF, out_path=out, **inputs)
    assert result["status"] == "ok" and result["outputs"] == []
    assert "14 session" in result["reason"]
    assert out.read_text(encoding="utf-8") == "LAST GOOD"


def test_night_slot_failure_keeps_the_last_good_file(tmp_path):
    inputs = _write_inputs(tmp_path)
    inputs["features_path"] = tmp_path / "missing_features.csv"
    out = tmp_path / "family_side_evidence.json"
    out.write_text("LAST GOOD", encoding="utf-8")
    result = fse.run_family_side_evidence(session_date=AS_OF, out_path=out, **inputs)
    assert result["status"] == "failed"
    assert "last file kept" in result["reason"]
    assert out.read_text(encoding="utf-8") == "LAST GOOD"


def test_the_evidence_path_is_beside_the_horizon_outcomes():
    import project_paths as pp

    assert Path(pp.FAMILY_SIDE_EVIDENCE_FILE).name == "family_side_evidence.json"
    assert Path(pp.FAMILY_SIDE_EVIDENCE_FILE).parent == Path(pp.MASTER_AVWAP_SESSION_HORIZON_OUTCOMES_FILE).parent


def test_the_slot_closes_stage_one_directly_after_day_review_facts(tmp_path):
    from ai_jobs import runner

    slots = runner.default_slots()
    names = [slot.name for slot in slots]
    slot = next(s for s in slots if s.name == "family_side_evidence")
    assert names.index("family_side_evidence") == names.index("day_review_facts") + 1
    # S15 (2026-09-26): `swing_path_facts` follows it and now closes stage 1.
    assert runner._STAGE_ONE_LAST_SLOT == "swing_path_facts"
    assert names.index("swing_path_facts") == names.index("family_side_evidence") + 1
    assert slot.goal == "setup_quality" and slot.uses_model is False and slot.max_attempts == 3
    for kind in ("weeknight", "saturday", "sunday"):
        slate = runner.slots_for(kind, session_date="2026-09-25", ledger_path=tmp_path / "ledger.jsonl")
        assert "family_side_evidence" in [s.name for s in slate], kind
