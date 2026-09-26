"""P8 packet P6: swing grades read wins vs SPY, and PROVEN/A need cum R >= 0.

A down month makes every short look PROVEN when a win is "moved my way over 5
sessions". Here a swing pick wins only when its 5-session side return beats
SPY's same-side return over the same sessions; a missing SPY close is unknown.
PROVEN and A also need the family's cumulative R over the window to be >= 0.
Presentation only: the grade still only orders lists and prints a badge.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import setup_grades as sg  # noqa: E402

# SPY closes: down 5% from 09-01 to 09-08, up 5% from 09-02 to 09-09.
SPY = {
    "2026-09-01": 100.0,
    "2026-09-08": 95.0,
    "2026-09-02": 100.0,
    "2026-09-09": 105.0,
}


def _horizon(symbol, side, scan, target, side_return, **overrides):
    row = {
        "symbol": symbol,
        "side": side,
        "scan_date": scan,
        "target_session": target,
        "horizon_sessions": "5",
        "side_return_pct": str(side_return),
        "measured": "True",
        "maturity": "mature",
        "outcome_kind": "favorable_direction_session_v2",
    }
    row.update(overrides)
    return row


def _pick(symbol, side, scan, r=None, bucket="near_favorite_zone", family="general"):
    return {"symbol": symbol, "side": side, "session": scan, "r": r,
            "bucket": bucket, "family": family}


# ---------------------------------------------------------------------------
# one pick vs the tape
# ---------------------------------------------------------------------------


def test_a_short_that_fell_less_than_spy_loses_to_the_tape():
    # SPY -5%: a short's SPY-relative bar is +5%. Down 3% is a plain win, a tape loss.
    row = _horizon("AAA", "SHORT", "2026-09-01", "2026-09-08", 3.0)
    assert sg.tape_result({"side": "SHORT"}, row, SPY) == sg.LOSS
    row = _horizon("AAA", "SHORT", "2026-09-01", "2026-09-08", 6.0)
    assert sg.tape_result({"side": "SHORT"}, row, SPY) == sg.WIN


def test_a_long_must_beat_spy_and_a_tie_is_not_a_win():
    # SPY +5% from 09-02 to 09-09.
    assert sg.tape_result({"side": "LONG"}, _horizon("B", "LONG", "2026-09-02", "2026-09-09", 4.0), SPY) == sg.LOSS
    assert sg.tape_result({"side": "LONG"}, _horizon("B", "LONG", "2026-09-02", "2026-09-09", 5.0), SPY) == sg.LOSS
    assert sg.tape_result({"side": "LONG"}, _horizon("B", "LONG", "2026-09-02", "2026-09-09", 7.0), SPY) == sg.WIN
    # A long that fell less than a falling SPY beat the tape.
    assert sg.tape_result({"side": "LONG"}, _horizon("B", "LONG", "2026-09-01", "2026-09-08", -2.0), SPY) == sg.WIN


def test_missing_spy_or_an_unmeasured_row_is_unknown_never_a_win_or_loss():
    pick = {"side": "LONG"}
    no_spy = _horizon("C", "LONG", "2026-09-03", "2026-09-10", 9.0)
    assert sg.tape_result(pick, no_spy, SPY) == sg.UNKNOWN
    half = dict(SPY)
    del half["2026-09-09"]
    assert sg.tape_result(pick, _horizon("C", "LONG", "2026-09-02", "2026-09-09", 9.0), half) == sg.UNKNOWN
    immature = _horizon("C", "LONG", "2026-09-02", "2026-09-09", 9.0, maturity="immature")
    assert sg.tape_result(pick, immature, SPY) == sg.UNKNOWN
    unmeasured = _horizon("C", "LONG", "2026-09-02", "2026-09-09", "", measured="False")
    assert sg.tape_result(pick, unmeasured, SPY) == sg.UNKNOWN
    assert sg.tape_result(pick, None, SPY) == sg.UNKNOWN


def test_spy_is_read_only_through_the_target_session():
    row = _horizon("D", "LONG", "2026-09-02", "2026-09-09", 9.0)
    # Built as of the day before the target: the target close is not known yet.
    assert sg.tape_result({"side": "LONG"}, row, SPY, as_of="2026-09-08") == sg.UNKNOWN
    assert sg.tape_result({"side": "LONG"}, row, SPY, as_of="2026-09-09") == sg.WIN


# ---------------------------------------------------------------------------
# per family: tape stats and cum R
# ---------------------------------------------------------------------------


def test_tape_stats_count_per_key_and_sum_each_picks_own_r():
    picks = [
        _pick("S1", "SHORT", "2026-09-01", r=0.4),
        _pick("S2", "SHORT", "2026-09-01", r=-1.0),
        _pick("S3", "SHORT", "2026-09-03", r=None),  # open, no SPY: unknown
    ]
    horizon = [
        _horizon("S1", "SHORT", "2026-09-01", "2026-09-08", 6.0),  # beat SPY
        _horizon("S2", "SHORT", "2026-09-01", "2026-09-08", 1.0),  # plain win, tape loss
        _horizon("S1", "SHORT", "2026-09-01", "2026-09-04", 9.0, horizon_sessions="3"),  # other horizon
    ]
    stats = sg.swing_tape_stats(picks, horizon, SPY)
    cell = stats["SHORT|near_favorite_zone|general"]
    assert (cell["wins"], cell["n"], cell["unknown"], cell["sessions"]) == (1, 2, 1, 1)
    assert cell["cum_r_lately"] == pytest.approx(-0.6)


def test_a_key_with_no_closed_pick_has_unknown_cum_r():
    stats = sg.swing_tape_stats([_pick("S3", "SHORT", "2026-09-03", r=None)], [], SPY)
    assert stats["SHORT|near_favorite_zone|general"]["cum_r_lately"] is None


# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------


def test_a_down_month_short_that_only_rode_the_tape_is_not_proven():
    # Plain: 96 of 100 moved its way -> PROVEN before. Vs SPY: 45 of 100.
    plain = sg.grade_for(n=100, sessions=20, wins=96, avg_r=0.1, cum_r_lately=10.0)
    assert plain["grade"] == sg.PROVEN
    taped = sg.grade_for(
        n=100, sessions=20, wins=96, avg_r=0.1, cum_r_lately=10.0,
        tape={"wins": 45, "n": 100, "sessions": 20},
    )
    assert taped["grade"] == sg.D
    assert taped["grade_basis"] == "tape"
    assert taped["tape_win_rate"] == pytest.approx(0.45)
    assert taped["tape_low_bound"] == pytest.approx(sg.wilson_lower_bound(45, 100))
    assert taped["low_bound"] == pytest.approx(sg.wilson_lower_bound(96, 100))  # plain kept beside


def test_a_family_that_beat_spy_can_still_be_proven():
    cell = sg.grade_for(
        n=150, sessions=20, wins=140, avg_r=0.3, cum_r_lately=40.0,
        tape={"wins": 110, "n": 140, "sessions": 18},
    )
    assert cell["grade"] == sg.PROVEN


def test_proven_and_a_need_cum_r_at_or_above_zero_and_unknown_blocks_them():
    base = dict(n=200, sessions=20, wins=150, avg_r=0.3)
    assert sg.grade_for(**base, cum_r_lately=0.0)["grade"] == sg.PROVEN
    assert sg.grade_for(**base, cum_r_lately=-0.1)["grade"] == sg.B
    assert sg.grade_for(**base, cum_r_lately=None)["grade"] == sg.B
    a_base = dict(n=60, sessions=10, wins=42, avg_r=0.2)
    assert sg.grade_for(**a_base, cum_r_lately=3.0)["grade"] == sg.A
    assert sg.grade_for(**a_base, cum_r_lately=-3.0)["grade"] == sg.B


def test_under_the_floor_the_tape_is_unknown_and_the_plain_win_is_used_and_said():
    cell = sg.grade_for(
        n=200, sessions=20, wins=150, avg_r=0.3, cum_r_lately=5.0,
        tape={"wins": 2, "n": 29, "sessions": 3},
    )
    assert cell["grade"] == sg.PROVEN
    assert cell["grade_basis"] == "plain" and cell["tape_note"] == "tape: unknown"


def test_swing_cells_take_tape_stats_by_key_and_missing_is_unknown():
    row = {
        "side": "SHORT", "priority_bucket": "near_favorite_zone", "setup_family": "general",
        "namespace": "live", "n_wins": "96", "n_losses": "4", "n_flats": "0",
        "n_entry_sessions": "20", "representative_closed_r": "0.1",
    }
    [bare] = sg.swing_cells([row])
    assert bare["grade"] == sg.B  # cum R unknown blocks PROVEN
    assert bare["cum_r_lately"] is None and bare["tape_note"] == "tape: unknown"
    tape = {"SHORT|near_favorite_zone|general": {"wins": 45, "n": 100, "sessions": 20,
                                                   "unknown": 0, "cum_r_lately": 10.0}}
    [cell] = sg.swing_cells([row], tape)
    assert cell["grade"] == sg.D
    assert (cell["tape_n"], cell["cum_r_lately"]) == (100, 10.0)


def test_day_trade_cells_keep_bracket_wins_and_gain_cum_r():
    results = [
        {"event_id": f"e{i}", "trade_date": f"2026-09-{(i % 12) + 1:02d}", "side": "LONG",
         "bounce_type": "vwap", "result": sg.WIN if i < 30 else sg.LOSS}
        for i in range(40)
    ]
    [cell] = sg.daytrade_cells(results)
    assert cell["cum_r_lately"] == pytest.approx(20.0)
    assert "tape_n" not in cell and "grade_basis" not in cell
    assert cell["grade"] == sg.A


def test_the_rules_text_says_wins_vs_spy_and_cum_r():
    payload = sg.build_payload(recent_rows=[], outcome_rows=[], as_of="2026-09-25")
    assert "SPY" in payload["rules"] and "cum R >= 0" in payload["rules"]


# ---------------------------------------------------------------------------
# what the desk prints
# ---------------------------------------------------------------------------


def test_the_cell_line_reads_grade_win_vs_spy_cum_r_and_n():
    cell = sg.grade_for(
        n=150, sessions=20, wins=140, avg_r=0.3, cum_r_lately=4.1,
        tape={"wins": 71, "n": 114, "sessions": 18},
    )
    low = round(sg.wilson_lower_bound(71, 114) * 100)
    assert sg.cell_line(cell) == f"{cell['grade']} · win vs SPY 62% (low {low}%) · cum R +4.1 · n 114"
    plain = sg.grade_for(n=40, sessions=12, wins=30, avg_r=0.2, cum_r_lately=None,
                         tape={"wins": 0, "n": 0, "sessions": 0})
    assert "tape: unknown" in sg.cell_line(plain) and "cum R unknown" in sg.cell_line(plain)
    day = sg.daytrade_cells([
        {"event_id": f"e{i}", "trade_date": "2026-09-01", "side": "LONG",
         "bounce_type": "vwap", "result": sg.WIN if i < 20 else sg.LOSS}
        for i in range(30)
    ])[0]
    # S1 + S10c (trader 2026-09-26): a day-trade cell reads its 1:1 bracket grade,
    # 2R grade, EOD close R and n; these results carry no 2R or EOD data.
    assert sg.cell_line(day) == f"1:1 {sg.badge(day['grade'])} · 2R NEW · EOD unknown · n 30"
    assert "SPY" not in sg.cell_line(day)


def test_the_holdout_rows_carry_tape_and_cum_r_per_window():
    recent = sg.swing_cells(
        [{"side": "LONG", "priority_bucket": "b", "setup_family": "f", "namespace": "live",
          "n_wins": "40", "n_losses": "10", "n_flats": "0", "n_entry_sessions": "12",
          "representative_closed_r": "0.3"}],
        {"LONG|b|f": {"wins": 35, "n": 50, "sessions": 12, "unknown": 0, "cum_r_lately": 7.5}},
    )
    [row] = sg.holdout_view(recent, [])
    assert row["recent_tape_n"] == 50 and row["recent_cum_r_lately"] == 7.5
    assert row["recent_tape_win_rate"] == pytest.approx(0.7)
    assert row["prior_tape_n"] is None and row["prior_cum_r_lately"] is None
    assert "vs SPY 70%" in row["recent_text"] and "cum +7.5R" in row["recent_text"]


def test_best_first_breaks_ties_on_the_bound_the_grade_was_read_from():
    payload = {"daytrade": [], "swing": [
        {"key": "x", "grade": sg.B, "low_bound": 0.9, "tape_low_bound": 0.50, "grade_basis": "tape"},
        {"key": "y", "grade": sg.B, "low_bound": 0.6, "tape_low_bound": 0.58, "grade_basis": "tape"},
    ]}
    lookup = sg.swing_lookup(payload)
    ordered = sorted(lookup.values(), key=sg.swing_sort_key)
    assert [cell["key"] for cell in ordered] == ["y", "x"]


# ---------------------------------------------------------------------------
# the service: built on the Working-lately worker from temp copies
# ---------------------------------------------------------------------------


def _setup(symbol, scan, r, side="SHORT"):
    return {
        "setup_id": f"{scan}:{symbol}:{side}:2026-08-01:near_favorite_zone",
        "symbol": symbol,
        "side": side,
        "scan_date": scan,
        "anchor_date": "2026-08-01",
        "priority_bucket": "near_favorite_zone",
        "setup_family": "general",
        "_scoring_outcome_summary": {
            "tradeable_scenario_count": 2,
            "closed_tradeable_scenario_count": 2 if r is not None else 0,
            "representative_closed_r": r,
            "representative_status": "closed" if r is not None else "pending",
        },
    }


@pytest.fixture()
def tape_files(tmp_path, monkeypatch):
    import csv
    import json
    from datetime import date

    import pandas as pd

    import looking_back
    from ui.services import working_lately_service as svc

    setups = {
        "a": _setup("S1", "2026-09-01", 0.4),
        "b": _setup("S2", "2026-09-01", -1.0),
        "c": _setup("S3", "2026-09-02", None),
        "old": _setup("S4", "2026-07-01", 5.0),  # outside the 30-day window
    }
    snapshot = tmp_path / "scoring.json"
    snapshot.write_text(json.dumps({"setups": setups}), encoding="utf-8")
    horizon = tmp_path / "horizon.csv"
    rows = [
        _horizon("S1", "SHORT", "2026-09-01", "2026-09-08", 6.0),
        _horizon("S2", "SHORT", "2026-09-01", "2026-09-08", 1.0),
        _horizon("S3", "SHORT", "2026-09-02", "2026-09-09", 9.0),
    ]
    with horizon.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    spy = tmp_path / "SPY.parquet"
    pd.DataFrame(
        {"datetime": pd.to_datetime(list(SPY)), "close": list(SPY.values())}
    ).to_parquet(spy)

    monkeypatch.setattr(svc, "_scoring_snapshot_path", lambda: snapshot)
    monkeypatch.setattr(svc, "_horizon_outcomes_path", lambda: horizon, raising=False)
    monkeypatch.setattr(svc, "_spy_bars_path", lambda: spy, raising=False)
    monkeypatch.setattr(svc, "_last_completed_session", lambda: date(2026, 9, 18))
    monkeypatch.setattr(
        looking_back, "_default_context",
        lambda s: (s["side"], s["priority_bucket"], s["setup_family"]),
    )
    monkeypatch.setattr(svc, "_LOOKING_BACK_CACHE", {})
    return svc


def test_the_service_reads_tape_and_cum_r_over_the_tracker_window(tape_files):
    from datetime import date

    svc = tape_files
    stats = svc.read_swing_tape(date(2026, 9, 18), as_of="2026-09-18")
    cell = stats["SHORT|near_favorite_zone|general"]
    # S1 beat a falling SPY, S2 did not, S3 rode a rising SPY short and beat it.
    assert (cell["wins"], cell["n"]) == (2, 3)
    assert cell["cum_r_lately"] == pytest.approx(-0.6)  # S4 is outside the window


def test_the_service_grades_carry_the_tape_fields(tape_files):
    svc = tape_files
    row = {
        "side": "SHORT", "priority_bucket": "near_favorite_zone", "setup_family": "general",
        "namespace": "live", "n_wins": "96", "n_losses": "4", "n_flats": "0",
        "n_entry_sessions": "20", "representative_closed_r": "0.1",
        "tracker_saved_at": "2026-09-18T16:20:00-04:00",
    }
    svc._OUTCOME_ROWS_THIS_BUILD = []
    try:
        grades = svc.read_setup_grades([row])
    finally:
        svc._OUTCOME_ROWS_THIS_BUILD = None
    [cell] = grades["swing"]
    assert cell["tape_n"] == 3 and cell["cum_r_lately"] == pytest.approx(-0.6)
    assert cell["tape_note"] == "tape: unknown"  # 3 is under the floor
    assert cell["grade"] == sg.B  # cum R < 0 blocks PROVEN


def test_no_scoring_snapshot_means_tape_and_cum_r_unknown(tape_files, tmp_path, monkeypatch):
    from datetime import date

    svc = tape_files
    monkeypatch.setattr(svc, "_scoring_snapshot_path", lambda: tmp_path / "absent.json")
    assert svc.read_swing_tape(date(2026, 9, 18), as_of="2026-09-18") is None


def test_the_prior_holdout_tape_is_rebuilt_when_the_day_rolls(tape_files, monkeypatch):
    from datetime import date

    svc = tape_files
    seen: list[str] = []
    real = svc._swing_tape_for

    def spy(setups, reference, *, as_of):
        seen.append(as_of)
        return real(setups, reference, as_of=as_of)

    monkeypatch.setattr(svc, "_swing_tape_for", spy)
    svc._swing(date(2026, 10, 10))
    monkeypatch.setattr(svc, "_last_completed_session", lambda: date(2026, 9, 21))
    svc._swing(date(2026, 10, 10))  # same files, new session
    assert seen == ["2026-09-18", "2026-09-21"]


def test_one_build_parses_the_scoring_snapshot_once(tape_files, tmp_path, monkeypatch):
    import json

    svc = tape_files
    setups = json.loads(svc._scoring_snapshot_path().read_text(encoding="utf-8"))["setups"]
    calls: list[int] = []

    def counted():
        calls.append(1)
        return setups

    monkeypatch.setattr(svc, "_scoring_setups", counted)
    monkeypatch.setattr(svc, "read_recent_rows", lambda: [{
        "side": "SHORT", "priority_bucket": "near_favorite_zone", "setup_family": "general",
        "namespace": "live", "n_wins": "96", "n_losses": "4", "n_flats": "0",
        "n_entry_sessions": "20", "representative_closed_r": "0.1",
        "tracker_saved_at": "2026-09-18T16:20:00-04:00",
    }])
    monkeypatch.setattr(svc, "read_favorable_read", lambda: None)
    monkeypatch.setattr(svc, "read_held_run_summaries", lambda: None)
    monkeypatch.setattr(svc, "_outcome_rows", lambda: [])
    service = svc.WorkingLatelyService(store_dir=tmp_path / "wl")
    payload = service.build_payload()
    assert payload["setup_grades"]["swing"][0]["tape_n"] == 3
    assert len(calls) == 1, calls
    assert svc._SETUPS_THIS_BUILD is None  # never held between builds


def test_the_line_says_why_tape_n_is_smaller_only_when_some_are_unknown():
    some = sg.grade_for(n=150, sessions=20, wins=140, avg_r=0.3, cum_r_lately=4.1,
                        tape={"wins": 71, "n": 114, "sessions": 18, "unknown": 36})
    assert sg.cell_line(some).endswith("n 114 (tape n excludes the newest ~5 sessions)")
    none = sg.grade_for(n=150, sessions=20, wins=140, avg_r=0.3, cum_r_lately=4.1,
                        tape={"wins": 71, "n": 114, "sessions": 18, "unknown": 0})
    assert sg.cell_line(none).endswith("n 114")


# ---------------------------------------------------------------------------
# the surfaces print the line
# ---------------------------------------------------------------------------


@pytest.fixture()
def _qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


_TAPE_CELL = sg.grade_for(
    n=150, sessions=20, wins=140, avg_r=0.3, cum_r_lately=4.1,
    tape={"wins": 71, "n": 114, "sessions": 18},
)
_DAY_CELL = sg.grade_for(n=40, sessions=12, wins=30, avg_r=0.5, cum_r_lately=20.0)


def test_the_setups_bucket_tooltip_leads_with_the_cell_line(_qapp):
    from PySide6.QtCore import Qt

    from ui.models.setup import SetupRow
    from ui.models.setup_table_model import SetupTableModel

    cell = {**_TAPE_CELL, "key": "SHORT|near_favorite_zone|general"}
    model = SetupTableModel([
        SetupRow(symbol="TSLA", side="SHORT", score=80.0, bucket="near_favorite_zone",
                 raw={"setup_family": "general"}),
    ])
    model.set_setup_grades({"swing": [cell]})
    bucket_col = [key for key, _ in model.COLUMNS].index("bucket")
    tip = model.index(0, bucket_col).data(Qt.ItemDataRole.ToolTipRole)
    assert tip.startswith(sg.cell_line(cell)), tip
    assert "win vs SPY 62%" in tip and "cum R +4.1" in tip and "n 114" in tip


def test_the_m5_row_tooltip_carries_the_grade_line(_qapp):
    from tests.test_st6_working_lately import _m5_alert
    from ui.widgets import m5_alert_bar as bar_module

    cell = {**_DAY_CELL, "key": "vwap|LONG", "bounce_type": "vwap", "side": "LONG"}
    bar = bar_module.M5AlertBar()
    try:
        bar.post(_m5_alert("BBB", "vwap", at="07:02:00"))
        bar.set_setup_grades({"daytrade": [cell]})
        tip = bar.list.item(0).toolTip()
    finally:
        bar.deleteLater()
    assert sg.cell_line(cell) in tip, tip
    assert "cum R +20.0" in tip


def test_the_live_strip_tooltip_carries_the_grade_line(_qapp, monkeypatch):
    import live_alert_results
    from tests.test_live_results_strip import NY, _alert, _strip

    from datetime import datetime

    monkeypatch.setattr(live_alert_results, "desk_zone", lambda: NY)
    strip, clock = _strip()
    strip.record(_alert("NVDA"))
    clock.moment = datetime(2026, 9, 22, 10, 45, 5, tzinfo=NY)
    strip.refresh()
    cell = {**_DAY_CELL, "key": "vwap|LONG", "bounce_type": "vwap", "side": "LONG"}
    strip.set_setup_grades({"daytrade": [cell]})
    assert sg.cell_line(cell) in strip.toolTip(), strip.toolTip()
