"""S1 + S10c golden fixtures: a 2R grade, EOD close R and reach-2R beside the 1:1 grade.

The 1:1 bracket grade (+1R before -1R) is unchanged; these fixtures pin the
NEW fields only, plus the 1:1 fields of the same fixture so a regression in
either shows here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import setup_grades as sg  # noqa: E402

EID = "AAPL_long_20260921_09_45_00_{}"


def _row(event_id, bars, *, t1=False, t2=False, stop=False, final=False,
         close_r="", eod_close="", entry="100", risk="1.0", trade_date="2026-09-21"):
    return {
        "event_id": event_id,
        "event_type": "final" if final else "update",
        "trade_date": trade_date,
        "direction": "long",
        "bars_elapsed": str(bars),
        "target_1r_hit": str(t1),
        "target_2r_hit": str(t2),
        "stop_hit": str(stop),
        "close_r": close_r,
        "eod_close": eod_close,
        "entry_price": entry,
        "risk_per_share": risk,
    }


def test_bracket_results_carry_the_2r_result_eod_r_and_reach_2r():
    rows = [
        # a: +1R, then +2R, closes +1.5R
        _row(EID.format("a"), 1),
        _row(EID.format("a"), 2, t1=True),
        _row(EID.format("a"), 3, t1=True, t2=True),
        _row(EID.format("a"), 4, t1=True, t2=True, final=True, close_r="1.5"),
        # b: +1R, then the stop - 1:1 win, 2R loss
        _row(EID.format("b"), 1, t1=True),
        _row(EID.format("b"), 2, t1=True, stop=True),
        _row(EID.format("b"), 3, t1=True, stop=True, final=True, close_r="-0.8"),
        # c: +2R and the stop first on the same row - a loss, but it did reach 2R
        _row(EID.format("c"), 2, t1=True, t2=True, stop=True),
        _row(EID.format("c"), 3, t1=True, t2=True, stop=True, final=True, close_r="-1.1"),
        # d: nothing hit, finished
        _row(EID.format("d"), 5, final=True, close_r="0.3"),
        # e: nothing hit, still open
        _row(EID.format("e"), 5),
        # f: the old unsettled sentinel (close_r 0, eod_close == entry)
        _row(EID.format("f"), 5, final=True, close_r="0", eod_close="100"),
        # g: a blank close_r
        _row(EID.format("g"), 5, final=True),
        # h: a stop under the risk floor makes R meaningless
        _row(EID.format("h"), 5, final=True, close_r="50", risk="0.001"),
    ]
    got = {r["bounce_type"]: r for r in sg.bracket_results(rows)}
    table = {k: (v["result"], v["result_2r"], v["reached_2r"], v["eod_r"]) for k, v in got.items()}
    assert table == {
        "a": (sg.WIN, sg.WIN, True, 1.5),
        "b": (sg.WIN, sg.LOSS, False, -0.8),
        "c": (sg.LOSS, sg.LOSS, True, -1.1),
        "d": (sg.UNDECIDED, sg.UNDECIDED, False, 0.3),
        "e": (sg.OPEN, sg.OPEN, None, None),
        "f": (sg.UNDECIDED, sg.UNDECIDED, False, None),
        "g": (sg.UNDECIDED, sg.UNDECIDED, False, None),
        "h": (sg.UNDECIDED, sg.UNDECIDED, False, None),
    }


def _fixture_rows():
    """41 vwap LONG alerts: 12 reach +2R, 18 reach +1R then stop, 10 stop, 1 undecided."""
    rows = []
    for i in range(40):
        eid = f"AAPL_long_202609{(i % 12) + 1:02d}_09_{i:02d}_00_vwap"
        day = f"2026-09-{(i % 12) + 1:02d}"
        if i < 12:
            rows += [_row(eid, 1, t1=True, trade_date=day),
                     _row(eid, 2, t1=True, t2=True, final=True, close_r="2.0", trade_date=day)]
        elif i < 30:
            rows += [_row(eid, 1, t1=True, trade_date=day),
                     _row(eid, 2, t1=True, stop=True, final=True, close_r="-0.5", trade_date=day)]
        else:
            rows += [_row(eid, 1, stop=True, final=True, close_r="-1.2", trade_date=day)]
    rows.append(_row("AAPL_long_20260901_10_00_00_vwap", 3, final=True, close_r="0.1",
                     trade_date="2026-09-01"))
    return rows


def test_daytrade_cells_golden_2r_grade_eod_and_reach():
    cells = {c["key"]: c for c in sg.daytrade_cells(sg.bracket_results(_fixture_rows()))}
    cell = cells["vwap|LONG"]
    # 1:1 unchanged: 30 of 40 decided, 12 sessions, A.
    assert (cell["grade"], cell["n"], cell["wins"], cell["sessions"]) == (sg.A, 40, 30, 12)
    assert cell["avg_r"] == pytest.approx(0.5)
    assert cell["cum_r_lately"] == pytest.approx(20.0)
    assert cell["undecided"] == 1
    # 2R: 12 of 40 decided, avg R = 3p - 1, cum R = 2w - l.
    assert (cell["grade_2r"], cell["n_2r"], cell["wins_2r"], cell["sessions_2r"]) == (sg.D, 40, 12, 12)
    assert cell["win_rate_2r"] == pytest.approx(0.3)
    assert cell["low_bound_2r"] == pytest.approx(sg.wilson_lower_bound(12, 40))
    assert cell["avg_r_2r"] == pytest.approx(-0.1)
    assert cell["cum_r_2r"] == pytest.approx(-4.0)
    assert cell["undecided_2r"] == 1
    # EOD close R over the 41 finished alerts.
    assert cell["eod_n"] == 41
    assert cell["eod_r_mean"] == pytest.approx(3.1 / 41, abs=1e-4)
    assert cell["eod_r_median"] == pytest.approx(-0.5)
    # Reach 2R over the 41 finished alerts.
    assert (cell["reach_2r_hits"], cell["reach_2r_n"]) == (12, 41)
    assert cell["reach_2r_rate"] == pytest.approx(12 / 41)


def test_the_2r_grade_climbs_the_same_ladder():
    rows = []
    for i in range(40):
        eid = f"AAPL_long_202609{(i % 12) + 1:02d}_09_{i:02d}_00_vwap"
        day = f"2026-09-{(i % 12) + 1:02d}"
        hit = i < 30
        rows.append(_row(eid, 1, t1=True, t2=hit, stop=not hit, final=True,
                         close_r="2.0" if hit else "-1.0", trade_date=day))
    cell = sg.daytrade_cells(sg.bracket_results(rows))[0]
    assert cell["grade_2r"] == sg.A  # 30/40, low ~0.60, 12 sessions, avg R +1.25
    assert cell["avg_r_2r"] == pytest.approx(1.25)


def test_the_cell_line_reads_1to1_2r_eod_and_n():
    cell = sg.daytrade_cells(sg.bracket_results(_fixture_rows()))[0]
    assert sg.cell_line(cell) == "1:1 A · 2R D · EOD +0.08R · n 40"


def test_the_cell_line_says_unknown_eod_and_new_2r_when_unmeasured():
    rows = [_row(EID.format("vwap"), 1, t1=True)]
    cell = sg.daytrade_cells(sg.bracket_results(rows))[0]
    assert sg.cell_line(cell) == "1:1 NEW · 2R NEW · EOD unknown · n 1"


def test_old_results_without_the_new_fields_still_grade_1to1():
    results = [{"event_id": f"e{i}", "trade_date": "2026-09-01", "side": "LONG",
                "bounce_type": "vwap", "result": sg.WIN if i < 20 else sg.LOSS} for i in range(30)]
    cell = sg.daytrade_cells(results)[0]
    assert (cell["n"], cell["wins"]) == (30, 20)
    assert cell["n_2r"] == 0 and cell["grade_2r"] == sg.NEW
    assert cell["eod_n"] == 0 and cell["eod_r_mean"] is None and cell["reach_2r_rate"] is None


def test_the_rules_name_both_brackets():
    assert "+1R before -1R" in sg.RULES_TEXT
    assert "2R grade: +2R before -1R" in sg.RULES_TEXT


# ---------------------------------------------------------------------------
# the Daytrade Tracker shows them
# ---------------------------------------------------------------------------


def test_the_tracker_columns_name_the_1to1_bracket_and_the_new_numbers():
    from ui.panels.daytrade_tracker_panel import PERFORMANCE_COLUMNS

    labels = dict(PERFORMANCE_COLUMNS)
    assert labels["grade_1r"] == "1:1 bracket"
    assert labels["grade_2r"] == "2R grade"
    for key in ("eod_r_mean", "eod_r_median", "reach_2r_rate", "grade_n"):
        assert key in labels, key


def test_the_tracker_joins_grades_onto_bounce_type_rows_only():
    from ui.panels.daytrade_tracker_panel import apply_setup_grades

    payload = {"daytrade": sg.daytrade_cells(sg.bracket_results(_fixture_rows()))}
    rows = apply_setup_grades(
        [{"dimension": "bounce_type", "direction": "long", "segment": "vwap"},
         {"dimension": "bounce_type", "direction": "short", "segment": "vwap"},
         {"dimension": "time_bucket", "direction": "long", "segment": "vwap"}],
        payload,
    )
    assert rows[0]["grade_1r"] == "A" and rows[0]["grade_2r"] == "D"
    assert rows[0]["eod_r_mean"] == pytest.approx(3.1 / 41, abs=1e-4)
    assert rows[0]["eod_r_median"] == pytest.approx(-0.5)
    assert rows[0]["reach_2r_rate"] == pytest.approx(12 / 41)
    assert rows[0]["grade_n"] == 40
    for blank in rows[1:]:
        assert blank["grade_1r"] == "" and blank["eod_r_mean"] is None, blank
