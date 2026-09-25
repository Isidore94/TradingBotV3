"""The setup grade ladder (trader, 2026-09-22): PROVEN / A / B / C / D / New.

Swing grades come from the Setup Tracker's recent family rows; day-trade grades
from the outcome log as a +1R-before-1R bracket. The desk shows the grade and,
with the priority switch on (now the default), puts the best-graded rows first.
Presentation only - every row stays on screen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import setup_grades as sg  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------


def test_the_ladder_uses_counts_sessions_the_low_bound_and_avg_r():
    # 29 closed is New however good it looks.
    assert sg.grade_for(n=29, sessions=20, wins=29, avg_r=1.0)["grade"] == sg.NEW
    # 100+ closed, 15+ sessions, low bound >= 60%, avg R > 0, cum R >= 0 -> PROVEN.
    proven = sg.grade_for(n=200, sessions=20, wins=150, avg_r=0.3, cum_r_lately=60.0)
    assert proven["grade"] == sg.PROVEN
    assert proven["low_bound"] == pytest.approx(0.6857, abs=1e-3)
    # The same record with a non-positive avg R cannot be PROVEN or A.
    assert sg.grade_for(n=200, sessions=20, wins=150, avg_r=0.0, cum_r_lately=0.0)["grade"] == sg.B
    # Too few sessions for PROVEN, enough for A.
    assert sg.grade_for(n=200, sessions=12, wins=150, avg_r=0.3, cum_r_lately=60.0)["grade"] == sg.A
    # 60 closed at 70%: low bound ~0.57 -> A with 10+ sessions, B with fewer.
    assert sg.grade_for(n=60, sessions=10, wins=42, avg_r=0.2, cum_r_lately=12.0)["grade"] == sg.A
    assert sg.grade_for(n=60, sessions=9, wins=42, avg_r=0.2, cum_r_lately=12.0)["grade"] == sg.B
    # 55% on 100: low bound ~0.45, rate >= 50% -> C; 45% -> D.
    assert sg.grade_for(n=100, sessions=20, wins=55, avg_r=0.1)["grade"] == sg.C
    assert sg.grade_for(n=100, sessions=20, wins=45, avg_r=-0.1)["grade"] == sg.D


def test_new_sorts_above_d_and_below_c():
    ranks = [sg.sort_rank(g) for g in (sg.PROVEN, sg.A, sg.B, sg.C, sg.NEW, sg.D)]
    assert ranks == sorted(ranks)


# ---------------------------------------------------------------------------
# swing
# ---------------------------------------------------------------------------


def _family_row(**overrides):
    row = {
        "side": "SHORT",
        "priority_bucket": "near_favorite_zone",
        "setup_family": "avwape_to_1stdev",
        "namespace": "live",
        "n_wins": "90",
        "n_losses": "8",
        "n_flats": "2",
        "n_entry_sessions": "18",
        "win_rate_closed": "0.99",  # recency-weighted: must NOT be used
        "representative_closed_r": "0.38",
        "avg_closed_r": "-5",
    }
    row.update(overrides)
    return row


def test_swing_cells_count_wins_and_leave_study_groups_out():
    # P8-P6: PROVEN also needs a known cum R >= 0 (here with the tape unknown).
    cells = sg.swing_cells(
        [_family_row(), _family_row(namespace="study", setup_family="hv_level_break")],
        {"SHORT|near_favorite_zone|avwape_to_1stdev": {"cum_r_lately": 38.0}},
    )
    assert len(cells) == 1
    cell = cells[0]
    assert cell["n"] == 100 and cell["wins"] == 90  # a flat is not a win
    assert cell["win_rate"] == pytest.approx(0.90)
    assert cell["avg_r"] == pytest.approx(0.38)  # representative R wins over the mean
    assert cell["grade"] == sg.PROVEN
    assert cell["key"] == "SHORT|near_favorite_zone|avwape_to_1stdev"


def test_swing_avg_r_falls_back_to_the_mean_and_blank_is_unmeasured():
    [cell] = sg.swing_cells([_family_row(representative_closed_r="", avg_closed_r="-0.2")])
    assert cell["avg_r"] == pytest.approx(-0.2)
    assert cell["grade"] == sg.B  # positive R needed for A/PROVEN
    [cell] = sg.swing_cells([_family_row(representative_closed_r="", avg_closed_r="")])
    assert cell["avg_r"] is None and cell["grade"] == sg.B


# ---------------------------------------------------------------------------
# day trade: +1R before -1R
# ---------------------------------------------------------------------------


def _out(event_id, bars, target, stop, event_type="update", trade_date="2026-09-21", direction="long"):
    return {
        "event_id": event_id,
        "event_type": event_type,
        "trade_date": trade_date,
        "direction": direction,
        "bars_elapsed": str(bars),
        "target_1r_hit": "True" if target else "False",
        "stop_hit": "True" if stop else "False",
    }


def test_the_first_row_that_sets_a_flag_decides_and_a_tie_is_a_loss():
    eid = "AAPL_long_20260921_09_45_00_{}"
    rows = [
        # win: target first, stop later (flags are cumulative)
        _out(eid.format("vwap"), 4, True, True),
        _out(eid.format("vwap"), 2, True, False),  # out of file order on purpose
        _out(eid.format("vwap"), 1, False, False),
        # loss: stop first
        _out(eid.format("ema"), 3, False, True),
        _out(eid.format("ema"), 6, True, True),
        # tie on the same row: loss
        _out(eid.format("tie"), 2, True, True),
        # neither, finished: undecided; neither, unfinished: open
        _out(eid.format("flat"), 5, False, False, event_type="final"),
        _out(eid.format("live"), 5, False, False),
    ]
    results = {r["bounce_type"]: r["result"] for r in sg.bracket_results(rows)}
    assert results == {
        "vwap": sg.WIN,
        "ema": sg.LOSS,
        "tie": sg.LOSS,
        "flat": sg.UNDECIDED,
        "live": sg.OPEN,
    }


def test_daytrade_cells_count_each_bounce_type_and_derive_bracket_r():
    results = []
    for index in range(40):
        results.append(
            {
                "event_id": f"e{index}",
                "trade_date": f"2026-09-{(index % 12) + 1:02d}",
                "side": "LONG",
                "bounce_type": "eod_vwap-vwap",
                "result": sg.WIN if index < 30 else sg.LOSS,
            }
        )
    results.append({"event_id": "u", "trade_date": "2026-09-01", "side": "LONG",
                    "bounce_type": "vwap", "result": sg.UNDECIDED})
    cells = {cell["key"]: cell for cell in sg.daytrade_cells(results)}
    assert set(cells) == {"eod_vwap|LONG", "vwap|LONG"}
    vwap = cells["vwap|LONG"]
    assert vwap["n"] == 40 and vwap["wins"] == 30 and vwap["undecided"] == 1
    assert vwap["sessions"] == 12
    assert vwap["avg_r"] == pytest.approx(0.5)  # 2 * 0.75 - 1
    assert vwap["grade"] == sg.A  # low bound ~0.60, 12 sessions, positive R


def test_daytrade_order_lists_graded_types_best_first_and_leaves_new_and_d_out():
    payload = {
        "daytrade": [
            {"bounce_type": "c_type", "side": "LONG", "grade": sg.C, "low_bound": 0.47, "n": 90},
            {"bounce_type": "a_type", "side": "SHORT", "grade": sg.A, "low_bound": 0.58, "n": 60},
            {"bounce_type": "d_type", "side": "LONG", "grade": sg.D, "low_bound": 0.30, "n": 90},
            {"bounce_type": "new_type", "side": "LONG", "grade": sg.NEW, "low_bound": 0.9, "n": 5},
        ]
    }
    assert sg.daytrade_order(payload) == [("a_type", "SHORT"), ("c_type", "LONG")]


def test_an_alert_takes_its_best_graded_type_and_a_measured_d_stays_d():
    lookup = sg.daytrade_lookup(
        {
            "daytrade": [
                {"key": "vwap|LONG", "grade": sg.C},
                {"key": "ema|LONG", "grade": sg.A},
                {"key": "h1|LONG", "grade": sg.D},
            ]
        }
    )
    assert sg.daytrade_grade_for_alert(lookup, "vwap;ema", "long") == sg.A
    assert sg.daytrade_grade_for_alert(lookup, "h1", "LONG") == sg.D
    assert sg.daytrade_grade_for_alert(lookup, "unseen", "LONG") == sg.NEW


# ---------------------------------------------------------------------------
# the desk
# ---------------------------------------------------------------------------


def _set_switch(on) -> None:
    import project_paths

    if on is None:
        path = project_paths.LOCAL_SETTINGS_FILE
        try:
            settings = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            settings = {}
        settings.pop("prioritise_working_lately", None)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings), encoding="utf-8")
    else:
        project_paths.save_local_setting("prioritise_working_lately", bool(on))
    project_paths.invalidate_local_settings_cache()


def test_the_switch_now_defaults_on():
    import working_lately

    _set_switch(None)
    assert working_lately.prioritise_enabled() is True
    _set_switch(False)
    assert working_lately.prioritise_enabled() is False
    _set_switch(None)


_SWING_GRADES = {
    "swing": [
        {"key": "LONG|favorite_setup|avwape_to_1stdev", "grade": sg.C, "low_bound": 0.50,
         "n": 335, "sessions": 20, "win_rate": 0.55, "avg_r": -0.32},
        {"key": "SHORT|near_favorite_zone|avwape_to_1stdev", "grade": sg.PROVEN,
         "low_bound": 0.83, "n": 390, "sessions": 20, "win_rate": 0.87, "avg_r": 0.38},
        {"key": "LONG|near_favorite_zone|thin", "grade": sg.D, "low_bound": 0.2,
         "n": 40, "sessions": 10, "win_rate": 0.3, "avg_r": -0.5},
    ]
}


def _setup_rows():
    from ui.models.setup import SetupRow

    return [
        SetupRow(symbol="NVDA", side="LONG", score=120.0, bucket="favorite_setup",
                 raw={"setup_family": "avwape_to_1stdev"}),
        SetupRow(symbol="XYZ", side="LONG", score=50.0, bucket="near_favorite_zone",
                 raw={"setup_family": "thin"}),
        SetupRow(symbol="NEWCO", side="LONG", score=60.0, bucket="near_favorite_zone",
                 raw={"setup_family": "never_seen"}),
        SetupRow(symbol="TSLA", side="SHORT", score=80.0, bucket="near_favorite_zone",
                 raw={"setup_family": "avwape_to_1stdev"}),
    ]


def test_the_setups_table_puts_the_best_grade_first_and_hides_nothing():
    from ui.panels.master_avwap_panel import MasterAvwapPanel

    def _run(switch):
        _set_switch(switch)
        panel = MasterAvwapPanel(None)
        try:
            panel.set_rows(_setup_rows())
            panel.set_setup_grades(_SWING_GRADES)
            symbols = [row.symbol for row in panel.filtered_rows()]
            bucket_col = [key for key, _ in panel.model.COLUMNS].index("bucket")
            cells = {
                panel.model.rows()[i].symbol: panel.model.index(i, bucket_col).data()
                for i in range(panel.model.rowCount())
            }
            return symbols, cells
        finally:
            panel.deleteLater()

    on_symbols, cells = _run(None)  # the default
    assert on_symbols == ["TSLA", "NVDA", "NEWCO", "XYZ"], on_symbols  # PROVEN, C, New, D
    assert cells["TSLA"].startswith("PROVEN · ")
    assert cells["NVDA"].startswith("C · ")
    assert cells["NEWCO"].startswith("NEW · ")
    off_symbols, _ = _run(False)
    assert off_symbols == ["NVDA", "XYZ", "NEWCO", "TSLA"], off_symbols  # arrival order
    assert sorted(on_symbols) == sorted(off_symbols)
    _set_switch(None)


def test_without_grades_the_bucket_cell_reads_as_before():
    from ui.models.setup_table_model import SetupTableModel

    model = SetupTableModel(_setup_rows())
    bucket_col = [key for key, _ in model.COLUMNS].index("bucket")
    assert model.index(0, bucket_col).data() == model.rows()[0].bucket_display


def test_the_m5_bar_shows_the_tracker_grade_and_sorts_by_it():
    from tests.test_st6_working_lately import _m5_alert
    from ui.widgets import m5_alert_bar as bar_module

    grades = {
        "daytrade": [
            {"key": "vwap|LONG", "bounce_type": "vwap", "side": "LONG", "grade": sg.C,
             "low_bound": 0.47, "n": 90},
            {"key": "ema_15|LONG", "bounce_type": "ema_15", "side": "LONG", "grade": sg.D,
             "low_bound": 0.24, "n": 50},
        ]
    }
    _set_switch(None)
    bar = bar_module.M5AlertBar()
    try:
        bar.set_working_lately_order(sg.daytrade_order(grades))
        bar.post(_m5_alert("AAA", "ema_15", at="07:01:00"))
        bar.post(_m5_alert("BBB", "vwap", at="07:02:00"))
        bar.post(_m5_alert("CCC", "unseen", at="07:03:00"))
        bar.set_setup_grades(grades)
        texts = [bar.list.item(i).text() for i in range(bar.list.count())]
    finally:
        bar.deleteLater()
    assert texts[0].startswith("[C]") and "BBB" in texts[0], texts
    assert {t.split()[0] for t in texts} == {"[C]", "[D]", "[NEW]"}, texts


def test_the_service_writes_the_grades_beside_the_snapshot(tmp_path, monkeypatch):
    from ui.services import working_lately_service as svc

    monkeypatch.setattr(svc, "read_recent_rows", lambda: [_family_row()])
    monkeypatch.setattr(svc, "read_favorable_read", lambda: None)
    monkeypatch.setattr(svc, "read_held_run_summaries", lambda: None)
    monkeypatch.setattr(
        svc, "_outcome_rows", lambda: [_out("AAPL_long_20260921_09_45_00_vwap", 2, True, False)]
    )
    # P8-P6: the swing cell's cum R (PROVEN needs it >= 0), off the live stores.
    monkeypatch.setattr(
        svc, "read_swing_tape",
        lambda *_a, **_k: {"SHORT|near_favorite_zone|avwape_to_1stdev": {"cum_r_lately": 38.0}},
    )
    service = svc.WorkingLatelyService(store_dir=tmp_path / "wl")
    payload = service.build_payload()
    grades = payload["setup_grades"]
    assert grades["schema"] == sg.SCHEMA
    assert grades["swing"][0]["grade"] == sg.PROVEN
    assert grades["daytrade"][0]["key"] == "vwap|LONG"
    on_disk = json.loads((tmp_path / "wl" / svc.GRADES_FILE_NAME).read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(grades, default=str))
    snapshot = json.loads(service.snapshot_path.read_text(encoding="utf-8"))
    assert "setup_grades" not in snapshot, "the persisted snapshot is unchanged"
