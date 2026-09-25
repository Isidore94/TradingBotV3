"""P2-9 9b: the hold-out window beside the window, in setup_grades and working_lately.

The last 20 sessions next to the 20 before them, the same statistic split by
date, so a leader that only led lately is visible. Under the floor says "n<30".
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_the_prior_window_is_the_twenty_sessions_just_before_the_recent_twenty():
    import evidence_stats
    import looking_back as lb
    import market_calendar

    windows = lb.split_windows("2026-09-18")
    recent, prior = windows["recent"], windows["prior"]
    assert recent == evidence_stats.lately_window("2026-09-18")
    assert recent[1] == "2026-09-18"
    assert market_calendar.previous_session(date.fromisoformat(recent[0])).isoformat() == prior[1]
    assert market_calendar.trading_days_between(
        date.fromisoformat(prior[0]), date.fromisoformat(prior[1])
    ) in (19, 20)
    assert prior[1] < recent[0]
    # The tracker's calendar-day window: the prior span ends the day before.
    assert lb.tracker_prior_reference(date(2026, 9, 24), 30) == date(2026, 8, 24)


def _bracket(event, day, won, side="long", bounce="vwap"):
    stamp = day.replace("-", "")
    return [
        {
            "event_id": f"{event}_{side}_{stamp}_09_45_00_{bounce}",
            "event_type": "final",
            "trade_date": day,
            "direction": side,
            "bars_elapsed": "1",
            "target_1r_hit": "True" if won else "False",
            "stop_hit": "False" if won else "True",
        }
    ]


def test_setup_grades_holdout_shows_the_same_ladder_on_each_window_and_the_floor():
    import setup_grades as sg

    recent_rows, prior_rows = [], []
    for index in range(40):
        recent_rows += _bracket(f"R{index}", "2026-09-1" + str(index % 9), index < 34)
    for index in range(12):
        prior_rows += _bracket(f"P{index}", "2026-08-1" + str(index % 9), index < 3)
    recent = sg.daytrade_cells(sg.bracket_results(recent_rows))
    prior = sg.daytrade_cells(sg.bracket_results(prior_rows))
    [row] = sg.holdout_view(recent, prior)
    assert row["key"] == "vwap|LONG"
    assert row["recent_grade"] == recent[0]["grade"] and row["recent_n"] == 40
    assert row["recent_text"].startswith(sg.badge(recent[0]["grade"]))
    assert "win 85%" in row["recent_text"] and "n=40" in row["recent_text"]
    # Twelve in the prior window: under the floor, and it says so.
    assert row["prior_text"] == "n<30 (n=12)"
    assert sg.holdout_view(recent, [])[0]["prior_text"] == sg.NO_PRIOR


def test_working_lately_holdout_puts_each_cell_beside_its_prior_window():
    import working_lately as wl

    common = dict(
        outcome_kind="k", outcome_version="v", knowledge_basis="b", horizon="h",
        window_sessions=20, latest_measured_session="2026-09-18", n_pending=0,
        n_excluded=0, n_symbols=5, n_sessions=5, top_symbol_share=None,
        top_session_share=None, statistic_name="win rate", uncertainty_kind="wilson",
        namespace="live", n_floor=30,
    )
    recent = [
        wl.EvidenceCell(kind="swing_trade_r", side="LONG", family="breakout", n_eligible=50,
                        statistic=0.7, uncertainty_low=0.56, n_graded=50, meets_floor=True, **common),
    ]
    prior = [
        wl.EvidenceCell(kind="swing_trade_r", side="long", family="breakout", n_eligible=12,
                        statistic=0.4, uncertainty_low=0.2, n_graded=12, meets_floor=False, **common),
        wl.EvidenceCell(kind="swing_trade_r", side="SHORT", family="fade", n_eligible=40,
                        statistic=0.6, uncertainty_low=0.45, n_graded=40, meets_floor=True, **common),
    ]
    rows = wl.holdout_view(recent, prior)
    assert [(r["side"], r["family"]) for r in rows] == [("LONG", "breakout"), ("SHORT", "fade")]
    assert rows[0]["recent_text"] == "0.70 (>= 0.56) n=50"
    assert rows[0]["prior_text"] == "n<30 (n=12)"
    assert rows[1]["recent_text"] == wl.HOLDOUT_NOT_IN_WINDOW
    assert rows[1]["prior_text"] == "0.60 (>= 0.45) n=40"


def test_the_service_splits_the_day_grades_by_date(tmp_path, monkeypatch):
    import looking_back as lb
    import setup_grades as sg
    from ui.services import working_lately_service as svc

    windows = lb.split_windows()
    recent_day, prior_day = windows["recent"][1], windows["prior"][1]
    recent_rows, prior_rows = [], []
    for index in range(35):
        recent_rows += _bracket(f"R{index}", recent_day, index < 30)
    for index in range(10):
        prior_rows += _bracket(f"P{index}", prior_day, index < 2)

    svc._LOOKING_BACK_CACHE.clear()
    monkeypatch.setattr(svc, "read_recent_rows", lambda: [])
    monkeypatch.setattr(svc, "read_favorable_read", lambda: None)
    monkeypatch.setattr(svc, "read_held_run_summaries", lambda: None)
    monkeypatch.setattr(svc, "_outcome_rows", lambda: list(recent_rows))
    monkeypatch.setattr(
        svc, "_stream_outcome_rows",
        lambda window: [row for row in prior_rows if lb.in_window(row["trade_date"], window)],
    )
    monkeypatch.setattr(svc, "_scoring_snapshot_path", lambda: tmp_path / "absent.json")
    service = svc.WorkingLatelyService(store_dir=tmp_path / "wl")
    payload = service.build_payload()

    day = payload["looking_back"]["holdout"]["day"]
    assert day["windows"]["recent"] == list(windows["recent"])
    assert day["windows"]["prior"] == list(windows["prior"])
    [row] = day["grades"]
    current = sg.daytrade_lookup(payload["setup_grades"])["vwap|LONG"]
    assert row["recent_grade"] == current["grade"], "the recent side is the published grade"
    assert row["recent_n"] == 35 and row["prior_n"] == 10
    assert row["prior_text"] == "n<30 (n=10)"
    # The grades file is untouched by the hold-out.
    on_disk = json.loads((tmp_path / "wl" / svc.GRADES_FILE_NAME).read_text(encoding="utf-8"))
    assert "holdout" not in on_disk

    # The Results view prints it: the day block, the floor stated.
    from PySide6.QtWidgets import QApplication

    from ui.widgets.looking_back_view import LookingBackView

    _app = QApplication.instance() or QApplication([])
    view = LookingBackView()
    try:
        view.set_reading(payload["looking_back"], "day")
        rows = view.holdout_rows()
        assert rows[0][0] == "Grade" and rows[0][1] == "vwap LONG"
        assert rows[0][3] == "n<30 (n=10)"
        assert windows["prior"][0] in view.holdout_label.text()
        view.set_reading(payload["looking_back"], "swing")
        assert view.holdout_rows() == []
    finally:
        view.deleteLater()
