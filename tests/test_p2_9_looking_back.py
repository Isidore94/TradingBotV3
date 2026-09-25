"""P2-9 Looking back: pick equity curves (9a) and the hold-out window (9b).

Display only: cumulative R with n per population, built from the per-pick R the
existing cells already average, and the same statistics split by date.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# 9a - the curve arithmetic
# ---------------------------------------------------------------------------


def test_the_curve_sums_r_by_session_in_date_order_and_counts_n():
    import looking_back as lb

    results = [
        {"session": "2026-09-03", "r": -1.0},
        {"session": "2026-09-01", "r": 0.5},
        {"session": "2026-09-01", "r": 1.5},
        {"session": "2026-09-02", "r": None},  # pending: counted apart, adds nothing
        {"session": "2026-09-03", "r": 2.25},
        {"session": "", "r": 1.0},  # undated: not graded
    ]
    curve = lb.equity_curve(results, population=lb.SWING)
    assert [
        (p["session"], p["day_r"], p["day_n"], p["cum_r"], p["cum_n"]) for p in curve["points"]
    ] == [
        ("2026-09-01", 2.0, 2, 2.0, 2),
        ("2026-09-03", 1.25, 2, 3.25, 4),
    ]
    assert curve["n"] == 4 and curve["total_r"] == 3.25
    assert curve["avg_r"] == pytest.approx(0.8125)
    assert curve["not_graded"] == 2
    assert (curve["first_session"], curve["last_session"]) == ("2026-09-01", "2026-09-03")
    assert "+3.25R over n=4" in lb.curve_line(curve)
    assert lb.curve_line(lb.equity_curve([], population=lb.M5)) == "no graded picks yet"


def _out(event_id, bars, target, stop, *, trade_date, event_type="update", direction="long"):
    return {
        "event_id": event_id,
        "event_type": event_type,
        "trade_date": trade_date,
        "direction": direction,
        "bars_elapsed": str(bars),
        "target_1r_hit": "True" if target else "False",
        "stop_hit": "True" if stop else "False",
    }


def test_m5_alerts_are_the_bracket_results_as_plus_or_minus_one_r():
    import looking_back as lb

    rows = [
        _out("A_long_20260901_09_45_00_vwap", 1, True, False, trade_date="2026-09-01"),
        _out("B_long_20260901_09_50_00_ema", 1, False, True, trade_date="2026-09-01"),
        _out("C_long_20260902_09_45_00_vwap", 1, True, False, trade_date="2026-09-02"),
        _out("D_long_20260902_09_45_00_vwap", 3, False, False, trade_date="2026-09-02",
             event_type="final"),
    ]
    results = sorted(lb.m5_alert_results(rows), key=lambda r: r["family"] + r["session"])
    assert [(r["session"], r["r"]) for r in results] == [
        ("2026-09-01", -1.0),
        ("2026-09-01", 1.0),
        ("2026-09-02", 1.0),
        ("2026-09-02", None),
    ]
    curve = lb.equity_curve(results, population=lb.M5)
    assert [(p["session"], p["cum_r"], p["cum_n"]) for p in curve["points"]] == [
        ("2026-09-01", 0.0, 2),
        ("2026-09-02", 1.0, 3),
    ]
    assert curve["not_graded"] == 1


def _setup(scan_date, *, symbol="AAA", anchor="2026-08-01", status="closed", r=0.5,
           family="general", exit_date=""):
    return {
        "setup_id": f"{scan_date}:{symbol}:LONG:{anchor}:favorite_setup",
        "symbol": symbol,
        "side": "LONG",
        "scan_date": scan_date,
        "anchor_date": anchor,
        "priority_bucket": "favorite_setup",
        "setup_family": family,
        "_scoring_outcome_summary": {
            "tradeable_scenario_count": 4,
            "closed_tradeable_scenario_count": 4 if status == "closed" else 0,
            "representative_closed_r": r if status == "closed" else None,
            "representative_status": status,
            "representative_exit_date": exit_date,
        },
    }


def _context(setup):
    return (setup["side"], setup["priority_bucket"], setup["setup_family"])


def test_swing_picks_are_one_per_episode_dated_by_scan_date_and_open_ones_are_pending():
    import looking_back as lb

    setups = {
        # One idea re-scanned on three days: ONE episode, the first actionable row.
        "a1": _setup("2026-09-01", r=1.2, exit_date="2026-09-05"),
        "a2": _setup("2026-09-02", r=0.9, exit_date="2026-09-05"),
        "a3": _setup("2026-09-03", r=0.7, exit_date="2026-09-05"),
        "b": _setup("2026-09-02", symbol="BBB", r=-1.0),
        "c": _setup("2026-09-04", symbol="CCC", status="pending", r=None),
        # Untradeable and future rows are outside the population.
        "d": {**_setup("2026-09-02", symbol="DDD"),
              "_scoring_outcome_summary": {"tradeable_scenario_count": 0}},
        "e": _setup("2026-09-30", symbol="EEE", r=3.0),
    }
    results = lb.swing_pick_results(setups, reference=date(2026, 9, 10), context=_context)
    assert sorted((r["symbol"], r["session"], r["r"]) for r in results) == [
        ("AAA", "2026-09-01", 1.2),
        ("BBB", "2026-09-02", -1.0),
        ("CCC", "2026-09-04", None),
    ]
    curve = lb.equity_curve(results, population=lb.SWING)
    assert [(p["session"], p["cum_r"], p["cum_n"]) for p in curve["points"]] == [
        ("2026-09-01", 1.2, 1),
        ("2026-09-02", 0.2, 2),
    ]
    assert curve["not_graded"] == 1


def test_the_two_populations_are_never_pooled():
    import looking_back as lb

    payload = lb.build_payload(
        swing_results=[{"session": "2026-09-01", "r": 2.0}],
        m5_results=[{"session": "2026-09-01", "r": -1.0}],
        as_of="2026-09-01",
    )
    assert payload["schema"] == lb.SCHEMA
    assert payload["curves"][lb.SWING]["total_r"] == 2.0
    assert payload["curves"][lb.M5]["total_r"] == -1.0
    assert set(payload["curves"]) == {lb.SWING, lb.M5}
    json.dumps(payload)  # the service writes it as JSON


# ---------------------------------------------------------------------------
# 9a - the service builds it on the worker and the Results page draws it
# ---------------------------------------------------------------------------


def _service_with_fixture(tmp_path, monkeypatch):
    import looking_back as lb
    from ui.services import working_lately_service as svc

    windows = lb.split_windows()
    recent_day = windows["recent"][1]
    prior_day = windows["prior"][1]
    stamp_recent = recent_day.replace("-", "")
    stamp_prior = prior_day.replace("-", "")
    recent_rows = [
        _out(f"A_long_{stamp_recent}_09_45_00_vwap", 1, True, False, trade_date=recent_day),
        _out(f"B_long_{stamp_recent}_09_50_00_ema", 1, False, True, trade_date=recent_day),
        _out(f"C_long_{stamp_recent}_09_55_00_ema", 1, True, False, trade_date=recent_day),
    ]
    prior_rows = [
        _out(f"D_long_{stamp_prior}_09_45_00_vwap", 1, True, False, trade_date=prior_day),
    ]
    streamed = []

    def fake_stream(window):
        streamed.append(tuple(window))
        return [row for row in prior_rows if lb.in_window(row["trade_date"], window)]

    snapshot = tmp_path / "scoring_snapshot.json"
    snapshot.write_text(
        json.dumps(
            {"setups": {"a": _setup("2026-09-01", r=2.5), "b": _setup("2026-09-02", symbol="BBB", r=-1.0)}}
        ),
        encoding="utf-8",
    )
    svc._LOOKING_BACK_CACHE.clear()
    monkeypatch.setattr(svc, "read_recent_rows", lambda: [])
    monkeypatch.setattr(svc, "read_favorable_read", lambda: None)
    monkeypatch.setattr(svc, "read_held_run_summaries", lambda: None)
    monkeypatch.setattr(svc, "read_setup_grades", lambda _rows: None)
    monkeypatch.setattr(svc, "_outcome_rows", lambda: list(recent_rows))
    monkeypatch.setattr(svc, "_stream_outcome_rows", fake_stream)
    monkeypatch.setattr(svc, "_scoring_snapshot_path", lambda: snapshot)
    # The prior-window cache keys on the log's mtime/size; a missing log is never cached.
    log = tmp_path / "intraday_bounce_outcomes.csv"
    log.write_text("event_id,trade_date\n", encoding="utf-8")
    monkeypatch.setattr(svc, "_outcome_log_path", lambda: log)
    return svc, streamed, windows


def test_the_service_builds_both_curves_and_writes_them_beside_the_snapshot(tmp_path, monkeypatch):
    import looking_back as lb

    svc, streamed, windows = _service_with_fixture(tmp_path, monkeypatch)
    service = svc.WorkingLatelyService(store_dir=tmp_path / "wl")
    payload = service.build_payload()
    looking = payload["looking_back"]
    swing = looking["curves"][lb.SWING]
    m5 = looking["curves"][lb.M5]
    assert (swing["n"], swing["total_r"]) == (2, 1.5)
    # prior window +1, recent window +1 -1 +1: one curve over both windows.
    assert (m5["n"], m5["total_r"]) == (4, 2.0)
    assert [p["session"] for p in m5["points"]] == [windows["prior"][1], windows["recent"][1]]
    on_disk = json.loads((tmp_path / "wl" / svc.LOOKING_BACK_FILE_NAME).read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(looking))
    snapshot = json.loads(service.snapshot_path.read_text(encoding="utf-8"))
    assert "looking_back" not in snapshot, "the persisted snapshot is unchanged"
    # The prior window is read once per window, not once per build.
    service.build_payload()
    assert streamed == [tuple(windows["prior"])]


def test_the_results_page_draws_the_curve_for_the_chosen_horizon(monkeypatch):
    import looking_back as lb
    from test_g5_research_results_panel import _forget_the_saved_selection, _settle
    from ui.panels import research_results_panel as module

    reading = lb.build_payload(
        swing_results=[{"session": "2026-09-01", "r": 2.0}, {"session": "2026-09-02", "r": -0.5}],
        m5_results=[{"session": "2026-09-03", "r": 1.0}],
        as_of="2026-09-03",
    )
    monkeypatch.setattr(module, "read_persisted_snapshot", lambda *a, **k: {})
    monkeypatch.setattr(module, "read_persisted_looking_back", lambda *a, **k: {})
    monkeypatch.setattr(module, "load_trades", lambda *a, **k: [])
    _forget_the_saved_selection()
    panel = module.ResearchResultsPanel()
    try:
        _settle(panel)
        panel.set_working_lately_snapshot({"as_of": "2026-09-03", "looking_back": reading})
        _settle(panel)
        view = panel.looking_back_view
        assert not view.isHidden()
        assert view.population() == lb.SWING
        assert [p["cum_r"] for p in view.curve_points()] == [2.0, 1.5]
        assert "+1.50R over n=2" in view.line_label.text()

        panel.horizon_buttons["day"].setChecked(True)
        _settle(panel)
        assert view.population() == lb.M5
        assert [p["cum_r"] for p in view.curve_points()] == [1.0]

        panel.population_buttons["mine"].setChecked(True)
        _settle(panel)
        assert view.isHidden(), "My trades has no bot pick curve"
    finally:
        panel.shutdown()
        _settle(panel)
        panel.deleteLater()
        _forget_the_saved_selection()
