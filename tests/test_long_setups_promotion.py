"""p9 long setups, promoted: the Setup Tracker's Long leaders section, its grades, the
Focus candidates and the phone-report line. The trader, 2026-09-26: "The bot should
really promote these." Never the priority points or the buckets (SP4 stays the points
challenger).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import long_setups as ls  # noqa: E402

PAYLOAD = {
    "schema_version": 1, "as_of": "2026-09-25", "market_working": "yes", "market_rule": "trader",
    "rows": [
        {"symbol": "LEAD", "setup": ls.LEADER_PULLBACK, "as_of": "2026-09-25", "entry_limit": 99.5,
         "stop": 95.0, "stop_basis": "under the pullback low", "target": 101.5, "strength": 3.4,
         "exit": "take +1 ATR at 101.50 or sell after 10 sessions", "promoted": True, "status": "ready",
         "reasons": ["made a 52-week high in the last 120 sessions"]},
        {"symbol": "GAPR", "setup": ls.POST_EARNINGS_DRIFT, "as_of": "2026-09-25", "entry_limit": 50.0,
         "stop": 48.0, "target": 51.0, "strength": 1.5, "promoted": True, "status": "ready"},
    ],
}


# --- grading

def _hist(setup, day, outcome, ret=None, spy=None):
    return {"setup": setup, "as_of": day, "outcome": outcome, "return_pct": ret, "spy_return_pct": spy}


def test_long_setup_cells_grade_raw_first_and_count_no_fills_apart():
    import setup_grades

    rows = [
        _hist(ls.LEADER_PULLBACK, "2026-09-01", "filled", 3.0, 2.0),   # raw win, beats SPY
        _hist(ls.LEADER_PULLBACK, "2026-09-01", "filled", -1.0, 2.0),  # raw loss
        _hist(ls.LEADER_PULLBACK, "2026-09-02", "filled", 1.0, 0.0),   # flat SPY: tape only
        _hist(ls.LEADER_PULLBACK, "2026-09-03", "no_fill"),            # never a loss
        _hist(ls.LEADER_PULLBACK, "2026-09-04", ""),                   # unsettled: left out
        _hist(ls.LEADER_PULLBACK, "2026-09-05", "filled", 1.0, None),  # SPY unknown: left out
    ]
    cells = {cell["family"]: cell for cell in setup_grades.long_setup_cells(rows)}
    lead = cells[ls.LEADER_PULLBACK]
    assert lead["headline"] == "raw"
    assert (lead["raw"]["n"], lead["raw"]["wins"]) == (2, 1)
    assert (lead["tape"]["n"], lead["tape"]["wins"]) == (3, 2)
    assert lead["no_fill"] == 1
    line = setup_grades.long_setup_line(lead)
    assert line.startswith("leader_pullback LONG: raw in SPY-up")
    assert line.index("raw in SPY-up") < line.index("vs SPY") and "limit not filled 1" in line
    assert setup_grades.long_setup_line(cells[ls.POST_EARNINGS_DRIFT]) == \
        "post_earnings_drift LONG: no settled filled rows yet."


# --- the Setup Tracker section

def test_the_worker_reader_builds_the_section_from_the_files(tmp_path, monkeypatch):
    import project_paths
    from diagnostics.artifact_io import atomic_write_json
    from ui.services import working_lately_service as service

    current, history = tmp_path / "long_setups.json", tmp_path / "long_setups_history.json"
    atomic_write_json(current, PAYLOAD)
    atomic_write_json(history, {"rows": [_hist(ls.LEADER_PULLBACK, "2026-09-01", "filled", 3.0, 2.0)]})
    monkeypatch.setattr(project_paths, "LONG_SETUPS_FILE", current)
    monkeypatch.setattr(project_paths, "LONG_SETUPS_HISTORY_FILE", history)
    service._LOOKING_BACK_CACHE.clear()
    try:
        lines = service.read_long_leader_lines()
    finally:
        service._LOOKING_BACK_CACHE.clear()
    assert lines[0].startswith("Long leaders (scan session 2026-09-25): 2 setups, 2 ready")
    assert lines[1].startswith("1. LEAD leader pullback | buy limit 99.50 | stop 95.00")
    assert any(line.startswith("leader_pullback LONG: raw in SPY-up") for line in lines)


def test_the_tracker_worker_carries_the_section_and_survives_a_failure(monkeypatch):
    from ui.panels import setup_tracker_panel as module
    from ui.services import working_lately_service

    monkeypatch.setattr(working_lately_service, "read_long_leader_lines", lambda: ["a", "b"])
    assert module._read_tracker_exports(1)["long_leader_lines"] == ["a", "b"]

    def _boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(working_lately_service, "read_long_leader_lines", _boom)
    assert module._read_tracker_exports(1)["long_leader_lines"] == ["Long leaders: unreadable right now."]


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.mark.qt
def test_long_leaders_sit_at_the_top_of_the_setup_tracker(qapp):
    from ui.panels import setup_tracker_panel as module

    panel = module.SetupTrackerPanel()
    try:
        panel._on_exports_loaded({"signatures": {}, "ranked": {}, "raw": {}, "long_leader_lines": ["x", "y"]})
        assert panel.long_leaders_label.text() == "x\ny"
        layout = panel.layout()
        widgets = [layout.itemAt(i).widget() for i in range(layout.count())]
        # Right under the page header, above the KPI tiles and every other section.
        assert widgets.index(panel.long_leaders_label) == 1
    finally:
        panel.shutdown()
        panel.deleteLater()


# --- Focus and the phone

def test_promoted_rows_join_the_auto_populate_candidates(monkeypatch):
    import autopilot_core as core
    import long_setups_store

    monkeypatch.setattr(long_setups_store, "read_long_setups", lambda *a, **k: PAYLOAD)
    got = core.build_long_setup_candidates(today=date(2026, 9, 28))
    assert [row["symbol"] for row in got["longs"]] == ["LEAD", "GAPR"]
    assert got["shorts"] == []
    assert got["longs"][0]["reason"] == "Long leaders: leader pullback"
    waiting = {**PAYLOAD, "rows": [{**row, "promoted": False} for row in PAYLOAD["rows"]]}
    monkeypatch.setattr(long_setups_store, "read_long_setups", lambda *a, **k: waiting)
    assert core.build_long_setup_candidates(today=date(2026, 9, 28))["longs"] == []
    monkeypatch.setattr(long_setups_store, "read_long_setups", lambda *a, **k: None)
    assert core.build_long_setup_candidates(today=date(2026, 9, 28)) == {"longs": [], "shorts": []}


def test_the_auto_populate_pass_merges_them_and_still_runs_the_focus_gate(monkeypatch):
    import autopilot_core as core

    seen = {}
    monkeypatch.setattr(core, "load_universe_pool", lambda: ["AAA"])
    monkeypatch.setattr(core, "build_long_setup_candidates", lambda **k: {
        "longs": [{"symbol": "LEAD", "score": 6.0, "reason": "Long leaders: leader pullback"}], "shorts": []})

    def _context(pool, **_k):
        seen["pool"] = list(pool)
        return {}

    monkeypatch.setattr(core, "load_daily_context", _context)
    monkeypatch.setattr(core, "fetch_intraday_profiles", lambda pool, **k: {sym: {} for sym in pool})
    monkeypatch.setattr(core, "build_aggressive_regime_candidates", lambda *a, **k: {"longs": [], "shorts": []})
    monkeypatch.setattr(core, "build_relative_weakness_candidates", lambda *a, **k: {"longs": [], "shorts": []})
    monkeypatch.setattr(core, "build_adr_breakout_candidates", lambda *a, **k: {"longs": [], "shorts": []})

    def _gate(candidates, profiles, daily_context, log=None):
        seen["gated"] = [row["symbol"] for row in candidates["longs"]]
        return {"longs": [], "shorts": []}

    monkeypatch.setattr(core, "filter_candidates_by_prev_day_extremes", _gate)
    monkeypatch.setattr(core, "apply_auto_populated_watchlists", lambda candidates, env, **k: {})
    core.refresh_auto_populated_watchlists("neutral")
    assert "LEAD" in seen["pool"], "a long leader outside the universe is still measured"
    assert seen["gated"] == ["LEAD"], "the Focus gate judges every long leader"


def test_the_phone_report_prints_one_line_when_a_row_is_promoted():
    import autopilot_core as core

    line = ls.phone_line(PAYLOAD)
    assert line == ("Long leaders 2026-09-25: LEAD (leader pullback, limit 99.50, stop 95.00), "
                    "GAPR (post-earnings drift, limit 50.00, stop 48.00)")
    text = core.render_away_report({"long_leaders_line": line})
    assert "== LONG LEADERS ==\n" + line in text
    assert text.index("== LONG LEADERS ==") < text.index("== BEST SWING TRADES ==")
    assert "== LONG LEADERS ==" not in core.render_away_report({"long_leaders_line": ""})


def test_the_service_reads_the_phone_line_from_the_file(monkeypatch):
    import long_setups_store
    from ui.services import autopilot_service

    monkeypatch.setattr(long_setups_store, "read_long_setups", lambda *a, **k: PAYLOAD)
    assert autopilot_service._long_leaders_report_line().startswith("Long leaders 2026-09-25: LEAD")

    def _boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(long_setups_store, "read_long_setups", _boom)
    assert autopilot_service._long_leaders_report_line() == ""
