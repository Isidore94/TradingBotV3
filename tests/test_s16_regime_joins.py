"""S16 item 3: everything joins on the trader's regime of its date.

Trader, 2026-09-26: "what's important is KNOWING the market regime and then having
setups you KNOW work in it ... it just is what it is."
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import regime_grades as rg  # noqa: E402
import regime_join as rj  # noqa: E402
import setup_grades as sg  # noqa: E402

SEGMENTS = [
    {"segment_id": 1, "start_date": "2026-03-01", "regime": "bull_run"},
    {"segment_id": 2, "start_date": "2026-06-01", "regime": "weekly_hh_then_compression"},
    {"segment_id": 3, "start_date": "2026-08-01", "regime": "bear_channel_lower_highs"},
]
TIMELINE = rj.timeline(SEGMENTS)


# ---------------------------------------------------------------- the join


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        ("2026-02-28", rj.UNKNOWN),  # before the first segment
        ("2026-03-01", "bull_run"),  # a start date is inside its segment
        ("2026-05-31", "bull_run"),
        ("2026-06-01", "weekly_hh_then_compression"),
        ("2026-07-31", "weekly_hh_then_compression"),
        ("2026-08-01", "bear_channel_lower_highs"),
        ("2026-09-25T15:59:00-04:00", "bear_channel_lower_highs"),
        ("", rj.UNKNOWN),
        ("not a date", rj.UNKNOWN),
    ],
)
def test_the_join_boundaries(day, expected):
    assert rj.label_for(day, TIMELINE) == expected


def test_no_segments_is_unknown_never_a_guess():
    assert rj.label_for("2026-09-01", []) == rj.UNKNOWN
    assert rj.label_for("2026-09-01", None) == rj.UNKNOWN


def test_a_backdated_segment_typed_later_labels_older_rows():
    # The trader typed the bull run on 09-26 with start 03-01: it is the trader's label.
    rows = [{"segment_id": 7, "start_date": "2026-03-01", "regime": "bull_run",
             "entered_at": "2026-09-26T10:00:00-04:00"}]
    assert rj.label_for("2026-04-15", rj.timeline(rows)) == "bull_run"


def test_a_superseded_segment_does_not_label():
    rows = [
        {"segment_id": 1, "start_date": "2026-08-01", "regime": "range"},
        {"segment_id": 2, "start_date": "2026-08-01", "regime": "bear_channel_lower_highs", "supersedes": 1},
    ]
    assert rj.label_for("2026-08-05", rj.timeline(rows)) == "bear_channel_lower_highs"


def test_split_and_order_current_first_unknown_last():
    parts = rj.split(["2026-02-01", "2026-04-01", "2026-09-01"], lambda d: d, TIMELINE)
    assert set(parts) == {rj.UNKNOWN, "bull_run", "bear_channel_lower_highs"}
    order = rj.ordered_regimes(parts, "bear_channel_lower_highs", TIMELINE)
    assert order == ["bear_channel_lower_highs", "bull_run", rj.UNKNOWN]
    # The current regime is listed even with no rows in it.
    assert rj.ordered_regimes({"bull_run"}, "range", TIMELINE)[0] == "range"


def test_the_reader_is_read_only_and_reads_the_journal_table(tmp_path):
    from journal_store import JournalStore

    db = tmp_path / "journal.sqlite3"
    store = JournalStore(db)
    store.append_structural_regime(start_date="2026-08-01", regime="bear_channel_lower_highs")
    before = db.stat().st_mtime_ns
    segments = rj.read_segments(db)
    assert [s["regime"] for s in segments] == ["bear_channel_lower_highs"]
    assert db.stat().st_mtime_ns == before
    assert rj.read_segments(tmp_path / "missing.sqlite3") == []


# ---------------------------------------------------------------- swing, per regime


def _pick(day, r, side="LONG", symbol="AAA", family="avwap_breakout"):
    return {"session": day, "r": r, "side": side, "bucket": "near_favorite_zone",
            "family": family, "symbol": symbol, "status": "closed"}


def _bull_and_bear_picks():
    bull = [_pick(f"2026-04-{d:02d}", 1.0, symbol=f"B{i}") for i in range(3) for d in range(1, 21)]
    bear = [_pick(f"2026-08-{d:02d}", -1.0, symbol=f"R{i}") for i in range(3) for d in range(3, 23)]
    return bull, bear


def test_per_regime_cells_are_never_pooled():
    bull, bear = _bull_and_bear_picks()
    cells = rg.swing_cells_by_regime(bull + bear, {}, {}, rj.Joiner(TIMELINE))
    key = sg.swing_key("LONG", "near_favorite_zone", "avwap_breakout")
    by_regime = cells[key]
    assert by_regime["bull_run"]["n"] == 60 and by_regime["bull_run"]["win_rate"] == 1.0
    assert by_regime["bear_channel_lower_highs"]["n"] == 60
    assert by_regime["bear_channel_lower_highs"]["win_rate"] == 0.0
    assert by_regime["bull_run"]["grade"] != by_regime["bear_channel_lower_highs"]["grade"]
    assert all(cell.get("n") != 120 for name, cell in by_regime.items() if name != "_meta")


def test_longs_are_judged_raw_and_shorts_vs_spy():
    # 40 shorts that made money (raw win) but lost to SPY falling harder.
    shorts = [_pick(f"2026-08-{d:02d}", 0.5, side="SHORT", symbol=f"S{i}") for i in range(2) for d in range(3, 23)]
    index = {
        (p["symbol"], "SHORT", p["session"]): {
            "measured": "true", "maturity": "mature", "side_return_pct": "1.0",
            "scan_date": p["session"], "target_session": "2026-08-28",
        }
        for p in shorts
    }
    spy = {p["session"]: 100.0 for p in shorts}
    spy["2026-08-28"] = 95.0  # SPY -5%: the short side of SPY made +5%
    cells = rg.swing_cells_by_regime(shorts, index, spy, rj.Joiner(TIMELINE), as_of="2026-09-01")
    cell = cells[sg.swing_key("SHORT", "near_favorite_zone", "avwap_breakout")]["bear_channel_lower_highs"]
    assert cell["win_rate"] == 1.0  # raw
    assert cell["grade_basis"] == "tape" and cell["tape_win_rate"] == 0.0
    assert cell["grade"] == sg.D
    longs = [_pick(p["session"], 0.5, symbol=p["symbol"]) for p in shorts]
    long_cell = rg.swing_cells_by_regime(longs, {}, spy, rj.Joiner(TIMELINE))[
        sg.swing_key("LONG", "near_favorite_zone", "avwap_breakout")]["bear_channel_lower_highs"]
    assert "grade_basis" not in long_cell and long_cell["basis"] == "raw"


# ---------------------------------------------------------------- untested


def test_untested_in_this_regime_rather_than_a_guess():
    bull, _bear = _bull_and_bear_picks()
    payload = rg.build_payload(segments=TIMELINE, today="2026-09-26", swing_picks=bull)
    assert payload["current"]["regime"] == "bear_channel_lower_highs"
    assert payload["regimes"][0] == "bear_channel_lower_highs"
    entry = next(iter(payload["swing"].values()))
    assert rg.this_regime_text(entry["by_regime"], payload["current"]) == rg.UNTESTED
    row = rg.table_rows(payload)[0]
    assert row["this_regime"] == "untested in this regime"
    assert row["other_regimes"].startswith("bull run: ")
    assert row["all_regimes"] == "none"


def test_no_regime_typed_says_so():
    payload = rg.build_payload(segments=[], today="2026-09-26", swing_picks=[_pick("2026-08-03", 1.0)])
    assert payload["current"] is None
    assert payload["regimes"] == [rj.UNKNOWN]
    assert "No regime typed yet" in rg.status_sentence(payload)
    assert rg.table_rows(payload)[0]["this_regime"] == "no regime typed yet"


def test_the_pooled_grade_rides_beside_labelled_all_regimes():
    bull, bear = _bull_and_bear_picks()
    grades = {"swing": [{"key": sg.swing_key("LONG", "near_favorite_zone", "avwap_breakout"),
                         "grade": sg.C, "n": 120}]}
    payload = rg.build_payload(segments=TIMELINE, today="2026-09-26", swing_picks=bull + bear, grades=grades)
    entry = next(iter(payload["swing"].values()))
    text = rg.by_regime_text(entry["by_regime"], payload, pooled=entry["all"])
    assert text.startswith("bear channel, lower highs (now): D")
    assert text.endswith("all regimes: C n 120")
    assert rg.table_rows(payload)[0]["all_regimes"] == "C n 120"


# ---------------------------------------------------------------- day trade, journal, cohorts


def _bracket(day, result, bounce="vwap", side="LONG"):
    return {"event_id": f"{day}-{bounce}", "trade_date": day, "side": side, "bounce_type": bounce,
            "result": result, "result_2r": result, "eod_r": None, "reached_2r": None}


def test_daytrade_cells_split_by_regime():
    results = [_bracket("2026-07-10", sg.WIN)] * 3 + [_bracket("2026-08-10", sg.LOSS)] * 2
    cells = rg.daytrade_cells_by_regime(results, rj.Joiner(TIMELINE))
    by_regime = cells[sg.daytrade_key("vwap", "LONG")]
    assert by_regime["weekly_hh_then_compression"]["wins"] == 3
    assert by_regime["bear_channel_lower_highs"]["wins"] == 0
    assert by_regime["bear_channel_lower_highs"]["n"] == 2


def test_journal_trades_and_cohorts_join_on_their_entry_date():
    trades = [
        {"status": "CLOSED", "direction": "LONG", "opened_at": "2026-07-31T15:00:00-04:00", "net_pnl": 100},
        {"status": "CLOSED", "direction": "LONG", "opened_at": "2026-08-01T09:31:00-04:00", "net_pnl": -40},
        {"status": "OPEN", "direction": "LONG", "opened_at": "2026-08-02T09:31:00-04:00", "net_pnl": 0},
    ]
    journal = rg.journal_by_regime(trades, rj.Joiner(TIMELINE))
    assert journal["LONG"]["weekly_hh_then_compression"]["wins"] == 1
    assert journal["LONG"]["bear_channel_lower_highs"] == {
        "n": 1, "wins": 0, "sessions": 1, "win_rate": 0.0,
        "low_bound": sg.wilson_lower_bound(0, 1), "pnl": -40.0,
    }
    outcomes = [
        {"trade_date": "2026-08-04", "side": "LONG", "source": "like_h1", "h5_return": "0.02"},
        {"trade_date": "2026-05-04", "side": "LONG", "source": "veto_bad_chart", "h5_return": "-0.01"},
        {"trade_date": "2026-08-04", "side": "SHORT", "source": "focus_swing", "h5_return": "0.01",
         "entry_date": "2026-08-04", "h5_date": "2026-08-11"},
        {"trade_date": "2026-08-05", "side": "LONG", "source": "focus__swing_dislike", "h5_return": ""},
    ]
    spy = {"2026-08-04": 100.0, "2026-08-11": 98.0}
    cohorts = rg.cohorts_by_regime(outcomes, rj.Joiner(TIMELINE), spy)
    assert cohorts["human_focus_like|LONG"]["bear_channel_lower_highs"]["wins"] == 1
    assert cohorts["human_focus_veto|LONG"]["bull_run"]["wins"] == 0
    short = cohorts["human_focus_swing|SHORT"]["bear_channel_lower_highs"]
    assert short["win_rate"] == 1.0 and short["tape_win_rate"] == 0.0  # +1% raw, SPY short +2%
    assert "human_focus_rejection|LONG" not in cohorts  # an unmatured return is not a row


# ---------------------------------------------------------------- the service (worker side)


def test_the_service_writes_the_regime_file_and_leaves_the_grades_file_alone(tmp_path, monkeypatch):
    """Golden: the pooled grades (badges, sorting, Show) are byte-identical with regimes typed."""
    from journal_store import JournalStore
    from tests.test_setup_grades import _family_row, _out
    from ui.services import working_lately_service as svc

    monkeypatch.setattr(svc, "read_recent_rows", lambda: [_family_row()])
    monkeypatch.setattr(svc, "read_favorable_read", lambda: None)
    monkeypatch.setattr(svc, "read_held_run_summaries", lambda: None)
    monkeypatch.setattr(
        svc, "_outcome_rows", lambda: [_out("AAPL_long_20260921_09_45_00_vwap", 2, True, False)]
    )
    monkeypatch.setattr(
        svc, "read_swing_tape",
        lambda *_a, **_k: {"SHORT|near_favorite_zone|avwape_to_1stdev": {"cum_r_lately": 38.0}},
    )
    monkeypatch.setattr(svc, "_prior_m5", lambda _w: {"results": [], "grades": [], "held": [], "bracket": []})
    monkeypatch.setattr(svc, "_swing", lambda _ref: {"picks": [], "grades": [], "trade_r": [], "windows": {}})
    db = tmp_path / "journal.sqlite3"
    monkeypatch.setattr(svc, "_journal_db_path", lambda: db)
    monkeypatch.setattr(svc, "_cohort_paths", lambda: [])
    svc._LOOKING_BACK_CACHE.clear()

    first = svc.WorkingLatelyService(store_dir=tmp_path / "a").build_payload()
    JournalStore(db).append_structural_regime(start_date="2026-08-01", regime="bear_channel_lower_highs")
    svc._LOOKING_BACK_CACHE.clear()
    second = svc.WorkingLatelyService(store_dir=tmp_path / "b").build_payload()

    grades_a = (tmp_path / "a" / svc.GRADES_FILE_NAME).read_bytes()
    grades_b = (tmp_path / "b" / svc.GRADES_FILE_NAME).read_bytes()
    assert grades_a == grades_b, "live badges and sorting read an unchanged pooled grade"
    assert "by_regime" not in json.loads(grades_b)
    assert first["setup_grades"] == second["setup_grades"]
    by_regime = second["setup_grades_by_regime"]
    assert by_regime["schema"] == rg.SCHEMA
    assert by_regime["current"]["regime"] == "bear_channel_lower_highs"
    cell = by_regime["daytrade"][sg.daytrade_key("vwap", "LONG")]
    assert cell["all"]["grade"] == second["setup_grades"]["daytrade"][0]["grade"]
    assert set(cell["by_regime"]) == {"bear_channel_lower_highs"}
    on_disk = json.loads((tmp_path / "b" / svc.REGIME_GRADES_FILE_NAME).read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(by_regime, default=str))
    assert first["setup_grades_by_regime"]["current"] is None


# ---------------------------------------------------------------- the two trackers


def _regime_payload():
    results = [_bracket("2026-07-10", sg.WIN)] * 3 + [_bracket("2026-07-11", sg.LOSS, bounce="ema_15")]
    grades = {"daytrade": [{"key": sg.daytrade_key("vwap", "LONG"), "grade": sg.C, "n": 90}]}
    bull, _bear = _bull_and_bear_picks()
    return rg.build_payload(segments=TIMELINE, today="2026-09-26", swing_picks=bull,
                            bracket=results, grades=grades, m5_window=("2026-07-01", "2026-09-25"))


def test_the_daytrade_tracker_shows_this_regime_and_every_regime():
    from ui.panels.daytrade_tracker_panel import PERFORMANCE_COLUMNS, apply_regime_grades

    keys = [key for key, _label in PERFORMANCE_COLUMNS]
    assert keys.index("regime_now") == keys.index("grade_n") + 1
    assert dict(PERFORMANCE_COLUMNS)["regime_now"] == "This regime"
    rows = apply_regime_grades(
        [{"dimension": "bounce_type", "direction": "long", "segment": "vwap"},
         {"dimension": "time_bucket", "direction": "long", "segment": "vwap"}],
        _regime_payload(),
    )
    assert rows[0]["regime_now"] == rg.UNTESTED  # bear channel now; the alerts were in June-July
    assert rows[0]["regime_all"] == (
        "bear channel, lower highs (now): untested in this regime · "
        "weekly higher highs then compression: NEW · win 100% · n 3 · all regimes: C n 90"
    )
    assert rows[1]["regime_now"] == rows[1]["regime_all"] == ""
    # No published file: blanks, never a guess.
    assert apply_regime_grades(rows[:1], {})[0]["regime_now"] == ""


def test_the_setup_tracker_worker_builds_the_by_regime_rows(monkeypatch):
    from ui.panels import setup_tracker_panel as module
    from ui.services import working_lately_service

    payload = _regime_payload()
    monkeypatch.setattr(working_lately_service, "read_persisted_regime_grades", lambda *_a: payload)
    data = module._read_tracker_exports(1)
    rows = data["ranked"]["regime_grades"]
    assert rows == rg.table_rows(payload)
    assert [row["kind"] for row in rows] == ["swing", "day trade", "day trade"]
    assert rows[0]["this_regime"] == rg.UNTESTED
    assert data["regime_sentence"].startswith("Regime now: bear channel, lower highs since 2026-08-01 (day 57).")
    plan = {name: memo for name, _m, _rows, memo in module._table_render_plan(
        data["ranked"], data["signatures"], 1, "")}
    assert data["signatures"]["regime_grades"] in plan["regime_table"]


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.mark.qt
def test_the_setup_tracker_has_a_by_regime_tab_that_only_formats(qapp):
    from ui.panels import setup_tracker_panel as module

    panel = module.SetupTrackerPanel()
    try:
        titles = [panel.tabs.tabText(index) for index in range(panel.tabs.count())]
        assert titles[:2] == ["Current Picks", "By regime"]
        payload = _regime_payload()
        panel._on_exports_loaded({
            "signatures": {"regime_grades": "x"}, "raw": {}, "min_closed": 1,
            "ranked": {"regime_grades": rg.table_rows(payload)},
            "regime_sentence": rg.status_sentence(payload),
        })
        assert panel.regime_model.rows()[0]["this_regime"] == rg.UNTESTED
        assert panel.regime_status_label.text() == rg.status_sentence(payload)
    finally:
        panel.shutdown()
        panel.deleteLater()


# ---------------------------------------------------------------- SP4 (shadow) per regime

_POOLED = {"n": 200, "sessions": 30, "beat_low_h5": 0.60, "mean_move_atr_h10": 0.5}  # adjust +16
_BEAR_OK = {"n": 90, "sessions": 16, "beat_low_h5": 0.40, "mean_move_atr_h10": -0.5}  # adjust -16
_BEAR_THIN = {"n": 40, "sessions": 8, "beat_low_h5": 0.10, "mean_move_atr_h10": -2.0}


def _evidence(bear_cell, regime="bear_channel_lower_highs"):
    return {
        "families": {"LONG|alpha": _POOLED},
        "current_regime": regime,
        "current_regime_label": "bear channel, lower highs",
        "families_by_regime": {"bear_channel_lower_highs": {"LONG|alpha": bear_cell}},
    }


def test_sp4_reads_the_current_regimes_cell_when_its_gates_are_met():
    import points_challenger as pc

    cell, basis = pc.effective_cell("LONG", "alpha", _evidence(_BEAR_OK))
    assert basis == "bear_channel_lower_highs" and cell is _BEAR_OK
    assert pc.adjust_for("LONG", "alpha", _evidence(_BEAR_OK)) == -16.0
    assert pc.sp4_points({"priority_score": 50, "side": "LONG", "setup_family": "alpha"},
                         _evidence(_BEAR_OK)) == 34.0


def test_sp4_falls_back_to_all_regimes_and_says_so():
    import points_challenger as pc

    evidence = _evidence(_BEAR_THIN)
    cell, basis = pc.effective_cell("LONG", "alpha", evidence)
    assert basis == pc.ALL_REGIMES and cell is _POOLED
    assert pc.adjust_for("LONG", "alpha", evidence) == 16.0
    chip = pc.chip_text(evidence)
    assert "Regime bear channel, lower highs: 0 families read this regime's cells; " \
           "1 fell back to all regimes (under n 80 / 15 sessions in this regime)." in chip
    # A family with no cell in this regime at all falls back too.
    assert pc.effective_cell("LONG", "alpha", _evidence(_BEAR_OK, regime="range"))[1] == pc.ALL_REGIMES
    # Evidence written before S16 reads the pooled cell exactly as before.
    assert pc.adjust_for("LONG", "alpha", {"families": {"LONG|alpha": _POOLED}}) == 16.0


def test_the_night_writes_per_regime_cells_and_records_what_sp4_read():
    from ai_jobs import family_side_evidence as fse
    from tests.test_s12_points_challenger import AS_OF, ATR, SPY, fixture_rows

    segments = rj.timeline([
        {"segment_id": 1, "start_date": "2026-07-01", "regime": "range"},
        {"segment_id": 2, "start_date": "2026-08-02", "regime": "bear_channel_lower_highs"},
    ])
    payload = fse.build_payload(fixture_rows(16), SPY, ATR, {}, [], {}, None, as_of=AS_OF, segments=segments)
    assert payload["current_regime"] == "bear_channel_lower_highs"
    bear = payload["families_by_regime"]["bear_channel_lower_highs"]["SHORT|alpha"]
    pooled = payload["families"]["SHORT|alpha"]
    assert (bear["n"], bear["sessions"]) == (90, 15)  # 08-02..08-16 only, never pooled
    assert (pooled["n"], pooled["sessions"]) == (96, 16)
    assert payload["families_by_regime"]["range"]["SHORT|alpha"]["n"] == 6
    assert payload["regime_basis"]["SHORT|alpha"] == "bear_channel_lower_highs"
    assert payload["adjust_history"][AS_OF]["SHORT|alpha"] == bear["adjust"]
    # Without a typed regime the file is what S12 wrote.
    plain = fse.build_payload(fixture_rows(16), SPY, ATR, {}, [], {}, None, as_of=AS_OF)
    assert plain["current_regime"] == "" and plain["families_by_regime"] == {}
    assert plain["adjust_history"][AS_OF] == {k: c["adjust"] for k, c in plain["families"].items()}


# ---------------------------------------------------------------- the Saturday search


SEARCH_SEGMENTS = rj.timeline([
    {"segment_id": 1, "start_date": "2026-01-01", "regime": "bull_run"},
    {"segment_id": 2, "start_date": "2026-02-10", "regime": "bear_channel_lower_highs"},
])


def test_the_saturday_search_gets_a_regime_facet(tmp_path):
    import setup_permutation_search as search
    from research_warehouse import trial_ledger
    from tests.test_setup_permutation_search import _population

    search.build_report(_population(), ledger_root=tmp_path, segments=SEARCH_SEGMENTS)
    declared = [cell for trial in trial_ledger.load(tmp_path) for cell in trial.get("declared_cells") or []]
    assert "regime=bull_run" in declared
    # Before 02-10 the bear channel has 3 selection sessions: under the floor, never searched.
    assert "regime=bear_channel_lower_highs" not in declared


def test_the_saturday_search_reports_per_regime_never_pooled(tmp_path):
    import setup_permutation_search as search
    from tests.test_setup_permutation_search import _population

    rows = _population()
    report = search.build_report(rows, ledger_root=tmp_path, segments=SEARCH_SEGMENTS)
    block = report["by_regime"]
    assert block["current"] == "bear_channel_lower_highs"
    assert block["order"] == ["bear_channel_lower_highs", "bull_run"]
    cells = block["populations"]["swing"]["1"]["avwap_band_bounce LONG"]
    assert cells["bull_run"]["n"] == 36 * 20 and cells["bull_run"]["sessions"] == 36
    assert cells["bear_channel_lower_highs"]["n"] == 24 * 20
    bull_wins = sum(1 for row in rows if row["session"] < "2026-02-10" and row["win"])
    assert cells["bull_run"]["wins"] == bull_wins
    # No timeline given: the report is what it was (no facet, no block).
    plain = search.build_report(rows, ledger_root=tmp_path / "plain")
    assert "by_regime" not in plain
    assert not any("regime" in str(key.get("facets")) for fam in plain["populations"]["swing"]["horizons"]["1"][
        "families"].values() for key in fam["keys"])


def test_the_research_pack_reads_the_file_the_build_writes():
    import project_paths
    from ui.services import working_lately_service as svc

    assert Path(project_paths.SETUP_GRADES_BY_REGIME_FILE).name == svc.REGIME_GRADES_FILE_NAME
    assert Path(project_paths.SETUP_GRADES_BY_REGIME_FILE).parent == svc.default_store_dir()
