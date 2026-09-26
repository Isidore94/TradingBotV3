"""S14: the long study families - one definition, tagged in the sidecar and the horizons."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import long_study_families as lsf  # noqa: E402
import swing_path_facts as spf  # noqa: E402
from master_avwap_lib import session_horizon_outcomes as sho  # noqa: E402

LPL, BBL = lsf.LEADER_PULLBACK_LONG, lsf.BAND_BOUNCE_LEADER_LONG


@pytest.mark.parametrize(
    ("side", "family", "pct", "sector", "expected"),
    [
        ("LONG", "top_pattern_tracking", -5.0, "", True),
        ("LONG", "avwap_breakout", -10.0, "Technology", True),
        ("LONG", "avwap_breakout", -3.0, "Technology", True),
        ("LONG", "avwap_breakout", -2.9, "Technology", False),
        ("LONG", "avwap_breakout", -5.0, "Energy", False),
        ("SHORT", "top_pattern_tracking", -5.0, "Technology", False),
        ("LONG", "avwap_breakout", None, "Technology", None),
        ("LONG", "avwap_breakout", -5.0, "", None),
    ],
)
def test_leader_pullback_long(side, family, pct, sector, expected):
    assert lsf.leader_pullback_long(side, family, pct, sector) is expected


@pytest.mark.parametrize(
    ("side", "family", "sector", "rs_top", "spy", "expected"),
    [
        ("LONG", "avwap_band_bounce", "Technology", True, "True", True),
        ("LONG", "avwap_band_bounce", "Healthcare", True, True, True),
        ("LONG", "avwap_band_bounce", "Energy", True, "True", False),
        ("LONG", "avwap_band_bounce", "Technology", False, "True", False),
        ("LONG", "avwap_band_bounce", "Technology", True, "False", False),
        ("LONG", "top_pattern_tracking", "Technology", True, "True", False),
        ("SHORT", "avwap_band_bounce", "Technology", True, "True", False),
        ("LONG", "avwap_band_bounce", "", True, "True", None),
        ("LONG", "avwap_band_bounce", "Technology", None, "True", None),
        ("LONG", "avwap_band_bounce", "Technology", True, "", None),
    ],
)
def test_band_bounce_leader_long(side, family, sector, rs_top, spy, expected):
    assert lsf.band_bounce_leader_long(side, family, sector, rs_top, spy) is expected


def test_rs_top_tercile_is_the_sessions_own_long_cross_section():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert lsf.rs_top_tercile(6.0, values) is True
    assert lsf.rs_top_tercile(5.0, values) is True
    assert lsf.rs_top_tercile(4.0, values) is False
    assert lsf.rs_top_tercile(None, values) is None
    assert lsf.rs_top_tercile(9.0, [1.0, 2.0]) is None  # too few to cut a tercile
    rows = [
        {"side": "LONG", "scan_date": "2026-09-01", "rs_vs_industry": "1.5"},
        {"side": "SHORT", "scan_date": "2026-09-01", "rs_vs_industry": "9"},
        {"side": "LONG", "scan_date": "2026-09-02", "rs_vs_industry": ""},
    ]
    assert lsf.session_rs_values(rows) == {"2026-09-01": [1.5]}


def test_the_tag_round_trips_and_drops_unknown_names():
    assert lsf.parse_tag(lsf.tag_text([LPL, BBL])) == [LPL, BBL]
    assert lsf.parse_tag("nope;" + LPL) == [LPL]
    assert lsf.parse_tag(None) == []


def test_swing_path_facts_has_no_definition_of_its_own():
    source = (SCRIPTS_DIR / "swing_path_facts.py").read_text(encoding="utf-8")
    assert "LEADER_PULLBACK_VWAP_RANGE" not in source
    assert '"top_pattern_tracking"' not in source
    assert "-10.0" not in source


def test_the_promotion_to_a_scored_family_is_ask_first_in_the_code():
    assert "ASK-FIRST" in lsf.__doc__


# --- the session-horizon export


def _history_rows():
    import pandas as pd

    base = {"run_id": "r1", "run_timestamp": "2026-09-01T13:00:00", "run_date": "2026-09-01",
            "last_trade_date": "2026-09-01", "last_close": 50.0, "spy_above_sma20": True}
    rows = [
        # A Technology pullback (leader_pullback_long) and a top-RS Tech band bounce.
        {**base, "symbol": "TECH", "side": "LONG", "setup_family": "avwap_breakout", "sector": "Technology",
         "pct_from_current_vwap": -6.0, "rs_vs_industry": 0.1},
        {**base, "symbol": "BNCE", "side": "LONG", "setup_family": "avwap_band_bounce", "sector": "Healthcare",
         "pct_from_current_vwap": 2.0, "rs_vs_industry": 5.0},
        {**base, "symbol": "LOW1", "side": "LONG", "setup_family": "avwap_band_bounce", "sector": "Technology",
         "pct_from_current_vwap": -6.0, "rs_vs_industry": -2.0},
        {**base, "symbol": "SHRT", "side": "SHORT", "setup_family": "avwap_band_bounce", "sector": "Technology",
         "pct_from_current_vwap": -6.0, "rs_vs_industry": 9.0},
        {**base, "symbol": "NORS", "side": "LONG", "setup_family": "avwap_band_bounce", "sector": "Technology",
         "pct_from_current_vwap": 1.0, "rs_vs_industry": None},
    ]
    return pd.DataFrame(rows)


def test_session_horizon_rows_carry_the_study_families():
    built = sho.build_session_horizon_observation_rows(
        _history_rows(), lambda symbol: None, horizons=(5,), last_completed_session=date(2026, 9, 30),
        window_sessions=None,
    )
    tags = {row["symbol"]: row["study_families"] for row in built.rows}
    assert tags == {"TECH": LPL, "BNCE": BBL, "LOW1": LPL, "SHRT": "", "NORS": ""}
    assert sho.SESSION_HORIZON_OUTCOME_COLUMNS[-1] == "study_families"


def test_a_history_without_the_inputs_tags_nothing():
    frame = _history_rows().drop(columns=["sector", "rs_vs_industry", "spy_above_sma20", "pct_from_current_vwap"])
    built = sho.build_session_horizon_observation_rows(
        frame, lambda symbol: None, horizons=(5,), last_completed_session=date(2026, 9, 30), window_sessions=None,
    )
    assert {row["study_families"] for row in built.rows} == {""}


def test_the_sidecar_tags_the_same_families():
    rows = [{"outcome_kind": spf.SOURCE_OUTCOME_KIND, "scan_row_id": f"{s}:2026-09-01:r1", "symbol": s,
             "side": side, "scan_date": "2026-09-01", "entry_close": "50", "setup_family": fam,
             "horizon_sessions": "5"}
            for s, side, fam in (("TECH", "LONG", "avwap_breakout"), ("BNCE", "LONG", "avwap_band_bounce"),
                                 ("LOW1", "LONG", "avwap_band_bounce"))]
    facts = {
        "TECH:2026-09-01:r1": spf.ScanFacts(2.0, -6.0, "Technology", 0.1, "True"),
        "BNCE:2026-09-01:r1": spf.ScanFacts(2.0, 2.0, "Healthcare", 5.0, "True"),
        "LOW1:2026-09-01:r1": spf.ScanFacts(2.0, -6.0, "Technology", -2.0, "True"),
    }
    build = spf.build_path_fact_rows(rows, lambda s: None, facts, last_completed_session=date(2026, 9, 30),
                                     horizons=(5,))
    assert {row["symbol"]: row["study_families"] for row in build.rows} == {"TECH": LPL, "BNCE": BBL, "LOW1": LPL}
    assert spf.COLUMNS[-1] == "study_families"


# --- the permutation facets, the backfill copies and the search's study block


def test_trend20_and_htf_trend_4h_facets():
    import setup_permutations as sp

    key = sp.facets_for_row({"side": "LONG", "setup_family": "x", "trend_20d": "DOWN", "htf_trend_4h": "NEUTRAL"})
    assert key.get("trend20") == "trend20_down"
    assert key.get("htf_trend_4h") == "h4_neutral"
    blank = sp.facets_for_row({"side": "LONG", "setup_family": "x", "trend_20d": "", "htf_trend_4h": None})
    assert blank.get("trend20") == sp.UNKNOWN and blank.get("htf_trend_4h") == sp.UNKNOWN
    assert set(lsf.STUDY_SEARCH_FACETS) <= set(sp.FACETS)


def test_the_backfill_copies_a_tagged_row_under_each_study_family(tmp_path):
    import csv

    import setup_permutation_backfill as bf

    path = tmp_path / "features.csv"
    rows = [
        {"symbol": "TECH", "side": "LONG", "setup_family": "avwap_breakout", "sector": "Technology",
         "pct_from_current_vwap": "-6", "rs_vs_industry": "0.1"},
        {"symbol": "BNCE", "side": "LONG", "setup_family": "avwap_band_bounce", "sector": "Healthcare",
         "pct_from_current_vwap": "2", "rs_vs_industry": "5"},
        {"symbol": "LOW1", "side": "LONG", "setup_family": "avwap_band_bounce", "sector": "Energy",
         "pct_from_current_vwap": "2", "rs_vs_industry": "-2"},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*rows[0], "last_trade_date", "run_id", "last_close",
                                                    "spy_above_sma20", "atr20"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "last_trade_date": "2026-09-01", "run_id": "r1", "last_close": "50",
                             "spy_above_sma20": "True", "atr20": "2"})
    keyed = bf.key_representatives(path, bf.session_representatives(path), bf.ContextStores())
    study = {identity[0]: row.study for identity, row in keyed.items()}
    assert study == {"TECH": (LPL,), "BNCE": (BBL,), "LOW1": ()}
    out = bf._swing_rows(keyed[("TECH", "LONG", "2026-09-01")], 5, True, 3.0, 50.0)
    assert [row["family"] for row in out] == ["avwap_breakout", LPL]
    assert out[1]["f_trend20"] == out[0]["f_trend20"]


def _study_population():
    import random
    from datetime import timedelta

    rng = random.Random(3)
    sessions = [(date(2026, 1, 5) + timedelta(days=i)).isoformat() for i in range(60)]
    rows = []
    for day in sessions:
        for n in range(20):
            spy_up = rng.random() < 0.5
            win = rng.random() < (0.8 if spy_up else 0.35)
            base = {"population": "swing", "side": "LONG", "horizon": 5, "session": day,
                    "episode_id": f"{day}:{n}", "win": win, "r": 1.0 if win else -1.0,
                    "f_spy_trend": "spy_above_sma20_above_sma50" if spy_up else "spy_below_sma20_below_sma50",
                    "f_trend20": rng.choice(["trend20_up", "trend20_down"]),
                    "f_htf_trend_4h": rng.choice(["h4_up", "h4_neutral"]),
                    "f_noise": rng.choice(["a", "b"])}
            rows.append({**base, "family": "avwap_breakout"})
            rows.append({**base, "family": LPL})
    return rows


def test_the_search_runs_study_families_first_on_their_own_three_facets(tmp_path):
    import setup_permutation_search as search
    from research_warehouse import trial_ledger

    report = search.build_report(_study_population(), ledger_root=tmp_path)
    block = report["populations"]["swing"]["horizons"]["5"]
    assert f"{LPL} LONG" not in block["families"]
    study = block["study_families"][f"{LPL} LONG"]
    assert study["facets_searched"] == list(lsf.STUDY_SEARCH_FACETS)
    assert study["verdict"] == search.VERDICT_KEY
    assert {"spy_trend": "spy_above_sma20_above_sma50"} in [key["facets"] for key in study["keys"]]
    for key in study["keys"] + study["top_rejected"]:
        assert set(key["facets"]) <= set(lsf.STUDY_SEARCH_FACETS)
    # The ordinary family still searches every facet, noise included.
    assert "avwap_breakout LONG" in block["families"]
    trials = [row["trial_id"] for row in trial_ledger.load(tmp_path)]
    assert trials[0].split(":")[2] == LPL  # registered before any other family


# --- the Setup Tracker: graded like any family, raw first for longs


def _hrow(symbol, scan, target, ret, tag, side="LONG"):
    return {"symbol": symbol, "side": side, "scan_date": scan, "target_session": target,
            "horizon_sessions": "5", "side_return_pct": str(ret), "measured": "True", "maturity": "mature",
            "outcome_kind": "favorable_direction_session_v2", "study_families": tag}


SPY = {"2026-09-01": 100.0, "2026-09-08": 102.0,   # SPY-up window (+2%)
       "2026-09-02": 100.0, "2026-09-09": 100.5}   # flat window (+0.5%)


def test_study_cells_grade_raw_in_spy_up_windows_and_tape_on_every_row():
    import setup_grades

    rows = [
        _hrow("A1", "2026-09-01", "2026-09-08", 3.0, LPL),   # raw win, beats SPY
        _hrow("A2", "2026-09-01", "2026-09-08", 1.0, LPL),   # raw win, lags SPY
        _hrow("A3", "2026-09-01", "2026-09-08", -1.0, f"{LPL};{BBL}"),
        _hrow("B1", "2026-09-02", "2026-09-09", 4.0, LPL),   # flat window: tape only
        _hrow("C1", "2026-09-01", "2026-09-08", 9.0, ""),    # untagged: never counted
        _hrow("D1", "2026-09-02", "2026-09-09", 1.0, LPL) | {"measured": "False"},
    ]
    cells = {cell["family"]: cell for cell in setup_grades.study_family_cells(rows, SPY, as_of="2026-09-25")}
    lpl = cells[LPL]
    assert lpl["headline"] == "raw"
    assert (lpl["raw"]["n"], lpl["raw"]["wins"]) == (3, 2)
    assert lpl["raw"]["mean_pct"] == pytest.approx(1.0)
    assert (lpl["tape"]["n"], lpl["tape"]["wins"]) == (4, 2)
    assert lpl["tape"]["mean_pct"] == pytest.approx(((3 - 2) + (1 - 2) + (-1 - 2) + (4 - 0.5)) / 4)
    assert lpl["raw"]["grade"] == setup_grades.NEW  # under 30: the same ladder as every family
    assert (cells[BBL]["raw"]["n"], cells[BBL]["tape"]["n"]) == (1, 1)
    line = setup_grades.study_family_line(lpl)
    assert line.index("raw in SPY-up") < line.index("vs SPY")
    assert "(study)" in line and "n 3" in line and "n 4" in line


def test_a_study_family_with_no_rows_says_so():
    import setup_grades

    cells = setup_grades.study_family_cells([], SPY)
    assert [cell["family"] for cell in cells] == list(lsf.STUDY_FAMILIES)
    assert setup_grades.STUDY_NO_DATA in setup_grades.study_family_line(cells[0])


def test_the_worker_reader_builds_the_lines_from_the_horizon_file(tmp_path, monkeypatch):
    import csv

    from ui.services import working_lately_service as service

    horizon = tmp_path / "horizon.csv"
    row = _hrow("A1", "2026-09-01", "2026-09-08", 3.0, LPL)
    with horizon.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    monkeypatch.setattr(service, "_horizon_outcomes_path", lambda: horizon)
    monkeypatch.setattr(service, "_spy_bars_path", lambda: tmp_path / "SPY.parquet")
    monkeypatch.setattr(service, "read_spy_closes", lambda: dict(SPY))
    service._LOOKING_BACK_CACHE.clear()
    try:
        lines = service.read_study_family_lines("2026-09-25")
        assert len(lines) == 2 and lines[0].startswith(f"{LPL} LONG (study): raw in SPY-up")
        assert "n 1" in lines[0]
    finally:
        service._LOOKING_BACK_CACHE.clear()


def test_the_tracker_worker_carries_the_lines_and_survives_a_failure(monkeypatch):
    from ui.panels import setup_tracker_panel as module
    from ui.services import working_lately_service

    monkeypatch.setattr(working_lately_service, "read_study_family_lines", lambda as_of="": ["a", "b"])
    assert module._read_tracker_exports(1)["study_family_lines"] == ["a", "b"]

    def _boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(working_lately_service, "read_study_family_lines", _boom)
    assert module._read_tracker_exports(1)["study_family_lines"] == []


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.mark.qt
def test_the_tracker_prints_the_study_lines_it_was_handed(qapp):
    from ui.panels import setup_tracker_panel as module

    panel = module.SetupTrackerPanel()
    try:
        panel._on_exports_loaded({"signatures": {}, "ranked": {}, "raw": {}, "study_family_lines": ["x", "y"]})
        assert panel.study_family_label.text() == "x\ny"
        panel._on_exports_loaded({"signatures": {}, "ranked": {}, "raw": {}, "min_closed": 1})
        assert panel.study_family_label.text() == ""
    finally:
        panel.shutdown()
        panel.deleteLater()
