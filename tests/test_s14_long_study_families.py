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
