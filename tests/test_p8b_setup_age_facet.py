"""Plan to 8 / P8b C4a - the `setup_age` D1 facet on the scan row.

Shadow only. The age is completed sessions since the setup tracker first saw this
(symbol, side, setup family); no first-seen record is unknown, never "new". The
scan golden proves every non-facet column and the detector/scoring output are
unchanged with the hook on.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import setup_permutation_backfill as bf  # noqa: E402
import setup_permutation_search as search  # noqa: E402
import setup_permutations as sp  # noqa: E402

AGE = "perm_setup_age_sessions"
SESSIONS = [(date(2026, 9, 1) + timedelta(days=i)).isoformat() for i in range(30)
            if (date(2026, 9, 1) + timedelta(days=i)).weekday() < 5]


def _setup(scan_date, symbol="AAA", side="LONG", family="general"):
    return {"symbol": symbol, "side": side, "scan_date": scan_date, "setup_family": family}


def _payload(*setups):
    return {"setups": {f"id{index}": setup for index, setup in enumerate(setups)}}


def _row(as_of, symbol="AAA", side="LONG", family="general"):
    return {"symbol": symbol, "side": side, "setup_family": family, "last_trade_date": as_of}


def _age(row, payload, sessions=SESSIONS):
    rows = [row]
    sp.setup_age_columns(rows, payload, sessions)
    return rows[0].get(AGE, "absent")


# ---------------------------------------------------------------------------
# the column: completed sessions since the tracker's first-seen record
# ---------------------------------------------------------------------------
def test_age_counts_completed_sessions_after_the_first_seen_scan_date():
    first, today = SESSIONS[2], SESSIONS[7]
    assert _age(_row(today), _payload(_setup(first), _setup(SESSIONS[5]))) == 5


def test_first_seen_on_the_scan_date_is_age_zero():
    assert _age(_row(SESSIONS[4]), _payload(_setup(SESSIONS[4]))) == 0


def test_a_weekend_is_not_a_session():
    # SESSIONS[3] is a Friday, SESSIONS[4] the next Monday.
    assert date.fromisoformat(SESSIONS[3]).weekday() == 4
    assert _age(_row(SESSIONS[4]), _payload(_setup(SESSIONS[3]))) == 1


@pytest.mark.parametrize("payload", [
    _payload(),
    _payload(_setup(SESSIONS[1], symbol="BBB")),
    _payload(_setup(SESSIONS[1], side="SHORT")),
    _payload(_setup(SESSIONS[1], family="avwap_band_bounce")),
    {},
    None,
    {"setups": "junk"},
])
def test_no_first_seen_record_for_the_key_is_unknown(payload):
    assert _age(_row(SESSIONS[6]), payload) is None


def test_a_record_after_the_scan_date_is_not_known_yet():
    assert _age(_row(SESSIONS[3]), _payload(_setup(SESSIONS[5]))) is None


def test_a_blank_family_has_no_setup_key():
    assert _age(_row(SESSIONS[6], family=""), _payload(_setup(SESSIONS[1], family=""))) is None


def test_a_first_seen_date_older_than_the_calendar_is_unknown():
    assert _age(_row(SESSIONS[6]), _payload(_setup("2026-08-03"))) is None


def test_no_calendar_is_unknown():
    assert _age(_row(SESSIONS[6]), _payload(_setup(SESSIONS[1])), sessions=()) is None


def test_the_key_normalises_symbol_side_and_family_case():
    payload = _payload(_setup(SESSIONS[1], symbol="aaa ", side="long", family=" General"))
    assert _age(_row(SESSIONS[3]), payload) == 2


def test_a_junk_record_is_skipped_not_fatal():
    payload = {"setups": {"a": "junk", "b": _setup("not-a-date"), "c": _setup(SESSIONS[0])}}
    assert _age(_row(SESSIONS[2]), payload) == 2


def test_the_column_is_appended_after_the_p11_columns():
    # S6 appends its trendline columns after the age; nothing else follows it.
    columns = sp.SCAN_ROW_COLUMNS[:-len(sp.TRENDLINE_COLUMNS)]
    assert sp.SCAN_ROW_COLUMNS[-len(sp.TRENDLINE_COLUMNS):] == sp.TRENDLINE_COLUMNS
    assert columns[-1] == AGE
    assert columns[-1 - len(sp.D1_HISTORY_COLUMNS):-1] == sp.D1_HISTORY_COLUMNS
    assert AGE.startswith("perm_")


# ---------------------------------------------------------------------------
# the facet: value and unknown
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("value", "expected"), [
    (0, "setup_age_0"), (1, "setup_age_1_2"), (2, "setup_age_1_2"), (3, "setup_age_3_5"), (5, "setup_age_3_5"),
    (6, "setup_age_6_10"), (10, "setup_age_6_10"), (11, "setup_age_11_plus"), ("40", "setup_age_11_plus"),
    ("", "unknown"), (None, "unknown"), (float("nan"), "unknown"), (-1, "unknown"), ("junk", "unknown"),
])
def test_setup_age_facet(value, expected):
    key = sp.facets_for_row({"side": "LONG", "setup_family": "general", AGE: value})
    assert key.get("setup_age") == expected


def test_a_row_without_the_column_keys_exactly_as_before():
    row = {"side": "LONG", "setup_family": "general", "atr20": 2.0}
    key = sp.facets_for_row(row)
    assert key.get("setup_age") == sp.UNKNOWN
    assert "setup_age" not in key.compact_key
    assert key.permutation_rule_version == "setup_permutations.v1"


# ---------------------------------------------------------------------------
# the Saturday search sees it, under the same floors as every other facet
# ---------------------------------------------------------------------------
def _search_rows(sessions, per_session):
    facet_columns = [column for column in bf.output_columns() if column.startswith("f_")]
    rng = random.Random(7)
    rows = []
    for day in sessions:
        for n in range(per_session):
            fresh = rng.random() < 0.3
            win = rng.random() < (0.85 if fresh else 0.4)
            row = {column: sp.UNKNOWN for column in facet_columns}
            row.update({"population": bf.POPULATION_SWING, "family": "general", "side": "LONG", "horizon": 5,
                        "session": day, "episode_id": f"{day}:{n}", "win": win, "r": 1.0 if win else -1.0,
                        "f_setup_age": "setup_age_0" if fresh else "setup_age_6_10"})
            rows.append(row)
    return rows


def test_the_backfill_writes_an_f_setup_age_column():
    assert "f_setup_age" in bf.output_columns()


def test_the_search_finds_a_setup_age_key(tmp_path):
    sessions = [(date(2026, 1, 5) + timedelta(days=i)).isoformat() for i in range(60)]
    report = search.build_report(_search_rows(sessions, 20), ledger_root=tmp_path)
    family = report["populations"]["swing"]["horizons"]["5"]["families"]["general LONG"]
    assert family["verdict"] == search.VERDICT_KEY
    assert family["keys"][0]["facets"] == {"setup_age": "setup_age_0"}


def test_the_search_floors_apply_to_setup_age(tmp_path):
    # Too few sessions for the family floor: no key, whatever the win rate.
    sessions = [(date(2026, 1, 5) + timedelta(days=i)).isoformat() for i in range(search.MIN_SESSIONS - 1)]
    report = search.build_report(_search_rows(sessions, 20), ledger_root=tmp_path)
    family = report["populations"]["swing"]["horizons"]["5"]["families"].get("general LONG") or {}
    assert family.get("verdict") != search.VERDICT_KEY
    assert not family.get("keys")


# ---------------------------------------------------------------------------
# the scan golden: output unchanged, the age column filled from the tracker
# ---------------------------------------------------------------------------
def _parity_module():
    path = Path(__file__).with_name("test_setup_permutation_scan_parity.py")
    spec = importlib.util.spec_from_file_location("_p8b_scan_parity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scan_runs(tmp_path_factory):
    parity = _parity_module()
    base = tmp_path_factory.mktemp("p8b-scan")
    # The synthetic PKEY LONG scores as "general"; the tracker first saw it 3 sessions back.
    seed = base / "tracker_seed.json"
    seed.write_text(json.dumps({"first_seen_offset": 3, "symbol": "PKEY", "side": "LONG",
                                "setup_family": "general"}), encoding="utf-8")
    return parity, parity._run(base, "on", 300, seed=seed), parity._run(base, "off", 300, seed=seed)


def test_the_scan_writes_the_setup_age(scan_runs):
    _parity, stamped, _plain = scan_runs
    row = stamped["history"][-1]
    assert list(row)[-1 - len(sp.TRENDLINE_COLUMNS)] == AGE
    assert row[AGE] == "3"
    key = dict(part.split("=", 1) for part in row["permutation_key"].split("|")[3].split(";"))
    assert key["setup_age"] == "setup_age_3_5"


def test_the_scan_output_is_identical_with_and_without_the_setup_age(scan_runs):
    parity, stamped, plain = scan_runs
    assert stamped["priority_row"] and plain["priority_row"]
    assert parity._strip(stamped["priority_row"]) == parity._strip(plain["priority_row"])
    assert parity._strip(stamped["ai_state_entry"]) == parity._strip(plain["ai_state_entry"])
    assert len(stamped["history"]) == len(plain["history"])
    for on_row, off_row in zip(stamped["history"], plain["history"], strict=True):
        assert parity._strip(on_row) == parity._strip(off_row)
    assert plain["history"][-1][AGE] == ""
