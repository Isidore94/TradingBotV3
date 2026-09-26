"""Golden: bucket, points, tier, D1 alert and tracker record of every D1 row (favourite-zone long gut).

Frozen BEFORE the gut from a scratch copy of the 2026-09-25 after-close scan's
`master_avwap_ai_state.json` (see `build_favzone_long_gut_fixture.py`), plus a
grid of `build_priority_setup_summary` inputs for the points.
"""

from conftest import load_fixture_contract
from favzone_long_gut_replay import observe_points, observe_scan

GOLDEN = load_fixture_contract("favzone_long_gut_v1")
RAW_ROWS = GOLDEN["raw"]["rows"]
EXPECTED_SCAN = GOLDEN["expected"]["scan"]
EXPECTED_POINTS = GOLDEN["expected"]["points"]


def test_golden_covers_both_sides_and_every_bucket():
    buckets = {(rec["side"], rec["bucket"]) for rec in EXPECTED_SCAN.values()}
    assert {("LONG", "favorite_setup"), ("LONG", "near_favorite_zone"), ("LONG", ""),
            ("SHORT", "favorite_setup"), ("SHORT", "near_favorite_zone"), ("SHORT", "")} <= buckets


def test_scan_replay_matches_golden(tmp_path):
    actual = observe_scan(RAW_ROWS, tmp_path)
    assert set(actual) == set(EXPECTED_SCAN)
    for symbol, record in EXPECTED_SCAN.items():
        assert actual[symbol] == record, symbol


def test_points_match_golden():
    assert observe_points() == EXPECTED_POINTS
