"""Golden: bucket, points, tier, D1 alert and tracker record of every D1 row (favourite-zone long gut).

Frozen BEFORE the gut from a scratch copy of the 2026-09-25 after-close scan's
`master_avwap_ai_state.json` (see `build_favzone_long_gut_fixture.py`), plus a
grid of `build_priority_setup_summary` inputs for the points.

The gut (trader 2026-09-26, "Gut it and replace it with what does work"): every
SHORT row and every non-favourite LONG row is byte-identical; a LONG row that had
`favorite_setup` / `near_favorite_zone` keeps no bucket, no high conviction, no
best-swing or tracker slot, no Focus favourite entry and no favourite D1 reason,
and is recorded as a tracker control `favzone_long_retired`. The LONG
favourite zone arms no D1 trigger and earns no points (18, +10 with a retest).
"""

import pytest
from conftest import load_fixture_contract
from favzone_long_gut_replay import observe_points, observe_scan

GOLDEN = load_fixture_contract("favzone_long_gut_v1")
RAW_ROWS = GOLDEN["raw"]["rows"]
EXPECTED_SCAN = GOLDEN["expected"]["scan"]
EXPECTED_POINTS = GOLDEN["expected"]["points"]


FAVOURITE_BUCKETS = {"favorite_setup", "near_favorite_zone"}
FAVOURITE_SECTIONS = {"high_conviction", "favorites", "near_favorite_zones"}


@pytest.fixture(scope="module")
def actual(tmp_path_factory):
    return observe_scan(RAW_ROWS, tmp_path_factory.mktemp("favzone"))


def test_golden_covers_both_sides_and_every_bucket():
    buckets = {(rec["side"], rec["bucket"]) for rec in EXPECTED_SCAN.values()}
    assert {("LONG", "favorite_setup"), ("LONG", "near_favorite_zone"), ("LONG", ""),
            ("SHORT", "favorite_setup"), ("SHORT", "near_favorite_zone"), ("SHORT", "")} <= buckets
    retired = [rec for rec in EXPECTED_SCAN.values() if rec["side"] == "LONG" and rec["bucket"] in FAVOURITE_BUCKETS]
    assert len(retired) == 112
    assert sum(1 for rec in retired if rec["high_conviction"]) == 34


def test_every_short_row_is_byte_identical(actual):
    assert set(actual) == set(EXPECTED_SCAN)
    for symbol, record in EXPECTED_SCAN.items():
        if record["side"] == "SHORT":
            assert actual[symbol] == record, symbol


def _without_favourite_zone_triggers(before, after):
    """The LONG favourite-zone triggers are gone; the rest keep their order. A level the
    generic A/S upgrade path also arms (same trigger id) may now come from that path."""
    zone_ids = {item["trigger_id"] for item in before if item["source"] == "favorite_zone"}
    kept = [item for item in before if item["source"] != "favorite_zone"]
    assert not [item for item in after if item["source"] == "favorite_zone"]
    assert [item for item in after if item in kept] == kept
    assert {item["trigger_id"] for item in after if item not in kept} <= zone_ids


def test_non_favourite_long_rows_keep_everything_but_the_zone_triggers(actual):
    for symbol, record in EXPECTED_SCAN.items():
        if record["side"] != "LONG" or record["bucket"] in FAVOURITE_BUCKETS:
            continue
        now = actual[symbol]
        assert {**now, "d1_triggers": None} == {**record, "d1_triggers": None}, symbol
        _without_favourite_zone_triggers(record["d1_triggers"], now["d1_triggers"])


def test_long_favourite_rows_are_retired_and_recorded(actual):
    for symbol, record in EXPECTED_SCAN.items():
        if record["side"] != "LONG" or record["bucket"] not in FAVOURITE_BUCKETS:
            continue
        now = actual[symbol]
        assert now["bucket"] == "", symbol
        assert now["control"] == "favzone_long_retired", symbol
        assert not now["tracked"] and not now["high_conviction"] and not now["best_swing"], symbol
        assert not set(now["focus_sections"]) & FAVOURITE_SECTIONS, symbol
        assert set(now["focus_sections"]) <= set(record["focus_sections"]), symbol
        assert now["tier"] in {"", "B"} or "post_earnings_plays" in now["focus_sections"], symbol
        assert not set(now["d1_reasons"]) & FAVOURITE_BUCKETS, symbol
        kept = [reason for reason in record["d1_reasons"] if reason not in FAVOURITE_BUCKETS]
        assert [reason for reason in now["d1_reasons"] if reason in kept] == kept, symbol
        _without_favourite_zone_triggers(record["d1_triggers"], now["d1_triggers"])
        assert (now["score"], now["setup_family"]) == (record["score"], record["setup_family"]), symbol


def test_no_long_row_is_left_in_a_favourite_bucket_or_list(actual):
    longs = [rec for rec in actual.values() if rec["side"] == "LONG"]
    assert not [rec for rec in longs if rec["bucket"] or rec["high_conviction"] or rec["best_swing"] or rec["tracked"]]
    assert not [rec for rec in longs if set(rec["focus_sections"]) & FAVOURITE_SECTIONS]


def test_points_drop_the_long_favourite_zone_bonus_only():
    for before, now in zip(EXPECTED_POINTS, observe_points(), strict=True):
        inputs = before["inputs"]
        assert now["inputs"] == inputs
        assert now["setup_family"] == before["setup_family"], inputs
        if inputs["side"] == "LONG" and inputs["zone"]:
            assert now["score"] == before["score"] - 18 - (10 if inputs["retest"] else 0), inputs
            assert {**now, "score": 0} == {**before, "score": 0}, inputs
        else:
            assert now == before, inputs
