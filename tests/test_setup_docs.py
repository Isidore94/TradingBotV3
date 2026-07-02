"""Tests for the setup encyclopedia and trade-plan calculator."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import setup_docs  # noqa: E402


REQUIRED_FIELDS = ("label", "group", "what", "detection", "entry", "stop", "targets", "evidence")


def test_every_doc_is_complete():
    assert len(setup_docs.SETUP_DOCS) >= 15
    for key, doc in setup_docs.SETUP_DOCS.items():
        for field in REQUIRED_FIELDS:
            assert field in doc, f"{key} missing {field}"
        assert isinstance(doc["detection"], list) and doc["detection"], f"{key} detection empty"


def test_resolve_families_aliases_and_fallback():
    key, doc = setup_docs.resolve_setup_doc("avwape_to_1stdev")
    assert key == "avwape_to_1stdev" and "Favorite" in doc["label"]
    key, _doc = setup_docs.resolve_setup_doc("mid_earnings_first_dev_retest")  # alias
    assert key == "mid_earnings_1stdev_retest"
    key, _doc = setup_docs.resolve_setup_doc("Something Unknown")
    assert key == "general"
    key, _doc = setup_docs.resolve_setup_doc("")
    assert key == "general"


def test_resolve_accepts_report_display_labels():
    # The priority report renders families as human labels; they must round-trip.
    key, _ = setup_docs.resolve_setup_doc("mid earnings EMA15 retest")
    assert key == "mid_earnings_ema15_retest"
    key, _ = setup_docs.resolve_setup_doc("mid earnings 1st-dev retest")
    assert key == "mid_earnings_1stdev_retest"
    key, _ = setup_docs.resolve_setup_doc("AVWAP retest followthrough")
    assert key == "avwap_retest_followthrough"
    key, _ = setup_docs.resolve_setup_doc("mid earnings above 2nd stdev")
    assert key == "playbook_second_dev_power_hold"


def test_resolve_family_from_candidates_prefers_first_real_match():
    family = setup_docs.resolve_setup_family_from_candidates(
        [None, "near_favorite_zone", "mid earnings ema15 retest", "avwap_breakout"]
    )
    assert family == "mid_earnings_ema15_retest"
    assert setup_docs.resolve_setup_family_from_candidates(["nonsense", ""]) == "general"
    assert setup_docs.resolve_setup_family_from_candidates([]) == "general"


def test_docs_grouped_for_display():
    groups = setup_docs.all_setup_docs_by_group()
    names = [name for name, _entries in groups]
    assert names[0] == "Main swing"
    assert sum(len(entries) for _name, entries in groups) == len(setup_docs.SETUP_DOCS)


def test_trade_plan_long_first_band_bounce_stops_at_avwape():
    bands = {"UPPER_1": 105.0, "UPPER_2": 110.0, "UPPER_3": 115.0, "LOWER_1": 95.0}
    plan = setup_docs.build_trade_plan(
        side="LONG",
        setup_family="avwap_band_bounce",
        favorite_signals=["BOUNCE_UPPER_1"],
        bands=bands,
        vwap=100.0,
        last_close=106.0,
    )
    assert plan["stop_label"] == "AVWAPE"
    assert plan["stop_price"] == 100.0
    assert plan["risk_per_share"] == 6.0
    assert plan["partial_label"] == "UPPER_2"
    assert abs(plan["partial_r"] - (110.0 - 106.0) / 6.0) < 1e-9
    assert abs(plan["final_r"] - (115.0 - 106.0) / 6.0) < 1e-9
    assert plan["stop_close_failures"] == 2


def test_trade_plan_short_defaults_to_protective_band():
    bands = {"UPPER_1": 105.0, "LOWER_1": 95.0, "LOWER_2": 90.0, "LOWER_3": 85.0}
    plan = setup_docs.build_trade_plan(
        side="SHORT",
        setup_family="avwap_breakout",
        bands=bands,
        vwap=100.0,
        last_close=94.0,
    )
    assert plan["stop_label"] == "UPPER_1"
    assert plan["stop_price"] == 105.0
    assert plan["risk_per_share"] == 11.0
    assert plan["partial_label"] == "LOWER_2"
    assert plan["final_label"] == "LOWER_3"


def test_trade_plan_stale_when_price_beyond_stop():
    bands = {"LOWER_1": 95.0, "UPPER_2": 110.0, "UPPER_3": 115.0}
    plan = setup_docs.build_trade_plan(
        side="LONG", setup_family="general", bands=bands, vwap=100.0, last_close=94.0
    )
    assert plan["risk_per_share"] is None
    assert plan["partial_r"] is None


def test_trade_plan_post_earnings_uses_tighter_close_discipline():
    plan = setup_docs.build_trade_plan(side="LONG", setup_family="post_earnings_52w_break", last_close=50.0)
    assert plan["stop_label"] == "POST_EARNINGS_AVWAPE"
    assert plan["stop_close_failures"] == 1


def test_trade_plan_power_hold_stops_at_zone_floor():
    bands = {"UPPER_1": 105.0, "UPPER_2": 110.0, "UPPER_3": 115.0}
    plan = setup_docs.build_trade_plan(
        side="LONG", setup_family="playbook_second_dev_power_hold", bands=bands, last_close=112.0
    )
    assert plan["stop_label"] == "UPPER_1"
    assert plan["stop_price"] == 105.0
