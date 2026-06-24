import json
import sys
from datetime import date
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_is_bucket_upgrade_transition_table():
    from master_avwap_bucket_state import is_bucket_upgrade

    # genuine upgrades into a target bucket
    assert is_bucket_upgrade(None, "favorite_setup") is True
    assert is_bucket_upgrade("", "favorite_setup") is True
    assert is_bucket_upgrade("unbucketed", "high_conviction") is True
    assert is_bucket_upgrade("near_favorite_zone", "favorite_setup") is True
    assert is_bucket_upgrade("near_favorite_zone", "high_conviction") is True
    assert is_bucket_upgrade("favorite_setup", "high_conviction") is True  # strengthen
    assert is_bucket_upgrade("sma_breakout_tracking", "favorite_setup") is True

    # not upgrades
    assert is_bucket_upgrade("favorite_setup", "favorite_setup") is False
    assert is_bucket_upgrade("high_conviction", "high_conviction") is False
    assert is_bucket_upgrade("high_conviction", "favorite_setup") is False  # downgrade
    assert is_bucket_upgrade("favorite_setup", "near_favorite_zone") is False
    assert is_bucket_upgrade(None, "near_favorite_zone") is False  # not a target
    assert is_bucket_upgrade(None, "study") is False


def test_compute_bucket_upgrades_only_emits_target_entries():
    from master_avwap_bucket_state import compute_bucket_upgrades

    rows = [
        {"symbol": "NVDA", "side": "LONG", "priority_bucket": "favorite_setup"},
        {"symbol": "AAPL", "side": "LONG", "priority_bucket": "near_favorite_zone"},
    ]
    upgrades = compute_bucket_upgrades(rows, previous_state={})
    assert [u["symbol"] for u in upgrades] == ["NVDA"]  # AAPL is only near, not a target

    # already favorite last scan -> no re-fire
    prev = {"NVDA|LONG": {"bucket": "favorite_setup"}}
    assert compute_bucket_upgrades(rows, prev) == []


def test_record_scan_round_trip_fires_once(tmp_path):
    from master_avwap_bucket_state import record_scan_bucket_upgrades

    path = tmp_path / "master_avwap_bucket_state.json"
    rows = [
        {"symbol": "NVDA", "side": "LONG", "priority_bucket": "favorite_setup"},
        {"symbol": "AAPL", "side": "LONG", "priority_bucket": "near_favorite_zone"},
    ]

    first = record_scan_bucket_upgrades(rows, "2026-06-23", path=path)
    assert [u["symbol"] for u in first] == ["NVDA"]
    assert path.exists()

    # same scan again -> NVDA now favorite->favorite, no upgrade
    second = record_scan_bucket_upgrades(rows, "2026-06-24", path=path)
    assert second == []

    # AAPL upgrades near -> favorite on a later scan
    rows[1]["priority_bucket"] = "favorite_setup"
    third = record_scan_bucket_upgrades(rows, "2026-06-25", path=path)
    assert [u["symbol"] for u in third] == ["AAPL"]


def test_bucket_upgrades_feed_d1_focus_alert_payload(tmp_path):
    from master_avwap_lib.legacy import (
        build_master_avwap_d1_upgrade_alert_payload,
        format_master_avwap_d1_upgrade_alert_report,
    )
    from master_avwap_shared import (
        build_master_avwap_d1_flag_events,
        load_master_avwap_d1_upgrade_alerts,
    )

    payload = build_master_avwap_d1_upgrade_alert_payload(
        [
            {
                "symbol": "AAPL",
                "side": "LONG",
                "priority_bucket": "favorite_setup",
                "score": 242,
                "setup_family": "earnings_gap",
            },
            {
                "symbol": "TSLA",
                "side": "SHORT",
                "priority_bucket": "near_favorite_zone",
                "score": 190,
            },
        ],
        {"symbols": {}},
        bucket_upgrades=[
            {
                "symbol": "AAPL",
                "side": "LONG",
                "previous_bucket": "near_favorite_zone",
                "bucket": "favorite_setup",
            }
        ],
    )

    assert payload["alert_mode"] == "bucket_upgrades"
    assert list(payload["symbols"]) == ["AAPL"]
    assert payload["symbols"]["AAPL"]["bucket_upgrade_events"][0]["previous_bucket"] == "near_favorite_zone"

    report = format_master_avwap_d1_upgrade_alert_report(payload)
    assert "MASTER AVWAP D1 FOCUS ALERTS" in report
    assert "Near favorite zone -> Favorite setup" in report

    path = tmp_path / "master_avwap_d1_upgrade_alerts.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_master_avwap_d1_upgrade_alerts(path)
    flags = build_master_avwap_d1_flag_events(
        {},
        {},
        {},
        date(2026, 6, 23),
        d1_upgrade_alerts=loaded,
    )

    assert len(flags) == 1
    assert flags[0]["symbol"] == "AAPL"
    assert flags[0]["source"] == "bucket_upgrade"
    assert flags[0]["previous_bucket"] == "near_favorite_zone"
