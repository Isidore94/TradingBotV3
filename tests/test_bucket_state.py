import sys
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
