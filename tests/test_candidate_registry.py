"""Packet D (plan.md sec 14.3/16.6): candidate registry with provenance."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from candidate_registry import (  # noqa: E402
    CandidateRegistry,
    SOURCE_USER,
    StaleWriterError,
)

NOW = datetime(2026, 7, 10, 10, 0)


def test_simultaneous_sources_do_not_duplicate_a_symbol():
    reg = CandidateRegistry()
    reg.add("AAPL", "LONG", "open_scan", now=NOW)
    reg.add("AAPL", "LONG", "focus", now=NOW)
    active = reg.active_candidates("LONG")
    assert len(active) == 1
    assert set(active[0].memberships) == {"open_scan", "focus"}


def test_removing_one_source_retains_other_memberships():
    reg = CandidateRegistry()
    reg.add("AAPL", "LONG", "open_scan", now=NOW)
    reg.add("AAPL", "LONG", "focus", now=NOW)
    reg.remove_source("AAPL", "LONG", "open_scan")
    candidate = reg.get("AAPL", "LONG")
    assert candidate.active
    assert set(candidate.memberships) == {"focus"}
    reg.remove_source("AAPL", "LONG", "focus")
    assert not reg.get("AAPL", "LONG").active


def test_user_entries_survive_every_automation_pass():
    reg = CandidateRegistry()
    reg.import_user_watchlist("LONG", ["NVDA", "AMD"], now=NOW)
    # automation churns: open scan replaces its set, leases expire, rotation
    reg.sync_source("open_scan", {"LONG": ["NVDA", "TSLA"]}, lease_minutes=30, now=NOW)
    reg.sync_source("open_scan", {"LONG": ["MSFT"]}, now=NOW)
    reg.expire_leases(now=NOW + timedelta(hours=5))
    assert not reg.remove_source("NVDA", "LONG", SOURCE_USER)  # automation actor
    for sym in ("NVDA", "AMD"):
        candidate = reg.get(sym, "LONG")
        assert candidate is not None and candidate.active
        assert SOURCE_USER in candidate.memberships
    # user edit is the only way user names leave
    reg.import_user_watchlist("LONG", ["AMD"], now=NOW)
    assert SOURCE_USER not in (reg.get("NVDA", "LONG").memberships or {})


def test_leases_expire_only_for_automation_sources():
    reg = CandidateRegistry()
    reg.add("TSLA", "SHORT", "auto_populate", lease_minutes=30, now=NOW)
    reg.add("META", "SHORT", SOURCE_USER, now=NOW)
    removed = reg.expire_leases(now=NOW + timedelta(minutes=31))
    assert ("TSLA", "SHORT", "auto_populate") in removed
    assert not reg.get("TSLA", "SHORT").active
    assert reg.get("META", "SHORT").active


def test_one_lifecycle_transition_produces_one_event():
    reg = CandidateRegistry()
    reg.add("AAPL", "LONG", "focus", now=NOW)
    event = reg.set_stage("AAPL", "LONG", "READY", now=NOW)
    assert event is not None and event.from_stage == "DEVELOPING" and event.to_stage == "READY"
    assert reg.set_stage("AAPL", "LONG", "READY", now=NOW) is None, "repeat stage must dedupe"
    again = reg.set_stage("AAPL", "LONG", "TRIGGERED", now=NOW)
    assert again is not None and again.to_stage == "TRIGGERED"


def test_stale_writer_cannot_erase_current_state(tmp_path):
    path = tmp_path / "registry.json"
    reg_a = CandidateRegistry()
    reg_a.add("AAPL", "LONG", "focus", now=NOW)
    reg_a.save(path)

    reg_b = CandidateRegistry.load(path)
    reg_b.add("NVDA", "LONG", "focus", now=NOW)
    reg_b.save(path)

    stale = CandidateRegistry()  # never saw the disk state
    stale.add("OLD", "LONG", "open_scan", now=NOW)
    with pytest.raises(StaleWriterError):
        stale.save(path)

    # the newer state survives
    reloaded = CandidateRegistry.load(path)
    assert reloaded.get("NVDA", "LONG") is not None


def test_restart_reconstructs_identical_active_candidates(tmp_path):
    path = tmp_path / "registry.json"
    reg = CandidateRegistry()
    reg.import_user_watchlist("LONG", ["NVDA"], now=NOW)
    reg.add("AAPL", "LONG", "focus", rank_score=77.0, now=NOW)
    reg.set_stage("AAPL", "LONG", "READY", now=NOW)
    reg.save(path)

    restarted = CandidateRegistry.load(path)
    assert [c.symbol for c in restarted.active_candidates("LONG")] == ["AAPL", "NVDA"]
    aapl = restarted.get("AAPL", "LONG")
    assert aapl.stage == "READY"
    assert aapl.rank_score == 77.0
    assert restarted.get("NVDA", "LONG").memberships[SOURCE_USER]


def test_live_pool_orders_by_stage_then_source_then_rank_and_caps():
    reg = CandidateRegistry()
    reg.add("BG1", "LONG", "background", rank_score=99.0, now=NOW)
    reg.add("FOC", "LONG", "focus", rank_score=10.0, now=NOW)
    reg.add("TRG", "LONG", "auto_populate", rank_score=5.0, now=NOW)
    reg.set_stage("TRG", "LONG", "TRIGGERED", now=NOW)
    reg.add("DEF", "LONG", "pullback_defiance", rank_score=50.0, now=NOW)
    reg.set_stage("DEF", "LONG", "DEFIANT", now=NOW)
    reg.add("BAD", "LONG", "focus", now=NOW)
    reg.set_stage("BAD", "LONG", "INVALID", now=NOW)

    pool = reg.derive_live_pool("LONG", max_size=3)
    assert pool == ["TRG", "DEF", "FOC"], pool
    assert "BAD" not in reg.derive_live_pool("LONG", max_size=10)


def test_watchlist_export_is_a_derived_view():
    reg = CandidateRegistry()
    reg.import_user_watchlist("SHORT", ["WOLF"], now=NOW)
    reg.add("MRNA", "SHORT", "open_scan", now=NOW)
    assert reg.export_watchlist_lines("SHORT") == ["MRNA", "WOLF"]
