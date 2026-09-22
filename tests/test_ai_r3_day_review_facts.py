"""AI-R3: the night refreshes Day Review facts before a story can read them."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys


SESSION = "2026-09-21"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_day_review_facts_is_the_stage_one_tail_and_is_on_every_night_slate(tmp_path):
    """Facts must exist before model slots, including an unattended Sunday retry."""
    from ai_jobs import runner

    slots = runner.default_slots()
    names = [slot.name for slot in slots]
    facts = next(slot for slot in slots if slot.name == "day_review_facts")

    assert names.index("day_review_facts") == names.index("measured_report") + 1
    assert runner._STAGE_ONE_LAST_SLOT == "day_review_facts"
    assert facts.uses_model is False
    assert facts.max_attempts == 3
    assert facts.reserve_minutes == 5.0
    for kind in ("weeknight", "saturday", "sunday"):
        slate = runner.slots_for(kind, session_date=SESSION, ledger_path=tmp_path / "ledger.jsonl")
        assert "day_review_facts" in [slot.name for slot in slate], kind


def test_day_review_facts_refreshes_a_stale_pack_through_the_canonical_four_service_calls():
    """A pack is built even when no Day Review page was opened after the close."""
    from ai_jobs.day_review_facts import run_day_review_facts

    calls: list[tuple[str, str]] = []

    class Service:
        def build_index_for(self, session_date, **_kwargs):
            calls.append(("index", session_date))
            return {"session": session_date}

        def build_session_bars_for(self, session_date, **_kwargs):
            calls.append(("bars", session_date))
            return {"SPY": [{"close": 100.0}]}

        def build_reads_for(self, session_date, **_kwargs):
            calls.append(("reads", session_date))
            return [{"verdict": "right"}]

        def build_pack_for(self, session_date, **_kwargs):
            calls.append(("pack", session_date))
            return {"schema": "day_review_pack_v1", "session_date": session_date, "inputs_hash": "fresh"}

    outcome = run_day_review_facts(
        session_date=SESSION,
        now=datetime(2026, 9, 22, 1, 0),
        service=Service(),
    )

    assert calls == [("index", SESSION), ("bars", SESSION), ("reads", SESSION), ("pack", SESSION)]
    assert outcome["status"] == "ok", outcome
    assert outcome["pack"]["inputs_hash"] == "fresh"
