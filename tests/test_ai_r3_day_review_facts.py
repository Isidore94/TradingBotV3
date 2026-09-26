"""AI-R3: the night refreshes Day Review facts before a story can read them."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import pytest


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
    # S12 (2026-09-26): `family_side_evidence` now closes stage 1 directly after
    # the facts, so the facts are still inside it on every slate.
    assert runner._STAGE_ONE_LAST_SLOT == "family_side_evidence"
    assert names.index("family_side_evidence") == names.index("day_review_facts") + 1
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
    bar_options: dict = {}

    class Service:
        def build_index_for(self, session_date, **_kwargs):
            calls.append(("index", session_date))
            return {"session": session_date}

        def build_session_bars_for(self, session_date, **_kwargs):
            calls.append(("bars", session_date))
            bar_options.update(_kwargs)
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
    assert bar_options == {"reuse_existing": True}
    assert outcome["status"] == "ok", outcome
    assert outcome["pack"]["inputs_hash"] == "fresh"


def test_day_review_facts_failure_keeps_the_prior_verified_pack_bytes(tmp_path):
    """A partial night result must not replace the facts the page last verified."""
    from ai_jobs.day_review_facts import run_day_review_facts

    root = tmp_path / "day_review"
    prior = root / "sessions" / SESSION / "pack.json"
    prior.parent.mkdir(parents=True)
    prior.write_bytes(b'{"inputs_hash":"last-verified"}\n')

    class FailingService:
        def build_index_for(self, *_args, **_kwargs):
            return {"session": SESSION}

        def build_session_bars_for(self, *_args, **_kwargs):
            return {"SPY": [{"close": 100.0}]}

        def build_reads_for(self, *_args, **_kwargs):
            return []

        def build_pack_for(self, *_args, **_kwargs):
            return None

    outcome = run_day_review_facts(
        session_date=SESSION,
        now=datetime(2026, 9, 22, 1, 0),
        root=root,
        service=FailingService(),
    )

    assert outcome["status"] == "failed", outcome
    assert prior.read_bytes() == b'{"inputs_hash":"last-verified"}\n'


def test_day_review_facts_reuses_a_verified_exact_session_tape_without_a_download(monkeypatch):
    """The night refreshes facts; it must not replace a good tape with an empty retry."""
    import day_review_bars
    from ui.services.day_review_service import DayReviewService

    stored = {
        "SPY": [{
            "dt": datetime(2026, 9, 21, 6, 30), "open": 100.0,
            "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10,
        }]
    }
    monkeypatch.setattr(day_review_bars, "session_is_closed", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(day_review_bars, "read_session_bars", lambda session: stored if session == SESSION else None)
    monkeypatch.setattr(day_review_bars, "decided_symbols", lambda *_args, **_kwargs: pytest.fail("reused tape chose download symbols"))
    monkeypatch.setattr(day_review_bars, "fetch_session_bars", lambda *_args, **_kwargs: pytest.fail("reused tape downloaded"))
    monkeypatch.setattr(day_review_bars, "write_session_bars", lambda *_args, **_kwargs: pytest.fail("reused tape rewrote"))

    assert DayReviewService().build_session_bars_for(SESSION, reuse_existing=True) is stored


def test_day_review_facts_fetches_and_writes_when_the_exact_session_tape_is_missing(monkeypatch, tmp_path):
    """Reuse is a protection for verified bars, not a reason to invent a tape."""
    import day_review_bars
    from ui.services.day_review_service import DayReviewService

    fresh = {"SPY": [{"dt": datetime(2026, 9, 21, 6, 30), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10}]}
    writes: list[tuple[str, dict]] = []
    monkeypatch.setattr(day_review_bars, "session_is_closed", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(day_review_bars, "read_session_bars", lambda _session: None)
    monkeypatch.setattr(day_review_bars.daily_recap_reader, "RecapSources", lambda: object())
    monkeypatch.setattr(day_review_bars, "decided_symbols", lambda session, _sources: {"SPY"} if session == SESSION else set())
    monkeypatch.setattr(day_review_bars, "fetch_session_bars", lambda names, session: fresh if (names, session) == ({"SPY"}, SESSION) else pytest.fail("wrong canonical fetch"))
    monkeypatch.setattr(day_review_bars, "write_session_bars", lambda session, bars: writes.append((session, bars)) or tmp_path / "tape.parquet")

    assert DayReviewService().build_session_bars_for(SESSION, reuse_existing=True) == tmp_path / "tape.parquet"
    assert writes == [(SESSION, fresh)]


def test_day_review_facts_calls_an_empty_written_tape_unmeasured(monkeypatch, tmp_path):
    """A parquet path is not proof that the fetch supplied evidence rows."""
    import day_review_bars
    from ai_jobs.day_review_facts import run_day_review_facts

    empty_tape = tmp_path / "empty.parquet"
    empty_tape.write_bytes(b"empty")
    monkeypatch.setattr(day_review_bars, "read_session_bars", lambda _session: {})

    class Service:
        def build_index_for(self, *_args, **_kwargs):
            return {"session": SESSION}

        def build_session_bars_for(self, *_args, **_kwargs):
            return empty_tape

        def build_reads_for(self, *_args, **_kwargs):
            return []

        def build_pack_for(self, *_args, **_kwargs):
            return {"session_date": SESSION, "inputs_hash": "fresh"}

    outcome = run_day_review_facts(
        session_date=SESSION, now=datetime(2026, 9, 22, 1, 0), service=Service()
    )

    assert outcome["status"] == "ok", outcome
    assert outcome["tape"] == "unmeasured"
