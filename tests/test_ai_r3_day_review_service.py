"""AI-R3: the night gets a source failure, while the page keeps its empty list."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_strict_day_review_reads_names_an_unreadable_journal_without_changing_page_behavior():
    from ui.services.day_review_service import DayReviewService

    class BrokenJournal:
        def entries_about(self, _session):
            raise OSError("journal segment is unavailable")

    service = DayReviewService(journal_service=BrokenJournal())

    assert service.build_reads_for("2026-09-18") == []
    assert service.build_reads_for("2026-09-18", strict=True) is None


def test_night_facts_keeps_the_prior_pack_when_the_real_service_cannot_read_the_journal(
    monkeypatch, tmp_path
):
    """The night must not call a source failure a fresh empty Day Review pack."""
    import day_review_pack
    from ai_jobs.day_review_facts import run_day_review_facts
    from ui.services.day_review_service import DayReviewService

    session = "2026-09-18"
    prior = day_review_pack.pack_path(session, root=tmp_path)
    prior.parent.mkdir(parents=True)
    prior.write_bytes(b'{"inputs_hash":"last-verified"}\n')

    class BrokenJournal:
        def entries_about(self, _session):
            raise OSError("journal segment is unavailable")

    service = DayReviewService(journal_service=BrokenJournal())
    monkeypatch.setattr(service, "build_index_for", lambda *_args, **_kwargs: {"session": session})
    monkeypatch.setattr(service, "build_session_bars_for", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        service,
        "build_pack_for",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("source failure built a pack")),
    )

    outcome = run_day_review_facts(
        session_date=session,
        now=datetime(2026, 9, 21, 5, 0),
        service=service,
        root=tmp_path,
    )

    assert outcome["status"] == "failed", outcome
    assert "reads could not be built" in outcome["reason"]
    assert prior.read_bytes() == b'{"inputs_hash":"last-verified"}\n'
