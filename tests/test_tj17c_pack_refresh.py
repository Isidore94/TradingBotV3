"""Saved stories follow the facts they were written from."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)
OLD = "2026-09-18"
END = "2026-09-21"


def _service(monkeypatch):
    from ui.services.day_review_service import DayReviewService

    service = DayReviewService()
    monkeypatch.setattr(service, "_regime_shifts", lambda _session: [])
    monkeypatch.setattr(service, "_d1_label_for", lambda _session: "")
    monkeypatch.setattr(service, "_internals_marks", lambda *_args: [])
    return service


def _payload(raw: str = "") -> dict:
    return {
        "entries": [], "forecast": {}, "story": None, "walkaway": None,
        "reads": [], "congruence": [], "report_card": {},
        "trades": [{"trade_id": "t1", "symbol": "ABC", "net_pnl": 12, "currency": "USD"}],
        "trade_reviews": [{"trade_id": "t1", "entry_raw": {"text": raw}}],
    }


def test_late_trade_words_make_old_story_stale_and_unchanged_pack_keeps_bytes(monkeypatch, tmp_path):
    import day_review_pack
    import project_paths

    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", tmp_path)
    service = _service(monkeypatch)
    before = service.build_pack_for(OLD, payload=_payload(), now=NOW, strict=True)
    path = day_review_pack.pack_path(OLD, root=tmp_path)
    original = path.read_bytes()
    assert service.build_pack_for(OLD, payload=_payload(), now=NOW, strict=True) == before
    assert path.read_bytes() == original

    late = _payload("I chased the open")
    late["day_story"] = {"inputs_hash": before["inputs_hash"]}
    assert service._story_freshness(OLD, late, NOW)["state"] == "stale"
    after = service.build_pack_for(OLD, payload=late, now=NOW, strict=True)
    assert after["inputs_hash"] != before["inputs_hash"]
    assert path.read_bytes() != original
    assert service._story_freshness(OLD, late, NOW)["state"] == "stale"


def test_recent_refresh_updates_only_existing_prior_pack_and_keeps_failure_bytes(monkeypatch, tmp_path):
    import day_review_pack
    import project_paths
    from ai_jobs.day_review_facts import refresh_recent_packs

    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", tmp_path)
    service = _service(monkeypatch)
    service.build_pack_for(OLD, payload=_payload(), now=NOW, strict=True)
    path = day_review_pack.pack_path(OLD, root=tmp_path)
    prior = path.read_bytes()
    monkeypatch.setattr(service, "read_day", lambda *_args, **_kwargs: _payload("late answer"))
    result = refresh_recent_packs(end_session=END, now=NOW, root=tmp_path, service=service, sessions=2)
    assert result["refreshed"] == [OLD]
    assert path.read_bytes() != prior
    updated = path.read_bytes()
    monkeypatch.setattr(service, "read_day", lambda *_args, **_kwargs: {**_payload(), "pack_sources_unread": ("journal",)})
    result = refresh_recent_packs(end_session=END, now=NOW, root=tmp_path, service=service, sessions=2)
    assert result["failed"][0]["session"] == OLD
    assert path.read_bytes() == updated
