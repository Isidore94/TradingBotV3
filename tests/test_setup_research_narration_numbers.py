"""Setup research narration keeps only statements that quote a number from the facts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import ai_summary  # noqa: E402
from ai_jobs import setup_research  # noqa: E402

VIEW = {"eligible_policies": [{"recipe_id": "r1", "stats": {"n": 41, "mean_r": 0.734, "win_rate": 0.56}}]}


def _line(text: str) -> dict:
    return {"statement": text, "confidence": "medium", "evidence_refs": ["setup_research.facts"]}


def test_boilerplate_with_no_number_from_the_facts_is_not_published():
    """2026-09-24: twelve minutes for 'several recipes show promising results' and no number."""
    summary = {
        "executive_summary": "Mixed results across various recipes.",
        "what_is_working": [_line("Several recipes show promising results.")],
        "risk_notes": [_line("Limited data raises concerns.")],
    }
    with pytest.raises(ValueError, match="no number"):
        setup_research._keep_quoted_statements(summary, VIEW)


def test_only_statements_quoting_the_facts_survive():
    summary = {
        "executive_summary": "Mixed.",
        "what_is_working": [_line("r1 has n=41 at +0.73R."), _line("Several recipes work.")],
        "what_is_not_working": [_line("Win rate 56% only.")],
        "risk_notes": [_line("Invented: n=987654.")],
    }
    kept = setup_research._keep_quoted_statements(summary, VIEW)
    assert [row["statement"] for row in kept["what_is_working"]] == ["r1 has n=41 at +0.73R."]
    assert [row["statement"] for row in kept["what_is_not_working"]] == ["Win rate 56% only."]
    assert kept["risk_notes"] == []
    assert kept["executive_summary"] == "Mixed."


def test_narrate_applies_the_filter(monkeypatch):
    monkeypatch.setattr(ai_summary, "local_provider_enabled", lambda: True)
    monkeypatch.setattr(ai_summary, "local_model", lambda tier="medium": "stub")
    monkeypatch.setattr(setup_research, "_evidence_package", lambda pack: {
        "package_id": "p", "sources": [{"sha256": "x", "content": {**VIEW, "narrated": {}}}],
    })
    monkeypatch.setattr(ai_summary, "request_ai_summary", lambda **_kw: {
        "model": "stub", "summary": {"what_is_working": [_line("Nothing measured.")]},
    })
    with pytest.raises(ValueError, match="no number"):
        setup_research._narrate({})
