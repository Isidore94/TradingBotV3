"""Trade Mentor citations: a foreign id rejects the reply, an uncited bullet is dropped."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_packs.citations import CitationRejected, check_citations, cited_ids  # noqa: E402

ALLOWED = {"ctx:auto_mode", "plan:risk:1"}


def test_cited_bullets_are_kept_and_uncited_are_dropped():
    reply = {
        "bullets": [
            {"text": "Auto is DESK", "evidence_refs": ["ctx:auto_mode"]},
            {"text": "Feels toppy", "evidence_refs": []},
        ]
    }
    result = check_citations(reply, ALLOWED)
    assert [row["text"] for row in result.kept] == ["Auto is DESK"]
    assert [row["text"] for row in result.dropped] == ["Feels toppy"]


def test_an_invented_id_rejects_the_whole_reply():
    reply = {
        "bullets": [
            {"text": "x", "evidence_refs": ["ctx:auto_mode"]},
            {"text": "y", "evidence_refs": ["pick:nope"]},
        ]
    }
    with pytest.raises(CitationRejected):
        check_citations(reply, ALLOWED)


@pytest.mark.parametrize("reply", [None, [], {"no": 1}, {"bullets": ["text"]}])
def test_a_malformed_reply_is_rejected(reply):
    with pytest.raises(CitationRejected):
        check_citations(reply, ALLOWED)


def test_inline_citations_are_found_in_order():
    text = "Short it [plan:risk:1] while [ctx:auto_mode] and [plan:risk:1]."
    assert cited_ids(text) == ["plan:risk:1", "ctx:auto_mode"]
