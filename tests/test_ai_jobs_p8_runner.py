"""Plan to 8/10, packets P3 and P4 part 2: night AI runner plumbing."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))



def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ---------------------------------------------------------------------------
# Budget order: plan_review and improvement_ideas rank after enrichment
# ---------------------------------------------------------------------------


def test_budget_priority_puts_plan_review_and_ideas_between_enrichment_and_tags():
    from ai_jobs import runner

    assert runner.MODEL_SLOT_PRIORITY == (
        "daily_digest",
        "day_review_narration",
        "market_story_narration",
        "setup_research",
        "journal_enrichment",
        "plan_review",
        "improvement_ideas",
        "observation_tags",
    )
    order = sorted(
        ["ticker_briefs", "observation_tags", "improvement_ideas", "econ_brief",
         "plan_review", "journal_enrichment"],
        key=runner.model_slot_priority,
    )
    assert order == [
        "journal_enrichment", "plan_review", "improvement_ideas", "observation_tags",
        "econ_brief", "ticker_briefs",
    ]


# ---------------------------------------------------------------------------
# note_vocabulary_audit is gone: its report had no reader
# ---------------------------------------------------------------------------


def test_note_vocabulary_audit_slot_and_module_are_gone():
    import importlib.util

    from ai_jobs import runner

    names = [slot.name for slot in runner.default_slots() + runner.optional_slots()]
    assert "note_vocabulary_audit" not in names
    assert importlib.util.find_spec("ai_jobs.note_vocabulary_audit") is None
