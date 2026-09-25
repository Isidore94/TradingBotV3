"""Plan to 8/10, packets P3 and P4 part 2: night AI runner plumbing."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

ET = ZoneInfo("America/New_York")


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


# ---------------------------------------------------------------------------
# Goal map: every slot names the goal it serves; its ledger rows carry it
# ---------------------------------------------------------------------------


LEAD_GOAL_MAP = {
    "journal_import": "journal",
    "journal_auto_tag": "journal",
    "journal_enrichment": "journal",
    "preference_trade_outcomes": "journal",
    "setup_keys_narration": "permutations",
    "read_grades_mature": "market_read",
    "prediction_contrast": "market_read",
    "market_story_rollups": "market_read",
    "market_story_narration": "market_read",
    "econ_brief": "market_read",
    "daily_digest": "market_read",
    "day_review_facts": "coaching",
    "day_review_narration": "coaching",
    "week_review_narration": "coaching",
    "week_questions": "coaching",
    "exit_note_fields": "coaching",
    "observation_tags": "coaching",
    "plan_review": "coaching",
    "improvement_ideas": "coaching",
    "ticker_briefs": "coaching",
    "outcome_sweep": "ops",
    "evidence_report": "ops",
    "measured_report": "ops",
    "sidecar_completion": "ops",
    "review_policy_draft": "ops",
    "ai_summary": "ops",
}
SETUP_GOALS = {"setup_quality", "permutations"}
SETUP_SLOTS = (
    "veto_cohort_grading", "like_cohort_grading", "pass_cohort_grading",
    "rejection_cohort_grading", "setup_research", "theta_pick_grading", "miss_contrast",
)


def test_every_slot_declares_a_goal_from_the_fixed_set():
    from ai_jobs import runner

    assert runner.SLOT_GOALS == (
        "trade_identification", "setup_quality", "permutations", "coaching",
        "market_read", "journal", "ops",
    )
    slots = runner.default_slots() + runner.optional_slots()
    for slot in slots:
        assert slot.goal in runner.SLOT_GOALS, slot.name
    goals = {slot.name: slot.goal for slot in slots}
    for name, goal in LEAD_GOAL_MAP.items():
        assert goals[name] == goal, name
    for name in SETUP_SLOTS:
        assert goals[name] in SETUP_GOALS, name


def test_a_slot_ledger_row_carries_its_goal(tmp_path, monkeypatch):
    from ai_jobs import runner, store, window

    monkeypatch.setattr(store, "store_available", lambda: (True, "ready"))
    monkeypatch.setattr(window, "launch_allowed", lambda *a, **k: (True, "window open"))
    monkeypatch.setattr(window, "market_session_block", lambda *a, **k: "")
    led = tmp_path / "ledger.jsonl"
    slots = [
        runner.JobSlot(name="a", run=lambda **k: {"status": "ok"}, goal="journal"),
        runner.JobSlot(name="b", run=lambda **k: {"status": "failed"}, goal="coaching"),
    ]
    runner.run_slots(slots, now=datetime(2026, 8, 12, 2, 0, tzinfo=ET), ledger_path=led)
    assert [(row["job"], row.get("goal")) for row in _rows(led)] == [
        ("a", "journal"), ("b", "coaching"),
    ]


def test_the_ollama_probe_row_is_an_ops_row(tmp_path):
    from ai_jobs import ollama_probe

    led = tmp_path / "ledger.jsonl"
    ollama_probe.record_probe(True, "ok", session_date="2026-08-11", path=led)
    assert _rows(led)[0]["goal"] == "ops"
