"""TJ-14B item 2 - the overnight `mentor_question`, now with click options.

RED BEFORE THE FIX. On `e8c04f88` `NARRATION_JSON_SCHEMA` has no
`mentor_question_options`, nothing validates the narration beyond its
`sources`, and `latest_coaching_question` puts the same sentence on EVERY card
forever - there is no "once a day" and no way to answer it.

THE CONTRACT THESE PIN (plan.md 12.4 TJ-14 item 2)
--------------------------------------------------
* *"at most ONE a day: the overnight `mentor_question`, now with optional
  closed click options (schema extended, validated, <=4 options)"*.
* An output that fails the schema is rejected WHOLE and the last verified file
  stays (plan.md 12.3, the AI ground rule).

NO MODEL RUNS HERE. `request` is a fake; nothing touches the AI store, the DAS
or the off-hours lock.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

from tj14b_support import SESSION, keys_of, pacific, slot_at, state  # noqa: E402

DAY = SESSION.isoformat()


def _rollups(root: Path) -> str:
    """One weekly pack with one entry, so `allowed_source_ids` is decidable."""
    weekly = root / "weekly"
    weekly.mkdir(parents=True, exist_ok=True)
    pack = {
        "period_id": "2026-W38",
        "sessions": [{"session_date": DAY, "entries": [{"entry_id": "e-1"}]}],
    }
    (weekly / "2026-W38.json").write_text(json.dumps(pack), encoding="utf-8")
    return "journal:e-1"


def _narration(options: list[str], source: str) -> dict:
    return {
        "summary": "The week was quiet.",
        "changes": [],
        "open_questions": [],
        "mentor_question": "Did the open drive hold into the afternoon?",
        "mentor_question_options": options,
        "sources": [source],
    }


def _prior_file(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{DAY}.json"
    path.write_text(
        json.dumps(
            {
                "schema": "market_story_narration_v1",
                "session_date": DAY,
                "inputs_hash": "an-older-hash",
                "prompt_version": "market_story_narration_v1",
                "model": "prior",
                "narration": {
                    "summary": "The verified narration from last night.",
                    "changes": [],
                    "open_questions": [],
                    "mentor_question": "What did the gap do?",
                    "sources": [],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_an_ai_question_with_five_options_is_rejected_and_the_prior_file_stands(tmp_path):
    """Five clicks is not a closed set the card can draw, and a schema that is
    only advice to the model is not a schema. The last verified file stays."""
    from ai_jobs.market_story_narration import run_market_story_narration

    rollups = tmp_path / "rollups"
    out_dir = tmp_path / "narrations"
    source = _rollups(rollups)
    path = _prior_file(out_dir)
    before = path.read_bytes()

    result = run_market_story_narration(
        session_date=DAY,
        rollups_dir=rollups,
        out_dir=out_dir,
        request=lambda **_kw: {
            "summary": _narration(["A", "B", "C", "D", "E"], source),
            "model": "fake",
        },
    )

    assert result["status"] == "degraded_no_narrative"
    assert path.read_bytes() == before, "the prior verified narration was overwritten"


def test_four_options_are_accepted_and_stored_with_the_question(tmp_path):
    """Four is the ceiling, so four must pass."""
    from ai_jobs.market_story_narration import run_market_story_narration

    rollups = tmp_path / "rollups"
    out_dir = tmp_path / "narrations"
    source = _rollups(rollups)

    result = run_market_story_narration(
        session_date=DAY,
        rollups_dir=rollups,
        out_dir=out_dir,
        request=lambda **_kw: {
            "summary": _narration(["Up", "Down", "Chop", "No view"], source),
            "model": "fake",
        },
    )

    assert result["status"] == "ok"
    payload = json.loads((out_dir / f"{DAY}.json").read_text(encoding="utf-8"))
    assert payload["narration"]["mentor_question_options"] == ["Up", "Down", "Chop", "No view"]


def test_the_schema_names_the_options_field_with_its_ceiling():
    """The closed JSON schema is what the local grammar is compiled from, so a
    field the card reads that the schema never mentions is a field no verified
    output can carry."""
    from ai_jobs.market_story_narration import NARRATION_JSON_SCHEMA

    options = NARRATION_JSON_SCHEMA["properties"]["mentor_question_options"]

    assert options["maxItems"] == 4
    assert options["items"]["type"] == "string"


def test_the_ai_question_carries_its_click_options_onto_the_card():
    """*"I'm happy to click boxes"* - the overnight question becomes a click,
    not a sentence the trader has to answer in prose or ignore."""
    import mentor_questions

    payload = state(
        ai_question={
            "question": "Did the open drive hold into the afternoon?",
            "options": ["Yes", "No", "Partly", "Did not watch"],
        }
    )

    result = mentor_questions.pending(payload, slot_at(SESSION, 11))
    subject = next(
        item for item in list(result.asked) + list(result.carried)
        if item.kind == "ai_question"
    )

    assert set(subject.options) >= {"Yes", "No", "Partly", "Did not watch"}


def test_the_ai_question_is_asked_at_most_once_a_day():
    """*"at most ONE a day"*. Today the same coaching line is printed on every
    card of every day until a newer narration replaces it."""
    import mentor_questions

    payload = state(
        ai_question={"question": "Did the open drive hold?", "options": ["Yes", "No"]}
    )
    first = mentor_questions.pending(payload, slot_at(SESSION, 7))
    subject = next(
        item for item in list(first.asked) + list(first.carried)
        if item.kind == "ai_question"
    )

    payload["answered"] = {
        f"ai_question:{subject.subject_id}": {"answered_at": SESSION.isoformat()}
    }
    payload["now"] = pacific(SESSION, 8)
    later = mentor_questions.pending(payload, slot_at(SESSION, 8))

    assert ("ai_question", subject.subject_id) not in (
        keys_of(later.asked) | keys_of(later.carried)
    )


def test_no_ai_question_is_asked_when_the_night_produced_none():
    """A degraded night asks nothing rather than repeating an old question."""
    import mentor_questions

    result = mentor_questions.pending(state(ai_question={}), slot_at(SESSION, 11))

    assert "ai_question" not in [item.kind for item in list(result.asked) + list(result.carried)]
