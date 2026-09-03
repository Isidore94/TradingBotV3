"""P10 A4 - the words the trader keeps writing that no code says.

The nightly `journal_enrichment` scope is gated, so this is deterministic: it
reads the annotation log and the two shipped vocabularies and writes one Markdown
page. **It proposes no code and adds none.** A vocabulary code is permanent and
never reused, and coining one from a frequency count would be inventing the
trader's categories for them.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _log(path: Path, notes: list[tuple[str, str]]) -> Path:
    import json

    lines = []
    for symbol, note in notes:
        lines.append(
            json.dumps(
                {
                    "schema_version": 1,
                    "event_id": symbol,
                    "event_type": "veto",
                    "symbol": symbol,
                    "session_date": "2026-09-02",
                    "created_at": "2026-09-02T10:00:00",
                    "source": "chart_review",
                    "surface": "chart_review",
                    "note": note,
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_it_writes_a_page_even_on_a_day_with_no_notes(tmp_path):
    """A missing page and a page saying "no notes" are different facts."""
    from ai_jobs import note_vocabulary_audit as audit

    source = tmp_path / "annotations.jsonl"
    source.write_text("", encoding="utf-8")

    result = audit.run_note_vocabulary_audit(
        session_date="2026-09-02",
        annotations_path=source,
        output_dir=tmp_path / "reports",
        now=datetime(2026, 9, 3, 4, 0),
    )

    assert result["status"] == "ok"
    page = Path(result["outputs"][0])
    assert page.exists()
    assert "No notes today" in page.read_text(encoding="utf-8")


def test_a_recurring_word_with_no_code_is_named_and_counted(tmp_path):
    from ai_jobs import note_vocabulary_audit as audit

    source = _log(
        tmp_path / "annotations.jsonl",
        [
            ("AAA", "gap is unfilled and I do not trust it"),
            ("BBB", "unfilled gap again, waiting"),
            ("CCC", "another unfilled one"),
        ],
    )

    result = audit.run_note_vocabulary_audit(
        session_date="2026-09-02",
        annotations_path=source,
        output_dir=tmp_path / "reports",
    )
    text = Path(result["outputs"][0]).read_text(encoding="utf-8")

    assert "| unfilled | 3 |" in text
    assert "3 note(s)" in result["reason"]
    # Every note is quoted whole as well as counted: the sentence is the value.
    assert "gap is unfilled and I do not trust it" in text


def test_a_word_the_vocabulary_already_uses_is_not_reported(tmp_path):
    """The page is about what has NO code. Listing coded words would bury it."""
    from ai_jobs import note_vocabulary_audit as audit
    from ui.annotations.vocabulary import load_veto_vocabulary

    label_word = ""
    for reason in load_veto_vocabulary().reasons:
        for word in audit._tokens(reason.label):
            if word not in audit._STOP_WORDS and len(word) > 4:
                label_word = word
                break
        if label_word:
            break
    assert label_word, "the vocabulary has no usable label word to test with"

    source = _log(
        tmp_path / "annotations.jsonl",
        [("AAA", f"{label_word} {label_word} zorbling")],
    )
    counts = dict(
        audit.uncoded_word_counts(
            audit.collect_notes(
                [
                    __import__("json").loads(line)
                    for line in source.read_text(encoding="utf-8").splitlines()
                ]
            )
        )
    )

    assert "zorbling" in counts
    assert label_word not in counts


def test_no_code_is_ever_added_by_machine():
    """The refusal, pinned in the source rather than trusted."""
    source = (SCRIPTS_DIR / "ai_jobs" / "note_vocabulary_audit.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("save_local_setting", "write_text(json", "vocab_version ="):
        assert forbidden not in source, forbidden
    assert "No code is ever added by machine" in source


def test_the_slot_is_appended_and_calls_no_model():
    """Later phases append; they never reorder."""
    from ai_jobs.runner import default_slots

    names = [slot.name for slot in default_slots()]
    assert "note_vocabulary_audit" in names
    # PAIRWISE, not by index: three packets have now edited an index assertion.
    assert names.index("journal_import") < names.index("note_vocabulary_audit")
    assert names.index("veto_cohort_grading") < names.index("note_vocabulary_audit")
    assert names.index("note_vocabulary_audit") < names.index("evidence_report")

    slot = next(item for item in default_slots() if item.name == "note_vocabulary_audit")
    assert slot.reserve_minutes == 5.0
    assert "no model" in slot.description
