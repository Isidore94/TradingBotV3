"""P13 mentor eval: the 40 plain questions, offline attach recall pinned at >= 90 %. Never runs --live."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mentor_eval  # noqa: E402
from mentor_packs import registry  # noqa: E402


def test_the_fixture_has_forty_plain_questions_with_real_pack_names():
    fixture = mentor_eval.load_fixture()
    questions = fixture["questions"]
    assert len(questions) == 40 and len({q["q"] for q in questions}) == 40
    names = set(registry.names())
    for item in questions:
        assert item["expected_packs"] and set(item["expected_packs"]) <= names, item["q"]
        assert item["must_mention"], item["q"]
        assert not item["q"].startswith("/"), "plain words, never commands"
    gates = [q for q in questions if "gate_pack" in q["expected_packs"]]
    assert gates and all(set(q["must_mention"]) == set(fixture["checklist"]) for q in gates)


def test_offline_attach_recall_is_at_least_ninety_percent():
    report = mentor_eval.offline_report(mentor_eval.load_fixture())
    assert report["questions"] == 40
    assert report["attach_recall"] >= 0.90, [row for row in report["rows"] if row["missed"]]


def test_the_trader_s_two_first_day_questions_attach_the_right_packs():
    rows = {row["q"]: row for row in mentor_eval.offline_report(mentor_eval.load_fixture())["rows"]}
    assert "journal_pack" in rows["what trades did I take today"]["attached"]
    assert rows["thinking of taking a short like ALL, thoughts?"]["attached"][0] == "gate_pack"


def test_offline_is_the_default_and_prints_the_recall(capsys):
    assert mentor_eval.main([]) == 0
    assert "attach recall" in capsys.readouterr().out
