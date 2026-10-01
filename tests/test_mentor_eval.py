"""P13-P18 mentor eval: the 128 plain questions (P18: three journal statements), offline attach recall pinned at >= 97.5 %, and the style score.

Never runs --live."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mentor_eval  # noqa: E402
from mentor_packs import registry  # noqa: E402


def test_the_fixture_has_the_128_eval_questions_with_real_pack_names():
    fixture = mentor_eval.load_fixture()
    questions = fixture["questions"]
    assert len(questions) == 128 and len({q["q"] for q in questions}) == 128
    assert sum(1 for q in questions if q.get("concept")) == 2
    journal = [q for q in questions if q.get("journal")]
    # P18: a journal statement expects no pack: it is kept as a journal line and answered "Noted".
    assert len(journal) == 3 and all(q["expected_packs"] == [] and q["must_mention"] == ["Noted"] for q in journal)
    names = set(registry.names())
    for item in questions:
        # Only a concept question ("explain ... in one paragraph") or a journal statement needs no desk data.
        assert (item["expected_packs"] or item.get("concept") is True or item.get("journal") is True) and set(
            item["expected_packs"]) <= names, item["q"]
        assert item["must_mention"], item["q"]
        assert not item["q"].startswith("/"), "plain words, never commands"
    gates = [q for q in questions if "gate_pack" in q["expected_packs"]]
    assert gates and all(set(q["must_mention"]) == set(fixture["checklist"]) for q in gates)


def test_offline_attach_recall_is_at_least_ninety_seven_and_a_half_percent():
    report = mentor_eval.offline_report(mentor_eval.load_fixture())
    assert report["questions"] == 128
    assert report["attach_recall"] >= 0.975, [row for row in report["rows"] if row["missed"]]
    assert report["attach_recall"] == 1.0, [row for row in report["rows"] if row["missed"]]


def test_the_trader_s_two_first_day_questions_attach_the_right_packs():
    rows = {row["q"]: row for row in mentor_eval.offline_report(mentor_eval.load_fixture())["rows"]}
    assert "journal_pack" in rows["what trades did I take today"]["attached"]
    assert rows["thinking of taking a short like ALL, thoughts?"]["attached"][0] == "gate_pack"


def test_offline_is_the_default_and_prints_the_recall(capsys):
    assert mentor_eval.main([]) == 0
    assert "attach recall" in capsys.readouterr().out


def test_forty_simple_questions_are_tagged_and_no_pre_trade_one_is():
    questions = mentor_eval.load_fixture()["questions"]
    simple = [q for q in questions if q.get("simple")]
    assert len(simple) == 40
    assert not any("gate_pack" in q["expected_packs"] for q in simple)
    assert {"green or red so far?", "how am i doing today", "should I stop trading for today"} <= {q["q"] for q in simple}


def test_no_regime_is_expected_without_a_market_cue():
    from mentor_app import attach

    for item in mentor_eval.load_fixture()["questions"]:
        if "regime_pack" in item["expected_packs"]:
            assert attach.market_cue(item["q"]), item["q"]


# ---------------------------------------------------------------- P14 style score, three live replies
HOW_AM_I = ("You've had a productive day so far, net **+$11.75** [jrn:2026-09-30:totals].\n\n**Today's Performance:**\n"
            "*   **Wins:** 1 [jrn:2026-09-30:totals]\n*   **Trend:** a **bear channel** regime [tape:regime].\n\n"
            "You're holding steady. Anything specific you want to review?")
DAY_GOING = ("You've closed 2 trades today, resulting in a net gain of +$11.75 [jrn:2026-09-30:totals].\n\n*   **WIN:** "
             "LONG SHOP (+12.02 $) [jrn:2026-09-30:663018367e2e7d25bb43db6a]\n*   **LOSS:** LONG TWLO (-0.28 $) "
             "[jrn:2026-09-30:73a77b0232cb58f82f0a8281]\n\nYou have 21 positions still open [ctx:today].")
STOP = ("You have a net win of +$11.75 from two closed trades today [jrn:2026-09-30:totals]. \n\nYour current regime is "
        "a **bear channel** with lower highs since 2026-09-28 [tape:regime]. \n\nDo you have a specific trade in mind, "
        "or are you feeling the urge to overtrade?")


def test_the_style_score_on_three_live_replies():
    rows = [{"q": "how am i doing today", "reply": HOW_AM_I}, {"q": "how's the day going", "reply": DAY_GOING},
            {"q": "should I stop trading for today", "reply": STOP}, {"q": "what's the tape doing", "error": "boom"}]
    summary = mentor_eval.style_summary(rows, mentor_eval.load_fixture())
    how, day, stop = rows[0], rows[1], rows[2]
    assert how["simple"] and how["style"]["headers"] == 1 and not how["style_pass"]
    assert how["style_pass_after_app"], "the guard strips the header and the closing offer"
    assert day["style_pass"] and day["style"]["context_lines_unasked"] == 0
    assert stop["style"]["closing_question"] and stop["style"]["context_lines_unasked"] == 1 and not stop["style_pass"]
    assert summary["simple_questions"] == 3 and summary["style_pass_rate"] == round(1 / 3, 4)
    assert summary["style_pass_rate_after_app"] == 1.0
    assert summary["headers_share"] == round(1 / 3, 4) and summary["offer_or_closing_share"] == round(2 / 3, 4)
    assert summary["unasked_context_share"] == round(2 / 3, 4)
    assert summary["reply_chars_p50"] == float(sorted(len(r) for r in (HOW_AM_I, DAY_GOING, STOP))[1])
    assert "style" not in rows[3], "an errored question is not scored"


def test_rescore_reads_an_earlier_live_report(tmp_path, capsys):
    import json

    report = {"mode": "live", "rows": [{"q": "green or red so far?", "reply": "Green, +$11.75 [jrn:d:totals]."}]}
    path = tmp_path / "live.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert mentor_eval.main(["--rescore", str(path)]) == 0
    out = capsys.readouterr().out
    assert "ok " in out and "style: pass 1.0 on 1 simple" in out


def test_the_book_is_part_of_the_fixture_and_a_book_question_never_reads_focus():
    fixture = mentor_eval.load_fixture()
    assert "book" in fixture["raw_input_keys"] and set(fixture["book"]) <= set(fixture["known_symbols"])
    rows = {row["q"]: row for row in mentor_eval.offline_report(fixture)["rows"]}
    assert rows["which of my open shorts is most at risk into earnings"]["attached"][:2] == ["earnings_pack", "book_pack"]


def test_journal_statements_are_kept_and_a_question_taken_for_self_talk_scores_zero():
    rows = {row["q"]: row for row in mentor_eval.offline_report(mentor_eval.load_fixture())["rows"]}
    noted = [row for row in rows.values() if row.get("journal")]
    assert len(noted) == 3 and all(row["attached"] == [mentor_eval.JOURNAL] and row["recall"] == 1.0 for row in noted)
    fixture = {"now": "2026-09-30T15:00:00+00:00", "questions": [
        {"q": "I'm annoyed, I chased TWLO", "expected_packs": ["journal_pack"], "must_mention": ["x"]}]}
    report = mentor_eval.offline_report(fixture)
    assert report["rows"][0]["recall"] == 0.0 and report["rows"][0]["attached"] == []
