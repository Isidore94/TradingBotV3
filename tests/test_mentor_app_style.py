"""P14 tone: the coach answers first and short; the app strips headers and closing offers, never the substance.

The three replies below are verbatim from the 2026-09-30 live eval on gemma4:12b (the trader's
"not spamming me with useless info" examples).
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import grounding, style  # noqa: E402
from mentor_app.chat_model import PERSONA_PROMPT, SYSTEM_PROMPT, TOOL_PROMPT  # noqa: E402

HOW_AM_I_DOING = (
    "You've had a productive day so far, finishing with a net gain of **+$11.75** [jrn:2026-09-30:totals] across 2 "
    "closed trades.\n\n**Today's Performance:**\n*   **Wins:** 1 [jrn:2026-09-30:totals]\n*   **Losses:** 1 (TWLO at "
    "-0.28 $) [jrn:2026-09-30:totals]\n*   **Trend:** You are successfully navigating the current **bear channel** "
    "regime [tape:regime], which has seen lower highs since 2026-09-28 (day 3) [tape:regime].\n\n**Current Status:**\n"
    "*   You have **21 open positions** [ctx:today].\n*   The desk is in **Auto mode** [ctx:auto_mode].\n*   Breadth "
    "shows **Technology (XLK)** as the strongest sector today [tape:breadth].\n\nYou're holding steady in a bearish "
    "environment. Anything specific you want to review?"
)
OVERTRADING = (
    "To determine if you are overtrading, I've checked today's activity and your historical patterns:\n\n"
    "**Today's Activity:**\n*   **Trades:** You closed 2 trades today [tilt:asof].\n*   **PnL:** You are currently "
    "net +$11.75 [tilt:asof].\n*   **Tilt Patterns:** There were no patterns of overtrading identified today (no bursts "
    "of opens, re-entries after losses, or size increases after losses) [tilt:none].\n\n**Historical Context:**\n*   "
    "Your historical data for patterns like \"bursts of opens\" or \"re-entries\" after a loss is currently too low to "
    "establish a reliable statistical trend (n < 30) [tilt:base:burst, tilt:base:reentry].\n\n**Verdict:**\nBased on "
    "today's activity alone, you have been very disciplined with only 2 closes. However, to give a broader answer on "
    "your general tendency, I'd need to look at your trade frequency over the last week or month.\n\nWould you like "
    "me to pull your trade count for the last week to see the broader trend?"
)
GREEN_OR_RED = (
    "Green so far today.\n\nYou've closed 2 trades for a net of **+$11.75** [jrn:2026-09-30:totals].\n\n**Today's "
    "performance:**\n*   **Win:** SHOP (swing) +$12.02 [jrn:2026-09-30:663018367e2e7d25bb43db6a]\n*   **Loss:** TWLO "
    "(swing) -$0.28 [jrn:2026-09-30:73a77b0232cb58f82f0a8281]\n\nThe regime remains a **bear channel** with lower "
    "highs since 2026-09-28 [ctx:regime]."
)
GATE_CHALLENGE = (
    "The M5 bounce cell has a C grade and a 51% win rate [gate:ALL:pick:ALL:m5cell:lrsi_cross_20]. Given the bear "
    "regime, does the setup give enough conviction to risk $100?"
)


# ---------------------------------------------------------------- the prompt
def test_the_system_prompt_carries_the_tone_rules_and_stays_byte_stable():
    for rule in ("Answer the question first", "Simple questions get 1-4 sentences", "No headers",
                 "Bullets only for lists of trades, names or dates", "No closing questions or offers",
                 "is background", "call the tool; never ask permission", "Cite ids inline"):
        assert rule in SYSTEM_PROMPT, rule
    assert "No headers" in PERSONA_PROMPT, "the frontier persona gets the tone too"
    assert "never ask permission" in TOOL_PROMPT and "never ask permission" not in PERSONA_PROMPT
    assert "earnings_pack" in TOOL_PROMPT and "No id, no claim" in SYSTEM_PROMPT
    assert '"how did today go" -> journal_pack(day=\'today\')\n' in TOOL_PROMPT, "no regime in the example"


def test_a_wrong_premise_is_said_first():
    """15:27 retest: "why did I lose money tuesday" accepted the premise on a +$53.47 day."""
    assert "If the question's premise is wrong by the data, say so in the first sentence, then answer" in PERSONA_PROMPT


# ---------------------------------------------------------------- the scorer
def test_the_scorer_sees_the_how_am_i_doing_wrapper():
    got = style.measure(HOW_AM_I_DOING, "how am i doing today")
    assert got["headers"] == 2 and got["bullets"] == 6 and got["chars"] == len(HOW_AM_I_DOING)
    assert got["closing_question"] and got["offer_phrases"] == 1
    assert got["context_lines_unasked"] == 3, "regime, Auto mode and breadth lines on a first-person question"
    assert not got["market_cue"] and not style.passes(got, simple=True)


def test_the_scorer_on_the_overtrading_and_green_or_red_replies():
    over = style.measure(OVERTRADING, "am I overtrading")
    assert over["headers"] == 3 and over["offer_phrases"] == 1 and over["closing_question"]
    green = style.measure(GREEN_OR_RED, "green or red so far?")
    assert green["headers"] == 1 and not green["closing_question"] and green["offer_phrases"] == 0
    assert green["context_lines_unasked"] == 1 and green["chars"] < 600
    assert not style.passes(green, simple=True), "one header fails the style pass"


def test_market_questions_never_count_their_context_as_unasked():
    reply = "The tape is a bear channel [tape:regime]; breadth favours tech [tape:breadth]."
    assert style.measure(reply, "whats the market like this morning")["context_lines_unasked"] == 0
    assert style.measure(reply, "should I take TSLA here")["context_lines_unasked"] == 0, "pre-trade intent"
    assert style.measure(reply, "green or red so far?")["context_lines_unasked"] == 1


def test_a_number_carrying_challenge_is_not_a_closing_question():
    got = style.measure(GATE_CHALLENGE, "im thinking of shorting ALL thoughts?")
    assert not got["closing_question"] and style.passes(got, simple=False)
    assert style.guard(GATE_CHALLENGE) == (GATE_CHALLENGE, [])


def test_passes_holds_simple_questions_to_600_chars_only():
    long_clean = {"chars": 900, "headers": 0, "offer_phrases": 0, "closing_question": False}
    assert style.passes(long_clean, simple=False) and not style.passes(long_clean, simple=True)
    assert style.passes(dict(long_clean, chars=600), simple=True)


# ---------------------------------------------------------------- the guard
def test_the_guard_leaves_how_am_i_doing_with_no_header_and_no_closing_question():
    out, removed = style.guard(HOW_AM_I_DOING)
    after = style.measure(out, "how am i doing today")
    assert after["headers"] == 0 and not after["closing_question"] and after["offer_phrases"] == 0
    assert removed == ["**Today's Performance:**", "**Current Status:**", "Anything specific you want to review?"]
    assert out.endswith("You're holding steady in a bearish environment.")
    for line in HOW_AM_I_DOING.split("\n"):
        if line.strip() and line.strip() not in removed and "Anything specific" not in line:
            assert line in out, f"substance kept byte for byte: {line!r}"


def test_the_guard_drops_an_offer_paragraph_and_bold_labels_but_keeps_every_citation():
    out, removed = style.guard(OVERTRADING)
    assert "Would you like" not in out and "**Verdict:**" not in out and removed[-1].startswith("Would you like")
    assert grounding.CITATION_RE.findall(out) == grounding.CITATION_RE.findall(OVERTRADING)
    assert out.endswith("over the last week or month.")


def test_a_markdown_header_becomes_bold_text():
    out, removed = style.guard("### Market Context\nBear channel [tape:regime].\n## Trades:\n- SHOP +12 [jrn:x:1]")
    assert out == "**Market Context**\nBear channel [tape:regime].\n**Trades**\n- SHOP +12 [jrn:x:1]" and removed == []
    assert style.measure(out, "what's the tape doing")["headers"] == 0


def test_a_bold_sentence_that_is_the_answer_is_never_dropped():
    reply = "**Green so far.**\nNet +$11.75 [jrn:2026-09-30:totals]."
    assert style.guard(reply) == (reply, [])


def test_a_trailing_question_with_data_or_without_an_ask_is_kept():
    for reply in ("Net +$11.75 [jrn:d:totals]. Is that enough for $100 of risk?",
                  "Net +$11.75 [jrn:d:totals]. Why did TWLO lose? It stopped out at the low [jrn:d:t2]."):
        assert style.guard(reply)[0] == reply


def test_the_guard_never_removes_a_line_with_a_citation_a_number_or_a_ticker():
    """Review of bba9077c: a bold label with a total, and offers carrying data, were deleted."""
    bold_total = "**Net +$11.75 over 2 trades [jrn:2026-09-30:totals]:**\n- SHOP +12.02 [jrn:a]"
    offer_with_data = "Net +$5 [jrn:x].\nI can also pull AMD, which reports in 3 days [earn:AMD]."
    happy = "Happy to walk through the 2 losses [jrn:l1] [jrn:l2]"
    ticker_ask = "Net +$5 [jrn:x].\nWant me to check AMD too?"
    for reply in (bold_total, offer_with_data, happy, ticker_ask):
        out, removed = style.guard(reply)
        assert out == reply and removed == [], reply


def _numbers_and_citations(text):
    return (sorted(grounding.NUMBER_RE.findall(grounding.CITATION_RE.sub(" ", text))),
            grounding.CITATION_RE.findall(text), sorted(re.findall(r"\[[^\[\]]+\]", text)))


def test_every_live_reply_keeps_all_its_citations_and_numbers_through_the_guard():
    """Invariant over the 100 replies of the 14:54 and 15:27 live evals (gemma4:12b, 2026-09-30)."""
    from conftest import load_fixture_contract

    replies = load_fixture_contract("mentor_eval_replies")["replies"]
    assert len(replies) == 100 and {r["run"] for r in replies} == {"14:54", "15:27"}
    changed = 0
    for item in replies:
        out, removed = style.guard(item["reply"])
        assert _numbers_and_citations(out) == _numbers_and_citations(item["reply"]), (item["run"], item["q"])
        assert all(not style.carries_substance(text) for text in removed), (item["q"], removed)
        changed += out != item["reply"]
    assert changed, "the guard still does its job on some replies"


# ---------------------------------------------------------------- earnings symbol grounding
PICK_ROWS = ("## pick_pack\n[pick:IOT:earn] Own earnings: unknown (no upcoming date in the earnings calendar)\n"
             "[pick:IOT:peer:PRGS] Peer PRGS earnings Wed 2026-09-30, today\n[pick:BFLY:earn] Own earnings: unknown\n"
             "[pick:BFLY:peer:NEOG] Peer NEOG earnings Tue 2026-10-06, in 6 days")
UNSEEN_CLAIM = ("**Note:** While IOT has a peer reporting today, your primary open longs (e.g., **QTUM**, **FCEL**, "
                "**QCOM**) do not have upcoming earnings listed in their specific pick packs.")


def test_an_earnings_claim_about_names_no_pack_showed_is_greyed_whole():
    reply = ("*   **IOT**: Peer **PRGS** reports today [pick:IOT:peer:PRGS].\n"
             "*   **BFLY**: a peer **NEOG** reports in 6 days [pick:BFLY:peer:NEOG].\n\n" + UNSEEN_CLAIM)
    out = grounding.mark_ungrounded_earnings(reply, [PICK_ROWS])
    lines = out.split("\n")
    assert lines[0] == reply.split("\n")[0] and lines[1] == reply.split("\n")[1], "grounded lines untouched"
    assert lines[-1] == (f'<span class="{grounding.UNCITED_CLAIM_CLASS}" style="color:{grounding.UNCITED_COLOR}">'
                         f"{UNSEEN_CLAIM}</span>")


def test_the_earnings_pack_grounds_every_name_it_listed():
    earn = "## earnings_pack\n[earn:QTUM] QTUM: no own report within 14 days (next 2026-11-10); no peer reports"
    rows = earn + "\n[earn:FCEL] FCEL: own report unknown\n[earn:QCOM] QCOM: REPORTS Wed 2026-10-07, in 7 days"
    assert grounding.mark_ungrounded_earnings(UNSEEN_CLAIM, [PICK_ROWS, rows]) == UNSEEN_CLAIM
    assert grounding.mark_ungrounded_earnings(UNSEEN_CLAIM, [rows]) != UNSEEN_CLAIM, "IOT was in no pack"


def test_lines_without_an_earnings_claim_or_with_desk_words_only_are_left_alone():
    packs = [PICK_ROWS]
    for line in ("QTUM is up 3% today [book:pos:1:QTUM].", "No earnings for the LONG book before CPI.",
                 "EPS for IOT is unknown [pick:IOT:earn]."):
        assert grounding.mark_ungrounded_earnings(line, packs) == line


# ---------------------------------------------------------------- the window
def _answer(text):
    return [
        json.dumps({"message": {"content": text}, "done": False}).encode(),
        json.dumps({"message": {"content": ""}, "done": True, "prompt_eval_count": 50, "eval_count": 5}).encode(),
    ]


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_the_window_shows_and_stores_the_guarded_reply_with_its_style(app, tmp_path, monkeypatch):
    from mentor_app import settings
    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.store import MentorChatStore
    from mentor_app.window import MentorWindow
    from mentor_packs.registry import make_pack

    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    monkeypatch.setattr(settings, "context_tokens", lambda: 8192)
    journal = make_pack("journal_pack", [{"id": "jrn:2026-09-30:totals", "text": "2 closed, net +11.75 $"}])
    win = MentorWindow(
        store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"), queue=PrefetchQueue(),
        stream_post=lambda url, payload, cancelled: _answer(HOW_AM_I_DOING),
        post=lambda url, payload, timeout: {}, pack_builder=lambda name, args: journal,
    )
    try:
        win._brain_ok, win._endpoint, win._model, win._native_tools = True, "http://x", "gemma4:12b", True
        win.send("how am i doing today")
        worker = win._worker
        assert worker is not None and worker.wait(5000)
        deadline = time.monotonic() + 5
        while win._worker is not None and time.monotonic() < deadline:
            app.processEvents()
        win._io.submit(lambda: None).result(5)
        shown = win.transcript.toPlainText()
        assert "Anything specific" not in shown and "Today's Performance:" not in shown
        assert "You're holding steady in a bearish environment." in shown and "21 open positions" in shown
        row = win.store.turns()[-1]
        assert row["text"] == HOW_AM_I_DOING, "the turn log keeps the model's raw words"
        assert "Anything specific" not in win.chat.turns[-1].text, "the conversation carries what was shown"
        kept = json.loads(row["timings_json"])["style"]
        assert kept["headers"] == 2 and kept["closing_question"] and kept["context_lines_unasked"] == 3
        assert kept["stripped"][-1] == "Anything specific you want to review?"
    finally:
        win.shutdown()
        win.deleteLater()


def test_names_the_earnings_pack_only_listed_as_not_read_never_ground_a_claim():
    rows = "## earnings_pack\n[earn:QCOM] QCOM: REPORTS Wed 2026-10-07, in 7 days\n[earn:more] 2 more not listed (no earnings read for them; say so, never guess): QTUM, FCEL"
    claim = "QTUM and FCEL have no earnings this week."
    assert grounding.mark_ungrounded_earnings(claim, [rows]) != claim
    assert grounding.mark_ungrounded_earnings("QCOM reports Wednesday [earn:QCOM].", [rows]) == "QCOM reports Wednesday [earn:QCOM]."
