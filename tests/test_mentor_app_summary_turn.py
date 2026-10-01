"""P20: a brief / tape summary ask is answered in three plain sentences (live eval 2026-10-01).

Six simple questions ran 700-1800 chars with bold headers and bullets the guard could not shorten (every
line cited). The app now adds a per-turn instruction after the user message, caps num_predict for that
turn, and the guard strips markdown emphasis and bullet markers on those turns - formatting only."""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import attach, brain, style  # noqa: E402

NOW = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)
KNOWN = {"AMD": "LONG", "NVDA": "LONG", "TSLA": "SHORT", "QCOM": ""}
BOOK = ["AMD", "NVDA", "TSLA"]
#: The six simple questions that failed the style check after the guard in the 2026-10-01 live eval.
SIX = ["what's the macro brief say today", "whats the macro brief say today", "is the playbook bullish or bearish",
       "anything about yields in the brief", "what did the night brief say", "whats the market like this morning"]
NOT_SUMMARY = ["should I buy NVDA here", "im thinking of shorting TSLA thoughts?", "what does the brief say about NVDA",
               "anything about AMD in the brief", "is the playbook today bullish or bearish and does my book match it",
               "how did I do today", "whats SPY doing",
               # review of 91498f4a: a tickerless pre-trade question plans only regime_pack ("trade intent")
               "should I buy here?", "about to short this, ok?", "should I take this trade",
               "should I enter now or wait", "should I short the open", "buying calls here?",
               "whats the market like, should I buy?",
               # a lowercase universe ticker anywhere is a ticker question; the indexes count too
               "what does the brief say about amd", "what does the brief say about spy",
               "whats qqq like this morning", "what about the vix in the brief",
               # a calendar is a list, not a brief
               "what's on the econ calendar tomorrow"]
#: Re-review of dcf5a45a: every simple turn gets the plain guard (formatting only), never the instruction.
SIMPLE_PLAIN = ["how did I do last week", "how did I do today", "whats SPY doing", "what's on the econ calendar tomorrow",
                "am I tilting", "what are my rules on shorts", "any news on TSLA",
                "in one line, how should I feel about my trading this week",
                # an explicit one-line ask wins over the planner's "trade intent": plain only, never a cap
                "briefly, should I buy here?", "should I short the open in one line"]
#: Not simple: a pre-trade check, a list, a comparison or an explanation keeps the default guard.
DEFAULT_TURN = ["should I buy NVDA here", "im thinking of shorting TSLA thoughts?", "should I buy here?",
                "compare NVDA and AMD for a long", "what am I holding", "anything reporting this week in my longs?",
                "explain my win rate by setup and tell me which to stop trading", "what alerts fired today",
                "is the playbook today bullish or bearish and does my book match it"]


def _shape(text):
    return attach.turn_shape(text, attach.plan_attachments(text, KNOWN, NOW, book=BOOK), KNOWN)


@pytest.mark.parametrize("text", SIX)
def test_the_six_summary_asks_get_the_three_sentence_instruction_and_a_token_cap(text):
    shape = _shape(text)
    assert shape.pop("turn_instruction").startswith(attach.SUMMARY_INSTRUCTION)
    assert shape == {"max_tokens": attach.SUMMARY_MAX_TOKENS, "plain": True}
    assert attach.SUMMARY_INSTRUCTION == ("Answer in at most three short sentences, under 500 characters in all. "
                                          "Prose only: no bullets, no bold, no headers. Cite ids inline.")
    assert attach.BRIEF_RECIPE == "Lead with the bottom line, then the playbook, then one risk."
    assert attach.SUMMARY_MAX_TOKENS == 260


def test_the_brief_recipe_rides_only_with_the_fundamentals_pack():
    brief = _shape("is the playbook bullish or bearish")["turn_instruction"]
    assert brief == f"{attach.SUMMARY_INSTRUCTION} {attach.BRIEF_RECIPE}"
    for night_only in ("what did the night say I'm missing", "what changed in the tape since yesterday",
                       "whats the market like this morning"):
        assert _shape(night_only)["turn_instruction"] == attach.SUMMARY_INSTRUCTION, night_only


@pytest.mark.parametrize("text", NOT_SUMMARY)
def test_a_pre_trade_ticker_or_calendar_question_never_gets_the_summary_instruction(text):
    assert "turn_instruction" not in _shape(text) and "max_tokens" not in _shape(text)


@pytest.mark.parametrize("text", SIMPLE_PLAIN)
def test_a_simple_turn_gets_the_plain_guard_and_no_instruction(text):
    assert _shape(text) == {"plain": True}


@pytest.mark.parametrize("text", DEFAULT_TURN)
def test_a_question_that_is_not_simple_keeps_the_default_turn(text):
    assert _shape(text) == {}


def test_the_simple_rule_never_marks_a_pre_trade_fixture_question_simple():
    import mentor_eval

    fixture = mentor_eval.load_fixture()
    now = datetime.fromisoformat(fixture["now"])
    for item in fixture["questions"]:
        plan = attach.plan_attachments(item["q"], fixture["known_symbols"], now, book=fixture["book"])
        if "gate_pack" in item["expected_packs"]:
            assert attach.turn_shape(item["q"], plan, fixture["known_symbols"]) == {}, item["q"]


def _answer(text):
    return [json.dumps({"message": {"content": text}, "done": False}).encode(),
            json.dumps({"message": {"content": ""}, "done": True}).encode()]


def _turn(model="gemma4:12b", **shape):
    sent = []
    brain.run_turn([{"role": "system", "content": "s"}, {"role": "user", "content": "is the playbook bullish"}],
                   model=model, endpoint="http://x", native_tools=True, tools=[{"function": {"name": "t"}}],
                   stream_post=lambda url, payload, cancelled: sent.append(payload) or _answer("ok [a:b:c]."),
                   **shape)
    return sent[0]


def test_the_instruction_follows_the_user_message_and_num_predict_is_capped_for_that_turn_only():
    payload = _turn(turn_instruction=attach.SUMMARY_INSTRUCTION, max_tokens=attach.SUMMARY_MAX_TOKENS)
    user = [m for m in payload["messages"] if m["role"] == "user"][-1]["content"]
    assert user.startswith("is the playbook bullish") and user.endswith(attach.SUMMARY_INSTRUCTION)
    assert payload["options"]["num_predict"] == attach.SUMMARY_MAX_TOKENS
    plain = _turn()
    assert "num_predict" not in plain["options"]
    assert [m for m in plain["messages"] if m["role"] == "user"][-1]["content"] == "is the playbook bullish"


def test_a_thinking_model_gets_the_instruction_but_no_cap_its_reasoning_would_eat():
    payload = _turn(model="gpt-oss:20b", turn_instruction=attach.SUMMARY_INSTRUCTION,
                    max_tokens=attach.SUMMARY_MAX_TOKENS)
    assert "num_predict" not in payload["options"]
    assert [m for m in payload["messages"] if m["role"] == "user"][-1]["content"].endswith(attach.SUMMARY_INSTRUCTION)


LONG_REPLY = """### Macro brief
**Bottom line:** risk-on into CPI, futures +0.4% [fund:2026-09-30:bottom_line].

**Playbook:**
- Buy pullbacks to the 5 day AVWAP in NVDA and AMD [fund:2026-09-30:playbook:1]
* Fade gaps above 5,820 on SPY [fund:2026-09-30:playbook:2]
+ **Risk:** CPI at 08:30 ET; a hot print flips it [fund:2026-09-30:risk:1]"""


def _words(text):
    return re.findall(r"[A-Za-z0-9.:,+%\-\[\]]+", text.replace("*", " ").replace("#", " "))


def test_the_plain_guard_strips_formatting_only_and_keeps_every_id_number_and_ticker():
    text, _removed = style.guard(LONG_REPLY, plain=True)
    assert "**" not in text and "#" not in text
    assert not any(re.match(r"\s*[-*+]\s", line) for line in text.split("\n"))
    assert attach.cited_ids(text) == attach.cited_ids(LONG_REPLY) and len(attach.cited_ids(text)) == 4
    assert re.findall(r"\d+", text) == re.findall(r"\d+", LONG_REPLY)
    assert re.findall(r"\b[A-Z]{2,5}\b", text) == re.findall(r"\b[A-Z]{2,5}\b", LONG_REPLY)
    kept = [w for w in _words(LONG_REPLY) if w not in ("-", "+", "Playbook:")]
    assert all(w in text for w in kept), [w for w in kept if w not in text]
    assert "Bottom line: risk-on into CPI" in text and "Buy pullbacks to the 5 day AVWAP" in text


def test_the_default_guard_leaves_bold_and_bullets_alone():
    text, _ = style.guard(LONG_REPLY)
    assert "**Bottom line:**" in text and "- Buy pullbacks" in text


def test_a_spaced_sign_is_never_read_as_a_bullet_marker():
    for line in ("- 0.8% SPY [tape:breadth]", "+ 3 pts on QQQ [tape:qqq]", "- .5 ATR", "- $2 gap [news:AMD:1]"):
        assert style.guard(line, plain=True)[0] == line
    assert style.guard('- "risk-on" [fund:x:bl]', plain=True)[0] == '"risk-on" [fund:x:bl]'
    assert style.guard("* [fund:x:bl] says risk-on", plain=True)[0] == "[fund:x:bl] says risk-on"


def test_the_eval_guards_a_row_the_way_the_live_run_turned_it():
    import mentor_eval

    rows = [{"q": "is the playbook bullish or bearish", "reply": LONG_REPLY, "plain": True},
            {"q": "what am I holding", "reply": LONG_REPLY},
            {"q": "is the playbook bullish or bearish", "reply": LONG_REPLY}]
    mentor_eval.style_summary(rows, mentor_eval.load_fixture())
    assert rows[0]["style"]["bullets"] == 3 and rows[0]["style_after_app"]["bullets"] == 0
    assert rows[1]["style_after_app"]["bullets"] == 3, "a non-summary question keeps the default guard"
    assert rows[2]["style_after_app"]["bullets"] == 3, "an older report without the run's flag is not guessed"


def test_the_live_eval_stores_the_turn_shape_on_its_row(monkeypatch, tmp_path):
    import mentor_eval
    from mentor_app import checklist, settings
    from mentor_packs import context_pack, journal_pack, registry

    seen = []

    def fake_turn(messages, **kwargs):
        seen.append(kwargs)
        return {"text": "Risk-on [fund:x:bl].", "attached": [], "tool_calls": [], "first_token_ms": 5,
                "turn_instruction": kwargs.get("turn_instruction", ""), "max_tokens": kwargs.get("max_tokens")}

    monkeypatch.setattr(settings, "gpu_block_reason", lambda: "")
    monkeypatch.setattr(settings, "mentor_model", lambda present: "gemma4:12b")
    monkeypatch.setattr(brain, "model_capabilities", lambda *a, **k: ("tools",))
    monkeypatch.setattr(brain, "run_turn", fake_turn)
    monkeypatch.setattr(context_pack, "build", lambda: registry.make_pack("context_pack", []))
    monkeypatch.setattr(journal_pack, "recent_symbols", lambda: [])
    monkeypatch.setattr(checklist, "covered", lambda text: set())
    fixture = {"now": NOW.isoformat(), "known_symbols": KNOWN, "book": BOOK,
               "questions": [{"q": "is the playbook bullish or bearish", "expected_packs": [], "simple": True},
                             {"q": "should I buy here?", "expected_packs": []},
                             {"q": "how did I do last week", "expected_packs": [], "simple": True}]}
    report = mentor_eval.live_report(fixture, out_dir=tmp_path)
    assert [row["plain"] for row in report["rows"]] == [True, False, True]
    assert seen[0]["turn_instruction"].startswith(attach.SUMMARY_INSTRUCTION) and "turn_instruction" not in seen[1]
    assert "turn_instruction" not in seen[2] and "max_tokens" not in seen[2] and seen[2]["plain"] is True


def test_a_summary_turn_shows_plain_text_and_stores_the_raw_reply(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from mentor_app import settings
    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.store import MentorChatStore
    from mentor_app.window import MentorWindow
    from mentor_packs.registry import make_pack

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    monkeypatch.setattr(settings, "context_tokens", lambda: 8192)
    sent: list = []
    rows = [{"id": "fund:2026-09-30:bottom_line", "text": "risk-on into CPI, futures +0.4%"},
            {"id": "fund:2026-09-30:playbook:1", "text": "buy pullbacks to the 5 day AVWAP"},
            {"id": "fund:2026-09-30:playbook:2", "text": "fade gaps above 5,820"},
            {"id": "fund:2026-09-30:risk:1", "text": "CPI at 08:30 ET"}]
    win = MentorWindow(store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"), queue=PrefetchQueue(),
                       stream_post=lambda url, payload, cancelled: sent.append(payload) or _answer(LONG_REPLY),
                       post=lambda url, payload, timeout: {}, pack_builder=lambda name, args: make_pack(name, rows))
    try:
        win._brain_ok, win._endpoint, win._model, win._native_tools = True, "http://x", "gemma4:12b", True
        win.send("is the playbook bullish or bearish")
        assert win._worker is not None and win._worker.wait(5000)
        deadline = time.monotonic() + 5
        while win._worker is not None and time.monotonic() < deadline:
            app.processEvents()
        win._io.submit(lambda: None).result(5)
        assert sent[0]["options"]["num_predict"] == attach.SUMMARY_MAX_TOKENS
        assert [m for m in sent[0]["messages"] if m["role"] == "user"][-1]["content"].endswith(
            f"{attach.SUMMARY_INSTRUCTION} {attach.BRIEF_RECIPE}")
        shown = win._blocks[-1]
        assert "**Bottom line" not in shown and "- Buy pullbacks" not in shown and "Bottom line: risk-on" in shown
        assert win.store.turns()[-1]["text"] == LONG_REPLY, "the turn log keeps the model's raw words"
    finally:
        win.shutdown()
        win.deleteLater()


def test_a_simple_journal_turn_shows_plain_text_with_no_instruction_or_cap(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from mentor_app import settings
    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.store import MentorChatStore
    from mentor_app.window import MentorWindow
    from mentor_packs.registry import make_pack

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    monkeypatch.setattr(settings, "context_tokens", lambda: 8192)
    sent: list = []
    reply = "- **Net:** green week [jrn:week:totals]\n- **Best:** the NVDA long [jrn:week:best]"
    win = MentorWindow(store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"), queue=PrefetchQueue(),
                       stream_post=lambda url, payload, cancelled: sent.append(payload) or _answer(reply),
                       post=lambda url, payload, timeout: {}, pack_builder=lambda name, args: make_pack(name, []))
    try:
        win._brain_ok, win._endpoint, win._model, win._native_tools = True, "http://x", "gemma4:12b", True
        win.send("how did I do last week")
        assert win._worker is not None and win._worker.wait(5000)
        deadline = time.monotonic() + 5
        while win._worker is not None and time.monotonic() < deadline:
            app.processEvents()
        win._io.submit(lambda: None).result(5)
        assert "num_predict" not in sent[0]["options"]
        assert [m for m in sent[0]["messages"] if m["role"] == "user"][-1]["content"] == "how did I do last week"
        shown = win._blocks[-1]
        assert "Net: green week [jrn:week:totals]" in shown and "**Net" not in shown and "- " not in shown
        assert win.store.turns()[-1]["text"] == reply
    finally:
        win.shutdown()
        win.deleteLater()
