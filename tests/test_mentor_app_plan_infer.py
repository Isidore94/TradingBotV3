"""Trade Mentor app: plan lines inferred live from the trader's own chat words (trader 2026-09-30).

The citation check, the trader's lines untouchable, the 5-a-day cap, dedupe, /plan and /drop,
`/remember rule:` flowing to the plan, a model-down skip, a failed write surfaced, and the side
call never on the Qt thread. Plans live under tmp_path; no network.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import trading_plan  # noqa: E402
from mentor_app import commands, plan_infer  # noqa: E402
from mentor_app.chat_model import SYSTEM_PROMPT  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
NOW = datetime(2026, 9, 30, 8, 0, tzinfo=PT)  # a Wednesday session
DAY = "2026-09-30"
PLAN = (
    "## Rules\n\n- Respect the stop.\n- No trades before 06:45. [ai 2026-09-29]\n\n"
    "## Risk\n\n## What I am testing\n\n## Decisions\n"
)


def _turns(*texts, first_id=1, role="user"):
    return [{"id": first_id + i, "role": role, "text": text} for i, text in enumerate(texts)]


def _inputs(after=0):
    turns = _turns("I never trade before 06:45.", "Stop after two losses, always.")
    turns.append({"id": 3, "role": "assistant", "text": "You could also size down after a loss."})
    return plan_infer.build_inputs(turns, trading_plan.parse_plan(PLAN), after=after)


def _add(text="Stop after two losses.", **extra):
    return {"op": "add", "section": "Rules", "text": text, "turn_ids": ["turn:2"], **extra}


# ---------------------------------------------------------------- inputs and the citation check
def test_only_the_traders_turns_are_evidence():
    inputs = _inputs()
    assert [row["id"] for row in inputs["trader_turns"]] == ["turn:1", "turn:2"], "the Mentor's reply is not evidence"
    assert {row["id"]: row["ai"] for row in inputs["plan_lines"]} == {"plan:rules:1": False, "plan:rules:2": True}
    assert inputs["allowed_sections"] == ["Goals", "Rules", "Setups I trade", "Risk", "What I am testing"]
    with pytest.raises(plan_infer.InferRejected):
        plan_infer.check_reply({"ops": [{**_add(), "turn_ids": ["turn:3"]}]}, inputs)


def test_a_foreign_turn_or_plan_id_rejects_the_whole_reply():
    inputs = _inputs()
    with pytest.raises(plan_infer.InferRejected):
        plan_infer.check_reply({"ops": [_add(), {**_add("x"), "turn_ids": ["turn:99"]}]}, inputs)
    with pytest.raises(plan_infer.InferRejected):
        plan_infer.check_reply({"ops": [_add(), {"op": "retire", "plan_id": "plan:rules:9", "turn_ids": ["turn:2"]}]},
                               inputs)
    with pytest.raises(plan_infer.InferRejected):
        plan_infer.check_reply({"nope": []}, inputs)


def test_bad_ops_are_dropped_one_at_a_time():
    inputs = _inputs(after=1)  # only turn:2 is new
    reply = {"ops": [
        {"op": "add", "section": "Rules", "text": "no turn", "turn_ids": []},
        {"op": "add", "section": "Rules", "text": "old turn only", "turn_ids": ["turn:1"]},
        {"op": "add", "section": "Decisions", "text": "bad section", "turn_ids": ["turn:2"]},
        {"op": "add", "section": "Rules", "text": "", "turn_ids": ["turn:2"]},
        {"op": "add", "section": "Rules", "text": "x" * 141, "turn_ids": ["turn:2"]},
        {"op": "update", "plan_id": "plan:rules:1", "text": "Respect it", "turn_ids": ["turn:2"]},
        {"op": "retire", "plan_id": "plan:rules:1", "turn_ids": ["turn:2"]},
        {"op": "delete", "plan_id": "plan:rules:2", "turn_ids": ["turn:2"]},
        "not an object",
        {"op": "add", "section": "risk", "text": "Stop after two losses.", "turn_ids": [2]},
    ]}
    ops, drops = plan_infer.check_reply(reply, inputs)
    assert drops == {"no_turn": 1, "no_new_turn": 1, "bad_section": 1, "bad_text": 2, "not_ai_line": 2,
                     "bad_op": 1, "not_an_object": 1}
    assert [(op["op"], op["section"], op["text"], op["turn_ids"]) for op in ops] == [
        ("add", "Risk", "Stop after two losses.", ["turn:2"])]
    assert ops[0]["quote"] == "Stop after two losses, always."


def test_at_most_three_ops_per_reply():
    ops, drops = plan_infer.check_reply({"ops": [_add(f"rule {n}") for n in range(5)]}, _inputs())
    assert len(ops) == 3 and drops == {"too_many": 2}


def test_the_prompt_says_questions_hypotheticals_and_mentor_ideas_are_not_rules():
    text = plan_infer.INSTRUCTIONS.lower()
    assert "a question, a hypothetical" in text and "the mentor's own suggestion is not a rule" in text
    assert "say nothing when unsure" in text
    assert "[ai yyyy-mm-dd]" in SYSTEM_PROMPT.lower() and "inferred" in SYSTEM_PROMPT


# ---------------------------------------------------------------- writing
@pytest.fixture()
def world(tmp_path):
    plan = tmp_path / "trading_plan.md"
    plan.write_text(PLAN, encoding="utf-8")
    return {"plan": plan, "store": MentorChatStore(tmp_path / "chat.sqlite3")}


def _checked(*ops):
    return plan_infer.check_reply({"ops": list(ops)}, _inputs())[0]


def test_an_add_writes_an_ai_line_and_one_chat_line(world):
    shown = plan_infer.apply(_checked(_add()), store=world["store"], day=DAY, now=NOW, path=world["plan"])
    assert shown == ["Added to your plan: Stop after two losses. [plan:rules:3] — /drop plan:rules:3 to undo"]
    assert "- Stop after two losses. [ai 2026-09-30]\n" in world["plan"].read_text(encoding="utf-8")


def test_a_clarification_updates_the_ai_line(world):
    update = {"op": "update", "plan_id": "plan:rules:2", "text": "No trades before 07:00.", "turn_ids": ["turn:2"]}
    shown = plan_infer.apply(_checked(update), store=world["store"], day=DAY, now=NOW, path=world["plan"])
    assert shown == ["Updated in your plan: No trades before 07:00. [plan:rules:2] — /drop plan:rules:2 to undo"]
    assert trading_plan.parse_plan(world["plan"].read_text(encoding="utf-8"))["sections"]["Rules"] == [
        "Respect the stop.", "No trades before 07:00. [ai 2026-09-30]"]


def test_a_duplicate_add_is_dropped_silently(world):
    before = world["plan"].read_text(encoding="utf-8")
    shown = plan_infer.apply(_checked(_add("respect the stop")), store=world["store"], day=DAY, now=NOW,
                             path=world["plan"])
    assert shown == [] and world["plan"].read_text(encoding="utf-8") == before
    assert world["store"].get_state(plan_infer.WRITES_KEY.format(day=DAY)) is None, "a dropped add costs nothing"


def test_at_most_five_writes_a_day_then_one_line_says_so(world):
    shown = []
    for batch in range(3):
        ops = _checked(*(_add(f"rule {batch}-{n}") for n in range(3)))
        shown += plan_infer.apply(ops, store=world["store"], day=DAY, now=NOW, path=world["plan"])
    added = [line for line in shown if line.startswith("Added")]
    assert len(added) == 5
    assert shown.count(plan_infer.CAPPED.format(cap=5)) == 1
    rules = trading_plan.parse_plan(world["plan"].read_text(encoding="utf-8"))["sections"]["Rules"]
    assert len(rules) == 7
    # a retire is not capped, and a new PT day starts a new count
    retire = {"op": "retire", "plan_id": "plan:rules:2", "turn_ids": ["turn:2"]}
    assert plan_infer.apply(_checked(retire), store=world["store"], day=DAY, now=NOW, path=world["plan"]) == [
        "Dropped from your plan: No trades before 06:45. (a Decisions line keeps it)"]
    assert plan_infer.apply(_checked(_add("next day rule")), store=world["store"], day="2026-10-01", now=NOW,
                            path=world["plan"])[0].startswith("Added")


def test_a_failed_plan_write_is_one_loud_line_and_writes_nothing(world, monkeypatch):
    before = world["plan"].read_text(encoding="utf-8")

    def boom(*_a, **_k):
        raise OSError("read-only")

    monkeypatch.setattr(trading_plan.os, "replace", boom)
    shown = plan_infer.apply(_checked(_add(), _add("second")), store=world["store"], day=DAY, now=NOW,
                             path=world["plan"])
    assert len(shown) == 1 and shown[0].startswith("Not saved to your plan: Stop after two losses.")
    assert world["plan"].read_text(encoding="utf-8") == before
    assert world["store"].get_state(plan_infer.WRITES_KEY.format(day=DAY)) is None


def test_the_traders_own_line_is_never_touched_even_if_a_check_is_bypassed(world):
    before = world["plan"].read_text(encoding="utf-8")
    forged = [{"op": "retire", "plan_id": "plan:rules:1", "expect_text": None, "text": "", "turn_ids": ["turn:2"]},
              {"op": "update", "plan_id": "plan:rules:1", "expect_text": None, "text": "x", "turn_ids": ["turn:2"]}]
    assert plan_infer.apply(forged, store=world["store"], day=DAY, now=NOW, path=world["plan"]) == []
    assert world["plan"].read_text(encoding="utf-8") == before


def test_drop_refuses_the_traders_line_and_retires_an_ai_line(world):
    store = world["store"]
    plan_infer.listing(store=store, path=world["plan"])  # /plan shows the ids first
    assert plan_infer.drop("plan:rules:1", store=store, day=DAY, now=NOW, path=world["plan"]) == (
        "plan:rules:1 is your own line, so I will not drop it. Edit trading_plan.md to change it.")
    assert "no AI line" in plan_infer.drop("plan:rules:9", store=store, day=DAY, now=NOW, path=world["plan"])
    assert plan_infer.drop("plan:rules:2", store=store, day=DAY, now=NOW,
                           path=world["plan"]).startswith("Dropped from your plan")
    parsed = trading_plan.parse_plan(world["plan"].read_text(encoding="utf-8"))
    assert parsed["decisions"][-1] == {"day": DAY, "text": "dropped AI rule: No trades before 06:45.", "dated": True}


def test_a_stale_drop_id_never_drops_a_different_ai_line(world):
    store = world["store"]
    for rule in ("First AI rule", "Second AI rule"):
        plan_infer.apply(_checked(_add(rule)), store=store, day=DAY, now=NOW, path=world["plan"])
    # Added lines showed plan:rules:3 = First, plan:rules:4 = Second; drop First, so Second becomes plan:rules:3
    assert plan_infer.drop("plan:rules:3", store=store, day=DAY, now=NOW,
                           path=world["plan"]).startswith("Dropped from your plan: First AI rule")
    before = world["plan"].read_text(encoding="utf-8")
    stale = plan_infer.drop("plan:rules:3", store=store, day=DAY, now=NOW, path=world["plan"])
    assert stale == plan_infer.MOVED.format(plan_id="plan:rules:3")
    assert world["plan"].read_text(encoding="utf-8") == before, "Second AI rule is still there"
    assert plan_infer.drop("plan:rules:4", store=store, day=DAY, now=NOW, path=world["plan"]).startswith(
        "There is no AI line")
    # after /plan shows the new ids, the drop lands on the rule it showed
    plan_infer.listing(store=store, path=world["plan"])
    assert plan_infer.drop("plan:rules:3", store=store, day=DAY, now=NOW,
                           path=world["plan"]).startswith("Dropped from your plan: Second AI rule")


def test_remember_rule_after_the_cap_says_it_is_a_note_only(world):
    store = world["store"]
    plan_infer.apply(_checked(*(_add(f"rule {n}") for n in range(3))), store=store, day=DAY, now=NOW, path=world["plan"])
    plan_infer.apply(_checked(*(_add(f"rule {n}") for n in range(3, 5))), store=store, day=DAY, now=NOW,
                     path=world["plan"])
    before = world["plan"].read_text(encoding="utf-8")
    shown = plan_infer.remember_rule("rule: No new entries after 12:30", store=store, day=DAY, now=NOW,
                                     path=world["plan"])
    assert shown == [plan_infer.REMEMBER_CAPPED.format(cap=plan_infer.MAX_WRITES_PER_DAY)]
    assert "kept as a note" in shown[0].lower() and "not added to your plan today" in shown[0]
    assert world["plan"].read_text(encoding="utf-8") == before


def test_remember_rule_flows_to_the_plan_with_the_same_dedupe(world):
    store = world["store"]
    assert plan_infer.remember_rule("I like NVDA", store=store, day=DAY, now=NOW, path=world["plan"]) == []
    shown = plan_infer.remember_rule("rule: Stop after two losses", store=store, day=DAY, now=NOW, path=world["plan"])
    assert shown == ["Added to your plan: Stop after two losses [plan:rules:3] — /drop plan:rules:3 to undo"]
    again = plan_infer.remember_rule("Rule: stop after two losses.", store=store, day=DAY, now=NOW, path=world["plan"])
    assert again == ["Already in your plan: [plan:rules:3]."]
    listing = plan_infer.listing(store=store, path=world["plan"])
    assert '[plan:rules:3] Stop after two losses [ai 2026-09-30] *(AI, from: "rule: Stop after two losses")*' in listing
    assert "[plan:rules:1] Respect the stop." in listing and "[plan:rules:2] No trades before 06:45. [ai 2026-09-29] *(AI)*" in listing


def test_model_down_skips_and_writes_nothing(world):
    def down(url, payload, timeout):
        raise ConnectionError("tunnel down")

    ops = plan_infer.infer(_turns("I never trade before 06:45."), after=0, endpoint="http://x", model="m",
                           post=down, path=world["plan"])
    assert ops == []
    garbled = plan_infer.infer(_turns("x"), after=0, endpoint="http://x", model="m",
                               post=lambda url, payload, timeout: {"message": {"content": "not json"}},
                               path=world["plan"])
    assert garbled == []


def test_the_side_call_is_one_structured_call_to_the_chat_model(world):
    calls = []

    def post(url, payload, timeout):
        calls.append((url, payload))
        return {"message": {"content": json.dumps({"ops": [_add(turn_ids=["turn:1"])]})}}

    ops = plan_infer.infer(_turns("Stop after two losses."), after=0, endpoint="http://h:11436/", model="qwen3:14b",
                           post=post, path=world["plan"])
    assert [op["text"] for op in ops] == ["Stop after two losses."]
    url, payload = calls[0]
    assert url == "http://h:11436/api/chat" and payload["stream"] is False and payload["format"] == plan_infer.SCHEMA
    assert payload["options"]["num_predict"] == plan_infer.MAX_OUTPUT_TOKENS
    # nothing new since the last read: no call at all
    assert plan_infer.infer(_turns("x"), after=1, endpoint="http://h", model="m", post=post, path=world["plan"]) == []
    assert len(calls) == 1


# ---------------------------------------------------------------- commands
def test_plan_and_drop_parse():
    assert commands.handle("/plan").action == "plan"
    assert commands.handle("/drop plan:rules:3") == commands.CommandResult("drop", "", "plan:rules:3")
    assert commands.handle("/drop [plan:Setups_I_trade:2]").arg == "plan:setups_i_trade:2"
    for bad in ("/drop", "/drop 3", "/drop rules:3"):
        assert commands.handle(bad).action == "error"
    assert "/plan" in commands.HELP_TEXT and "/drop" in commands.HELP_TEXT


# ---------------------------------------------------------------- the window
def _answer(text):
    return [
        json.dumps({"message": {"content": text}, "done": False}).encode(),
        json.dumps({"message": {"content": ""}, "done": True}).encode(),
    ]


@pytest.fixture()
def window(tmp_path, monkeypatch):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from mentor_app import settings
    from mentor_app.inbox import Inbox
    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.window import MentorWindow

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    monkeypatch.setattr(settings, "context_tokens", lambda: 8192)
    plan = tmp_path / "trading_plan.md"
    plan.write_text(PLAN, encoding="utf-8")
    calls: list = []
    reply = {"ops": [_add("Stop after two losses.", turn_ids=["turn:1"])]}

    def post(url, payload, timeout):
        if payload.get("format") == plan_infer.SCHEMA:
            calls.append(threading.current_thread())
            if reply.get("down"):
                raise ConnectionError("tunnel down")
            return {"message": {"content": json.dumps(reply)}}
        return {}

    win = MentorWindow(
        store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"),
        queue=PrefetchQueue(blocked=lambda: "", model_ready=lambda: True),
        inbox=Inbox(per_day_cap=6, now=lambda: NOW),
        stream_post=lambda url, payload, cancelled: _answer("Two losses is a good stop."),
        post=post,
        now=lambda: NOW,
        mentor_enabled=False,
        liked_source=lambda: [],
        memory_root=tmp_path / "ai",
        plan_path=plan,
    )
    win.app, win.plan, win.calls, win.reply = app, plan, calls, reply
    win._submit_io(win._open_session)
    win._io.submit(lambda: None).result(5)
    yield win
    win.shutdown()
    win.deleteLater()


def _wait(win, done, seconds=10.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        win._io.submit(lambda: None).result(5)
        win.app.processEvents()
        if done():
            return True
        time.sleep(0.02)
    return False


def _chat(win, text):
    win._brain_ok, win._endpoint, win._model = True, "http://127.0.0.1:11436", "qwen3:14b"
    win.send(text)
    assert win._worker is not None and win._worker.wait(5000)
    assert _wait(win, lambda: win._worker is None)


def test_a_stated_rule_lands_in_the_plan_off_the_qt_thread(window):
    window.queue.start()
    _chat(window, "I stop after two losses.")
    assert _wait(window, lambda: "Added to your plan" in window.transcript.toPlainText())
    assert "Stop after two losses. [plan:rules:3]" in window.transcript.toPlainText()
    assert "- Stop after two losses. [ai 2026-09-30]" in window.plan.read_text(encoding="utf-8")
    assert window.calls and all(thread is not threading.main_thread() for thread in window.calls)
    # the same turn is never read twice: a second reply with no new trader turn asks nothing
    window._queue_plan_inference()
    _wait(window, lambda: not window.queue.pending(), 2)
    assert len(window.calls) == 1


def test_model_down_in_the_window_writes_nothing_and_says_nothing(window):
    window.reply["down"] = True
    before = window.plan.read_text(encoding="utf-8")
    window.queue.start()
    _chat(window, "I stop after two losses.")
    assert _wait(window, lambda: window.calls)
    _wait(window, lambda: not window.queue.pending(), 2)
    assert window.plan.read_text(encoding="utf-8") == before
    assert "plan" not in window.transcript.toPlainText().lower()


def test_plan_drop_and_remember_rule_commands_in_the_window(window):
    window.send("/remember rule: No new entries after 12:30")
    assert _wait(window, lambda: "Added to your plan: No new entries after 12:30" in window.transcript.toPlainText())
    assert window.store.profile_notes()[0]["text"] == "rule: No new entries after 12:30", "the note is still kept"
    window.send("/drop plan:rules:2")  # not shown by /plan yet: refused, nothing dropped
    assert _wait(window, lambda: "That rule moved" in window.transcript.toPlainText())
    window.send("/plan")
    assert _wait(window, lambda: "**Your plan**" in window.transcript.toPlainText()
                 or "Your plan (lines" in window.transcript.toPlainText())
    window.send("/drop plan:rules:1")
    assert _wait(window, lambda: "is your own line" in window.transcript.toPlainText())
    window.send("/drop plan:rules:2")
    assert _wait(window, lambda: "Dropped from your plan: No trades before 06:45." in window.transcript.toPlainText())
    window.send("/plan")
    assert _wait(window, lambda: "**Your plan**" in window.transcript.toPlainText()
                 or "Your plan (lines" in window.transcript.toPlainText())
    text = window.transcript.toPlainText()
    assert "[plan:rules:2] No new entries after 12:30 [ai 2026-09-30]" in text
    assert "dropped AI rule: No trades before 06:45." in text
