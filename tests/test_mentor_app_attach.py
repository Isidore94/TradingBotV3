"""P13 auto-attach: plain questions pick their packs; the brain injects them as tool results, deduped and capped."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import attach, brain, checklist  # noqa: E402
from mentor_app.chat_model import ChatModel  # noqa: E402
from mentor_packs.registry import make_pack  # noqa: E402

NOW = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)  # Wed 11:00 ET
KNOWN = {"ALL": "SHORT", "NVDA": "LONG", "AMD": "SHORT", "V": "LONG", "TSLA": "SHORT", "MSFT": ""}
TOOLS = [{"type": "function", "function": {"name": "context_pack", "description": "desk", "parameters": {}}}]


def _names(requests):
    return [(r.name, r.args) for r in requests]


def test_a_known_uppercase_ticker_attaches_its_pick_and_news():
    got = _names(attach.plan_attachments("is NVDA still worth it with AMD reporting", KNOWN, NOW))
    assert ("pick_pack", {"symbol": "NVDA"}) in got and ("pick_pack", {"symbol": "AMD"}) in got
    assert ("news_pack", {"symbol": "NVDA"}) in got and ("news_pack", {"symbol": "AMD"}) in got


def test_all_v_and_it_are_tickers_only_inside_the_traders_universe():
    assert attach.find_symbols("ALL of it went well", {}) == []
    assert attach.find_symbols("ALL of it went well", KNOWN) == ["ALL"]
    assert attach.find_symbols("IT is fine, V too", KNOWN) == ["V"]
    assert attach.find_symbols("all good, nvda?", KNOWN) == [], "plain tickers are case-sensitive"
    assert attach.find_symbols("what about $pltr and $nvda", {}) == ["PLTR", "NVDA"], "$SYM is any case, anywhere"
    assert attach.find_symbols("I think A is fine", {"I": "", "A": ""}) == [], "desk words are never tickers"


def test_at_most_three_symbols():
    got = attach.plan_attachments("NVDA AMD TSLA MSFT ALL", KNOWN, NOW)
    assert len({r.args["symbol"] for r in got if r.name == "news_pack"}) == 3


def test_time_words_attach_the_journal_and_the_tape():
    assert _names(attach.plan_attachments("how did today go", KNOWN, NOW)) == [
        ("journal_pack", {"day": "2026-09-30"}), ("regime_pack", {})]
    assert ("journal_pack", {"day": "2026-09-29"}) in _names(attach.plan_attachments("what about yesterday", KNOWN, NOW))
    assert ("journal_pack", {"day": "week"}) in _names(attach.plan_attachments("how is this week", KNOWN, NOW))
    assert ("journal_pack", {"day": "last_week"}) in _names(attach.plan_attachments("and last week?", KNOWN, NOW))
    assert ("journal_pack", {"day": "2026-09-29"}) in _names(
        attach.plan_attachments("why did I lose money tuesday", KNOWN, NOW))
    monday = datetime(2026, 10, 5, 15, 0, tzinfo=timezone.utc)
    assert attach.resolve_day("yesterday", monday) == "2026-10-02", "Monday's yesterday is Friday's session"


def test_what_trades_did_i_take_today_is_the_journal_not_a_gate():
    got = _names(attach.plan_attachments("what trades did I take today", KNOWN, NOW))
    assert got[0] == ("journal_pack", {"day": "2026-09-30"})
    assert not any(name == "gate_pack" for name, _ in got)


def test_thinking_of_shorting_a_known_name_attaches_the_gate_with_the_side():
    got = attach.plan_attachments("im thinking of shorting ALL thoughts?", KNOWN, NOW)
    assert got[0].name == "gate_pack" and got[0].args == {"side": "SHORT", "symbol": "ALL"}
    assert not any(r.name == "pick_pack" and r.args["symbol"] == "ALL" for r in got), "the gate carries the pick"
    long_ = attach.plan_attachments("should I take NVDA here", KNOWN, NOW)[0]
    assert long_.args == {"side": "LONG", "symbol": "NVDA"}, "no side word: the Focus side"
    assert attach.plan_attachments("thinking of taking a short like ALL, thoughts?", KNOWN, NOW)[0].args["side"] == "SHORT"


def test_intent_on_a_name_with_no_side_known_attaches_the_pick_only():
    got = _names(attach.plan_attachments("should I take MSFT", KNOWN, NOW))
    assert ("pick_pack", {"symbol": "MSFT"}) in got and not any(n == "gate_pack" for n, _ in got)


def test_veto_tape_book_and_group_words():
    assert ("veto_pack", {"date": "2026-09-29"}) in _names(
        attach.plan_attachments("what did I veto yesterday and was I right", KNOWN, NOW))
    assert _names(attach.plan_attachments("what's the tape doing", KNOWN, NOW)) == [("regime_pack", {})]
    assert ("regime_pack", {}) in _names(attach.plan_attachments("what did SPY do", KNOWN, NOW))
    assert ("book_pack", {}) in _names(attach.plan_attachments("what am I holding right now", KNOWN, NOW))
    longs = [r.args["symbol"] for r in attach.plan_attachments("anything reporting this week in my longs?", KNOWN, NOW)
             if r.name == "pick_pack"]
    assert longs == ["NVDA", "V"]


def test_the_plan_is_deterministic_and_sorted_by_priority():
    one = attach.plan_attachments("thinking of shorting ALL today, what's the tape?", KNOWN, NOW)
    two = attach.plan_attachments("thinking of shorting ALL today, what's the tape?", KNOWN, NOW)
    assert one == two
    assert [r.priority for r in one] == sorted(r.priority for r in one)


def test_known_symbols_come_from_focus_positions_likes_and_the_journal():
    rows = [
        {"kind": "focus", "category": "swing", "side": "long", "names": ["NVDA"]},
        {"kind": "focus", "category": "m5", "side": "short", "names": ["ALL"]},
        {"kind": "position", "symbol": "MSFT", "direction": "LONG"},
    ]
    known = attach.known_symbols(rows, [("TSLA", "SHORT")], ["AMD"])
    assert known == {"NVDA": "LONG", "ALL": "SHORT", "MSFT": "LONG", "TSLA": "SHORT", "AMD": ""}


# ---------------------------------------------------------------- brain injection
def _lines(*chunks):
    return [json.dumps(chunk).encode() for chunk in chunks]


def _answer(text):
    return _lines({"message": {"content": text}, "done": False},
                  {"message": {"content": ""}, "done": True, "prompt_eval_count": 900, "eval_count": 30})


def _pack(name, args):
    sym = str(args.get("symbol") or "X")
    return make_pack(name, [{"id": f"{name}:{sym}:a", "text": f"{sym} row a"}, {"id": f"{name}:{sym}:b", "text": "b"}])


def _messages(question):
    chat = ChatModel()
    chat.add("user", question)
    return chat.messages(context_text="[ctx:auto_mode] Auto mode: DESK")


def test_native_models_get_the_attachments_as_tool_results_after_the_question():
    sent: list[dict] = []
    requests = attach.plan_attachments("im thinking of shorting ALL thoughts?", KNOWN, NOW)
    result = brain.run_turn(_messages("im thinking of shorting ALL thoughts?"), model="gemma4:12b", endpoint="http://x",
                            tools=TOOLS, native_tools=True, build_pack=_pack, attachments=requests,
                            stream_post=lambda u, p, c: sent.append(p) or _answer("ok"))
    roles = [m["role"] for m in sent[0]["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "tool"]
    call_names = [c["function"]["name"] for c in sent[0]["messages"][2]["tool_calls"]]
    assert call_names == ["gate_pack", "news_pack"]
    assert "[gate_pack:ALL:a]" in sent[0]["messages"][3]["content"], "ids intact"
    assert sent[0]["tools"] == TOOLS, "the model can still call more"
    assert [a["name"] for a in result["attached"]] == ["gate_pack", "news_pack"]
    assert all(a["source"] == "auto" for a in result["attached"]) and result["tool_calls"] == []
    assert "gate_pack:ALL:a" in result["pack_ids"]


def test_the_system_prefix_is_the_same_bytes_with_or_without_attachments():
    sent: list[dict] = []
    for requests in ((), attach.plan_attachments("how did today go", KNOWN, NOW)):
        brain.run_turn(_messages("how did today go"), model="gemma4:12b", endpoint="http://x", tools=TOOLS,
                       native_tools=True, build_pack=_pack, attachments=requests,
                       stream_post=lambda u, p, c: sent.append(p) or _answer("ok"))
    assert sent[0]["messages"][0] == sent[1]["messages"][0]
    assert sent[0]["messages"][1] == sent[1]["messages"][1]


def test_the_fallback_gets_a_packs_block_before_the_question_and_no_choice_call():
    posted: list = []
    sent: list[dict] = []
    brain.run_turn(_messages("how did today go"), model="gemma3:12b", endpoint="http://x", tools=TOOLS,
                   native_tools=False, build_pack=_pack, post=lambda u, p, t: posted.append(p) or {},
                   attachments=attach.plan_attachments("how did today go", KNOWN, NOW),
                   stream_post=lambda u, p, c: sent.append(p) or _answer("ok"))
    assert posted == [], "the app already chose: one model call per turn"
    roles = [m["role"] for m in sent[0]["messages"]]
    assert roles == ["system", "system", "user"] and "attached by the app" in sent[0]["messages"][1]["content"]


def test_the_budget_drops_the_lowest_priority_pack_first():
    def big(name, args):
        return make_pack(name, [{"id": f"{name}:{i}", "text": "x" * 400} for i in range(10)])  # ~1000 tokens

    requests = [attach.AttachRequest("news_pack", {"symbol": "A"}, 6), attach.AttachRequest("gate_pack", {}, 0),
                attach.AttachRequest("regime_pack", {}, 4)]
    result = brain.run_turn(_messages("x"), model="m", endpoint="http://x", native_tools=True, tools=TOOLS,
                            build_pack=big, attachments=requests, attach_budget_tokens=2200,
                            stream_post=lambda u, p, c: _answer("ok"))
    kept = {a["name"]: a["dropped"] for a in result["attached"]}
    assert kept == {"gate_pack": False, "regime_pack": False, "news_pack": True}


def test_rows_cited_in_the_last_turns_are_not_reattached_unless_the_question_names_them():
    sent: list[dict] = []

    def tape(name, args):
        return make_pack(name, [{"id": "tape:spy:pause", "text": "Fifteen long names held up through a SPY pause"},
                                {"id": "tape:mode", "text": "Auto mode: DESK"}])

    turns = [{"role": "assistant", "text": "Longs held up [tape:spy:pause]."}]
    seen = attach.recent_cited_ids(turns)
    assert seen == {"tape:spy:pause"}
    request = [attach.AttachRequest("regime_pack", {}, 4)]
    for question in ("what's the tape doing", "and what did SPY do"):
        brain.run_turn(_messages(question), model="m", endpoint="http://x", native_tools=True, tools=TOOLS,
                       build_pack=tape, attachments=request, seen_ids=seen, question=question,
                       stream_post=lambda u, p, c: sent.append(p) or _answer("ok"))
    first, second = (payload["messages"][3]["content"] for payload in sent)
    assert "Fifteen long names" not in first and "1 row(s) you cited" in first and "[tape:mode]" in first
    assert "Fifteen long names" in second, "the question names SPY: the row comes back"


def test_recent_cited_ids_reads_only_the_last_six_turns():
    turns = [{"role": "assistant", "text": "[old:row:1]"}] + [{"role": "user", "text": "q"}] * 6
    assert attach.recent_cited_ids(turns) == set()


# ---------------------------------------------------------------- checklist
def _gate():
    rows = [
        ("gate:ALL:req", "Request: SHORT ALL"),
        ("gate:ALL:pick:ALL:earn", "Own earnings: Thu 2026-10-29, in 29 days"),
        ("gate:ALL:pick:ALL:peer:TRV", "Peer TRV earnings tomorrow"),
        ("gate:ALL:pick:ALL:cell", "Setup cell top_pattern SHORT: too few, n=12"),
        ("gate:ALL:pick:ALL:news:1", "Allstate cat losses"),
        ("gate:ALL:tape:d1env", "D1 environment: bearish"),
        ("gate:ALL:book:industry", "Open book: 1 open trade; 0 in Insurance"),
        ("gate:ALL:plan:risk:1", "Plan [plan:risk:1]: Max 3 open shorts"),
    ]
    return make_pack("gate_pack", [{"id": i, "text": t} for i, t in rows])


def test_a_reply_that_skips_two_sections_gets_them_appended_from_the_pack():
    reply = ("Earnings are far [gate:ALL:pick:ALL:earn], the cell is thin [gate:ALL:pick:ALL:cell], D1 is bearish "
             "[gate:ALL:tape:d1env] and your plan allows it [gate:ALL:plan:risk:1].")
    assert checklist.missing(reply) == ["book", "news"]
    extra = checklist.appendix(reply, _gate())
    assert extra.startswith("**Not covered:** book exposure, news")
    assert "[gate:ALL:book:industry]" in extra and "[gate:ALL:pick:ALL:news:1]" in extra
    assert "[gate:ALL:pick:ALL:earn]" not in extra, "covered sections are not repeated"


def test_a_full_reply_needs_no_appendix_and_the_brain_attaches_it_for_a_gate_turn():
    full = ("[gate:ALL:pick:ALL:earn] [gate:ALL:plan:risk:1] [gate:ALL:tape:d1env] [gate:ALL:pick:ALL:cell] "
            "[gate:ALL:book:industry] [gate:ALL:pick:ALL:news:1]")
    assert checklist.appendix(full, _gate()) == ""
    result = brain.run_turn(_messages("thinking of shorting ALL"), model="m", endpoint="http://x", native_tools=True,
                            tools=TOOLS, build_pack=lambda n, a: _gate() if n == "gate_pack" else _pack(n, a),
                            attachments=[attach.AttachRequest("gate_pack", {"side": "SHORT", "symbol": "ALL"}, 0)],
                            stream_post=lambda u, p, c: _answer("Looks fine [gate:ALL:tape:d1env]."))
    assert result["appendix"].startswith("**Not covered:** earnings (own + peers), plan lines, setup cell / cohort")
    assert result["timings"]["auto_packs"] == 1 and result["timings"]["attach_ms"] >= 0
