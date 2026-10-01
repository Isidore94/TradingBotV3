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


def test_a_dollar_common_word_is_a_ticker_only_in_capitals_or_inside_the_universe():
    assert attach.find_symbols("what about $IT", {}) == ["IT"], "capitals: a ticker anywhere"
    assert attach.find_symbols("is $it worth it, $all of it?", {}) == [], "lowercase English words are not tickers"
    assert attach.find_symbols("is $it worth it", {"IT": "LONG"}) == ["IT"], "inside the universe it is"
    assert attach.find_symbols("$pltr and $Nvda", {}) == ["PLTR", "NVDA"], "not a common word: any case"


def test_skip_and_skipping_are_veto_words():
    for question in ("what did I skip monday", "was skipping AMD right", "I skip too many"):
        assert any(r.name == "veto_pack" for r in attach.plan_attachments(question, KNOWN, NOW)), question


def test_at_most_three_symbols():
    got = attach.plan_attachments("NVDA AMD TSLA MSFT ALL", KNOWN, NOW)
    assert len({r.args["symbol"] for r in got if r.name == "news_pack"}) == 3


def test_time_words_attach_the_journal_and_not_the_tape():
    # P14 (trader 2026-09-30, "not spamming me"): a day question is the journal alone; the tape needs a market cue.
    assert _names(attach.plan_attachments("how did today go", KNOWN, NOW)) == [("journal_pack", {"day": "2026-09-30"})]
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
    # P14: earnings over a group is one earnings_pack over every name with that side, not three pick packs.
    # P16: "my longs" is the open book only; "my focus longs" is Focus (with each name's origin).
    assert _names(attach.plan_attachments("anything reporting this week in my focus longs?", KNOWN, NOW)) == [
        ("earnings_pack", {"symbols": ["NVDA", "V"], "book": [], "liked": [], "focus": ["NVDA", "V"], "side": "LONG"})]
    assert _names(attach.plan_attachments("anything reporting this week in my longs?", KNOWN, NOW, book=["NVDA"])) == [
        ("earnings_pack", {"symbols": ["NVDA"], "book": ["NVDA"], "liked": [], "focus": [], "side": "LONG"}),
        ("book_pack", {})]
    news = [r.args["symbol"] for r in attach.plan_attachments("any news in my focus longs?", KNOWN, NOW)
            if r.name == "pick_pack"]
    assert news == ["NVDA", "V"], "news over a group is still the pick packs"


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
    # P15b: a pre-trade question also carries today's brief (bottom line + playbook).
    assert roles == ["system", "user", "assistant", "tool", "tool", "tool"]
    call_names = [c["function"]["name"] for c in sent[0]["messages"][2]["tool_calls"]]
    assert call_names == ["gate_pack", "fundamentals_pack", "news_pack"]
    assert "[gate_pack:ALL:a]" in sent[0]["messages"][3]["content"], "ids intact"
    assert sent[0]["tools"] == TOOLS, "the model can still call more"
    assert [a["name"] for a in result["attached"]] == ["gate_pack", "fundamentals_pack", "news_pack"]
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


def test_a_smaller_lower_priority_pack_never_displaces_a_higher_one():
    sizes = {"gate_pack": 2278, "journal_pack": 4027, "pick_pack": 278}  # tokens, the reviewer's repro

    def sized(name, args):
        rows = sizes[name] * 4 // 100
        return make_pack(name, [{"id": f"{name}:{i}", "text": "x" * 92} for i in range(rows)])

    requests = [attach.AttachRequest("pick_pack", {"symbol": "ALL"}, 2), attach.AttachRequest("gate_pack", {}, 0),
                attach.AttachRequest("journal_pack", {"day": "today"}, 1)]
    sent: list[dict] = []
    result = brain.run_turn(_messages("x"), model="m", endpoint="http://x", native_tools=True, tools=TOOLS,
                            build_pack=sized, attachments=requests, attach_budget_tokens=6000,
                            stream_post=lambda u, p, c: sent.append(p) or _answer("ok"))
    by_name = {a["name"]: a for a in result["attached"]}
    assert not by_name["gate_pack"]["dropped"] and not by_name["gate_pack"]["truncated"]
    assert not by_name["journal_pack"]["dropped"] and by_name["journal_pack"]["truncated"]
    assert by_name["pick_pack"]["dropped"], "the lowest priority goes first"
    assert sum(a["tokens"] for a in result["attached"] if not a["dropped"]) <= 6000
    assert [m.get("tool_name") for m in sent[0]["messages"] if m["role"] == "tool"] == ["gate_pack", "journal_pack"]


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


# ---------------------------------------------------------------- P14: no spam, the right window, the whole book
def _only(question):
    return sorted(r.name for r in attach.plan_attachments(question, KNOWN, NOW))


def test_the_tape_is_attached_only_on_a_market_or_pre_trade_cue():
    for question in ("green or red so far?", "how am i doing today", "what did I skip monday", "any headlines on MSFT today",
                     "did I do ok yesterday", "how was my week", "am I tilting", "what's my win rate this month"):
        assert "regime_pack" not in _only(question), question
    for question in ("whats the market like this morning", "what's the tape doing", "is QQQ holding up",
                     "breadth any good?", "cpi tomorrow, should I be careful", "how are futures",
                     "what should I look at tomorrow morning", "should I take MSFT"):
        assert "regime_pack" in _only(question), question
    assert attach.market_cue("thinking of shorting ALL") and not attach.market_cue("should I stop trading for today")
    assert attach.market_cue("anything new this morning") and not attach.market_cue("how did I do this morning")


def test_the_journal_needs_a_first_person_or_a_named_day():
    assert _only("whats the market like this morning") == ["regime_pack"]
    assert _only("any headlines on MSFT today") == ["news_pack", "pick_pack"]
    for question in ("green or red so far?", "how am i doing today", "how did today go", "what trades did I take today",
                     "did I do ok yesterday", "what happened with AMD friday", "how did I do last week"):
        assert "journal_pack" in _only(question), question
    assert "book_pack" not in _only("is QQQ holding up"), "'holding up' is not the book"
    assert _only("what am I holding") == ["book_pack"]


def test_this_month_is_the_journal_month_not_the_mirror():
    assert _names(attach.plan_attachments("what's my win rate this month", KNOWN, NOW)) == [
        ("journal_pack", {"day": "month"})]
    assert attach.resolve_day("how was last month", NOW) == "last_month"
    assert _only("how has my record been lately") == ["mirror_pack"]
    assert _only("when during the day do I trade best") == ["mirror_pack"]
    # P16: a month of vetoes is the aggregate by reason, with the mirror alongside.
    assert _only("which of my vetoes this month would have worked") == ["mirror_pack", "veto_pack"]


def test_stopping_for_the_day_reads_tilt_and_todays_journal():
    for question in ("should I stop trading for today", "should I call it a day", "am I overtrading",
                     "time to walk away?"):
        got = _names(attach.plan_attachments(question, KNOWN, NOW))
        assert ("tilt_pack", {}) in got and ("journal_pack", {"day": "2026-09-30"}) in got, question
        assert not any(name == "regime_pack" for name, _ in got), question


def test_earnings_across_the_book_passes_every_sided_name_and_leaves_the_cap_to_the_pack():
    known = {f"L{index:02d}": "LONG" for index in range(30)} | {f"S{index:02d}": "SHORT" for index in range(15)}
    known |= {"SPY": "LONG", "JRNL": ""}
    # P16: a book question never falls back to Focus; with no book there is no earnings read, only the book pack.
    got = attach.plan_attachments("anything reporting in my book this week?", known, NOW)
    assert [r.name for r in got] == ["book_pack"]
    got = attach.plan_attachments("anything reporting in my focus or my book this week?", known, NOW)
    earn = next(r for r in got if r.name == "earnings_pack")
    assert len(earn.args["symbols"]) == 45 and earn.args["book"] == [], "both groups: every sided name, uncapped"
    assert "SPY" not in earn.args["symbols"] and "JRNL" not in earn.args["symbols"], "index and side-less names out"
    assert not any(r.name == "journal_pack" for r in got), "an earnings question is not the journal"
    shorts = next(r for r in attach.plan_attachments("my focus shorts reporting soon?", known, NOW)
                  if r.name == "earnings_pack")
    assert shorts.args["symbols"] == [f"S{index:02d}" for index in range(15)]


def _desk(swing_longs=38, liked=18, m5=70):
    """The reviewer's desk: 38 swing longs + 18 liked + 70 M5 Focus names, and two open longs QCOM and FCEL."""
    rows = [{"kind": "focus", "category": "swing", "side": "long", "names": [f"W{i:02d}" for i in range(swing_longs)]},
            {"kind": "focus", "category": "m5", "side": "long", "names": [f"M{i:02d}" for i in range(m5)]},
            {"kind": "position", "symbol": "QCOM", "direction": "LONG"},
            {"kind": "position", "symbol": "FCEL", "direction": "LONG"}]
    likes = [(f"K{i:02d}", "LONG") for i in range(liked)]
    return rows, likes, attach.known_symbols(rows, likes, ["JRNL"])


def test_the_open_book_comes_first_and_is_never_capped_off():
    """Review of bba9077c: the 40-name cap cut in Focus order, so the open book fell off after 38 swing longs."""
    from mentor_packs import earnings_pack, pick_pack

    rows, likes, known = _desk()
    assert list(known)[:4] == ["QCOM", "FCEL", "K00", "K01"], "book, then likes, then Focus"
    assert attach.book_symbols(rows) == ["QCOM", "FCEL"]
    earn = next(r for r in attach.plan_attachments("anything reporting this week in my longs or focus longs?", known, NOW,
                                                   book=attach.book_symbols(rows), liked=likes)
                if r.name == "earnings_pack")
    assert earn.args["symbols"][:3] == ["QCOM", "FCEL", "K00"] and len(earn.args["symbols"]) == 2 + 18 + 38 + 70
    assert earn.args["book"] == ["QCOM", "FCEL"]
    pack = earnings_pack.build(**earn.args, now=pick_pack.FIXTURE_NOW, paths=pick_pack.write_fixture_world(
        __import__("tempfile").mkdtemp()))
    ids = pack.ids
    assert "earn:QCOM" in ids and "earn:FCEL" in ids and "earn:more" in ids
    more = next(row for row in pack.rows if row["id"] == "earn:more")
    assert more["text"].startswith("88 more not listed") and "M69" in more["text"]


def test_a_book_question_covers_the_whole_book_even_past_forty():
    from mentor_packs import earnings_pack, pick_pack

    rows = [{"kind": "position", "symbol": f"B{i:02d}", "direction": "LONG"} for i in range(45)]
    rows.append({"kind": "focus", "category": "swing", "side": "long", "names": ["W00", "W01"]})
    known = attach.known_symbols(rows, [], [])
    earn = next(r for r in attach.plan_attachments("any earnings in my book?", known, NOW,
                                                   book=attach.book_symbols(rows)) if r.name == "earnings_pack")
    assert earn.args["symbols"] == [f"B{i:02d}" for i in range(45)], "a book question is the book, not the Focus"
    pack = earnings_pack.build(earn.args["symbols"] + ["W00"], book=earn.args["book"], now=pick_pack.FIXTURE_NOW,
                               paths=pick_pack.write_fixture_world(__import__("tempfile").mkdtemp()))
    assert all(f"earn:B{i:02d}" in pack.ids for i in range(45)) and "earn:W00" not in pack.ids
    assert "1 more not listed" in next(row for row in pack.rows if row["id"] == "earn:more")["text"]


def test_earnings_with_no_group_word_reads_the_book_and_the_likes_not_the_journal():
    rows, likes, known = _desk(swing_longs=3, liked=2, m5=3)
    for question in ("earnings this week?", "anything reporting tomorrow?", "who reports this week"):
        got = attach.plan_attachments(question, known, NOW, book=attach.book_symbols(rows), liked=likes)
        names = [r.name for r in got]
        assert names == ["earnings_pack"], (question, names)
        assert got[0].args["symbols"] == ["QCOM", "FCEL", "K00", "K01"], question
    assert "earnings_pack" not in _only("what does my plan say about shorting into earnings")


# ---------------------------------------------------------------- P15a: the night's reads
def test_night_words_attach_the_night_pack():
    for question in ("what did the night say?", "anything overnight I should know", "what did you find last night",
                     "any ideas for me", "what am I missing", "what did I get wrong yesterday",
                     "what did I get right", "show me the week review", "give me the night read"):
        names = [request.name for request in attach.plan_attachments(question, KNOWN, NOW)]
        assert "night_pack" in names, question


def test_a_ticker_question_carries_the_brief_through_the_pick_pack_not_the_night_pack():
    names = [request.name for request in attach.plan_attachments("how does NVDA look", KNOWN, NOW)]
    assert "pick_pack" in names and "night_pack" not in names


# ---------------------------------------------------------------- P15b fundamentals
def test_fundamentals_words_attach_the_brief():
    for question in ("what's the macro brief say today", "what did claude flag as the catalyst",
                     "is the playbook bullish or bearish", "anything about yields in the brief",
                     "what did the paste say about oil", "fed speakers today?", "how is the dollar",
                     "cpi or nfp this week?", "pce came in soft, what's the bottom line", "fomc scenario"):
        got = _names(attach.plan_attachments(question, KNOWN, NOW))
        assert ("fundamentals_pack", {"section": "all"}) in got, question


def test_a_pre_trade_question_carries_the_compact_brief_and_a_plain_one_does_not():
    got = _names(attach.plan_attachments("thinking of taking a short like ALL, thoughts?", KNOWN, NOW))
    assert got[0][0] == "gate_pack" and ("fundamentals_pack", {"section": "compact"}) in got
    for question in ("how did today go", "what's the tape doing", "what am I holding", "any news on TSLA"):
        assert "fundamentals_pack" not in _only(question), question


# ---------------------------------------------------------------- P18 review: pre-trade intents by verb
def _gate_sides(text, known=KNOWN, book=()):
    return [(r.args["side"], r.args["symbol"]) for r in attach.plan_attachments(text, known, NOW, book=book)
            if r.name == "gate_pack"]


def test_pre_trade_verbs_route_to_the_gate_with_the_side_from_the_verb():
    # MSFT has no known side: the verb alone sets it.
    assert _gate_sides("I'm about to buy MSFT") == [("LONG", "MSFT")]
    assert _gate_sides("about to sell MSFT") == [("SHORT", "MSFT")]
    assert _gate_sides("going long MSFT here") == [("LONG", "MSFT")]
    assert _gate_sides("going short MSFT here") == [("SHORT", "MSFT")]
    # No side in the verb: the known side; an add takes the held side; an add on a name not held is nothing
    # (round 5 rule, the table in test_mentor_app_intent.py).
    assert _gate_sides("entering TSLA") == [("SHORT", "TSLA")]
    assert _gate_sides("adding to AMD", book=["AMD"]) == [("SHORT", "AMD")]
    assert _gate_sides("adding to NVDA") == []


def test_a_sell_off_is_not_a_sell():
    assert _gate_sides("NVDA sell-off today, what happened") == []


#: P18 re-review: the verb is read against the book first.
HELD = {"AMD": "LONG", "NVDA": "LONG", "TSLA": "SHORT"}


def _gate_args(text, book):
    return [dict(r.args) for r in attach.plan_attachments(text, HELD, NOW, book=book) if r.name == "gate_pack"]


def test_selling_a_held_long_is_an_exit_of_that_long_never_a_new_short():
    for text, sym in (("should I sell AMD here?", "AMD"), ("selling AMD, taking profit", "AMD"),
                      ("about to sell half my NVDA", "NVDA")):
        assert _gate_args(text, [sym]) == [{"side": "LONG", "symbol": sym, "exit": True}], text
        names = [r.name for r in attach.plan_attachments(text, HELD, NOW, book=[sym])]
        assert "journal_pack" in names and "pick_pack" in names, text


def test_covering_a_held_short_is_an_exit_of_that_short():
    assert _gate_args("cover TSLA", ["TSLA"]) == [{"side": "SHORT", "symbol": "TSLA", "exit": True}]


def test_selling_a_name_not_held_is_a_new_short():
    assert _gate_args("about to sell AMD", []) == [{"side": "SHORT", "symbol": "AMD"}]
    assert _gate_args("adding to AMD", ["AMD"]) == [{"side": "LONG", "symbol": "AMD", "add": True}]  # not an exit


#: P18 re-review 2: "close" is a price word unless it is an action on the name; each verb binds to its ticker.
BOOK = ["AMD", "NVDA", "TSLA"]


def _packs(text):
    return [(r.name, dict(r.args)) for r in attach.plan_attachments(text, HELD, NOW, book=BOOK)]


def test_close_as_a_price_word_is_never_an_exit():
    assert [a for n, a in _packs("AMD closing strong, add more?") if n == "gate_pack"] == [
        {"side": "LONG", "symbol": "AMD", "add": True}]  # an add, never an exit
    for text in ("did AMD close above vwap", "what is the AMD close today", "where did AMD close yesterday?"):
        packs = _packs(text)
        assert not [a for n, a in packs if n == "gate_pack"], text
        assert {"bars_pack", "journal_pack"} & {n for n, _a in packs}, text


def test_close_out_and_close_my_are_exits():
    assert [a for n, a in _packs("close out AMD") if n == "gate_pack"] == [
        {"side": "LONG", "symbol": "AMD", "exit": True}]
    assert [a for n, a in _packs("close my NVDA") if n == "gate_pack"] == [
        {"side": "LONG", "symbol": "NVDA", "exit": True}]


def test_each_verb_binds_to_its_own_ticker():
    gates = [a for n, a in _packs("sell AMD and buy NVDA") if n == "gate_pack"]
    assert gates == [{"side": "LONG", "symbol": "AMD", "exit": True}, {"side": "LONG", "symbol": "NVDA", "add": True}]
    from mentor_app import intent

    assert intent.bind("sell AMD and buy NVDA", ["AMD", "NVDA"]) == {"AMD": {"sell"}, "NVDA": {"long"}}
