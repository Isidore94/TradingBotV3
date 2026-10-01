"""P16: the gaps the 100-question live eval found (book scope, hold by outcome, veto aggregates, tape diff, routes)."""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import attach  # noqa: E402

NOW = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)  # Wed 11:00 ET


def _desk():
    """Focus shorts LULU, MKC, GIS; a liked short WULF; the book holds NVDA and DRAM short and QCOM long."""
    rows = [{"kind": "focus", "category": "swing", "side": "short", "names": ["LULU", "MKC", "GIS"]},
            {"kind": "position", "symbol": "NVDA", "direction": "SHORT"},
            {"kind": "position", "symbol": "DRAM", "direction": "SHORT"},
            {"kind": "position", "symbol": "QCOM", "direction": "LONG"}]
    likes = [("WULF", "SHORT")]
    return rows, likes, attach.known_symbols(rows, likes, [])


def _plan(question):
    rows, likes, known = _desk()
    return attach.plan_attachments(question, known, NOW, book=attach.book_symbols(rows), liked=likes)


# ---------------------------------------------------------------- step 1: book-only scope
def test_my_open_shorts_into_earnings_is_the_book_only_never_focus():
    for question in ("which of my open shorts is most at risk into earnings", "my shorts reporting soon?",
                     "any earnings in my positions?"):
        got = _plan(question)
        earn = next(r for r in got if r.name == "earnings_pack")
        assert set(earn.args["symbols"]) <= {"NVDA", "DRAM", "QCOM"}, (question, earn.args)
        assert not {"LULU", "MKC", "GIS", "WULF"} & set(earn.args["symbols"]), question
        assert "book_pack" in [r.name for r in got], question
    shorts = next(r for r in _plan("my shorts reporting soon?") if r.name == "earnings_pack")
    assert shorts.args["symbols"] == ["NVDA", "DRAM"] and shorts.args["book"] == ["NVDA", "DRAM"]


def test_my_focus_shorts_and_watchlist_are_focus_and_likes_not_the_book():
    for question in ("any of my focus shorts reporting this week?", "earnings in my watchlist shorts?"):
        earn = next(r for r in _plan(question) if r.name == "earnings_pack")
        assert earn.args["symbols"] == ["WULF", "LULU", "MKC", "GIS"], (question, earn.args)
        assert earn.args["book"] == [] and earn.args["liked"] == ["WULF"]
        assert earn.args["focus"] == ["LULU", "MKC", "GIS"]


def test_every_earnings_row_says_book_or_focus():
    from mentor_packs import earnings_pack, pick_pack

    world = pick_pack.write_fixture_world(tempfile.mkdtemp())
    pack = earnings_pack.build(["NVDA", "TSLA", "ZZZ"], book=["NVDA"], liked=["ZZZ"], focus=["TSLA"], side="SHORT",
                               now=pick_pack.FIXTURE_NOW, paths=world)
    rows = {row["id"]: row for row in pack.rows}
    assert rows["earn:NVDA"]["origin"] == "book" and rows["earn:NVDA"]["text"].startswith("NVDA (book short):")
    assert rows["earn:TSLA"]["origin"] == "focus" and rows["earn:TSLA"]["text"].startswith("TSLA (Focus short):")
    assert rows["earn:ZZZ"]["origin"] == "liked" and rows["earn:ZZZ"]["text"].startswith("ZZZ (liked short):")
    assert "never positions" in rows["earn:asof"]["text"]


def test_news_over_my_shorts_reads_book_names_with_their_origin():
    picks = [r for r in _plan("any news on my shorts?") if r.name == "pick_pack"]
    assert [(r.args["symbol"], r.args["origin"]) for r in picks] == [("NVDA", "book"), ("DRAM", "book")]
    picks = [r for r in _plan("any news on my focus shorts?") if r.name == "pick_pack"]
    assert [(r.args["symbol"], r.args["origin"]) for r in picks] == [
        ("WULF", "liked"), ("LULU", "focus"), ("MKC", "focus")]


def test_a_pick_pack_for_a_focus_name_says_it_is_not_a_position():
    from mentor_packs import pick_pack

    world = pick_pack.write_fixture_world(tempfile.mkdtemp())
    pack = pick_pack.build("TSLA", origin="focus", now=pick_pack.FIXTURE_NOW, paths=world)
    assert "a Focus name, NOT a position" in pack.rows[0]["text"]
    assert all(row["origin"] == "focus" for row in pack.rows)


def test_do_i_have_any_shorts_on_reads_the_book():
    assert "book_pack" in [r.name for r in _plan("do I have any shorts on")]


# ---------------------------------------------------------------- step 2: hold time by outcome, best / worst
def _journal_pack(day="today"):
    from mentor_packs import journal_pack

    tmp = Path(tempfile.mkdtemp())
    return journal_pack.build(day, now=journal_pack.FIXTURE_NOW,
                              journal=journal_pack.write_fixture_journal(tmp / "trade_journal.sqlite3"),
                              chat_db=tmp / "none.sqlite3")


def test_the_journal_says_hold_time_on_winners_and_losers():
    rows = {row["id"]: row for row in _journal_pack("today").rows}
    # Fixture: winners NVDA 45 min and the ALL spread 40 min; loser AMD 15 min.
    win, lose = rows["jrn:2026-09-30:hold:winners"], rows["jrn:2026-09-30:hold:losers"]
    assert (win["n"], win["median_min"], win["mean_min"]) == (2, 42.5, 42.5)
    assert win["text"] == "Winners held a median 42 min, mean 42 min (n=2)"
    assert (lose["n"], lose["median_min"]) == (1, 15.0) and "median 15 min" in lose["text"]


def test_best_and_worst_rank_by_r_when_a_stop_exists_and_say_why():
    rows = {row["id"]: row for row in _journal_pack("today").rows}
    best, worst = rows["jrn:2026-09-30:best"], rows["jrn:2026-09-30:worst"]
    assert best["by"] == "R" and best["symbol"] == "NVDA" and "+1.00R" in best["text"]
    assert "ranked by R (1 of 3 trade(s) have a planned stop" in best["text"]
    assert worst["symbol"] == "NVDA", "only stopped trades are ranked by R"
    week = {row["id"]: row for row in _journal_pack("week").rows}
    assert week["jrn:wk2026-09-28:best"]["symbol"] == "NVDA"
    assert week["jrn:wk2026-09-28:worst"]["symbol"] == "TSLA" and "-0.50R" in week["jrn:wk2026-09-28:worst"]["text"]


def test_best_by_dollars_when_no_trade_has_a_stop():
    from mentor_packs import journal_pack
    from mentor_packs.journal_read import Unit

    t = datetime(2026, 9, 30, 10, 0, tzinfo=timezone.utc)
    units = [Unit([{"direction": "LONG"}], "day", t, t, 50.0, None, symbol="AAA", ids=["A"]),
             Unit([{"direction": "SHORT"}], "day", t, t, -80.0, None, symbol="BBB", ids=["B"])]
    rows = {row["id"]: row for row in journal_pack.outcome_rows("d", units)}
    assert rows["jrn:d:best"]["symbol"] == "AAA" and rows["jrn:d:worst"]["symbol"] == "BBB"
    assert "ranked by $ (no trade here has a planned stop" in rows["jrn:d:best"]["text"]


def test_hold_and_best_questions_route_to_the_journal_month_and_the_mirror():
    for question in ("whats my average hold time on winners vs losers", "what's the best setup I've had this month and why",
                     "best trade this month?"):
        got = [(r.name, r.args) for r in attach.plan_attachments(question, {}, NOW)]
        assert ("journal_pack", {"day": "month"}) in got and ("mirror_pack", {}) in got, (question, got)


# ---------------------------------------------------------------- step 3: follow-your-vetoes aggregate
def _veto_world():
    from mentor_packs import veto_pack

    return veto_pack, veto_pack.write_fixture_world(Path(tempfile.mkdtemp()) / "desk")


def test_every_daily_slice_says_clears_baseline_and_which_lb_is_larger():
    veto_pack, world = _veto_world()
    slices = {row["id"]: row for row in veto_pack.build(now=veto_pack.FIXTURE_NOW, paths=world).rows
              if row.get("kind") == "slice"}
    assert all(row["clears_baseline"] in ("yes", "no") for row in slices.values())
    aaa, bbb = slices["veto:2026-09-29:AAA:1:slice"], slices["veto:2026-09-29:BBB:1:slice"]
    assert aaa["clears_baseline"] == "yes" and "LB=0.71 is ABOVE the LONG baseline LB=0.45" in aaa["text"]
    assert bbb["clears_baseline"] == "no" and "LB=0.28 is BELOW the SHORT baseline LB=0.43" in bbb["text"]
    assert slices["veto:2026-09-29:CCC:1:slice"]["clears_baseline"] == "no", "too few never clears"


def test_a_month_of_vetoes_aggregates_by_reason_with_a_verdict():
    veto_pack, world = _veto_world()
    pack = veto_pack.build(scope="month", now=datetime(2026, 8, 25, 20, 0, tzinfo=timezone.utc), paths=world)
    rows = {row["id"]: row for row in pack.rows}
    long_ = rows["veto:agg:compressed:LONG"]
    assert long_["vetoes"] == 40 and long_["h5"]["n"] == 32 and long_["h5"]["pending"] == 8
    assert round(long_["h5"]["mean"], 4) == 0.03 and long_["h5"]["win_rate"] == 1.0 and long_["h10"]["n"] == 32
    assert long_["verdict"] == "cost" and "cost a winning cohort" in long_["text"]
    assert long_["clears_baseline"] == "yes" and "is ABOVE the LONG baseline" in long_["text"]
    short = rows["veto:agg:too_extended_from_base:SHORT"]
    assert short["verdict"] == "too_few" and short["clears_baseline"] == "no"
    assert "is BELOW the SHORT baseline" in short["text"]
    total = rows["veto:agg:total"]
    assert total["vetoes"] == 85 and "as a whole" in total["text"]
    assert pack.rows[1]["id"] == "veto:agg:compressed:LONG", "the reason that cost the most comes first"


def test_a_losing_cohort_is_an_avoided_loss():
    veto_pack, world = _veto_world()
    lines = world.veto_outcomes.read_text(encoding="utf-8").splitlines()
    world.veto_outcomes.write_text("\n".join([lines[0]] + [line.replace(",0.03,", ",-0.03,") for line in lines[1:]])
                                   + "\n", encoding="utf-8")
    pack = veto_pack.build(scope="month", now=datetime(2026, 8, 25, 20, 0, tzinfo=timezone.utc), paths=world)
    row = next(r for r in pack.rows if r["id"] == "veto:agg:compressed:LONG")
    assert row["verdict"] == "avoided" and "avoided a losing cohort" in row["text"]


def test_veto_record_questions_route_to_the_aggregate_and_the_mirror():
    cases = {"if I had followed my vetoes exactly this month how would I have done": "month",
             "which veto reason of mine has the worst track record": "month",
             "did my vetoes this week work out": "week",
             "which of my vetoes this month would have worked": "month"}
    for question, scope in cases.items():
        got = [(r.name, r.args) for r in attach.plan_attachments(question, {}, NOW)]
        assert ("veto_pack", {"scope": scope}) in got and ("mirror_pack", {}) in got, (question, got)
    assert ("veto_pack", {"date": "2026-09-29"}) in [
        (r.name, r.args) for r in attach.plan_attachments("what did I veto yesterday and was I right", {}, NOW)]


# ---------------------------------------------------------------- step 4: comparison discipline, no announced fetch
REGIME_OR_ME = (
    "The current regime is a bear channel with lower highs, which has been in place since 2026-09-28 [tape:regime]. "
    "In this environment, the underlying tape is showing bearish weakness [tape:night:5].\n\n"
    "To determine if the issue is the regime or your execution, we need to look at your recent performance.\n\n"
    "I am checking your recent journal entries and vetoes to see if your long entries are aligning with the bear "
    "channel reality.")


def _lines(*chunks):
    import json

    return [json.dumps(chunk).encode() for chunk in chunks]


def _answer(text):
    return _lines({"message": {"content": text}, "done": False},
                  {"message": {"content": ""}, "done": True, "prompt_eval_count": 10, "eval_count": 5})


def _turn(replies, attachments):
    from mentor_app import brain
    from mentor_app.chat_model import ChatModel
    from mentor_packs.registry import make_pack

    chat = ChatModel()
    question = "I keep getting stopped out on longs in this regime, is that the regime or me"
    chat.add("user", question)
    sent: list[dict] = []
    queue = list(replies)
    built: list[str] = []

    def pack(name, args):
        built.append(name)
        return make_pack(name, [{"id": f"{name}:row", "text": f"{name} says 3 of 9 longs stopped"}])

    result = brain.run_turn(chat.messages(context_text="[ctx:auto_mode] DESK"), model="gemma4:12b",
                            endpoint="http://x", tools=[{"type": "function", "function": {"name": "journal_pack"}}],
                            native_tools=True, build_pack=pack, attachments=attachments,
                            stream_post=lambda u, p, c: sent.append(p) or _answer(queue.pop(0)), question=question)
    return result, sent, built


def test_the_regime_or_me_reply_that_announces_a_fetch_is_re_asked_once_with_the_packs():
    from mentor_app import style

    assert style.announces_fetch(REGIME_OR_ME)
    assert not style.announces_fetch("Mostly you: 3 of 9 longs stopped [jrn:mo2026-09:totals].")
    requests = attach.plan_attachments("I keep getting stopped out on longs in this regime, is that the regime or me",
                                       {}, NOW)
    result, sent, built = _turn([REGIME_OR_ME, "Mostly you: 3 of 9 longs stopped [journal_pack:row]."], requests)
    assert len(sent) == 2, "one re-ask"
    assert result["fetch_retry"] is True and result["text"] == "Mostly you: 3 of 9 longs stopped [journal_pack:row]."
    assert result["first_reply"] == REGIME_OR_ME
    last = sent[1]["messages"][-1]["content"]
    assert last.startswith(style.FETCH_RETRY_PROMPT) and f"[{requests[0].name}:row]" in last
    assert built.count(requests[0].name) == 2, "the planner's packs are built once more"


def test_a_retry_that_still_announces_or_no_packs_gets_the_note():
    from mentor_app import style

    requests = attach.plan_attachments("is that the regime or me", {}, NOW)
    result, sent, _ = _turn([REGIME_OR_ME, "Let me pull your journal."], requests)
    assert len(sent) == 2 and result["text"].endswith(style.NO_FETCH_NOTE)
    result, sent, _ = _turn([REGIME_OR_ME], [])
    assert len(sent) == 1 and result["text"] == f"{REGIME_OR_ME} {style.NO_FETCH_NOTE}"


def test_the_persona_says_how_to_compare_and_never_to_announce_a_fetch():
    from mentor_app.chat_model import PERSONA_PROMPT

    assert ("When comparing two numbers, write both and say which is larger; call something better or worse only "
            "when the pack row says so (`clears_baseline`, `verdict`).") in PERSONA_PROMPT
    assert "Never write 'I am checking...' or 'let me pull...': call the tool or answer." in PERSONA_PROMPT


# ---------------------------------------------------------------- step 6: watch tomorrow
def test_what_to_watch_tomorrow_reads_the_brief_watch_list_the_econ_and_the_day_review():
    for question in ("what's the single most important thing to watch at the open tomorrow", "what to watch tomorrow",
                     "most important thing to watch tomorrow?"):
        got = [(r.name, r.args) for r in attach.plan_attachments(question, {}, NOW)]
        assert ("fundamentals_pack", {"section": "watch"}) in got, (question, got)
        assert ("regime_pack", {}) in got and ("night_pack", {"section": "day_review"}) in got, (question, got)


def test_the_brief_watch_section_puts_scheduled_data_first():
    from mentor_packs import fundamentals_pack

    assert "watch" in fundamentals_pack.SCHEMA["function"]["parameters"]["properties"]["section"]["enum"]
    pack = fundamentals_pack.build("today", "watch", now=fundamentals_pack.FIXTURE_NOW,
                                   paths=fundamentals_pack.write_fixture_world(tempfile.mkdtemp()))
    sections = [row["section"] for row in pack.rows]
    assert sections[0] == "asof" and sections[1] == "events", "scheduled data first"
    assert sections.index("signals") < sections.index("bottom_line"), "then the watch list, then the bottom line"
    assert "Nonfarm Payrolls" in pack.as_text()
    from mentor_app.chat_model import SYSTEM_PROMPT

    assert ("rank by impact: scheduled high-impact data first, then the brief's watch list, then levels"
            in SYSTEM_PROMPT)


# ---------------------------------------------------------------- step 7: regime or me, this week vs last week
def test_regime_or_me_reads_the_month_the_mirror_and_the_tape():
    for question in ("I keep getting stopped out on longs in this regime, is that the regime or me",
                     "am I the problem", "my fault or the market?"):
        got = [(r.name, r.args) for r in attach.plan_attachments(question, {}, NOW)]
        assert ("journal_pack", {"day": "month"}) in got, (question, got)
        assert ("mirror_pack", {}) in got and ("regime_pack", {}) in got, (question, got)
    from mentor_app.chat_model import SYSTEM_PROMPT

    assert "cite the mirror's regime row and kind row and the journal totals" in SYSTEM_PROMPT


def test_this_week_vs_last_week_reads_both_weeks_and_never_today():
    for question in ("compare this week to last week in one paragraph", "how is my week vs last week"):
        got = [(r.name, r.args) for r in attach.plan_attachments(question, {}, NOW) if r.name == "journal_pack"]
        assert got == [("journal_pack", {"day": "week"}), ("journal_pack", {"day": "last_week"})], (question, got)
