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
    # Review advisory: one stopped trade (a winner) cannot be the worst by R; fewer than 2 stopped losers = by $.
    assert worst["symbol"] == "AMD" and worst["by"] == "$" and "fewer than 2 to rank by R" in worst["text"]
    week = {row["id"]: row for row in _journal_pack("week").rows}
    assert week["jrn:wk2026-09-28:best"]["symbol"] == "NVDA"
    assert week["jrn:wk2026-09-28:worst"]["symbol"] == "AMD", "one stopped loser (TSLA): by $"


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


# ---------------------------------------------------------------- step 5: tape diff
def _tape_store(tmp_path, snapshots):
    import sqlite3

    db = tmp_path / "mentor_chat.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE app_state (key TEXT PRIMARY KEY, value TEXT, updated_utc TEXT)")
    for day, rows in snapshots.items():
        conn.execute("INSERT INTO app_state VALUES (?, ?, '')", (f"tape:snapshot:{day}", __import__("json").dumps(rows)))
    conn.commit()
    conn.close()
    return db


def test_the_tape_diff_says_what_changed_since_the_previous_snapshot(tmp_path):
    from dataclasses import replace

    from mentor_packs import regime_pack

    now = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)
    yesterday = regime_pack.build(now=datetime(2026, 9, 29, 15, 0, tzinfo=timezone.utc), sources=replace(
        regime_pack.fixture_sources(), econ=lambda session: {
            "today": [{"id": "t1", "date": session, "time_et": "08:30", "label": "CPI (Aug)"},
                      {"id": "t2", "date": session, "time_et": "10:00", "label": "JOLTS"}],
            "week": [{"id": "w1", "date": "2026-09-30", "time_et": "09:45", "label": "Chicago  PMI"},
                     {"id": "w2", "date": "2026-09-30", "time_et": "10:00", "label": "ISM manufacturing"}]}))
    older = [dict(row, regime="strong since 2026-08-01") if row["id"] == "tape:regime" else row
             for row in __import__("json").loads(regime_pack.snapshot_json(yesterday))]
    db = _tape_store(tmp_path, {"2026-09-28": older, "2026-09-29": __import__("json").loads(
        regime_pack.snapshot_json(yesterday))})
    sources = replace(regime_pack.fixture_sources(), d1_env=lambda day: "bullish_trend", econ=lambda session: {
        "today": [{"id": "t1", "date": session, "time_et": "13:00", "label": "Fed Governor Waller speaks"},
                  {"id": "t2", "date": session, "time_et": "09:45", "label": "Chicago PMI"},
                  {"id": "t3", "date": session, "time_et": "10:00", "label": "ISM Manufacturing"}]})
    pack = regime_pack.build(diff=True, now=now, sources=sources, chat_db=db)
    rows = {row["id"]: row for row in pack.rows}
    assert rows["tape:diff:d1env"]["changed"] is True
    assert rows["tape:diff:d1env"]["text"] == "Since 2026-09-29: the D1 environment CHANGED from bearish_trend to bullish_trend"
    assert rows["tape:diff:regime"]["changed"] is False, "compared with the newest earlier snapshot, not the 28th"
    assert rows["tape:diff:leaders"]["changed"] is False and "Technology" in rows["tape:diff:leaders"]["text"]
    assert rows["tape:diff:night"]["changed"] is False
    econ = rows["tape:diff:econ"]["text"]
    assert econ == "Since 2026-09-29: new econ events today: Econ 2026-09-30 13:00 ET: Fed Governor Waller speaks", econ
    assert not any(row["id"].startswith("tape:diff") for row in regime_pack.build(
        now=now, sources=sources, chat_db=db).rows), "no diff unless asked"


def test_no_snapshot_yesterday_is_one_first_day_row(tmp_path):
    from mentor_packs import regime_pack

    pack = regime_pack.build(diff=True, now=datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc),
                             sources=regime_pack.fixture_sources(), chat_db=_tape_store(tmp_path, {}))
    diff = [row for row in pack.rows if row["id"].startswith("tape:diff")]
    assert [row["text"] for row in diff] == [
        "Tape diff: no snapshot for 2026-09-29 (first day); what changed is unknown"]


def test_tape_change_questions_attach_the_diff_and_the_night():
    for question in ("what changed in the tape since yesterday", "anything different today?", "did the tape change"):
        got = [(r.name, r.args) for r in attach.plan_attachments(question, {}, NOW)]
        assert ("regime_pack", {"diff": True}) in got and ("night_pack", {}) in got, (question, got)
        assert not any(name == "journal_pack" for name, _ in got), (question, got)


def test_the_app_snapshots_the_first_tape_of_the_pt_day_and_never_overwrites(tmp_path):
    import json
    from types import SimpleNamespace

    from mentor_app.store import MentorChatStore
    from mentor_app.window import MentorWindow
    from mentor_packs import regime_pack

    store = MentorChatStore(tmp_path / "mentor_chat.sqlite3")
    clock = {"now": datetime(2026, 9, 30, 13, 0, tzinfo=timezone.utc)}  # 06:00 PT
    host = SimpleNamespace(store=store, _now=lambda: clock["now"])
    first = regime_pack.build(now=clock["now"], sources=regime_pack.fixture_sources())
    MentorWindow._snapshot_tape(host, first)
    later = regime_pack.build(now=clock["now"], sources=__import__("dataclasses").replace(
        regime_pack.fixture_sources(), d1_env=lambda day: "bullish_trend"))
    MentorWindow._snapshot_tape(host, later)
    stored = json.loads(store.get_state("tape:snapshot:2026-09-30"))
    assert next(row for row in stored if row["id"] == "tape:d1env")["label"] == "bearish_trend", "first build wins"
    assert not any(row["kind"] == "asof" for row in stored), "rows only"
    pack = regime_pack.build(diff=True, now=datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc),
                             sources=regime_pack.fixture_sources(), chat_db=store.path)
    assert any(row["id"] == "tape:diff:d1env" and row["changed"] is False for row in pack.rows)


# ---------------------------------------------------------------- live rerun fixes (2026-09-30 evening)
def _live_context():
    """The context pack the app builds: journal positions as ``ctx:pos:*`` rows, plus Focus shorts LULU and MKC."""
    from dataclasses import replace

    from mentor_packs import context_pack

    positions = [{"trade_id": f"T{i}", "symbol": sym, "direction": side, "quantity_opened": 10, "quantity_closed": 0,
                  "average_entry_price": 50.0, "opened_at": "2026-09-20T10:00:00-04:00"}
                 for i, (sym, side) in enumerate((("NVDA", "SHORT"), ("DRAM", "Short"), ("QCOM", "LONG")))]
    sources = replace(context_pack.fixture_sources(), open_positions=lambda: positions)
    rows = [dict(row) for row in context_pack.build(now=NOW, sources=sources).rows]
    rows.append({"kind": "focus", "category": "swing", "side": "short", "names": ["LULU", "MKC"]})
    return rows


def test_the_live_context_book_pools_the_real_book_by_its_side():
    rows = _live_context()
    assert [r["direction"] for r in rows if r.get("kind") == "position"] == ["SHORT", "SHORT", "LONG"]
    known = attach.known_symbols(rows, [], [])
    assert attach.book_symbols(rows) == ["NVDA", "DRAM", "QCOM"]
    got = attach.plan_attachments("which of my open shorts is most at risk into earnings", known, NOW,
                                  book=attach.book_symbols(rows))
    earn = next(r for r in got if r.name == "earnings_pack")
    assert earn.args["symbols"] == ["NVDA", "DRAM"] and earn.args["book"] == ["NVDA", "DRAM"]
    assert "book_pack" in [r.name for r in got]
    # A book name never takes a Focus side, even when its own side is unknown.
    rows = [{"kind": "position", "symbol": "NVDA", "side": ""},
            {"kind": "focus", "category": "swing", "side": "long", "names": ["NVDA", "V"]}]
    assert attach.known_symbols(rows, [("NVDA", "LONG")], []) == {"NVDA": "", "V": "LONG"}


def test_an_empty_reply_never_renders_empty(caplog):
    from mentor_app import brain

    requests = attach.plan_attachments("what's the tape doing", {}, NOW)
    with caplog.at_level("WARNING"):
        result, sent, _ = _turn([""], requests)
    assert result["empty_fallback"] is True
    assert result["text"].startswith("(the model returned no text; here is the data)\n\n")
    assert "[regime_pack:row]" in result["text"] and "returned no text" in caplog.text
    result, _, _ = _turn([""], [])
    assert result["text"] == "(the model returned no text; no data was attached)" and brain.EMPTY_REPLY_LINE


def _veto_agg_world():
    """Three LONG veto reasons over the floor with 5d means +1 %, -2.83 % and -4.53 %."""
    veto_pack, world = _veto_world()
    from datetime import date, timedelta

    notes, cohort = [], ["trade_date,symbol,side,source,h1_date,h1_return,h3_date,h3_return,h5_date,h5_return,"
                         "h10_date,h10_return"]
    for reason, h5, h10 in (("compressed", 0.01, 0.05), ("incoming_trendline", -0.0283, 0.02),
                            ("too_extended_from_base", -0.0453, -0.01)):
        for index in range(30):
            day = (date(2026, 8, 3) + timedelta(days=index % 10)).isoformat()
            sym = f"{reason[:2].upper()}{index:02d}"
            notes.append(__import__("json").dumps({"event_type": "veto", "symbol": sym, "side": "LONG",
                                                   "session_date": day, "created_at": f"{day}T09:00:00-07:00",
                                                   "decision_session": day, "reason_code": reason,
                                                   "event_id": f"{sym}-{day}"}))
            cohort.append(f"{day},{sym},LONG,veto,{day},0,{day},0,{day},{h5},{day},{h10}")
    world.annotations.write_text("\n".join(notes) + "\n", encoding="utf-8")
    world.veto_outcomes.write_text("\n".join(cohort) + "\n", encoding="utf-8")
    return veto_pack, world


def test_veto_aggregates_carry_their_rank_and_the_verdict_names_its_basis():
    veto_pack, world = _veto_agg_world()
    pack = veto_pack.build(scope="month", now=datetime(2026, 8, 25, 20, 0, tzinfo=timezone.utc), paths=world)
    rows = {row["id"]: row for row in pack.rows}
    ranks = {key: rows[f"veto:agg:{key}:LONG"]["rank"] for key in
             ("compressed", "incoming_trendline", "too_extended_from_base")}
    assert ranks == {"compressed": 1, "incoming_trendline": 2, "too_extended_from_base": 3}
    assert rows["veto:agg:compressed:LONG"]["text"].startswith("worst #1 of 3 veto records (cost the most): ")
    assert rows["veto:agg:incoming_trendline:LONG"]["text"].startswith("#2 of 3 veto records: ")
    assert rows["veto:agg:too_extended_from_base:LONG"]["text"].startswith(
        "#3 of 3 veto records (avoided the most loss): ")
    text = rows["veto:agg:too_extended_from_base:LONG"]["text"]
    assert "verdict (5d mean -4.53%, n=30): vetoing this reason: avoided a losing cohort" in text
    assert "rank" not in rows["veto:agg:total"]


def test_mirror_veto_rows_carry_their_rank():
    from mentor_packs import mirror_pack

    ranked = [row for row in mirror_pack.fixture().rows if row.get("kind") == "veto" and "rank" in row]
    assert [row["rank"] for row in ranked] == [1] and ranked[0]["text"].startswith("worst #1 of 1 veto records")
    assert all("rank" not in row for row in mirror_pack.fixture().rows if row.get("too_few"))
    from mentor_app.chat_model import PERSONA_PROMPT

    assert "For best/worst questions, read the row's rank; never re-rank yourself." in PERSONA_PROMPT


def test_recap_issues_carry_their_rank_by_sessions():
    from mentor_packs import recaps_pack

    issues = [row for row in recaps_pack.fixture().rows if row.get("kind") == "issue"]
    assert issues and [row["rank"] for row in issues] == list(range(1, len(issues) + 1))
    assert issues[0]["text"].startswith(f"#1 of {len(issues)} by sessions: ")
    assert [row["count"] for row in issues] == sorted((row["count"] for row in issues), reverse=True)


def test_a_tape_with_an_unknown_row_is_not_snapshotted_and_a_gap_is_said(tmp_path):
    from types import SimpleNamespace

    from mentor_app.store import MentorChatStore
    from mentor_app.window import MentorWindow
    from mentor_packs import regime_pack
    from mentor_packs.registry import make_pack

    store = MentorChatStore(tmp_path / "mentor_chat.sqlite3")
    host = SimpleNamespace(store=store, _now=lambda: datetime(2026, 9, 29, 15, 0, tzinfo=timezone.utc))
    good = regime_pack.build(now=host._now(), sources=regime_pack.fixture_sources())
    broken = make_pack(regime_pack.NAME, [*good.rows, {"id": "tape:breadth", "kind": "unknown", "text": "x"}])
    MentorWindow._snapshot_tape(host, broken)
    assert store.get_state("tape:snapshot:2026-09-29") is None, "an unknown row: retry at the next build"
    host._now = lambda: datetime(2026, 9, 28, 15, 0, tzinfo=timezone.utc)
    MentorWindow._snapshot_tape(host, good)
    pack = regime_pack.build(diff=True, now=datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc),
                             sources=regime_pack.fixture_sources(), chat_db=store.path)
    gap = next(row for row in pack.rows if row["id"] == "tape:diff:gap")
    assert gap["text"] == "Tape diff: no clean snapshot for 2026-09-29; comparing with 2026-09-28 instead"


def test_a_plan_to_cover_is_not_an_announced_fetch():
    from mentor_app import style

    assert not style.announces_fetch("Hold it; I'll look to cover into the 10:00 print.")
    assert not style.announces_fetch("I am checking that against your record: 3 of 9 stopped [jrn:x:totals].")
    assert style.announces_fetch("I will check your journal for the stops.")
    assert style.announces_fetch("Let me pull your recent trades.")


def test_an_empty_reply_data_dump_does_not_count_as_citations():
    from mentor_app import brain

    turns = [{"role": "assistant", "text": f"{brain.EMPTY_REPLY_LINE}\n\n## book_pack\n[book:pos:1:NVDA] NVDA 10"},
             {"role": "assistant", "text": "Short NVDA [book:pos:2:NVDA]."}]
    assert attach.recent_cited_ids(turns) == {"book:pos:2:NVDA"}
