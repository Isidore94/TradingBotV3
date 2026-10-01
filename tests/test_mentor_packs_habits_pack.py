"""P18 B: habits - the night counts what the trader keeps saying; the app shows it (/habits, attach, brief, Inbox)."""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ai_jobs import mentor_habits, mentor_review  # noqa: E402
from mentor_app import attach, commands, journal_mode, memory  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402
from mentor_packs import habits_pack, reads_pack, registry  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
ET = ZoneInfo("America/New_York")
SESSION = "2026-09-29"
NIGHT = datetime(2026, 9, 29, 23, 0, tzinfo=PT)


def _day_at(day: str, hh: int, mm: int = 0) -> datetime:
    return datetime.fromisoformat(f"{day}T{hh:02d}:{mm:02d}:00").replace(tzinfo=ET)


def _entry(store, day, hh, text, *, after=""):
    fields = journal_mode.entry_fields(text, _day_at(day, hh))
    fields["last_trade_text"] = after
    fields["regime"] = "Trader's regime: chop since 2026-09-20 (day 3)"
    return store.add_journal_entry(fields)


@pytest.fixture
def chat(tmp_path):
    path = tmp_path / "mentor_chat.sqlite3"
    store = MentorChatStore(path)
    for day in ("2026-09-25", "2026-09-26", "2026-09-29"):
        _entry(store, day, 10, "I'm annoyed, I chased the open again", after="the AMD stop")
    _entry(store, "2026-09-29", 15, "bored now")
    session = store.start_session("m")
    store.add_turn(session, "user", "Why do I keep chasing?")
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE turns SET ts_utc = ?", ("2026-09-29T17:00:00.000+00:00",))
    return path


def _journal(tmp_path, closes):
    path = tmp_path / "trade_journal.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE trades (trade_id TEXT, account_number TEXT, symbol TEXT, security_type TEXT, direction TEXT,"
        " status TEXT, opened_at TEXT, closed_at TEXT, quantity_opened REAL, quantity_closed REAL,"
        " average_entry_price REAL, average_exit_price REAL, net_pnl REAL, net_pnl_usd REAL);"
        "CREATE TABLE trade_annotations (trade_id TEXT, planned_stop REAL, planned_risk REAL);")
    for n, (stamp, pnl) in enumerate(closes):
        conn.execute("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (f"T{n}", "M1", "NVDA", "STK", "LONG", "CLOSED", stamp, stamp, 1, 1, 1, 1, pnl, pnl))
    conn.commit()
    conn.close()
    return path


# ------------------------------------------------------------------ the night's counting
def test_phrases_are_three_words_trimmed_of_stopwords():
    assert "chased the open" in mentor_habits.phrases("I'm annoyed, I chased the open again")
    assert mentor_habits.phrases("I chased it") == set()  # a stopword at either end
    assert mentor_habits.phrases("the and of") == set()


def test_habits_need_three_days_and_carry_context_examples_and_red_days(chat, tmp_path):
    said = mentor_habits.said(chat, *mentor_habits.window(SESSION))
    assert [item["id"] for item in said][:3] == ["journal:1", "journal:2", "journal:3"]
    assert any(item["id"] == "turn:1" for item in said)
    journal = _journal(tmp_path, [(f"{day}T11:00:00-04:00", -50.0) for day in ("2026-09-25", "2026-09-26",
                                                                                   "2026-09-29")])
    habits = mentor_habits.find_habits(said, red_after=mentor_habits.red_after_from_journal(journal))
    keys = [h["key"] for h in habits]
    assert "tag:fomo" in keys and "tag:tilted" in keys and "phrase:chased the open" in keys
    assert "tag:bored" not in keys  # one day only
    fomo = next(h for h in habits if h["key"] == "tag:fomo")
    assert (fomo["first_seen"], fomo["last_seen"], fomo["days_seen"]) == ("2026-09-25", "2026-09-29", 3)
    assert fomo["examples"] == ["journal:1", "journal:2", "journal:3"] and fomo["id"] == "habit:tag_fomo"
    assert fomo["context"]["after_loss"] == 3 and fomo["context"]["regimes"] == {"chop": 3}
    assert fomo["red_rest_of_day_days"] == 3 and fomo["inbox_ok"] is True
    # No closes after the line = unknown, never counted red.
    quiet = mentor_habits.find_habits(said, red_after=mentor_habits.red_after_from_journal(tmp_path / "none.db"))
    assert all(h["red_rest_of_day_days"] == 0 and not h["inbox_ok"] for h in quiet)
    facts = mentor_habits.day_facts(said, SESSION)
    assert facts["mood_tags"] == {"bored": 1, "fomo": 1, "tilted": 1} and facts["after_loss"] == 1
    assert facts["journal_lines"] == 2 and facts["turns"] == 1


def test_the_habits_reply_is_citation_checked_and_code_words_the_rest(chat):
    habits = mentor_habits.find_habits(mentor_habits.said(chat, *mentor_habits.window(SESSION)))
    rows = mentor_habits.habit_rows(habits)
    allowed = [r["id"] for r in rows] + [e for r in rows for e in r["examples"]]
    reply = {"habits": [{"key": habits[0]["key"], "text": "You chase after a stop.",
                         "evidence_refs": [habits[0]["id"], "journal:1"]},
                        {"key": "tag:nonsense", "text": "x", "evidence_refs": [habits[0]["id"]]}]}
    kept, dropped = mentor_habits.check_habits(reply, habits, allowed)
    assert kept[0]["text"] == "You chase after a stop." and dropped == 1
    assert [k["key"] for k in kept] == [h["key"] for h in habits[:5]]
    with pytest.raises(mentor_habits.HabitsRejected):
        mentor_habits.check_habits({"habits": [{"key": habits[0]["key"], "text": "x",
                                                "evidence_refs": ["turn:999"]}]}, habits, allowed)


def _review(chat, tmp_path, **kwargs):
    kwargs.setdefault("reads_sources", reads_pack.Sources(entries=lambda: [], grades=lambda: []))
    return mentor_review.run_mentor_review(session_date=SESSION, now=NIGHT, chat_db=chat, ai_root=tmp_path / "ai",
                                           veto_outcomes=tmp_path / "veto.csv",
                                           journal_db=tmp_path / "no_journal.sqlite3", **kwargs)


def test_the_night_publishes_the_registry_and_a_facts_brief_with_habits(chat, tmp_path):
    out = _review(chat, tmp_path, ask=False)
    assert out["status"] == "ok", out["reason"]
    registry_file = json.loads((tmp_path / "ai" / "mentor_habits.json").read_text(encoding="utf-8"))
    assert registry_file["schema"] == "mentor_habits_v1" and registry_file["session_date"] == SESSION
    assert {h["key"] for h in registry_file["habits"]} >= {"tag:fomo", "phrase:chased the open"}
    brief = json.loads((tmp_path / "ai" / f"mentor_coach_brief_{SESSION}.json").read_text(encoding="utf-8"))
    assert brief["habits"] and brief["habits"][0]["evidence_refs"][0].startswith("habit:")
    # first_seen never moves later on a rerun with fewer days in view.
    registry_file["habits"][0]["first_seen"] = "2026-09-01"
    (tmp_path / "ai" / "mentor_habits.json").write_text(json.dumps(registry_file), encoding="utf-8")
    _review(chat, tmp_path, ask=False)
    again = json.loads((tmp_path / "ai" / "mentor_habits.json").read_text(encoding="utf-8"))
    assert again["habits"][0]["first_seen"] == "2026-09-01"


def test_the_night_words_habits_in_a_third_call_capped_at_300_tokens(chat, tmp_path):
    calls, sent = [], []

    def request(**kwargs):
        calls.append(kwargs)
        sent.append(kwargs["post"])
        if len(calls) == 1:
            return {"model": "m", "summary": {"digest": [{"text": "x", "evidence_refs": ["turn:1"]}],
                                              "open_questions": []}}
        if len(calls) == 2:
            return {"model": "m", "summary": {"watch": [], "missing": [], "issues": [],
                                              "one_line": {"text": "", "evidence_refs": []}}}
        return {"model": "m", "summary": {"habits": [{"key": "tag:fomo", "text": "You chase after stops.",
                                                      "evidence_refs": ["habit:tag_fomo", "journal:3"]}]}}

    posted = []
    out = _review(chat, tmp_path, request=request, post=lambda url, **kw: posted.append(kw["json"]["max_tokens"]))
    assert out["status"] == "ok", out["reason"]
    assert len(calls) == 3 and calls[2]["schema_name"] == "tradingbot_mentor_habits"
    calls[2]["post"]("u", json={})
    assert posted[-1] == mentor_habits.MAX_HABIT_TOKENS == 300
    brief = json.loads((tmp_path / "ai" / f"mentor_coach_brief_{SESSION}.json").read_text(encoding="utf-8"))
    assert brief["habits"][0] == {"key": "tag:fomo", "text": "You chase after stops.",
                                  "evidence_refs": ["habit:tag_fomo", "journal:3"], "days_seen": 3,
                                  "first_seen": "2026-09-25"}
    # The digest call saw every journal line of the day and the habits, with ids.
    night = calls[0]["evidence"]["night_reads"]
    assert [row["id"] for row in night["journal"]] == ["journal:3", "journal:4"]
    assert "habit:tag_fomo" in calls[0]["evidence"]["allowed_evidence_ids"]


def test_every_turn_rides_the_digest_not_twenty(tmp_path):
    path = tmp_path / "mentor_chat.sqlite3"
    store = MentorChatStore(path)
    session = store.start_session("m")
    for n in range(25):
        store.add_turn(session, "user", f"question {n}?")
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE turns SET ts_utc = ?", ("2026-09-29T17:00:00.000+00:00",))
    inputs = mentor_review.build_inputs(path, SESSION, mentor_review.day_facts(path, SESSION))
    assert len(inputs["turns"]) == 25
    many = [{"id": f"turn:{n}", "text": "x" * 1000} for n in range(20)]
    kept = mentor_review.budgeted(many, mentor_review.TURN_BUDGET_CHARS)
    assert [row["id"] for row in kept] == [f"turn:{n}" for n in range(11, 20)]  # the oldest dropped first


def test_a_journal_only_day_still_loads_the_model(tmp_path):
    path = tmp_path / "mentor_chat.sqlite3"
    store = MentorChatStore(path)
    _entry(store, SESSION, 10, "I'm tired")
    assert mentor_review.model_wanted(session_date=SESSION, chat_db=path) is True
    assert mentor_review.model_wanted(session_date="2026-09-28", chat_db=path) is False


def test_read_and_tape_disagreeing_two_sessions_running_is_an_issue(chat, tmp_path):
    grades = [{"entry_id": "mj-a", "session": "2026-09-26", "horizon": "rest_of_day", "source": "click",
               "verdict": "wrong"},
              {"entry_id": "mj-b", "session": SESSION, "horizon": "rest_of_day", "source": "click",
               "verdict": "wrong"}]
    out = _review(chat, tmp_path, ask=False, reads_sources=reads_pack.Sources(entries=lambda: [],
                                                                               grades=lambda: grades))
    assert out["status"] == "ok"
    brief = json.loads((tmp_path / "ai" / f"mentor_coach_brief_{SESSION}.json").read_text(encoding="utf-8"))
    issue = next(item for item in brief["issues"] if item["key"] == "read_vs_tape")
    assert "disagreed 2 sessions running (2026-09-26, 2026-09-29)" in issue["text"]
    assert reads_pack.disagreement_run([{**grades[0], "verdict": "right"}, grades[1]], SESSION) == []
    assert reads_pack.disagreement_run(grades, "2026-09-30") == []  # nothing graded tonight: no run


# ------------------------------------------------------------------ the app
def test_the_routine_table_rides_in_the_habits_pack(tmp_path):
    table = {"session_date": SESSION, "session_days": ["d"] * 10, "routines": [
        {"bucket": "06:30", "packs": [{"name": "regime_pack", "days": 7}]}]}
    path = tmp_path / "mentor_routines.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    src = habits_pack.fixture_sources(tmp_path)
    pack = habits_pack.build(sources=habits_pack.Sources(registry=src.registry, routines=lambda: path))
    row = next(r for r in pack.rows if r["id"] == "routine:0630")
    assert row["text"] == "Usual ask at 06:30 PT (last 10 session days): regime_pack on 7 days"
    only = habits_pack.build(sources=habits_pack.Sources(registry=lambda: tmp_path / "nope.json",
                                                         routines=lambda: path))
    assert [r["id"] for r in only.rows] == ["habits:none", "routine:0630"]
    names = [r.name for r in attach.plan_attachments("what do I usually ask at the open", set(), NIGHT)]
    assert "habits_pack" in names


def test_habits_pack_reads_the_nights_file_and_says_unknown_without_it(tmp_path, monkeypatch):
    assert "habits_pack" in registry.names()
    pack = habits_pack.fixture()
    assert pack.ids[0] == "habits:asof" and "habit:tag_fomo" in pack.ids and pack.ids[-1] == "habits:day"
    fomo = next(row for row in pack.rows if row["id"] == "habit:tag_fomo")
    assert "after a loss 4" in fomo["text"] and "journal:12" in fomo["text"] and "I chased TWLO again" in fomo["text"]
    missing = habits_pack.build(sources=habits_pack.Sources(registry=lambda: tmp_path / "nope.json"))
    assert not missing.rows and "not counted your habits yet; unknown" in missing.empty_text


def test_one_habit_inbox_item_a_week_only_for_a_red_day_habit():
    pack = habits_pack.fixture()
    now = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)
    week, line = habits_pack.inbox_line(pack, None, now)
    assert week == "2026-W40" and line.endswith("[habit:tag_fomo]")
    assert habits_pack.inbox_line(pack, "2026-W40", now) is None
    assert habits_pack.inbox_line(pack, "2026-W40", now + timedelta(days=6))[0] == "2026-W41"
    calm = habits_pack.make_pack("habits_pack", [{**row, "inbox_ok": False} for row in pack.rows])
    assert habits_pack.inbox_line(calm, None, now) is None


def test_habits_command_attach_words_and_the_brief_section_in_memory():
    assert commands.handle("/habits") == commands.CommandResult("pack", "", ("habits_pack", {}))
    assert commands.handle("/habits now").action == "error" and "/habits" in commands.HELP_TEXT
    now = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)
    for question in ("what are my bad habits", "what do I keep saying", "is there a pattern in what I say",
                     "when do I get frustrated", "what have I been feeling lately"):
        assert "habits_pack" in [r.name for r in attach.plan_attachments(question, set(), now)], question
    brief = {"session_date": SESSION, "one_line": {"text": "Lead [a:1]", "evidence_refs": ["a:1"]},
             "habits": [{"key": "tag:fomo", "text": "You chase after stops.", "evidence_refs": ["habit:tag_fomo"],
                         "first_seen": "2026-09-25"}]}
    texts = [item.text for item in memory.coach_items(brief)]
    assert "habit (since 2026-09-25): You chase after stops. (cites habit:tag_fomo)" in texts


@pytest.fixture
def window(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from mentor_app import settings
    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.window import MentorWindow

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    monkeypatch.setattr(settings, "context_tokens", lambda: 8192)
    win = MentorWindow(store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"), queue=PrefetchQueue(),
                       stream_post=lambda url, payload, cancelled: [], post=lambda url, payload, timeout: {},
                       habits_sources=habits_pack.fixture_sources(tmp_path),
                       now=lambda: datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc))
    yield win
    win.shutdown()
    win.deleteLater()


def test_the_window_posts_the_weeks_habit_item_once_and_habits_shows_the_card(window):
    from PySide6.QtWidgets import QApplication

    def settle():
        window._io.submit(lambda: None).result(5)
        QApplication.processEvents()
        window._io.submit(lambda: None).result(5)

    window._io.submit(window._check_habit_inbox).result(5)
    settle()
    posted = [item for item in window.inbox.items() if item.kind == "habits"]
    assert len(posted) == 1 and posted[0].text.endswith("[habit:tag_fomo]")
    assert window.store.get_state(habits_pack.INBOX_WEEK_KEY) == "2026-W40"
    window._io.submit(window._check_habit_inbox).result(5)
    settle()
    assert len([item for item in window.inbox.items() if item.kind == "habits"]) == 1
    card = window._pack_card("habits_pack", {})
    assert card.startswith("**Habits**") and "[habit:tag_fomo]" in card
