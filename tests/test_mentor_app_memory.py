"""Mentor app P4: morning memory (byte-stable block, budget), /memory /forget /keep /recall, and the
once-a-week "still true?" Inbox item."""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import memory  # noqa: E402
from mentor_app.chat_model import SYSTEM_PROMPT, ChatModel  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
NOW = datetime(2026, 9, 29, 10, 0, tzinfo=PT)  # a Tuesday session
OLD = "2026-09-15T17:00:00.000+00:00"  # 14 days before NOW


def _digest(root: Path, day: str, items: list[str], questions: list[str] = ()) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {"session_date": day, "digest": [{"text": text, "evidence_refs": ["fact:turns"]} for text in items],
               "open_questions": [{"text": text, "evidence_refs": ["fact:turns"]} for text in questions]}
    (root / f"mentor_day_digest_{day}.json").write_text(json.dumps(payload), encoding="utf-8")


def _note(store: MentorChatStore, text: str, stamp: str = OLD) -> int:
    note_id = store.add_profile_note(text)
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE profile_notes SET ts_utc = ? WHERE id = ?", (stamp, note_id))
    return int(note_id)


# ---------------------------------------------------------------- the block
def test_the_memory_block_is_byte_stable_and_sits_between_rules_and_context(tmp_path):
    root = tmp_path / "ai"
    _digest(root, "2026-09-28", ["Asked about NVDA twice."], ["Is two losses still the stop?"])
    store = MentorChatStore(tmp_path / "chat.sqlite3")
    _note(store, "rule: I stop after two losses")
    one = memory.load(store, ai_root=root)
    two = memory.load(store, ai_root=root)
    assert one.text == two.text and one.text.encode() == two.text.encode()
    assert "[mem:digest:2026092800] (2026-09-28) Asked about NVDA twice." in one.text
    assert "[mem:digest:2026092850] (2026-09-28) open question: Is two losses still the stop?" in one.text
    assert "[mem:note:1] (2026-09-15) rule: I stop after two losses" in one.text
    body = ChatModel.system_message("[ctx:auto_mode] DESK", one.text)["content"]
    assert body.startswith(SYSTEM_PROMPT) and body.index("# Memory") < body.index("# Desk context")
    first = ChatModel().messages(context_text="[ctx:a] x", memory_block=one.text)[0]
    assert first == ChatModel().messages(context_text="[ctx:a] x", memory_block=two.text)[0]


def test_the_budget_drops_the_oldest_first_and_is_deterministic():
    items = [memory.MemoryItem(f"mem:note:{i}", "note", i, f"2026-09-{i:02d}", "x" * 200) for i in range(1, 21)]
    one = memory.render(items, budget_tokens=300)
    two = memory.render(list(reversed(items)), budget_tokens=300)
    assert one.text == two.text and one.dropped == two.dropped > 0
    kept = [item.ref_id for item in one.items]
    assert kept == sorted(kept) and kept[-1] == 20 and 1 not in kept, "newest kept, oldest dropped"
    assert memory.estimate_tokens(one.text) <= 300


def test_only_the_last_five_digests_are_loaded(tmp_path):
    root = tmp_path / "ai"
    for day in range(20, 28):
        _digest(root, f"2026-09-{day}", [f"day {day}"])
    loaded = memory.load(MentorChatStore(tmp_path / "chat.sqlite3"), ai_root=root)
    assert [item.day for item in loaded.items] == [f"2026-09-{day}" for day in range(23, 28)]


def test_a_night_digest_becomes_morning_memory(tmp_path):
    from ai_jobs import mentor_review

    store = MentorChatStore(tmp_path / "chat.sqlite3")
    store.add_turn(store.start_session(), "user", "NVDA?")
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE turns SET ts_utc = '2026-09-29T17:00:00.000+00:00'")
    reply = {"digest": [{"text": "He asked about NVDA.", "evidence_refs": ["turn:1"]}], "open_questions": []}
    mentor_review.run_mentor_review(session_date="2026-09-29", now=datetime(2026, 9, 29, 23, 0, tzinfo=PT),
                                    chat_db=store.path, ai_root=tmp_path / "ai",
                                    request=lambda **_: {"summary": reply, "model": "m"})
    loaded = memory.load(store, ai_root=tmp_path / "ai")
    assert "[mem:digest:2026092900] (2026-09-29) He asked about NVDA." in loaded.text


def test_still_true_candidate_rules():
    rows = [
        {"id": 1, "text": "I like NVDA", "ts_utc": OLD},
        {"id": 2, "text": "rule: two losses and I stop", "ts_utc": OLD, "retired_utc": "x"},
        {"id": 3, "text": "Rule: no trades before 06:45", "ts_utc": OLD},
        {"id": 4, "text": "rule: fresh one", "ts_utc": "2026-09-28T17:00:00+00:00"},
    ]
    assert memory.still_true_candidate(rows, NOW)["id"] == 3
    rows[2]["asked_utc"] = "2026-09-27T17:00:00+00:00"
    assert memory.still_true_candidate(rows, NOW) is None, "asked within 7 days"
    rows[2]["asked_utc"] = None
    rows[2]["checked_utc"] = "2026-09-26T17:00:00+00:00"
    assert memory.still_true_candidate(rows, NOW) is None, "/keep restarted its age"


# ---------------------------------------------------------------- the window
@pytest.fixture(scope="module")
def app():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture()
def window(app, tmp_path, monkeypatch):
    from mentor_app import settings
    from mentor_app.inbox import Inbox
    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.window import MentorWindow

    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    monkeypatch.setattr(settings, "context_tokens", lambda: 8192)
    clock = {"now": NOW}
    win = MentorWindow(
        store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"),
        queue=PrefetchQueue(),
        inbox=Inbox(per_day_cap=6, now=lambda: clock["now"]),
        stream_post=lambda url, payload, cancelled: [],
        post=lambda url, payload, timeout: {},
        now=lambda: clock["now"],
        mentor_enabled=False,
        liked_source=lambda: [],
        memory_root=tmp_path / "ai",
    )
    win.clock = clock
    yield win
    win.shutdown()
    win.deleteLater()


def _drain(win, app):
    for _ in range(50):
        win._io.submit(lambda: None).result(5)
        app.processEvents()
        if not win.queue.run_one():
            app.processEvents()
            if not win.queue.pending():
                break
    win._io.submit(lambda: None).result(5)
    app.processEvents()


def _text(win):
    return win.transcript.toPlainText()


def test_forget_retires_a_note_and_never_deletes_it(window, app):
    note_id = _note(window.store, "rule: I stop after two losses")
    window.send(f"/forget mem:note:{note_id}")
    _drain(window, app)
    rows = window.store.profile_notes(include_retired=True)
    assert len(rows) == 1 and rows[0]["retired_utc"], "kept, marked retired"
    assert window.store.profile_notes() == []
    assert "Retired [mem:note:1]" in _text(window)


def test_memory_command_lists_what_was_loaded_with_ids(window, app):
    _note(window.store, "rule: I stop after two losses")
    _digest(window._memory_root, "2026-09-28", ["Asked about NVDA twice."])
    window._submit_io(window._open_session)
    _drain(window, app)
    window.send("/memory")
    text = _text(window)
    assert "[mem:note:1]" in text and "[mem:digest:2026092800]" in text
    assert "# Memory" in window._memory_block


def test_recall_answers_by_substring_with_the_brain_off(window, app):
    from mentor_packs import recall

    window.install_recall_fallback()  # what start_background does; the brain stays off
    _note(window.store, "I stop trading after two losses")
    recall.set_searcher(None)
    window.send("/recall losses")
    _drain(window, app)
    assert "[mem:note:1]" in _text(window)


def test_still_true_posts_once_per_note_per_seven_days(window, app):
    _note(window.store, "rule: I stop after two losses")
    window.send("/help")
    _drain(window, app)
    items = [item for item in window.inbox.items() if item.kind == "memory"]
    assert len(items) == 1 and items[0].text.startswith("You said: rule: I stop after two losses. Still true?")
    assert "(/keep 1 | /forget 1)" in items[0].text
    assert window.store.profile_notes()[0]["asked_utc"]
    window.send("/help")  # same day: not again
    window.clock["now"] = NOW + timedelta(days=1)  # Wednesday: asked yesterday
    window.send("/help")
    _drain(window, app)
    assert len([item for item in window.inbox.items() if item.kind == "memory"]) == 1
    window.clock["now"] = NOW + timedelta(days=8)  # the next Wednesday: a week later, ask again
    window.send("/help")
    _drain(window, app)
    assert len([item for item in window.inbox.items() if item.kind == "memory"]) == 2


def test_still_true_never_posts_when_the_cap_is_used(window, app):
    _note(window.store, "rule: I stop after two losses")
    window.inbox.per_day_cap = 0
    window.send("/help")
    _drain(window, app)
    assert window.inbox.items() == []
    assert window.store.profile_notes()[0]["asked_utc"] is None, "not asked, so it may ask another day"


def test_keep_refreshes_the_note(window, app):
    _note(window.store, "rule: I stop after two losses")
    window.send("/keep 1")
    _drain(window, app)
    assert window.store.profile_notes()[0]["checked_utc"]
    assert "Still true: [mem:note:1]" in _text(window)


@pytest.mark.parametrize("wall", ["2026-01-05T12:00:00.000+00:00", "2027-06-01T12:00:00.000+00:00"])
def test_still_true_uses_the_windows_clock_on_any_date(window, app, monkeypatch, wall):
    from mentor_app import store as store_module

    monkeypatch.setattr(store_module, "utc_now", lambda: wall)
    _note(window.store, "rule: I stop after two losses")
    window.send("/help")
    _drain(window, app)
    assert window.store.profile_notes()[0]["asked_utc"].startswith("2026-09-29")
    window.clock["now"] = NOW + timedelta(days=8)
    window.send("/help")
    _drain(window, app)
    assert len([item for item in window.inbox.items() if item.kind == "memory"]) == 2


# ---------------------------------------------------------------- P15a: the memory stands on the night
COACH = {"schema": "mentor_coach_brief_v1", "session_date": "2026-09-28", "worded": True,
         "one_line": {"text": "Bounces in a bear channel have been early for you.",
                      "evidence_refs": ["night:day_review:2026-09-28:2"]},
         "watch": [{"text": "SPY at its 50 SMA", "evidence_refs": ["night:story:2026-09-28:0"]}],
         "missing": [{"text": "Trendline vetoes ran 21% of the time", "evidence_refs": ["night:miss:2026-09-28:1"]}],
         "issues": [{"key": "wrong_reads", "text": "Two reads graded wrong", "first_seen": "2026-09-24", "count": 3,
                     "evidence_refs": ["night:day_review:2026-09-28:2"]}]}


def _night_world(tmp_path):
    from mentor_packs import night_pack

    world = night_pack.write_fixture_world(tmp_path / "night")
    root = tmp_path / "ai"
    _digest(root, "2026-09-28", ["Asked about NVDA twice."])
    (root / "mentor_coach_brief_2026-09-28.json").write_text(json.dumps(COACH), encoding="utf-8")
    return world, root


def test_the_memory_loads_the_night_in_priority_order_with_its_ids(tmp_path):
    world, root = _night_world(tmp_path)
    store = MentorChatStore(tmp_path / "chat.sqlite3")
    _note(store, "rule: I stop after two losses")
    one = memory.load(store, ai_root=root, night_paths=world, now=NOW)
    two = memory.load(store, ai_root=root, night_paths=world, now=NOW)
    assert one.text.encode() == two.text.encode(), "byte-stable"
    ids = [item.id for item in one.items]
    tiers = [item.priority for item in one.items]
    assert tiers == sorted(tiers), "coach brief, digest, ideas, day review, week, notes"
    assert ids[0] == "night:coach:2026-09-28:0" and "night:coach:2026-09-28:i1" in ids
    assert ids.index("mem:digest:2026092800") < ids.index("night:ideas:2026-09-29:1")
    assert len([i for i in ids if i.startswith("night:ideas:")]) == memory.IDEA_LIMIT
    assert "night:day_review:2026-09-29:2" in ids and ids[-1] == "mem:note:1"
    assert len([i for i in ids if i.startswith("night:week:")]) == memory.WEEK_LINES
    assert "[night:day_review:2026-09-29:2] (2026-09-29) You said" in one.text
    assert "recurring issue (since 2026-09-24): Two reads graded wrong" in one.text
    assert memory.MEMORY_BUDGET_TOKENS == 3000 and memory.estimate_tokens(one.text) <= 3000


def test_the_budget_drops_the_lowest_tier_first_then_the_oldest():
    night = [memory.MemoryItem(f"night:ideas:2026-09-{i:02d}:1", "night", i, f"2026-09-{i:02d}", "y" * 200,
                               memory.PRIORITY_IDEA) for i in range(1, 6)]
    notes = [memory.MemoryItem(f"mem:note:{i}", "note", i, "2026-09-29", "x" * 200) for i in range(1, 6)]
    kept = memory.render([*notes, *night], budget_tokens=400)
    ids = [item.id for item in kept.items]
    assert all(item.id in ids for item in night), "the notes tier goes before any idea"
    left = [i for i in ids if i.startswith("mem:note:")]
    assert left == [f"mem:note:{i}" for i in range(6 - len(left), 6)] and len(left) < 5, "oldest notes go first"
    assert kept.dropped and memory.estimate_tokens(kept.text) <= 400
    tight = memory.render([*notes, *night], budget_tokens=200)
    assert all(not i.id.startswith("mem:note:") for i in tight.items) and tight.items[-1].id.endswith("09-05:1")


def test_no_night_on_file_leaves_digests_and_notes_as_before(tmp_path):
    from mentor_packs import night_pack

    root = tmp_path / "ai"
    _digest(root, "2026-09-28", ["Asked about NVDA twice."])
    loaded = memory.load(MentorChatStore(tmp_path / "chat.sqlite3"), ai_root=root,
                         night_paths=night_pack.NightPaths(), now=NOW)
    assert [item.id for item in loaded.items] == ["mem:digest:2026092800"], "a none row is never memory"


@pytest.mark.parametrize("one_line", [{"text": "Trust your gut.", "evidence_refs": []}, "Trust your gut.", None])
def test_without_a_cited_one_line_the_first_watch_item_leads(tmp_path, one_line):
    """Review advisory 3: uncited model text never leads the memory; the first watch item does, once."""
    root = tmp_path / "ai"
    root.mkdir(parents=True)
    payload = {**COACH, "one_line": one_line}
    (root / "mentor_coach_brief_2026-09-28.json").write_text(json.dumps(payload), encoding="utf-8")
    from mentor_packs import night_pack

    loaded = memory.load(MentorChatStore(tmp_path / "c.sqlite3"), ai_root=root,
                         night_paths=night_pack.NightPaths(), now=NOW)
    assert loaded.items[0].id == "night:coach:2026-09-28:0"
    assert loaded.items[0].text == "coach brief: watch: SPY at its 50 SMA (cites night:story:2026-09-28:0)"
    assert "Trust your gut" not in loaded.text
    assert "night:coach:2026-09-28:w1" not in [item.id for item in loaded.items], "not shown twice"


def test_a_coach_brief_dated_after_today_is_not_loaded(tmp_path):
    world, root = _night_world(tmp_path)
    (root / "mentor_coach_brief_2026-09-30.json").write_text(json.dumps({**COACH, "session_date": "2026-09-30"}),
                                                            encoding="utf-8")
    ids = [item.id for item in memory.load(MentorChatStore(tmp_path / "c.sqlite3"), ai_root=root,
                                           night_paths=world, now=NOW).items]
    assert "night:coach:2026-09-28:0" in ids and not any("2026-09-30" in i for i in ids)


def test_embed_candidates_cover_night_rows_and_briefs_once_per_version(tmp_path):
    from mentor_packs import night_pack

    world, root = _night_world(tmp_path)
    loaded = memory.load(MentorChatStore(tmp_path / "c.sqlite3"), ai_root=root, night_paths=world, now=NOW)
    first = memory.embed_candidates(loaded, world, NOW)
    assert {kind for kind, _ref, _text in first} == {"night", "brief"}
    assert any(text.startswith("[night:ideas:2026-09-29:1] ") for _k, _r, text in first)
    assert any(text.startswith("[brief:NVDA:2026-09-29] ") for _k, _r, text in first)
    assert first == memory.embed_candidates(loaded, world, NOW), "same artifacts, same refs"
    (world.day_review / "narration" / "2026-09-29.json").write_text(json.dumps({"narration": {
        "headline": "Rewritten.", "were_you_right": []}}), encoding="utf-8")
    again = {ref for _k, ref, _t in memory.embed_candidates(loaded, world, NOW)}
    changed = night_pack.stable_ref("night:day_review:2026-09-29:0",
                                    "Day review 2026-09-29: Rewritten. (src: day_review:2026-09-29)")
    assert changed in again, "a changed artifact is a new ref, embedded once more"


def test_recall_cites_a_night_hit_by_its_own_id():
    from mentor_packs import recall

    stored = [{"kind": "night", "ref_id": 7, "vector": [1.0, 0.0],
               "text": "[night:ideas:2026-09-29:1] Idea (process): ask why liked names ran."},
              {"kind": "brief", "ref_id": 8, "vector": [0.9, 0.1], "text": "[brief:NVDA:2026-09-29] NVDA held."},
              {"kind": "note", "ref_id": 2, "vector": [0.0, 1.0], "text": "[x:y] a note"}]
    searcher = recall.make_searcher(lambda texts: [[1.0, 0.0] for _ in texts], lambda: stored)
    ids = recall.build("liked names", searcher=searcher).ids
    assert ids == ("night:ideas:2026-09-29:1", "brief:NVDA:2026-09-29", "mem:note:2")


def test_memory_brief_and_issues_commands_show_the_night(window, app, tmp_path):
    from mentor_packs import night_pack

    window.night_paths = night_pack.write_fixture_world(tmp_path / "night")
    window._memory_root.mkdir(parents=True, exist_ok=True)
    (window._memory_root / "mentor_coach_brief_2026-09-28.json").write_text(json.dumps(COACH), encoding="utf-8")
    window._submit_io(window._open_session)
    _drain(window, app)
    window.send("/memory")
    window.send("/brief")
    window.send("/issues")
    _drain(window, app)
    text = _text(window)
    assert "[night:coach:2026-09-28:0]" in text and "[night:day_review:2026-09-29:1]" in text
    assert "Coach brief from the night of 2026-09-28" in text
    assert "you may be missing: Trendline vetoes ran 21% of the time" in text
    assert "first seen 2026-09-24, 3 times: Two reads graded wrong" in text


def test_brief_with_no_night_says_so(window, app):
    window.send("/brief")
    window.send("/issues")
    _drain(window, app)
    text = _text(window)
    assert "No coach brief from the night yet" in text and "No recurring issues" in text


def test_the_idle_embed_queue_adds_night_and_brief_kinds_once(window, app, tmp_path):
    from mentor_app import settings
    from mentor_packs import night_pack

    calls = []

    def post(url, payload, timeout):
        calls.append(url)
        return {"embeddings": [[1.0, 0.0]]} if url.endswith("/api/embed") else {}

    window._post = post
    window._endpoint = "http://h"
    window.night_paths = night_pack.write_fixture_world(tmp_path / "night")
    window._queue_memory_embeddings()
    _drain(window, app)
    assert window.store.embedded_refs("night", settings.EMBED_MODEL)
    assert window.store.embedded_refs("brief", settings.EMBED_MODEL)
    first = len(calls)
    assert first
    window._queue_memory_embeddings()
    _drain(window, app)
    assert len(calls) == first, "nothing new: every night row and brief is embedded once per version"
