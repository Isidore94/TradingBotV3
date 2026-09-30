"""Mentor app P4: the night's `mentor_review` slot - registration, the facts half, the cited digest,
and the write watch (the only chat-DB write is a challenge's grade)."""

from __future__ import annotations

import dataclasses
import functools
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ai_jobs import mentor_review  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
ET = ZoneInfo("America/New_York")
SESSION = "2026-09-29"
#: 23:00 PT Tuesday = 02:00 ET Wednesday: inside the night, processing Tuesday.
NIGHT = datetime(2026, 9, 29, 23, 0, tzinfo=PT)
DAY_STAMP = "2026-09-29T17:00:00.000+00:00"  # 10:00 PT on the session


@pytest.fixture
def chat(tmp_path):
    path = tmp_path / "mentor_chat.sqlite3"
    store = MentorChatStore(path)
    session = store.start_session("m")
    store.add_turn(session, "user", "Is NVDA still worth it?")
    store.add_turn(session, "assistant", "NVDA held its AVWAP [pick:NVDA:cell].", latency_ms=1200,
                   tool_calls=[{"name": "pick_pack", "arguments": {"symbol": "NVDA"}}])
    store.add_turn(session, "assistant", "Two vetoes today.", latency_ms=2400,
                   tool_calls=[{"name": "veto_pack", "arguments": {}}, {"name": "pick_pack", "arguments": {}}])
    store.add_profile_note("rule: I stop after two losses")
    store.add_challenge("veto:2026-09-29:AAA:1", kind="veto", symbol="AAA", claim="vetoes like it won",
                        issued_utc=DAY_STAMP, outcome={"status": "open", "session": SESSION,
                                                       "session_date": SESSION, "side": "LONG"})
    store.put_pack("pick_assessment", {"symbol": "NVDA", "hash": "h"}, "{}", DAY_STAMP)
    store.bump_day_stats(SESSION, uncited_numbers=2, numbers=10, brain_offline_min=3)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE turns SET ts_utc = ?", (DAY_STAMP,))
        conn.execute("UPDATE profile_notes SET ts_utc = ?", (DAY_STAMP,))
    return path


def _dump(path: Path) -> dict:
    """Every table's rows, minus the two grading columns the night owns."""
    out = {}
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        out["schema"] = sorted(tuple(row) for row in conn.execute("SELECT type, name, sql FROM sqlite_master"))
        for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall():
            rows = []
            for row in conn.execute(f"SELECT * FROM {name}").fetchall():
                item = dict(row)
                if name == "challenges":
                    item.pop("outcome_json")
                    item.pop("graded_utc")
                rows.append(sorted(item.items(), key=lambda kv: kv[0]))
            out[name] = sorted(map(repr, rows))
    return out


def _run(chat, tmp_path, **kwargs):
    return mentor_review.run_mentor_review(
        session_date=SESSION, now=NIGHT, chat_db=chat, ai_root=tmp_path / "ai",
        veto_outcomes=tmp_path / "veto_cohort_outcomes.csv", **kwargs,
    )


# ---------------------------------------------------------------- registration
def test_mentor_review_is_registered_after_ticker_briefs_and_cut_first():
    from ai_jobs import runner

    slots = runner.default_slots()
    names = [slot.name for slot in slots]
    assert names.index("mentor_review") == names.index("ticker_briefs") + 1
    slot = next(slot for slot in slots if slot.name == "mentor_review")
    assert slot.goal == "journal" and slot.goal in runner.SLOT_GOALS
    assert slot.uses_model and slot.model_free_kwargs == {"ask": False}
    ranked = sorted(["mentor_review", "ticker_briefs", "observation_tags", "econ_brief", "daily_digest"],
                    key=runner.model_slot_priority)
    assert ranked[-1] == "mentor_review", "the first model slot the budget cuts"
    assert "mentor_review" not in runner.WEEKEND_ONLY_SLOTS


# ---------------------------------------------------------------- the deterministic half
def test_facts_half_grades_and_publishes_facts_without_a_model(chat, tmp_path):
    def no_model(**_):
        raise AssertionError("ask=False must not call the model")

    out = _run(chat, tmp_path, ask=False, request=no_model)
    assert out["status"] == "ok" and "no model asked" in out["reason"]
    facts = json.loads((tmp_path / "ai" / f"mentor_day_facts_{SESSION}.json").read_text(encoding="utf-8"))
    assert facts["turns"] == {"user": 1, "assistant": 2}
    assert facts["tool_calls"] == {"pick_pack": 2, "veto_pack": 1}
    assert facts["picks_assessed"] == 1 and facts["remember_notes"] == 1
    assert facts["challenges"]["issued"] == 1
    assert facts["uncited_numbers"] == 2 and facts["numbers"] == 10 and facts["brain_offline_min"] == 3
    assert facts["first_token_ms"]["n"] == 2 and facts["first_token_ms"]["p95"] == 2400
    assert facts["grading"] == {"owner": "night", "updated": 1}
    row = MentorChatStore(chat).challenges(kind="veto")[0]
    assert json.loads(row["outcome_json"])["reason"] == "no veto cohort row yet", "graded by the night"
    assert not (tmp_path / "ai" / f"mentor_day_digest_{SESSION}.json").exists()


def test_the_night_does_not_grade_outside_its_window(chat, tmp_path):
    out = mentor_review.run_mentor_review(session_date=SESSION, now=datetime(2026, 9, 29, 12, 0, tzinfo=PT),
                                          chat_db=chat, ai_root=tmp_path / "ai", ask=False)
    assert out["status"] == "ok"
    row = MentorChatStore(chat).challenges(kind="veto")[0]
    assert "reason" not in json.loads(row["outcome_json"]), "the app owns grading by day"


def test_the_slot_writes_only_the_grading_columns(chat, tmp_path):
    before = _dump(chat)
    reply = {"digest": [{"text": "Asked about NVDA once.", "evidence_refs": ["fact:turns"]}], "open_questions": []}
    _run(chat, tmp_path, request=lambda **_: {"summary": reply, "model": "m"})
    assert _dump(chat) == before, "the night wrote the chat DB beyond the grading columns"


def test_a_failed_probe_runs_the_facts_half_and_the_ledger_row_carries_goal(chat, tmp_path, monkeypatch):
    from ai_jobs import runner, store, window

    monkeypatch.setattr(store, "store_available", lambda: (True, "ready"))
    monkeypatch.setattr(window, "launch_allowed", lambda *a, **k: (True, "window open"))
    monkeypatch.setattr(window, "market_session_block", lambda *a, **k: "")
    real = next(slot for slot in runner.default_slots() if slot.name == "mentor_review")
    slot = dataclasses.replace(
        real,
        run=functools.partial(mentor_review.run_mentor_review, chat_db=chat, ai_root=tmp_path / "ai",
                              veto_outcomes=tmp_path / "veto.csv", request=lambda **_: pytest.fail("model")),
        model_wanted=lambda **_: True,
    )
    led = tmp_path / "ledger.jsonl"
    runner.run_slots([slot], now=NIGHT.astimezone(ET), ledger_path=led, probe=lambda: (False, "down"))
    rows = [json.loads(line) for line in led.read_text(encoding="utf-8").splitlines() if line]
    row = next(r for r in rows if r["job"] == "mentor_review")
    assert row["goal"] == "journal" and row["status"] == "degraded_no_narrative"
    assert row["turns"] == {"user": 1, "assistant": 2}
    assert (tmp_path / "ai" / f"mentor_day_facts_{SESSION}.json").exists()


# ---------------------------------------------------------------- the model half
def test_the_digest_drops_uncited_items_and_publishes(chat, tmp_path):
    seen = {}

    def request(**kwargs):
        seen.update(kwargs)
        return {"model": "m", "summary": {
            "digest": [
                {"text": "He asked about NVDA and read the pick pack.", "evidence_refs": ["turn:1", "fact:tools"]},
                {"text": "An uncited thought.", "evidence_refs": []},
            ],
            "open_questions": [{"text": "Is two losses still the stop?", "evidence_refs": ["note:1"]}],
        }}

    out = _run(chat, tmp_path, request=request)
    assert out["status"] == "ok", out["reason"]
    assert "fact:turns" in seen["evidence"]["allowed_evidence_ids"]
    assert seen["schema"] is mentor_review.DIGEST_JSON_SCHEMA
    digest = json.loads((tmp_path / "ai" / f"mentor_day_digest_{SESSION}.json").read_text(encoding="utf-8"))
    assert [item["text"] for item in digest["digest"]] == ["He asked about NVDA and read the pick pack."]
    assert digest["open_questions"][0]["evidence_refs"] == ["note:1"] and digest["dropped"] == 1


def test_a_foreign_id_rejects_the_reply_and_keeps_the_last_good_digest(chat, tmp_path):
    good = {"digest": [{"text": "Kept.", "evidence_refs": ["fact:turns"]}], "open_questions": []}
    _run(chat, tmp_path, request=lambda **_: {"summary": good, "model": "m"})
    path = tmp_path / "ai" / f"mentor_day_digest_{SESSION}.json"
    before = path.read_text(encoding="utf-8")
    bad = {"digest": [{"text": "Invented.", "evidence_refs": ["pick:TSLA:cell"]}], "open_questions": []}
    out = _run(chat, tmp_path, request=lambda **_: {"summary": bad, "model": "m"}, force=True)
    assert out["status"] == "failed" and "rejected" in out["reason"]
    assert path.read_text(encoding="utf-8") == before


def test_a_model_that_fails_leaves_the_facts_and_is_degraded(chat, tmp_path):
    def down(**_):
        raise ConnectionError("no host")

    out = _run(chat, tmp_path, request=down)
    assert out["status"] == "degraded_no_narrative"
    assert (tmp_path / "ai" / f"mentor_day_facts_{SESSION}.json").exists()


def test_no_chat_store_is_a_skip(tmp_path):
    out = mentor_review.run_mentor_review(session_date=SESSION, now=NIGHT, chat_db=tmp_path / "none.sqlite3",
                                          ai_root=tmp_path / "ai")
    assert out["status"] == "skipped" and not (tmp_path / "none.sqlite3").exists()
