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
def test_mentor_review_is_registered_after_exit_note_fields_and_ranked_after_the_day_review():
    # Was directly after `ticker_briefs`; the briefs moved last on 2026-09-30 (trader).
    # P15a (trader 2026-09-30): no longer cut first; the budget ranks it right after the day review
    # (story, then show). Priority changed only: the run order below is unchanged.
    from ai_jobs import runner

    slots = runner.default_slots()
    names = [slot.name for slot in slots]
    assert names.index("mentor_review") == names.index("exit_note_fields") + 1
    assert names.index("mentor_review") < names.index("ticker_briefs")
    slot = next(slot for slot in slots if slot.name == "mentor_review")
    assert slot.goal == "journal" and slot.goal in runner.SLOT_GOALS
    assert slot.uses_model and slot.model_free_kwargs == {"ask": False}
    ranked = sorted(["mentor_review", "ticker_briefs", "observation_tags", "econ_brief", "daily_digest",
                     "day_review_narration", "day_review_show", "market_story_narration"],
                    key=runner.model_slot_priority)
    assert ranked == ["daily_digest", "day_review_narration", "day_review_show", "mentor_review",
                      "market_story_narration", "observation_tags", "econ_brief", "ticker_briefs"]
    assert "mentor_review" not in runner.CUT_FIRST_SLOTS
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
    assert facts["grading"] == {"owner": "night", "graded": 0, "updated": 1}
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
    calls = []

    def request(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:  # P15a: the second call is the coach brief; the digest is the first
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
    assert [call["schema"] for call in calls] == [mentor_review.DIGEST_JSON_SCHEMA, mentor_review.BRIEF_JSON_SCHEMA]
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


def test_the_model_call_is_capped_at_2500_output_tokens(monkeypatch):
    import ai_summary

    monkeypatch.setattr(ai_summary, "local_reasoning_tokens", lambda: 8000)
    sent = {}
    wrapped = mentor_review.capped_post(lambda url, **kw: sent.update(kw["json"]), model="gpt-oss:20b")
    wrapped("http://h/v1/chat/completions", json={"max_tokens": 4000})
    # A thinking tag: the 2500-token answer cap plus the reasoning allowance, since its reasoning
    # counts against max_tokens (gemma4:12b under a bare 600 cap answered nothing, 2026-09-30).
    assert sent == {"max_tokens": 10500, "reasoning_effort": "high"}
    sent.clear()
    # gemma4 thinks unless told not to; the night tells it not to (it spent 11.5k tokens reasoning
    # and answered nothing in 8 of 8 night calls, 2026-10-01).
    mentor_review.capped_post(lambda url, **kw: sent.update(kw["json"]), model="gemma4:12b")("u", json={})
    assert sent == {"max_tokens": 2500, "reasoning_effort": "none"}
    sent.clear()
    mentor_review.capped_post(lambda url, **kw: sent.update(kw["json"]), model="gemma3:12b")("u", json={})
    assert sent == {"max_tokens": 2500}


def test_no_chat_store_is_a_skip(tmp_path):
    out = mentor_review.run_mentor_review(session_date=SESSION, now=NIGHT, chat_db=tmp_path / "none.sqlite3",
                                          ai_root=tmp_path / "ai")
    assert out["status"] == "skipped" and not (tmp_path / "none.sqlite3").exists()


def test_the_ledger_reason_counts_graded_and_updated_apart(chat, tmp_path, monkeypatch):
    """One challenge fully matures (graded), one only gains a horizon (updated): '1 graded, 2 updated'."""
    import annotations_reader

    MentorChatStore(chat).add_challenge(
        "veto:2026-09-29:BBB:1", kind="veto", symbol="BBB", claim="vetoes like it won", issued_utc=DAY_STAMP,
        outcome={"status": "open", "session": SESSION, "session_date": SESSION, "side": "LONG"})
    done = {h: ("2026-09-01", 0.01) for h in (1, 3, 5, 10)}
    part = {1: ("2026-09-01", 0.01), 10: ("2026-12-31", 0.02)}
    monkeypatch.setattr(annotations_reader, "veto_forward_returns",
                        lambda _path: {(SESSION, "AAA", "LONG"): done, (SESSION, "BBB", "LONG"): part})
    out = _run(chat, tmp_path, ask=False)
    assert "1 graded, 2 updated" in out["reason"]
    assert out["extra"]["graded"] == 1 and out["extra"]["updated"] == 2
    graded = [row for row in MentorChatStore(chat).challenges(kind="veto") if row.get("graded_utc")]
    assert len(graded) == out["extra"]["graded"]


# ---------------------------------------------------------------- P11 hypotheses
def _perm_history(tmp_path):
    from mentor_packs import hypothesis_pack

    folder = tmp_path / "permutation_report_history"
    folder.mkdir(exist_ok=True)
    (folder / "2026-09-26.json").write_text(json.dumps(hypothesis_pack.fixture_report()), encoding="utf-8")
    return folder


def _hyp_reply(*hypotheses):
    return {"digest": [{"text": "Asked about NVDA once.", "evidence_refs": ["fact:turns"]}], "open_questions": [],
            "hypotheses": list(hypotheses)}


HYP = {"query": {"population": "swing", "horizon": "5", "family": "avwap_breakout", "side": "LONG",
                 "facets": ["sma100_support=held", "spy_trend=up"]},
       "why": "He keeps liking SMA100 holds in an up tape.", "evidence_refs": ["turn:1", "hyp:report:asof"]}


def test_hypotheses_are_looked_up_recorded_and_published_as_hyp_lines(chat, tmp_path):
    seen = {}
    miss = {**HYP, "query": {**HYP["query"], "facets": ["made_up=yes"]}}

    def request(**kwargs):
        seen.update(kwargs)
        return {"summary": _hyp_reply(HYP, miss), "model": "m"}

    before = _dump(chat)
    out = _run(chat, tmp_path, request=request, permutation_history=_perm_history(tmp_path),
               permutation_report=tmp_path / "none.json")
    assert out["status"] == "ok" and out["extra"]["hypotheses"] == 2 and out["extra"]["hypotheses_recorded"] == 2
    evidence = seen["evidence"]
    assert "hyp:report:asof" in evidence["allowed_evidence_ids"]
    assert evidence["hypothesis_vocabulary"]["swing"]["facets"]["spy_trend"] == ["up"]
    assert set(mentor_review.DIGEST_JSON_SCHEMA["properties"]["hypotheses"]["items"]["properties"]) == {
        "query", "why", "evidence_refs"}
    digest = json.loads((tmp_path / "ai" / f"mentor_day_digest_{SESSION}.json").read_text(encoding="utf-8"))
    assert digest["permutation_report_asof"] == "2026-09-26"
    lines = digest["hyp_lines"]
    assert lines[0].startswith(f"[hyp:{SESSION}:1] swing avwap_breakout LONG 5") and "n=64" in lines[0]
    assert "not in vocabulary: facet made_up" in lines[1]
    rows = {row["id"]: row for row in MentorChatStore(chat).challenges(kind="hypothesis")}
    assert set(rows) == {f"hyp:{SESSION}:1", f"hyp:{SESSION}:2"}
    first = json.loads(rows[f"hyp:{SESSION}:1"]["outcome_json"])
    assert first["status"] == "open" and first["lookup"]["cell"]["n"] == 64
    assert first["lookup"]["report_asof"] == "2026-09-26" and rows[f"hyp:{SESSION}:1"]["graded_utc"] is None
    assert datetime.fromisoformat(rows[f"hyp:{SESSION}:1"]["issued_utc"]).tzinfo is not None
    after = _dump(chat)
    assert {k: v for k, v in after.items() if k != "challenges"} == {k: v for k, v in before.items()
                                                                       if k != "challenges"}
    assert set(before["challenges"]) < set(after["challenges"]), "only hypothesis rows were added"


def test_a_hypothesis_citing_a_foreign_id_rejects_the_reply(chat, tmp_path):
    bad = {**HYP, "evidence_refs": ["pick:TSLA:cell"]}
    out = _run(chat, tmp_path, request=lambda **_: {"summary": _hyp_reply(bad), "model": "m"},
               permutation_history=_perm_history(tmp_path), permutation_report=tmp_path / "none.json")
    assert out["status"] == "failed" and "hypotheses 0 cited" in out["reason"]
    assert MentorChatStore(chat).challenges(kind="hypothesis") == []


def test_uncited_or_extra_hypotheses_are_dropped(chat, tmp_path):
    uncited = {**HYP, "evidence_refs": []}
    out = _run(chat, tmp_path, request=lambda **_: {"summary": _hyp_reply(uncited, HYP, HYP, HYP, HYP),
                                                     "model": "m"},
               permutation_history=_perm_history(tmp_path), permutation_report=tmp_path / "none.json")
    assert out["extra"]["hypotheses"] == 3 and out["extra"]["dropped"] == 2


def test_a_daytime_run_publishes_the_lookup_but_records_nothing(chat, tmp_path):
    out = mentor_review.run_mentor_review(
        session_date=SESSION, now=datetime(2026, 9, 29, 12, 0, tzinfo=PT), chat_db=chat, ai_root=tmp_path / "ai",
        request=lambda **_: {"summary": _hyp_reply(HYP), "model": "m"},
        permutation_history=_perm_history(tmp_path), permutation_report=tmp_path / "none.json")
    assert out["extra"]["hypotheses"] == 1 and out["extra"]["hypotheses_recorded"] == 0
    assert MentorChatStore(chat).challenges(kind="hypothesis") == []
    digest = json.loads((tmp_path / "ai" / f"mentor_day_digest_{SESSION}.json").read_text(encoding="utf-8"))
    assert digest["hypotheses"][0]["recorded"] is False and "n=64" in digest["hyp_lines"][0]


def test_a_side_written_into_the_family_is_moved_to_side_before_the_lookup(monkeypatch):
    # 2026-09-30 night: family "avwap_band_bounce SHORT" + side "SHORT" missed as "family ... SHORT SHORT".
    from mentor_packs import hypothesis_pack

    seen = []
    monkeypatch.setattr(hypothesis_pack, "lookup_record", lambda query, report: seen.append(dict(query)) or {})
    item = {"query": {"population": "swing", "horizon": "5", "family": "avwap_band_bounce SHORT", "side": "SHORT",
                      "facets": ["spy_trend=up"]}, "why": "w", "evidence_refs": ["turn:1"]}
    plain = {**item, "query": {**item["query"], "family": "avwap_breakout", "side": "LONG"}}
    clash = {**item, "query": {**item["query"], "family": "avwap_breakout long", "side": "SHORT"}}
    out = mentor_review.look_up_hypotheses([item, plain, clash], None, session=SESSION, issued_utc=DAY_STAMP,
                                           night_store=None)
    assert (seen[0]["family"], seen[0]["side"]) == ("avwap_band_bounce", "SHORT")
    assert [row["normalised"] for row in out] == [True, False, False]
    assert (seen[1]["family"], seen[1]["side"]) == ("avwap_breakout", "LONG")
    assert seen[2]["family"] == "avwap_breakout long", "a side that disagrees is left for the lookup to miss"
    assert "families are listed without a side" in mentor_review.INSTRUCTIONS


def test_bare_facet_values_are_stored_as_name_value_and_found(chat, tmp_path):
    # 2026-10-01: facets ["rvol_below_1"] / ["no_trigger"] missed as "facets must be 1 to 3 'name=value' items".
    bare = {**HYP, "query": {**HYP["query"], "facets": ["held", "up"]}}
    _run(chat, tmp_path, request=lambda **_: {"summary": _hyp_reply(bare), "model": "m"},
         permutation_history=_perm_history(tmp_path), permutation_report=tmp_path / "none.json")
    digest = json.loads((tmp_path / "ai" / f"mentor_day_digest_{SESSION}.json").read_text(encoding="utf-8"))
    assert "n=64" in digest["hyp_lines"][0]
    assert digest["hypotheses"][0]["query"]["facets"] == ["sma100_support=held", "spy_trend=up"]
    assert "facet held read as sma100_support=held" in digest["hypotheses"][0]["lookup"]["facet_notes"]


def test_a_side_in_the_family_still_finds_its_published_cell(chat, tmp_path):
    sided = {**HYP, "query": {**HYP["query"], "family": "avwap_breakout LONG"}}
    _run(chat, tmp_path, request=lambda **_: {"summary": _hyp_reply(sided), "model": "m"},
         permutation_history=_perm_history(tmp_path), permutation_report=tmp_path / "none.json")
    digest = json.loads((tmp_path / "ai" / f"mentor_day_digest_{SESSION}.json").read_text(encoding="utf-8"))
    assert "n=64" in digest["hyp_lines"][0] and digest["hypotheses"][0]["normalised"] is True
    row = MentorChatStore(chat).challenges(kind="hypothesis")[0]
    assert json.loads(row["outcome_json"])["normalised"] is True


def test_no_permutation_report_leaves_the_vocabulary_empty(chat, tmp_path):
    seen = {}

    def request(**kwargs):
        seen.update(kwargs)
        return {"summary": _hyp_reply(), "model": "m"}

    out = _run(chat, tmp_path, request=request, permutation_history=tmp_path / "none",
               permutation_report=tmp_path / "none.json")
    assert out["status"] == "ok" and seen["evidence"]["hypothesis_vocabulary"] == {}
    assert "hyp:report:asof" not in seen["evidence"]["allowed_evidence_ids"]


# ---------------------------------------------------------------- P15a: the night feeds the day coach
def _stamp(day: str, hour: int = 17) -> str:
    return f"{day}T{hour:02d}:00:00.000+00:00"


def _assessment(symbol: str, verdict: str, *, breaks: str = "") -> str:
    flags = [{"plan_id": breaks, "breaks": True, "text": "over the limit"}] if breaks else []
    return json.dumps({"symbol": symbol, "verdict": verdict, "pack_hash": "h",
                       "bullets": [{"text": f"{symbol} held its AVWAP", "evidence_refs": [f"pick:{symbol}:cell"]}],
                       "rule_flags": flags})


@pytest.fixture
def rich(chat, tmp_path):
    """The chat day plus ten sessions of the app's own records and one night of artifacts."""
    from mentor_packs import mirror_pack, night_pack

    store = MentorChatStore(chat)
    store.put_pack("pick_assessment", {"symbol": "NVDA", "hash": "a"}, _assessment("NVDA", "wait", breaks="plan:risk:1"),
                   _stamp(SESSION))
    store.put_pack("pick_assessment", {"symbol": "AMD", "hash": "b"}, _assessment("AMD", "pass", breaks="plan:risk:1"),
                   _stamp("2026-09-25"))
    for cid, day in (("veto:2026-09-22:X:1", "2026-09-22"), ("veto:2026-09-23:Y:1", "2026-09-23")):
        store.add_challenge(cid, kind="veto", symbol=cid.split(":")[2], claim="vetoes like it won",
                            issued_utc=_stamp(day), outcome={"status": "graded", "hit": True})
    store.add_challenge("gate:2026-09-24:TSLA:d", kind="gate", symbol="TSLA", claim="wait: extended",
                        issued_utc=_stamp("2026-09-24"), outcome={"status": "graded", "result": "taken", "r": 1.2,
                                                                   "hit": True})
    for cid, day in (("tilt:2026-09-25:t1", "2026-09-25"), ("tilt:2026-09-29:t2", SESSION)):
        store.add_challenge(cid, kind="tilt", symbol="", claim="re-entry 4 min after a loss", issued_utc=_stamp(day),
                            outcome={"status": "open", "day": day, "at": f"{day}T10:00:00-04:00",
                                     "pattern": "fast_reentry"})
    with sqlite3.connect(chat) as conn:
        conn.execute("UPDATE challenges SET graded_utc = issued_utc WHERE kind IN ('veto', 'gate') AND id != ?",
                     ("veto:2026-09-29:AAA:1",))
    world = night_pack.write_fixture_world(tmp_path / "night")
    return {"chat": chat, "night_paths": world, "mirror_builder": mirror_pack.fixture}


def _rich_run(rich, tmp_path, **kwargs):
    return _run(rich["chat"], tmp_path, night_paths=rich["night_paths"], mirror_builder=rich["mirror_builder"],
                **kwargs)


def _brief(tmp_path):
    return json.loads((tmp_path / "ai" / f"mentor_coach_brief_{SESSION}.json").read_text(encoding="utf-8"))


GOOD_DIGEST = {"digest": [{"text": "He asked about NVDA.", "evidence_refs": ["turn:1"]}], "open_questions": []}
EMPTY_BRIEF = {"watch": [], "missing": [], "issues": [], "one_line": {"text": "", "evidence_refs": []}}


def _two_calls(brief_reply):
    calls = []

    def request(**kwargs):
        calls.append(kwargs)
        return {"model": "m", "summary": GOOD_DIGEST if len(calls) == 1 else brief_reply}

    return request, calls


def test_the_digest_inputs_carry_the_nights_reads_with_ids(rich, tmp_path):
    request, calls = _two_calls(EMPTY_BRIEF)
    out = _rich_run(rich, tmp_path, request=request)
    assert out["status"] == "ok", out["reason"]
    evidence = calls[0]["evidence"]
    night, allowed = evidence["night_reads"], set(evidence["allowed_evidence_ids"])
    assert [row["id"] for row in night["pick_assessments"]] == ["assess:NVDA"], "the day's assessments only"
    assert "wait; NVDA held its AVWAP; breaks plan:risk:1" in night["pick_assessments"][0]["text"]
    assert [row["id"] for row in night["gates"]] == ["challenge:gate:2026-09-24:TSLA:d"]
    assert "graded: result taken, r 1.2, hit True" in night["gates"][0]["text"]
    assert [row["id"] for row in night["tilt"]] == ["challenge:tilt:2026-09-29:t2"]
    assert 0 < len(night["mirror"]) <= 5 and all(row["id"].startswith("mirror:") for row in night["mirror"])
    assert 0 < len(night["digest_facts"]) <= 10
    assert "night:miss:2026-09-29:1" in allowed and "night:prediction:2026-09-29:1" in allowed
    assert "night:day_review:2026-09-29:2" in allowed and "night:ideas:2026-09-29:1" in allowed
    assert len(night["ideas"]) <= 5 and "issue:tilt:fast_reentry" in allowed
    assert all(row["id"] in allowed for rows in night.values() for row in rows)
    assert len(evidence["allowed_evidence_ids"]) == len(allowed), "each id once"
    assert out["extra"]["inputs"] == sum(out["extra"]["night_inputs"].values()) > 10
    assert "night input(s)" in out["reason"]


def test_the_facts_half_publishes_the_issue_candidates_without_a_model(rich, tmp_path):
    out = _rich_run(rich, tmp_path, ask=False)
    assert out["status"] == "ok" and "issue candidate(s)" in out["reason"]
    brief = _brief(tmp_path)
    assert brief["schema"] == "mentor_coach_brief_v1" and brief["worded"] is False
    assert brief["watch"] == [] and brief["missing"] == [] and brief["one_line"] == {"text": "", "evidence_refs": []}
    keys = [item["key"] for item in brief["issues"]]
    assert {"veto_won", "tilt:fast_reentry", "rule:plan:risk:1", "miss:incoming_trendline"} <= set(keys)
    assert len(keys) <= 5 and all(item["first_seen"] == SESSION for item in brief["issues"])
    assert all(item["evidence_refs"] == [f"issue:{item['key']}"] for item in brief["issues"])


def test_an_issue_keeps_its_first_seen_across_nights(rich, tmp_path):
    root = tmp_path / "ai"
    root.mkdir(parents=True, exist_ok=True)
    earlier = {"session_date": "2026-09-25", "issues": [{"key": "tilt:fast_reentry", "first_seen": "2026-09-24"}]}
    (root / "mentor_coach_brief_2026-09-25.json").write_text(json.dumps(earlier), encoding="utf-8")
    _rich_run(rich, tmp_path, ask=False)
    seen = {item["key"]: item["first_seen"] for item in _brief(tmp_path)["issues"]}
    assert seen["tilt:fast_reentry"] == "2026-09-24" and seen["veto_won"] == SESSION


def test_the_coach_brief_is_worded_cited_and_bounded(rich, tmp_path):
    reply = {
        "watch": [{"text": f"watch {i}", "evidence_refs": ["night:day_review:2026-09-29:1"]} for i in range(6)],
        "missing": [{"text": "Trendline vetoes ran anyway", "evidence_refs": ["night:miss:2026-09-29:1"]},
                    {"text": "uncited", "evidence_refs": []}],
        "issues": [{"key": "tilt:fast_reentry", "text": "You re-enter fast after a loss",
                    "evidence_refs": ["issue:tilt:fast_reentry"]},
                   {"key": "made_up", "text": "An invented issue", "evidence_refs": ["issue:veto_won"]}],
        "one_line": {"text": "Slow down after a loss today.", "evidence_refs": ["night:day_review:2026-09-29:1"]},
    }
    request, calls = _two_calls(reply)
    out = _rich_run(rich, tmp_path, request=request)
    assert out["status"] == "ok" and "coach brief 4 watch, 1 missing" in out["reason"]
    assert calls[1]["schema"] is mentor_review.BRIEF_JSON_SCHEMA
    brief = _brief(tmp_path)
    assert brief["worded"] is True and brief["one_line"]["text"] == "Slow down after a loss today."
    assert len(brief["watch"]) == 4 and [m["text"] for m in brief["missing"]] == ["Trendline vetoes ran anyway"]
    assert brief["issues"][0] == {"key": "tilt:fast_reentry", "text": "You re-enter fast after a loss",
                                  "evidence_refs": ["issue:tilt:fast_reentry"], "first_seen": SESSION, "count": 2}
    keys = [item["key"] for item in brief["issues"]]
    assert "made_up" not in keys and "veto_won" in keys, "the model ranks; an issue it skipped still stands"
    assert brief["dropped"] == 4 and len(keys) <= 5


def test_a_foreign_id_in_the_brief_keeps_the_facts_brief(rich, tmp_path):
    request, _calls = _two_calls({**EMPTY_BRIEF, "watch": [{"text": "x", "evidence_refs": ["pick:TSLA:cell"]}]})
    out = _rich_run(rich, tmp_path, request=request)
    assert out["status"] == "ok" and "coach brief rejected" in out["reason"]
    assert _brief(tmp_path)["worded"] is False


def test_no_brief_call_when_the_digest_fails(rich, tmp_path):
    calls = []

    def request(**kwargs):
        calls.append(kwargs)
        return {"model": "m", "summary": {"digest": [{"text": "x", "evidence_refs": ["pick:TSLA:cell"]}],
                                          "open_questions": []}}

    out = _rich_run(rich, tmp_path, request=request)
    assert out["status"] == "failed"
    # The digest call and its one retry; never the brief call.
    assert [call["schema"] for call in calls] == [mentor_review.DIGEST_JSON_SCHEMA] * 2
    assert _brief(tmp_path)["worded"] is False, "the facts part is published regardless"


def test_no_brief_call_without_time_left_in_the_reserve(rich, tmp_path, monkeypatch):
    monkeypatch.setattr(mentor_review, "BRIEF_MIN_SECONDS_LEFT", mentor_review.RESERVE_MINUTES * 60 + 1)
    request, calls = _two_calls(EMPTY_BRIEF)
    out = _rich_run(rich, tmp_path, request=request)
    assert out["status"] == "ok" and len(calls) == 1 and "no time left" in out["reason"]


def test_a_later_facts_only_run_never_overwrites_a_worded_brief(rich, tmp_path):
    request, _calls = _two_calls({**EMPTY_BRIEF, "one_line": {"text": "Worded.", "evidence_refs": ["night:day_review:2026-09-29:1"]}})
    _rich_run(rich, tmp_path, request=request)
    _rich_run(rich, tmp_path, ask=False)
    assert _brief(tmp_path)["one_line"]["text"] == "Worded."


def test_the_brief_call_is_capped_at_1500_output_tokens(rich, tmp_path):
    sent = []

    def post(url, **kwargs):
        sent.append(kwargs["json"]["max_tokens"])

    def request(**kwargs):
        kwargs["post"]("u", json={"max_tokens": 4000})
        return {"model": "m", "summary": GOOD_DIGEST if len(sent) == 1 else EMPTY_BRIEF}

    _rich_run(rich, tmp_path, request=request, post=post)
    assert sent == [mentor_review.MAX_OUTPUT_TOKENS, mentor_review.MAX_BRIEF_TOKENS] == [2500, 1500]


def test_the_brief_writes_nothing_to_the_chat_db(rich, tmp_path):
    before = _dump(rich["chat"])
    request, _calls = _two_calls({**EMPTY_BRIEF, "one_line": {"text": "x", "evidence_refs": ["night:day_review:2026-09-29:1"]}})
    out = mentor_review.run_mentor_review(  # daytime: the app grades, so the night writes no grading column
        session_date=SESSION, now=datetime(2026, 9, 29, 12, 0, tzinfo=PT), chat_db=rich["chat"],
        ai_root=tmp_path / "ai", request=request, night_paths=rich["night_paths"],
        mirror_builder=rich["mirror_builder"])
    assert out["status"] == "ok" and _brief(tmp_path)["worded"] is True
    assert _dump(rich["chat"]) == before


def test_an_unreadable_night_input_never_costs_the_review(rich, tmp_path):
    def broken():
        raise PermissionError("locked")

    out = _run(rich["chat"], tmp_path, ask=False, night_paths=rich["night_paths"], mirror_builder=broken)
    assert out["status"] == "ok" and out["extra"]["night_inputs"]["mirror"] == 0
    assert out["extra"]["unread"] == ["mirror (PermissionError)"]


# ---------------------------------------------------------------- review fixes (2026-09-30)
def test_the_night_inputs_and_issues_read_the_superseding_siblings(rich, tmp_path):
    """Blocker: a rerun writes `<name>.1.json` (D6); the review reads the correction, never the first file."""
    from mentor_packs import night_pack

    night_pack.write_fixture_corrections(rich["night_paths"])
    night = mentor_review.night_inputs(rich["chat"], SESSION, NIGHT, night_paths=rich["night_paths"],
                                       mirror_builder=rich["mirror_builder"])
    contrasts = " ".join(row["text"] for row in night["contrasts"])
    assert "CORRECTED" in contrasts and "incoming_trendline" not in contrasts
    assert "miss_contrast-2026-09-29.1:group:corrected_group" in contrasts
    assert "close_r 9.99 (n=99)" in night["digest_facts"][0]["text"]
    assert "facts/2026/2026-09-29.1.json" in night["digest_facts"][0]["text"]
    keys = [item["key"] for item in mentor_review.issue_candidates(rich["chat"], SESSION,
                                                                   night_paths=rich["night_paths"])]
    assert "miss:corrected_group" in keys and "miss:incoming_trendline" not in keys


def test_an_issue_left_out_of_the_brief_keeps_its_first_seen_in_the_registry(rich, tmp_path):
    """Advisory 1: the registry remembers every candidate, also the sixth that never reaches the brief."""
    store = MentorChatStore(rich["chat"])
    for cid, day in (("tilt:2026-09-25:s1", "2026-09-25"), ("tilt:2026-09-29:s2", SESSION)):
        store.add_challenge(cid, kind="tilt", symbol="", claim="size up after a win", issued_utc=_stamp(day),
                            outcome={"status": "open", "day": day, "at": f"{day}T11:00:00-04:00",
                                     "pattern": "size_up"})
    _rich_run(rich, tmp_path, ask=False)
    registry = json.loads((tmp_path / "ai" / "mentor_issue_registry.json").read_text(encoding="utf-8"))["issues"]
    published = [item["key"] for item in _brief(tmp_path)["issues"]]
    left_out = sorted(set(registry) - set(published))
    assert len(published) == 5 and left_out, "six candidates, five in the brief"
    assert all(registry[key]["first_seen"] == SESSION and registry[key]["nights_seen"] == 1 for key in left_out)
    later = "2026-09-30"
    mentor_review.run_mentor_review(session_date=later, now=datetime(2026, 9, 30, 23, 0, tzinfo=PT),
                                    chat_db=rich["chat"], ai_root=tmp_path / "ai", ask=False,
                                    night_paths=rich["night_paths"], mirror_builder=rich["mirror_builder"])
    again = mentor_review.read_issue_registry(tmp_path / "ai")
    assert all(again[key]["first_seen"] == SESSION and again[key]["last_seen"] == later for key in left_out)
    candidates = mentor_review.issue_candidates(rich["chat"], later, night_paths=rich["night_paths"],
                                                registry=again)
    assert all(item["first_seen"] == SESSION for item in candidates if item["key"] in left_out)
    mentor_review.run_mentor_review(session_date=later, now=datetime(2026, 9, 30, 23, 30, tzinfo=PT),
                                    chat_db=rich["chat"], ai_root=tmp_path / "ai", ask=False,
                                    night_paths=rich["night_paths"], mirror_builder=rich["mirror_builder"])
    assert mentor_review.read_issue_registry(tmp_path / "ai")[left_out[0]]["nights_seen"] == 2, "a rerun counts once"


def test_an_uncited_one_line_is_dropped_and_a_foreign_one_rejects_the_brief(rich, tmp_path):
    """Advisory 3: the one line is model text loaded into memory, so it must cite like every other item."""
    request, _calls = _two_calls({**EMPTY_BRIEF, "one_line": {"text": "Trust your gut today.", "evidence_refs": []},
                                  "watch": [{"text": "SPY at its 50 SMA", "evidence_refs": ["night:day_review:2026-09-29:1"]}]})
    out = _rich_run(rich, tmp_path, request=request)
    brief = _brief(tmp_path)
    assert out["status"] == "ok" and brief["worded"] is True
    assert brief["one_line"] == {"text": "", "evidence_refs": []} and brief["dropped"] == 1
    request, _calls = _two_calls({**EMPTY_BRIEF, "one_line": {"text": "x", "evidence_refs": ["pick:TSLA:cell"]}})
    out = _rich_run(rich, tmp_path, request=request, force=True)
    assert "coach brief rejected" in out["reason"] and _brief(tmp_path)["watch"], "the last good brief stays"
    request, _calls = _two_calls({**EMPTY_BRIEF, "one_line": "a plain string"})
    out = _rich_run(rich, tmp_path, request=request, force=True)
    assert "coach brief rejected" in out["reason"]


def test_a_foreign_id_is_retried_once_with_the_rejection_quoted_and_the_second_reply_kept(chat, tmp_path):
    """2026-09-30: the digest cited 'assess:ALL' once and the whole slot failed; one fed-back retry fixes that."""
    calls = []
    bad = {"digest": [{"text": "Invented.", "evidence_refs": ["assess:ALL"]}], "open_questions": []}
    good = {"digest": [{"text": "Kept.", "evidence_refs": ["fact:turns"]}], "open_questions": []}

    def request(**kwargs):
        calls.append(kwargs)
        digest_calls = [call for call in calls if call["schema"] is mentor_review.DIGEST_JSON_SCHEMA]
        return {"model": "m", "summary": bad if len(digest_calls) == 1 else good}

    out = _run(chat, tmp_path, request=request)
    assert out["status"] == "ok", out["reason"]
    digest_calls = [call for call in calls if call["schema"] is mentor_review.DIGEST_JSON_SCHEMA]
    assert len(digest_calls) == 2
    assert "assess:ALL" not in digest_calls[0]["evidence"]["instructions"]
    assert "assess:ALL" in digest_calls[1]["evidence"]["instructions"]
    assert "retried after" in out["reason"] and "assess:ALL" in out["reason"]
    digest = json.loads((tmp_path / "ai" / f"mentor_day_digest_{SESSION}.json").read_text(encoding="utf-8"))
    assert [item["text"] for item in digest["digest"]] == ["Kept."]


def test_a_foreign_id_on_both_attempts_still_fails_after_the_one_retry(chat, tmp_path):
    calls = []
    bad = {"digest": [{"text": "Invented.", "evidence_refs": ["assess:ALL"]}], "open_questions": []}

    def request(**kwargs):
        calls.append(kwargs)
        return {"model": "m", "summary": bad}

    out = _run(chat, tmp_path, request=request)
    assert out["status"] == "failed" and "rejected" in out["reason"] and "assess:ALL" in out["reason"]
    assert len(calls) == 2  # one retry, never more; no brief call after a failed digest
    assert not (tmp_path / "ai" / f"mentor_day_digest_{SESSION}.json").exists()
