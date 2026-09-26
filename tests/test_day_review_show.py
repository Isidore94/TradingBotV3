"""R1 Day Review Show: the pure deck module (schema, verifier, fallback)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "scripts"))

import day_review_pack  # noqa: E402
import day_review_show as show  # noqa: E402

SESSION = "2026-09-25"


def _pack() -> dict:
    pack = day_review_pack.build_pack(
        SESSION,
        story={"measured": [{"symbol": "SPY", "close": 571.25, "change_pct": 0.8123}]},
        d1_label="trend day",
        walkaway={
            "liked_not_traded": [{"symbol": "NVDA", "ran_after_pct": 3.4}],
            "rejected": [{"symbol": "AMD", "ran_after_pct": -1.2}],
        },
        reads=[{"read_id": "r1", "entry_id": "e1", "verdict": "right"}],
        trades=[{"trade_id": "t1", "symbol": "TSLA", "direction": "LONG",
                 "status": "closed", "net_pnl": 125.5}],
        report_card={"lines": [
            {"key": "did_well", "text": "Liked 2 names, 1 ran.", "n": 2},
            {"key": "missed", "text": "Rejected 1 name.", "n": 1},
        ]},
    )
    # A recorded mood, as `mood_section` would mint it.
    pack["mood"] = {"n": 1, "recorded": [{"entry_id": "m1", "score": 4, "source_id": "mood:m1"}]}
    return pack


def _slide(kind, title="A title", body="Plain words.", ids=("measured:SPY",), stat="", label=""):
    return {"kind": kind, "title": title, "body": body, "stat_source_id": stat,
            "stat_label": label, "source_ids": list(ids)}


def _good_reply() -> dict:
    return {
        "title": "A steady trend day",
        "slides": [
            _slide("open", "How it opened", "The desk called it a trend day.", ["env:d1_label"]),
            _slide("tape", "SPY climbed", "SPY closed at 571.25.", ["measured:SPY"],
                   stat="measured:SPY", label="SPY on the day"),
            _slide("trade", "TSLA long", "One long, closed.", ["trade:t1"],
                   stat="trade:t1", label="net"),
            _slide("miss", "NVDA ran", "NVDA ran after you liked it.", ["walkaway:0:NVDA"]),
            _slide("read", "Your read", "Your call on the tape.", ["read:r1"]),
            _slide("close", "That was the day", "See you tomorrow.", ["report_card:did_well"]),
        ],
    }


def test_a_good_deck_passes_and_the_stat_value_comes_from_the_pack():
    pack = _pack()
    deck = show.verify_show(_good_reply(), pack)
    assert len(deck["slides"]) == 6
    tape = deck["slides"][1]
    assert tape["stat"] == {"value": "+0.81%", "label": "SPY on the day", "source_id": "measured:SPY"}
    assert deck["slides"][2]["stat"]["value"] == "+125.50"
    assert deck["slides"][0]["stat"] is None


def test_the_model_schema_has_no_stat_value_field():
    item = show.MODEL_JSON_SCHEMA["properties"]["slides"]["items"]["properties"]
    assert "stat_source_id" in item and "value" not in item and "stat" not in item
    assert show.MODEL_JSON_SCHEMA["properties"]["title"]["maxLength"] == 60
    assert item["body"]["maxLength"] == 220 and item["title"]["maxLength"] == 48


@pytest.mark.parametrize(
    "mutate, why",
    [
        (lambda r: r["slides"].pop(), "6 to 10"),
        (lambda r: r["slides"].extend([_slide("number")] * 5), "6 to 10"),
        (lambda r: r["slides"][0].update(source_ids=["made:up"]), "does not carry"),
        (lambda r: r["slides"][0].update(source_ids=[]), "cites 1 to 4"),
        (lambda r: r["slides"][1].update(body="SPY closed at 999."), "999"),
        (lambda r: r["slides"][1].update(stat_label="up 7 points"), "7"),
        (lambda r: r["slides"][3].update(body="AAPL ran after you liked it."), "AAPL"),
        (lambda r: r["slides"][4].update(kind="open"), "repeats kind"),
        (lambda r: r["slides"][2].update(body="You felt calm and won."), "mood"),
        (lambda r: r["slides"][2].update(source_ids=["trade:t1", "mood:m1"]), "mood"),
        (lambda r: r["slides"][4].update(stat_source_id="mood:m1", source_ids=["mood:m1"]), "mood"),
        (lambda r: r["slides"][1].update(stat_source_id="env:d1_label"), "does not cite"),
        (lambda r: r["slides"][1].update(stat_label=""), "no caption"),
        (lambda r: r["slides"][0].update(stat_label="orphan"), "does not name"),
        (lambda r: r["slides"][0].update(value="1"), "exactly"),
        (lambda r: r.update(title="x" * 61), "longer than 60"),
        (lambda r: r["slides"][0].update(kind="party"), "not a slide kind"),
    ],
)
def test_one_breach_rejects_the_whole_deck(mutate, why):
    reply = _good_reply()
    mutate(reply)
    with pytest.raises(show.ShowRejected, match=why):
        show.verify_show(reply, _pack())


def test_number_and_trade_may_repeat():
    reply = _good_reply()
    reply["slides"][4:4] = [
        _slide("number", "Card", "Liked names.", ["report_card:did_well"],
               stat="report_card:did_well", label="liked"),
        _slide("number", "Card again", "Rejected names.", ["report_card:missed"]),
        _slide("trade", "TSLA again", "Same trade.", ["trade:t1"]),
    ]
    deck = show.verify_show(reply, _pack())
    assert deck["slides"][4]["stat"]["value"] == "2"


def test_mood_on_its_own_slide_is_reported_not_rejected():
    reply = _good_reply()
    reply["slides"][4] = _slide("lesson", "How you were", "You noted feeling calm.", ["mood:m1"])
    assert show.verify_show(reply, _pack())["slides"][4]["kind"] == "lesson"


def test_the_fallback_deck_is_deterministic_and_well_formed():
    pack = _pack()
    bars = [{"open": 570.0, "high": 572.0, "low": 569.5, "close": 571.0}]
    one = show.fallback_deck(pack, truth_lines=["Last 20 sessions:", "Stocks 3-2"],
                             alerts_line="Alerts: 12 shown", spy_bars=bars)
    two = show.fallback_deck(copy.deepcopy(pack), truth_lines=["Last 20 sessions:", "Stocks 3-2"],
                             alerts_line="Alerts: 12 shown", spy_bars=bars)
    assert one == two
    assert show.MIN_SLIDES <= len(one["slides"]) <= show.MAX_SLIDES
    kinds = [slide["kind"] for slide in one["slides"]]
    assert kinds[0] == "open" and kinds[-1] == "close"
    for kind in set(kinds) - show.REPEATABLE_KINDS:
        assert kinds.count(kind) == 1
    board = one["slides"][kinds.index("scoreboard")]
    assert board["lines"] == ["Liked 2 names, 1 ran.", "Rejected 1 name."]
    assert any("Alerts: 12 shown" in s["body"] for s in one["slides"])
    assert one["slides"][1]["stat"]["value"] == "+0.18%"
    allowed = set(day_review_pack.allowed_source_ids(pack))
    for slide in one["slides"]:
        assert len(slide["title"]) <= 48 and len(slide["body"]) <= 220
        assert set(slide["source_ids"]) <= allowed and len(slide["source_ids"]) <= 4


def test_the_fallback_deck_works_with_no_pack_at_all():
    deck = show.fallback_deck(None, session_date=SESSION)
    assert len(deck["slides"]) >= show.MIN_SLIDES
    assert deck["slides"][0]["title"].endswith(SESSION)


def _stored(pack, reply=None):
    deck = show.verify_show(reply or _good_reply(), pack)
    return {"schema": show.SCHEMA, "session_date": SESSION, "pack_hash": pack["inputs_hash"],
            "model": "gemma3:12b", "show": deck}


def test_the_desk_shows_a_verified_deck_and_reprints_stats_from_the_pack():
    pack = _pack()
    stored = _stored(pack)
    stored["show"]["slides"][1]["stat"]["value"] = "+99%"  # a doctored file value
    chosen = show.desk_deck(stored, pack, session_date=SESSION)
    assert chosen["facts_only"] is False and chosen["model"] == "gemma3:12b"
    assert chosen["deck"]["slides"][1]["stat"]["value"] == "+0.81%"


@pytest.mark.parametrize(
    "spoil, why",
    [
        (lambda s: None, "no show"),
        (lambda s: s.update(pack_hash="old"), "changed"),
        (lambda s: s.update(session_date="2026-09-24"), "another session"),
        (lambda s: s["show"]["slides"][0].update(source_ids=["made:up"]), "failed its checks"),
    ],
)
def test_the_desk_falls_back_to_facts_only(spoil, why):
    pack = _pack()
    stored = _stored(pack)
    result = spoil(stored)
    chosen = show.desk_deck(None if why == "no show" else stored, pack, session_date=SESSION)
    del result
    assert chosen["facts_only"] is True
    assert why in chosen["reason"]
    assert chosen["deck"]["slides"][-1]["title"] == "Facts only"


def test_inputs_hash_moves_with_the_pack_and_the_narration():
    pack = _pack()
    base = show.inputs_hash(pack, None)
    assert base == show.inputs_hash(copy.deepcopy(pack), None)
    assert base != show.inputs_hash(pack, {"headline": "x"})
    moved = dict(pack, inputs_hash="other")
    assert base != show.inputs_hash(moved, None)


def test_every_kind_has_one_glyph():
    assert set(show.KIND_GLYPHS) == set(show.KINDS)
    assert len(set(show.KIND_GLYPHS.values())) == len(show.KINDS)


def test_the_show_lives_under_shows_by_date(tmp_path):
    assert show.show_path(SESSION, root=tmp_path) == tmp_path / "shows" / f"{SESSION}.json"


# ---------------------------------------------------------------------------
# the night slot `day_review_show`
# ---------------------------------------------------------------------------
def _night(tmp_path, reply=None, *, narration=None):
    from ai_jobs import day_review_narration

    pack = _pack()
    day_review_pack.write_pack(pack, root=tmp_path)
    if narration is not None:
        day_review_narration._atomic_write(
            day_review_narration.narration_path(SESSION, root=tmp_path), narration
        )
    calls = []

    def _request(**kwargs):
        calls.append(kwargs)
        return {"summary": copy.deepcopy(reply or _good_reply()), "model": "gemma3:12b"}

    return pack, calls, _request


def test_the_night_writes_a_verified_show_with_its_stamps(tmp_path):
    from ai_jobs import day_review_show_night as night

    pack, calls, request = _night(tmp_path)
    outcome = night.run_day_review_show(session_date=SESSION, root=tmp_path, request=request)
    assert outcome["status"] == "ok", outcome
    stored = json.loads(show.show_path(SESSION, root=tmp_path).read_text(encoding="utf-8"))
    assert stored["model"] == "gemma3:12b"
    assert stored["prompt_version"] == night.PROMPT_VERSION
    assert stored["inputs_hash"] == show.inputs_hash(pack, None)
    assert stored["pack_hash"] == pack["inputs_hash"]
    assert stored["show"]["slides"][1]["stat"]["value"] == "+0.81%"
    assert calls[0]["evidence"]["allowed_source_ids"]
    assert show.desk_deck(stored, pack, session_date=SESSION)["facts_only"] is False
    # Unchanged inputs cost no second call.
    again = night.run_day_review_show(session_date=SESSION, root=tmp_path, request=request)
    assert again["status"] == "ok" and len(calls) == 1


def test_a_rejected_deck_is_degraded_and_keeps_the_last_good_file(tmp_path):
    from ai_jobs import day_review_show_night as night

    _pack_, _calls, request = _night(tmp_path)
    night.run_day_review_show(session_date=SESSION, root=tmp_path, request=request)
    path = show.show_path(SESSION, root=tmp_path)
    before = path.read_bytes()
    bad = _good_reply()
    bad["slides"][1]["body"] = "SPY closed at 999."
    # The facts moved, so a call is owed.
    pack = day_review_pack.build_pack(SESSION, d1_label="range day")
    day_review_pack.write_pack(pack, root=tmp_path)

    def _bad(**_kwargs):
        return {"summary": bad, "model": "gemma3:12b"}

    outcome = night.run_day_review_show(session_date=SESSION, root=tmp_path, request=_bad)
    assert outcome["status"] == "degraded_no_narrative"
    assert "prior show was kept" in outcome["reason"]
    assert path.read_bytes() == before


def test_the_night_reads_only_a_story_written_for_this_pack(tmp_path):
    from ai_jobs import day_review_show_night as night

    pack = _pack()
    story = {"inputs_hash": pack["inputs_hash"], "narration": {"headline": "Trend day", "sources": []}}
    _p, calls, request = _night(tmp_path, narration=story)
    night.run_day_review_show(session_date=SESSION, root=tmp_path, request=request)
    assert calls[0]["evidence"]["previous_story"] == {"headline": "Trend day"}

    stale = {"inputs_hash": "old", "narration": {"headline": "Old day"}}
    other = tmp_path / "other"
    _p, calls, request = _night(other, narration=stale)
    night.run_day_review_show(session_date=SESSION, root=other, request=request)
    assert calls[0]["evidence"]["previous_story"] == {}


def test_no_pack_is_skipped_without_a_call(tmp_path):
    from ai_jobs import day_review_show_night as night

    calls = []
    outcome = night.run_day_review_show(
        session_date=SESSION, root=tmp_path, request=lambda **k: calls.append(k)
    )
    assert outcome["status"] == "skipped" and calls == []


def test_the_slot_runs_directly_after_the_day_story():
    from ai_jobs import runner

    slots = runner.default_slots()
    names = [slot.name for slot in slots]
    assert names[names.index("day_review_narration") + 1] == "day_review_show"
    slot = slots[names.index("day_review_show")]
    assert slot.uses_model and slot.reserve_minutes == 5.0 and slot.goal == "coaching"
    priority = runner.MODEL_SLOT_PRIORITY
    assert priority[priority.index("day_review_narration") + 1] == "day_review_show"
