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
        (lambda r: r["slides"][1].update(body="SPY fell nine hundred points."), "spelled-out"),
        (lambda r: r["slides"][1].update(stat_label="twenty names ran"), "spelled-out"),
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


def test_the_prompt_asks_for_tickers_in_capitals():
    # The ticker check reads capitals only, so the model must write tickers that way.
    assert "Do not write in capitals" not in show.INSTRUCTIONS
    assert "tickers in capitals" in show.INSTRUCTIONS


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


# ---------------------------------------------------------------------------
# P19: a heavy pack fits the call; the model is told its tickers (2026-09 rejections)
_INTERNALS_LINE = (
    "Breadth (RSP-SPY) -0.26%  ·  Fear VXX down / SPY down  (both the same way)  ·  "
    "Rates (TLT) -0.58%  ·  Oil (USO) -2.05%\nLeaders XLK, XLU, XLI  ·  Laggards XLV, XLP, XLE  ·  "
    "Offense-defense +0.85%  ·  Sectors above VWAP 2 of 11"
)


def _heavy_pack() -> dict:
    """A pack shaped like the live 2026-09-29 one (~27k chars after the old trim)."""
    pack = _pack()
    words = "gap up that was instantly filled, now testing the downside and the level below us "
    pack["trader_said"] = list(pack["trader_said"]) + [
        {"at": f"2026-09-25T14:{i:02d}:58+00:00", "direction": "", "entry_id": f"mj-heavy-{i:02d}",
         "horizon": "", "kind": "observation", "source_id": f"said:mj-heavy-{i:02d}:observation",
         "text": words * 2, "timeframe": "M5"}
        for i in range(16)
    ]
    pack["environment"] = list(pack["environment"]) + [
        {"detail": "Bullish Weak", "event_at": f"2026-09-25T13:{i:02d}:55+00:00", "event_type": "regime_shift",
         "from_regime": "bullish_strong", "kind": "regime_shift", "schema": "market_regime_shift_v1",
         "session_date": SESSION, "source": "auto", "source_id": f"env:regime_shift:{i}",
         "spy_day_pct": None, "to_regime": "bullish_weak", "writer_host": "NucBox_K8_Plus", "writer_pid": 16920}
        for i in range(10)
    ]
    pack["measured"] = list(pack["measured"]) + [
        {"atr": 6.519590712596484, "bars_through": SESSION, "bars_used": 60, "change_pct": -0.17110507747521198,
         "close": 764.2999877929688, "completed_only": True, "kind": "measured",
         "position_vs_sma20": {"distance_atr": -0.11135823763187615, "side": "below", "sma20": 765.0259979248046},
         "range_atr": 0.705571380762604, "reason": "",
         "rule_versions": {"change_pct": "close_over_prior_close_pct_v1",
                           "position_vs_sma20": "close_minus_sma20_in_atr_v1",
                           "range_atr": "session_range_over_wilder_atr14_v1"},
         "source_id": f"measured:{symbol}", "status": "measured", "symbol": symbol}
        for symbol in ("QQQ", "IWM", "TLT", "USO", "VXX")
    ]
    pack["internals"] = [
        {"source_id": f"internals:mentor:2026-09-25T08:{i:02d}:15-07:00", "kind": "mentor",
         "at": f"2026-09-25T08:{i:02d}:15-07:00", "context": {"common": {
             "availability": "available", "internals": _INTERNALS_LINE, "reason": ""}}}
        for i in range(10)
    ]
    pack["reads"] = list(pack["reads"]) + [
        {"because": words, "benchmark": "SPY", "confidence": "medium", "direction": "range",
         "entry_id": f"mj-heavy-{i:02d}", "flat_band_rule": "atr_0.25_v1", "grader_gap": "",
         "horizon": "next_5_sessions", "move_atr": None, "observation": words * 2,
         "read_id": f"rd-heavy-{i}", "schema": "market_read_v1", "session": SESSION, "source": "click",
         "source_id": f"read:rd-heavy-{i}", "span": [], "stamp": "2026-09-25T08:03:15-07:00",
         "timeframe": "D1", "verdict": "pending 2026-10-02",
         "checkpoints": [{"close": None, "move": None, "move_atr": None, "session": "2026-09-26",
                          "sessions": n, "status": "pending"} for n in (1, 3, 5)]}
        for i in range(8)
    ]
    pack["forecast"] = {"entry_id": "mj-fc", "fields": {}, "text": ("- a brief line about the morning tape\n" * 160)}
    pack["trades"] = {**pack["trades"], "rows": list(pack["trades"]["rows"]) + [
        {"source_id": f"trade:h{i}", "trade_id": f"h{i}", "symbol": "WYNN", "direction": "LONG",
         "status": "closed", "net_pnl": 12.5, "notes": words * 3}
        for i in range(9)
    ]}
    pack["report_card"] = {**pack["report_card"], "lines": list(pack["report_card"]["lines"]) + [
        {"key": f"extra_{i}", "text": words, "n": i, "source_id": f"report_card:extra_{i}"} for i in range(4)
    ]}
    return pack


def _heavy_reply() -> dict:
    return {
        "title": "A slow range day",
        "slides": [
            _slide("open", "How it opened", "You watched the gap fill.", ["said:mj-heavy-00:observation"]),
            _slide("tape", "SPY on the day", "SPY ended the day lower.", ["measured:SPY"],
                   stat="measured:SPY", label="SPY on the day"),
            _slide("read", "Your read", "You called a range.", ["read:rd-heavy-0"]),
            _slide("trade", "TSLA long", "One long, closed.", ["trade:t1"]),
            _slide("lesson", "Wait for the bounce", "You waited for proof first.", ["said:mj-heavy-01:observation"]),
            _slide("close", "That was the day", "See you tomorrow.", ["measured:QQQ"]),
        ],
    }


def _write_heavy(tmp_path, reply):
    pack = _heavy_pack()
    day_review_pack.write_pack(pack, root=tmp_path)
    calls = []

    def _request(**kwargs):
        calls.append(kwargs)
        return {"summary": copy.deepcopy(reply), "model": "gemma3:12b"}

    return pack, calls, _request


def test_the_heavy_pack_used_to_overflow_the_old_trim():
    from ai_jobs import day_review_narration as story
    from ai_jobs import day_review_show_night as night

    pack = _heavy_pack()
    view = story._model_pack(pack)
    for part in story.DAY_TRIM_ORDER:
        story._trim_part(view, part)
    # Every part the day story may drop is gone and the evidence still does not fit.
    assert len(json.dumps(view, default=str)) > night.MAX_EVIDENCE_CHARS


def test_a_heavy_pack_now_fits_and_the_show_is_called_and_kept(tmp_path):
    from ai_jobs import day_review_show_night as night

    _pack_, calls, request = _write_heavy(tmp_path, _heavy_reply())
    outcome = night.run_day_review_show(session_date=SESSION, root=tmp_path, request=request)
    assert len(calls) == 1, outcome
    assert outcome["status"] == "ok", outcome
    evidence = calls[0]["evidence"]
    assert night._chars(evidence) <= night.MAX_EVIDENCE_CHARS
    # Caps come first; a capped or dropped section's ids leave allowed_source_ids.
    assert evidence["pack_trimmed"][:2] == ["report_card_lines", "trades_rows"]
    allowed = set(evidence["allowed_source_ids"])
    assert "trade:t1" in allowed and "measured:SPY" in allowed
    assert "report_card:extra_3" not in allowed


def test_a_deck_citing_an_id_trimmed_away_is_still_rejected(tmp_path):
    from ai_jobs import day_review_show_night as night

    reply = _heavy_reply()
    reply["slides"][5]["source_ids"] = ["report_card:extra_3"]
    _pack_, calls, request = _write_heavy(tmp_path, reply)
    outcome = night.run_day_review_show(session_date=SESSION, root=tmp_path, request=request)
    assert len(calls) == 1
    assert outcome["status"] == "degraded_no_narrative"
    assert "dropped to fit the call" in outcome["reason"]


def test_the_show_view_hides_no_id_before_trimming():
    from ai_jobs import day_review_show_night as night

    for pack in (_pack(), _heavy_pack()):
        assert day_review_pack.allowed_source_ids(night._show_view(pack)) == day_review_pack.allowed_source_ids(pack)


def test_the_model_is_handed_the_packs_tickers_and_xlk_still_rejects(tmp_path):
    from ai_jobs import day_review_show_night as night

    _pack_, calls, request = _write_heavy(tmp_path, _heavy_reply())
    night.run_day_review_show(session_date=SESSION, root=tmp_path, request=request)
    evidence = calls[0]["evidence"]
    assert evidence["pack_tickers"] == sorted(show.pack_tickers(_heavy_pack()))
    assert "XLK" not in evidence["pack_tickers"] and "SPY" in evidence["pack_tickers"]
    assert "pack_tickers" in evidence["instructions"]
    # The verifier is unchanged: XLK, written only in the internals text, still rejects.
    reply = _good_reply()
    reply["slides"][1]["body"] = "XLK led the day."
    with pytest.raises(show.ShowRejected, match="XLK"):
        show.verify_show(reply, _heavy_pack())
