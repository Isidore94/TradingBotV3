"""P8-P4 part 1: three night AI slots that failed or read as failure every night.

1. `day_review_narration` timed out at 540 s on 7 of 11 nights with a ~40 KB
   pack: the pack is now trimmed to a time budget and the ledger names its size.
2. `econ_brief` rejected "1:00 pm ET" / "1 p.m." against a 13:00 ET event.
3. `theta_pick_grading`: picks that were never quoted can never be graded.

No model is called; every transport is faked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
for _path in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import tj4_support as day_fx  # noqa: E402
from test_ai_night_fixes_2026_09_23 import _live_sized_pack  # noqa: E402


# ---------------------------------------------------------------------------
# 1. day story: trimmed to the time budget, deterministically, still verifiable
# ---------------------------------------------------------------------------
def _bulky_pack():
    """The live-sized fixture with its uncited sections grown past the budget."""
    pack = _live_sized_pack()
    pack["report_card"]["lines"] = [
        {"source_id": f"report_card:k{i}", "key": f"k{i}", "text": "a line " * 60}
        for i in range(30)
    ]
    pack["congruence"] = [
        {"source_id": f"congruence:c{i}", "text": "agreed " * 80, "source_ids": []}
        for i in range(20)
    ]
    return pack


def test_a_day_pack_over_the_budget_is_trimmed_deterministically(tmp_path):
    import day_review_pack
    from ai_jobs import day_review_narration as narration

    pack = _bulky_pack()
    whole = narration._model_pack(pack)
    assert len(json.dumps(whole, default=str)) > narration.MAX_DAY_EVIDENCE_CHARS

    first = narration._day_evidence(pack, tmp_path)
    second = narration._day_evidence(pack, tmp_path)
    text = json.dumps(first, sort_keys=True, default=str)

    assert text == json.dumps(second, sort_keys=True, default=str)
    assert len(text) <= narration.MAX_DAY_EVIDENCE_CHARS
    trimmed = first["pack_trimmed"]
    assert trimmed == list(narration.DAY_TRIM_ORDER[: len(trimmed)])
    assert trimmed[0] == "report_card"
    assert first["pack"]["report_card"] != whole["report_card"]
    # Everything the verifier needs is still in front of the model.
    for source_id in day_review_pack.allowed_source_ids(pack):
        assert json.dumps(source_id) in text
    assert first["pack"]["reads"] == pack["reads"]
    assert first["pack"]["trader_said"] == pack["trader_said"]


def _v2_answer(evidence: dict) -> dict:
    return {
        "headline": "A day.",
        "what_happened": "It moved.",
        "what_you_thought": "Up.",
        "read_explanations": {key: "Explained." for key in evidence["read_explanations"]},
        "chased_against_news": {"verdict": "unknown", "evidence_id": ""},
        "process": "Fine.",
        "sources": evidence["allowed_source_ids"][:2],
    }


def test_a_trimmed_day_pack_still_verifies_and_the_ledger_names_what_was_sent(tmp_path):
    import day_review_pack
    from ai_jobs import day_review_narration as narration

    root = tmp_path / "day_review"
    pack = _bulky_pack()
    day_review_pack.write_pack(pack, root=root)
    seen: list[dict] = []

    def request(**kwargs):
        seen.append(kwargs["evidence"])
        return {"summary": _v2_answer(kwargs["evidence"]), "model": "fake-12b"}

    outcome = narration.run_day_review_narration(
        session_date=day_fx.SESSION, now=day_fx.OVERNIGHT, root=root,
        request=request, only_this_session=True,
    )

    assert outcome["status"] == "ok", outcome
    assert seen and seen[0]["pack_trimmed"][0] == "report_card"
    assert "grounded day story written" in outcome["reason"]
    assert "bytes (~" in outcome["reason"] and "tokens est.)" in outcome["reason"]
    assert "dropped report_card" in outcome["reason"]
    assert narration.narration_path(day_fx.SESSION, root=root).exists()


def test_a_failed_day_call_still_names_the_size_it_sent(tmp_path):
    import day_review_pack
    from ai_jobs import day_review_narration as narration

    root = tmp_path / "day_review"
    day_review_pack.write_pack(day_fx.build(), root=root)

    def request(**kwargs):
        raise RuntimeError("Read timed out. (read timeout=540)")

    outcome = narration.run_day_review_narration(
        session_date=day_fx.SESSION, now=day_fx.OVERNIGHT, root=root,
        request=request, only_this_session=True,
    )

    assert outcome["status"] == "degraded_no_narrative"
    assert "Read timed out" in outcome["reason"]
    assert "bytes (~" in outcome["reason"]


# ---------------------------------------------------------------------------
# 2. econ brief: clock times normalise; a past event in the prose is not today's
# ---------------------------------------------------------------------------
BRIEF_0924 = (
    Path(__file__).resolve().parent / "fixtures" / "day_review" / "forecast_2026-09-24.md"
).read_text(encoding="utf-8")


def _econ_pack(target: str) -> dict:
    import econ_brief

    forecasts = [{"session": "2026-09-24", "text": BRIEF_0924, "entry_id": "e",
                  "created_at": "2026-09-24T08:55:00-04:00"}]
    return econ_brief.build_pack(forecasts, target_session=target)


def _auction_reply(pack: dict, text: str) -> dict:
    rows = list(pack["today"]) + list(pack["week"])
    auction = next(row["id"] for row in rows if row["kind"] == "auction")
    assert next(row for row in rows if row["id"] == auction)["time_et"] == "13:00"
    other = [row["id"] for row in rows if row["id"] != auction][:2]
    return {"lines": [
        {"text": text, "event_ids": [auction]},
        {"text": "Watch the rates tape.", "event_ids": [other[0]]},
        {"text": "Keep size small.", "event_ids": [other[1]]},
    ]}


def test_every_way_of_writing_one_pm_matches_a_1300_et_event():
    from ai_jobs import econ_brief_narration as job

    pack = _econ_pack("2026-09-24")  # the brief's own day, when the auction is live
    for text in (
        "The 7-year auction is at 1:00 pm ET.",   # rejected live 2026-09-24 23:37
        "Observe the 1 p.m. Treasury auction.",   # rejected live 2026-09-24 22:35
        "The 7-year auction lands at 1pm.",
        "The 7-year auction is at 13:00 ET.",
        "The 7-year auction is at 1:00 P.M.",
    ):
        assert job.validate(_auction_reply(pack, text), pack)[0] == text


def test_a_wrong_time_for_the_auction_still_rejects():
    import pytest

    from ai_jobs import econ_brief_narration as job

    pack = _econ_pack("2026-09-24")
    with pytest.raises(ValueError, match="is not the time of an event it cites"):
        job.validate(_auction_reply(pack, "The 7-year auction is at 2:00 pm ET."), pack)


def test_the_live_rejection_was_a_past_auction_and_the_model_is_told_so():
    """The night of 09-24 plans 09-25; the 13:00 auction was on 09-24's brief day."""
    from ai_jobs import econ_brief_narration as job

    pack = _econ_pack("2026-09-25")
    kinds = {row["kind"] for row in list(pack["today"]) + list(pack["week"])}
    assert "auction" not in kinds
    assert "1 p.m. 7-year auction" in pack["bottom_line"]
    instructions = job._evidence(pack)["instructions"]
    assert "2026-09-24" in instructions
    assert "already past" in instructions
    assert job.PROMPT_VERSION != "econ_brief_narration_v1"


# ---------------------------------------------------------------------------
# 3. theta: a pick that was never quoted is dead once its window has passed
# ---------------------------------------------------------------------------
def _no_quote_pick(scan_date: str, *, expiry: str = "", symbol: str = "DDD") -> dict:
    return {"symbol": symbol, "scan_date": scan_date, "bar_date": scan_date,
            "strike": None, "expiry": expiry, "close": 100.0, "atr": 2.0,
            "play_type": "put", "support_combo": "SMA_50", "supports": []}


def _quoted_pick(scan_date: str) -> dict:
    return {**_no_quote_pick(scan_date, symbol="BBB"), "strike": 95.0, "expiry": "2026-07-17"}


def _grade(tmp_path, picks, as_of):
    import market_calendar
    import theta_pick_tracker

    return theta_pick_tracker.grade_theta_picks(
        picks, closes_for=lambda symbol: {}, calendar=market_calendar,
        as_of=as_of, path=tmp_path / "theta_outcomes.csv",
    )


def test_a_never_quoted_pick_past_thirty_sessions_is_dead(tmp_path):
    from datetime import date

    # 2026-06-01 + 30 sessions is 2026-07-15 (06-19 and 07-03 are holidays).
    rows = _grade(tmp_path, [_no_quote_pick("2026-06-01")], date(2026, 7, 15))
    assert rows[0]["status"] == "dead"
    assert rows[0]["unmeasured_reason"] == "never_quoted"
    assert rows[0]["held_20"] is None
    young = _grade(tmp_path, [_no_quote_pick("2026-06-01")], date(2026, 7, 14))
    assert young[0]["status"] == "unmeasured"
    assert "no_option_quote" in young[0]["unmeasured_reason"]


def test_a_never_quoted_pick_with_an_expiry_is_dead_once_that_expiry_passes(tmp_path):
    from datetime import date

    pick = _no_quote_pick("2026-06-01", expiry="2026-06-12")
    assert _grade(tmp_path, [pick], date(2026, 6, 12))[0]["status"] == "dead"
    assert _grade(tmp_path, [pick], date(2026, 6, 11))[0]["status"] == "unmeasured"


def test_the_theta_ledger_sentence_names_dead_picks_apart(tmp_path, monkeypatch):
    from datetime import date, datetime

    import market_calendar
    import theta_pick_tracker
    from ai_jobs import theta_grading

    picks = [_no_quote_pick("2026-06-01"), _no_quote_pick("2026-07-10", symbol="EEE"),
             _quoted_pick("2026-07-10")]
    monkeypatch.setattr(theta_pick_tracker, "read_theta_picks", lambda path: picks)
    outcomes = tmp_path / "outcomes.csv"
    result = theta_grading.run_theta_pick_grading(
        picks_path=tmp_path / "p.jsonl", outcomes_path=outcomes,
        daily_bars_dir=tmp_path / "bars",
        now=datetime(2026, 7, 16, 2, 0, tzinfo=market_calendar.MARKET_TZ),
    )
    assert result["as_of"] == date(2026, 7, 15).isoformat()
    assert result["reason"].endswith(
        "0 measured, 1 pending, 1 dead (never quoted), 1 unmeasured"
    )
    assert result["dead"] == 1
    # The CSV is still rewritten in full, dead rows included.
    assert outcomes.read_text(encoding="utf-8").count("\n") == 4
