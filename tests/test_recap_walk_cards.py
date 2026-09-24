"""Review my day: which cards the walk shows, in what order, and which Mentor
questions a trade card carries (trader, 2026-09-23)."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ui.widgets import recap_walk_cards as cards  # noqa: E402

SESSION = "2026-09-22"


def _trade(n: int, pnl: float, *, stop=None, entry=100.0, exit_=101.0, side="LONG") -> dict:
    return {
        "trade_id": f"t{n}", "symbol": f"SYM{n}", "direction": side, "status": "CLOSED",
        "opened_at": f"{SESSION}T10:0{n % 10}:00-04:00", "closed_at": f"{SESSION}T11:00:00-04:00",
        "net_pnl": pnl, "average_entry_price": entry, "average_exit_price": exit_, "planned_stop": stop,
        "quantity_opened": 100,
    }


def _pick(symbol: str, category: str, ran: float) -> dict:
    return {
        "symbol": symbol, "side": "LONG", "category": category,
        "result": f"ran {ran:+.1f}% after (run)", "reason": "too extended",
        "recipe": "D1 pullback · long", "recipe_lately": {"core": {"words": "too few to tell (n 4)"}},
        "where_to_look": ["Master AVWAP page"], "surfaced": {"first_at": "", "surfaces": []},
    }


def _payload(trades=(), reads=()) -> dict:
    return {"session_date": SESSION, "trades": list(trades), "reads": list(reads), "name_charts": {}}


def test_cards_are_capped_ranked_and_end_with_the_lesson():
    trades = [_trade(n, 50.0 * n) for n in range(1, 7)]
    picks = [_pick(f"M{n}", "real_miss", 5.0 * n) for n in range(1, 6)]
    picks += [_pick(f"G{n}", "good_pass_failed", -2.0 * n) for n in range(1, 4)]
    reads = [{"stamp": f"{SESSION}T09:40:00-04:00", "horizon": "m5", "direction": "bullish", "verdict": "right"}]
    out = cards.build_cards(SESSION, _payload(trades, reads), findability={"picks": picks}, environment={})

    assert len(out) == cards.WALK_CAP
    assert out[-1]["kind"] == "lesson"
    ranked = [c["value"] for c in out[:-1]]
    assert ranked == sorted(ranked, reverse=True)
    assert sum(1 for c in out if c["kind"] == "miss") <= cards.MISS_MAX
    assert sum(1 for c in out if c["kind"] == "good_pass") <= cards.GOOD_PASS_MAX


def test_the_biggest_real_miss_outranks_a_plain_trade_and_the_misses_keep_the_biggest_runs():
    trades = [_trade(1, 20.0)]
    picks = [_pick(f"M{n}", "real_miss", float(n)) for n in (2, 9, 30, 4)]
    out = cards.build_cards(SESSION, _payload(trades), findability={"picks": picks}, environment={})
    kinds = [c["kind"] for c in out]
    assert kinds[0] == "miss" and out[0]["symbol"] == "M30"
    misses = [c["symbol"] for c in out if c["kind"] == "miss"]
    assert misses == ["M30", "M9", "M4"]  # max three, biggest runs first
    assert kinds[-1] == "lesson"
    assert "environment" in kinds


def test_a_small_day_still_walks_market_and_lesson():
    out = cards.build_cards(SESSION, _payload(), findability={}, environment={})
    assert [c["kind"] for c in out] == ["environment", "lesson"]


def test_trade_card_says_not_known_for_what_it_cannot_measure():
    card = cards.build_cards(SESSION, _payload([_trade(1, -40.0)]), findability={}, environment={})[0]
    assert card["kind"] == "trade"
    text = " ".join(card["lines"]) + card["what_if"]
    assert "not known" in text
    assert "0.00R" not in text


def test_what_if_reads_the_tape_after_the_exit():
    bars = [
        {"dt": f"{SESSION}T10:00:00", "open": 100, "high": 101, "low": 99.5, "close": 100.5},
        {"dt": f"{SESSION}T10:05:00", "open": 100.5, "high": 103, "low": 100, "close": 102},
        {"dt": f"{SESSION}T11:00:00", "open": 102, "high": 102.5, "low": 101, "close": 101.5},
        {"dt": f"{SESSION}T15:55:00", "open": 101.5, "high": 105, "low": 101, "close": 104},
    ]
    trade = _trade(1, 100.0, stop=99.0, entry=100.0, exit_=101.0)
    trade["opened_at"] = f"{SESSION}T10:00:00-04:00"
    values = cards.what_if(trade, bars)
    assert values["unit"] == "R"
    assert values["held_to_close"] == pytest.approx(4.0)
    assert values["held_to_stop"] == pytest.approx(4.0)  # never stopped: held to the close
    assert values["mfe"] == pytest.approx(3.0)
    assert values["mae"] == pytest.approx(-0.5)


def test_progress_text_counts_cards_and_minutes():
    out = cards.build_cards(SESSION, _payload([_trade(1, 5.0)]), findability={}, environment={})
    text = cards.progress_text(out, 0)
    assert text.startswith(f"1 of {len(out)} · ")
    assert "min left" in text


# -- the Mentor ---------------------------------------------------------------
def _mentor_state(trades) -> dict:
    return {
        "session": SESSION, "auto_mode": "", "trades": trades, "open_positions": [],
        "decisions": [], "claims": [], "focus_adds": [], "armed": [],
        "likes": [{"event_id": "like1", "symbol": "SYM1", "like_mode": "quick", "matched_trade_id": "t1"}],
        "exit_drafts": [], "answered": {}, "retired": [], "carried": (),
    }


def test_the_trade_card_asks_the_awake_trade_origin_question():
    trades = [_trade(1, 10.0)]
    asked = cards.mentor_by_trade(_mentor_state(trades), ["t1"])
    kinds = sorted(s.kind for s in asked.get("t1", []))
    assert kinds == ["trade_origin"], kinds


def test_a_dormant_kind_is_never_asked(monkeypatch):
    import mentor_questions

    registry = tuple(
        replace(kind, dormant_until="TEST") if kind.kind == "trade_origin" else kind
        for kind in mentor_questions.REGISTRY
    )
    monkeypatch.setattr(mentor_questions, "REGISTRY", registry)
    asked = cards.mentor_by_trade(_mentor_state([_trade(1, 10.0)]), ["t1"])
    assert asked == {}


def test_an_answered_or_retired_question_is_not_asked_again():
    state = _mentor_state([_trade(1, 10.0)])
    state["answered"] = {"trade_origin:t1": {"answered_at": SESSION}}
    assert cards.mentor_by_trade(state, ["t1"]) == {}
    state = _mentor_state([_trade(1, 10.0)])
    state["retired"] = ["trade_origin:t1"]
    assert cards.mentor_by_trade(state, ["t1"]) == {}


def test_mentor_questions_ride_their_trade_card_without_stop_asking():
    import mentor_questions

    trades = [_trade(1, 10.0)]
    asked = cards.mentor_by_trade(_mentor_state(trades), ["t1"])
    out = cards.build_cards(SESSION, _payload(trades), mentor_by_trade=asked)
    card = next(c for c in out if c["kind"] == "trade")
    assert [q["kind"] for q in card["mentor"]] == ["trade_origin"]
    values = [v for v, _label in card["mentor"][0]["options"]]
    assert "planned_off_the_desk" in values
    assert mentor_questions.STOP_ASKING not in values


def test_an_exit_draft_becomes_a_looks_like_line():
    import mentor_questions

    subject = mentor_questions.Subject(
        kind="exit_draft_review", subject_id="t1@2026-09-22",
        detail={"trade_id": "t1", "key": "t1@2026-09-22", "exit_session": SESSION,
                "fields": {"watching": [{"quote": "the VWAP"}]}, "raw_text": "lost vwap"},
    )
    out = cards.build_cards(SESSION, _payload([_trade(1, 10.0)]), mentor_by_trade={"t1": [subject]})
    card = next(c for c in out if c["kind"] == "trade")
    assert card["exit_draft"]["sentence"].startswith("Looks like")
    assert card["exit_draft"]["sentence"].endswith("right?")
    assert card["exit_note"] is None


def test_a_pending_call_is_not_counted_as_measured():
    reads = [
        {"stamp": f"{SESSION}T09:40:00-04:00", "horizon": "m5", "direction": "up", "verdict": "right"},
        {"stamp": f"{SESSION}T09:41:00-04:00", "horizon": "d1", "direction": "up", "verdict": "pending 2026-09-29"},
        {"stamp": f"{SESSION}T09:42:00-04:00", "horizon": "m5", "direction": "up", "verdict": "unmeasured:no_bars"},
    ]
    card = cards.calls_card(_payload(reads=reads))
    assert card["measured"] == 1
    assert card["title"].endswith("1 of 3 measured")


def test_a_swing_leg_on_another_day_shows_its_date():
    trade = _trade(1, 10.0)
    trade["opened_at"] = "2026-09-18T10:00:00-04:00"
    card = cards.trade_card(trade, _payload([trade]))
    assert card["lines"][0].startswith("Opened 09/18 10:00 · closed 11:00")
    assert "R not known" in card["lines"][1]
