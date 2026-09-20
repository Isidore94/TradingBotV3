"""TJ-12 wakes the two Mentor kinds whose reader it is (lead, 2026-09-20).

`scripts/mentor_questions.py` registers `trade_origin` and
`open_position_check` with `dormant_until="TJ-12"`, because a question is ASKED
only when its answer has a reader (decision 0021 answer 28, TJ-14B decision 1).
TJ-12 builds that reader: `day_report_card.process_line` reads the
``trade_origin`` answer and `day_report_card.long_hold_lines` the
``open_position_state`` one. So this packet clears the field on those TWO kinds
and fills their lane in `MainWindow._mentor_question_state`.

`grader_gap` and `quick_like_followup` stay dormant: plan.md §12.5 names TJ-10
(as a small follow-up nobody has done) and TJ-14C for them, and this packet
grants neither.

Nothing here lifts dormancy. `tests/tj14b_lift_dormancy.py` exists so the
tester's files can make a DORMANT kind fire; a WOKEN kind must reach a live
card without it, which is what these tests prove.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

from tj14b_support import (  # noqa: E402
    REVIEWED,
    SESSION,
    keys_of,
    slot_at,
    state,
    trade_row,
)

#: The two kinds TJ-12 wakes, and the answer each one files.
WOKEN = {"trade_origin": "trade_origin", "open_position_check": "open_position_state"}

#: The two that stay asleep, and the packet each still waits for.
STILL_DORMANT = {"grader_gap": "TJ-10", "quick_like_followup": "TJ-14C"}


# ---------------------------------------------------------------------------
# 1. the one field
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(WOKEN))
def test_the_two_kinds_tj12_builds_the_reader_for_are_awake(name):
    import mentor_questions

    assert mentor_questions.kind_named(name).dormant_until == "", (
        f"{name} names TJ-12 as its packet and TJ-12 has shipped its reader"
    )


@pytest.mark.parametrize("name,packet", sorted(STILL_DORMANT.items()))
def test_the_other_two_are_untouched(name, packet):
    """Waking is per-kind. A packet that cleared the field on all four would be
    asking three questions whose readers nobody has written."""
    import mentor_questions

    assert mentor_questions.kind_named(name).dormant_until == packet


@pytest.mark.parametrize("name", sorted(WOKEN))
def test_a_woken_kind_passes_the_registrys_own_consumer_probe(name):
    """The rule the registry exists for: nothing ASKED that nothing reads.

    A woken kind is no longer excused by `dormant`, so it must import AND touch
    its own `answer_key` - the sharper probe, not a text search.
    """
    import mentor_questions

    row = {item["kind"]: item for item in mentor_questions.consumer_report()}[name]

    assert row["dormant"] is False
    assert row["imports"] is True, row["reason"]
    assert row["reads"] is True, row["reason"]


def test_no_live_kind_is_left_without_a_reader():
    import mentor_questions

    broken = {
        row["kind"]: row["reason"]
        for row in mentor_questions.consumer_report()
        if not row["dormant"] and not (row["imports"] and row["reads"])
    }

    assert broken == {}


# ---------------------------------------------------------------------------
# 2. it reaches a live card, with nothing lifted
# ---------------------------------------------------------------------------
def test_an_unplanned_trade_is_asked_without_the_lift_helper():
    """The fixture `test_a_dormant_kind_never_reaches_a_live_card` used to prove
    the opposite of. Same shape, and now the question is asked."""
    import mentor_questions

    payload = state(trades=[trade_row("T-1", symbol="AAPL", day=REVIEWED)])

    result = mentor_questions.pending(payload, slot_at(SESSION, 11))

    assert ("trade_origin", "T-1") in keys_of(result.asked) | keys_of(result.carried)


def test_a_long_held_open_position_is_asked_without_the_lift_helper():
    import mentor_questions

    payload = state(
        open_positions=[trade_row("P-1", symbol="TLT", day="2026-08-17", status="OPEN")]
    )

    result = mentor_questions.pending(payload, slot_at(SESSION, 11))

    assert ("open_position_check", "P-1") in keys_of(result.asked) | keys_of(result.carried)


def test_a_planned_trade_is_still_never_asked():
    """`trade_origin.planned_state` is the ONE rule and waking changed none of
    it: a trade the trader spoke about before its first fill is not asked."""
    import mentor_questions

    trade = trade_row("T-1", symbol="AAPL", day=REVIEWED)
    payload = state(
        trades=[trade],
        decisions=[
            {
                "symbol": "AAPL",
                "side": "LONG",
                "created_at": f"{REVIEWED}T06:00:00-07:00",
                "verdict": "like",
            }
        ],
    )

    result = mentor_questions.pending(payload, slot_at(SESSION, 11))

    assert ("trade_origin", "T-1") not in keys_of(result.asked) | keys_of(result.carried)


def test_away_still_asks_nothing_at_all():
    import mentor_questions

    payload = state(
        auto_mode="AWAY",
        trades=[trade_row("T-1", symbol="AAPL", day=REVIEWED)],
        open_positions=[trade_row("P-1", symbol="TLT", day="2026-08-17", status="OPEN")],
    )

    result = mentor_questions.pending(payload, slot_at(SESSION, 11))

    assert result.asked == () and result.forced == () and result.carried == ()


def test_the_budget_of_three_still_holds_with_two_more_kinds_awake():
    """Waking adds questions to the queue, never to the CARD. Over budget is
    carried - counted on the card, never dropped, never a fourth."""
    import mentor_questions

    payload = state(
        trades=[
            trade_row("T-1", symbol="AAPL", day=REVIEWED),
            trade_row("T-2", symbol="MSFT", day=REVIEWED),
            trade_row("T-3", symbol="NVDA", day=REVIEWED),
        ],
        open_positions=[
            trade_row("P-1", symbol="TLT", day="2026-08-17", status="OPEN"),
            trade_row("P-2", symbol="GLD", day="2026-08-17", status="OPEN"),
        ],
    )

    result = mentor_questions.pending(payload, slot_at(SESSION, 11))

    assert len(result.asked) == mentor_questions.BUDGET == 3
    assert len(result.carried) == 2
    assert not (keys_of(result.asked) & keys_of(result.carried))


# ---------------------------------------------------------------------------
# 3. the lane the desk fills
# ---------------------------------------------------------------------------
def test_the_desk_fills_the_origin_lanes_it_can_read(monkeypatch, tmp_path):
    """`MainWindow._mentor_question_state` names five lanes and TJ-14B left the
    origin ones EMPTY, because nothing read them. Empty lanes would now ask
    about every trade the desk has: `planned_state` answers `unplanned` when
    nothing was said, and "nothing was said" is what an unread store looks like.
    """
    pytest.importorskip("PySide6", reason="the desk window needs PySide6")
    import claimed_picks
    import ui.app as app_module
    from ui.app import MainWindow

    annotations = [
        {"symbol": "AAPL", "side": "LONG", "created_at": f"{REVIEWED}T06:00:00-07:00"}
    ]
    claims = [
        {
            "symbol": "MSFT",
            "side": "LONG",
            "claim_at_utc": f"{REVIEWED}T13:00:00+00:00",
            "session_date": REVIEWED,
            "claimed_setup_id": "steady",
        }
    ]
    monkeypatch.setattr(
        MainWindow, "_mentor_annotation_lane", staticmethod(lambda _days: list(annotations))
    )
    monkeypatch.setattr(claimed_picks, "load_rows", lambda *_a, **_k: list(claims))

    lanes = app_module.MainWindow._mentor_origin_lanes((SESSION.isoformat(), REVIEWED))

    assert [row["symbol"] for row in lanes["decisions"]] == ["AAPL"]
    assert [row["symbol"] for row in lanes["claims"]] == ["MSFT"]
    assert "focus_adds" in lanes and "armed" in lanes


def test_a_lane_that_cannot_be_read_is_empty_and_never_raises(monkeypatch):
    """An unreadable store asks MORE questions, never fewer answers - and it
    never takes the Mentor card down with it."""
    pytest.importorskip("PySide6", reason="the desk window needs PySide6")
    import claimed_picks
    from ui.app import MainWindow

    def _boom(*_args, **_kwargs):
        raise OSError("the store is not mounted")

    monkeypatch.setattr(MainWindow, "_mentor_annotation_lane", staticmethod(_boom))
    monkeypatch.setattr(claimed_picks, "load_rows", _boom)

    lanes = MainWindow._mentor_origin_lanes((SESSION.isoformat(),))

    assert lanes == {"decisions": [], "claims": [], "focus_adds": (), "armed": ()}
