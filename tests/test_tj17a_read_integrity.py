"""TJ-17A — the Day/Week read summary must preserve what was measured.

The figures below are deliberately hand-counted.  They travel through the
append-only grade ledger, the Day Review owner, a packed card, and the weekly
re-cut so a scalar-only repair cannot make the page look correct while losing
the horizon data before Week Review receives it.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))


def _grade(
    grade_id: str,
    session: str,
    horizon: str,
    verdict: str,
    *,
    direction: str = "up",
) -> dict:
    """A stored, current clicked grade with the small real baseline context."""
    return {
        "grade_id": grade_id,
        "read_id": grade_id,
        "session": session,
        "source": "click",
        "horizon": horizon,
        "verdict": verdict,
        "direction": direction,
        "move_atr": 1.0,
        "context": {
            "availability": "measured",
            "last_hour_spy": "up",
            "d1_environment": "trending_up",
        },
        "read": {"horizon": horizon, "direction": direction, "confidence": "medium"},
    }


def _your_reads_line(card):
    return next(line for line in card.lines if line["key"] == "your_reads")


def _two_session_cards(tmp_path):
    """Two packed cards, written through the actual append-only ledger."""
    import day_report_card
    import market_read_grades as grades
    import prediction_ledger

    root = tmp_path / "day_review"
    first = "2026-09-21"
    second = "2026-09-22"
    grades.append_grades(
        first,
        [
            _grade("d1", first, "rest_of_day", grades.VERDICT_RIGHT),
            _grade("d2", first, "rest_of_day", grades.VERDICT_RIGHT),
            _grade("d3", first, "rest_of_day", grades.VERDICT_RIGHT),
            _grade("d4", first, "rest_of_day", grades.VERDICT_FLAT, direction="chop"),
            _grade("s1", first, "next_5_sessions", "pending 2026-09-28"),
        ],
        root=root,
    )
    grades.append_grades(
        second,
        [
            _grade("d5", second, "rest_of_day", grades.VERDICT_WRONG, direction="down"),
            _grade("s2", second, "next_5_sessions", grades.VERDICT_RIGHT),
            _grade("s3", second, "next_5_sessions", "pending 2026-09-29"),
        ],
        root=root,
    )
    first_tally = prediction_ledger.your_reads(first, root=root)
    first_card = day_report_card.build({"session": first, "your_reads": first_tally})
    second_card = day_report_card.build(
        {"session": second, "your_reads": prediction_ledger.your_reads(second, root=root)}
    )
    return first_tally, first_card, second_card


def test_closed_read_coverage_does_not_subtract_a_waiting_horizon_twice(tmp_path):
    """Four closed Sep. 21 calls stay four measured when one swing call waits."""
    _tally, first_card, _second_card = _two_session_cards(tmp_path)
    first_line = _your_reads_line(first_card)

    # `n` is already the finished/closed count. Pending is separate inventory.
    assert (first_line["n"], first_line["measured"], first_line["right"]) == (4, 4, 3)
    assert first_line["pending"] == 1
    assert first_line["unmeasured"] == 0


def test_read_horizons_survive_the_day_pack_and_week(tmp_path):
    """A rest-of-day result and a five-session result never become one rate.

    Sep. 21's real count shape is four closed rest-of-day calls (three right,
    one flat) and one still-open five-session call.  The next card adds one
    closed rest-of-day wrong and one closed five-session right.  Thus the week
    must retain *two* cells: day 3 right of 5 closed, swing 1 right of 1
    closed; the two open five-session calls remain waiting, outside both rates.
    """
    import day_report_card

    first_tally, first_card, second_card = _two_session_cards(tmp_path)
    first_line = _your_reads_line(first_card)

    assert first_tally["horizons"]["rest_of_day"]["accuracy"] == {
        "right": 3,
        "wrong": 0,
        "flat": 1,
        "pending": 0,
        "unmeasured": 0,
        "n": 4,
        "rate": pytest.approx(0.75),
        "rate_lb": first_line["horizons"]["rest_of_day"]["accuracy"]["rate_lb"],
        "meets_floor": False,
    }
    swing = first_line["horizons"]["next_5_sessions"]["accuracy"]
    assert (swing["n"], swing["right"], swing["flat"], swing["pending"], swing["unmeasured"], swing["rate"]) == (0, 0, 0, 1, 0, None)
    assert first_line["horizons"]["rest_of_day"]["baselines"]
    assert first_line["horizons"]["next_5_sessions"]["baselines"]

    packed_first = day_report_card.pack_card(first_card)
    packed_second = day_report_card.pack_card(second_card)
    assert _your_reads_line(packed_first)["horizons"] == first_line["horizons"]

    # Repeating the first packed session is ignored.  Stored-card pooling keeps
    # the two horizons apart: 3/5 day versus 1/1 five-session, never 4/6.
    week = day_report_card.week_from_cards([packed_first, packed_second, packed_first])
    week_line = _your_reads_line(week)
    assert (week_line["n"], week_line["measured"], week_line["right"], week_line["flat"], week_line["pending"]) == (6, 6, 4, 1, 2)
    day = week_line["horizons"]["rest_of_day"]
    five = week_line["horizons"]["next_5_sessions"]
    assert (day["n"], day["right"], day["wrong"], day["flat"], day["pending"], day["rate"]) == (5, 3, 1, 1, 0, pytest.approx(0.6))
    assert (five["n"], five["right"], five["wrong"], five["flat"], five["pending"], five["rate"]) == (1, 1, 0, 0, 2, pytest.approx(1.0))


def test_a_legacy_scalar_card_is_named_unseparated_and_cannot_supply_a_horizon_rate():
    """Old packed cards cannot be reverse-classified from a pooled total."""
    import day_report_card

    legacy = {
        "session": "2026-09-18",
        "lines": ({"key": "your_reads", "text": "Your reads: 1 right of 2", "n": 2, "measured": 2, "right": 1},),
    }
    week = day_report_card.week_from_cards([legacy])
    line = _your_reads_line(week)

    assert line["unseparated_sessions"] == ("2026-09-18",)
    assert "unseparated" in line["text"].lower()
    for horizon in ("rest_of_day", "next_5_sessions"):
        assert line["horizons"][horizon]["n"] == 0
        assert line["horizons"][horizon]["rate"] is None


def test_contrast_encodes_only_measured_canonical_internals_without_mutating_rows():
    """Recorded v2 internals become bounded contrast features, including zero/False."""
    import trade_mentor_context
    from ai_jobs import prediction_contrast
    import tj14a_support as fixture

    snapshot = fixture.context()  # The real v2 builder, not a hand-shaped nested dict.
    row = {
        "source": "click",
        "verdict": "right",
        "context": {"internals": snapshot, "hour": 10, "d1_environment": "trending_up"},
    }
    original = copy.deepcopy(row)
    encoded = prediction_contrast.encode_rows([row])[0]

    assert row == original, "encoding is a reader and never rewrites a recorded grade"
    assert encoded["internals.breadth.value"] == pytest.approx(0.5)
    assert encoded["internals.fear.vxx_direction:down"] == 1.0
    assert encoded["internals.fear.spy_direction:up"] == 1.0
    assert encoded["internals.fear.divergence:False"] == 1.0
    assert encoded["internals.rates.value"] == pytest.approx(0.8)
    assert encoded["internals.oil.value"] == pytest.approx(-2.1)
    assert encoded["internals.offense_vs_defense.value"] == pytest.approx(2.0)
    assert encoded["internals.sectors_above_vwap.count"] == 9.0
    assert encoded["internals.sectors_above_vwap.denominator"] == float(len(trade_mentor_context.SECTORS))
    assert encoded["internals.sector_leaders.day:XLE"] == 1.0
    assert encoded["internals.sector_laggards.m30:XLF"] == 1.0
    assert all(key.startswith(("internals.", "hour", "d1_environment:")) for key in encoded)

    unavailable = copy.deepcopy(row)
    unavailable["context"]["internals"]["derived"]["breadth"]["status"] = "unmeasured"
    unavailable["context"]["internals"]["derived"]["breadth"]["value"] = 999.0
    encoded_unavailable = prediction_contrast.encode_rows([unavailable])[0]
    assert "internals.breadth.value" not in encoded_unavailable
