"""TJ-10 item 4 - what a naive rule would have scored on the SAME stamps.

Packet `.claude/packets/TJ-10.md` item 4 ("a pure helper, used by TJ-12 and
TJ-16"); `plan.md` §12.4 "TJ-16" item 2 ("skill against naive baselines, never a
bare hit rate ... on the SAME stamps"). RED before the build.

The contract these tests pin
----------------------------

``BASELINES = ("always_up", "same_as_the_last_hour", "with_the_d1_environment")``

``baseline_reads(rows, name) -> list[dict]``
    The same read rows with the direction a naive rule would have given, read
    out of **each row's own context snapshot** and nothing else - so a baseline
    is point-in-time by construction and cannot look at a bar the trader could
    not see. One row in, one row out, in the same order, with the SAME
    ``stamp``, ``session``, ``benchmark`` and ``horizon``. Ungradable rows
    (``no_view``) stay ungradable: a baseline never answers a question the
    trader declined.

``accuracy(grades) -> dict``
    ``{"right", "wrong", "flat", "pending", "unmeasured", "n", "rate",
    "rate_lb"}``. Integer counts; ``n`` is the CLOSED horizons only (TJ-11:
    pending is printed and is in neither half); ``rate_lb`` is the ONE Wilson
    (`swing_headline.wilson_lower_bound`, z ``WILSON_Z``); a cell under
    ``evidence_stats.MIN_REPORTABLE_N`` is never NAMED as a leader.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

import tj10_support as fx  # noqa: E402



def _rows_with_context():
    """Three reads, each carrying the context its baseline has to read."""
    import market_read_grades as grades

    made = []
    for index, (direction, last_hour, label) in enumerate((
        ("down", "up", "trending_down"),
        ("up", "down", "trending_up"),
        ("no_view", "up", "compressed"),
    )):
        entry = fx.mentor_entry(
            direction=direction, horizon="rest_of_day", timeframe="M5",
            created_at=fx.STAMP + timedelta(minutes=60 * index),
            confidence="" if direction == "no_view" else "medium",
        )
        row = grades.read_rows([entry], session=fx.SESSION)[0]
        row["context"] = {
            "last_hour_spy": last_hour,
            "d1_environment": label,
        }
        made.append(row)
    return made


def test_the_three_baselines_are_named_once():
    import market_read_grades as grades

    assert grades.BASELINES == (
        "always_up", "same_as_the_last_hour", "with_the_d1_environment"
    )


def test_every_baseline_answers_on_exactly_the_traders_own_stamps():
    import market_read_grades as grades

    rows = _rows_with_context()

    for name in grades.BASELINES:
        baseline = grades.baseline_reads(rows, name)
        assert fx.stamps_of(baseline) == fx.stamps_of(rows), name
        assert [row["session"] for row in baseline] == [
            row["session"] for row in rows
        ], name
        assert [row["horizon"] for row in baseline] == [
            row["horizon"] for row in rows
        ], name


def test_always_up_says_up_on_every_stamp_the_trader_answered():
    import market_read_grades as grades

    baseline = grades.baseline_reads(_rows_with_context(), "always_up")

    assert [row["direction"] for row in baseline[:2]] == ["up", "up"]


def test_the_last_hour_baseline_reads_the_rows_own_context_and_no_bars():
    """Point-in-time by construction: the fact is already ON the row."""
    import market_read_grades as grades

    baseline = grades.baseline_reads(_rows_with_context(), "same_as_the_last_hour")

    assert [row["direction"] for row in baseline[:2]] == ["up", "down"]


def test_the_environment_baseline_maps_the_desks_own_label():
    """`trending_down` -> down, `trending_up` -> up, and a label with no
    direction (`compressed`, `mixed`, `unknown`) never invents one."""
    import market_read_grades as grades

    rows = _rows_with_context()
    baseline = grades.baseline_reads(rows, "with_the_d1_environment")

    assert baseline[0]["direction"] == "down"
    assert baseline[1]["direction"] == "up"
    assert grades.is_gradable(baseline[2]) is False


def test_a_no_view_stamp_is_never_answered_by_a_baseline():
    """A baseline measures skill on the questions the trader ANSWERED."""
    import market_read_grades as grades

    rows = _rows_with_context()

    for name in grades.BASELINES:
        baseline = grades.baseline_reads(rows, name)
        assert grades.is_gradable(baseline[2]) is False, name


def test_accuracy_counts_integers_and_leaves_pending_out_of_both_halves():
    """TJ-11: a pooled rate counts CLOSED horizons only; pending is printed."""
    import market_read_grades as grades
    import swing_headline

    rows = [
        {"verdict": grades.VERDICT_RIGHT},
        {"verdict": grades.VERDICT_RIGHT},
        {"verdict": grades.VERDICT_RIGHT},
        {"verdict": grades.VERDICT_WRONG},
        {"verdict": grades.VERDICT_FLAT},
        {"verdict": "pending 2026-09-25"},
        {"verdict": "unmeasured:no_completed_bars"},
    ]

    cell = grades.accuracy(rows)

    assert cell["right"] == 3
    assert cell["wrong"] == 1
    assert cell["flat"] == 1
    assert cell["pending"] == 1
    assert cell["unmeasured"] == 1
    # Five closed horizons: three right, one wrong, one flat.
    assert cell["n"] == 5
    assert cell["rate"] == pytest.approx(3 / 5)
    assert cell["rate_lb"] == pytest.approx(
        swing_headline.wilson_lower_bound(3, 5)
    )


def test_a_cell_under_the_floor_is_never_named_a_leader():
    import evidence_stats
    import market_read_grades as grades

    small = grades.accuracy([{"verdict": grades.VERDICT_RIGHT}] * 3)

    assert small["n"] == 3
    assert small["n"] < evidence_stats.MIN_REPORTABLE_N
    assert small["meets_floor"] is False


def test_a_cell_at_the_floor_meets_it():
    import evidence_stats
    import market_read_grades as grades

    rows = [{"verdict": grades.VERDICT_RIGHT}] * evidence_stats.MIN_REPORTABLE_N
    cell = grades.accuracy(rows)

    assert cell["n"] == evidence_stats.MIN_REPORTABLE_N
    assert cell["meets_floor"] is True
