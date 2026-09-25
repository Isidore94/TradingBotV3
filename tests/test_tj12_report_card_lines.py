"""TJ-12 item 1 - the six lines, their order, and what each one counts.

`plan.md` §12.4 TJ-12 with its AMENDED 2026-09-19 block, and decision 0021
answers 17, 22, 24 and 27. The trader's words: *"I want what I missed to be very
apparent. I want what I did well with to also be very apparent."*

THE CONTRACT THESE TESTS PIN (the builder may add keys, never remove one)
------------------------------------------------------------------------
``scripts/day_report_card.py`` - PURE. It opens no store, reads no clock, calls
no model and touches no Qt.

    LINE_KEYS = ("did_well", "missed", "your_reads", "congruence",
                 "process", "how_fresh")
        The SIX lines, in the order the packet names them. `how_fresh` is the
        sixth and smaller one the trader added on the second look.

    LINE_TARGETS: {key: target}
        What a click on that line opens. `did_well` -> `liked_not_traded` and
        `missed` -> `rejected` are `walkaway_day.TABLES` names, because the
        table under the line is the one the count came from.

    build(day_inputs) -> ReportCard
        `.session`, `.lines` - a tuple of six mappings, each
        `{key, text, n, measured, target}`.

        `n`        how many things the line is ABOUT (the population).
        `measured` how many of those the desk could actually measure.

        n - measured is never printed as a zero and never enters a rate
        (`plan.md` §12.3: missing data is uncertainty, never confirmation).

``day_inputs`` is a mapping of things the Day Review worker ALREADY read:
``session``, ``walkaway`` (a `walkaway_day.WalkawayDay`), ``your_reads``
(`prediction_ledger.your_reads`' dict), ``congruence``
(`market_read_grades.congruence_lines`' tuple), ``trades``, ``origin_lanes``
(the four lanes `trade_origin.planned_state` reads) and ``freshness``.

Every number asserted here is hand-counted in `tests/tj12_support.py`'s module
docstring or read back out of the owner that computed it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import tj12_support as fx  # noqa: E402

PLAN_ORDER = ("did_well", "missed", "your_reads", "congruence", "process", "how_fresh")


@pytest.fixture()
def card(tmp_path):
    import day_report_card

    fx.one_session_of_clicks(tmp_path)
    return day_report_card.build(fx.day_inputs(tmp_path))


def _line(card, key):
    for line in card.lines:
        if line["key"] == key:
            return line
    raise AssertionError(f"no {key!r} line on the card: {[item['key'] for item in card.lines]}")


# ---------------------------------------------------------------------------
# the six lines and their order
# ---------------------------------------------------------------------------
def test_the_card_is_six_lines_in_the_order_the_packet_names():
    import day_report_card

    assert tuple(day_report_card.LINE_KEYS) == PLAN_ORDER


def test_the_built_card_carries_those_six_lines_in_that_order(card):
    assert tuple(line["key"] for line in card.lines) == PLAN_ORDER


def test_every_line_carries_its_key_text_n_measured_and_target(card):
    import walkaway_day

    for line in card.lines:
        assert set(line) >= {"key", "text", "n", "measured", "target"}, line["key"]
        assert isinstance(line["n"], int), line["key"]
        assert isinstance(line["measured"], int), line["key"]
        assert line["text"].strip(), f"{line['key']} says nothing"
        assert line["measured"] <= line["n"], line["key"]
        assert line["target"], line["key"]
    assert _line(card, "did_well")["target"] in walkaway_day.TABLES
    assert _line(card, "missed")["target"] in walkaway_day.TABLES


# ---------------------------------------------------------------------------
# Did well
# ---------------------------------------------------------------------------
def test_did_well_counts_the_runs_among_the_likes_and_never_reads_a_blank_as_zero(card):
    """Four likes: AAA and BBB ran, CCC did not, DDD could not be measured.

    So the line is about FOUR and measured THREE - and the fourth is stated as
    unmeasured, never folded into the denominator as a miss.
    """
    line = _line(card, "did_well")
    assert line["n"] == 4
    assert line["measured"] == 3
    assert "2" in line["text"], line["text"]
    assert "unmeasured" in line["text"].lower()


def test_did_well_quotes_the_skill_line_so_a_count_never_stands_alone(card):
    """The AMENDED block: both lines quote TJ-11's skill line.

    Its sentence is `walkaway_day`'s own, and the card prints it rather than
    rebuilding a base rate of its own.
    """
    sentence = fx.skill_window()["sentence"]
    assert sentence in _line(card, "did_well")["text"], _line(card, "did_well")["text"]


def test_did_well_names_the_family_with_the_best_wilson_bound_not_the_best_rate(card):
    """`steady` 55/100 (bound ~0.452) against `flashy` 18/30 (rate 0.60, bound ~0.423).

    A card ranked on the RATE names `flashy`. The packet says "best family by
    the ONE Wilson bound", so it names `steady`, and the bound it printed is the
    cell's own `low` - not a second Wilson computed here.
    """
    text = _line(card, "did_well")["text"]
    assert fx.BEST_BOUND_FAMILY in text, text
    assert fx.BEST_RATE_FAMILY not in text, text

    window = fx.skill_window()
    best = fx.family_cell(window, fx.BEST_BOUND_FAMILY)
    worse = fx.family_cell(window, fx.BEST_RATE_FAMILY)
    assert best["low"] > worse["low"]
    assert best["rate"] < worse["rate"], "the fixture has to tempt the wrong rule"
    assert _line(card, "did_well")["best_family"] == {
        "setup_family": fx.BEST_BOUND_FAMILY,
        "runs": best["runs"],
        "measured": best["measured"],
        "low": best["low"],
    }


def test_a_family_under_the_reporting_floor_is_never_named(card):
    """`tiny` is 9 of 10 - the best rate on the board and under the floor."""
    import evidence_stats

    assert fx.FAMILY_PLAN[fx.UNDER_FLOOR_FAMILY][1] < evidence_stats.MIN_REPORTABLE_N
    for line in card.lines:
        assert fx.UNDER_FLOOR_FAMILY not in line["text"], line["key"]


def test_no_family_is_named_at_all_when_none_clears_the_floor(tmp_path):
    import day_report_card
    import walkaway_day

    fx.one_session_of_clicks(tmp_path)
    inputs = fx.day_inputs(tmp_path)
    thin = walkaway_day.WalkawayDay(
        liked_not_traded=inputs["walkaway"].liked_not_traded,
        rejected=inputs["walkaway"].rejected,
        skill={"session": {"cells": (), "overlapping": (), "sentence": "Of this scan: no scan rows to compare against."},
               "lately": None},
        sentences=inputs["walkaway"].sentences,
        money=inputs["walkaway"].money,
    )
    inputs["walkaway"] = thin
    line = _line(day_report_card.build(inputs), "did_well")
    assert line.get("best_family") is None
    assert "too few" in line["text"].lower(), line["text"]


# ---------------------------------------------------------------------------
# Missed
# ---------------------------------------------------------------------------
def test_missed_counts_the_real_misses_among_rejections_and_names_their_reason(card):
    """Five vetoes: EEE, FFF and GGG ran, HHH did not, III was unmeasured.

    Three of them share the coded reason `extended` - `walkaway_day`'s own
    `_reason_clause`, quoted, never recounted here.
    """
    line = _line(card, "missed")
    assert line["n"] == 5
    assert line["measured"] == 4
    assert "3" in line["text"]
    assert "extended" in line["text"], line["text"]
    assert fx.skill_window()["sentence"] in line["text"]


def test_missed_never_prints_a_rate_over_the_unmeasured_row(card):
    line = _line(card, "missed")
    assert line["measured"] == 4 and line["n"] == 5
    assert "5 of 5" not in line["text"], line["text"]


# ---------------------------------------------------------------------------
# Your reads
# ---------------------------------------------------------------------------
def test_your_reads_prints_both_integers_with_their_own_n(card):
    """Hand-counted: the trader 3 right of 4; `same as the last hour` 2 of 2."""
    line = _line(card, "your_reads")
    assert line["n"] == 4
    assert line["measured"] == 4
    assert "3" in line["text"] and "4" in line["text"]
    assert "2 of 2" in line["text"], line["text"]


def test_more_right_answers_on_a_bigger_base_is_never_called_beating_a_baseline(card):
    """3 right of 4 is 75%; 2 of 2 is 100%. The trader did NOT beat it.

    `prediction_ledger.your_reads` names the baseline with the most `right`, so
    the page that owns the wording has to compare the RATES before it says
    anybody won - and here nobody did.
    """
    text = _line(card, "your_reads")["text"].lower()
    assert "beat" not in text, text
    assert "better than" not in text, text


def test_a_session_with_no_graded_reads_says_so_and_never_zero(tmp_path):
    import day_report_card

    inputs = fx.day_inputs(tmp_path, with_reads=False)
    line = _line(day_report_card.build(inputs), "your_reads")
    assert line["n"] == 0
    assert line["measured"] == 0
    assert "no graded reads" in line["text"].lower(), line["text"]
    assert "0%" not in line["text"], line["text"]


# ---------------------------------------------------------------------------
# Congruence
# ---------------------------------------------------------------------------
def test_congruence_says_too_few_to_call_and_never_agreement(card):
    """Three D1 likes, all LONG, against an `up` read.

    The direction matches, and three is under `MIN_REPORTABLE_N`, so
    `market_read_grades` answered `too_few`. A card that read the match as
    agreement would tell the trader they are consistent on three names.
    """
    line = _line(card, "congruence")
    assert line["n"] == 3, "three congruence lines were handed in"
    assert line["measured"] == 2, "the fills line is unmeasured - no fills today"
    text = line["text"].lower()
    assert "too few" in text, text
    assert "picks" in text or "likes" in text, text


def test_congruence_names_the_missing_side_rather_than_reading_it_as_agreement(card):
    assert "unmeasured" in _line(card, "congruence")["text"].lower()


# ---------------------------------------------------------------------------
# Process
# ---------------------------------------------------------------------------
def test_process_counts_planned_and_unplanned_from_trade_origin(card):
    """AAA and BBB were spoken about before their fills; ZZZ was not.

    YYY's only stamp is midnight market-local - a broker file is blind to time -
    so `trade_origin` refuses to place it and the card says `unmeasured`.
    """
    line = _line(card, "process")
    assert line["n"] == 4
    assert line["measured"] == 3
    assert line["planned"] == 2
    assert line["unplanned"] == 1
    assert line["unmeasured"] == 1
    assert "unmeasured" in line["text"].lower()


def test_process_prints_the_label_provenance_mix_and_the_unlabelled_count(card):
    """Two confirmed labels, one made before entry and one the same session.

    Two rows carry the key PRESENT and EMPTY - an old row, not a missing one -
    and they count as unlabelled, never as a third provenance.
    """
    import trade_origin

    line = _line(card, "process")
    assert line["label_provenance"] == {
        trade_origin.CLAIMED_BEFORE_ENTRY: 1,
        trade_origin.SAME_SESSION: 1,
        trade_origin.RECALLED_AFTER: 0,
    }
    assert line["unlabelled"] == 2


def test_an_option_is_not_judged_here(card):
    """Decision 0021 answer 24: an option gets its own rule or is not judged."""
    line = _line(card, "process")
    assert "not judged here" in line["text"].lower(), line["text"]


def test_a_day_with_no_trades_says_so_rather_than_printing_zero(tmp_path):
    """Live gate #157's second clause, in a unit test."""
    import day_report_card

    fx.one_session_of_clicks(tmp_path)
    inputs = fx.day_inputs(tmp_path, with_trades=False)
    line = _line(day_report_card.build(inputs), "process")
    assert line["n"] == 0
    assert "no trades" in line["text"].lower(), line["text"]
    assert "0%" not in line["text"], line["text"]


# ---------------------------------------------------------------------------
# a missing input
# ---------------------------------------------------------------------------
def test_a_line_whose_input_is_missing_says_so_and_prints_no_number(tmp_path):
    """Item 2: *a line whose input is missing SAYS so, never a zero.*

    This is what the trader sees FIRST: on 2026-09-20 the live
    `C:\\TradingBotData\\day_review` folder holds `bars/` and three
    `sessions/<date>/outcomes.json` files, and no `narration/`, no `reads/` and
    no `pack.json` at all.
    """
    import day_report_card

    inputs = fx.day_inputs(
        tmp_path, with_reads=False, with_trades=False, with_walkaway=False
    )
    built = day_report_card.build(inputs)
    assert tuple(line["key"] for line in built.lines) == PLAN_ORDER
    for line in built.lines:
        assert line["text"].strip(), line["key"]
        assert "%" not in line["text"], (line["key"], line["text"])
