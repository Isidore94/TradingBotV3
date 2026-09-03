"""The desk's strength scan equals the trader's TC2000 scan — V1 item 1.

Decision 0016 answer 9: the trader's own scan is the SPECIFICATION for this
board, so "does the desk agree with TC2000?" has to be a test rather than an
impression. The golden's expected values are computed in
`tests/build_tc2000_parity_fixture.py` by a second, naive implementation written
straight from the trader's two formula lines — not by calling the module these
tests check, which would pin whatever the module does including its mistakes.

Seven symbols, sixteen sessions of M5 bars each. Sixteen because the relative
volume compares each of the last twelve bars with the same offset over the prior
FIFTEEN sessions; under the old five-day fetch every RVOL on the board would have
been blank.

R4 A7 added the two symbols the first five could not catch. AAA-EEE are clean
78-bar sessions, and on those a flat positional stride and a session-relative one
give the SAME answer - which is why their pinned values are unchanged by A7 and
why they could never have found the defect. `FFF` carries one early-close session
and `GGG` one missing bar, and both have a volume series that is a pure function
of the time of day, so the honest answer is 1.00 and any deviation is the
alignment rather than the tape.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

FIXTURE_NAME = "tc2000_parity_v1"


@pytest.fixture(scope="module")
def golden():
    from conftest import load_fixture_contract

    contract = load_fixture_contract(FIXTURE_NAME)
    return contract.data


def test_the_fixture_meets_the_milestone_3_contract(golden):
    """Loading it validates the contract; this states that it was checked."""
    assert golden["schema"] == "tc2000_parity_v1"
    assert golden["numeric_tolerance"] == pytest.approx(1e-4)
    assert golden["raw_input_keys"] == ["bars"]
    assert set(golden["bars"]) == {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"}


def test_the_strength_score_matches_the_hand_computation(golden):
    from strength_scan import strength_score

    for symbol, bars in golden["bars"].items():
        measured = strength_score(bars)
        expected = golden["expected"][symbol]["strength"]
        assert measured == pytest.approx(expected, abs=1e-4), symbol


def test_the_relative_volume_matches_the_hand_computation(golden):
    """`AVG(V / mean(V at the same bar offset over the prior 15 sessions), 12)`."""
    from strength_scan import relative_volume

    for symbol, bars in golden["bars"].items():
        measured = relative_volume(bars)
        expected = golden["expected"][symbol]["rvol"]
        assert measured is not None, symbol
        assert measured == pytest.approx(expected, abs=1e-4), symbol


def test_a_flat_day_reads_as_a_relative_volume_of_about_one(golden):
    """A sanity anchor the trader can check by eye.

    EEE trades the same volume today as on every prior session, so its ratio is
    the ratio of today's flat volume to the prior mean - which the generator
    varies by bar-of-day, so it is near 1 rather than exactly 1. A number wildly
    off 1 here would mean the offsets are lining up on the wrong bars.
    """
    from strength_scan import relative_volume

    measured = relative_volume(golden["bars"]["EEE"])
    assert 0.5 < measured < 1.5, measured


def test_too_little_history_is_blank_and_never_zero(golden):
    """A zero would rank the name at the bottom of a filter it was never in.

    "Too little history" is counted in SESSIONS since R4 A7, because the offset
    is counted inside a session. Fifteen prior sessions or the answer is blank;
    a session that is merely SHORT is a different thing and is handled by
    dropping the offsets it never reached, never by blanking the symbol.
    """
    from strength_scan import relative_volume

    bars = golden["bars"]["AAA"]
    assert relative_volume(bars) is not None
    # Fourteen prior sessions is not fifteen.
    assert relative_volume(bars[78 * 2 :]) is None
    assert relative_volume(bars[:100]) is None
    assert relative_volume([]) is None


def test_an_early_close_does_not_move_a_time_of_day_volume_series(golden):
    """The reproduction, pinned. FFF's volume depends only on the time of day.

    So the honest relative volume is exactly 1.00 on every bar, and anything
    else is the alignment rather than the tape. V1's flat positional stride
    reads 1.2949 here: one 39-bar early close shifts every offset past it, so a
    late bar is compared with a middling one.
    """
    from strength_scan import relative_volume

    measured = relative_volume(golden["bars"]["FFF"])
    assert measured == pytest.approx(1.0, abs=1e-6), measured
    assert golden["expected"]["FFF"]["rvol"] == pytest.approx(1.0, abs=1e-6)


def test_a_positional_stride_gets_the_early_close_wrong(golden):
    """The old rule, run here so the difference is a number and not a claim."""
    volumes = [bar["volume"] for bar in golden["bars"]["FFF"]]
    ratios = []
    for step in range(12):
        index = len(volumes) - 1 - step
        prior = [volumes[index - 78 * back] for back in range(1, 16)]
        ratios.append(volumes[index] / (sum(prior) / 15))
    positional = sum(ratios) / len(ratios)

    assert positional == pytest.approx(1.2949, abs=1e-4), positional
    assert positional != pytest.approx(1.0, abs=0.01), (
        "a series whose volume is a pure function of the time of day has a "
        "relative volume of exactly 1.00; this is the error the stride adds"
    )


def test_a_prior_session_missing_a_bar_contributes_nothing_and_never_a_zero(golden):
    """GGG drops one bar from one prior session. The number stays measured.

    A zero in the denominator's mean would read as "that day was dead at that
    minute", which is a claim about volume rather than about a gap in the tape.
    The residual is stated rather than hidden: the offset is the bar's INDEX
    inside its session, so a hole shifts that session's later offsets by one -
    seven basis points here, against the 29% a 39-bar early close cost the
    positional stride.
    """
    from strength_scan import relative_volume

    measured = relative_volume(golden["bars"]["GGG"])
    assert measured is not None
    assert measured == pytest.approx(golden["expected"]["GGG"]["rvol"], abs=1e-4)
    assert 0.99 < measured < 1.01, measured


def test_a_halted_prior_window_is_unmeasurable_rather_than_infinite(golden):
    """Dividing by no volume is not a very large relative volume."""
    from strength_scan import relative_volume

    bars = [dict(bar) for bar in golden["bars"]["AAA"]]
    for bar in bars[:-12]:
        bar["volume"] = 0.0
    assert relative_volume(bars) is None


def _materialised(bars):
    """The fixture stores `dt` as ISO text; the pipeline hands over datetimes.

    `strength_score` never looks at `dt`. `relative_volume` and `score_symbol`
    both group by session, and since R4 A7 `_session_groups` parses ISO text as
    well as datetimes - it had to, or every stored bar fell into ONE session and
    the relative volume compared today's bar with itself. This helper stays
    because the pipeline hands over real datetimes and the fixture should be
    read both ways.
    """
    from datetime import datetime

    return [{**bar, "dt": datetime.fromisoformat(bar["dt"])} for bar in bars]


def test_the_score_is_unchanged_by_v1(golden):
    """V1 added RVOL and floors; it must not have moved the strength number.

    The golden's `strength` values come from the hand implementation, so this
    passing means the formula in `strength_scan.py` is still the trader's - which
    is the same thing the fenced-function test asserts textually.
    """
    from strength_scan import score_symbol

    for symbol, bars in golden["bars"].items():
        row = score_symbol(symbol, _materialised(bars))
        assert row is not None, symbol
        assert row["strength"] == pytest.approx(
            golden["expected"][symbol]["strength"], abs=1e-4
        )
        # And the row now carries the new measurements beside it.
        assert row["rvol"] == pytest.approx(golden["expected"][symbol]["rvol"], abs=1e-4)
        assert row["session_volume"] is not None


def _load_raw() -> dict:
    return json.loads(
        (ROOT / "tests" / "fixtures" / f"{FIXTURE_NAME}.json").read_text(encoding="utf-8")
    )


def test_the_builder_still_reproduces_the_fixture():
    """A golden nobody can regenerate is a number nobody can check."""
    import build_tc2000_parity_fixture as builder

    rebuilt = builder.build_payload()
    stored = _load_raw()
    assert rebuilt["raw_input_sha256"] == stored["raw_input_sha256"]
    assert rebuilt["expected"] == stored["expected"]
