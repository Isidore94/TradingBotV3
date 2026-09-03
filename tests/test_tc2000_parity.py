"""The desk's strength scan equals the trader's TC2000 scan — V1 item 1.

Decision 0016 answer 9: the trader's own scan is the SPECIFICATION for this
board, so "does the desk agree with TC2000?" has to be a test rather than an
impression. The golden's expected values are computed in
`tests/build_tc2000_parity_fixture.py` by a second, naive implementation written
straight from the trader's two formula lines — not by calling the module these
tests check, which would pin whatever the module does including its mistakes.

Five symbols, sixteen sessions of M5 bars each. Sixteen because the relative
volume compares each of the last twelve bars with the same offset over the prior
FIFTEEN sessions; under the old five-day fetch every RVOL on the board would have
been blank.
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
    assert set(golden["bars"]) == {"AAA", "BBB", "CCC", "DDD", "EEE"}


def test_the_strength_score_matches_the_hand_computation(golden):
    from strength_scan import strength_score

    for symbol, bars in golden["bars"].items():
        measured = strength_score(bars)
        expected = golden["expected"][symbol]["strength"]
        assert measured == pytest.approx(expected, abs=1e-4), symbol


def test_the_relative_volume_matches_the_hand_computation(golden):
    """`AVG(V / mean(V78, V156, ... V1170), 12)`, to four decimals."""
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
    """A zero would rank the name at the bottom of a filter it was never in."""
    from strength_scan import relative_volume

    bars = golden["bars"]["AAA"]
    assert relative_volume(bars) is not None
    assert relative_volume(bars[-1181:]) is None, "one bar short must be blank"
    assert relative_volume(bars[:100]) is None
    assert relative_volume([]) is None


def test_a_halted_prior_window_is_unmeasurable_rather_than_infinite(golden):
    """Dividing by no volume is not a very large relative volume."""
    from strength_scan import relative_volume

    bars = [dict(bar) for bar in golden["bars"]["AAA"]]
    for bar in bars[:-12]:
        bar["volume"] = 0.0
    assert relative_volume(bars) is None


def _materialised(bars):
    """The fixture stores `dt` as ISO text; the pipeline hands over datetimes.

    `strength_score` and `relative_volume` never look at `dt`, which is why they
    read the fixture as it stands. `score_symbol` groups by session and needs
    real datetimes - without them every bar falls into one session and the row
    is refused for having no prior session to compare with.
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
