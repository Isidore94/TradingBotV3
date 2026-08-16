"""R8 §9 step 2 - the M5 strength functions do not move while R8 shares them.

``scripts/strength_scan.py`` is not edited by R8. But `weekend_strength` imports
its pure functions, which turns them from one packet's private code into two
packets' shared surface - and a shared function is one that can now be broken
from a direction its own tests were never written to watch.

Bit-identical, not approximately: the fixture declares a 1e-09 tolerance, which
is float-comparison slack rather than a licence for the formula to drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from conftest import load_fixture_contract  # noqa: E402
from m5_strength_characterization import CASES, FIXTURE_NAME, capture  # noqa: E402


@pytest.fixture
def golden():
    return load_fixture_contract(FIXTURE_NAME)


def test_the_frozen_bars_are_the_ones_the_module_ships(golden):
    assert golden["bars"] == CASES
    assert golden.raw_input_digest() == golden["raw_input_sha256"]


@pytest.mark.parametrize("case", sorted(CASES))
def test_every_shared_function_answers_what_it_used_to(golden, case):
    measured = capture()
    expected = golden["measured"][case]
    for key, value in expected.items():
        golden.assert_matches(measured[case][key], value, context=f"{case}.{key}")


def test_the_percentile_cut_still_orders_the_same_way(golden):
    measured = capture()["_percentile"]
    expected = golden["measured"]["_percentile"]
    # JSON has no tuples; compare the names in order, which is what the cut means.
    for side in ("long_top_50pct", "short_bottom_50pct"):
        assert [row[0] for row in measured[side]] == [row[0] for row in expected[side]]


def test_short_history_still_refuses_rather_than_approximates(golden):
    """The property the whole board rests on.

    A row that cannot be measured is not a weak row, and scoring it as one would
    rank a data problem against real setups. 50 bars is deliberately in the list:
    it is exactly one short of what ATR50 needs.
    """
    measured = capture()["_refusals"]
    assert measured == golden["measured"]["_refusals"]
    assert set(measured.values()) == {None}


def test_the_capture_is_reproducible_within_one_process():
    """A golden built on anything non-deterministic is a coin flip, not a golden."""
    assert capture() == capture()


def test_r8_has_not_edited_the_m5_scanner():
    """The hard rule, checked rather than remembered.

    ``strength_scan.py`` is fenced by the spec's §2 and §8: R8 reimplements the
    ~20-line board orchestration in its own module and imports the formula. If
    this file ever needs editing, the spec says stop and ask - so this test
    exists to make "it seemed necessary" visible instead of quiet.
    """
    import subprocess

    base = subprocess.run(
        ["git", "merge-base", "HEAD", "4420bbf"],
        capture_output=True, text=True, cwd=ROOT_DIR,
    ).stdout.strip() or "4420bbf"
    changed = subprocess.run(
        ["git", "diff", "--name-only", base, "--", "scripts/strength_scan.py"],
        capture_output=True, text=True, cwd=ROOT_DIR,
    ).stdout.strip()
    assert changed == "", (
        "scripts/strength_scan.py is fenced by the R8 spec and must not be edited; "
        "stop and ask the trader first"
    )
