"""Golden: the live M5 tier and alert text on 280 real September alerts (S9 / P14).

Frozen from a scratch copy of the live learning state (2026-09-25) and the
confirmed rows of `intraday_bounce_candidates.csv`, BEFORE S9 or P14 touched
anything. The live tier (`evaluate_bounce_quality`) must stay byte-identical.
"""

import sys
from pathlib import Path

from conftest import load_fixture_contract

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from bounce_bot_lib import learning  # noqa: E402

GOLDEN = load_fixture_contract("s9_p14_tier_golden_v1")
STATE = GOLDEN["raw"]["learning_state"]
CASES = GOLDEN["raw"]["cases"]
EXPECTED = GOLDEN["expected"]


def _quality(case):
    return learning.evaluate_bounce_quality(STATE, **case["inputs"])


def _message(case, quality):
    from bounce_bot_lib.legacy import _format_bounce_alert_message

    inputs = case["inputs"]
    return _format_bounce_alert_message(
        case["symbol"],
        inputs["direction"],
        inputs["bounce_types"],
        case["event_row"],
        quality,
        exit_note=case["exit_note"],
    )


def test_golden_covers_proven_and_every_tier():
    tiers = {EXPECTED[case["id"]]["quality"]["tier"] for case in CASES}
    assert tiers == {"S", "A", "B", "C", "D"}
    assert sum(1 for case in CASES if EXPECTED[case["id"]]["quality"]["proven"]) == 80
    assert len(CASES) == 280


def test_live_tier_is_byte_identical():
    for case in CASES:
        assert _quality(case) == EXPECTED[case["id"]]["quality"], case["id"]


def test_alert_text_is_byte_identical():
    for case in CASES:
        expected = EXPECTED[case["id"]]
        assert _message(case, expected["quality"]) == expected["message"], case["id"]
