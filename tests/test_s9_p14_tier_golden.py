"""Golden: the live M5 tier and alert text on 280 real September alerts (S9 / P14).

Frozen from a scratch copy of the live learning state (2026-09-25) and the
confirmed rows of `intraday_bounce_candidates.csv`, BEFORE S9 or P14 touched
anything. The live tier (`evaluate_bounce_quality`) must stay byte-identical.

P14 (trader 2026-09-26, "retire PROVEN"): every alert whose text carried no
proven evidence is byte-identical; each of the 80 that did loses the
"PROVEN " stamp and has its "proven: ..." part replaced by "grade <1:1 grade>"
from the frozen setup grades. Nothing else in the text moves.
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


def _grade_text(case):
    import setup_grades

    lookup = setup_grades.daytrade_lookup({"daytrade": GOLDEN["raw"]["daytrade_grades"]})
    inputs = case["inputs"]
    return learning.daytrade_grade_text(
        lookup, direction=inputs["direction"], bounce_types=inputs["bounce_types"]
    )


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
        grade_text=_grade_text(case) if quality.get("proven_reasons") else "",
    )


def _was_proven(case):
    return bool(EXPECTED[case["id"]]["quality"]["proven_reasons"])


def _p14_text(case):
    """The frozen text with the stamp dropped and the proven part replaced by the grade."""
    old = EXPECTED[case["id"]]["message"]
    tier = EXPECTED[case["id"]]["quality"]["tier"]
    assert old.startswith(f"[{tier}-TIER] PROVEN ")
    parts = old.replace(f"[{tier}-TIER] PROVEN ", f"[{tier}-TIER] ", 1).split(" | ")
    swapped = [f"grade {_grade_text(case)}" if part.startswith("proven: ") else part for part in parts]
    assert swapped != parts
    return " | ".join(swapped)


def test_golden_covers_proven_and_every_tier():
    tiers = {EXPECTED[case["id"]]["quality"]["tier"] for case in CASES}
    assert tiers == {"S", "A", "B", "C", "D"}
    assert sum(1 for case in CASES if EXPECTED[case["id"]]["quality"]["proven"]) == 80
    assert len(CASES) == 280


def test_live_tier_is_byte_identical():
    for case in CASES:
        assert _quality(case) == EXPECTED[case["id"]]["quality"], case["id"]


def test_alert_text_without_proven_evidence_is_byte_identical():
    checked = 0
    for case in CASES:
        if _was_proven(case):
            continue
        expected = EXPECTED[case["id"]]
        assert _message(case, expected["quality"]) == expected["message"], case["id"]
        checked += 1
    assert checked == 200


def test_formerly_proven_alerts_carry_their_grade_and_nothing_else_moves():
    changed = 0
    for case in CASES:
        if not _was_proven(case):
            continue
        message = _message(case, EXPECTED[case["id"]]["quality"])
        assert "PROVEN" not in message and "proven:" not in message, case["id"]
        assert message == _p14_text(case), case["id"]
        changed += 1
    assert changed == 80


def test_no_golden_alert_says_proven_any_more():
    for case in CASES:
        assert "PROVEN" not in _message(case, EXPECTED[case["id"]]["quality"]), case["id"]
