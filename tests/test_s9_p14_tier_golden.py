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


# --------------------------------------------------------------------------- the bot seam
REGIME = "bear_channel_lower_highs"


def _state_with_extreme_regime_segments():
    """A COPY of the golden state plus regime segments no live tier may feel."""
    import copy

    state = copy.deepcopy(STATE)
    state["segments"]["structural_regime"] = {
        # A mute-grade negative with a big n, and a huge positive.
        f"short|{REGIME}": {
            "avg_close_r": -0.9, "entry_r": -0.9, "production_r": -0.9, "sample_count": 900,
            "session_count": 40, "std_close_r": 1.2, "median_close_r": -1.0, "muted": True, "proven": False,
        },
        f"long|{REGIME}": {
            "avg_close_r": 2.5, "entry_r": 2.5, "production_r": 2.5, "sample_count": 900,
            "session_count": 40, "std_close_r": 1.2, "median_close_r": 2.0, "muted": False, "proven": True,
        },
    }
    return state


def test_the_bot_seam_keeps_live_tier_text_and_mutes_with_extreme_regime_segments(monkeypatch):
    """S9 review blocker: neither the shadow nor the regime reaches a live verdict at the seam."""
    from bounce_bot_lib import legacy

    state = _state_with_extreme_regime_segments()
    monkeypatch.setattr(learning, "load_bounce_learning_state", lambda path=None: state)
    monkeypatch.setattr(learning, "structural_regime_on", lambda day: REGIME)
    shadow_differs = 0
    for case in CASES:
        inputs = case["inputs"]
        monkeypatch.setattr(learning, "time_bucket_for", lambda when, bucket=inputs["time_bucket"]: bucket)
        row = {
            "market_environment": inputs["market_environment"],
            "master_avwap_priority_bucket": inputs["priority_bucket"],
            "master_avwap_focus_label": inputs["focus_label"],
            "master_avwap_setup_family": inputs["setup_family"],
            "master_avwap_swing_traits": ";".join(inputs["swing_traits"]),
            "signal_time": f"{case['trade_date']} 10:00:00",
        }
        levels = {bounce_type: 1.0 for bounce_type in inputs["bounce_types"]}
        quality = legacy.BounceBot._evaluate_bounce_alert_quality(
            legacy.BounceBot, inputs["direction"], levels, row
        )
        shadow = quality.pop("shadow_s9")
        expected = EXPECTED[case["id"]]
        assert quality == expected["quality"], case["id"]  # tier, composite, mutes, reasons
        if quality.get("proven_reasons"):
            assert _message(case, quality) == _p14_text(case), case["id"]
        else:
            assert _message(case, quality) == expected["message"], case["id"]
        plain = learning.evaluate_shadow_tier(STATE, **{k: v for k, v in inputs.items() if k != "swing_traits"})
        shadow_differs += int(shadow != plain)
    # The regime segments DO move the shadow, so the seam really fed it the regime.
    assert shadow_differs >= 100
