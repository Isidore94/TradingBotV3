"""Golden characterization of ``build_d1_zone_arms`` (R5 section 8.3, step 4).

``test_d1_zone_arms.py`` says of itself that it is not a golden fixture, and
section 8.3 requires one BEFORE the prior-anchor AVWAP line is added to the
zone-arms output. This is that fixture: the whole side/zone matrix, with and
without prior bands and EMAs, plus the still-armed gating edges, pinned as
exact output.

Its job is narrow and it is the whole point of the packet's plumbing decision:
after the edit, every existing key must be byte-identical and
``trigger_levels`` unchanged, with only the new optional top-level key added
where a prior anchor exists. If a future edit changes what the D1 scan arms,
this file fails and the change has to be a deliberate detector decision with
its own evidence - not a side effect of plumbing.

Regenerate deliberately, never casually::

    python tests/test_d1_zone_arms_golden.py --write
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from master_avwap_lib.d1_zone_arms import build_d1_zone_arms  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "d1_zone_arms_golden_v1.json"

#: One case per row: a name, and the exact keyword arguments the scan passes.
#: The band ladder is deliberately round so a reader can see which zone each
#: close lands in without arithmetic.
CASES: list[tuple[str, dict]] = [
    (
        "long_zone1_full_ladder",
        dict(
            symbol="aaa",
            close=101.0,
            avwape=100.0,
            upper_1=104.0,
            upper_2=108.0,
            upper_3=112.0,
            lower_1=96.0,
            lower_2=92.0,
            lower_3=88.0,
            ema15=99.0,
            ema21=98.0,
            prev_upper_1=102.5,
            prev_lower_1=94.0,
            stdev=4.0,
            atr=2.0,
            anchor_date="2026-05-01",
            armed_at="2026-08-18T09:31:00",
        ),
    ),
    (
        "long_zone1_no_prior_bands",
        dict(
            symbol="BBB",
            close=101.0,
            avwape=100.0,
            upper_1=104.0,
            upper_2=108.0,
            upper_3=112.0,
            ema15=99.0,
            ema21=98.0,
            stdev=4.0,
            atr=2.0,
            anchor_date="2026-05-01",
            armed_at="2026-08-18T09:31:00",
        ),
    ),
    (
        "long_zone2",
        dict(
            symbol="CCC",
            close=105.0,
            avwape=100.0,
            upper_1=104.0,
            upper_2=108.0,
            upper_3=112.0,
            ema15=102.0,
            ema21=101.0,
            prev_upper_1=103.0,
            stdev=4.0,
            atr=2.0,
            anchor_date="2026-05-01",
            armed_at="2026-08-18T09:31:00",
        ),
    ),
    (
        "long_zone3_sustained",
        dict(
            symbol="DDD",
            close=109.0,
            avwape=100.0,
            upper_1=104.0,
            upper_2=108.0,
            upper_3=112.0,
            ema15=105.0,
            ema21=103.0,
            stdev=4.0,
            atr=2.0,
            sustained_2nd_3rd=True,
            anchor_date="2026-05-01",
            armed_at="2026-08-18T09:31:00",
        ),
    ),
    (
        "long_zone3_not_sustained",
        dict(
            symbol="EEE",
            close=109.0,
            avwape=100.0,
            upper_1=104.0,
            upper_2=108.0,
            upper_3=112.0,
            ema15=105.0,
            ema21=103.0,
            stdev=4.0,
            atr=2.0,
            sustained_2nd_3rd=False,
            anchor_date="2026-05-01",
            armed_at="2026-08-18T09:31:00",
        ),
    ),
    (
        "short_zone1_full_ladder",
        dict(
            symbol="FFF",
            close=99.0,
            avwape=100.0,
            upper_1=104.0,
            upper_2=108.0,
            upper_3=112.0,
            lower_1=96.0,
            lower_2=92.0,
            lower_3=88.0,
            ema15=101.0,
            ema21=102.0,
            prev_upper_1=106.0,
            prev_lower_1=97.5,
            stdev=4.0,
            atr=2.0,
            anchor_date="2026-05-01",
            armed_at="2026-08-18T09:31:00",
        ),
    ),
    (
        "short_zone2",
        dict(
            symbol="GGG",
            close=95.0,
            avwape=100.0,
            lower_1=96.0,
            lower_2=92.0,
            lower_3=88.0,
            ema15=98.0,
            ema21=99.0,
            prev_lower_1=97.0,
            stdev=4.0,
            atr=2.0,
            anchor_date="2026-05-01",
            armed_at="2026-08-18T09:31:00",
        ),
    ),
    (
        "short_zone3_sustained",
        dict(
            symbol="HHH",
            close=91.0,
            avwape=100.0,
            lower_1=96.0,
            lower_2=92.0,
            lower_3=88.0,
            ema15=95.0,
            ema21=97.0,
            stdev=4.0,
            atr=2.0,
            sustained_2nd_3rd=True,
            anchor_date="2026-05-01",
            armed_at="2026-08-18T09:31:00",
        ),
    ),
    (
        "no_ema_no_atr_falls_back_to_stdev_tolerance",
        dict(
            symbol="III",
            close=101.0,
            avwape=100.0,
            upper_1=104.0,
            upper_2=108.0,
            stdev=4.0,
            anchor_date="2026-05-01",
            armed_at="2026-08-18T09:31:00",
        ),
    ),
    (
        "no_ema_no_atr_no_stdev_falls_back_to_price_tolerance",
        dict(
            symbol="JJJ",
            close=101.0,
            avwape=100.0,
            upper_1=104.0,
            upper_2=108.0,
            anchor_date="2026-05-01",
            armed_at="2026-08-18T09:31:00",
        ),
    ),
    (
        "gating_edge_ema_already_passed",
        dict(
            # The 15EMA sits ABOVE the close, so a bounce-up arm on it would be
            # a level price has already lost. It must not be armed.
            symbol="KKK",
            close=105.0,
            avwape=100.0,
            upper_1=104.0,
            upper_2=108.0,
            ema15=106.0,
            ema21=101.0,
            stdev=4.0,
            atr=2.0,
            anchor_date="2026-05-01",
            armed_at="2026-08-18T09:31:00",
        ),
    ),
    (
        "gating_edge_prior_band_above_current",
        dict(
            # The prior-anchor UPPER_1 sits above the current one, so zone 1's
            # prior-band bounce arm does not apply.
            symbol="LLL",
            close=101.0,
            avwape=100.0,
            upper_1=104.0,
            upper_2=108.0,
            prev_upper_1=105.0,
            stdev=4.0,
            atr=2.0,
            anchor_date="2026-05-01",
            armed_at="2026-08-18T09:31:00",
        ),
    ),
    (
        "nothing_armed_returns_none",
        dict(symbol="MMM", close=101.0, avwape=None),
    ),
]


def build_all() -> dict:
    return {name: build_d1_zone_arms(**kwargs) for name, kwargs in CASES}


def case_inputs() -> dict:
    """The exact arguments each case is built from - the fixture's raw input."""
    return {name: dict(sorted(kwargs.items())) for name, kwargs in CASES}


def build_payload() -> dict:
    """The fixture file: the plan.md Milestone 3 contract, then the cases."""
    canonical = json.dumps(
        case_inputs(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema": "d1_zone_arms_golden_v1",
        "feature_version": "zone_arm_schema_1",
        "raw_input_keys": ["case_inputs"],
        "raw_input_sha256": hashlib.sha256(canonical).hexdigest(),
        "acquired_at": "2026-08-18T00:00:00-07:00",
        "as_of": "2026-08-18T00:00:00-07:00",
        "universe_version": "synthetic_zone_ladder_v1",
        "provider_assumptions": (
            "Synthetic band ladders and closes only; no broker, network or daily "
            "store is touched. build_d1_zone_arms is pure, so every value here is "
            "a function of the arguments in case_inputs."
        ),
        "expected_keys": ["cases"],
        "numeric_tolerance": 0.0,
        "intentional_difference": "",
        "schema_version": 1,
        "intent": (
            "Fixture before edit (plan.md Milestone 3). Pins the whole side/zone "
            "matrix, the tolerance fallbacks and the still-armed gating edges, so "
            "the R5 section 8.3 prior-anchor AVWAP line can be PROVEN additive: "
            "after that edit every key here still matches and trigger_levels is "
            "unchanged."
        ),
        "case_inputs": case_inputs(),
        "cases": build_all(),
    }


def test_the_zone_arm_output_is_unchanged():
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
    actual = json.loads(json.dumps(build_all(), sort_keys=True))
    assert actual == expected


def test_the_matrix_actually_covers_both_sides_and_all_three_zones():
    """A golden file over a thin matrix is a false sense of safety."""
    built = [entry for entry in build_all().values() if entry]
    assert {entry["side"] for entry in built} == {"LONG", "SHORT"}
    assert {entry["zone"] for entry in built} >= {1, 2, 3}
    assert any(not entry["trigger_levels"] for entry in built) is False


class TestThePriorAnchorAvwapIsPurelyAdditive:
    """R5 section 8.3: carried, not computed, and invisible to the rubric."""

    BASE = dict(CASES[0][1])

    def test_supplying_it_changes_nothing_but_adds_one_key(self):
        without = build_d1_zone_arms(**self.BASE)
        with_prior = build_d1_zone_arms(**dict(self.BASE, prev_avwape=97.1234567))

        assert with_prior["prev_avwape"] == 97.1235
        assert with_prior["trigger_levels"] == without["trigger_levels"]
        assert {
            key: value for key, value in with_prior.items() if key != "prev_avwape"
        } == without

    def test_no_prior_anchor_means_the_key_is_absent_not_null(self):
        """A no-prior symbol must read like a file written before the change."""
        for value in (None, 0, 0.0, "", "n/a", float("nan")):
            entry = build_d1_zone_arms(**dict(self.BASE, prev_avwape=value))
            assert "prev_avwape" not in entry

    def test_the_schema_version_does_not_move(self):
        """Additive only: a bump could trip a reader that gates on it."""
        entry = build_d1_zone_arms(**dict(self.BASE, prev_avwape=97.0))
        assert entry["schema_version"] == 1

    def test_the_trigger_walker_cannot_see_it(self):
        """The zone-arm alert rubric provably cannot gain a trigger."""
        from master_avwap_lib.d1_zone_arms import detect_zone_arm_triggers

        entry = build_d1_zone_arms(**dict(self.BASE, prev_avwape=97.0))
        bars = [
            {"high": 97.05, "low": 96.9, "close": 97.02},
            {"high": 98.0, "low": 97.0, "close": 97.9},
        ]
        fired = detect_zone_arm_triggers(entry, bars)
        assert all(arm.get("level") != 97.0 for arm in fired)


if __name__ == "__main__":  # pragma: no cover - deliberate regeneration
    if "--write" in sys.argv:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(
            json.dumps(build_payload(), indent=2, sort_keys=True) + chr(10),
            encoding="utf-8",
        )
        print(f"wrote {FIXTURE}")
    else:
        print("pass --write to regenerate the fixture")
