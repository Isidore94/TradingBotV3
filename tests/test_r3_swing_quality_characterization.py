"""R3 golden characterization fixtures, committed before production changes."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autopilot_core  # noqa: E402
from master_avwap_lib import legacy  # noqa: E402


FIXTURE = json.loads(
    (ROOT_DIR / "tests" / "fixtures" / "r3_swing_quality_v1.json").read_text(
        encoding="utf-8"
    )
)


def _finite(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _reference_verdict(case: dict) -> tuple[list[str], bool]:
    settings = FIXTURE["settings"]
    side = str(case.get("side") or "").upper()
    close = _finite(case.get("last_close"))
    ema21 = _finite(case.get("ema21"))
    atr20 = _finite(case.get("atr20"))
    rules: list[str] = []
    if close is not None and ema21 is not None and atr20 is not None and atr20 > 0:
        directional_distance = (close - ema21) / atr20
        if side == "SHORT":
            directional_distance *= -1.0
        if directional_distance > settings["ema_atr_max"]:
            rules.append("ema_atr_extension")
    zone = str(case.get("current_band_zone") or "")
    if side == "LONG" and zone in {
        "UPPER_1 to UPPER_2",
        "UPPER_2 to UPPER_3",
        "UPPER_3",
    }:
        rules.append("band_extension")
    if side == "SHORT" and zone in {
        "LOWER_2 to LOWER_1",
        "LOWER_3 to LOWER_2",
        "LOWER_3",
    }:
        rules.append("band_extension")
    daytrade = bool(
        rules
        and (_finite(case.get("relvol")) or 0.0) >= settings["daytrade_rvol_min"]
        and (_finite(case.get("score")) or 0.0) >= settings["daytrade_score_min"]
        and (_finite(case.get("expected_r")) or 0.0)
        >= settings["daytrade_expected_r_min"]
    )
    return rules, daytrade


def _live_row(case: dict) -> dict:
    row = dict(case)
    for fixture_only in ("would_demote", "rules", "daytrade_candidate"):
        row.pop(fixture_only, None)
    row.update(
        {
            "symbol": str(case["id"]).upper(),
            "priority_bucket": "favorite_setup",
            "setup_family": "avwap_retest_followthrough",
            "favorite_signals": ["CROSS_UP_VWAP"],
            "context_signals": [],
            "retest_followthrough": True,
            "previous_day_range_break": True,
        }
    )
    return row


def _series_frame(spec: dict, *, forming: bool) -> pd.DataFrame:
    closes: list[float] = []
    for index in range(int(spec["bars"])):
        gap = float(spec["gap_size"]) if index >= int(spec["gap_index"]) else 0.0
        closes.append(float(spec["start"]) + index * float(spec["step"]) + gap)
    if forming:
        closes[-1] += float(spec["forming_delta"])
    missing_index = spec.get("missing_index")
    if missing_index is not None:
        closes[int(missing_index)] = float("nan")
    return pd.DataFrame(
        {
            "datetime": pd.bdate_range("2026-05-01", periods=len(closes)),
            "open": [value - 0.2 for value in closes],
            "high": [value + 1.0 for value in closes],
            "low": [value - 1.0 for value in closes],
            "close": closes,
            "volume": [1_000_000.0 + index * 10_000.0 for index in range(len(closes))],
        }
    )


def test_fixture_reference_verdicts_and_daytrade_carveout_are_explicit():
    for case in FIXTURE["quality_cases"]:
        rules, daytrade = _reference_verdict(case)
        assert rules == case["rules"], case["id"]
        assert bool(case.get("daytrade_candidate")) is daytrade, case["id"]
        assert bool(rules) is bool(case["would_demote"]), case["id"]


def test_fixture_kills_flipped_inclusive_wrong_field_and_wrong_side_mutations():
    cases = {case["id"]: case for case in FIXTURE["quality_cases"]}
    threshold = float(FIXTURE["settings"]["ema_atr_max"])
    edge = cases["long_exact_edge"]
    over = cases["long_over_edge"]
    short = cases["short_over_edge"]
    opposite_band = cases["opposite_band_field_must_not_fire"]

    def distance(case):
        return (float(case["last_close"]) - float(case["ema21"])) / float(case["atr20"])

    assert not (distance(edge) > threshold)
    assert distance(edge) >= threshold, "kills an accidental inclusive comparison"
    assert distance(over) > threshold
    assert not (distance(over) < threshold), "kills a flipped comparison"
    assert distance(short) < -threshold
    assert abs(distance(short)) > threshold, "kills use of the long-only sign"
    assert opposite_band["current_band_zone"].startswith("LOWER_")
    assert not opposite_band["would_demote"], "kills a side-blind band-zone lookup"


def test_trending_gap_missing_and_forming_pairs_are_not_flat_placebos():
    directions = []
    for spec in FIXTURE["series_cases"]:
        completed = _series_frame(spec, forming=False)
        forming = _series_frame(spec, forming=True)
        completed_indicators = legacy.compute_indicator_frame(completed)
        forming_indicators = legacy.compute_indicator_frame(forming)
        assert completed["close"].nunique(dropna=True) > 20, spec["id"]
        gap_index = int(spec["gap_index"])
        ordinary_move = abs(float(spec["step"]))
        gap_move = abs(
            float(completed["close"].iloc[gap_index])
            - float(completed["close"].iloc[gap_index - 1])
        )
        assert gap_move > ordinary_move * 5, spec["id"]
        assert forming_indicators["ema_21"].iloc[-1] != pytest.approx(
            completed_indicators["ema_21"].iloc[-1]
        ), spec["id"]
        directions.append(
            math.copysign(
                1.0,
                float(completed["close"].dropna().iloc[-1])
                - float(completed["close"].dropna().iloc[0]),
            )
        )
        if spec.get("missing_index") is not None:
            assert pd.isna(completed["close"].iloc[int(spec["missing_index"])])
    assert set(directions) == {-1.0, 1.0}


def test_shadow_packet_starts_with_live_best_swing_membership_unchanged():
    cases = {case["id"]: case for case in FIXTURE["quality_cases"]}
    rows = [
        _live_row(cases[row_id])
        for row_id in FIXTURE["legacy_live_membership"]["row_ids"]
    ]
    selected = legacy._priority_best_swing_trade_rows(
        rows, per_side=20, total_limit=40
    )
    assert {row["id"] for row in selected} == set(
        FIXTURE["legacy_live_membership"]["row_ids"]
    )
    assert all("would_demote" not in row for row in rows)


def test_tracker_and_slot_timing_is_characterized_before_the_honesty_fix():
    schedule = FIXTURE["legacy_schedule"]
    reference = datetime.fromisoformat(schedule["reference"])
    slots = autopilot_core.get_autopilot_swing_slots(
        reference, local_timezone_name="America/Los_Angeles"
    )
    assert slots == schedule["slots"]
    assert [
        slot
        for slot in slots
        if autopilot_core.slot_writes_setup_tracker(
            slot, reference, local_timezone_name="America/Los_Angeles"
        )
    ] == schedule["tracker_write_slots"]


def test_daily_volume_thrust_characterization_keeps_forming_volume_unscaled():
    spec = FIXTURE["series_cases"][0]
    completed = _series_frame(spec, forming=False)
    completed.loc[:, "volume"] = 1_000_000.0
    forming = completed.copy()
    forming.loc[forming.index[-1], "close"] = float(forming["close"].iloc[-2]) * 1.02
    forming.loc[forming.index[-1], "volume"] = 2_100_000.0

    assert legacy._playbook_detect_volume_thrust(
        completed, "LONG", anchor_vwap=1.0
    ) is None
    note = legacy._playbook_detect_volume_thrust(forming, "LONG", anchor_vwap=1.0)
    assert note is not None and "2.1x avg volume" in note


def test_shadow_classifier_matches_the_mutation_seeded_golden():
    rows = [dict(case) for case in FIXTURE["quality_cases"]]
    expected = {
        row["id"]: {
            "would_demote": row.pop("would_demote"),
            "rules": row.pop("rules"),
            "daytrade_candidate": bool(row.pop("daytrade_candidate", False)),
        }
        for row in rows
    }
    count = legacy.apply_swing_quality_demotion(rows, FIXTURE["settings"])
    assert count == sum(1 for value in expected.values() if value["would_demote"])
    for row in rows:
        frozen = expected[row["id"]]
        assert row["would_demote"] is frozen["would_demote"], row["id"]
        assert row["would_demote_rules"] == frozen["rules"], row["id"]
        assert row["daytrade_candidate"] is frozen["daytrade_candidate"], row["id"]


def test_shadow_stamps_never_change_live_selection_or_tier_membership():
    cases = {case["id"]: case for case in FIXTURE["quality_cases"]}
    rows = [
        _live_row(cases[row_id])
        for row_id in FIXTURE["legacy_live_membership"]["row_ids"]
    ]
    before_best = legacy._priority_best_swing_trade_rows(
        rows, per_side=20, total_limit=40
    )
    before_tiers = legacy._priority_partition_tier_rows(
        actionable_rows=rows,
        report_rows=rows,
        high_conviction_rows=rows,
        best_swing_rows=before_best,
    )
    legacy.apply_swing_quality_demotion(rows, FIXTURE["settings"])
    after_best = legacy._priority_best_swing_trade_rows(
        rows, per_side=20, total_limit=40
    )
    after_tiers = legacy._priority_partition_tier_rows(
        actionable_rows=rows,
        report_rows=rows,
        high_conviction_rows=rows,
        best_swing_rows=after_best,
    )
    assert [row["id"] for row in after_best] == [row["id"] for row in before_best]
    assert {
        tier["label"]: [row["id"] for row in tier["rows"]] for tier in after_tiers
    } == {
        tier["label"]: [row["id"] for row in tier["rows"]] for tier in before_tiers
    }


def test_report_duplicates_shadow_evidence_but_keeps_live_best_swing(tmp_path):
    case = next(
        row for row in FIXTURE["quality_cases"] if row["id"] == "long_over_edge"
    )
    row = _live_row(case)
    legacy.apply_swing_quality_demotion([row], FIXTURE["settings"])
    path = tmp_path / "priority.txt"
    legacy.write_priority_setup_report(path, [row])
    text = path.read_text(encoding="utf-8")
    assert "Best swing trades today\n" in text
    assert row["symbol"] in text[text.index("Best swing trades today\n") :]
    assert "Stretched - shadow would demote (NO LIVE CHANGE)" in text
    assert "quality shadow: EMA21 distance 2.01 ATR > 2.00" in text

    pytest.importorskip("PySide6")
    from ui.models.setup import SetupRow
    from ui.panels import master_avwap_panel

    assert master_avwap_panel._shadow_would_demote_symbols(path) == {row["symbol"]}
    desk_row = SetupRow(symbol=row["symbol"], raw={})
    original = master_avwap_panel._shadow_would_demote_symbols
    try:
        master_avwap_panel._shadow_would_demote_symbols = lambda: {row["symbol"]}
        master_avwap_panel._apply_swing_quality_shadow_badges([desk_row])
    finally:
        master_avwap_panel._shadow_would_demote_symbols = original
    assert desk_row.raw["classification_badges"] == ["Stretched? (shadow)"]
