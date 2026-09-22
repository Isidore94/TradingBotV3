"""Red contract tests for SP2's sample-safe proven-quality score repair.

The fixture is hand-pinned from ``fb95854e`` before any SP2 change.  These
tests drive the production family builder, lookup and ranking application;
they do not fabricate the lookup result that the score is supposed to audit.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from master_avwap_lib import legacy  # noqa: E402
from master_avwap_lib.expected_r import compute_proven_quality_score  # noqa: E402
from test_st4_first_actionable import (  # noqa: E402
    GOLDEN_LOOKBACK_DAYS as ST4_GOLDEN_LOOKBACK_DAYS,
    GOLDEN_REFERENCE_DATE as ST4_GOLDEN_REFERENCE_DATE,
    _golden_setups as st4_golden_setups,
    _rows_to_csv_text,
)


GOLDENS = json.loads(
    (Path(__file__).parent / "fixtures" / "sp2_pqs_policy_goldens.json").read_text(
        encoding="utf-8"
    )
)["cases"]
REFERENCE_DAY = date(2026, 9, 18)


def _tracked_setup(
    symbol: str,
    scan_date: str,
    total_r: float | None,
    *,
    status: str = "TARGET_HIT",
    family: str = "post_earnings_52w_break",
    extra_scenarios: dict | None = None,
) -> dict:
    """One real tracker-record shape with a representative baseline scenario."""
    scenarios = {
        "full_band2": {
            "tradeable": True,
            "status": status,
            "total_r": total_r,
            "stop_reference_label": "LOWER_1",
            "exit_template_id": "full_band2",
        }
    }
    scenarios.update(extra_scenarios or {})
    return {
        "setup_id": f"{scan_date}:{symbol}:{family}",
        "symbol": symbol,
        "side": "LONG",
        "scan_date": scan_date,
        "priority_bucket": "favorite_setup",
        "setup_family": family,
        "setup_status": "CLOSED" if status != "OPEN" else "OPEN",
        "favorite_signals": [],
        "scenarios": scenarios,
    }


def _family_row(setups: dict) -> dict:
    rows = legacy.build_recent_tracker_setup_family_rows(
        setups,
        reference_date=REFERENCE_DAY,
        lookback_days=90,
    )
    assert len(rows) == 1
    return rows[0]


def test_explicit_v1_replays_the_hand_pinned_pre_repair_goldens():
    """Rollback/replay must name the old scoring policy and preserve its output."""
    for case_name in ("n0", "n1_all_win", "n6_all_win"):
        case = GOLDENS[case_name]
        score = compute_proven_quality_score(
            static_points=case["static_points"],
            win_rate=case["win_rate"],
            profit_factor=case["profit_factor"],
            closed_samples=case["closed_samples"],
            freshness=case["freshness"],
            policy="pqs_v1",
        )
        assert score["policy"] == "pqs_v1"
        assert score["score"] == pytest.approx(case["score"], abs=0.01)
        assert score["evidence"] == pytest.approx(case["evidence"], abs=0.01)


def test_v2_withholds_positive_evidence_below_episode_or_session_floor():
    """One or six winners, and one hundred same-day winners, remain unproven."""
    for case_name, entry_sessions in (
        ("n1_all_win", 1),
        ("n6_all_win", 5),
        ("same_day_concentration", 1),
    ):
        case = GOLDENS[case_name]
        n_wins = int(case["n_wins"] if "n_wins" in case else case["closed_samples"])
        score = compute_proven_quality_score(
            static_points=260.0,
            win_rate=1.0,
            profit_factor=99.0,
            closed_samples=n_wins,
            n_wins=n_wins,
            n_losses=0,
            n_flats=0,
            measured_entry_sessions=entry_sessions,
            gross_win=float(n_wins),
            gross_loss=0.0,
        )
        assert score["policy"] == "pqs_v2"
        assert score["proven"] is False
        assert score["evidence"] <= 40.0
        assert score["profit_factor"] is None
        assert score["payoff_evidence"] == 0.0
        assert "unproven" in score["reason"]


def test_v2_uses_counted_closed_population_and_scales_payoff_by_losses():
    """A weighted rate cannot supply Wilson n or a full payoff bonus."""
    case = GOLDENS["mature_gains_losses"]
    score = compute_proven_quality_score(
        static_points=260.0,
        win_rate=GOLDENS["weighted_vs_counted"]["weighted_win_rate"],
        profit_factor=case["profit_factor"],
        closed_samples=30,
        n_wins=case["n_wins"],
        n_losses=case["n_losses"],
        n_flats=case["n_flats"],
        measured_entry_sessions=case["measured_entry_sessions"],
        gross_win=case["gross_win"],
        gross_loss=case["gross_loss"],
    )
    assert score["policy"] == "pqs_v2"
    assert score["closed_samples"] == 30
    assert score["counted_win_rate"] == pytest.approx(0.8)
    assert score["confident_win_rate"] > 0.5
    assert score["profit_factor"] == pytest.approx(10.0)
    assert score["payoff_evidence"] == pytest.approx(case["payoff_evidence"])


def test_family_builder_exports_only_finite_representative_closed_pqs_basis():
    """Pending/alternate/unreadable entries are not PQS evidence or fake losses."""
    setups = {}
    for index in range(24):
        scan_day = REFERENCE_DAY - timedelta(days=index % 5 + 2)
        setups[f"win-{index}"] = _tracked_setup(
            f"W{index:02d}", scan_day.isoformat(), 2.5
        )
    for index in range(6):
        scan_day = REFERENCE_DAY - timedelta(days=index % 5 + 2)
        setups[f"loss-{index}"] = _tracked_setup(
            f"L{index:02d}", scan_day.isoformat(), -1.0, status="STOPPED"
        )
    setups["pending-representative"] = _tracked_setup(
        "PENDING", "2026-09-17", 0.0, status="OPEN",
        extra_scenarios={
            "alternate_closed": {
                "tradeable": True,
                "status": "TARGET_HIT",
                "total_r": 8.0,
                "stop_reference_label": "LOWER_1",
                "exit_template_id": "alternate",
            }
        },
    )
    setups["unreadable"] = _tracked_setup("UNREAD", "2026-09-11", math.nan, status="STOPPED")

    row = _family_row(setups)
    assert row["pqs_n_wins"] == 24
    assert row["pqs_n_losses"] == 6
    assert row["pqs_n_flats"] == 0
    assert row["pqs_n_closed"] == 30
    assert row["pqs_n_entry_sessions"] == 5
    assert row["pqs_gross_win"] == pytest.approx(60.0)
    assert row["pqs_gross_loss"] == pytest.approx(6.0)
    assert row["pqs_profit_factor"] == pytest.approx(10.0)
    assert row["pqs_counted_win_rate"] == pytest.approx(0.8)
    assert row["pqs_basis"] == "finite_representative_closed_r"
    assert row["pqs_n_closed"] == row["pqs_n_wins"] + row["pqs_n_losses"] + row["pqs_n_flats"]
    assert row["pqs_n_closed"] != row["closed_setups"]


def test_v2_marks_missing_or_invalid_numeric_inputs_unmeasured_and_keeps_zero_freshness_zero():
    """Neither NaN/inf nor a falsey zero may manufacture evidence."""
    no_coverage = compute_proven_quality_score(
        static_points=260.0,
        win_rate=None,
        profit_factor=None,
        closed_samples=0,
        n_wins=0,
        n_losses=0,
        n_flats=0,
        measured_entry_sessions=0,
        gross_win=0.0,
        gross_loss=0.0,
    )
    invalid = compute_proven_quality_score(
        static_points=260.0,
        win_rate=math.nan,
        profit_factor=math.inf,
        closed_samples=30,
        n_wins=30,
        n_losses=0,
        n_flats=0,
        measured_entry_sessions=5,
        gross_win=math.inf,
        gross_loss=math.nan,
    )
    stale = compute_proven_quality_score(
        static_points=260.0,
        win_rate=1.0,
        profit_factor=3.0,
        closed_samples=30,
        n_wins=30,
        n_losses=0,
        n_flats=0,
        measured_entry_sessions=5,
        gross_win=30.0,
        gross_loss=0.0,
        freshness=0.0,
    )
    assert no_coverage["proven"] is False
    assert no_coverage["evidence"] == 40.0
    assert invalid["proven"] is False
    assert invalid["profit_factor"] is None
    assert invalid["payoff_evidence"] == 0.0
    assert math.isfinite(invalid["score"])
    assert stale["freshness"] == 0.0
    assert stale["evidence"] == 0.0


def test_v2_payoff_needs_thirty_measured_losing_episodes_for_full_weight():
    """A PF cap alone cannot award sixty points from six losses."""
    partial_case = GOLDENS["mature_gains_losses"]
    partial = compute_proven_quality_score(
        static_points=260.0,
        win_rate=0.8,
        profit_factor=partial_case["profit_factor"],
        closed_samples=30,
        n_wins=partial_case["n_wins"],
        n_losses=partial_case["n_losses"],
        n_flats=0,
        measured_entry_sessions=5,
        gross_win=partial_case["gross_win"],
        gross_loss=partial_case["gross_loss"],
    )
    full = compute_proven_quality_score(
        static_points=260.0,
        win_rate=0.5,
        profit_factor=3.0,
        closed_samples=60,
        n_wins=30,
        n_losses=30,
        n_flats=0,
        measured_entry_sessions=5,
        gross_win=90.0,
        gross_loss=30.0,
    )
    assert partial["payoff_evidence"] == pytest.approx(12.0)
    assert full["payoff_evidence"] == pytest.approx(60.0)


def test_v2_adds_pqs_basis_without_moving_the_pinned_legacy_family_columns():
    """SP2 may append provenance, never rewrite the existing family values."""
    golden_path = Path(__file__).parent / "fixtures" / "st4_family_rows_golden.csv"
    golden_text = golden_path.read_text(encoding="utf-8")
    old_columns = golden_text.splitlines()[0].split(",")
    rows = legacy.build_recent_tracker_setup_family_rows(
        st4_golden_setups(),
        reference_date=ST4_GOLDEN_REFERENCE_DATE,
        lookback_days=ST4_GOLDEN_LOOKBACK_DAYS,
        selection_policy="closed_first_v1",
    )
    assert _rows_to_csv_text(rows, old_columns) == golden_text
    assert all(row["pqs_basis"] == "finite_representative_closed_r" for row in rows)


def test_apply_routes_v2_basis_to_row_state_and_features_once():
    """The real builder -> lookup -> apply route carries one named v2 basis."""
    setups = {
        f"w{index}": _tracked_setup(
            f"W{index:02d}", (REFERENCE_DAY - timedelta(days=index % 5 + 2)).isoformat(), 1.0
        )
        for index in range(30)
    }
    family_row = _family_row(setups)
    priority = {
        "symbol": "LIVE",
        "side": "LONG",
        "priority_bucket": "favorite_setup",
        "setup_family": "post_earnings_52w_break",
        "score": 260.0,
        "signal_date": REFERENCE_DAY.isoformat(),
    }
    ai_state = {"symbols": {"LIVE": {}}}
    features = {"LIVE": {}}

    legacy.apply_expected_r_ranking(
        [priority], ai_state, features, recent_family_rows=[family_row], reference_date=REFERENCE_DAY
    )
    first_score = priority["score"]
    legacy.apply_expected_r_ranking(
        [priority], ai_state, features, recent_family_rows=[family_row], reference_date=REFERENCE_DAY
    )

    assert priority["score"] == first_score
    for destination in (priority, ai_state["symbols"]["LIVE"], features["LIVE"]):
        assert destination["pqs_policy"] == "pqs_v2"
        assert destination["pqs_basis"] == "finite_representative_closed_r"
        assert destination["pqs_n_closed"] == 30
        assert destination["pqs_n_entry_sessions"] == 5
        assert destination["pqs_profit_factor"] is None
    assert "PF 9.9" not in priority["proven_quality_note"]
