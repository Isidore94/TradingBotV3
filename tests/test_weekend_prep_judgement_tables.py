"""P2 items 1-2: the judgement tables show the robust half, and the callouts
are named rather than counted.

Both surfaces were showing strictly less than the store on disk already held.

* The two cohort tables kept six columns and dropped `median_return`,
  `trimmed_mean_return`, `ci_low`/`ci_high`, `symbols`, `sessions`,
  `top_symbol_share`, `evidence_label` and `meets_n_floor` - every one of them
  written by `human_focus_tracking` since R10.C, and most of them already on
  screen in the Focus performance table on the SAME page. What survived was a
  bare mean on a ratio, which is the statistic R10.C published the robust half
  to stop anyone reading alone.
* The week page printed `Blind Spots: 3`. The scoreboard has always known which
  segment, how often it was shown, and what the two halves measured.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6")

from ui.panels import weekend_prep_panel as panel_module  # noqa: E402


def _raw(**overrides) -> dict:
    """One rollup row exactly as `human_focus_tracking` writes it."""
    row = {
        "cohort": "human_focus_veto_v2_compressed",
        "side": "ALL",
        "horizon_sessions": "3",
        "sample_count": "18",
        "win_rate": "0.3889",
        "avg_side_return": "-0.0123",
        "profit_factor": "0.3932",
        "updated_at": "2026-09-01T05:00:00",
        "median_return": "-0.0056",
        "trimmed_mean_return": "-0.0098",
        "p10_return": "-0.0400",
        "p90_return": "0.0210",
        "symbols": "14",
        "sessions": "5",
        "top_symbol_share": "0.1667",
        "ci_low": "-0.0301",
        "ci_high": "0.0055",
        "ci_basis": "block bootstrap by session",
        "evidence_label": "discovery",
        "meets_n_floor": "1",
    }
    row.update(overrides)
    return row


# ==========================================================================
# item 1 - the robust half reaches the two judgement tables
# ==========================================================================
def test_the_projection_carries_every_robust_column():
    """Fail-before-fix: `_cohort_robust_fields` does not exist, and the two
    readers project six keys."""
    fields = panel_module._cohort_robust_fields(_raw())

    assert fields["median"] == "-0.56%"
    assert fields["trimmed"] == "-0.98%"
    assert fields["symbols"] == "14"
    assert fields["sessions"] == "5"
    assert fields["top_share"] == "16.7%"
    assert fields["ci"] == "[-0.0301, 0.0055]"
    assert fields["ci_basis"] == "block bootstrap by session"
    assert fields["evidence"] == "discovery"
    assert fields["_meets_floor"] is True
    assert fields["_sort_value"] == pytest.approx(-0.0098)


def test_an_uncomputable_interval_stays_blank_and_never_becomes_zero():
    """A substituted zero would be a false lesson about the trader's own
    judgement: a sample spanning one session cannot have a block interval."""
    fields = panel_module._cohort_robust_fields(
        _raw(ci_low="", ci_high="", median_return="", trimmed_mean_return="")
    )
    assert fields["ci"] == ""
    assert fields["median"] == ""
    assert fields["trimmed"] == ""
    assert fields["_sort_value"] is None


def test_a_row_under_the_n_floor_sorts_after_every_row_above_it():
    """The ORDER is the honesty. An n=3 cohort with a profit factor of 165 must
    not sit at the top of a table read on a Saturday - which is exactly what
    the live rollup carried for `human_focus_veto_compressed`."""
    strong_but_thin = {
        **panel_module._cohort_robust_fields(
            _raw(trimmed_mean_return="0.5000", meets_n_floor="", sample_count="3")
        ),
        "cohort": "thin",
        "horizon": "3",
    }
    weak_but_real = {
        **panel_module._cohort_robust_fields(_raw(trimmed_mean_return="-0.0098")),
        "cohort": "real",
        "horizon": "3",
    }
    ordered = panel_module._cohort_view([strong_but_thin, weak_but_real], "3")

    assert [row["cohort"] for row in ordered] == ["real", "thin"]


def test_rows_above_the_floor_order_by_the_TRIMMED_mean():
    def row(name, trimmed, avg):
        return {
            **panel_module._cohort_robust_fields(
                _raw(trimmed_mean_return=trimmed, avg_side_return=avg)
            ),
            "cohort": name,
            "horizon": "3",
        }

    # `flatterer` wins on the bare average and loses on the trimmed one.
    ordered = panel_module._cohort_view(
        [row("flatterer", "0.0100", "0.9000"), row("honest", "0.0400", "0.0500")], "3"
    )
    assert [r["cohort"] for r in ordered] == ["honest", "flatterer"]


def test_the_view_shows_one_horizon_and_defaults_to_h3():
    rows = [
        {**panel_module._cohort_robust_fields(_raw()), "cohort": "a", "horizon": "1"},
        {**panel_module._cohort_robust_fields(_raw()), "cohort": "b", "horizon": "3"},
        {**panel_module._cohort_robust_fields(_raw()), "cohort": "c", "horizon": "5"},
    ]
    assert panel_module.DEFAULT_COHORT_HORIZON == "3"
    assert [r["cohort"] for r in panel_module._cohort_view(rows, "3")] == ["b"]
    assert [r["cohort"] for r in panel_module._cohort_view(rows, "5")] == ["c"]


def test_a_row_with_no_trimmed_mean_sorts_last_inside_its_own_group():
    """Not promoted by a default of zero, and not mixed with the thin rows
    either - it cleared the floor, its ordering key is simply absent."""
    def row(name, trimmed):
        return {
            **panel_module._cohort_robust_fields(_raw(trimmed_mean_return=trimmed)),
            "cohort": name,
            "horizon": "3",
        }

    ordered = panel_module._cohort_view(
        [row("blank", ""), row("negative", "-0.9000")], "3"
    )
    assert [r["cohort"] for r in ordered] == ["negative", "blank"]


def test_the_floor_sentence_says_what_being_under_the_floor_means():
    under = {**panel_module._cohort_robust_fields(_raw(meets_n_floor="")), "horizon": "3"}
    over = {**panel_module._cohort_robust_fields(_raw()), "horizon": "3"}

    said = panel_module._floor_sentence([under, over])
    assert "1 of 2" in said
    assert "not a finding" in said
    assert "TRIMMED" in said
    assert panel_module._floor_sentence([over]) == (
        "Every row shown clears the reportable-n floor."
    )
    assert panel_module._floor_sentence([]) == ""


def test_the_claim_picklist_caveat_is_the_one_the_AI_already_gets():
    """Never a second copy of the sentence: the model has been told the
    picklist is bounded on every package, and the trader reading the same
    table had not been."""
    from ai_summary import _offered_claim_caveat

    assert panel_module._claim_picklist_caveat() == _offered_claim_caveat()
    assert "not a trader preference" in panel_module._claim_picklist_caveat()


def test_the_caveat_degrades_to_UNKNOWN_rather_than_disappearing(monkeypatch):
    import ai_summary

    def boom():
        raise RuntimeError("picklist unreadable")

    monkeypatch.setattr(ai_summary, "_offered_claim_caveat", boom)
    said = panel_module._claim_picklist_caveat()
    assert "UNKNOWN" in said
    assert "preference" in said


# ==========================================================================
# item 2 - the callouts are named
# ==========================================================================
def test_callouts_name_the_segment_instead_of_counting_it():
    """Fail-before-fix: `callout_lines` does not exist and the page printed
    `Blind Spots: 1`."""
    state = {
        "overall_take_rate": 0.324,
        "blind_spots": [
            {
                "dimension": "tier",
                "segment": "B",
                "shown": 24,
                "take_rate": 0.125,
                "passed_r_avg": 0.62,
                "passed_r_n": 14,
            }
        ],
        "leaks": [],
    }
    text = "\n".join(panel_module.callout_lines(state))

    assert "tier=B" in text
    assert "take 12% of 24 shown" in text  # 12.5 rounds to even, as Python formats it
    assert "passed +0.62R (n=14)" in text
    assert "overall take rate this window: 32%" in text
    assert "none at current sample sizes." in text  # the empty leaks class


def test_the_r_gap_class_is_rendered_when_the_state_carries_it():
    state = {
        "overall_take_rate": 0.25,
        "blind_spots": [],
        "leaks": [],
        "r_gaps": [
            {
                "dimension": "bounce_type",
                "segment": "lrsi_cross_20",
                "shown": 60,
                "take_rate": 0.283,
                "taken_r_avg": -0.376,
                "taken_r_n": 8,
                "passed_r_avg": 0.962,
                "passed_r_n": 24,
                "r_difference": -1.338,
            }
        ],
    }
    text = "\n".join(panel_module.callout_lines(state))

    assert "R GAPS" in text
    assert "bounce_type=lrsi_cross_20" in text
    assert "taken -0.38R (n=8)" in text
    assert "passed +0.96R (n=24)" in text
    assert "gap -1.34R" in text


def test_a_state_without_the_r_gap_class_still_renders():
    """The scoreboard file is written by whichever build last ran. A state with
    no `r_gaps` key must print the two classes it does have rather than raise
    or invent an empty third."""
    state = {"overall_take_rate": 0.2, "blind_spots": [], "leaks": []}
    text = "\n".join(panel_module.callout_lines(state))

    assert "BLIND SPOTS" in text
    assert "LEAKS" in text
    assert "R GAPS" not in text


def test_a_quiet_window_says_quiet_not_clean():
    text = "\n".join(
        panel_module.callout_lines({"overall_take_rate": 0.3, "blind_spots": [], "leaks": []})
    )
    assert "quiet week, not a clean one" in text


def test_a_forward_percent_callout_prints_in_percent_not_R():
    state = {
        "overall_take_rate": 0.3,
        "blind_spots": [
            {
                "dimension": "setup_family",
                "segment": "post_earnings",
                "shown": 12,
                "take_rate": 0.1,
                "passed_fwd_avg_pct": 3.4,
                "passed_fwd_n": 11,
            }
        ],
        "leaks": [],
    }
    text = "\n".join(panel_module.callout_lines(state))
    assert "passed +3.4% (n=11)" in text
    assert "R" not in text.split("post_earnings")[1].split("\n")[0]


def test_a_malformed_state_is_no_lines_rather_than_a_crash():
    assert panel_module.callout_lines(None) == []
    assert panel_module.callout_lines("not a state") == []
