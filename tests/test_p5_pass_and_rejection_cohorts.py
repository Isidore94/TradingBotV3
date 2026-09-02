"""P5: the last three verdicts get a forward record.

The veto cohort has graded what the trader threw away since it shipped, and the
like cohort what they endorsed. Three verdicts had no forward record at all:

* the day-trade **pass** - "I really like this stock for a daytrade but it has
  this ONE issue" (trader, 2026-08-31);
* **not_today** - one auto-adopted pick thrown back for one session, 223 of
  them on the live log;
* **dislike** - the name itself, 34 of them.

The rules these tests hold, each one a decision rather than a preference:

* the pass vocabulary is a SEPARATE family and is never folded into the veto's;
* cohort identity on write is (vocab_version, reason_code), and rows are never
  rewritten;
* a pass is MULTI-SELECT, so it grades in k code cohorts AND `pass_all`, which
  makes the code cohorts overlapping and unsummable;
* `not_today` and `dislike` are separate cohorts and are never pooled;
* a missing intraday grade is BLANK with a stated reason, never a zero;
* no test asserts a literal `vocab_version`.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))



class _Lazy:
    """Resolve the module at ATTRIBUTE time, not import time.

    Both modules are new in P5, so importing them at the top would make this
    whole file fail to COLLECT on the un-fixed tree - one error instead of the
    per-test failures that show which contract is missing.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attribute: str):
        import importlib

        return getattr(importlib.import_module(self._name), attribute)


rejection_cohort = _Lazy("rejection_cohort")
pass_cohort = _Lazy("ui.annotations.pass_cohort")

NOW = datetime(2026, 9, 1, 20, 0, tzinfo=timezone.utc)
PASSED_AT = datetime(2026, 9, 1, 7, 0, tzinfo=timezone(timedelta(hours=-7)))


def _loaded_pass_version() -> int:
    """The CURRENT pass vocabulary version, LOADED - never a literal.

    House rule: a version bump must not have to edit a test.
    """
    from ui.annotations.vocabulary import load_pass_vocabulary

    return int(load_pass_vocabulary().vocab_version)


def _pass(symbol="AAA", side="LONG", codes=("poor_market_conditions",), **extra) -> dict:
    row = {
        "event_type": "pass",
        "event_id": "e" * 32,
        "symbol": symbol,
        "side": side,
        "session_date": "2026-09-01",
        "created_at": PASSED_AT.isoformat(),
        "reason_codes": list(codes),
        "vocab_version": _loaded_pass_version(),
        "vocabulary_id": "pass_reasons",
        "timeframe": "D1",
        "source": "chart_review",
    }
    row.update(extra)
    return row


def _bar(minute, *, close, high=None, low=None, day=1, hour=7):
    stamp = datetime(2026, 9, day, hour, minute)
    return {
        "dt": stamp.isoformat(),
        "open": close,
        "high": high if high is not None else close + 0.5,
        "low": low if low is not None else close - 0.5,
        "close": close,
    }


# ==========================================================================
# the cohort key
# ==========================================================================
def test_the_pass_cohort_key_carries_its_vocabulary_version():
    """Identity on write is (vocab_version, reason_code), exactly as the veto
    cohort does it. Asserted against the LOADED vocabulary, never a literal.

    Fail-before-fix: `pass_cohort` does not exist.
    """
    version = _loaded_pass_version()
    assert pass_cohort.pass_cohort_source("too_extended", version) == f"pass_v{version}_too_extended"
    # No version returns the historical unversioned form, which is what keeps
    # a row already on disk grading where it was filed.
    assert pass_cohort.pass_cohort_source("too_extended") == "pass_too_extended"
    assert pass_cohort.pass_cohort_source("") == "pass_uncoded"


def test_the_pass_family_is_never_folded_into_the_veto_family():
    """Separate vocabularies, separate cohorts (trader decision 2026-08-31).
    The veto cohorts are already accruing forward returns."""
    from human_focus_tracking import _outcome_base_cohort

    version = _loaded_pass_version()
    pass_source = pass_cohort.pass_cohort_source("too_extended_from_base", version)
    assert _outcome_base_cohort({"source": pass_source}) == "human_focus_pass"
    assert not pass_source.startswith("veto")


def test_the_rejection_prefix_cannot_reach_a_focus_cohort():
    """The DOUBLE underscore is load-bearing: `_outcome_base_cohort` matches
    `startswith(prefix + "_")`, so `focus_` claims `focus__not_today` and can
    never claim `focus_swing`, `focus_m5` or `focus_pick`."""
    from human_focus_tracking import _outcome_base_cohort

    assert _outcome_base_cohort({"source": "focus__not_today"}) == "human_focus_rejection"
    assert _outcome_base_cohort({"source": "focus__dislike"}) == "human_focus_rejection"
    for untouched in ("focus_swing", "focus_swing_vetted", "focus_m5", "focus_pick"):
        assert _outcome_base_cohort({"source": untouched}) != "human_focus_rejection"


# ==========================================================================
# a pass is multi-select
# ==========================================================================
def test_a_pass_with_two_codes_appears_in_exactly_three_cohorts():
    """k code cohorts plus the pooled `pass_all`.

    Fail-before-fix: nothing produces pass cohort rows at all.
    """
    rows, skipped = pass_cohort.pass_pick_rows(
        [_pass(codes=("poor_market_conditions", "spread_too_wide"))], now=NOW
    )

    version = _loaded_pass_version()
    assert skipped == 0
    assert {row["source"] for row in rows} == {
        pass_cohort.pass_cohort_source("poor_market_conditions", version),
        pass_cohort.pass_cohort_source("spread_too_wide", version),
        pass_cohort.PASS_ALL_SOURCE,
    }
    assert len(rows) == 3
    # Every row says how many cohorts this one pass entered, so the overlap is
    # readable from the file itself.
    assert {row["reason_code_count"] for row in rows} == {2}


def test_the_overlap_is_stated_where_the_numbers_are():
    """The code cohorts share samples by construction. A reader who never opens
    the module must still be unable to add them up by accident."""
    assert "never be summed" in pass_cohort.OVERLAP_NOTE
    assert "pass_all" in pass_cohort.OVERLAP_NOTE


def test_a_pass_with_no_codes_still_grades_in_the_pool():
    rows, _ = pass_cohort.pass_pick_rows([_pass(codes=())], now=NOW)
    assert [row["source"] for row in rows] == [pass_cohort.PASS_ALL_SOURCE]
    assert rows[0]["reason_code_count"] == 0


def test_a_sideless_pass_is_counted_and_never_graded():
    """The veto and like cohorts' rule, kept verbatim: forward returns are
    side-adjusted and a blank side reads as LONG downstream."""
    rows, skipped = pass_cohort.pass_pick_rows([_pass(side="")], now=NOW)
    assert rows == []
    assert skipped == 1


def test_the_multi_source_key_keeps_the_cohorts_apart():
    """Every row of one pass shares a date, a symbol and a side. Under the
    default (date, symbol, side) key they would collapse into one outcome row
    and k of the k+1 cohorts would silently vanish."""
    from human_focus_tracking import _pick_key, pick_key_with_source

    rows, _ = pass_cohort.pass_pick_rows(
        [_pass(codes=("poor_market_conditions", "spread_too_wide"))], now=NOW
    )
    assert len({_pick_key(row) for row in rows}) == 1, "the default key collapses them"
    assert len({pick_key_with_source(row) for row in rows}) == 3


def test_the_default_pick_key_is_unchanged_for_every_existing_caller():
    import inspect

    from human_focus_tracking import update_human_focus_outcomes

    assert inspect.signature(update_human_focus_outcomes).parameters["pick_key"].default is None


# ==========================================================================
# the same-session grade
# ==========================================================================
def test_a_pass_without_a_sidecar_has_blank_intraday_columns():
    """Never a zero: a pass the desk held no bars for is unmeasured, and a zero
    would read as "it went nowhere"."""
    rows, _ = pass_cohort.pass_pick_rows([_pass()], now=NOW)
    for row in rows:
        assert row["intraday_close_r"] == ""
        assert row["intraday_entry_price"] == ""
        assert row["intraday_first_hit"] == ""
        assert row["intraday_unmeasured_reason"] == "no_sidecar_bars"


def test_the_entry_is_the_first_completed_close_AFTER_the_pass():
    """The trader could not have traded a bar that had not finished, and
    entering on one would be reading a price they never saw."""
    bars = [
        _bar(0, close=100.0, low=99.0),
        _bar(5, close=101.0, low=99.5),  # last bar before the pass
        _bar(10, close=102.0, low=101.0),  # first bar at/after 07:00... see below
    ]
    # Pass at 07:07, so the 07:10 bar is the first one starting after it.
    passed = datetime(2026, 9, 1, 7, 7, tzinfo=None)
    outcome = pass_cohort.intraday_pass_outcome(bars, side="LONG", passed_at=passed)

    assert outcome["intraday_entry_at"].endswith("07:10:00")
    assert outcome["intraday_entry_price"] == 102.0


def test_a_long_stop_is_the_session_low_up_to_entry_and_the_target_is_2R():
    bars = [
        _bar(0, close=100.0, low=98.0, high=100.5),
        _bar(5, close=101.0, low=100.0, high=101.5),
        _bar(10, close=102.0, low=101.0, high=102.5),  # entry
        _bar(15, close=103.0, low=102.0, high=110.0),  # target: 102 + 2*(102-98)=110
    ]
    outcome = pass_cohort.intraday_pass_outcome(
        bars, side="LONG", passed_at=datetime(2026, 9, 1, 7, 7)
    )
    assert outcome["intraday_stop_price"] == 98.0
    assert outcome["intraday_risk_per_share"] == 4.0
    assert outcome["intraday_first_hit"] == "TARGET"
    assert outcome["intraday_close_r"] == 2.0


def test_a_bar_that_touches_both_is_scored_STOP_FIRST():
    """Assuming the good fill on an ambiguous bar is how a backtest flatters
    itself - the same convention the warehouse and the scoreboard use."""
    bars = [
        _bar(0, close=100.0, low=98.0, high=100.5),
        _bar(10, close=102.0, low=101.0, high=102.5),  # entry, risk 4
        _bar(15, close=100.0, low=97.0, high=111.0),  # hits stop AND target
    ]
    outcome = pass_cohort.intraday_pass_outcome(
        bars, side="LONG", passed_at=datetime(2026, 9, 1, 7, 5)
    )
    assert outcome["intraday_first_hit"] == "STOP"
    assert outcome["intraday_close_r"] == -1.0


def test_an_unresolved_trade_is_marked_at_the_session_close():
    bars = [
        _bar(0, close=100.0, low=98.0, high=100.5),
        _bar(10, close=102.0, low=101.5, high=102.5),  # entry, risk 4
        _bar(15, close=104.0, low=103.0, high=104.5),
    ]
    outcome = pass_cohort.intraday_pass_outcome(
        bars, side="LONG", passed_at=datetime(2026, 9, 1, 7, 5)
    )
    assert outcome["intraday_first_hit"] == "SESSION_CLOSE"
    assert outcome["intraday_close_r"] == pytest.approx(0.5)


def test_a_short_uses_the_session_HIGH_and_mirrors():
    bars = [
        _bar(0, close=100.0, low=99.5, high=104.0),
        _bar(10, close=100.0, low=99.5, high=100.5),  # entry, risk 4
        _bar(15, close=92.0, low=91.0, high=99.0),  # target 100-8=92
    ]
    outcome = pass_cohort.intraday_pass_outcome(
        bars, side="SHORT", passed_at=datetime(2026, 9, 1, 7, 5)
    )
    assert outcome["intraday_stop_price"] == 104.0
    assert outcome["intraday_first_hit"] == "TARGET"


def test_a_stop_at_the_entry_is_not_a_trade_anybody_took():
    """No risk means no R. Dividing by it would manufacture a trade."""
    bars = [
        _bar(10, close=100.0, low=100.0, high=100.5),
        _bar(15, close=101.0, low=100.5, high=101.5),
    ]
    outcome = pass_cohort.intraday_pass_outcome(
        bars, side="LONG", passed_at=datetime(2026, 9, 1, 7, 5)
    )
    assert outcome["intraday_close_r"] == ""
    assert outcome["intraday_unmeasured_reason"] == "stop_at_or_through_entry"


def test_a_naive_bar_and_an_aware_pass_are_compared_on_one_clock():
    """CLAUDE.md: normalize by ATTACHING market-local to the naive side, never
    by stripping the aware side. Stripping would move the comparison by whole
    hours and pick the wrong entry bar."""
    bars = [_bar(10, close=102.0, low=98.0, high=102.5), _bar(15, close=103.0, low=102.0)]
    aware = datetime(2026, 9, 1, 7, 5, tzinfo=timezone(timedelta(hours=-7)))
    outcome = pass_cohort.intraday_pass_outcome(bars, side="LONG", passed_at=aware)
    assert outcome["intraday_entry_at"].endswith("07:10:00")


def test_a_sidecar_that_ends_before_the_entry_says_so():
    """MEASURED on the live desk: the sidecar is written from the bars the desk
    was ALREADY holding, so every bar in it starts before the pass. The reason
    is stated rather than left as an ambiguous blank."""
    bars = [_bar(0, close=100.0), _bar(5, close=101.0)]
    outcome = pass_cohort.intraday_pass_outcome(
        bars, side="LONG", passed_at=datetime(2026, 9, 1, 9, 0)
    )
    assert outcome["intraday_close_r"] == ""
    assert outcome["intraday_unmeasured_reason"] == "sidecar_ends_before_the_entry_bar"


# ==========================================================================
# the rejection cohort
# ==========================================================================
def _feedback(verdict="not_today", symbol="BBB", side="LONG", **extra) -> dict:
    row = {
        "verdict": verdict,
        "symbol": symbol,
        "side": side,
        "trade_date": "2026-08-20",
        "ts": "2026-08-20T08:13:31",
        "category": "m5",
        "origin": "auto_pick",
        "reason": "not today",
    }
    row.update(extra)
    return row


def test_not_today_and_dislike_are_separate_cohorts_and_never_pooled():
    """`pick_feedback` has kept them distinct since packet R2: a same-day
    throwback and a judgement on the name are different claims.

    Fail-before-fix: `rejection_cohort` does not exist.
    """
    rows, _ = rejection_cohort.rejection_pick_rows(
        [_feedback("not_today"), _feedback("dislike", symbol="CCC")], now=NOW
    )
    assert {row["source"] for row in rows} == {"focus__not_today", "focus__dislike"}


def test_the_free_text_reason_is_carried_and_never_coded():
    """The whole value of a dislike is the sentence. Turning it into machine
    categories is a separate decision with a vocabulary behind it."""
    said = "too weak today needs to be over the 20sma"
    rows, _ = rejection_cohort.rejection_pick_rows(
        [_feedback("dislike", reason=said)], now=NOW
    )
    assert rows[0]["reason"] == said
    assert "reason" in rejection_cohort.PICK_COLUMNS


def test_like_and_unfavorite_are_not_graded_here():
    """`like` belongs to the like cohort. `unfavorite` is a membership change
    rather than a verdict, and the live rows carry no side at all - grading one
    would manufacture a direction the trader never expressed."""
    rows, _ = rejection_cohort.rejection_pick_rows(
        [_feedback("like"), _feedback("unfavorite", side="")], now=NOW
    )
    assert rows == []
    assert rejection_cohort.GRADED_VERDICTS == ("not_today", "dislike")


def test_a_sideless_rejection_is_counted_and_never_graded():
    rows, skipped = rejection_cohort.rejection_pick_rows([_feedback(side="")], now=NOW)
    assert rows == []
    assert skipped == 1


def test_a_rejection_is_dated_by_the_session_it_is_about():
    """Never by `ts`, which is when it was typed. A verdict entered on Saturday
    about Friday belongs to Friday."""
    rows, _ = rejection_cohort.rejection_pick_rows(
        [_feedback(trade_date="2026-08-20", ts="2026-08-22T10:00:00")], now=NOW
    )
    assert rows[0]["trade_date"] == "2026-08-20"
    assert rows[0]["session_date"] == "2026-08-20"


def test_one_name_can_carry_both_verdicts_on_one_day():
    from human_focus_tracking import pick_key_with_source

    rows, _ = rejection_cohort.rejection_pick_rows(
        [_feedback("not_today"), _feedback("dislike")], now=NOW
    )
    assert len({pick_key_with_source(row) for row in rows}) == 2


def test_a_corrupt_feedback_line_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "pick_feedback.jsonl"
    path.write_text(
        json.dumps(_feedback()) + "\n{not json at all\n" + json.dumps(_feedback(symbol="DDD")) + "\n",
        encoding="utf-8",
    )
    assert len(rejection_cohort.load_rejection_feedback(path)) == 2


def test_a_missing_feedback_log_is_no_rows_rather_than_a_crash(tmp_path):
    assert rejection_cohort.load_rejection_feedback(tmp_path / "nothing.jsonl") == []


# ==========================================================================
# merges, slots and surfaces
# ==========================================================================
def test_both_merges_are_idempotent(tmp_path):
    """What makes it safe to run at capture time as well as nightly."""
    annotations = tmp_path / "trader_annotations.jsonl"
    annotations.write_text(json.dumps(_pass()) + "\n", encoding="utf-8")
    picks = tmp_path / "pass_picks.csv"

    first = pass_cohort.merge_pass_cohort_picks(
        annotations_path=annotations, picks_path=picks, now=NOW
    )
    second = pass_cohort.merge_pass_cohort_picks(
        annotations_path=annotations, picks_path=picks, now=NOW
    )
    assert first["added"] == 2 and second["added"] == 0
    assert first["total_rows"] == second["total_rows"]

    feedback = tmp_path / "pick_feedback.jsonl"
    feedback.write_text(json.dumps(_feedback()) + "\n", encoding="utf-8")
    rej = tmp_path / "rejection_picks.csv"
    one = rejection_cohort.merge_rejection_cohort_picks(
        feedback_path=feedback, picks_path=rej, now=NOW
    )
    two = rejection_cohort.merge_rejection_cohort_picks(
        feedback_path=feedback, picks_path=rej, now=NOW
    )
    assert one["added"] == 1 and two["added"] == 0


def test_both_slots_are_appended_and_nothing_was_reordered():
    from ai_jobs.runner import default_slots

    names = [slot.name for slot in default_slots()]
    assert names[:5] == [
        "journal_import",
        "ai_summary",
        "ticker_briefs",
        "veto_cohort_grading",
        "like_cohort_grading",
    ]
    assert names[5:7] == ["pass_cohort_grading", "rejection_cohort_grading"]
    for name in ("pass_cohort_grading", "rejection_cohort_grading"):
        slot = next(item for item in default_slots() if item.name == name)
        assert slot.reserve_minutes == 5.0


def test_both_slots_call_no_model():
    import inspect

    from ai_jobs import cohorts

    for func in (cohorts.run_pass_cohort_grading, cohorts.run_rejection_cohort_grading):
        source = inspect.getsource(func)
        for forbidden in ("request_ai_summary", "local_model", "narrate"):
            assert forbidden not in source


def test_the_trader_judgement_scope_reads_every_verdict():
    """It read the veto trio only, which asks "were your rejections wrong?" and
    never "were your endorsements right?" - the flattering half."""
    import inspect

    import ai_summary
    from project_paths import (
        LIKE_COHORT_PERFORMANCE_FILE,
        PASS_COHORT_PERFORMANCE_FILE,
        REJECTION_COHORT_PERFORMANCE_FILE,
    )

    text = inspect.getsource(ai_summary)
    for key in (
        "judgement.like_performance",
        "judgement.pass_performance",
        "judgement.rejection_performance",
    ):
        assert key in text, f"{key} is not in the trader_judgement scope"
    # And the scope addresses them BY CONSTANT, which is the spelling that
    # cannot drift from where the writers put them.
    for constant in (
        "LIKE_COHORT_PERFORMANCE_FILE",
        "PASS_COHORT_PERFORMANCE_FILE",
        "REJECTION_COHORT_PERFORMANCE_FILE",
    ):
        assert constant in text
    assert LIKE_COHORT_PERFORMANCE_FILE.name == "like_cohort_performance.csv"
    assert PASS_COHORT_PERFORMANCE_FILE.name == "pass_cohort_performance.csv"
    assert REJECTION_COHORT_PERFORMANCE_FILE.name == "rejection_cohort_performance.csv"


def test_the_evidence_report_carries_both_new_cohorts():
    import inspect

    from ai_jobs import evidence_report

    text = inspect.getsource(evidence_report)
    assert '("pass", _read_pass_cohort_rows)' in text
    assert '("rejection", _read_rejection_cohort_rows)' in text


def test_the_weekend_prep_page_shows_both():
    pytest.importorskip("PySide6")
    from ui.panels import weekend_prep_panel as panel_module

    assert hasattr(panel_module, "_read_pass_cohort")
    assert hasattr(panel_module, "_read_rejection_cohort")
    # The six legacy columns PLUS the two evidence columns the older tables
    # drop: a new table has no legacy shape to preserve.
    assert "meets_n_floor" in panel_module.P5_COHORT_COLUMNS
    assert "evidence" in panel_module.P5_COHORT_COLUMNS


def test_the_weekend_prep_reader_reformats_and_never_derives(tmp_path):
    pytest.importorskip("PySide6")
    from ui.panels import weekend_prep_panel as panel_module

    path = tmp_path / "pass_cohort_performance.csv"
    path.write_text(
        "cohort,side,horizon_sessions,sample_count,win_rate,avg_side_return,"
        "profit_factor,meets_n_floor,evidence_label\n"
        "human_focus_pass_all,ALL,3,42,0.5238,-0.0123,0.8100,1,discovery\n",
        encoding="utf-8",
    )
    rows = panel_module._read_p5_cohort(path)
    assert rows[0]["cohort"] == "human_focus_pass_all"
    assert rows[0]["win_rate"] == "52.4%"
    assert rows[0]["avg_return"] == "-1.23%"
    assert rows[0]["meets_n_floor"] == "1"
    assert rows[0]["evidence"] == "discovery"


def test_a_missing_rollup_is_a_quieter_page_not_an_error(tmp_path):
    pytest.importorskip("PySide6")
    from ui.panels import weekend_prep_panel as panel_module

    assert panel_module._read_p5_cohort(tmp_path / "nothing.csv") == []
