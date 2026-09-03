"""R8's retained joins, built 2026-08-18: RS/RW extremes and pick↔outcome.

Both were named as future scope in the Weekend Prep spec and explicitly not
claimed by the R8 build. What they have to get right is not arithmetic but
honesty:

- a pick with an unmatured horizon shows BLANK, never 0.00% - the review is
  read on a Saturday and a fabricated zero is a false lesson;
- an outcome whose pick snapshot is missing is still shown, and marked, because
  dropping it would quietly narrow the week;
- the RS/RW extremes fold per symbol (a name that led the tape all week must
  not bury the rest) and the weak-side bucket's "best" is its most NEGATIVE
  reading, not its largest.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6")

from ui.panels.weekend_prep_panel import (  # noqa: E402
    _join_focus_week,
    _read_rrs_week,
)

WEEK = (date(2026, 8, 10), date(2026, 8, 14))


def _redirect(monkeypatch, project_paths, tmp_path: Path) -> None:
    """Point every file this module reads at a sandbox, BY ITS CONSTANT.

    The original fixture patched `PERSISTENT_DATA_DIR` and wrote the CSVs
    directly beneath it, which is what let the panel read a directory the real
    files have never been in: the home ROOT rather than `data/runtime`. Naming
    each constant is what makes the test and the desk agree about the file.
    """
    monkeypatch.setattr(project_paths, "PERSISTENT_DATA_DIR", tmp_path)
    for name, filename in (
        ("HUMAN_FOCUS_DAILY_PICKS_FILE", "human_focus_daily_picks.csv"),
        ("HUMAN_FOCUS_OUTCOMES_FILE", "human_focus_outcomes.csv"),
        ("VETO_COHORT_PERFORMANCE_FILE", "veto_cohort_performance.csv"),
    ):
        monkeypatch.setattr(project_paths, name, tmp_path / filename)


def _write(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


def test_a_pick_carries_its_outcome_on_one_row(tmp_path, monkeypatch):
    import project_paths

    _redirect(monkeypatch, project_paths, tmp_path)
    _write(
        tmp_path / "human_focus_daily_picks.csv",
        "trade_date,symbol,side,source,snapshotted_at,active_at_snapshot",
        [
            "2026-08-11,AAPL,long,manual,2026-08-11T06:40:00,1",
            "2026-08-12,TSLA,short,auto,2026-08-12T06:40:00,1",
        ],
    )
    _write(
        tmp_path / "human_focus_outcomes.csv",
        "trade_date,symbol,side,source,h1_return,h3_return,h5_return,h10_return,matured_horizons",
        ["2026-08-11,AAPL,long,manual,0.0125,-0.004,,,2"],
    )

    rows = _join_focus_week(WEEK)
    assert [row["symbol"] for row in rows] == ["AAPL", "TSLA"]
    aapl = rows[0]
    assert aapl["h1"] == "+1.25%"
    assert aapl["h3"] == "-0.40%"
    # Not matured yet is BLANK. A zero here would say "it went nowhere".
    assert aapl["h5"] == ""
    assert aapl["h10"] == ""
    assert aapl["matured"] == "2"
    # A pick with no outcome row at all is still a pick.
    assert rows[1]["h1"] == ""


def test_the_same_name_on_both_lists_keeps_its_own_outcome(tmp_path, monkeypatch):
    """Since 2026-09-01 a name can carry a swing row AND an M5 row for one day.
    The join has to be per category or one of them shows the other's returns -
    and they are opposite trades, so that is not a rounding error.

    Fail-before-fix: on the un-fixed join both rows read +1.25%.
    """
    import project_paths

    _redirect(monkeypatch, project_paths, tmp_path)
    _write(
        tmp_path / "human_focus_daily_picks.csv",
        "trade_date,symbol,side,source,snapshotted_at,active_at_snapshot",
        [
            "2026-08-11,AMGN,long,focus_m5,2026-08-11T08:02:14,1",
            "2026-08-11,AMGN,long,focus_swing_vetted,2026-08-11T11:33:06,1",
        ],
    )
    _write(
        tmp_path / "human_focus_outcomes.csv",
        "trade_date,symbol,side,source,h1_return,h3_return,h5_return,h10_return,matured_horizons",
        [
            "2026-08-11,AMGN,long,focus_m5,0.0125,,,,1",
            "2026-08-11,AMGN,long,focus_swing_vetted,-0.0300,,,,1",
        ],
    )

    rows = _join_focus_week(WEEK)
    by_source = {row["source"]: row for row in rows}
    assert set(by_source) == {"focus_m5", "focus_swing_vetted"}
    assert by_source["focus_m5"]["h1"] == "+1.25%"
    assert by_source["focus_swing_vetted"]["h1"] == "-3.00%"


def test_an_outcome_without_a_pick_snapshot_is_kept_and_marked(tmp_path, monkeypatch):
    import project_paths

    _redirect(monkeypatch, project_paths, tmp_path)
    _write(
        tmp_path / "human_focus_outcomes.csv",
        "trade_date,symbol,side,source,h1_return,matured_horizons",
        ["2026-08-13,NVDA,long,auto,0.02,1"],
    )

    rows = _join_focus_week(WEEK)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "NVDA"
    assert "no pick snapshot" in rows[0]["source"]


def test_rows_outside_the_week_are_not_this_weeks_review(tmp_path, monkeypatch):
    import project_paths

    _redirect(monkeypatch, project_paths, tmp_path)
    _write(
        tmp_path / "human_focus_daily_picks.csv",
        "trade_date,symbol,side,source",
        ["2026-08-07,AAPL,long,manual", "2026-08-17,MSFT,long,manual"],
    )
    assert _join_focus_week(WEEK) == []


def test_missing_csvs_are_a_quiet_week_not_an_error(tmp_path, monkeypatch):
    import project_paths

    _redirect(monkeypatch, project_paths, tmp_path)
    assert _join_focus_week(WEEK) == []


def test_the_rrs_extremes_fold_per_symbol(tmp_path, monkeypatch):
    import project_paths

    log = tmp_path / "rrs_strength_extremes.csv"
    monkeypatch.setattr(project_paths, "RRS_STRENGTH_LOG_FILE", log)
    _write(
        log,
        "timestamp_local,timeframe,bucket,symbol,rrs,power_index",
        [
            "2026-08-11 07:40:42 PST,5m,strongest,RLAY,3.63,2.10",
            "2026-08-11 08:10:42 PST,5m,strongest,RLAY,4.10,2.40",
            "2026-08-12 07:40:42 PST,5m,strongest,RLAY,2.90,1.80",
            "2026-08-12 07:41:00 PST,5m,strongest,AMD,1.20,0.90",
            "2026-08-12 07:42:00 PST,5m,weakest,SNAP,-2.50,-1.10",
            "2026-08-13 07:42:00 PST,5m,weakest,SNAP,-3.80,-1.60",
        ],
    )

    rows = _read_rrs_week(WEEK)
    by_symbol = {row["symbol"]: row for row in rows}
    assert by_symbol["RLAY"]["sightings"] == 3
    assert by_symbol["RLAY"]["days"] == 2
    assert by_symbol["RLAY"]["best_rrs"] == 4.10
    assert by_symbol["RLAY"]["last_seen"] == "2026-08-12"
    # The weak bucket's "best" is its most negative reading.
    assert by_symbol["SNAP"]["best_rrs"] == -3.80
    # Most days first, so the consistent name leads its bucket.
    strongest = [row["symbol"] for row in rows if row["bucket"] == "strongest"]
    assert strongest == ["RLAY", "AMD"]


def test_rrs_rows_outside_the_week_and_unreadable_logs_are_silent(tmp_path, monkeypatch):
    import project_paths

    log = tmp_path / "rrs_strength_extremes.csv"
    monkeypatch.setattr(project_paths, "RRS_STRENGTH_LOG_FILE", log)
    _write(
        log,
        "timestamp_local,timeframe,bucket,symbol,rrs,power_index",
        ["2026-08-03 07:40:42 PST,5m,strongest,RLAY,3.63,2.10", "garbage,,,,,"],
    )
    assert _read_rrs_week(WEEK) == []

    monkeypatch.setattr(project_paths, "RRS_STRENGTH_LOG_FILE", tmp_path / "missing.csv")
    assert _read_rrs_week(WEEK) == []


# ==========================================================================
# AI-P1 - the mirror cohort (R8 §6's remaining DEFERRED join)
#
# The Focus Pick Review subtitle has promised "the veto cohort beside them"
# since the step shipped, and nothing loaded any veto_cohort file. The pane
# overclaimed; this is the join it was describing.
#
# The cohort is the trader's REJECTED picks graded forward, so it answers the
# question the accepted picks cannot: not "were my picks good" but "were the
# ones I threw away worse". On 2026-08-24 it said the short vetoes were right
# 91% of the time and the long vetoes wrong more than half - which is a
# DISCOVERY at one matured horizon, and the pane has to say so.
# ==========================================================================
def test_the_veto_cohort_is_read_and_shown_beside_the_picks(tmp_path, monkeypatch):
    import project_paths
    from ui.panels.weekend_prep_panel import _read_veto_cohort

    _redirect(monkeypatch, project_paths, tmp_path)
    _write(
        tmp_path / "veto_cohort_performance.csv",
        "cohort,side,horizon_sessions,sample_count,win_rate,avg_side_return,"
        "profit_factor,updated_at",
        [
            "human_focus_veto,ALL,1,78,0.3462,-0.002584,0.7006,2026-08-22T02:15:51-04:00",
            "human_focus_veto,LONG,1,43,0.5581,0.007353,3.3432,2026-08-22T02:15:51-04:00",
            "human_focus_veto,SHORT,1,35,0.0857,-0.014792,0.0380,2026-08-22T02:15:51-04:00",
        ],
    )

    rows = _read_veto_cohort()
    assert [row["side"] for row in rows] == ["ALL", "LONG", "SHORT"]
    longs = rows[1]
    assert longs["cohort"] == "human_focus_veto"
    assert longs["n"] == "43"
    assert longs["win_rate"] == "55.8%"
    assert longs["avg_return"] == "+0.74%"
    assert longs["profit_factor"] == "3.34"


def test_a_cohort_row_is_pooled_through_the_one_canonical_function(tmp_path, monkeypatch):
    """Never a second pooling implementation.

    The rollup is written already pooled, so calling it again is idempotent -
    but it is CALLED, so a future vocabulary bump cannot make this pane and the
    rollup disagree about which rows are the same reason.
    """
    import project_paths
    from ui.annotations import veto_cohort
    from ui.panels.weekend_prep_panel import _read_veto_cohort

    _redirect(monkeypatch, project_paths, tmp_path)
    monkeypatch.setattr(
        veto_cohort, "canonical_veto_cohort", lambda source: f"POOLED::{source}"
    )
    _write(
        tmp_path / "veto_cohort_performance.csv",
        "cohort,side,horizon_sessions,sample_count,win_rate,avg_side_return,"
        "profit_factor,updated_at",
        ["human_focus_veto_v1_too_extended,LONG,1,30,0.63,0.00885,4.42,2026-08-22T02:15:51-04:00"],
    )

    assert _read_veto_cohort()[0]["cohort"] == "POOLED::human_focus_veto_v1_too_extended"


def test_an_unmeasured_cohort_number_is_blank_never_zero(tmp_path, monkeypatch):
    """Same rule the pick join already keeps: a fabricated zero is a false
    lesson, and here it would be a false lesson about the trader's judgement."""
    import project_paths
    from ui.panels.weekend_prep_panel import _read_veto_cohort

    _redirect(monkeypatch, project_paths, tmp_path)
    _write(
        tmp_path / "veto_cohort_performance.csv",
        "cohort,side,horizon_sessions,sample_count,win_rate,avg_side_return,"
        "profit_factor,updated_at",
        ["human_focus_veto,LONG,3,4,,,,2026-08-22T02:15:51-04:00"],
    )

    row = _read_veto_cohort()[0]
    assert row["n"] == "4"
    assert row["win_rate"] == ""
    assert row["avg_return"] == ""
    assert row["profit_factor"] == ""


def test_a_missing_cohort_file_is_an_absent_state_not_a_crash(tmp_path, monkeypatch):
    import project_paths
    from ui.panels.weekend_prep_panel import _read_veto_cohort

    _redirect(monkeypatch, project_paths, tmp_path)
    assert _read_veto_cohort() == []


def test_the_subtitle_no_longer_promises_what_nothing_loaded():
    """The overclaim this packet exists to end.

    The pane has said "and the veto cohort beside them" since it shipped while
    loading no veto_cohort file at all. Either the loader exists or the
    sentence goes; this asserts the loader exists and is wired to the page.
    """
    from ui.panels import weekend_prep_panel

    source = (
        Path(weekend_prep_panel.__file__).read_text(encoding="utf-8")
    )
    assert "_read_veto_cohort" in source
    # and the page actually calls it, rather than merely defining it
    focus_page = source.split("class FocusReviewPage")[1].split("\nclass ")[0]
    assert "_read_veto_cohort()" in focus_page


def test_the_join_reads_the_directory_the_csvs_are_actually_in(tmp_path, monkeypatch):
    """The step rendered an empty table on the live desk from the day it shipped.

    `_join_focus_week` composed its paths as `PERSISTENT_DATA_DIR / name`, and
    that constant is the home ROOT (the TradingBotData folder) while both live
    under `data/runtime`. Every read missed, and the function's own "a missing
    CSV is a quiet week" forgiveness turned the miss into a plausible blank
    page rather than an error - so a whole review step quietly showed nothing
    for six days.

    This pins the files by their NAMED CONSTANTS, which is the only spelling
    that cannot drift from where the writers put them.
    """
    import project_paths

    runtime = tmp_path / "data" / "runtime"
    monkeypatch.setattr(project_paths, "PERSISTENT_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        project_paths, "HUMAN_FOCUS_DAILY_PICKS_FILE",
        runtime / "human_focus_daily_picks.csv",
    )
    monkeypatch.setattr(
        project_paths, "HUMAN_FOCUS_OUTCOMES_FILE",
        runtime / "human_focus_outcomes.csv",
    )
    _write(
        runtime / "human_focus_daily_picks.csv",
        "trade_date,symbol,side,source",
        ["2026-08-11,AAPL,long,manual"],
    )

    rows = _join_focus_week(WEEK)
    assert [row["symbol"] for row in rows] == ["AAPL"]


def test_the_cohort_reads_its_named_constant_too(tmp_path, monkeypatch):
    import project_paths
    from ui.panels.weekend_prep_panel import _read_veto_cohort

    runtime = tmp_path / "data" / "runtime"
    monkeypatch.setattr(project_paths, "PERSISTENT_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        project_paths, "VETO_COHORT_PERFORMANCE_FILE",
        runtime / "veto_cohort_performance.csv",
    )
    _write(
        runtime / "veto_cohort_performance.csv",
        "cohort,side,horizon_sessions,sample_count,win_rate,avg_side_return,"
        "profit_factor,updated_at",
        ["human_focus_veto,ALL,1,78,0.3462,-0.002584,0.7006,2026-08-22T02:15:51-04:00"],
    )

    assert [row["n"] for row in _read_veto_cohort()] == ["78"]


# ==========================================================================
# Packet 8b - the LIKE cohort beside the veto one
#
# R10.F graded the trader's endorsements for the first time, and nothing read
# the file. The two cohorts are the halves of one judgement: the veto table
# answers "was I right to throw that away", this one "was I right to like it".
# Reading either alone gives half an answer - and if you only kept the vetoes,
# the half you get is the flattering one.
# ==========================================================================
def test_the_like_cohort_is_read_by_its_named_constant(tmp_path, monkeypatch):
    """By CONSTANT, never by composing a filename: AI-P1 found this step had
    rendered an empty table for six days from exactly that mistake."""
    import project_paths
    from ui.panels.weekend_prep_panel import _read_like_cohort

    runtime = tmp_path / "data" / "runtime"
    monkeypatch.setattr(project_paths, "PERSISTENT_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        project_paths, "LIKE_COHORT_PERFORMANCE_FILE",
        runtime / "like_cohort_performance.csv",
    )
    _write(
        runtime / "like_cohort_performance.csv",
        "cohort,side,horizon_sessions,sample_count,win_rate,avg_side_return,"
        "profit_factor,updated_at",
        [
            "human_focus_like,LONG,1,21,0.8095,0.019011,6.5473,2026-08-24T16:48:57",
            "human_focus_like_post_earnings_52w_break,ALL,1,4,,,,2026-08-24T16:48:57",
        ],
    )

    rows = _read_like_cohort()

    assert [row["n"] for row in rows] == ["21", "4"]
    assert rows[0]["win_rate"] == "81.0%"
    assert rows[0]["avg_return"] == "+1.90%"
    assert rows[0]["profit_factor"] == "6.55"


def test_an_unmeasured_like_statistic_is_blank_never_zero(tmp_path, monkeypatch):
    """Same rule the veto table keeps: a fabricated zero is a false lesson,
    and here it would be a false lesson about the trader's own conviction."""
    import project_paths
    from ui.panels.weekend_prep_panel import _read_like_cohort

    runtime = tmp_path / "data" / "runtime"
    monkeypatch.setattr(project_paths, "PERSISTENT_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        project_paths, "LIKE_COHORT_PERFORMANCE_FILE",
        runtime / "like_cohort_performance.csv",
    )
    _write(
        runtime / "like_cohort_performance.csv",
        "cohort,side,horizon_sessions,sample_count,win_rate,avg_side_return,"
        "profit_factor,updated_at",
        ["human_focus_like,LONG,3,2,,,,2026-08-24T16:48:57"],
    )

    row = _read_like_cohort()[0]
    assert row["n"] == "2"
    assert row["win_rate"] == "" and row["avg_return"] == "" and row["profit_factor"] == ""


def test_a_missing_like_file_is_an_absent_state_not_a_crash(tmp_path, monkeypatch):
    import project_paths
    from ui.panels.weekend_prep_panel import _read_like_cohort

    monkeypatch.setattr(project_paths, "PERSISTENT_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        project_paths, "LIKE_COHORT_PERFORMANCE_FILE", tmp_path / "nope.csv"
    )
    assert _read_like_cohort() == []


def test_the_subtitle_promises_both_cohorts_and_the_page_loads_both():
    """The subtitle overclaimed once already (AI-P1). It must not again."""
    from ui.panels import weekend_prep_panel

    source = Path(weekend_prep_panel.__file__).read_text(encoding="utf-8")
    focus_page = source.split("class FocusReviewPage")[1].split("\nclass ")[0]

    assert "_read_like_cohort()" in focus_page
    assert "_read_veto_cohort()" in focus_page
    assert "what you vetoed and what you liked" in focus_page


# ---------------------------------------------------------------------------
# R8 §6's LAST deferred joins, built 2026-08-24 (packet W2). What was still
# owed after AI-P1: the `human_focus_performance.csv` rollup and the
# `pick_feedback.jsonl` verdicts in Focus Pick Review, and the
# `rrs_group_strength_extremes.csv` stream in Week in Review beside the symbol
# stream it already folds.
#
# Every one of them reads by NAMED CONSTANT. That rule is not stylistic: this
# step shipped an empty Focus table on the live desk for six days because one
# reader composed a filename under the wrong directory, and the fixture encoded
# the same wrong assumption, so nothing caught it.
# ---------------------------------------------------------------------------


def _redirect_w2(monkeypatch, project_paths, tmp_path: Path) -> None:
    for name, filename in (
        ("HUMAN_FOCUS_PERFORMANCE_FILE", "human_focus_performance.csv"),
        ("PICK_FEEDBACK_FILE", "pick_feedback.jsonl"),
        ("RRS_GROUP_STRENGTH_LOG_FILE", "rrs_group_strength_extremes.csv"),
    ):
        monkeypatch.setattr(project_paths, name, tmp_path / filename)


def test_the_focus_performance_rollup_is_read_by_its_named_constant(tmp_path, monkeypatch):
    import project_paths
    from ui.panels.weekend_prep_panel import _read_focus_performance

    _redirect_w2(monkeypatch, project_paths, tmp_path)
    _write(
        tmp_path / "human_focus_performance.csv",
        "cohort,side,horizon_sessions,sample_count,win_rate,avg_side_return,"
        "profit_factor,updated_at,median_return,symbols,sessions,ci_low,ci_high,ci_basis",
        [
            "human_focus_pick,LONG,1,44,0.5000,0.004210,1.4400,2026-08-24T02:00:00-04:00,"
            "0.0031,21,9,-0.001,0.009,5-95 percentile of a session-block bootstrap on the mean",
        ],
    )

    rows = _read_focus_performance()
    assert len(rows) == 1
    row = rows[0]
    assert row["cohort"] == "human_focus_pick"
    assert row["n"] == "44"
    assert row["win_rate"] == "50.0%"
    assert row["avg_return"] == "+0.42%"
    assert row["symbols"] == "21" and row["sessions"] == "9"
    assert row["ci"] == "[-0.001, 0.009]"
    assert row["updated_at"] == "2026-08-24T02:00:00-04:00"


def test_an_unmeasured_performance_number_is_blank_and_an_unmeasured_ci_says_why(
    tmp_path, monkeypatch
):
    """Blank, never zero - and an absent interval carries its reason."""
    import project_paths
    from ui.panels.weekend_prep_panel import _read_focus_performance

    _redirect_w2(monkeypatch, project_paths, tmp_path)
    _write(
        tmp_path / "human_focus_performance.csv",
        "cohort,side,horizon_sessions,sample_count,win_rate,avg_side_return,"
        "profit_factor,updated_at,median_return,symbols,sessions,ci_low,ci_high,ci_basis",
        [
            "human_focus_pick,SHORT,10,3,,,,2026-08-24T02:00:00-04:00,,3,1,,,"
            "unmeasured: only 1 session in the sample",
        ],
    )

    row = _read_focus_performance()[0]
    assert row["n"] == "3"
    assert row["win_rate"] == "" and row["avg_return"] == "" and row["profit_factor"] == ""
    assert row["ci"] == ""
    assert "only 1 session" in row["ci_basis"]


def test_a_missing_performance_rollup_is_an_absent_state_not_a_crash(tmp_path, monkeypatch):
    import project_paths
    from ui.panels.weekend_prep_panel import _read_focus_performance

    _redirect_w2(monkeypatch, project_paths, tmp_path)
    assert _read_focus_performance() == []


def test_pick_feedback_is_filtered_to_the_reviewed_week(tmp_path, monkeypatch):
    import json

    import project_paths
    from ui.panels.weekend_prep_panel import _read_pick_feedback_week

    _redirect_w2(monkeypatch, project_paths, tmp_path)
    rows = [
        {"ts": "2026-08-11T07:00:00", "trade_date": "2026-08-11", "symbol": "AAPL",
         "side": "LONG", "verdict": "like", "category": "", "origin": "swing", "reason": ""},
        {"ts": "2026-08-13T07:00:00", "trade_date": "2026-08-13", "symbol": "TSLA",
         "side": "SHORT", "verdict": "dislike", "category": "extended", "origin": "m5",
         "reason": "too far from the level"},
        {"ts": "2026-08-03T07:00:00", "trade_date": "2026-08-03", "symbol": "NVDA",
         "side": "LONG", "verdict": "like", "category": "", "origin": "", "reason": ""},
    ]
    (tmp_path / "pick_feedback.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    week = _read_pick_feedback_week(WEEK)
    assert [row["symbol"] for row in week] == ["AAPL", "TSLA"], "the prior week is not this week"
    assert week[1]["verdict"] == "dislike"
    assert week[1]["reason"] == "too far from the level"


def test_a_missing_feedback_file_is_an_absent_state_not_a_crash(tmp_path, monkeypatch):
    import project_paths
    from ui.panels.weekend_prep_panel import _read_pick_feedback_week

    _redirect_w2(monkeypatch, project_paths, tmp_path)
    assert _read_pick_feedback_week(WEEK) == []


def test_the_group_rs_stream_folds_per_group_and_keeps_both_extremes(tmp_path, monkeypatch):
    """The group log records no bucket, so direction is read from the sign.

    `_log_group_strength_extremes` writes the top and the bottom of each list
    with identical columns - unlike the symbol log, which stamps a `bucket`. So
    the fold reports the strongest AND the weakest reading it saw rather than
    inventing a direction the file never recorded.
    """
    import project_paths
    from ui.panels.weekend_prep_panel import _read_rrs_group_week

    _redirect_w2(monkeypatch, project_paths, tmp_path)
    _write(
        tmp_path / "rrs_group_strength_extremes.csv",
        "timestamp_local,timeframe,group_type,group_key,etf,rrs,power_index",
        [
            "2026-08-11 07:05:00 PDT,D1,sector,Information Technology,XLK,1.8000,0.9",
            "2026-08-12 07:05:00 PDT,D1,sector,Information Technology,XLK,-0.4000,0.2",
            "2026-08-12 07:05:00 PDT,D1,sector,Energy,XLE,-2.1000,0.1",
            "2026-08-12 07:05:00 PDT,D1,industry,SMH,SMH,2.4000,0.8",
            "2026-08-03 07:05:00 PDT,D1,sector,Energy,XLE,9.9000,0.9",
        ],
    )

    rows = _read_rrs_group_week(WEEK)
    tech = [row for row in rows if row["group_key"] == "Information Technology"][0]
    assert tech["group_type"] == "sector"
    assert tech["sightings"] == 2 and tech["days"] == 2
    assert tech["max_rrs"] == 1.8 and tech["min_rrs"] == -0.4
    assert tech["last_seen"] == "2026-08-12"
    energy = [row for row in rows if row["group_key"] == "Energy"][0]
    assert energy["sightings"] == 1, "the prior week is not this week"
    assert energy["max_rrs"] == -2.1, "a week with only weak readings has no positive best"
    assert {row["group_type"] for row in rows} == {"sector", "industry"}


def test_a_missing_group_log_is_a_quieter_week_not_an_error(tmp_path, monkeypatch):
    import project_paths
    from ui.panels.weekend_prep_panel import _read_rrs_group_week

    _redirect_w2(monkeypatch, project_paths, tmp_path)
    assert _read_rrs_group_week(WEEK) == []


def test_the_dead_reader_that_carried_the_wrong_path_is_gone():
    """`_read_focus_week` resolved its CSVs under the home ROOT.

    It was superseded by `_join_focus_week` in 2026-08-18 and left behind. AI-P1
    fixed the live reader; this removes the copy that still encodes the defect,
    so nobody restores it by reaching for the nearest-looking helper.
    """
    from ui.panels import weekend_prep_panel

    assert not hasattr(weekend_prep_panel, "_read_focus_week")


def test_the_new_streams_are_wired_to_their_pages_not_merely_defined():
    """The AI-P1 lesson, applied forward.

    A reader that exists but is never called renders exactly the same blank
    page as a reader that reads the wrong directory, and the page's own
    forgiveness makes both look like a quiet week.
    """
    from ui.panels import weekend_prep_panel

    source = Path(weekend_prep_panel.__file__).read_text(encoding="utf-8")
    week_page = source.split("class WeekReviewPage")[1].split("\nclass ")[0]
    focus_page = source.split("class FocusReviewPage")[1].split("\nclass ")[0]

    # CHANGED BY V2 item 2c: the RS/RW extremes left this page for the live
    # board that answers the same question. See
    # `test_the_rs_extremes_are_deliberately_unwired` below - the lesson this
    # test states still holds, and it now points the other way for these two.
    assert "_read_focus_performance()" in focus_page
    assert "_read_pick_feedback_week(" in focus_page


def test_the_rs_extremes_are_deliberately_unwired():
    """The AI-P1 lesson, pointing the other way.

    A reader that exists but is never called renders the same blank page as a
    broken one - so an uncalled reader has to SAY it is uncalled, or the next
    agent will "fix" a page by wiring it back and put the wall of text the
    trader complained about straight back on their Saturday.
    """
    from ui.panels import weekend_prep_panel

    source = Path(weekend_prep_panel.__file__).read_text(encoding="utf-8")
    week_page = source.split("class WeekReviewPage")[1].split(chr(10) + "class ")[0]

    assert "_read_rrs_week(" not in week_page
    assert "_read_rrs_group_week(" not in week_page
    assert "_rrs_lines" not in week_page
    assert "_rrs_group_lines" not in week_page

    # The scans are kept, and they say why they have no caller.
    assert "NO CALLER SINCE V2, AND KEPT ON PURPOSE" in source
    assert callable(weekend_prep_panel._read_rrs_week)
    assert callable(weekend_prep_panel._read_rrs_group_week)


def test_the_group_cap_and_the_absent_log_message_retired_with_the_block():
    """RETIRED BY V2 item 2c, and named rather than quietly deleted.

    Two tests lived here: one that the group cap PRINTS what it dropped (a
    silent top-N reads as "that was all of it"), and one that an absent log says
    so on the page rather than showing a blank. Both asserted the behaviour of
    `WeekReviewPage._rrs_group_lines`, which V2 removed along with the rest of
    the RS/RW prose - the desk has a live board for that question, and the block
    was the second-largest part of the wall of text the trader complained about.

    The two rules they protected are NOT retired; they are protected wherever
    those readers are next printed. What is gone is the printer, not the log, and
    `_read_rrs_group_week` is still exported and still tested by
    `test_the_group_rs_stream_folds_per_group_and_keeps_both_extremes` and
    `test_a_missing_group_log_is_a_quieter_week_not_an_error`.
    """
    from ui.panels import weekend_prep_panel

    assert not hasattr(weekend_prep_panel.WeekReviewPage, "_rrs_group_lines")
    assert callable(weekend_prep_panel._read_rrs_group_week)
