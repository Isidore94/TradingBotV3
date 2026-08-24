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
