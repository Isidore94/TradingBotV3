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


def _write(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


def test_a_pick_carries_its_outcome_on_one_row(tmp_path, monkeypatch):
    import project_paths

    monkeypatch.setattr(project_paths, "PERSISTENT_DATA_DIR", tmp_path)
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

    monkeypatch.setattr(project_paths, "PERSISTENT_DATA_DIR", tmp_path)
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

    monkeypatch.setattr(project_paths, "PERSISTENT_DATA_DIR", tmp_path)
    _write(
        tmp_path / "human_focus_daily_picks.csv",
        "trade_date,symbol,side,source",
        ["2026-08-07,AAPL,long,manual", "2026-08-17,MSFT,long,manual"],
    )
    assert _join_focus_week(WEEK) == []


def test_missing_csvs_are_a_quiet_week_not_an_error(tmp_path, monkeypatch):
    import project_paths

    monkeypatch.setattr(project_paths, "PERSISTENT_DATA_DIR", tmp_path)
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
