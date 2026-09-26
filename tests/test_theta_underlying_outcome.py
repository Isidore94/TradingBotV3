"""B7 Theta measured: an underlying-only outcome for every theta pick.

Most recorded theta picks carry no strike (recorded before the quote pass until
2026-09-22, or a name with no weekly options). They can never get an option
grade. This measures what the underlying did instead - did the close stay at or
above a reference level - and says plainly it is not option P&L. The option
grade (`held_*`, `status`) is untouched.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import market_calendar  # noqa: E402
from test_ws_th_theta_tracker import (  # noqa: E402
    SESSION_5,
    SESSION_10,
    SESSION_20,
    _bars,
    _closes_for,
    _put_row,
    _record_and_read,
    _sma_row,
)


def _grade(tmp_path, picks, bars, as_of=SESSION_20):
    import theta_pick_tracker

    return theta_pick_tracker.grade_theta_picks(
        picks,
        closes_for=_closes_for(bars),
        calendar=market_calendar,
        as_of=as_of,
        path=tmp_path / "theta_outcomes.csv",
    )


def _never_quoted(tmp_path):
    return _record_and_read(
        tmp_path, [_put_row("DDD", option=None, unavailable_reason="no_weekly_options")]
    )


def test_a_never_quoted_pick_is_measured_against_its_lowest_held_support(tmp_path):
    picks = _never_quoted(tmp_path)
    assert picks[0]["strike"] is None
    # The lowest level the pick's support set held on at the scan.
    lowest = min(s["level"] for s in picks[0]["supports"] if s["held"])
    row = _grade(tmp_path, picks, _bars({SESSION_10: {"close": lowest - 0.5}}))[0]
    assert row["ref_basis"] == "lowest_held_support"
    assert row["ref_level"] == pytest.approx(lowest)
    assert row["ref_held_5"] is True
    assert row["ref_held_10"] is False
    assert row["ref_held_20"] is True
    assert row["ref_status"] == "measured"
    # The option grade is untouched: no strike, no option outcome.
    assert row["held_20"] is None
    assert row["status"] == "unmeasured"


def test_an_immature_underlying_mark_is_pending_not_a_break(tmp_path):
    picks = _never_quoted(tmp_path)
    row = _grade(tmp_path, picks, _bars(), as_of=SESSION_5)[0]
    assert row["ref_held_5"] is True
    assert row["ref_held_20"] is None
    assert row["ref_status"] == "pending"


def test_a_pick_with_no_held_support_and_no_strike_stays_unmeasured(tmp_path):
    picks = _never_quoted(tmp_path)
    for entry in picks[0]["supports"]:
        entry["held"] = False
    row = _grade(tmp_path, picks, _bars())[0]
    assert row["ref_basis"] == "none"
    assert row["ref_held_20"] is None
    assert row["ref_status"] == "unmeasured"


def test_a_strike_pick_uses_its_strike_and_says_whether_it_was_quoted(tmp_path):
    picks = _record_and_read(tmp_path, [_sma_row()])
    row = _grade(tmp_path, picks, _bars({SESSION_20: {"close": 94.0}}))[0]
    assert row["ref_basis"] == "strike_quoted"
    assert row["ref_level"] == pytest.approx(95.0)
    assert row["ref_held_20"] is False
    assert row["held_20"] is False
    assert row["ref_status"] == "measured"

    unquoted = dict(picks[0], premium=None)
    row = _grade(tmp_path, [unquoted], _bars())[0]
    assert row["ref_basis"] == "strike_unquoted"


def test_the_new_columns_are_appended_after_the_old_ones(tmp_path):
    import theta_pick_tracker

    columns = tuple(theta_pick_tracker.THETA_OUTCOME_COLUMNS)
    assert columns[columns.index("rs_note") + 1 :] == (
        "ref_basis",
        "ref_level",
        "ref_held_5",
        "ref_held_10",
        "ref_held_20",
        "ref_status",
    )
    _grade(tmp_path, _never_quoted(tmp_path), _bars())
    with (tmp_path / "theta_outcomes.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["ref_basis"] == "lowest_held_support"
    assert row["ref_held_20"] == "True"


def _csv_rows(tmp_path, picks, bars, as_of=SESSION_20):
    _grade(tmp_path, picks, bars, as_of=as_of)
    with (tmp_path / "theta_outcomes.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_the_outcome_counts_and_line_say_underlying_only_and_n_measured(tmp_path):
    import theta_pick_tracker

    picks = _record_and_read(
        tmp_path,
        [
            _put_row("DDD", option=None, unavailable_reason="no_weekly_options"),
            _sma_row("BBB"),
        ],
    )
    rows = _csv_rows(tmp_path, picks, _bars())
    counts = theta_pick_tracker.theta_outcome_counts(rows)
    assert counts["first_appearances"] == 2
    assert counts["marks"][5] == {"n": 2, "held": 2}
    assert counts["marks"][20] == {"n": 2, "held": 2}
    assert counts["basis"] == {"lowest_held_support": 1, "strike_quoted": 1}
    line = theta_pick_tracker.theta_outcome_line(counts)
    assert "underlying only" in line
    assert "not option P&L" in line
    assert "2 measured" in line
    assert "no option quote" in line


def test_a_repeat_appearance_is_not_counted_twice():
    import theta_pick_tracker

    base = {
        "symbol": "A",
        "first_seen_scan_date": "2026-06-01",
        "ref_basis": "lowest_held_support",
        "ref_held_5": "True",
        "ref_status": "pending",
    }
    rows = [dict(base, scan_date="2026-06-01"), dict(base, scan_date="2026-06-02")]
    counts = theta_pick_tracker.theta_outcome_counts(rows)
    assert counts["first_appearances"] == 1
    assert counts["marks"][5] == {"n": 1, "held": 1}
    assert counts["pending"] == 1


def test_the_line_before_any_export_says_so():
    import theta_pick_tracker

    line = theta_pick_tracker.theta_outcome_line(theta_pick_tracker.theta_outcome_counts([]))
    assert "no theta pick measured yet" in line.lower()


def _qt_app():
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:  # pragma: no cover
        return None
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def test_the_theta_tab_shows_the_outcome_line_from_the_worker(monkeypatch, tmp_path):
    if _qt_app() is None:
        pytest.skip("PySide6 is not installed")
    from ui.panels import setup_tracker_panel

    from tests.conftest import refresh_setup_tracker

    picks = _never_quoted(tmp_path)
    _grade(tmp_path, picks, _bars())
    setup_tracker_panel.clear_setup_tracker_csv_cache()
    monkeypatch.setattr(
        setup_tracker_panel,
        "MASTER_AVWAP_THETA_OUTCOMES_FILE",
        tmp_path / "theta_outcomes.csv",
        raising=False,
    )
    panel = setup_tracker_panel.SetupTrackerPanel()
    refresh_setup_tracker(panel)
    try:
        text = panel.theta_outcome_label.text()
        assert "1 measured" in text
        assert "underlying only" in text
    finally:
        panel.deleteLater()


def test_the_night_slot_reports_the_underlying_only_count(tmp_path):
    from datetime import datetime

    from ai_jobs.theta_grading import run_theta_pick_grading
    from test_ws_th_theta_grading_slot import _pick_row, _write_daily_parquet, _write_store

    never_quoted = dict(_pick_row(), strike=None, short_strike=None, premium=None, expiry=None)
    picks = tmp_path / "theta_picks.jsonl"
    _write_store(picks, [never_quoted])
    _write_daily_parquet(tmp_path / "bars", "BBB", through=SESSION_20)
    result = run_theta_pick_grading(
        picks_path=picks,
        outcomes_path=tmp_path / "out.csv",
        daily_bars_dir=tmp_path / "bars",
        now=datetime(2026, 7, 1, 2, 0, tzinfo=market_calendar.MARKET_TZ),
    )
    assert result["measured"] == 0
    assert result["ref_measured"] == 1
    assert "1 measured underlying-only" in result["reason"]
