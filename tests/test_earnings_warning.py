"""S10b: earnings warning on SHORT setups. Annotate only - never hide, sort or mute."""

from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import earnings_warning as ew  # noqa: E402

STAT = ew.ShortEarningsStat(spy_rel_pct=-3.456, lookback_days=60, observations=525)

_FIELDS = (
    "lookback_days", "horizon_sessions", "side", "factor_key", "value_label",
    "observation_count", "avg_spy_relative_side_return_pct",
)


def _row(bucket, value, count, *, side="SHORT", horizon="5", factor="days_to_next_earnings"):
    return {
        "lookback_days": "60", "horizon_sessions": horizon, "side": side,
        "factor_key": factor, "value_label": bucket, "observation_count": str(count),
        "avg_spy_relative_side_return_pct": str(value),
    }


def _write_board(path: Path, rows) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture(autouse=True)
def _clean_cache():
    ew.reset_cache_for_tests()
    yield
    ew.reset_cache_for_tests()


@pytest.mark.parametrize("days", [0, 1, 7, 14])
def test_short_inside_window_is_warned_with_the_stat(days):
    text = ew.short_into_earnings(days, "SHORT", STAT)
    assert text == (
        f"earnings in {days} d - shorts 3-14 d before earnings: -3.5% vs SPY (60 d)"
    )


@pytest.mark.parametrize("days", [15, 30, -1])
def test_outside_window_is_silent(days):
    assert ew.short_into_earnings(days, "SHORT", STAT) == ""
    assert ew.badge_text(days, "SHORT") == ""


@pytest.mark.parametrize("days", [None, "", "nan", float("nan"), "soon"])
def test_unknown_date_is_silent(days):
    assert ew.short_into_earnings(days, "SHORT", STAT) == ""


@pytest.mark.parametrize("side", ["LONG", "long", "", None])
def test_long_side_is_silent(side):
    assert ew.short_into_earnings(5, side, STAT) == ""
    assert ew.badge_text(5, side) == ""


def test_lowercase_short_counts():
    assert ew.badge_text(3, "short") == "ER 3d"


def test_no_stat_gives_plain_wording():
    assert ew.short_into_earnings(5, "SHORT", None) == "earnings in 5 d - short into earnings"


def test_stat_is_observation_weighted_over_both_buckets():
    stat = ew.stat_from_rows([
        _row("3 to < 7", -2.0, 100),
        _row("7 to < 14", -5.0, 300),
        _row("0 to < 3", -99.0, 1000),
        _row("3 to < 7", 50.0, 100, side="LONG"),
        _row("3 to < 7", 50.0, 100, horizon="10"),
    ])
    assert stat is not None
    assert stat.spy_rel_pct == pytest.approx(-4.25)
    assert stat.observations == 400
    assert stat.lookback_days == 60


def test_leaderboard_row_absent_gives_plain_wording(tmp_path):
    board = _write_board(tmp_path / "board.csv", [_row("3 to < 7", -2.0, 100)])
    history = tmp_path / "history.json"
    history.write_text(json.dumps({"symbols": {}}), encoding="utf-8")
    ew.warm_cache(board, history)
    assert ew.cached_stat() is None
    assert ew.short_into_earnings(4, "SHORT", ew.cached_stat()) == (
        "earnings in 4 d - short into earnings"
    )


def test_missing_leaderboard_file_gives_no_stat(tmp_path):
    assert ew.read_stat(tmp_path / "absent.csv") is None


def test_warm_cache_reads_board_and_history(tmp_path):
    board = _write_board(
        tmp_path / "board.csv", [_row("3 to < 7", -2.0, 100), _row("7 to < 14", -5.0, 300)]
    )
    history = tmp_path / "history.json"
    history.write_text(json.dumps({"symbols": {
        "MU": {"events": [{"earnings_date": "2026-06-25"}, {"earnings_date": "2026-09-30"}]},
        "BRK-B": {"events": [{"earnings_date": "2026-10-10"}]},
        "OLD": {"events": [{"earnings_date": "2020-01-01"}]},
    }}), encoding="utf-8")
    assert ew.warm_cache(board, history) is True
    today = date(2026, 9, 26)
    assert ew.cached_days_to_next_earnings("MU", today=today) == 4
    assert ew.cached_days_to_next_earnings("BRK.B", today=today) == 14
    assert ew.cached_days_to_next_earnings("OLD", today=today) is None
    assert ew.cached_days_to_next_earnings("NOPE", today=today) is None
    assert ew.warning_for_symbol("MU", "SHORT", today=today) == (
        "earnings in 4 d - shorts 3-14 d before earnings: -4.2% vs SPY (60 d)"
    )
    assert ew.warning_for_symbol("MU", "LONG", today=today) == ""
    assert ew.warning_for_symbol("NOPE", "SHORT", today=today) == ""
    # An unchanged pair of files is not re-read.
    assert ew.warm_cache(board, history) is False


def test_earnings_day_itself_counts_as_zero():
    assert ew.days_to_next([date(2026, 9, 26)], date(2026, 9, 26)) == 0
    assert ew.days_to_next([date(2026, 9, 25)], date(2026, 9, 26)) is None
