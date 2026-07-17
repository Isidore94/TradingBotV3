"""TC2000-style relative volume math."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from rvol import bar_rvol, same_slot_baseline, session_rvol, split_sessions  # noqa: E402


def _flat_sessions(count: int, bars: int, volume: float) -> list[list[float]]:
    return [[volume] * bars for _ in range(count)]


class TestSameSlotBaseline:
    def test_averages_the_same_bar_across_sessions(self):
        prior = [[100.0, 200.0], [300.0, 400.0], [500.0, 600.0], [100.0, 200.0], [500.0, 600.0]]
        assert same_slot_baseline(prior, 0, min_sessions=5) == pytest.approx(300.0)
        assert same_slot_baseline(prior, 1, min_sessions=5) == pytest.approx(400.0)

    def test_uses_only_the_most_recent_window(self):
        prior = [[1_000_000.0]] * 10 + [[100.0]] * 15
        assert same_slot_baseline(prior, 0, sessions=15) == pytest.approx(100.0)

    def test_short_sessions_still_contribute_their_bars(self):
        prior = _flat_sessions(4, 78, 100.0) + [[100.0] * 30]  # one half day
        assert same_slot_baseline(prior, 10, min_sessions=5) == pytest.approx(100.0)
        # bar 40 exists in only 4 sessions -> below the evidence floor of 5
        assert same_slot_baseline(prior, 40, min_sessions=5) is None

    def test_insufficient_history_returns_none(self):
        assert same_slot_baseline(_flat_sessions(3, 78, 100.0), 0, min_sessions=5) is None
        assert same_slot_baseline([], 0) is None


class TestBarRvol:
    def test_matches_the_tc2000_script(self):
        prior = _flat_sessions(15, 78, 200.0)
        today = [999.0, 999.0, 300.0]  # only the latest bar matters
        assert bar_rvol(today, prior) == pytest.approx(1.5)

    def test_none_without_baseline_or_bars(self):
        assert bar_rvol([], _flat_sessions(15, 78, 200.0)) is None
        assert bar_rvol([100.0], _flat_sessions(2, 78, 200.0)) is None


class TestSessionRvol:
    def test_flat_tape_reads_exactly_one(self):
        prior = _flat_sessions(15, 78, 100.0)
        assert session_rvol([100.0] * 12, prior) == pytest.approx(1.0)

    def test_elevated_volume_reads_above_one(self):
        prior = _flat_sessions(15, 78, 100.0)
        assert session_rvol([250.0] * 12, prior) == pytest.approx(2.5)

    def test_cumulative_smooths_one_quiet_bar(self):
        prior = _flat_sessions(15, 78, 100.0)
        today = [300.0] * 11 + [0.0]
        assert session_rvol(today, prior) == pytest.approx(3300.0 / 1200.0)

    def test_none_when_no_baseline(self):
        assert session_rvol([100.0], []) is None
        assert session_rvol([], _flat_sessions(15, 78, 100.0)) is None


class TestSplitSessions:
    def test_groups_in_order_by_key(self):
        pairs = [("d1", 1.0), ("d1", 2.0), ("d2", 3.0), ("d3", 4.0), ("d3", 5.0)]
        assert split_sessions(pairs) == [[1.0, 2.0], [3.0], [4.0, 5.0]]

    def test_bad_volume_becomes_zero(self):
        assert split_sessions([("d1", "x")]) == [[0.0]]
