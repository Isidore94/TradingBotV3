"""The with-trend regime-pause rule is exactly two cases and nothing else."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from regime_pause_focus import day_bias, focus_side_for  # noqa: E402


@pytest.mark.parametrize(
    "env, expected",
    [
        ("bullish_weak", "bullish"),
        ("bullish_strong", "bullish"),
        ("BULLISH_STRONG", "bullish"),
        ("bearish_weak", "bearish"),
        ("bearish_strong", "bearish"),
        ("neutral", ""),
        ("", ""),
        (None, ""),
        ("bull", ""),
    ],
)
def test_day_bias_collapses_strength_and_rejects_the_rest(env, expected):
    assert day_bias(env) == expected


def test_a_long_holding_highs_on_a_bullish_day_joins_focus_longs():
    assert focus_side_for("bullish_weak", "LONG") == "long"
    assert focus_side_for("bullish_strong", "long") == "long"


def test_a_short_pressing_lows_on_a_bearish_day_joins_focus_shorts():
    assert focus_side_for("bearish_weak", "SHORT") == "short"
    assert focus_side_for("bearish_strong", "short") == "short"


@pytest.mark.parametrize(
    "env, side",
    [
        ("bullish_weak", "SHORT"),  # counter-trend: still a chart to look at
        ("bearish_strong", "LONG"),
        ("neutral", "LONG"),  # no directional read admits nothing
        ("", "LONG"),
        (None, "SHORT"),
        ("bullish_weak", "WATCH"),  # a sideless row never joins a side
        ("bullish_weak", ""),
    ],
)
def test_everything_else_stays_on_the_review_queue(env, side):
    assert focus_side_for(env, side) is None
