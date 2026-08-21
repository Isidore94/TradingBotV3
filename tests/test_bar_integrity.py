"""A malformed bar may not decide the scale, and may not pass for a candle.

The defect this covers is specific: a chart takes its y-range from bar lows
and highs, but draws the body from opens and closes. While
``low <= open, close <= high`` holds those are the same numbers. When it does
not, the body is drawn from a coordinate the range never saw - so one corrupt
row paints a solid column over an entire session while the axis still reads
normally (trader, 2026-08-21, GFS M5).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ui import bar_integrity  # noqa: E402


def _bar(minute: int, open_: float, high: float, low: float, close: float, volume=1000.0):
    return {
        "dt": datetime(2026, 8, 21, 6, 30) + timedelta(minutes=5 * minute),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def _healthy(count: int = 12) -> list[dict]:
    return [_bar(i, 47.5, 47.6, 47.4, 47.55) for i in range(count)]


# -- the judgement -------------------------------------------------------


def test_a_well_formed_bar_has_no_defect():
    assert bar_integrity.bar_defect(_bar(0, 47.5, 47.6, 47.4, 47.55)) is None


def test_open_and_close_may_sit_exactly_on_the_extremes():
    """A marubozu is not a defect. The bound is inclusive on both ends."""
    assert bar_integrity.bar_defect(_bar(0, 47.4, 47.6, 47.4, 47.6)) is None


def test_a_zero_open_is_named_not_tolerated():
    defect = bar_integrity.bar_defect(_bar(0, 0.0, 47.6, 47.4, 47.55))
    assert defect == bar_integrity.DEFECT_OPEN_OUTSIDE


def test_a_close_from_another_scale_is_named():
    defect = bar_integrity.bar_defect(_bar(0, 47.5, 47.6, 47.4, 4755.0))
    assert defect == bar_integrity.DEFECT_CLOSE_OUTSIDE


def test_nan_is_a_defect_even_though_it_fails_every_compare():
    """NaN would slip through an ordering check and reach the painter."""
    defect = bar_integrity.bar_defect(_bar(0, float("nan"), 47.6, 47.4, 47.55))
    assert defect == bar_integrity.DEFECT_NOT_FINITE


def test_an_inverted_range_is_a_defect():
    defect = bar_integrity.bar_defect(_bar(0, 47.5, 47.0, 47.9, 47.55))
    assert defect == bar_integrity.DEFECT_RANGE_INVERTED


def test_a_missing_price_is_a_defect_not_a_zero():
    bar = _bar(0, 47.5, 47.6, 47.4, 47.55)
    del bar["close"]
    assert bar_integrity.bar_defect(bar) == bar_integrity.DEFECT_NOT_NUMERIC


def test_only_a_salvageable_range_is_drawable():
    salvageable = bar_integrity.scan_bars([_bar(0, 0.0, 47.6, 47.4, 47.55)])[0]
    hopeless = bar_integrity.scan_bars([_bar(0, 47.5, float("inf"), 47.4, 47.55)])[0]
    assert salvageable.drawable is True
    assert hopeless.drawable is False


# -- the scale -----------------------------------------------------------


def test_the_bad_bar_does_not_move_the_price_range():
    """The failing case: one row with a zero open next to a normal session."""
    bars = _healthy() + [_bar(12, 0.0, 47.62, 47.40, 48.62)]
    low, high = bar_integrity.price_range(bars)
    assert low == pytest.approx(47.4)
    assert high == pytest.approx(47.6)


def test_a_series_of_only_bad_bars_still_yields_a_range_from_what_held():
    """Better a scale from salvaged lows/highs than no chart at all."""
    bars = [_bar(0, 0.0, 47.6, 47.4, 48.62), _bar(1, 0.0, 47.8, 47.3, 48.9)]
    assert bar_integrity.price_range(bars) == (pytest.approx(47.3), pytest.approx(47.8))


def test_nothing_usable_yields_no_range_rather_than_an_invented_one():
    bars = [_bar(0, 1.0, float("nan"), float("nan"), 1.0)]
    assert bar_integrity.price_range(bars) is None


def test_the_body_of_a_bad_bar_is_clamped_into_its_own_range():
    bottom, top = bar_integrity.clamped_body(_bar(0, 0.0, 47.6, 47.4, 48.62))
    assert (bottom, top) == (pytest.approx(47.4), pytest.approx(47.6))


# -- the diagnostic ------------------------------------------------------


def test_defects_are_logged_once_each_with_their_provenance(tmp_path):
    path = tmp_path / "bad_bars.jsonl"
    bars = _healthy() + [_bar(12, 0.0, 47.62, 47.40, 48.62)]
    written = bar_integrity.log_defects(
        "GFS", "M5", bars, source="latest_bars[GFS]", path=path
    )
    assert written == 1
    # The same cached series is rebuilt on every refresh; the second thousand
    # copies of one row teach nothing the first did not.
    assert bar_integrity.log_defects(
        "GFS", "M5", bars, source="latest_bars[GFS]", path=path
    ) == 0

    import json

    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["symbol"] == "GFS"
    assert row["timeframe"] == "M5"
    assert row["defect"] == bar_integrity.DEFECT_OPEN_OUTSIDE
    assert row["source"] == "latest_bars[GFS]"
    assert row["open"] == 0.0 and row["close"] == pytest.approx(48.62)


def test_a_healthy_series_writes_nothing_at_all(tmp_path):
    path = tmp_path / "bad_bars.jsonl"
    assert bar_integrity.log_defects("GFS", "M5", _healthy(), path=path) == 0
    assert not path.exists()


def test_a_diagnostics_failure_never_breaks_the_caller(tmp_path):
    """A directory where the file should be: unwritable, and harmless."""
    path = tmp_path / "bad_bars.jsonl"
    path.mkdir()
    bars = [_bar(0, 0.0, 47.6, 47.4, 48.62)]
    assert bar_integrity.log_defects("ZZZZ", "M5", bars, path=path) == 0


# -- the advisory observation --------------------------------------------


def test_an_aggregate_looking_row_is_observed():
    """A bar that summarises the whole series is what a daily row dropped
    into an M5 cache looks like from the outside."""
    bars = _healthy(20) + [_bar(20, 47.10, 48.62, 47.02, 48.50)]
    found = bar_integrity.range_outliers(bars)
    assert [d.index for d in found] == [20]
    assert found[0].defect == bar_integrity.DEFECT_RANGE_OUTLIER


def test_an_ordinary_wide_bar_is_not_observed():
    """A violent bar on a volatile session must not read as corruption."""
    bars = [_bar(i, 47.5, 47.9, 47.1, 47.55) for i in range(20)]
    bars.append(_bar(20, 47.5, 48.2, 47.0, 48.0))
    assert bar_integrity.range_outliers(bars) == []


def test_a_short_series_is_never_judged():
    """Three bars have no median worth trusting."""
    bars = [_bar(0, 47.5, 47.6, 47.4, 47.55), _bar(1, 47.1, 48.6, 47.0, 48.5)]
    assert bar_integrity.range_outliers(bars) == []


def test_an_observation_never_changes_what_is_drawn():
    """scan_bars is what the renderer asks; outliers are not defects."""
    bars = _healthy(20) + [_bar(20, 47.10, 48.62, 47.02, 48.50)]
    assert bar_integrity.scan_bars(bars) == []
    assert bar_integrity.price_range(bars) == (pytest.approx(47.02), pytest.approx(48.62))
