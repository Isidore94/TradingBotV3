"""Phase 0.10 B-3 - the challenger's D1 overlay, default OFF.

Three things this file protects:

1. the group exists and is built on the WORKER, in the levels payload, beside
   the champion's lines - never on the paint path;
2. it is **off until the trader turns it on**, which the paint-lines preference
   file did not previously have a way to express (every group defaults ON there,
   deliberately, so a group added by a later version appears switched on rather
   than silently missing). A challenger under test is the opposite case;
3. nothing else in the levels payload moved - the existing families are
   byte-identical with the new group present.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import chart_levels  # noqa: E402
from ui.services.paint_lines_prefs import PaintLinesPrefs  # noqa: E402


def _bars(count: int = 40) -> list[dict]:
    from datetime import datetime

    out = []
    start = date(2026, 6, 1)
    for index in range(count):
        base = 100.0 + index * 0.5
        out.append(
            {
                "dt": datetime.combine(start + timedelta(days=index), datetime.min.time()),
                "open": base,
                "high": base + 1.0,
                "low": base - 1.0,
                "close": base + 0.25,
                "volume": 1_000_000.0 + index,
            }
        )
    return out


# ---------------------------------------------------------------------------
# The group itself.
# ---------------------------------------------------------------------------


def test_the_group_is_registered_and_named():
    assert chart_levels.GROUP_AVWAP_VARIANT == "avwap_variant"
    assert chart_levels.GROUP_NAMES[chart_levels.GROUP_AVWAP_VARIANT] == "AVWAP σ variant"
    # Last in the control's list: a challenger does not push the trader's own
    # lines around in a menu they read by position.
    assert chart_levels.LEVEL_GROUPS[-1][0] == chart_levels.GROUP_AVWAP_VARIANT


def test_the_six_bands_are_built_with_stable_ids():
    bars = _bars()
    levels = chart_levels.avwap_variant_levels(bars, anchor_index=25)
    assert len(levels) == 6
    assert all(level["group"] == chart_levels.GROUP_AVWAP_VARIANT for level in levels)
    assert all(level["family"] == "avwap_variant" for level in levels)
    ids = [level["id"] for level in levels]
    assert len(set(ids)) == 6
    # Stable: the same bars and anchor give the same ids again.
    assert [level["id"] for level in chart_levels.avwap_variant_levels(bars, 25)] == ids


def test_each_band_is_a_sloped_series_aligned_to_the_bars():
    bars = _bars()
    levels = chart_levels.avwap_variant_levels(bars, anchor_index=25)
    for level in levels:
        assert level["values"] is not None
        assert len(level["values"]) == len(bars)
        # Nothing before the anchor, everything from it on.
        assert all(value is None for value in level["values"][:25])
        assert level["values"][-1] is not None
        assert level["price"] == pytest.approx(level["values"][-1])


def test_a_bar_the_challenger_cannot_measure_is_none_not_zero():
    """Anchored at bar 0 the 20-close window is not full until bar 19."""
    bars = _bars()
    levels = chart_levels.avwap_variant_levels(bars, anchor_index=0)
    upper = next(level for level in levels if level["label"].startswith("+1"))
    assert all(value is None for value in upper["values"][:19])
    assert upper["values"][19] is not None


def test_too_few_bars_yields_no_group_at_all():
    assert chart_levels.avwap_variant_levels(_bars(5), anchor_index=0) == []
    assert chart_levels.avwap_variant_levels([], anchor_index=0) == []


# ---------------------------------------------------------------------------
# It rides the existing payload and changes nothing else in it.
# ---------------------------------------------------------------------------


def test_the_payload_carries_the_group_and_nothing_else_moved(tmp_path):
    bars = _bars()
    anchor = bars[25]["dt"].date()
    common = dict(
        store_records=[],
        trendline_feed={},
        price_alerts_path=tmp_path / "alerts.json",
        d1_level_watches_path=tmp_path / "watches.json",
    )
    without = chart_levels.build_d1_levels("TEST", bars, avwap_anchor=None, **common)
    with_variant = chart_levels.build_d1_levels("TEST", bars, avwap_anchor=anchor, **common)

    added = [
        level
        for level in with_variant
        if level["group"] == chart_levels.GROUP_AVWAP_VARIANT
    ]
    assert len(added) == 6
    others = [
        level
        for level in with_variant
        if level["group"] != chart_levels.GROUP_AVWAP_VARIANT
    ]
    assert others == without


def test_an_anchor_that_is_not_a_session_adds_nothing(tmp_path):
    bars = _bars()
    levels = chart_levels.build_d1_levels(
        "TEST",
        bars,
        avwap_anchor=date(2020, 1, 1),
        store_records=[],
        trendline_feed={},
        price_alerts_path=tmp_path / "alerts.json",
        d1_level_watches_path=tmp_path / "watches.json",
    )
    assert not [
        level for level in levels if level["group"] == chart_levels.GROUP_AVWAP_VARIANT
    ]


def test_the_group_is_hidden_by_the_normal_visibility_filter():
    bars = _bars()
    levels = chart_levels.avwap_variant_levels(bars, anchor_index=25)
    assert chart_levels.visible_levels(levels, [chart_levels.GROUP_AVWAP_VARIANT]) == []
    assert len(chart_levels.visible_levels(levels, [])) == 6


# ---------------------------------------------------------------------------
# Default OFF, and it stays off across a reload.
# ---------------------------------------------------------------------------


def test_the_variant_group_is_off_by_default(tmp_path):
    prefs = PaintLinesPrefs(tmp_path / "paint.json")
    assert prefs.is_visible(chart_levels.GROUP_AVWAP_VARIANT) is False
    assert chart_levels.GROUP_AVWAP_VARIANT in prefs.hidden_groups()
    # ...while every other group is still ON by default, unchanged.
    for group, _label in chart_levels.LEVEL_GROUPS:
        if group != chart_levels.GROUP_AVWAP_VARIANT:
            assert prefs.is_visible(group) is True


def test_turning_it_on_survives_a_reload(tmp_path):
    path = tmp_path / "paint.json"
    prefs = PaintLinesPrefs(path)
    prefs.set_visible(chart_levels.GROUP_AVWAP_VARIANT, True)
    assert PaintLinesPrefs(path).is_visible(chart_levels.GROUP_AVWAP_VARIANT) is True
    prefs.set_visible(chart_levels.GROUP_AVWAP_VARIANT, False)
    assert PaintLinesPrefs(path).is_visible(chart_levels.GROUP_AVWAP_VARIANT) is False


def test_a_preference_file_written_before_this_group_existed_keeps_it_off(tmp_path):
    path = tmp_path / "paint.json"
    path.write_text(json.dumps({"hidden_groups": ["prev_day"]}), encoding="utf-8")
    prefs = PaintLinesPrefs(path)
    assert prefs.is_visible("prev_day") is False
    assert prefs.is_visible(chart_levels.GROUP_AVWAP_VARIANT) is False
    # And the trader's own hidden group is not lost when the file is rewritten.
    prefs.set_visible(chart_levels.GROUP_AVWAP_VARIANT, True)
    reloaded = PaintLinesPrefs(path)
    assert reloaded.is_visible("prev_day") is False
    assert reloaded.is_visible(chart_levels.GROUP_AVWAP_VARIANT) is True
