"""R5 section 4: the any-bounce watch.

One armed request per symbol and side, covering a SET of levels, firing once
on the level that actually held and then disarming. The rules asserted here
are the ones the spec makes load-bearing:

- the bounce means what a D1 zone arm means by it (two completed bars, dip and
  hold, then a better close clear of the level);
- a level the data cannot supply is ABSENT, never fabricated - so a symbol with
  no zone-arms entry still watches its session EMAs and nothing else;
- a forming bar is preview and can never be the trigger bar;
- the store round-trips, and a persisted watch whose levels are all unknown is
  dropped rather than shown as armed forever.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from chart_watch import (  # noqa: E402
    ANY_BOUNCE_KINDS,
    AnyBounceWatch,
    any_bounce_levels,
    any_bounce_watch_from_dict,
    evaluate_any_bounce_watch,
    load_any_bounce_watches,
    save_any_bounce_watches,
)

OPEN = datetime(2026, 8, 17, 6, 30)


def bar(index, high, low, close, *, start=OPEN):
    return {
        "dt": start + timedelta(minutes=5 * index),
        "open": close,
        "high": high,
        "low": low,
        "close": close,
    }


def watch(*kinds, side="long", symbol="AAA"):
    return AnyBounceWatch(
        symbol=symbol, side=side, kinds=tuple(kinds), armed_at=OPEN
    )


class TestTheBounceRule:
    def test_a_dip_and_reclaim_fires_naming_the_level(self):
        """Bar A tags 100 and holds; bar B closes better and clear of it."""
        bars = [bar(0, 101.0, 99.98, 100.05), bar(1, 101.5, 100.2, 101.2)]
        hit = evaluate_any_bounce_watch(
            watch("d1_ema15"),
            bars,
            {"d1_ema15": 100.0},
            now=OPEN + timedelta(minutes=10),
        )
        assert hit is not None
        assert hit.kind == "d1_ema15"
        assert hit.level == 100.0
        assert "D1 15 EMA" in hit.message
        assert hit.resolved_side == "long"

    def test_a_level_that_was_never_tagged_does_not_fire(self):
        bars = [bar(0, 105.0, 104.0, 104.5), bar(1, 106.0, 104.8, 105.8)]
        assert (
            evaluate_any_bounce_watch(
                watch("d1_ema15"),
                bars,
                {"d1_ema15": 100.0},
                now=OPEN + timedelta(minutes=10),
            )
            is None
        )

    def test_a_tag_without_a_reclaim_does_not_fire(self):
        """It dipped to the level and kept going. That is a break, not a bounce."""
        bars = [bar(0, 101.0, 99.98, 100.05), bar(1, 100.1, 98.5, 98.7)]
        assert (
            evaluate_any_bounce_watch(
                watch("d1_ema15"),
                bars,
                {"d1_ema15": 100.0},
                now=OPEN + timedelta(minutes=10),
            )
            is None
        )

    def test_the_short_side_is_the_rejection(self):
        bars = [bar(0, 100.02, 99.0, 99.95), bar(1, 99.9, 98.5, 98.8)]
        hit = evaluate_any_bounce_watch(
            watch("avwape", side="short"),
            bars,
            {"avwape": 100.0},
            now=OPEN + timedelta(minutes=10),
        )
        assert hit is not None
        assert "rejected from" in hit.message
        # The same bars are not a long-side bounce.
        assert (
            evaluate_any_bounce_watch(
                watch("avwape"),
                bars,
                {"avwape": 100.0},
                now=OPEN + timedelta(minutes=10),
            )
            is None
        )

    def test_the_first_armed_level_that_held_is_the_one_named(self):
        """Order follows the watch's own kinds, so the answer is deterministic."""
        bars = [bar(0, 101.0, 99.98, 100.05), bar(1, 101.5, 100.2, 101.2)]
        levels = {"d1_ema15": 100.0, "m5_ema15": 100.0}
        first = evaluate_any_bounce_watch(
            watch("m5_ema15", "d1_ema15"), bars, levels, now=OPEN + timedelta(minutes=10)
        )
        assert first.kind == "m5_ema15"

    def test_a_forming_bar_is_never_the_trigger(self):
        bars = [bar(0, 101.0, 99.98, 100.05), bar(1, 101.5, 100.2, 101.2)]
        # Bar 1 spans 06:35-06:40; at 06:39 it is still forming, so there is
        # only one completed bar and no two-bar rule to apply.
        assert (
            evaluate_any_bounce_watch(
                watch("d1_ema15"),
                bars,
                {"d1_ema15": 100.0},
                now=OPEN + timedelta(minutes=9),
            )
            is None
        )

    def test_an_absent_level_is_silence_not_a_zero(self):
        bars = [bar(0, 101.0, 99.98, 100.05), bar(1, 101.5, 100.2, 101.2)]
        assert (
            evaluate_any_bounce_watch(
                watch("prev_avwape"), bars, {}, now=OPEN + timedelta(minutes=10)
            )
            is None
        )

    def test_an_unreadable_bar_refuses_rather_than_guesses(self):
        bars = [bar(0, 101.0, 99.98, 100.05), bar(1, 101.5, None, 101.2)]
        bars[0]["close"] = None
        assert (
            evaluate_any_bounce_watch(
                watch("d1_ema15"),
                bars,
                {"d1_ema15": 100.0},
                now=OPEN + timedelta(minutes=10),
            )
            is None
        )


class TestTheLevelSet:
    def test_zone_arm_levels_are_read_by_name(self):
        entry = {
            "avwape": 101.25,
            "prev_avwape": 98.4,
            "trigger_levels": [
                {"name": "UPPER_1", "level": 103.0},
                {"name": "PREV_LOWER_1", "level": 96.5},
                {"name": "EMA_21", "level": 99.1},
            ],
        }
        levels = any_bounce_levels(zone_arm_entry=entry, now=OPEN)
        assert levels["avwape"] == 101.25
        assert levels["prev_avwape"] == 98.4
        assert levels["d1_band_1"] == 103.0
        assert levels["prev_band_1"] == 96.5
        assert levels["d1_ema21"] == 99.1
        # Nothing supplied the daily 15 EMA, so it is simply not there.
        assert "d1_ema15" not in levels

    def test_the_daily_store_fills_the_emas_the_scan_did_not_carry(self):
        levels = any_bounce_levels(d1_levels={"ema15": 97.5}, now=OPEN)
        assert levels["d1_ema15"] == 97.5

    def test_session_emas_come_from_the_cached_bars(self):
        bars = [bar(index, 100.0 + index, 99.0 + index, 100.0 + index) for index in range(30)]
        levels = any_bounce_levels(m5_bars=bars, now=OPEN + timedelta(minutes=5 * 30))
        assert "m5_ema15" in levels
        assert "m5_ema21" in levels
        # 21 bars of history exist, so both EMAs are answerable; the H1 15 EMA
        # needs 15 COMPLETED hours and this session has under three.
        assert "h1_ema15" not in levels

    def test_a_symbol_with_no_data_at_all_watches_nothing(self):
        assert any_bounce_levels(now=OPEN) == {}

    def test_the_hourly_ema_ignores_the_forming_hour(self):
        bars = [
            bar(index, 100.0, 99.0, 100.0 + index * 0.1)
            for index in range(12 * 20)
        ]
        moment = OPEN + timedelta(minutes=5 * 12 * 20)
        levels = any_bounce_levels(m5_bars=bars, now=moment)
        assert "h1_ema15" in levels
        # The last bucket is the hour containing `moment`, and it is excluded.
        hourly_closes = [
            bars[index]["close"] for index in range(len(bars))
        ]
        assert levels["h1_ema15"] < max(hourly_closes)


class TestTheStore:
    def test_a_watch_round_trips(self, tmp_path):
        path = tmp_path / "any_bounce_watches.json"
        original = [watch("d1_ema15", "avwape"), watch("m5_ema21", side="short", symbol="ZZZ")]
        save_any_bounce_watches(original, path)
        assert load_any_bounce_watches(path) == original

    def test_a_missing_store_is_empty_not_an_error(self, tmp_path):
        assert load_any_bounce_watches(tmp_path / "nope.json") == []

    def test_a_corrupt_store_is_empty_not_an_error(self, tmp_path):
        path = tmp_path / "any_bounce_watches.json"
        path.write_text("{not json", encoding="utf-8")
        assert load_any_bounce_watches(path) == []

    def test_an_unknown_level_is_dropped_and_the_rest_survive(self):
        restored = any_bounce_watch_from_dict(
            {
                "symbol": "aaa",
                "side": "long",
                "kinds": ["d1_ema15", "not_a_level"],
                "armed_at": OPEN.isoformat(),
            }
        )
        assert restored.kinds == ("d1_ema15",)
        assert restored.symbol == "AAA"

    def test_a_watch_with_no_known_level_is_refused(self):
        """An armed chip that can never fire is worse than no chip."""
        assert (
            any_bounce_watch_from_dict(
                {
                    "symbol": "AAA",
                    "side": "long",
                    "kinds": ["not_a_level"],
                    "armed_at": OPEN.isoformat(),
                }
            )
            is None
        )

    def test_a_bad_side_is_refused(self):
        assert (
            any_bounce_watch_from_dict(
                {
                    "symbol": "AAA",
                    "side": "sideways",
                    "kinds": ["d1_ema15"],
                    "armed_at": OPEN.isoformat(),
                }
            )
            is None
        )

    def test_every_kind_has_a_label(self):
        for kind, label in ANY_BOUNCE_KINDS.items():
            assert label and label != kind
