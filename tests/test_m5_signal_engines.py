"""R5 section 3 engine-layer tests: bars in, events out.

These are the rules the call site has historically got wrong, so they are
asserted here where they are cheap: completed bars only, a warm indicator with
a session-scoped event, and a short side that is a true mirror rather than an
inverted test.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from m5_signal_engines import (  # noqa: E402
    LrsiCrossEvent,
    latest_lrsi_cross,
    lrsi_cross_events,
)

OPEN = datetime(2026, 8, 17, 6, 30)

#: Eight bars of churn (the EMA9 saws, so efficiency sits at/near zero) then a
#: clean one-way run. The first fully efficient window lands on index 9, which
#: is what makes this series useful: the crossing has a known home.
CHURN_THEN_RUN = [
    100.0, 99.5, 100.2, 99.6, 100.1, 99.4, 100.0, 99.5,
    100.3, 100.9, 101.6, 102.4, 103.3, 104.3, 105.4,
]


def bars_from(closes, *, start=OPEN, minutes=5):
    return [
        {
            "dt": start + timedelta(minutes=minutes * index),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
        }
        for index, close in enumerate(closes)
    ]


def after(bar_count, *, start=OPEN, minutes=5):
    """A clock at which exactly ``bar_count`` bars have completed."""
    return start + timedelta(minutes=minutes * bar_count)


class TestLrsiCrossEvents:
    def test_a_run_out_of_churn_crosses_both_levels_on_one_bar(self):
        bars = bars_from(CHURN_THEN_RUN)
        events = lrsi_cross_events(
            bars, symbol="test", side="long", now=after(len(CHURN_THEN_RUN))
        )
        assert [(event.bar_index, event.level) for event in events] == [
            (9, 20.0),
            (9, 50.0),
        ]
        # Symbols are normalised at the boundary, not by every caller.
        assert {event.symbol for event in events} == {"TEST"}
        assert events[0].previous == pytest.approx(0.0)
        assert events[0].value == pytest.approx(65.0593, abs=1e-3)

    def test_the_lower_level_is_the_stronger_tell(self):
        bars = bars_from(CHURN_THEN_RUN)
        events = lrsi_cross_events(
            bars, symbol="T", side="long", now=after(len(CHURN_THEN_RUN))
        )
        strongest = [event for event in events if event.is_strongest]
        assert [event.level for event in strongest] == [20.0]

    def test_a_forming_bar_never_produces_an_event(self):
        """The crossing bar exists but has not closed. Preview, not signal."""
        bars = bars_from(CHURN_THEN_RUN)
        # Index 9 spans 07:15-07:20; at 07:19 it is still forming.
        forming = OPEN + timedelta(minutes=5 * 9 + 4)
        assert lrsi_cross_events(bars, symbol="T", side="long", now=forming) == ()
        # One minute later the same bar is complete and the event appears.
        complete = OPEN + timedelta(minutes=5 * 10)
        assert lrsi_cross_events(bars, symbol="T", side="long", now=complete)

    def test_too_little_history_is_silence_not_a_guess(self):
        assert lrsi_cross_events([], symbol="T", side="long", now=after(0)) == ()
        one = bars_from([100.0])
        assert lrsi_cross_events(one, symbol="T", side="long", now=after(1)) == ()

    def test_an_unreadable_close_refuses_the_whole_series(self):
        """A placeholder would corrupt the EMA for every later bar."""
        closes = list(CHURN_THEN_RUN)
        bars = bars_from(closes)
        bars[3]["close"] = None
        assert lrsi_cross_events(bars, symbol="T", side="long", now=after(len(closes))) == ()

    def test_events_are_scoped_to_a_session_but_the_indicator_is_not(self):
        """Yesterday warms the EMA; yesterday's crossing is not today's alert."""
        yesterday = OPEN - timedelta(days=1)
        bars = bars_from(CHURN_THEN_RUN, start=yesterday)
        # Today continues the same efficient run, so the oscillator stays high
        # and produces no NEW crossing today.
        bars += bars_from([106.5, 107.7, 109.0], start=OPEN)
        now = OPEN + timedelta(minutes=5 * 3)

        assert lrsi_cross_events(bars, symbol="T", side="long", now=now) == ()

        yesterdays = lrsi_cross_events(
            bars, symbol="T", side="long", now=now, session=yesterday.date()
        )
        assert [event.level for event in yesterdays] == [20.0, 50.0]


class TestShortsAreAMirror:
    def test_negated_prices_give_identical_events_on_the_short_side(self):
        """The oscillator clamps at zero, so shorts cannot be an inverted test."""
        longs = lrsi_cross_events(
            bars_from(CHURN_THEN_RUN),
            symbol="T",
            side="long",
            now=after(len(CHURN_THEN_RUN)),
        )
        mirrored = [200.0 - close for close in CHURN_THEN_RUN]
        shorts = lrsi_cross_events(
            bars_from(mirrored),
            symbol="T",
            side="short",
            now=after(len(mirrored)),
        )

        assert [event.level for event in shorts] == [event.level for event in longs]
        assert [event.bar_index for event in shorts] == [event.bar_index for event in longs]
        for short_event, long_event in zip(shorts, longs):
            assert short_event.value == pytest.approx(long_event.value)
        assert {event.side for event in shorts} == {"short"}

    def test_a_falling_name_produces_no_long_side_crossing(self):
        mirrored = [200.0 - close for close in CHURN_THEN_RUN]
        assert (
            lrsi_cross_events(
                bars_from(mirrored), symbol="T", side="long", now=after(len(mirrored))
            )
            == ()
        )

    def test_an_unknown_side_is_treated_as_a_long(self):
        events = lrsi_cross_events(
            bars_from(CHURN_THEN_RUN), symbol="T", side="", now=after(len(CHURN_THEN_RUN))
        )
        assert {event.side for event in events} == {"long"}


class TestLatestLrsiCross:
    def test_it_fires_only_on_the_most_recently_completed_bar(self):
        bars = bars_from(CHURN_THEN_RUN)
        event = latest_lrsi_cross(bars, symbol="T", side="long", now=after(10))
        assert isinstance(event, LrsiCrossEvent)
        assert event.bar_index == 9

    def test_an_older_crossing_is_not_re_emitted_every_cycle(self):
        """R4 section 6.3: one event is one alert, not one alert per scan."""
        bars = bars_from(CHURN_THEN_RUN)
        for completed in range(11, len(CHURN_THEN_RUN) + 1):
            assert latest_lrsi_cross(bars, symbol="T", side="long", now=after(completed)) is None

    def test_when_one_bar_crosses_both_levels_the_stronger_one_wins(self):
        bars = bars_from(CHURN_THEN_RUN)
        event = latest_lrsi_cross(bars, symbol="T", side="long", now=after(10))
        assert event.level == 20.0
        assert event.is_strongest

    def test_no_crossing_is_none(self):
        flat = bars_from([100.0] * 12)
        assert latest_lrsi_cross(flat, symbol="T", side="long", now=after(12)) is None
