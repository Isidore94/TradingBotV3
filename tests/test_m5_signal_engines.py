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
    ConfluenceEvent,
    LrsiCrossEvent,
    confluence_events,
    latest_confluence,
    latest_lrsi_cross,
    latest_orb_events,
    lrsi_cross_events,
    orb_events,
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


# ----------------------------------------------------------------------
# R5 section 3.2 -- the HA + SMI + LRSI confluence.
# ----------------------------------------------------------------------

def ohlc_bars(rows, *, start=OPEN, minutes=5):
    """Bars from explicit (open, high, low, close) rows."""
    return [
        {
            "dt": start + timedelta(minutes=minutes * index),
            "open": row[0],
            "high": row[1],
            "low": row[2],
            "close": row[3],
        }
        for index, row in enumerate(rows)
    ]


def _decline_then_reversal():
    """22 falling candles, then a sharp turn up.

    The three legs land on known bars: the SMI turns first (22), the
    Heikin-Ashi candle flips green next (23), and the efficiency oscillator
    needs its 4-step window to fill before it crosses (26). Span 4 -- the
    outside edge of the trader's "within 3-4 candles of each other".
    """
    rows = []
    price = 120.0
    for _ in range(22):
        close = price - 0.8
        rows.append((price, price + 0.15, close - 0.15, close))
        price = close
    for _ in range(8):
        close = price + 1.2
        rows.append((price, close + 0.2, price - 0.1, close))
        price = close
    return rows


DECLINE_THEN_REVERSAL = _decline_then_reversal()


class TestConfluenceEvents:
    def test_the_three_legs_report_on_the_bar_that_completes_them(self):
        bars = ohlc_bars(DECLINE_THEN_REVERSAL)
        events = confluence_events(
            bars, symbol="t", side="long", now=after(len(DECLINE_THEN_REVERSAL))
        )
        assert len(events) == 1
        event = events[0]
        # The LRSI is last, so the alert belongs to its bar - the first moment
        # the trader could have known all three had happened.
        assert (event.ha_index, event.smi_index, event.lrsi_index) == (23, 22, 26)
        assert event.bar_index == 26
        assert event.span_bars == 4
        assert event.lrsi_level == 20.0
        assert event.symbol == "T"

    def test_a_tighter_window_refuses_the_same_three_signals(self):
        """Window tuning is the trader's dial; 4 fires here, 3 does not."""
        bars = ohlc_bars(DECLINE_THEN_REVERSAL)
        now = after(len(DECLINE_THEN_REVERSAL))
        assert confluence_events(bars, symbol="t", side="long", now=now, window_bars=4)
        assert confluence_events(bars, symbol="t", side="long", now=now, window_bars=3) == ()

    def test_two_legs_are_not_a_confluence(self):
        """The efficient run alone crosses the LRSI and never reports."""
        bars = bars_from(CHURN_THEN_RUN)
        assert (
            confluence_events(
                bars, symbol="t", side="long", now=after(len(CHURN_THEN_RUN))
            )
            == ()
        )

    def test_a_forming_bar_never_completes_a_confluence(self):
        bars = ohlc_bars(DECLINE_THEN_REVERSAL)
        forming = OPEN + timedelta(minutes=5 * 26 + 4)
        assert confluence_events(bars, symbol="t", side="long", now=forming) == ()

    def test_an_unreadable_bar_refuses_the_whole_series(self):
        bars = ohlc_bars(DECLINE_THEN_REVERSAL)
        bars[5]["high"] = None
        assert (
            confluence_events(
                bars, symbol="t", side="long", now=after(len(DECLINE_THEN_REVERSAL))
            )
            == ()
        )

    def test_yesterdays_confluence_is_not_todays_alert(self):
        """The indicator warms across sessions; the EVENT belongs to one."""
        yesterday = OPEN - timedelta(days=1)
        bars = ohlc_bars(DECLINE_THEN_REVERSAL, start=yesterday)
        now = yesterday + timedelta(minutes=5 * len(DECLINE_THEN_REVERSAL))
        assert confluence_events(
            bars, symbol="t", side="long", now=now, session=OPEN.date()
        ) == ()
        assert confluence_events(
            bars, symbol="t", side="long", now=now, session=yesterday.date()
        )

    def test_the_short_side_is_the_same_chart_upside_down(self):
        longs = confluence_events(
            ohlc_bars(DECLINE_THEN_REVERSAL),
            symbol="t",
            side="long",
            now=after(len(DECLINE_THEN_REVERSAL)),
        )
        mirrored = [
            (200.0 - row[0], 200.0 - row[2], 200.0 - row[1], 200.0 - row[3])
            for row in DECLINE_THEN_REVERSAL
        ]
        shorts = confluence_events(
            ohlc_bars(mirrored),
            symbol="t",
            side="short",
            now=after(len(mirrored)),
        )
        assert [event.parts for event in shorts] == [event.parts for event in longs]
        assert {event.side for event in shorts} == {"short"}
        # And the un-mirrored series must produce nothing on the short side.
        assert (
            confluence_events(
                ohlc_bars(DECLINE_THEN_REVERSAL),
                symbol="t",
                side="short",
                now=after(len(DECLINE_THEN_REVERSAL)),
            )
            == ()
        )


class TestLatestConfluence:
    def test_it_fires_only_on_the_most_recently_completed_bar(self):
        bars = ohlc_bars(DECLINE_THEN_REVERSAL)
        event = latest_confluence(bars, symbol="t", side="long", now=after(27))
        assert isinstance(event, ConfluenceEvent)
        assert event.bar_index == 26

    def test_an_older_confluence_is_not_re_emitted_every_cycle(self):
        bars = ohlc_bars(DECLINE_THEN_REVERSAL)
        for completed in range(28, len(DECLINE_THEN_REVERSAL) + 1):
            assert (
                latest_confluence(bars, symbol="t", side="long", now=after(completed))
                is None
            )


# ----------------------------------------------------------------------
# R5 section 3.3 -- the first-candle ORB flow.
# ----------------------------------------------------------------------

PRIOR_SESSION = OPEN - timedelta(days=3)


def _orb_series():
    """Prior session flat at 100, then a gap-up open that fades and re-breaks.

    Session bars (index 10 onward): the first candle prints 104.5 and holds
    the high; eight candles fade it back toward 100.7, which is what finally
    drags the EMA9 down and takes the oscillator under 20; then seven candles
    push back through 104.5.
    """
    prior = [(100.0, 100.2, 99.8, 100.0)] * 10
    session = [(103.0, 104.5, 102.9, 104.2)]
    price = 104.2
    for _ in range(8):
        close = price - 0.5
        session.append((price, price + 0.1, close - 0.1, close))
        price = close
    for _ in range(7):
        close = price + 0.75
        session.append((price, close + 0.15, price - 0.1, close))
        price = close
    return prior, session


ORB_PRIOR, ORB_SESSION = _orb_series()


def orb_bars():
    return ohlc_bars(ORB_PRIOR, start=PRIOR_SESSION) + ohlc_bars(ORB_SESSION, start=OPEN)


ORB_NOW = OPEN + timedelta(minutes=5 * len(ORB_SESSION))


class TestOrbFlow:
    def test_the_whole_flow_in_order(self):
        events = orb_events(orb_bars(), symbol="t", side="long", now=ORB_NOW)
        assert [(event.kind, event.bar_index) for event in events] == [
            ("candidate", 10),
            ("lrsi_recross", 22),
            ("new_extreme", 24),
        ]
        candidate = events[0]
        assert candidate.first_extreme == pytest.approx(104.5)
        assert candidate.gap_from == pytest.approx(100.0)
        # The re-break is a NEW session high measured against the first
        # candle's, and it is armed only because the oscillator pulled back.
        assert events[2].level == pytest.approx(104.85)
        assert events[2].deep is True
        assert events[1].is_informational is True
        assert events[2].is_informational is False

    def test_no_gap_is_not_this_setup(self):
        prior, session = _orb_series()
        flat_open = (99.0, 99.5, 98.8, 99.2)
        bars = ohlc_bars(prior, start=PRIOR_SESSION) + ohlc_bars(
            [flat_open] + session[1:], start=OPEN
        )
        assert orb_events(bars, symbol="t", side="long", now=ORB_NOW) == ()

    def test_no_prior_close_is_silence_not_a_zero_gap(self):
        """Missing data is uncertainty. A gap needs something to gap FROM."""
        bars = ohlc_bars(ORB_SESSION, start=OPEN)
        assert orb_events(bars, symbol="t", side="long", now=ORB_NOW) == ()
        # Handed the prior close explicitly, the same bars do report.
        assert orb_events(bars, symbol="t", side="long", now=ORB_NOW, prior_close=100.0)

    def test_an_unreadable_open_refuses_the_series(self):
        bars = orb_bars()
        bars[10]["open"] = None
        assert orb_events(bars, symbol="t", side="long", now=ORB_NOW) == ()

    def test_the_break_needs_the_pullback_first(self):
        """A first candle that never fades arms nothing, so a higher high is
        the trend continuing - not the re-break the trader armed for."""
        prior = [(100.0, 100.2, 99.8, 100.0)] * 10
        price = 104.2
        session = [(103.0, 104.5, 102.9, 104.2)]
        for _ in range(8):
            close = price + 0.6
            session.append((price, close + 0.1, price - 0.1, close))
            price = close
        bars = ohlc_bars(prior, start=PRIOR_SESSION) + ohlc_bars(session, start=OPEN)
        now = OPEN + timedelta(minutes=5 * len(session))
        kinds = [event.kind for event in orb_events(bars, symbol="t", side="long", now=now)]
        assert kinds == ["candidate"]

    def test_the_re_break_reports_once(self):
        """The second higher bar of the same push is the move, not a signal."""
        events = orb_events(orb_bars(), symbol="t", side="long", now=ORB_NOW)
        assert [event.kind for event in events].count("new_extreme") == 1

    def test_a_forming_first_candle_is_not_a_candidate(self):
        bars = orb_bars()
        forming = OPEN + timedelta(minutes=4)
        assert orb_events(bars, symbol="t", side="long", now=forming) == ()

    def test_the_short_side_is_the_same_chart_upside_down(self):
        longs = orb_events(orb_bars(), symbol="t", side="long", now=ORB_NOW)
        mirrored = [
            {
                "dt": bar["dt"],
                "open": 200.0 - bar["open"],
                "high": 200.0 - bar["low"],
                "low": 200.0 - bar["high"],
                "close": 200.0 - bar["close"],
            }
            for bar in orb_bars()
        ]
        shorts = orb_events(mirrored, symbol="t", side="short", now=ORB_NOW)
        assert [(event.kind, event.bar_index) for event in shorts] == [
            (event.kind, event.bar_index) for event in longs
        ]
        assert {event.side for event in shorts} == {"short"}
        assert shorts[0].first_extreme == pytest.approx(200.0 - 104.5)
        # A gap-up name is not a short-side ORB candidate.
        assert orb_events(orb_bars(), symbol="t", side="short", now=ORB_NOW) == ()


class TestLatestOrbEvents:
    def test_only_the_steps_on_the_last_completed_bar(self):
        bars = orb_bars()
        # The re-break is session bar 14 (global 24), so it is the last
        # completed bar once 15 session bars have closed.
        at_the_break = OPEN + timedelta(minutes=5 * 15)
        assert [
            event.kind
            for event in latest_orb_events(
                bars, symbol="t", side="long", now=at_the_break
            )
        ] == ["new_extreme"]
        # One bar later the same break is history, not an alert.
        assert (
            latest_orb_events(
                bars,
                symbol="t",
                side="long",
                now=at_the_break + timedelta(minutes=5),
            )
            == ()
        )
