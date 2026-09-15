"""PCT-1 - the Pullback alert: the pure rule `pullback_sma_reclaim_v1`.

Written RED, before the module exists (`docs/PULLBACK_COMPRESSION_TRENDLINE_PLAN.md`
section 5, item 1). The desk-side half - the intraday cache, the watch kind, the
arm bar, the poll, the auto-arm and the three claim names - is in
`tests/test_pct1_pullback_desk.py`, which imports the fixtures from here.

=== The contract these tests assert ===

``scripts/indicators/pullback_sma_reclaim.py``:

``RULE_VERSION == "pullback_sma_reclaim_v1"``.

``evaluate(bars, *, side, sma_length, bar_minutes, armed_at, now, episode_state=None)``
returns ``None`` (NOT MEASURED) below the warm-up - 160 M15 bars, 85 M30 bars -
and for a series whose last completed bar is more than 24 h behind ``now``;
otherwise a frozen result carrying ``fired`` (a tuple of ``Fire``), the new
``episode_state`` and ``reason``.

A ``Fire`` carries ``trigger, timeframe, bar_dt, sma, close, lrsi,
lrsi_from_below_50, atr, message``. The trigger names are the packet's own:
``sma_reclaim_lrsi``, ``reclaim_then_lrsi``, ``sma_retest``.

SMA is the simple mean of the last ``sma_length`` completed closes. LRSI is the
champion's ``indicators.efficiency_lrsi.compute_efficiency_lrsi`` read at level
80 - never ``CROSS_LEVELS``, which stays ``(20, 50)`` - with the closes NEGATED
for a short, which is the idiom ``m5_signal_engines.latest_lrsi_cross`` already
uses. ATR is Wilder-14 over the same bars.

=== Why the fixtures are built the way they are ===

Every close series below is written by hand here and its LRSI is then computed
with ``compute_efficiency_lrsi`` INSIDE the test, so the 80-cross assertions are
the series' own arithmetic rather than a number copied out of the module under
test. Each key value is ALSO pinned as a literal, so an edited fixture fails
loudly instead of quietly re-deriving itself.

Two ambiguities in the packet are deliberately designed AROUND rather than
guessed at, so these tests hold under either reading:

* ``lrsi_from_below_50`` ("was below 50 in the 2-4 bars before") is measured
  from the cross bar or from the reclaim bar - in ``M15_LONG_CLOSES`` the two
  are the SAME bar, and in ``M15_LATE_FLAG_CLOSES`` both windows read above 50;
* whether the reclaim bar itself can also count as an ``sma_retest``. No test
  here asserts the whole ``fired`` tuple by equality; each asserts the trigger
  it is about, by name and by count.

Bars are session-aligned on the desk's market-local clock (06:30 -> 13:00
Pacific, i.e. the 09:30-16:00 ET regular session), which is the convention
``indicators.h1_ema_bounce.closed_h1_bars`` and the chart-watch store already
hold. 390 minutes is 26 M15 buckets and 13 M30 buckets a session, with no short
closing bucket for either.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

RULE_VERSION = "pullback_sma_reclaim_v1"
TRIGGER_RECLAIM = "sma_reclaim_lrsi"
TRIGGER_THEN_LRSI = "reclaim_then_lrsi"
TRIGGER_RETEST = "sma_retest"
TRIGGER_H1 = "h1_ema15_bounce"

#: The trader's two timeframes and the SMA each one is read with.
M15_SMA = 150
M30_SMA = 75
#: Warm-up: the SMA plus the LRSI's own ~13 bars, rounded to the packet's numbers.
M15_WARMUP = 160
M30_WARMUP = 85

# ---------------------------------------------------------------------------
# Session-aligned timestamps
# ---------------------------------------------------------------------------
#: Ten consecutive exchange sessions. Pinned against `market_calendar` by
#: `test_the_fixture_days_are_real_exchange_sessions`, so a holiday cannot
#: silently reshape a fixture.
SESSION_DAYS = (17, 18, 19, 20, 21, 24, 25, 26, 27, 28)
SESSION_YEAR = 2026
SESSION_MONTH = 8
SESSION_OPEN_HOUR = 6
SESSION_OPEN_MINUTE = 30
SESSION_MINUTES = 390  # 06:30 -> 13:00 market-local


def bar_dt(index: int, bar_minutes: int) -> datetime:
    """The start of the `index`-th session-aligned bar of this timeframe."""
    per_session = SESSION_MINUTES // bar_minutes
    day = SESSION_DAYS[index // per_session]
    slot = index % per_session
    return datetime(
        SESSION_YEAR, SESSION_MONTH, day, SESSION_OPEN_HOUR, SESSION_OPEN_MINUTE
    ) + timedelta(minutes=bar_minutes * slot)


def bar_end(index: int, bar_minutes: int) -> datetime:
    return bar_dt(index, bar_minutes) + timedelta(minutes=bar_minutes)


# ---------------------------------------------------------------------------
# Close paths
# ---------------------------------------------------------------------------
def _extend(closes: list[float], steps: list[float]) -> list[float]:
    out = list(closes)
    for step in steps:
        out.append(out[-1] + step)
    return out


#: 200 M15 bars of a steady advance, so the price sits well above SMA-150 and
#: there is nothing to reclaim yet.
M15_BASE = [100.0 + 0.20 * index for index in range(200)]
#: 90 M30 bars of the same shape against SMA-75.
M30_BASE = [100.0 + 0.30 * index for index in range(90)]

#: **The long M15 golden.** Ten bars down THROUGH the 150-SMA (the first close
#: below it is bar 204), then eight up. Bar 214 is the first completed close
#: back ABOVE the SMA *and* the only bar where the LRSI crosses up through 80.
M15_LONG_CLOSES = _extend(M15_BASE, [-3.0] * 10 + [4.0] * 8)
M15_RECLAIM_INDEX = 214
M15_FIRST_BELOW_INDEX = 204
M15_RECLAIM_CLOSE = 129.8
M15_RECLAIM_SMA = 126.04
M15_RECLAIM_LRSI = 86.81175688256107
M15_RECLAIM_ATR = 2.4883666625186223

#: The same shape, but the approach to the reclaim is a grind rather than a
#: collapse: the LRSI reads 71.5 / 74.4 / 76.3 in the three bars before the
#: cross, so it crosses 80 (at bar 223, one bar before the reclaim at 224 -
#: inside the "reclaim bar or the two before it" window) WITHOUT having been
#: under 50 two-to-four bars earlier. `lrsi_from_below_50` is therefore False.
M15_LATE_FLAG_CLOSES = _extend(
    M15_BASE, [-2.0] * 14 + [2.0] * 5 + [-4.0] + [2.0] * 10
)
M15_LATE_FLAG_RECLAIM_INDEX = 224
M15_LATE_FLAG_CROSS_INDEX = 223

#: A reclaim (bar 223) whose only 80-cross was FOUR bars back (bar 219): a pop
#: that ran out of steam and then crept over the line. Outside the window, so
#: `sma_reclaim_lrsi` must not fire.
M15_STALE_CROSS_CLOSES = _extend(
    M15_BASE, [-2.0] * 14 + [2.0] * 4 + [1.2] * 14
)
M15_STALE_CROSS_RECLAIM_INDEX = 223
M15_STALE_CROSS_INDEX = 219

#: Two complete episodes: reclaim at 214, a fresh close back below the SMA, and
#: a second reclaim at 228. Both carry their own 80-cross on the reclaim bar.
M15_TWO_EPISODE_CLOSES = _extend(
    M15_BASE, [-3.0] * 10 + [4.0] * 6 + [-3.0] * 8 + [4.0] * 8
)
M15_EPISODE_ONE_INDEX = 214
M15_EPISODE_TWO_INDEX = 228
M15_EPISODE_TWO_SMA = 126.74

#: **The long M30 golden**, against SMA-75: reclaim and 80-cross both at bar 105.
M30_LONG_CLOSES = _extend(M30_BASE, [-2.0] * 10 + [2.0] * 8)
M30_RECLAIM_INDEX = 105
M30_RECLAIM_CLOSE = 118.7
M30_RECLAIM_SMA = 117.34933333333335
M30_RECLAIM_ATR = 1.5806095172134358

#: M30 `reclaim_then_lrsi`: the reclaim prints at bar 102, every completed close
#: after it stays above the SMA, and the 80-cross lands THREE bars later at 105
#: - too late for `sma_reclaim_lrsi`, which is exactly the gap this trigger was
#: asked for.
M30_HOLD_CLOSES = _extend(M30_BASE, [-3.0] * 12 + [30.0, -1.0] + [2.0] * 8)
M30_HOLD_RECLAIM_INDEX = 102
M30_HOLD_CROSS_INDEX = 105

#: The same, with ONE completed M30 close back below the SMA (bar 105, 96.70
#: against an SMA of 116.096) before the 80-cross at bar 112. The episode is
#: over, so nothing fires - and the price never regains the SMA, so no NEW
#: episode can answer in its place.
M30_CANCELLED_CLOSES = _extend(
    M30_BASE, [-3.0] * 12 + [30.0, -1.0, 2.0, -25.0] + [2.0] * 8
)
M30_CANCELLED_BELOW_INDEX = 105
M30_CANCELLED_CROSS_INDEX = 112

#: The retest bar appended after the M15 reclaim: it closes 2.85 clear of the
#: SMA and dips its LOW to exactly 0.10 ATR-14 above the line (the tagging
#: variant) or 0.60 ATR above it (the variant that never comes near). Both lows
#: are fixed points of `wilder_atr` over the resulting series and are RE-DERIVED
#: in the test, so the 0.25 ATR threshold is asserted from both sides.
M15_RETEST_CLOSE = 129.0
M15_RETEST_SMA = 126.14666666666666
M15_RETEST_TAG_LOW = 126.4020007088374
M15_RETEST_TAG_ATR = 2.553340421707478
M15_RETEST_MISS_LOW = 127.62620502061496
M15_RETEST_MISS_ATR = 2.46589725658051
#: The third variant: the low goes clean THROUGH the SMA but the bar closes
#: below it. That is not a retest; it is the next episode starting.
M15_RETEST_FAIL_CLOSE = 124.0
M15_RETEST_FAIL_LOW = 123.5

HALF_RANGE = 0.10


def make_bars(closes, bar_minutes: int, *, half_range: float = HALF_RANGE):
    """Session-aligned OHLCV dicts, shaped like `closed_h1_bars` output."""
    bars = []
    for index, close in enumerate(closes):
        bars.append(
            {
                "dt": bar_dt(index, bar_minutes),
                "open": close,
                "high": close + half_range,
                "low": close - half_range,
                "close": close,
                "volume": 1_000 + index,
            }
        )
    return bars


def mirror(bars, axis: float = 300.0):
    """The same tape reflected: a LONG pullback becomes the SHORT one.

    Reflection is an affine map, so the EMA - and therefore every LRSI step -
    is exactly the negation `m5_signal_engines._closes` applies for a short.
    `test_the_short_mirror_is_the_negated_close_series` pins that equality.
    """
    return [
        {
            "dt": bar["dt"],
            "open": axis - bar["open"],
            "high": axis - bar["low"],
            "low": axis - bar["high"],
            "close": axis - bar["close"],
            "volume": bar["volume"],
        }
        for bar in bars
    ]


def lrsi_values(closes, *, side: str = "long"):
    """The champion oscillator over this fixture's own closes."""
    from indicators.efficiency_lrsi import compute_efficiency_lrsi

    sign = -1.0 if str(side).lower() == "short" else 1.0
    return compute_efficiency_lrsi([sign * float(close) for close in closes]).values


def cross_up_80(closes, *, side: str = "long") -> tuple[int, ...]:
    from indicators.efficiency_lrsi import compute_efficiency_lrsi

    sign = -1.0 if str(side).lower() == "short" else 1.0
    result = compute_efficiency_lrsi([sign * float(close) for close in closes])
    return result.cross_up_indices(80.0)


def simple_mean(closes, index: int, length: int) -> float:
    return sum(closes[index + 1 - length: index + 1]) / float(length)


def evaluate(
    bars,
    *,
    side="long",
    sma_length,
    bar_minutes,
    armed_at,
    now,
    episode_state=None,
):
    """The rule under test, imported inside the call so this file still loads."""
    from indicators.pullback_sma_reclaim import evaluate as rule_evaluate

    return rule_evaluate(
        bars,
        side=side,
        sma_length=sma_length,
        bar_minutes=bar_minutes,
        armed_at=armed_at,
        now=now,
        episode_state=episode_state,
    )


def triggers_of(result) -> list[str]:
    return [str(fire.trigger) for fire in (result.fired if result is not None else ())]


def one_fire(result, trigger: str):
    """The single fire with this trigger. Fails loudly on nought or two."""
    hits = [fire for fire in result.fired if str(fire.trigger) == trigger]
    assert len(hits) == 1, f"{trigger} fired {len(hits)} times: {triggers_of(result)}"
    return hits[0]


def m15_eval(closes, upto: int, **kwargs):
    bars = make_bars(closes[: upto + 1], 15)
    kwargs.setdefault("armed_at", bar_dt(0, 15))
    kwargs.setdefault("now", bar_end(upto, 15))
    return evaluate(bars, sma_length=M15_SMA, bar_minutes=15, **kwargs)


def m30_eval(closes, upto: int, **kwargs):
    bars = make_bars(closes[: upto + 1], 30)
    kwargs.setdefault("armed_at", bar_dt(0, 30))
    kwargs.setdefault("now", bar_end(upto, 30))
    return evaluate(bars, sma_length=M30_SMA, bar_minutes=30, **kwargs)


# ---------------------------------------------------------------------------
# 0. The fixtures are what this file says they are
# ---------------------------------------------------------------------------
def test_the_fixture_days_are_real_exchange_sessions():
    import market_calendar

    for day in SESSION_DAYS:
        stamp = date(SESSION_YEAR, SESSION_MONTH, day)
        assert market_calendar.is_session(stamp), f"{stamp} is not a session"
    # 390 minutes divides into whole buckets on both timeframes, so neither
    # has the short closing bucket the H1 series has to special-case.
    assert SESSION_MINUTES % 15 == 0 and SESSION_MINUTES % 30 == 0
    assert bar_dt(0, 15) == datetime(2026, 8, 17, 6, 30)
    assert bar_dt(25, 15) == datetime(2026, 8, 17, 12, 45)
    assert bar_dt(26, 15) == datetime(2026, 8, 18, 6, 30)
    assert bar_dt(12, 30) == datetime(2026, 8, 17, 12, 30)


def test_the_m15_golden_reclaims_and_crosses_eighty_on_the_same_bar():
    """The fixture's own arithmetic, before any rule reads it."""
    closes = M15_LONG_CLOSES
    values = lrsi_values(closes)

    assert cross_up_80(closes)[-1] == M15_RECLAIM_INDEX
    assert [index for index in cross_up_80(closes) if index >= 200] == [
        M15_RECLAIM_INDEX
    ]
    assert values[M15_RECLAIM_INDEX] == pytest.approx(M15_RECLAIM_LRSI, abs=1e-9)
    assert values[M15_RECLAIM_INDEX - 1] == pytest.approx(8.121287777090282, abs=1e-9)
    for back in (2, 3, 4):
        assert values[M15_RECLAIM_INDEX - back] == 0.0

    sma_at = simple_mean(closes, M15_RECLAIM_INDEX, M15_SMA)
    assert sma_at == pytest.approx(M15_RECLAIM_SMA, abs=1e-9)
    assert closes[M15_RECLAIM_INDEX] == pytest.approx(M15_RECLAIM_CLOSE, abs=1e-9)
    assert closes[M15_RECLAIM_INDEX] > sma_at
    assert closes[M15_RECLAIM_INDEX - 1] < simple_mean(
        closes, M15_RECLAIM_INDEX - 1, M15_SMA
    )
    assert closes[M15_FIRST_BELOW_INDEX] < simple_mean(
        closes, M15_FIRST_BELOW_INDEX, M15_SMA
    )


def test_the_short_mirror_is_the_negated_close_series():
    """A mirrored tape and a negated one are the same oscillator (M5 idiom)."""
    bars = make_bars(M15_LONG_CLOSES, 15)
    mirrored = [bar["close"] for bar in mirror(bars)]

    assert lrsi_values(mirrored, side="short")[M15_RECLAIM_INDEX] == pytest.approx(
        M15_RECLAIM_LRSI, abs=1e-9
    )
    assert cross_up_80(mirrored, side="short")[-1] == M15_RECLAIM_INDEX
    # And the mirror really did flip the side of the line.
    mirrored_sma = simple_mean(mirrored, M15_RECLAIM_INDEX, M15_SMA)
    assert mirrored[M15_RECLAIM_INDEX] < mirrored_sma
    assert mirrored[M15_RECLAIM_INDEX - 1] > simple_mean(
        mirrored, M15_RECLAIM_INDEX - 1, M15_SMA
    )


# ---------------------------------------------------------------------------
# 1. The rule sheet
# ---------------------------------------------------------------------------
def test_the_rule_sheet_names_its_version_and_leaves_the_live_levels_alone():
    from indicators import efficiency_lrsi
    from indicators import pullback_sma_reclaim as rule

    assert rule.RULE_VERSION == RULE_VERSION
    # The champion's M5 pair is untouched; 80 is this rule's own parameter.
    assert efficiency_lrsi.CROSS_LEVELS == (20.0, 50.0)


# ---------------------------------------------------------------------------
# 2. sma_reclaim_lrsi
# ---------------------------------------------------------------------------
def test_a_m15_reclaim_with_the_cross_on_the_reclaim_bar_fires_from_below_fifty():
    result = m15_eval(M15_LONG_CLOSES, M15_RECLAIM_INDEX)

    assert result is not None
    fire = one_fire(result, TRIGGER_RECLAIM)
    assert fire.timeframe == "M15"
    assert fire.bar_dt == bar_dt(M15_RECLAIM_INDEX, 15)
    assert fire.close == pytest.approx(M15_RECLAIM_CLOSE, abs=1e-9)
    assert fire.sma == pytest.approx(M15_RECLAIM_SMA, abs=1e-9)
    assert fire.sma == pytest.approx(
        simple_mean(M15_LONG_CLOSES, M15_RECLAIM_INDEX, M15_SMA), abs=1e-9
    )
    assert fire.lrsi == pytest.approx(M15_RECLAIM_LRSI, abs=1e-9)
    assert fire.lrsi_from_below_50 is True
    assert fire.atr == pytest.approx(M15_RECLAIM_ATR, abs=1e-9)
    # The line the trader reads names the trigger and the timeframe.
    assert "M15" in str(fire.message)
    assert TRIGGER_THEN_LRSI not in triggers_of(result)


def test_a_m30_reclaim_fires_against_the_seventy_five_sma():
    result = m30_eval(M30_LONG_CLOSES, M30_RECLAIM_INDEX)

    assert result is not None
    fire = one_fire(result, TRIGGER_RECLAIM)
    assert fire.timeframe == "M30"
    assert fire.bar_dt == bar_dt(M30_RECLAIM_INDEX, 30)
    assert fire.close == pytest.approx(M30_RECLAIM_CLOSE, abs=1e-9)
    assert fire.sma == pytest.approx(M30_RECLAIM_SMA, abs=1e-9)
    assert fire.sma == pytest.approx(
        simple_mean(M30_LONG_CLOSES, M30_RECLAIM_INDEX, M30_SMA), abs=1e-9
    )
    assert fire.lrsi_from_below_50 is True
    assert fire.atr == pytest.approx(M30_RECLAIM_ATR, abs=1e-9)
    assert "M30" in str(fire.message)


def test_the_short_side_fires_on_the_mirrored_tape():
    bars = mirror(make_bars(M15_LONG_CLOSES[: M15_RECLAIM_INDEX + 1], 15))
    closes = [bar["close"] for bar in bars]

    result = evaluate(
        bars,
        side="short",
        sma_length=M15_SMA,
        bar_minutes=15,
        armed_at=bar_dt(0, 15),
        now=bar_end(M15_RECLAIM_INDEX, 15),
    )

    assert result is not None
    fire = one_fire(result, TRIGGER_RECLAIM)
    assert fire.bar_dt == bar_dt(M15_RECLAIM_INDEX, 15)
    assert fire.close == pytest.approx(300.0 - M15_RECLAIM_CLOSE, abs=1e-9)
    assert fire.sma == pytest.approx(
        simple_mean(closes, M15_RECLAIM_INDEX, M15_SMA), abs=1e-9
    )
    assert fire.lrsi == pytest.approx(M15_RECLAIM_LRSI, abs=1e-9)
    assert fire.lrsi_from_below_50 is True


def test_a_grinding_approach_fires_but_says_it_was_never_under_fifty():
    """The trader's "ideally it was below 50 2-4 bars previously" is a LABEL.

    Here the oscillator reads 71.5 / 74.4 / 76.3 on its way up, so the flag is
    False under either reading of "2-4 bars before" (the cross at 223 or the
    reclaim at 224) - and the fire still happens, because the clause is quality,
    never a gate.
    """
    closes = M15_LATE_FLAG_CLOSES
    values = lrsi_values(closes)
    assert cross_up_80(closes)[-1] == M15_LATE_FLAG_CROSS_INDEX
    for index in range(M15_LATE_FLAG_CROSS_INDEX - 4, M15_LATE_FLAG_CROSS_INDEX):
        assert values[index] >= 50.0
    assert values[M15_LATE_FLAG_CROSS_INDEX - 1] == pytest.approx(
        76.2982279573176, abs=1e-9
    )
    assert closes[M15_LATE_FLAG_RECLAIM_INDEX] > simple_mean(
        closes, M15_LATE_FLAG_RECLAIM_INDEX, M15_SMA
    )
    assert closes[M15_LATE_FLAG_RECLAIM_INDEX - 1] < simple_mean(
        closes, M15_LATE_FLAG_RECLAIM_INDEX - 1, M15_SMA
    )

    result = m15_eval(closes, M15_LATE_FLAG_RECLAIM_INDEX)

    assert result is not None
    fire = one_fire(result, TRIGGER_RECLAIM)
    assert fire.lrsi_from_below_50 is False


def test_a_reclaim_whose_cross_was_four_bars_back_does_not_fire():
    closes = M15_STALE_CROSS_CLOSES
    assert cross_up_80(closes)[-1] == M15_STALE_CROSS_INDEX
    assert M15_STALE_CROSS_RECLAIM_INDEX - M15_STALE_CROSS_INDEX == 4
    assert closes[M15_STALE_CROSS_RECLAIM_INDEX] > simple_mean(
        closes, M15_STALE_CROSS_RECLAIM_INDEX, M15_SMA
    )

    result = m15_eval(closes, M15_STALE_CROSS_RECLAIM_INDEX)

    assert result is not None  # measurable, and the answer is "no"
    assert TRIGGER_RECLAIM not in triggers_of(result)


def test_a_reclaim_that_finished_before_the_arm_is_not_this_watch_s_event():
    """The armed-watch convention (`chart_watch.h1_event_is_post_arm`): the
    event bar's END must be strictly after `armed_at`."""
    after = bar_end(M15_RECLAIM_INDEX, 15) + timedelta(minutes=1)

    late = m15_eval(M15_LONG_CLOSES, M15_RECLAIM_INDEX, armed_at=after)
    early = m15_eval(
        M15_LONG_CLOSES,
        M15_RECLAIM_INDEX,
        armed_at=bar_dt(M15_RECLAIM_INDEX - 9, 15),
    )

    assert late is not None
    assert TRIGGER_RECLAIM not in triggers_of(late)
    assert TRIGGER_RECLAIM in triggers_of(early)


# ---------------------------------------------------------------------------
# 3. reclaim_then_lrsi (M30)
# ---------------------------------------------------------------------------
def test_reclaim_then_lrsi_fires_on_a_later_cross_while_every_close_holds_the_sma():
    closes = M30_HOLD_CLOSES
    assert cross_up_80(closes)[-1] == M30_HOLD_CROSS_INDEX
    assert M30_HOLD_CROSS_INDEX - M30_HOLD_RECLAIM_INDEX == 3  # too late to reclaim-fire
    assert closes[M30_HOLD_RECLAIM_INDEX - 1] < simple_mean(
        closes, M30_HOLD_RECLAIM_INDEX - 1, M30_SMA
    )
    for index in range(M30_HOLD_RECLAIM_INDEX, M30_HOLD_CROSS_INDEX + 1):
        assert closes[index] > simple_mean(closes, index, M30_SMA)

    result = m30_eval(closes, M30_HOLD_CROSS_INDEX)

    assert result is not None
    fire = one_fire(result, TRIGGER_THEN_LRSI)
    assert fire.timeframe == "M30"
    assert fire.bar_dt == bar_dt(M30_HOLD_CROSS_INDEX, 30)
    assert fire.sma == pytest.approx(
        simple_mean(closes, M30_HOLD_CROSS_INDEX, M30_SMA), abs=1e-9
    )
    # The cross is four bars past the reclaim window, so the reclaim trigger
    # must stay silent - the two are not the same event told twice.
    assert TRIGGER_RECLAIM not in triggers_of(result)


def test_one_completed_close_back_below_the_sma_ends_the_episode():
    closes = M30_CANCELLED_CLOSES
    assert cross_up_80(closes)[-1] == M30_CANCELLED_CROSS_INDEX
    assert closes[M30_CANCELLED_BELOW_INDEX] < simple_mean(
        closes, M30_CANCELLED_BELOW_INDEX, M30_SMA
    )
    # ...and the tape never regains the line, so no NEW episode can answer.
    for index in range(M30_CANCELLED_BELOW_INDEX, M30_CANCELLED_CROSS_INDEX + 1):
        assert closes[index] < simple_mean(closes, index, M30_SMA)

    result = m30_eval(closes, M30_CANCELLED_CROSS_INDEX)

    assert result is not None
    assert TRIGGER_THEN_LRSI not in triggers_of(result)
    assert TRIGGER_RECLAIM not in triggers_of(result)


# ---------------------------------------------------------------------------
# 4. sma_retest
# ---------------------------------------------------------------------------
def _retest_bars(low: float, close: float, high: float):
    """The M15 golden truncated at its reclaim, plus one retest candidate."""
    closes = list(M15_LONG_CLOSES[: M15_RECLAIM_INDEX + 1]) + [close]
    bars = make_bars(closes, 15)
    bars[-1] = {
        "dt": bar_dt(M15_RECLAIM_INDEX + 1, 15),
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": 2_000,
    }
    return bars, closes


def test_a_low_inside_a_quarter_atr_of_the_sma_that_closes_above_it_retests():
    from indicators.atr import wilder_atr

    bars, closes = _retest_bars(
        M15_RETEST_TAG_LOW, M15_RETEST_CLOSE, M15_RETEST_CLOSE + HALF_RANGE
    )
    index = M15_RECLAIM_INDEX + 1
    sma_at = simple_mean(closes, index, M15_SMA)
    atr = wilder_atr(bars, 14)
    assert sma_at == pytest.approx(M15_RETEST_SMA, abs=1e-9)
    assert atr == pytest.approx(M15_RETEST_TAG_ATR, abs=1e-9)
    # The number the rule has to read: a tenth of an ATR off the line.
    assert (M15_RETEST_TAG_LOW - sma_at) / atr == pytest.approx(0.10, abs=1e-9)
    assert closes[index] > sma_at

    result = evaluate(
        bars,
        sma_length=M15_SMA,
        bar_minutes=15,
        armed_at=bar_dt(0, 15),
        now=bar_end(index, 15),
    )

    assert result is not None
    fire = one_fire(result, TRIGGER_RETEST)
    assert fire.timeframe == "M15"
    assert fire.bar_dt == bar_dt(index, 15)
    assert fire.sma == pytest.approx(M15_RETEST_SMA, abs=1e-9)
    assert fire.atr == pytest.approx(M15_RETEST_TAG_ATR, abs=1e-9)


def test_a_low_six_tenths_of_an_atr_above_the_sma_never_tagged_it():
    from indicators.atr import wilder_atr

    bars, closes = _retest_bars(
        M15_RETEST_MISS_LOW, M15_RETEST_CLOSE, M15_RETEST_CLOSE + HALF_RANGE
    )
    index = M15_RECLAIM_INDEX + 1
    sma_at = simple_mean(closes, index, M15_SMA)
    atr = wilder_atr(bars, 14)
    assert atr == pytest.approx(M15_RETEST_MISS_ATR, abs=1e-9)
    assert (M15_RETEST_MISS_LOW - sma_at) / atr == pytest.approx(0.60, abs=1e-9)

    result = evaluate(
        bars,
        sma_length=M15_SMA,
        bar_minutes=15,
        armed_at=bar_dt(0, 15),
        now=bar_end(index, 15),
    )

    assert result is not None
    assert TRIGGER_RETEST not in triggers_of(result)


def test_a_bar_through_the_sma_that_closes_below_it_is_not_a_retest():
    bars, closes = _retest_bars(
        M15_RETEST_FAIL_LOW, M15_RETEST_FAIL_CLOSE, M15_RETEST_FAIL_CLOSE + 2.5
    )
    index = M15_RECLAIM_INDEX + 1
    sma_at = simple_mean(closes, index, M15_SMA)
    assert M15_RETEST_FAIL_LOW < sma_at  # it went clean through
    assert closes[index] < sma_at  # and stayed there

    result = evaluate(
        bars,
        sma_length=M15_SMA,
        bar_minutes=15,
        armed_at=bar_dt(0, 15),
        now=bar_end(index, 15),
    )

    assert result is not None
    assert TRIGGER_RETEST not in triggers_of(result)


# ---------------------------------------------------------------------------
# 5. Episodes
# ---------------------------------------------------------------------------
def test_a_trigger_fires_once_per_episode_and_again_in_the_next_one():
    closes = M15_TWO_EPISODE_CLOSES
    assert [index for index in cross_up_80(closes) if index >= 200] == [
        M15_EPISODE_ONE_INDEX,
        M15_EPISODE_TWO_INDEX,
    ]

    first = m15_eval(closes, M15_EPISODE_ONE_INDEX)
    assert first is not None
    assert triggers_of(first).count(TRIGGER_RECLAIM) == 1

    # The very next poll, same bars, carrying the state forward: the episode
    # has already spoken, so it does not speak twice.
    again = m15_eval(
        closes, M15_EPISODE_ONE_INDEX, episode_state=first.episode_state
    )
    assert again is not None
    assert triggers_of(again) == []

    # A completed close back below the SMA opens a NEW episode, and the second
    # reclaim is a second event, not a repeat of the first.
    second = m15_eval(
        closes, M15_EPISODE_TWO_INDEX, episode_state=again.episode_state
    )
    assert second is not None
    fire = one_fire(second, TRIGGER_RECLAIM)
    assert fire.bar_dt == bar_dt(M15_EPISODE_TWO_INDEX, 15)
    assert fire.sma == pytest.approx(M15_EPISODE_TWO_SMA, abs=1e-9)
    assert fire.lrsi_from_below_50 is True


# ---------------------------------------------------------------------------
# 6. Not measured: warm-up, staleness, the forming bar
# ---------------------------------------------------------------------------
def test_fewer_than_a_hundred_and_sixty_m15_bars_is_not_measured():
    short = m15_eval(M15_LONG_CLOSES, M15_WARMUP - 2)
    enough = m15_eval(M15_LONG_CLOSES, M15_WARMUP - 1)

    assert short is None, "159 M15 bars cannot answer"
    assert enough is not None, "160 M15 bars is the warm-up"


def test_fewer_than_eighty_five_m30_bars_is_not_measured():
    short = m30_eval(M30_LONG_CLOSES, M30_WARMUP - 2)
    enough = m30_eval(M30_LONG_CLOSES, M30_WARMUP - 1)

    assert short is None, "84 M30 bars cannot answer"
    assert enough is not None, "85 M30 bars is the warm-up"


def test_a_tape_more_than_a_day_old_is_not_measured():
    stale = m15_eval(
        M15_LONG_CLOSES,
        M15_RECLAIM_INDEX,
        now=bar_end(M15_RECLAIM_INDEX, 15) + timedelta(hours=25),
    )
    fresh = m15_eval(
        M15_LONG_CLOSES,
        M15_RECLAIM_INDEX,
        now=bar_end(M15_RECLAIM_INDEX, 15) + timedelta(hours=23),
    )

    assert stale is None
    assert fresh is not None
    assert TRIGGER_RECLAIM in triggers_of(fresh)


def test_the_forming_bar_is_never_read():
    """plan.md sec 5: completed bars only. One minute short of the bell the
    reclaim has not happened yet, and the rule must not peek."""
    forming = m15_eval(
        M15_LONG_CLOSES,
        M15_RECLAIM_INDEX,
        now=bar_dt(M15_RECLAIM_INDEX, 15) + timedelta(minutes=14),
    )
    complete = m15_eval(
        M15_LONG_CLOSES,
        M15_RECLAIM_INDEX,
        now=bar_dt(M15_RECLAIM_INDEX, 15) + timedelta(minutes=15),
    )

    assert forming is not None
    assert triggers_of(forming) == []
    assert TRIGGER_RECLAIM in triggers_of(complete)


def test_the_rule_reads_no_clock_of_its_own():
    """`now` is a parameter, so the same bars answer the same way twice."""
    first = m15_eval(M15_LONG_CLOSES, M15_RECLAIM_INDEX)
    second = m15_eval(M15_LONG_CLOSES, M15_RECLAIM_INDEX)

    assert first is not None and second is not None
    assert triggers_of(first) == triggers_of(second)
    assert one_fire(first, TRIGGER_RECLAIM).bar_dt == one_fire(
        second, TRIGGER_RECLAIM
    ).bar_dt
