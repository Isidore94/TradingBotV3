"""The group tape's RRS maths (plan.md Phase 0.5 item 11, packet T-1).

The point of this file is the parity test. The tape's complaint was never that
the number was wrong - an independent Yahoo recompute on 2026-08-27 ranked the
same window the same way - it was that the number was late and reached across
the overnight gap. So the formula is lifted out unchanged and PROVEN unchanged
against `legacy.real_relative_strength`, and the new behaviour (today only,
completed only, three windows, UNKNOWN never invented) is tested on top of it.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


SESSION_OPEN = datetime(2026, 8, 27, 6, 30)  # desk-local (PT) regular open


def _series(closes, *, start=SESSION_OPEN, spread=0.40, step_minutes=5):
    """Bar dicts with a deterministic range around each close.

    The range varies with the index so the Wilder ATR actually smooths
    something - a constant-range series would make the parity test pass even
    if the smoothing loop were dropped.
    """
    bars = []
    for index, close in enumerate(closes):
        wobble = spread * (1.0 + (index % 3) * 0.25)
        bars.append(
            {
                "dt": start + timedelta(minutes=step_minutes * index),
                "open": close - wobble / 3.0,
                "high": close + wobble,
                "low": close - wobble,
                "close": close,
                "volume": 1000.0 + index,
            }
        )
    return bars


def _as_ib(bars):
    from bounce_bot_lib.legacy import IbBar

    return [
        IbBar(
            dt=bar["dt"],
            open=bar["open"],
            high=bar["high"],
            low=bar["low"],
            close=bar["close"],
            volume=bar["volume"],
        )
        for bar in bars
    ]


def _rising(count, base=100.0, step=0.20):
    return [base + step * index for index in range(count)]


def _choppy(count, base=50.0):
    return [base + (index % 5) * 0.30 - (index % 3) * 0.17 for index in range(count)]


# --------------------------------------------------------------------- parity


@pytest.mark.parametrize("length", [6, 12, 18])
def test_the_formula_is_the_same_number_legacy_produces(length):
    """Hard rule 5: identical bars in, equal to 1e-9 out.

    This is the whole licence for not calling BounceBot. If it ever fails,
    the tape has started answering a different question than the RS Window
    tab and the scan log, and the difference is the bug.
    """
    import group_rrs
    from bounce_bot_lib.legacy import real_relative_strength as legacy_rrs

    symbol = _series(_rising(40))
    spy = _series(_choppy(40), spread=0.25)

    legacy_value, legacy_power = legacy_rrs(_as_ib(symbol), _as_ib(spy), length=length)
    mine, power = group_rrs.real_relative_strength(symbol, spy, length)

    assert legacy_value is not None, "the fixture must actually produce a number"
    assert mine == pytest.approx(legacy_value, abs=1e-9)
    assert power == pytest.approx(legacy_power, abs=1e-9)


def test_parity_holds_for_attribute_shaped_bars_too():
    """A bar's shape is a producer detail: IbBar objects and dicts agree."""
    import group_rrs
    from bounce_bot_lib.legacy import real_relative_strength as legacy_rrs

    symbol = _series(_rising(30, base=88.0, step=-0.15))
    spy = _series(_choppy(30), spread=0.25)
    objects = (_as_ib(symbol), _as_ib(spy))

    legacy_value, _ = legacy_rrs(*objects, length=12)
    from_objects, _ = group_rrs.real_relative_strength(*objects, 12)
    from_dicts, _ = group_rrs.real_relative_strength(symbol, spy, 12)

    assert from_objects == pytest.approx(legacy_value, abs=1e-9)
    assert from_dicts == pytest.approx(legacy_value, abs=1e-9)


def test_the_atr_matches_legacy_including_its_whole_series_dependence():
    """The ATR seeds on the first `length` ranges and smooths over the REST,
    so it depends on every bar handed in - not just the last `length + 1`."""
    import group_rrs
    from bounce_bot_lib.legacy import _wilder_atr_last

    bars = _series(_choppy(40))
    short = bars[-14:]

    assert group_rrs.wilder_atr_last(bars, 12) == pytest.approx(
        _wilder_atr_last(_as_ib(bars), 12), abs=1e-12
    )
    assert group_rrs.wilder_atr_last(short, 12) == pytest.approx(
        _wilder_atr_last(_as_ib(short), 12), abs=1e-12
    )
    assert group_rrs.wilder_atr_last(bars, 12) != pytest.approx(
        group_rrs.wilder_atr_last(short, 12), abs=1e-6
    ), "if these agreed the smoothing loop would be untested"


def test_the_sector_map_has_not_drifted_from_the_one_bouncebot_uses():
    """Copied, not imported (the tape must survive BounceBot being off) - so
    the copy is pinned rather than trusted."""
    import group_rrs
    from bounce_bot_lib.legacy import DEFAULT_SECTOR_ETF_MAP

    assert group_rrs.SECTOR_ETFS == DEFAULT_SECTOR_ETF_MAP
    assert len(set(group_rrs.SECTOR_ETFS.values())) == 11


# ------------------------------------------------------------ session filters


def test_the_forming_bar_is_excluded():
    """plan.md sec 5: a forming bar is preview. Including it would reshuffle
    the tape every few seconds against moves that had not finished."""
    import group_rrs

    bars = _series(_rising(10))
    # 09:05 is inside the bar that started at 09:05 and ends 09:10.
    now = SESSION_OPEN + timedelta(minutes=5 * 9 + 3)
    kept = group_rrs.session_bars(bars, now=now)

    assert len(kept) == 9, "the ninth bar closed; the tenth is still forming"
    assert kept[-1]["dt"] == SESSION_OPEN + timedelta(minutes=40)


def test_a_bar_that_just_closed_is_kept():
    """Inclusive at the boundary - `completed_bars`' rule. A strict `<` would
    discard the single most important bar on a 5-minute engine."""
    import group_rrs

    bars = _series(_rising(3))
    now = SESSION_OPEN + timedelta(minutes=15)
    assert len(group_rrs.session_bars(bars, now=now)) == 3


def test_a_gap_straddling_series_never_reaches_into_yesterday():
    """The whole reason for the rebuild. 06:36 read XLK +10.5 because a
    12-bar window on a 5-day fetch spanned the overnight gap."""
    import group_rrs

    yesterday = _series(
        _rising(30, base=100.0), start=SESSION_OPEN - timedelta(days=1)
    )
    # Today gapped down 8 points and has only four completed bars so far.
    today = _series(_rising(4, base=92.0), start=SESSION_OPEN)
    now = SESSION_OPEN + timedelta(minutes=20)

    kept = group_rrs.session_bars(yesterday + today, now=now)
    assert len(kept) == 4, "yesterday is a different session, not more history"
    assert all(bar["dt"].date() == SESSION_OPEN.date() for bar in kept)

    spy = group_rrs.session_bars(
        _series(_choppy(30), start=SESSION_OPEN - timedelta(days=1))
        + _series(_choppy(4), start=SESSION_OPEN),
        now=now,
    )
    # Four bars cannot answer any window, and the gap is not allowed to
    # manufacture one.
    assert group_rrs.rrs_windows(yesterday + today, spy, now=now) == {
        "30": None,
        "60": None,
        "90": None,
    }


def test_unequal_timestamps_are_aligned_rather_than_misread():
    """An ETF missing one bar must not have its move measured over a longer
    span than SPY's and read as strength."""
    import group_rrs

    spy = _series(_choppy(20), spread=0.25)
    symbol = _series(_rising(20))
    holed = [bar for index, bar in enumerate(symbol) if index != 5]

    now = SESSION_OPEN + timedelta(minutes=5 * 20)
    aligned_symbol, aligned_spy = group_rrs.align_bars(holed, spy, now=now)

    assert len(aligned_symbol) == len(aligned_spy) == 19
    assert [bar["dt"] for bar in aligned_symbol] == [bar["dt"] for bar in aligned_spy]
    assert all(bar["dt"] != symbol[5]["dt"] for bar in aligned_spy), "SPY dropped it too"


def test_alignment_survives_the_two_sides_carrying_different_zones():
    """Yahoo stamps intraday bars with a zone. Two series normalized
    differently would share no keys at all and the tape would go blank."""
    import group_rrs

    eastern = timezone(timedelta(hours=-4))
    pacific = timezone(timedelta(hours=-7))
    closes = _rising(20)
    symbol = _series(closes, start=datetime(2026, 8, 27, 9, 30, tzinfo=eastern))
    spy = _series(closes, start=datetime(2026, 8, 27, 6, 30, tzinfo=pacific))

    now = datetime(2026, 8, 27, 8, 30, tzinfo=pacific)
    aligned_symbol, aligned_spy = group_rrs.align_bars(symbol, spy, now=now)
    assert len(aligned_symbol) == len(aligned_spy) == 20


# ------------------------------------------------------------------- windows


def test_each_window_answers_only_once_it_has_its_own_bars():
    """UNKNOWN, never invented: 6/12/18 bars PLUS the ATR warm-up (+2)."""
    import group_rrs

    assert group_rrs.minimum_bars_for("30") == 8
    assert group_rrs.minimum_bars_for("60") == 14
    assert group_rrs.minimum_bars_for("90") == 20

    closes_symbol = _rising(30)
    closes_spy = _choppy(30)
    seen: dict[str, int] = {}
    for count in range(1, 31):
        now = SESSION_OPEN + timedelta(minutes=5 * count)
        windows = group_rrs.rrs_windows(
            _series(closes_symbol[:count]),
            _series(closes_spy[:count], spread=0.25),
            now=now,
        )
        for label, value in windows.items():
            if value is not None:
                seen.setdefault(label, count)

    assert seen == {"30": 8, "60": 14, "90": 20}


def test_a_short_window_is_none_and_never_a_zero():
    """0.0 on a tape reads as "exactly in line with SPY" - a claim. A blank
    reads as "no answer yet", which is the truth."""
    import group_rrs

    now = SESSION_OPEN + timedelta(minutes=30)
    windows = group_rrs.rrs_windows(
        _series(_rising(6)), _series(_choppy(6), spread=0.25), now=now
    )
    assert windows == {"30": None, "60": None, "90": None}
    assert not any(value == 0.0 for value in windows.values())


def test_the_three_windows_describe_the_same_bars():
    import group_rrs

    now = SESSION_OPEN + timedelta(minutes=5 * 30)
    symbol = _series(_rising(30))
    spy = _series(_choppy(30), spread=0.25)

    windows = group_rrs.rrs_windows(symbol, spy, now=now)
    for label, length in group_rrs.RRS_WINDOWS.items():
        assert windows[label] == pytest.approx(
            group_rrs.session_rrs(symbol, spy, now=now, length=length), abs=1e-12
        )
    assert group_rrs.WINDOW_ORDER == ("90", "60", "30")


def test_a_flat_series_has_no_atr_and_so_no_answer():
    """A zero ATR is the division that would produce an infinity; legacy
    returns None for it and so does this."""
    import group_rrs

    flat = [
        {"dt": SESSION_OPEN + timedelta(minutes=5 * i), "open": 10.0, "high": 10.0,
         "low": 10.0, "close": 10.0, "volume": 1.0}
        for i in range(30)
    ]
    now = SESSION_OPEN + timedelta(minutes=5 * 30)
    assert group_rrs.wilder_atr_last(flat, 12) is None
    assert group_rrs.rrs_windows(flat, flat, now=now) == {
        "30": None,
        "60": None,
        "90": None,
    }


def test_missing_bars_are_uncertainty_rather_than_a_number():
    import group_rrs

    now = SESSION_OPEN + timedelta(minutes=5 * 30)
    spy = _series(_choppy(30), spread=0.25)
    assert group_rrs.rrs_windows([], spy, now=now) == {"30": None, "60": None, "90": None}
    assert group_rrs.rrs_windows(spy, [], now=now) == {"30": None, "60": None, "90": None}
    assert group_rrs.session_rrs([], [], now=now, length=12) is None
