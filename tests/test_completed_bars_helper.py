"""R5 section 5: ONE completed-bars definition, shared.

Before this, the intraday "is this bar finished?" rule existed twice:

- `weekend_strength._completed_intraday` -- correct, and the one with real
  coverage. It normalizes a tz-aware stamp with `astimezone(...)`.
- BounceBot's ad hoc idiom at `legacy.py:4384-4386` and `4533-4535` --
  `cutoff = get_market_local_now().replace(tzinfo=None)`, then compare. That
  spelling is wrong for a tz-aware stamp: `replace(tzinfo=None)` DISCARDS the
  offset instead of converting through it, so a bar stamped in a different zone
  is judged against a wall-clock number that never meant the same instant.

R5 adds engines that need the rule again, so it is extracted once into
`scripts/completed_bars.py` and `weekend_strength` now delegates to it.

The first half of this file is a CHARACTERIZATION of `weekend_strength`: shipped
R8 code whose behaviour must not move. It was verified to fail against a
deliberately broken extraction before the real one went in.
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

from completed_bars import completed_intraday_bars  # noqa: E402

NOW = datetime(2026, 8, 17, 10, 32)  # naive, market-local, mid-session


def _bar(stamp, **extra):
    row = {"dt": stamp}
    row.update(extra)
    return row


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------
def test_a_finished_bar_is_kept():
    bar = _bar(datetime(2026, 8, 17, 10, 25))  # 10:25 + 5m = 10:30 <= 10:32
    assert completed_intraday_bars([bar], 5, now=NOW) == [bar]


def test_a_still_forming_bar_is_dropped():
    bar = _bar(datetime(2026, 8, 17, 10, 30))  # 10:30 + 5m = 10:35 > 10:32
    assert completed_intraday_bars([bar], 5, now=NOW) == []


def test_the_boundary_instant_counts_as_complete():
    """`bar_start + bar_minutes <= now`, inclusive. Mutation check: flipping to
    a strict `<` silently discards the bar that JUST closed - which on a 5-minute
    engine is the single most important bar there is."""
    bar = _bar(NOW - timedelta(minutes=5))
    assert completed_intraday_bars([bar], 5, now=NOW) == [bar]


def test_one_second_short_is_not_complete():
    bar = _bar(NOW - timedelta(minutes=5) + timedelta(seconds=1))
    assert completed_intraday_bars([bar], 5, now=NOW) == []


def test_the_bar_length_is_honoured():
    """Mutation check: a 60-minute bar starting 10 minutes ago is NOT done,
    while a 5-minute one is."""
    bar = _bar(NOW - timedelta(minutes=10))
    assert completed_intraday_bars([bar], 5, now=NOW) == [bar]
    assert completed_intraday_bars([bar], 60, now=NOW) == []


def test_a_bar_with_no_readable_timestamp_is_dropped():
    """Missing data is uncertainty, never confirmation - an undateable bar
    cannot be shown to be complete, so it does not trigger anything."""
    assert completed_intraday_bars([{"dt": None}, {"close": 1.0}], 5, now=NOW) == []


def test_order_is_preserved():
    bars = [_bar(NOW - timedelta(minutes=n)) for n in (20, 15, 10)]
    assert completed_intraday_bars(bars, 5, now=NOW) == bars


# --------------------------------------------------------------------------
# timezone handling - the reason this is one definition and not two
# --------------------------------------------------------------------------
def test_a_tz_aware_stamp_is_converted_not_stripped():
    """The defect the shared helper exists to prevent.

    A bar stamped 17:30 UTC is 10:30 PT. Against a naive 10:32 PT `now` it is
    still forming (10:30 + 5m = 10:35). `replace(tzinfo=None)` would compare
    17:30 to 10:32 and call it long finished."""
    utc_stamp = datetime(2026, 8, 17, 17, 30, tzinfo=timezone.utc)
    local = utc_stamp.astimezone().replace(tzinfo=None)
    bar = _bar(utc_stamp)
    kept = completed_intraday_bars([bar], 5, now=local + timedelta(minutes=2))
    assert kept == []


def test_a_tz_aware_stamp_that_really_is_finished_is_kept():
    """The other direction, so the fix is not just 'drop everything aware'."""
    utc_stamp = datetime(2026, 8, 17, 17, 30, tzinfo=timezone.utc)
    local = utc_stamp.astimezone().replace(tzinfo=None)
    bar = _bar(utc_stamp)
    assert completed_intraday_bars([bar], 5, now=local + timedelta(minutes=30)) == [bar]


def test_an_aware_now_with_naive_bars_is_handled():
    aware_now = NOW.replace(tzinfo=timezone.utc)
    bar = _bar(NOW - timedelta(minutes=10))
    assert completed_intraday_bars([bar], 5, now=aware_now) == [bar]


def test_the_helper_never_uses_the_replace_spelling():
    """A source-level guard. `replace(tzinfo=None)` on a BAR stamp is the exact
    defect; this asserts the helper does not reintroduce it."""
    import completed_bars

    source = Path(completed_bars.__file__).read_text(encoding="utf-8")
    body = source.split("def completed_intraday_bars", 1)[1]
    code = "\n".join(
        line for line in body.splitlines() if not line.strip().startswith("#")
    )
    assert "replace(tzinfo=None)" not in code


# --------------------------------------------------------------------------
# characterization: weekend_strength must not move
# --------------------------------------------------------------------------
_WEEKEND_NOW = datetime(2026, 8, 17, 11, 0)

#: (label, bar start times, how many survive). Expectations are FROZEN NUMBERS,
#: not a second call to the helper: `weekend_strength._completed_intraday` now
#: delegates to the helper, so comparing the two would be a tautology that a
#: mutation changes on both sides at once. It has to be pinned from outside.
_WEEKEND_CASES = [
    ("finished an hour ago", [datetime(2026, 8, 17, 9, 30)], 1),
    ("exactly finished", [datetime(2026, 8, 17, 10, 0)], 1),
    ("one minute short", [datetime(2026, 8, 17, 10, 1)], 0),
    ("still forming", [datetime(2026, 8, 17, 10, 30)], 0),
    ("undateable", [None], 0),
    ("empty", [], 0),
    (
        "mixed",
        [
            datetime(2026, 8, 17, 8, 0),
            datetime(2026, 8, 17, 9, 0),
            datetime(2026, 8, 17, 10, 45),
        ],
        2,
    ),
]


@pytest.mark.parametrize("label,stamps,survivors", _WEEKEND_CASES)
def test_weekend_strength_intraday_behaviour_is_unchanged(label, stamps, survivors):
    """The extraction must be behaviour-preserving on shipped R8 code.

    Verified: mutating the helper's boundary from `<=` to `<` turns the
    "exactly finished" row red here, and the pre-existing
    `test_weekend_strength.py::test_an_hourly_bar_is_complete_sixty_minutes_after_it_opened`
    red alongside it."""
    from weekend_strength import H1, completed_bars as weekend_completed

    bars = [_bar(stamp) for stamp in stamps]
    assert len(weekend_completed(H1, bars, now=_WEEKEND_NOW)) == survivors


def test_a_tz_aware_bar_still_reaches_the_weekend_board():
    """Through the real weekend_strength seam, not the helper directly: a UTC
    stamp for a long-finished hourly bar must survive the conversion."""
    from weekend_strength import H1, completed_bars as weekend_completed

    bars = [_bar(datetime(2026, 8, 17, 16, 30, tzinfo=timezone.utc))]
    kept = weekend_completed(H1, bars, now=_WEEKEND_NOW)
    local = datetime(2026, 8, 17, 16, 30, tzinfo=timezone.utc).astimezone().replace(tzinfo=None)
    assert len(kept) == (1 if local + timedelta(minutes=60) <= _WEEKEND_NOW else 0)


def test_weekend_strength_session_and_month_rules_are_untouched():
    """Only the INTRADAY rule is shared. D1 uses the NYSE calendar's last
    completed session and M1 uses month identity; neither is this helper's
    business, and folding them in would have been a real behaviour change."""
    from weekend_strength import D1, M1, completed_bars as weekend_completed

    now = datetime(2026, 8, 17, 11, 0)
    today = [_bar(datetime(2026, 8, 17))]
    assert weekend_completed(D1, today, now=now) == []
    this_month = [_bar(datetime(2026, 8, 1))]
    assert weekend_completed(M1, this_month, now=now) == []
