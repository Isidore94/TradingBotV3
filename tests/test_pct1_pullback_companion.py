"""PCT-1 - the M15 leg of ``reclaim_then_lrsi``. ADDED by the builder.

The tester left this leg untested on purpose and said so: the packet's
``evaluate()`` signature named one bar series, so the other timeframe's
crosses could not be handed to the pure rule without inventing an argument.
The lead named it (ruling 1, 2026-09-15):

    ``evaluate(...)`` gains keyword-only ``companion_bars=None,
    companion_minutes=None`` (the M15 series handed to the M30 evaluation); an
    LRSI 80 cross on the companion series after the M30 reclaim bar, while
    every completed M30 close holds the SMA, fires ``reclaim_then_lrsi`` with
    ``timeframe="M30"`` and a note naming the companion.

The fixtures are the tester's own, re-used rather than re-written, so these
two tests cannot drift away from the ones beside them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from test_pct1_pullback_alert import (  # noqa: E402
    OPEN_GATE_D1_ATR,
    M15_LONG_CLOSES,
    M15_RECLAIM_INDEX,
    M30_HOLD_CLOSES,
    M30_HOLD_RECLAIM_INDEX,
    M30_SMA,
    TRIGGER_RECLAIM,
    TRIGGER_THEN_LRSI,
    bar_dt,
    bar_end,
    cross_up_80,
    make_bars,
    one_fire,
    simple_mean,
    triggers_of,
)

#: The tester's M30 hold with the M30's OWN 80-cross cut off - their fixture
#: crosses at bar 105, so this series stops at 104. The M30 has reclaimed and
#: every completed close since holds SMA-75; the only oscillator reversal
#: available is the M15 companion's, which is exactly the case the trader
#: described: "a breakup then waiting for an M30 or M15 LRSI reversal while
#: staying above teh relevant SMA".
M30_COMPANION_LAST_INDEX = 104


def _companion_case(*, with_companion: bool):
    # The rule directly, not the tester's one-series wrapper: the companion
    # arguments are the whole point of these two tests.
    from indicators.pullback_sma_reclaim import evaluate

    m30_bars = make_bars(M30_HOLD_CLOSES[: M30_COMPANION_LAST_INDEX + 1], 30)
    m15_bars = make_bars(M15_LONG_CLOSES[: M15_RECLAIM_INDEX + 1], 15)
    # `now` is the moment the M15 cross bar finishes. The M30 bar that would
    # have carried the M30's own cross is not in this series at all.
    return evaluate(
        m30_bars,
        side="long",
        sma_length=M30_SMA,
        bar_minutes=30,
        armed_at=bar_dt(0, 30),
        now=bar_end(M15_RECLAIM_INDEX, 15),
        companion_bars=m15_bars if with_companion else None,
        companion_minutes=15 if with_companion else None,
        d1_atr=OPEN_GATE_D1_ATR,
    )


def test_an_m15_cross_answers_the_m30_hold_and_the_note_names_the_companion():
    # The premises, from the fixtures' own arithmetic, before the rule reads
    # a single bar of them.
    m30_closes = M30_HOLD_CLOSES[: M30_COMPANION_LAST_INDEX + 1]
    assert cross_up_80(m30_closes) == ()  # nothing on the M30 itself
    for index in range(M30_HOLD_RECLAIM_INDEX, M30_COMPANION_LAST_INDEX + 1):
        assert m30_closes[index] > simple_mean(m30_closes, index, M30_SMA)
    assert cross_up_80(M15_LONG_CLOSES[: M15_RECLAIM_INDEX + 1]) == (
        M15_RECLAIM_INDEX,
    )
    assert bar_end(M15_RECLAIM_INDEX, 15) > bar_end(M30_HOLD_RECLAIM_INDEX, 30)

    result = _companion_case(with_companion=True)

    assert result is not None
    fire = one_fire(result, TRIGGER_THEN_LRSI)
    # The SMA is the M30's, so the FIRE is the M30's; its event time is the
    # later M15 cross, while the M30 bar remains named as SMA evidence.
    assert fire.timeframe == "M30"
    assert fire.bar_dt == bar_dt(M15_RECLAIM_INDEX, 15)
    assert fire.cross_timeframe == "M15"
    assert fire.cross_bar_dt == fire.bar_dt
    assert fire.sma_bar_dt == bar_dt(M30_COMPANION_LAST_INDEX, 30)
    assert fire.sma == pytest.approx(
        simple_mean(m30_closes, M30_COMPANION_LAST_INDEX, M30_SMA), abs=1e-9
    )
    assert "M15" in str(fire.message), fire.message
    # The reclaim trigger stays silent: its window is the reclaim bar and the
    # two before it, and nothing crossed there.
    assert TRIGGER_RECLAIM not in triggers_of(result)


def test_without_the_companion_series_the_m30_hold_says_nothing():
    """Absence of the companion is silence, never a fire on a cross the M30
    series does not contain."""
    result = _companion_case(with_companion=False)

    assert result is not None
    assert triggers_of(result) == []
