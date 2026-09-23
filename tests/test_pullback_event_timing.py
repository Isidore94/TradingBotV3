"""AR-1 timing contracts for the M30 SMA / M15 LRSI companion leg.

The price math is deliberately inherited from PCT-1's frozen old-code tape.  These
tests only name when a completed cross may speak and the identity it must retain.
"""

from __future__ import annotations

import sys
from datetime import timedelta
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
    M30_LONG_CLOSES,
    M30_RECLAIM_INDEX,
    M30_SMA,
    TRIGGER_THEN_LRSI,
    bar_dt,
    bar_end,
    cross_up_80,
    lrsi_values,
    make_bars,
    mirror,
    one_fire,
)


def _side_bars(side: str, closes, minutes: int):
    bars = make_bars(closes, minutes)
    return mirror(bars) if side == "short" else bars


@pytest.mark.parametrize("side", ("long", "short"))
def test_a_companion_cross_uses_its_own_end_for_the_arm_fence(side):
    """A M15 reversal completed before arming cannot be made post-arm by M30."""
    from indicators.pullback_sma_reclaim import evaluate

    # The M15 cross ends at 08:15.  The M30 cache has subsequently completed
    # its 08:00--08:30 bar, which is intentionally irrelevant to the arm fence.
    companion = _side_bars(side, M15_LONG_CLOSES[: M15_RECLAIM_INDEX + 1], 15)
    primary = _side_bars(side, M30_LONG_CLOSES[:108], 30)
    armed_at = bar_end(M15_RECLAIM_INDEX, 15) + timedelta(minutes=1)

    result = evaluate(
        primary,
        side=side,
        sma_length=M30_SMA,
        bar_minutes=30,
        armed_at=armed_at,
        now=bar_end(107, 30),
        companion_bars=companion,
        companion_minutes=15,
        d1_atr=OPEN_GATE_D1_ATR,
    )

    assert result is not None
    assert [fire for fire in result.fired if fire.trigger == TRIGGER_THEN_LRSI] == []


@pytest.mark.parametrize("side", ("long", "short"))
def test_a_post_arm_m15_cross_keeps_its_cross_identity_beside_the_m30_sma(side):
    """The M30 alert names both its M30 SMA reading and the M15 cross that fired it."""
    from indicators.pullback_sma_reclaim import evaluate

    # The native M30 80-cross ended before this arm.  The M15 cross ends after
    # it, so the companion is the earliest eligible event.  These facts are
    # fixed by the old PCT-1 prices, before the rule sees a bar.
    primary_closes = M30_LONG_CLOSES[:108]
    primary = _side_bars(side, primary_closes, 30)
    assert cross_up_80([bar["close"] for bar in primary], side=side) == (
        M30_RECLAIM_INDEX,
    )
    assert bar_end(M30_RECLAIM_INDEX, 30) < bar_end(M15_RECLAIM_INDEX, 15)
    armed_at = bar_end(M30_RECLAIM_INDEX, 30) + timedelta(minutes=1)
    companion = _side_bars(side, M15_LONG_CLOSES[: M15_RECLAIM_INDEX + 1], 15)

    result = evaluate(
        primary,
        side=side,
        sma_length=M30_SMA,
        bar_minutes=30,
        armed_at=armed_at,
        now=bar_end(107, 30),
        companion_bars=companion,
        companion_minutes=15,
        d1_atr=OPEN_GATE_D1_ATR,
    )

    assert result is not None
    fire = one_fire(result, TRIGGER_THEN_LRSI)
    # M30 remains the SMA leg.  The source event is separately truthful and is
    # the durable identity used to prevent a restart re-announcement.
    assert fire.timeframe == "M30"
    assert fire.cross_timeframe == "M15"
    assert fire.cross_bar_dt == bar_dt(M15_RECLAIM_INDEX, 15)
    assert fire.cross_lrsi == pytest.approx(
        lrsi_values([bar["close"] for bar in companion], side=side)[M15_RECLAIM_INDEX],
        abs=1e-9,
    )
    assert fire.sma_bar_dt == bar_dt(107, 30)


def test_an_old_m30_stamp_does_not_reannounce_the_earlier_companion_cross_on_upgrade():
    """Old watches stored the latest M30 start; the corrected M15 identity is older."""
    from chart_watch import ChartWatch
    from ui.panels.alert_center_panel import AlertCenterPanel

    class _Cache:
        def __init__(self, bars):
            self._bars = bars

        def bars_for(self, _symbol):
            return list(self._bars)

    watch = ChartWatch(
        symbol="NVDA",
        kind="pullback",
        armed_at=bar_end(M30_RECLAIM_INDEX, 30) + timedelta(minutes=1),
        side="LONG",
        watch_id="ar1-old-stamp",
        triggers=(TRIGGER_THEN_LRSI,),
    )
    # This is how the old code persisted a first companion alert: a later M30
    # cache bar, rather than the actual M15 event.  A still newer M30 bar is
    # now present after restart, so equality-only de-duplication re-alerts it.
    old_m30_stamp = bar_dt(108, 30).isoformat()
    panel = AlertCenterPanel.__new__(AlertCenterPanel)
    result = panel._evaluate_one_pullback_job(
        {
            "watch": watch,
            "identity": watch.watch_id,
            "triggers": watch.triggers,
            "due": [(30, M30_SMA, None)],
            "caches": {
                30: _Cache(_side_bars("long", M30_LONG_CLOSES[:110], 30)),
                15: _Cache(
                    _side_bars("long", M15_LONG_CLOSES[: M15_RECLAIM_INDEX + 1], 15)
                ),
            },
            "marks": {
                AlertCenterPanel.pullback_fire_key(TRIGGER_THEN_LRSI, "M30"): old_m30_stamp
            },
            "states": {},
        },
        bar_end(109, 30),
    )

    assert result["fires"] == []
