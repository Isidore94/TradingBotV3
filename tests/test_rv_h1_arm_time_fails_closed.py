"""The arm-time fence fails CLOSED on an event it cannot date (lead ruling, 2026-09-13).

The independent reviewer of `claude/rv-h1-arm-time` found that
`chart_watch.h1_event_is_post_arm` answered True - "post-arm, nothing fenced" -
when the event bar or the arm time was not a datetime. Missing data is
uncertainty, never confirmation (plan.md section 5), so an undatable event is
NOT the trader's: the watch stays armed and answers on the next bar it can
date. Unreachable today (`armed_at` is a required field of `ChartWatch` and
the frozen rule stamps `confirm_bar_dt` on every fire and invalidation), which
is exactly why it needs a pin rather than a live proof.
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
    H1_EMA_BOUNCE_KIND,
    ChartWatch,
    h1_event_is_post_arm,
)

ARMED_AT = datetime(2026, 8, 26, 9, 0)


def _watch(armed_at) -> ChartWatch:
    return ChartWatch(
        symbol="AAPL",
        kind=H1_EMA_BOUNCE_KIND,
        side="LONG",
        armed_at=armed_at,
        watch_id="rv-fails-closed",
    )


def test_a_datable_post_arm_event_is_still_the_traders():
    assert h1_event_is_post_arm(_watch(ARMED_AT), ARMED_AT + timedelta(hours=2)) is True


def test_an_event_bar_without_a_date_is_not_the_traders():
    assert h1_event_is_post_arm(_watch(ARMED_AT), None) is False
    assert h1_event_is_post_arm(_watch(ARMED_AT), "2026-08-26T11:30:00") is False


def test_an_arm_without_a_date_fences_every_event():
    assert h1_event_is_post_arm(_watch(None), ARMED_AT + timedelta(hours=2)) is False
