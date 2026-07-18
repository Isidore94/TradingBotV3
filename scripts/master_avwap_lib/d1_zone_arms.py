"""Pure builder for the D1 band-zone M5 "zone arms".

The master scan classifies every scanned symbol by where its last daily close
sits inside the current-earnings anchored-VWAP band ladder, then arms a small,
rubric-driven set of M5 trigger levels. The bounce bot watches those levels on
every M5 cycle and, on a two-bar confirmation, fires a D1 Focus alert the moment
a tracked setup actually triggers.

Rubric (long side; the short side mirrors the bands and inverts the direction):

  Zone 1  AVWAPE <= close < UPPER_1
      - bounce off AVWAPE                    (reclaim support)
      - break of UPPER_1                     (1st-dev breakout)
      - bounce off the prior-anchor UPPER_1  (only when it sits below UPPER_1)
  Zone 2  UPPER_1 <= close < UPPER_2
      - break of UPPER_2                     (2nd-dev breakout)
      - bounce off UPPER_1 / 15EMA / 21EMA
  Zone 3  UPPER_2 <= close < UPPER_3 for >= 2 sessions   (the critical pullback)
      - bounce off UPPER_1 / 15EMA / 21EMA   (no breakout arm)

Decision-support only. Pure + fully unit-tested: no Qt, no I/O, no network. See
plan.md Milestone 3 (golden fixtures before detector wiring).
"""

from __future__ import annotations

import math
from typing import Any


ZONE_ARM_SCHEMA_VERSION = 1
ZONE_ARM_SOURCE = "zone_arm"

# How close price must come to a level to "tag" it for a bounce. The bot confirms
# the reclaim on the next completed bar (two-bar rule), so this only sets the tag
# radius. Expressed as a fraction of ATR(20); falls back to a fraction of the band
# stdev, then a tiny fraction of price when neither is available.
ZONE_ARM_BOUNCE_TOL_ATR = 0.20
ZONE_ARM_BOUNCE_TOL_STDEV = 0.20
ZONE_ARM_BOUNCE_TOL_PRICE = 0.001


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _append_arm(
    arms: list[dict],
    seen: set,
    *,
    side: str,
    zone: int,
    label: str,
    level: float | None,
    action: str,
    event_type: str,
    alert_label: str,
    reason: str,
    close: float,
    tolerance: float,
    critical: bool = False,
) -> None:
    level = _finite(level)
    if level is None or level <= 0:
        return
    # Still-armed gating: never arm a level price has already passed the wrong way.
    # A break needs room to break; a bounce needs price on the reclaim side of the
    # level it might pull back to.
    if action == "break_above" and not close < level:
        return
    if action == "break_below" and not close > level:
        return
    if action == "bounce_up" and not close > level:
        return
    if action == "bounce_down" and not close < level:
        return

    rounded = round(level, 4)
    key = (event_type, label, rounded)
    if key in seen:
        return
    seen.add(key)
    arms.append(
        {
            "schema_version": ZONE_ARM_SCHEMA_VERSION,
            "trigger_id": f"{event_type}:{label}:{rounded:.4f}",
            "side": side,
            "zone": zone,
            "action": action,
            "event_type": event_type,
            "label": label,
            "alert_label": alert_label,
            "level": rounded,
            "tolerance": round(float(tolerance), 4) if tolerance is not None else 0.0,
            "reason": reason,
            "source": ZONE_ARM_SOURCE,
            "critical": bool(critical),
            "armed_price": round(float(close), 4),
        }
    )


def build_d1_zone_arms(
    *,
    symbol: Any,
    close: Any,
    avwape: Any,
    upper_1: Any = None,
    upper_2: Any = None,
    upper_3: Any = None,
    lower_1: Any = None,
    lower_2: Any = None,
    lower_3: Any = None,
    ema15: Any = None,
    ema21: Any = None,
    prev_upper_1: Any = None,
    prev_lower_1: Any = None,
    stdev: Any = None,
    atr: Any = None,
    sustained_2nd_3rd: bool = False,
    anchor_date: str = "",
    armed_at: str = "",
) -> dict | None:
    """Return the per-symbol zone-arm entry, or ``None`` when nothing is armed.

    Side and zone are decided purely by where ``close`` sits versus the current
    anchor bands, independent of any watchlist side. ``sustained_2nd_3rd`` should
    be the caller's ``closes_between_bands`` result for the side price is on (the
    zone-3 pullback rubric only applies once the name has held the 2nd->3rd band
    for at least two sessions).
    """
    sym = str(symbol or "").strip().upper()
    close = _finite(close)
    avwape = _finite(avwape)
    if not sym or close is None or avwape is None:
        return None

    u1, u2, u3 = _finite(upper_1), _finite(upper_2), _finite(upper_3)
    l1, l2, l3 = _finite(lower_1), _finite(lower_2), _finite(lower_3)
    e15, e21 = _finite(ema15), _finite(ema21)
    pu1, pl1 = _finite(prev_upper_1), _finite(prev_lower_1)
    stdev_v, atr_v = _finite(stdev), _finite(atr)

    if atr_v is not None and atr_v > 0:
        tol = ZONE_ARM_BOUNCE_TOL_ATR * atr_v
    elif stdev_v is not None and stdev_v > 0:
        tol = ZONE_ARM_BOUNCE_TOL_STDEV * stdev_v
    else:
        tol = abs(close) * ZONE_ARM_BOUNCE_TOL_PRICE

    side: str | None = None
    zone: int | None = None
    zone_label = ""

    if close >= avwape:
        side = "LONG"
        if u1 is not None and avwape <= close < u1:
            zone, zone_label = 1, "AVWAPE_TO_UPPER_1"
        elif u1 is not None and u2 is not None and u1 <= close < u2:
            zone, zone_label = 2, "UPPER_1_TO_UPPER_2"
        elif u2 is not None and u3 is not None and u2 <= close < u3 and sustained_2nd_3rd:
            zone, zone_label = 3, "UPPER_2_TO_UPPER_3"
    else:
        side = "SHORT"
        if l1 is not None and l1 < close <= avwape:
            zone, zone_label = 1, "LOWER_1_TO_AVWAPE"
        elif l1 is not None and l2 is not None and l2 < close <= l1:
            zone, zone_label = 2, "LOWER_2_TO_LOWER_1"
        elif l2 is not None and l3 is not None and l3 < close <= l2 and sustained_2nd_3rd:
            zone, zone_label = 3, "LOWER_3_TO_LOWER_2"

    if zone is None:
        return None

    arms: list[dict] = []
    seen: set = set()

    if side == "LONG":
        if zone == 1:
            _append_arm(
                arms, seen, side=side, zone=1, label="AVWAPE", level=avwape,
                action="bounce_up", event_type="zone_bounce_avwape",
                alert_label="bounce off AVWAPE",
                reason="Zone 1: reclaim AVWAPE support and go.",
                close=close, tolerance=tol,
            )
            _append_arm(
                arms, seen, side=side, zone=1, label="UPPER_1", level=u1,
                action="break_above", event_type="zone_break_1st_dev",
                alert_label="1st-dev break",
                reason="Zone 1: break of UPPER_1 (1st deviation).",
                close=close, tolerance=0.0,
            )
            if pu1 is not None and u1 is not None and pu1 < u1:
                _append_arm(
                    arms, seen, side=side, zone=1, label="PREV_UPPER_1", level=pu1,
                    action="bounce_up", event_type="zone_bounce_prev_1st_dev",
                    alert_label="bounce off prior 1st-dev",
                    reason="Zone 1: prior-anchor UPPER_1 sits below the current UPPER_1; reclaim + go.",
                    close=close, tolerance=tol,
                )
        elif zone == 2:
            _append_arm(
                arms, seen, side=side, zone=2, label="UPPER_2", level=u2,
                action="break_above", event_type="zone_break_2nd_dev",
                alert_label="2nd-dev break",
                reason="Zone 2: break of UPPER_2 (2nd deviation).",
                close=close, tolerance=0.0,
            )
            _append_arm(
                arms, seen, side=side, zone=2, label="UPPER_1", level=u1,
                action="bounce_up", event_type="zone_bounce_1st_dev",
                alert_label="bounce off 1st-dev",
                reason="Zone 2: reclaim UPPER_1 (1st deviation) on the pullback.",
                close=close, tolerance=tol,
            )
            _append_arm(
                arms, seen, side=side, zone=2, label="EMA_15", level=e15,
                action="bounce_up", event_type="zone_bounce_ema15",
                alert_label="bounce off 15EMA",
                reason="Zone 2: reclaim the daily 15EMA on the pullback.",
                close=close, tolerance=tol,
            )
            _append_arm(
                arms, seen, side=side, zone=2, label="EMA_21", level=e21,
                action="bounce_up", event_type="zone_bounce_ema21",
                alert_label="bounce off 21EMA",
                reason="Zone 2: reclaim the daily 21EMA on the pullback.",
                close=close, tolerance=tol,
            )
        elif zone == 3:
            _append_arm(
                arms, seen, side=side, zone=3, label="UPPER_1", level=u1,
                action="bounce_up", event_type="zone_bounce_1st_dev",
                alert_label="bounce off 1st-dev (2nd-3rd pullback)",
                reason="Zone 3: multi-session 2nd->3rd extension pulling back to UPPER_1.",
                close=close, tolerance=tol, critical=True,
            )
            _append_arm(
                arms, seen, side=side, zone=3, label="EMA_15", level=e15,
                action="bounce_up", event_type="zone_bounce_ema15",
                alert_label="bounce off 15EMA (2nd-3rd pullback)",
                reason="Zone 3: multi-session 2nd->3rd extension pulling back to the daily 15EMA.",
                close=close, tolerance=tol, critical=True,
            )
            _append_arm(
                arms, seen, side=side, zone=3, label="EMA_21", level=e21,
                action="bounce_up", event_type="zone_bounce_ema21",
                alert_label="bounce off 21EMA (2nd-3rd pullback)",
                reason="Zone 3: multi-session 2nd->3rd extension pulling back to the daily 21EMA.",
                close=close, tolerance=tol, critical=True,
            )
    else:  # SHORT mirror
        if zone == 1:
            _append_arm(
                arms, seen, side=side, zone=1, label="AVWAPE", level=avwape,
                action="bounce_down", event_type="zone_bounce_avwape",
                alert_label="reject at AVWAPE",
                reason="Zone 1: reject AVWAPE resistance and go.",
                close=close, tolerance=tol,
            )
            _append_arm(
                arms, seen, side=side, zone=1, label="LOWER_1", level=l1,
                action="break_below", event_type="zone_break_1st_dev",
                alert_label="1st-dev break",
                reason="Zone 1: break of LOWER_1 (1st deviation).",
                close=close, tolerance=0.0,
            )
            if pl1 is not None and l1 is not None and pl1 > l1:
                _append_arm(
                    arms, seen, side=side, zone=1, label="PREV_LOWER_1", level=pl1,
                    action="bounce_down", event_type="zone_bounce_prev_1st_dev",
                    alert_label="reject at prior 1st-dev",
                    reason="Zone 1: prior-anchor LOWER_1 sits above the current LOWER_1; reject + go.",
                    close=close, tolerance=tol,
                )
        elif zone == 2:
            _append_arm(
                arms, seen, side=side, zone=2, label="LOWER_2", level=l2,
                action="break_below", event_type="zone_break_2nd_dev",
                alert_label="2nd-dev break",
                reason="Zone 2: break of LOWER_2 (2nd deviation).",
                close=close, tolerance=0.0,
            )
            _append_arm(
                arms, seen, side=side, zone=2, label="LOWER_1", level=l1,
                action="bounce_down", event_type="zone_bounce_1st_dev",
                alert_label="reject at 1st-dev",
                reason="Zone 2: reject LOWER_1 (1st deviation) on the bounce.",
                close=close, tolerance=tol,
            )
            _append_arm(
                arms, seen, side=side, zone=2, label="EMA_15", level=e15,
                action="bounce_down", event_type="zone_bounce_ema15",
                alert_label="reject at 15EMA",
                reason="Zone 2: reject the daily 15EMA on the bounce.",
                close=close, tolerance=tol,
            )
            _append_arm(
                arms, seen, side=side, zone=2, label="EMA_21", level=e21,
                action="bounce_down", event_type="zone_bounce_ema21",
                alert_label="reject at 21EMA",
                reason="Zone 2: reject the daily 21EMA on the bounce.",
                close=close, tolerance=tol,
            )
        elif zone == 3:
            _append_arm(
                arms, seen, side=side, zone=3, label="LOWER_1", level=l1,
                action="bounce_down", event_type="zone_bounce_1st_dev",
                alert_label="reject at 1st-dev (2nd-3rd pullback)",
                reason="Zone 3: multi-session 2nd->3rd extension bouncing back to LOWER_1.",
                close=close, tolerance=tol, critical=True,
            )
            _append_arm(
                arms, seen, side=side, zone=3, label="EMA_15", level=e15,
                action="bounce_down", event_type="zone_bounce_ema15",
                alert_label="reject at 15EMA (2nd-3rd pullback)",
                reason="Zone 3: multi-session 2nd->3rd extension bouncing back to the daily 15EMA.",
                close=close, tolerance=tol, critical=True,
            )
            _append_arm(
                arms, seen, side=side, zone=3, label="EMA_21", level=e21,
                action="bounce_down", event_type="zone_bounce_ema21",
                alert_label="reject at 21EMA (2nd-3rd pullback)",
                reason="Zone 3: multi-session 2nd->3rd extension bouncing back to the daily 21EMA.",
                close=close, tolerance=tol, critical=True,
            )

    if not arms:
        return None

    return {
        "schema_version": ZONE_ARM_SCHEMA_VERSION,
        "symbol": sym,
        "side": side,
        "zone": zone,
        "zone_label": zone_label,
        "close": round(close, 4),
        "tolerance": round(float(tol), 4),
        "anchor_date": str(anchor_date or ""),
        "armed_at": str(armed_at or ""),
        "active_current_scan": True,
        "trigger_levels": arms,
    }
