"""Pure builder + detector for MASTER_AVWAP_D1_TIER_FLIP alerts.

The D1 Focus contract (user rule 2026-07-09, restated 2026-07-22): the feed
should carry "S or A tier stocks that go from non S or A tier into S or A
tier and all it would take was a small move on the M5". The master scan
already arms exactly those conditions - the per-symbol "A/S upgrade target"
trigger levels in the D1 watchlist store, each a concrete price level whose
crossing the scan judged would upgrade the name toward the Favorite (S) /
Near-Favorite (A) buckets. This module turns those armed conditions into a
tier-flip watch:

- TRANSITION-ONLY: only symbols whose scan-time priority bucket is NOT
  already S/A (favorite_setup / near_favorite_zone) are watched. A name that
  is already great never fires here.
- SMALL MOVE ONLY: a condition arms only when its level sits within
  ``TIER_FLIP_SMALL_MOVE_MAX_ATR`` of the scan-time daily close, measured in
  daily ATR(20) - "all it would take was a small move".
- CLOSE-THROUGH ONLY: the flip confirms when the last COMPLETED M5 bar
  closes through the level with the prior completed close still on the armed
  side. A wick touch is not a flip; a gap that opened already beyond the
  level never fires (the next full scan's bucket-upgrade path owns that).

The flip claim is the scan's own curated ``target_tier`` ("A/S") and is
always presented as predicted - the next completed master scan is the
confirmation, and comparing fired flips against recorded bucket upgrades is
the feature's ground-truth precision measure.

Pure + unit-tested: no Qt, no I/O, no pandas. The caller (BounceBot) owns
gates that need live state: session rvol, learning-state context tier,
per-day caps and the persisted fired-set.
"""

from __future__ import annotations

import math
from typing import Any

TIER_FLIP_SCHEMA_VERSION = 1
TIER_FLIP_SOURCE = "tier_flip"

# "A small move on the M5": the armed level must sit within this many daily
# ATR(20) of the scan-time close. Guess pending live calibration - every
# armed condition records its distance so a week of funnel logs can replay
# alternative radii offline before this default changes (plan.md sec 6).
TIER_FLIP_SMALL_MOVE_MAX_ATR = 0.35
# When no daily ATR is available anywhere, approximate it as a fraction of
# price (a ~2% daily range assumption) so a missing ATR degrades gracefully
# instead of silently disarming the symbol.
TIER_FLIP_FALLBACK_ATR_PRICE_FRACTION = 0.02
# Scarcity is the feature: at most this many flip alerts per day.
TIER_FLIP_MAX_PER_DAY = 8
TIER_FLIP_MAX_PER_SIDE_PER_DAY = 4
# The measured-context chip (evaluate_bounce_quality on the mapped M5
# segment) must grade at least this tier or the flip is suppressed to a log
# line. "B" = neutral-or-better; raise to "A" for near-silence.
TIER_FLIP_MIN_CONTEXT_TIER = "B"
# Armed conditions from a scan older than this many calendar days never
# fire. Yesterday's arms stay valid through the next open (the window before
# the first scan of the day is exactly when flips happen); the allowance
# covers weekends without letting a week-old artifact keep firing.
TIER_FLIP_MAX_RUN_AGE_DAYS = 4

_TIER_ORDER = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4}

# Scan-time buckets that already carry the S/A bot-pick tier
# (_tier_for_priority_bucket in master_avwap_lib): these names are excluded -
# the feed only reports transitions INTO that set.
_ALREADY_S_A_BUCKETS = {"favorite_setup", "near_favorite_zone"}

# Side-aware map from an armed condition's event_type to the M5 bounce_type
# segment its confirmation would trade like, so the learning state's measured
# scalars grade the flip's entry context on the calibrated tier scale. An
# unmapped event type contributes no bounce_type segment: the composite then
# rides the context dimensions alone, which defaults toward the neutral B
# band - conservative, never a fabricated segment match.
_EVENT_BOUNCE_TYPES = {
    "avwape_reclaim": ("dynamic_vwap", "dynamic_vwap"),
    "avwape_breakdown": ("dynamic_vwap", "dynamic_vwap"),
    "avwape_retest_watch": ("dynamic_vwap", "dynamic_vwap"),
    "first_dev_break": ("dynamic_vwap_upper_band", "dynamic_vwap_lower_band"),
    "second_dev_break": ("dynamic_vwap_upper_band", "dynamic_vwap_lower_band"),
    "first_dev_retest_watch": ("dynamic_vwap_upper_band", "dynamic_vwap_lower_band"),
    "mid_earnings_first_dev_retest_watch": ("dynamic_vwap_upper_band", "dynamic_vwap_lower_band"),
    "ema15_retest_watch": ("ema_15", "ema_15"),
    "mid_earnings_ema15_retest_watch": ("ema_15", "ema_15"),
    "previous_day_high_break": ("prev_day_high", "prev_day_high"),
    "previous_day_low_break": ("prev_day_low", "prev_day_low"),
}


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def bounce_types_for_flip_event(event_type: Any, side: Any) -> list[str]:
    """M5 bounce_type segment(s) an armed condition's confirmation maps to."""
    key = str(event_type or "").strip().lower()
    pair = _EVENT_BOUNCE_TYPES.get(key)
    if not pair:
        return []
    long_type, short_type = pair
    chosen = short_type if str(side or "").strip().upper() == "SHORT" else long_type
    return [chosen] if chosen else []


def context_tier_passes(tier: Any, minimum: str = TIER_FLIP_MIN_CONTEXT_TIER) -> bool:
    rank = _TIER_ORDER.get(str(tier or "").strip().upper())
    floor = _TIER_ORDER.get(str(minimum or "").strip().upper())
    if rank is None or floor is None:
        return False
    return rank <= floor


def build_tier_flip_watch_entry(
    entry: Any,
    *,
    atr: Any = None,
    max_distance_atr: float = TIER_FLIP_SMALL_MOVE_MAX_ATR,
) -> dict | None:
    """The tier-flip watch for one D1 watchlist entry, or ``None``.

    ``entry`` is a per-symbol dict from the D1 watchlist store (the bot's
    ``master_avwap_d1_watchlist`` map). ``atr`` is the caller's daily ATR(20)
    fallback for when the entry's own ``atr20`` field is absent (the live
    store currently writes it as null).
    """
    if not isinstance(entry, dict):
        return None
    symbol = str(entry.get("symbol") or "").strip().upper()
    side = str(entry.get("side") or "").strip().upper()
    if not symbol or side not in {"LONG", "SHORT"}:
        return None
    if not entry.get("active_current_scan"):
        return None

    bucket = str(entry.get("priority_bucket") or "").strip().lower()
    if bucket in _ALREADY_S_A_BUCKETS:
        return None  # transition-only: already-S/A names never fire here

    last_close = _finite(entry.get("last_close"))
    if last_close is None or last_close <= 0:
        return None

    atr_value = _finite(entry.get("atr20"))
    atr_source = "atr20"
    if atr_value is None or atr_value <= 0:
        atr_value = _finite(atr)
        atr_source = "bot_atr"
    if atr_value is None or atr_value <= 0:
        atr_value = last_close * TIER_FLIP_FALLBACK_ATR_PRICE_FRACTION
        atr_source = "price_fraction"

    conditions: list[dict] = []
    for trigger in entry.get("trigger_levels") or []:
        if not isinstance(trigger, dict):
            continue
        level = _finite(trigger.get("level"))
        if level is None or level <= 0:
            continue
        action = str(trigger.get("action") or "").strip().lower()
        if action not in {"break_above", "break_below"}:
            continue
        # Still-armed vs the scan close: a level the daily close already
        # passed is not a pending flip.
        if action == "break_above" and not last_close < level:
            continue
        if action == "break_below" and not last_close > level:
            continue
        distance_atr = abs(level - last_close) / atr_value
        if distance_atr > max_distance_atr:
            continue
        conditions.append(
            {
                **trigger,
                "distance_atr": round(distance_atr, 3),
            }
        )

    if not conditions:
        return None
    conditions.sort(key=lambda c: (c.get("distance_atr") or 0.0, str(c.get("trigger_id") or "")))

    return {
        "schema_version": TIER_FLIP_SCHEMA_VERSION,
        "symbol": symbol,
        "side": side,
        "current_bucket": bucket,
        "priority_score": _finite(entry.get("priority_score")),
        "setup_family": str(entry.get("setup_family") or "").strip(),
        "run_date": str(entry.get("watchlist_run_date") or "").strip(),
        "last_close": round(last_close, 4),
        "atr_basis": round(atr_value, 4),
        "atr_source": atr_source,
        "conditions": conditions,
    }


def build_tier_flip_watch(
    watchlist_map: Any,
    *,
    atr_by_symbol: Any = None,
    max_distance_atr: float = TIER_FLIP_SMALL_MOVE_MAX_ATR,
) -> dict:
    """Map-level convenience wrapper over ``build_tier_flip_watch_entry``."""
    watch: dict[str, dict] = {}
    atr_map = atr_by_symbol or {}
    for symbol, entry in (watchlist_map or {}).items():
        key = str(symbol or "").strip().upper()
        built = build_tier_flip_watch_entry(
            entry,
            atr=atr_map.get(key),
            max_distance_atr=max_distance_atr,
        )
        if built:
            watch[key] = built
    return watch


def detect_tier_flip_triggers(watch_entry: Any, bars: Any) -> list[dict]:
    """Armed conditions whose flip confirmed on the latest completed bar.

    ``bars`` is a chronological list of *completed* M5 bars, each a mapping
    with ``close`` (high/low are accepted and ignored - the flip is a
    close-through, never a wick touch):

      - ``break_above``: prior completed close below the level AND the last
        completed close above it.
      - ``break_below``: mirror.

    Both closes strictly through means the cross happened earlier (or gapped
    overnight) - no fire; the two-bar shape pins the alert to the bar that
    actually completed the move. Returns shallow copies, nearest level first.
    """
    if not isinstance(watch_entry, dict):
        return []
    conditions = watch_entry.get("conditions") or []
    closes: list[float] = []
    for bar in bars or []:
        if not isinstance(bar, dict):
            continue
        close = _finite(bar.get("close"))
        if close is None:
            continue
        closes.append(close)
    if len(closes) < 2 or not conditions:
        return []

    prev_close, last_close = closes[-2], closes[-1]

    fired: list[dict] = []
    for condition in conditions:
        if not isinstance(condition, dict):
            continue
        level = _finite(condition.get("level"))
        if level is None:
            continue
        action = str(condition.get("action") or "").strip().lower()
        if action == "break_above":
            hit = prev_close < level < last_close
        elif action == "break_below":
            hit = prev_close > level > last_close
        else:
            hit = False
        if hit:
            fired.append(dict(condition))
    return fired


def format_tier_flip_message(event: Any) -> str:
    """One self-explaining D1 Focus row for a confirmed tier flip.

    Reads as a transition sentence: what the name was, the small move that
    just completed on a closed M5 bar, the scan's own upgrade rationale, the
    measured entry-context evidence, and the rvol stamp. The predicted flip
    is always labelled as pending the next scan's confirmation.
    """
    event = event if isinstance(event, dict) else {}
    symbol = str(event.get("symbol") or "").strip().upper()
    direction = str(event.get("direction") or "").strip().lower() or "watch"
    alert_label = str(event.get("alert_label") or event.get("label") or "level cross").strip()
    level = _finite(event.get("level"))
    price = _finite(event.get("current_price"))
    bar_time = str(event.get("bar_time") or "").strip()
    distance_atr = _finite(event.get("distance_atr"))
    scan_close = _finite(event.get("scan_close"))
    score = _finite(event.get("priority_score"))
    bucket = str(event.get("current_bucket") or "").strip() or "none"
    reason = str(event.get("reason") or "").strip()
    context_tier = str(event.get("context_tier") or "").strip().upper()
    context_note = str(event.get("context_note") or "").strip()
    rvol = _finite(event.get("rvol"))

    parts: list[str] = []
    if level is not None:
        parts.append(f"@{level:.2f}")
    if price is not None:
        parts.append(f"px={price:.2f}")
    if bar_time:
        parts.append(f"bar={bar_time}")
    if distance_atr is not None and scan_close is not None:
        parts.append(f"move {distance_atr:.2f} ATR from scan close {scan_close:.2f}")
    was = f"was: bucket {bucket}"
    if score is not None:
        was += f", score {score:.0f}"
    parts.append(was)
    if reason:
        parts.append(reason)
    if context_tier:
        context = f"ctx {context_tier}-tier"
        if context_note:
            context += f": {context_note}"
        parts.append(context)
    if rvol is not None:
        parts.append(f"rvol {rvol:.2f}")
    suffix = f" [{'; '.join(parts)}]" if parts else ""
    return (
        f"MASTER_AVWAP_D1_TIER_FLIP: {symbol} ({direction}) "
        f"non-S/A -> A/S predicted (next scan confirms) {alert_label}{suffix}"
    )
