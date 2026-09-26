"""Pure alert gates for the Alert Center: tier, loudness, D1 ready/developing, push labels.

No Qt import; `alert_center_panel` re-exports every name here.
"""

from __future__ import annotations

import re
from datetime import datetime

from chart_watch import D1_EVENT_KINDS, D1_LEVEL_KINDS
from intraday_history import (
    bucket_end as _intraday_bucket_end,
    last_completed_bucket as _intraday_last_bucket,
)
from ui.models.bounce import (
    BounceAlert,
    is_chart_watch_alert,
    is_entry_assist_text,
)

_TIER_RE = re.compile(r"\[([SABCD])-TIER\]", re.IGNORECASE)
_TIER_RANK = {"S": 4, "A": 3, "B": 2, "C": 1, "D": 0}


def intraday_last_bucket_end(moment: datetime, interval_minutes: int):
    """When this timeframe's most recent COMPLETED bucket finished, or None.

    The armed poll's "is there anything new to judge?" question, asked once
    per interval per tick rather than once per watch: a completed bar is the
    only thing that can change a pullback verdict, and resolving the session
    window is the expensive part (PCT-1 review blocker 3b). Pure bucket math
    on the desk's market-local clock, so a stubbed cache cannot change it.
    """
    bucket = _intraday_last_bucket(moment, interval_minutes)
    if bucket is None:
        return None
    return _intraday_bucket_end(bucket, interval_minutes)


# D1 focus alerts that mark a stock TURNING INTO a favorite / high-conviction
# name: the scan confirmed a genuine bucket upgrade. An armed-level crossing
# is still only developing evidence and stays out of both actionable feeds. A
# final Favorite / High Conviction bucket result belongs in the D1 Focus feed
# (user rule 2026-07-09: "only things that turn a stock into a favourite or
# high conviction bucket stock"). Generic champion D1 flags retain their live
# routing under the normal tier gate.
_D1_READY_PREFIXES = {
    # The D1 Focus feed is the M5 band-zone rubric: a scanned name bouncing off
    # AVWAPE / 1st-dev / 15-21EMA or breaking the next band, confirmed on two
    # completed bars. A fresh Favorite / High Conviction bucket upgrade still
    # surfaces here too.
    "MASTER_AVWAP_D1_ZONE",
    "MASTER_AVWAP_D1_BUCKET_UPGRADE",
    # Pre-armed tier flip: a non-S/A name closed through the A/S upgrade-target
    # level the scan armed one small move away - the headline D1 Focus event
    # (few per day, rvol/context gated, predicted pending next-scan confirm).
    "MASTER_AVWAP_D1_TIER_FLIP",
}
_D1_DEVELOPING_PREFIXES = {
    "MASTER_AVWAP_D1_RESEARCH",
    # Compatibility with messages already queued by an older bot process.
    "MASTER_AVWAP_D1_UPGRADE_TRIGGER",
    "MASTER_AVWAP_D1_UPGRADE_WATCH",
}


def _bar_close(bar: object) -> float | None:
    try:
        return float(bar["close"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        return None


def _d1_alert_prefix(alert: BounceAlert) -> str:
    return str(alert.raw_text or "").split(":", 1)[0].strip().upper()


def is_developing_d1_alert(alert: BounceAlert) -> bool:
    return _d1_alert_prefix(alert) in _D1_DEVELOPING_PREFIXES


def _is_feed_noise_alert(alert: BounceAlert) -> bool:
    if is_developing_d1_alert(alert):
        return True
    text = f"{alert.raw_text} {alert.trigger}".strip().lower()
    return not alert.is_d1 and alert.side == "WATCH" and "candle has closed" in text


def is_ready_d1_alert(alert: BounceAlert) -> bool:
    return _d1_alert_prefix(alert) in _D1_READY_PREFIXES


# Short labels for the scanner's own D1 focus alerts, so the hourly phone push
# reads "NVDA bucket upgrade" rather than a 200-character raw alert line.
_D1_PUSH_LABELS = {
    "MASTER_AVWAP_D1_ZONE": "D1 zone",
    "MASTER_AVWAP_D1_BUCKET_UPGRADE": "bucket upgrade",
    "MASTER_AVWAP_D1_TIER_FLIP": "tier flip",
}


def d1_push_event(alert: BounceAlert) -> dict[str, str] | None:
    """What the hourly D1 phone push should carry for this alert, if anything.

    One classifier, here rather than in the Auto Pilot service, because this
    module already owns which D1 alerts are actionable and which are developing
    research. The phone therefore names exactly the events the D1 Focus feed
    shows, and the two can never drift apart.
    """
    symbol = str(getattr(alert, "symbol", "") or "").strip().upper()
    if not symbol:
        return None
    kind = ""
    payload = getattr(alert, "payload", None)
    if isinstance(payload, dict):
        kind = str(payload.get("chart_watch_kind") or payload.get("focus_d1_kind") or "")
    label = ""
    if kind:
        # Armed D1 levels and D1 event watches: the trader asked for exactly
        # this condition, so it belongs on the phone by definition.
        label = D1_LEVEL_KINDS.get(kind) or D1_EVENT_KINDS.get(kind) or ""
        if not label:
            return None
    elif getattr(alert, "is_d1", False) and is_ready_d1_alert(alert):
        label = _D1_PUSH_LABELS.get(_d1_alert_prefix(alert), "D1 event")
    else:
        return None
    return {
        "symbol": symbol,
        "label": label,
        "time_text": str(getattr(alert, "time_text", "") or ""),
    }


def extract_alert_tier(alert: BounceAlert) -> str:
    match = _TIER_RE.search(str(alert.raw_text or ""))
    return match.group(1).upper() if match else ""


# "BANGER" was retired 2026-09-01 (trader: "We can probably remove this because
# idk what it is"). It was only ever a literal token match against alert text,
# and nothing in the tree ever emitted the token: 0 of 8,818 recorded review
# rows carried banger=True. PROVEN is the top alert class and is untouched.


# The retired learning-loop PROVEN stamp (P14, 2026-09-26): new alerts never
# carry it; it is read only for rows written before, and never passes the tier gate.
_PROVEN_RE = re.compile(r"\bPROVEN\b")


def is_proven_alert(alert: BounceAlert) -> bool:
    return bool(_PROVEN_RE.search(str(alert.raw_text or "")))


def is_entry_assist_alert(alert: BounceAlert) -> bool:
    return str(alert.tag or "") == "entry_assist" or is_entry_assist_text(alert.raw_text)


def alert_passes_min_tier(alert: BounceAlert, mode: str, *, grade_bypass: bool = False) -> bool:
    """Filter policy for the live feed (D1 alerts route to their own feed).

    P14: an M5 row whose setup grade is in `alert_show_filter.bypass_grades`
    (`grade_bypass`, decided by the panel) always passes; the PROVEN stamp no
    longer does. Entry-assist output passes too — the trader clicked a button
    asking for it. Chart-watch hits pass for the same reason: the trader armed
    that exact condition from the M5 chart. Untiered alerts (regime notes,
    pause-watch summaries) pass everything except the S-only mode.
    """
    if mode in ("", "all"):
        return True
    if grade_bypass or is_entry_assist_alert(alert) or is_chart_watch_alert(alert):
        return True
    tier = extract_alert_tier(alert)
    if not tier:
        return mode != "S"
    return _TIER_RANK.get(tier, 0) >= _TIER_RANK.get(mode, 0)


def alert_is_loud(alert: BounceAlert) -> bool:
    """Alerts worth a sound: proven configs, S/A tiers, ready D1, and
    chart-watch hits (the trader armed the exact condition and is
    waiting on it)."""
    return (
        is_proven_alert(alert)
        or is_ready_d1_alert(alert)
        or is_chart_watch_alert(alert)
        or extract_alert_tier(alert) in {"S", "A"}
    )


def alert_passes_feed_gate(
    alert: BounceAlert, mode: str, *, is_focus: bool = False, grade_bypass: bool = False
) -> bool:
    """Liked (focus) picks always surface; everything else obeys the tier gate."""
    return is_focus or alert_passes_min_tier(alert, mode, grade_bypass=grade_bypass)


def alert_should_sound(alert: BounceAlert, *, is_focus: bool = False) -> bool:
    """Liked (focus) picks always sound; everything else needs to be loud."""
    return is_focus or alert_is_loud(alert)


def favorite_category_for_alert(alert: BounceAlert) -> str:
    """Where the ★ files a pick: D1/H1 alerts are swing material, the rest M5.

    Matches the trader's split: longs/shorts.txt alerts are M5 day-trade
    based, while bot-generated D1/H1 output is multi-day swing evidence.
    """
    if alert.is_d1 or str(alert.timeframe or "").strip().lower() in {"d1", "h1", "1h"}:
        return "swing"
    return "m5"


def favorite_origin_for_alert(alert: BounceAlert) -> str:
    """Which alert flavor a verdict came from - logged so the tracker can grade
    H1-sourced picks separately from D1-sourced ones (and M5 likewise)."""
    if alert.is_d1:
        return "d1"
    if str(alert.timeframe or "").strip().lower() in {"h1", "1h"}:
        return "h1"
    return "m5"
