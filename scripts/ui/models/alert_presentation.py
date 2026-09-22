"""Pure presentation meanings shared by the Alert Center's two alert surfaces."""

from __future__ import annotations

from dataclasses import dataclass

from chart_watch import D1_LEVEL_KINDS, PULLBACK_KIND
from ui.models.bounce import (
    AUTO_PICK_TAG,
    FOCUS_D1_EVENT_TAG,
    FOCUS_REVIEW_TAG,
    MANUAL_CHART_TAG,
    BounceAlert,
    is_chart_watch_alert,
)


_MUTED_TAGS = frozenset({MANUAL_CHART_TAG, AUTO_PICK_TAG, FOCUS_REVIEW_TAG})
_LIVE_FOCUS_TAGS = frozenset({FOCUS_D1_EVENT_TAG, "d1_focus_pin"})


@dataclass(frozen=True)
class AlertPresentation:
    """Display-only answer for a live alert, with no routing or data effects."""

    frame_tone: str
    reason_tone: str
    source_timeframe: str = ""


def classify_alert(alert: BounceAlert, *, in_focus: bool = False) -> AlertPresentation:
    """Classify an alert once for its frame, badge source, and reason text.

    A persistent Pullback remains a D1 row for routing, but its M15/M30/H1
    source owns its visual colour.  ``in_focus`` is actual membership.  A
    chart's destination ``focus_category`` must never be passed here.
    """
    source_timeframe = pullback_source_timeframe(alert)
    return AlertPresentation(
        frame_tone=_frame_tone(alert, in_focus=in_focus, source_timeframe=source_timeframe),
        reason_tone=_reason_tone(alert, in_focus=in_focus, source_timeframe=source_timeframe),
        source_timeframe=source_timeframe,
    )


def pullback_source_timeframe(alert: BounceAlert) -> str:
    """Return the measured source bar for a persistent Pullback, if known."""
    payload = alert.payload if isinstance(alert.payload, dict) else {}
    if str(payload.get("chart_watch_kind") or "") != PULLBACK_KIND:
        return ""
    value = str(payload.get("timeframe") or "").strip().upper().replace(" ", "")
    return {
        "15M": "M15",
        "M15": "M15",
        "30M": "M30",
        "M30": "M30",
        "1H": "H1",
        "H1": "H1",
    }.get(value, "")


def _frame_tone(alert: BounceAlert, *, in_focus: bool, source_timeframe: str) -> str:
    if source_timeframe:
        return f"pullback-{source_timeframe.lower()}"
    payload = alert.payload if isinstance(alert.payload, dict) else {}
    watch_kind = str(payload.get("chart_watch_kind") or "")
    if is_chart_watch_alert(alert) and watch_kind in D1_LEVEL_KINDS:
        return "personal-d1"
    if _is_d1(alert) and in_focus:
        return "focus-d1"
    if _is_d1(alert):
        return "d1"
    return ""


def _reason_tone(alert: BounceAlert, *, in_focus: bool, source_timeframe: str) -> str:
    tag = str(alert.tag or "")
    if tag in _MUTED_TAGS:
        return "muted"
    if _is_price_alert(alert):
        return "price"
    # Source-bar colours carry more information than Focus membership.
    if source_timeframe:
        return f"pullback-{source_timeframe.lower()}"
    if in_focus or tag in _LIVE_FOCUS_TAGS or tag.startswith("master_avwap_focus_"):
        return "focus"
    if _is_d1(alert):
        return "d1"
    # An ordinary scanner or user-armed watch remains the established red.
    return "live"


def _is_d1(alert: BounceAlert) -> bool:
    return bool(alert.is_d1) or str(alert.timeframe or "").strip().upper() == "D1"


def _is_price_alert(alert: BounceAlert) -> bool:
    return str(alert.raw_text or alert.trigger or "").lstrip().upper().startswith("PRICE ALERT")
