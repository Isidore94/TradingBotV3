from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


SYMBOL_RE = re.compile(r"\b([A-Z][A-Z0-9.\-]{0,9})\b")


@dataclass
class BounceAlert:
    time_text: str
    symbol: str = ""
    side: str = "WATCH"
    trigger: str = ""
    timeframe: str = ""
    context: str = ""
    tag: str = ""
    raw_text: str = ""
    is_d1: bool = False
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_callback(cls, message: Any, tag: str, timestamp: datetime | None = None) -> "BounceAlert":
        timestamp = timestamp or datetime.now()
        tag_text = str(tag or "")
        payload = message if isinstance(message, dict) else {}
        raw_text = _message_text(message)
        feedback = payload.get("feedback") if isinstance(payload.get("feedback"), dict) else {}

        side = _side_from(feedback.get("direction") or tag_text or raw_text)
        symbol = str(feedback.get("symbol") or "").strip().upper()
        if not symbol:
            symbol = _extract_symbol(raw_text)

        trigger = str(feedback.get("bounce_types") or "").replace(";", ", ").strip()
        if not trigger:
            trigger = _trigger_from_text(raw_text)

        return cls(
            time_text=timestamp.strftime("%H:%M:%S"),
            symbol=symbol,
            side=side,
            trigger=trigger,
            timeframe=_timeframe_from_text(raw_text),
            context=_context_from_text(raw_text),
            tag=tag_text,
            raw_text=raw_text,
            is_d1=tag_text.startswith("d1_flag") or raw_text.startswith("MASTER_AVWAP_D1"),
            payload=dict(payload),
        )


def _message_text(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("text") or message.get("message") or message)
    return str(message or "")


def _extract_symbol(text: str) -> str:
    if ":" in text:
        candidate = text.split(":", 1)[0].strip().split()[-1].upper()
        if candidate and candidate not in {"MASTER_AVWAP_D1_FLAG", "MASTER_AVWAP_FOCUS_BOUNCE"}:
            return candidate
    match = SYMBOL_RE.search(text)
    return match.group(1).upper() if match else ""


def _side_from(value: Any) -> str:
    text = str(value or "").lower()
    if "long" in text or text == "green":
        return "LONG"
    if "short" in text or text == "red":
        return "SHORT"
    return "WATCH"


def _trigger_from_text(text: str) -> str:
    if "Bounce confirmed" in text:
        tail = text.split("from", 1)[-1].strip() if "from" in text else "Bounce confirmed"
        return tail or "Bounce confirmed"
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return text[:90]


def _timeframe_from_text(text: str) -> str:
    for token in ("5m", "15m", "30m", "1h", "H1", "D1"):
        if token in text:
            return token
    return ""


def _context_from_text(text: str) -> str:
    if "[" in text and "]" in text:
        return text.rsplit("[", 1)[-1].split("]", 1)[0].strip()
    return ""
