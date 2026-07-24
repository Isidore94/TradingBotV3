from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


SYMBOL_RE = re.compile(r"\b([A-Z][A-Z0-9.\-]{0,9})\b")

# Entry-assist outputs (window open/close summaries, trailing-movers lists,
# failure notes). They are list-style messages, not single-symbol alerts, so
# symbol extraction is skipped for them; the Alert Center also lets them
# bypass the tier gate since the trader explicitly asked for the output.
ENTRY_ASSIST_PREFIXES = ("ENTRY WINDOW", "ENTRY ASSIST", "STRONGEST ", "WEAKEST ")


def is_entry_assist_text(text: Any) -> bool:
    return str(text or "").strip().upper().startswith(ENTRY_ASSIST_PREFIXES)


# A user-armed chart watch (New HOD / New LOD / VWAP bounce) firing. These are
# armed only from the visual M5 review chart and flag red in the Alert Center.
CHART_WATCH_TAG = "chart_watch"


def is_chart_watch_alert(alert: Any) -> bool:
    return str(getattr(alert, "tag", "") or "") == CHART_WATCH_TAG


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
        if not symbol and not is_entry_assist_text(raw_text):
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

    @classmethod
    def from_callback_many(
        cls,
        message: Any,
        tag: str,
        timestamp: datetime | None = None,
    ) -> list["BounceAlert"]:
        """Turn list-style M5 output into one actionable alert per ticker."""
        timestamp = timestamp or datetime.now()
        alert = cls.from_callback(message, tag, timestamp=timestamp)
        rows = _list_m5_rows(alert.raw_text)
        if not rows:
            return [alert]

        alerts = []
        for rank, (symbol, metric) in enumerate(rows, start=1):
            payload = dict(alert.payload)
            payload.update(
                {
                    "list_parent": alert.raw_text,
                    "list_rank": rank,
                    "list_metric": metric,
                }
            )
            alerts.append(
                cls(
                    time_text=alert.time_text,
                    symbol=symbol,
                    side=alert.side,
                    trigger=metric,
                    timeframe="M5",
                    context=alert.context,
                    tag=alert.tag,
                    raw_text=alert.raw_text,
                    payload=payload,
                )
            )
        return alerts


def _message_text(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("text") or message.get("message") or message)
    return str(message or "")


def _extract_symbol(text: str) -> str:
    if ":" in text:
        candidate = text.split(":", 1)[0].strip().split()[-1].upper()
        if (
            candidate
            and not candidate.startswith("MASTER_AVWAP_D1")
            and candidate not in {"MASTER_AVWAP_FOCUS_BOUNCE"}
        ):
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


_ENTRY_WINDOW_LIST_RE = re.compile(
    r"^ENTRY WINDOW \((?:long|short)\):.*?(?:held strongest|stayed weakest) "
    r"through it:\s*(?P<rows>.+?)\s+\[[^\]]+\]\.?$",
    re.IGNORECASE,
)
_TRAILING_MOVERS_LIST_RE = re.compile(
    r"^(?:STRONGEST|WEAKEST)\s+\d+M\s+\((?:long|short)\):\s*"
    r"(?P<rows>.+?)\s+\[[^\]]+\]\.?$",
    re.IGNORECASE,
)
_REGIME_PAUSE_LIST_RE = re.compile(
    r"^REGIME PAUSE WATCH \((?:long|short)\):.*?"
    r"(?P<pattern>holding highs|pressing lows):\s*(?P<rows>.+?)"
    r"\s+\(\d+\s+today\)\.",
    re.IGNORECASE,
)
_M5_LIST_ROW_RE = re.compile(
    r"(?P<symbol>[A-Z][A-Z0-9.\-]{0,9})\s+"
    r"(?P<move>[+-]\d+(?:\.\d+)?)%"
    r"(?:\s+\(x(?P<excess>[+-]\d+(?:\.\d+)?)\))?",
)


def _list_m5_rows(text: str) -> list[tuple[str, str]]:
    """Parse list alerts sourced from cached M5 bars into clickable rows."""
    normalized = str(text or "").strip()
    match = _ENTRY_WINDOW_LIST_RE.match(normalized) or _TRAILING_MOVERS_LIST_RE.match(
        normalized
    )
    if match:
        rows = []
        for item in _M5_LIST_ROW_RE.finditer(match.group("rows")):
            symbol = item.group("symbol").upper()
            metric = f"M5 move {item.group('move')}%"
            if item.group("excess") is not None:
                metric += f" · vs SPY x{item.group('excess')}"
            rows.append((symbol, metric))
        return rows

    pause_match = _REGIME_PAUSE_LIST_RE.match(normalized)
    if not pause_match:
        return []
    pattern = pause_match.group("pattern").lower()
    symbols = [
        token.strip().upper() for token in pause_match.group("rows").split(",")
    ]
    return [
        (symbol, f"M5 regime-pause watch · {pattern}")
        for symbol in symbols
        if SYMBOL_RE.fullmatch(symbol)
    ]
