"""Strict trade-side typing (plan.md 22.3).

The legacy compatibility helper mapped every value other than exact "SHORT" -
including empty strings and typos - to "LONG", silently converting corrupt or
unlabeled rows into long candidates. `parse_side` is strict and returns
UNKNOWN for invalid input; only the explicit legacy-ingestion adapter
(`coerce_side_legacy`) may apply a default, and it counts every coercion as a
data-quality event in the active run manifest.
"""

from __future__ import annotations

import logging
from enum import Enum

_LONG_ALIASES = {"LONG", "L", "BUY", "BULL", "BULLISH"}
_SHORT_ALIASES = {"SHORT", "S", "SELL", "BEAR", "BEARISH"}

_warned_values: set[str] = set()


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    UNKNOWN = "UNKNOWN"


def parse_side(value) -> Side:
    """Strict parse: anything not clearly long/short is UNKNOWN, never a default."""
    text = str(value or "").strip().upper()
    if text in _LONG_ALIASES:
        return Side.LONG
    if text in _SHORT_ALIASES:
        return Side.SHORT
    return Side.UNKNOWN


def coerce_side_legacy(value, *, default: Side = Side.LONG, context: str = "") -> str:
    """Legacy-ingestion adapter: applies `default` for UNKNOWN input but makes
    the coercion visible - counted in the run manifest and warned once per
    distinct garbage value per process (empty values count silently; they are
    common in legacy rows that intend the default)."""
    side = parse_side(value)
    if side is not Side.UNKNOWN:
        return side.value

    text = str(value or "").strip().upper()
    counter = "side_coercions_empty" if not text else "side_coercions_invalid"
    try:
        from diagnostics import get_active_recorder

        recorder = get_active_recorder()
        if recorder is not None:
            recorder.incr(counter)
            if text:
                evidence = recorder.outputs.setdefault("invalid_side_values", {})
                key = text[:80]
                item = evidence.setdefault(key, {"count": 0, "contexts": []})
                item["count"] = int(item.get("count", 0) or 0) + 1
                context_text = str(context or "").strip()
                if context_text and context_text not in item["contexts"]:
                    item["contexts"].append(context_text)
    except Exception:
        pass
    if text and text not in _warned_values:
        _warned_values.add(text)
        logging.warning(
            "Invalid side value %r coerced to %s%s - fix the producing row.",
            value,
            default.value,
            f" ({context})" if context else "",
        )
    return default.value
