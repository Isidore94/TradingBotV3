"""One stable identity for a reviewed trading opportunity.

An opportunity is not just a ticker on a date.  Side and horizon are part of
the thesis, and a D1 chart must never absorb an M5 verdict (or vice versa).
This module is intentionally pure so review, journal and later lifecycle
readers can share the exact same normalization without importing Qt or stores.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

IDENTITY_VERSION = "opportunity_identity_v1"


def _text(value: Any) -> str:
    return str(value or "").strip()


def normalized_side(value: Any) -> str:
    text = _text(value).upper()
    if text.startswith("SHORT") or text in {"SELL", "S"}:
        return "SHORT"
    if text.startswith("LONG") or text in {"BUY", "B"}:
        return "LONG"
    return "UNKNOWN"


def timeframe_slot(row: Mapping[str, Any]) -> str:
    raw = _text(row.get("timeframe")).upper().replace(" ", "")
    if bool(row.get("is_d1")) or raw in {"D", "D1", "1D", "DAILY", "SWING"}:
        return "D1"
    if raw in {"M5", "5M", "5MIN", "5MINS", "INTRADAY", "DAY"}:
        return "M5"
    return raw or "UNKNOWN"


def thesis_slot(row: Mapping[str, Any]) -> str:
    """The narrowest durable thesis key the writer supplied.

    Event id wins because review actions for one shown chart carry it through.
    Older rows fall back through existing category fields and remain separate
    when a setup/horizon is known.  No result or outcome enters the identity.
    """

    event_id = _text(row.get("event_id"))
    if event_id:
        return f"event:{event_id}"
    for key in (
        "pick_source_family",
        "category_slot",
        "bucket",
        "setup_family",
        "bounce_types",
        "tag",
        "surface",
    ):
        value = _text(row.get(key))
        if value:
            return f"{key}:{value.lower()}"
    return "unclassified"


def opportunity_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        _text(row.get("trade_date") or row.get("session_date"))[:10],
        _text(row.get("symbol")).upper(),
        normalized_side(row.get("side") or row.get("direction")),
        timeframe_slot(row),
        thesis_slot(row),
    )


def opportunity_id(row: Mapping[str, Any]) -> str:
    explicit = _text(row.get("opportunity_id"))
    if explicit:
        return explicit
    payload = json.dumps(
        [IDENTITY_VERSION, *opportunity_key(row)],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "opp:" + hashlib.sha256(payload).hexdigest()[:24]

