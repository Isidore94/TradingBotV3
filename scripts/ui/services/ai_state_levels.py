"""Per-symbol level feed for the UI, read from the scan's ai_state file.

The 38MB ai_state JSON is the only on-disk source that carries the current
earnings-anchor bands per symbol, so the Setup Tracker detail pane reads it
lazily (once, cached with the file mtime) and extracts just the fields the
trade-plan calculator needs. Loading takes well under a second locally; the
cache means clicking through picks costs nothing after the first click.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from project_paths import MASTER_AVWAP_AI_STATE_FILE

_cache: dict[str, Any] = {"mtime": None, "levels": {}}


def load_symbol_levels(force: bool = False) -> dict[str, dict]:
    """symbol -> {vwap, bands, anchor_date, atr20, last_close, side}."""
    try:
        mtime = MASTER_AVWAP_AI_STATE_FILE.stat().st_mtime
    except OSError:
        return {}
    if not force and _cache["mtime"] == mtime and _cache["levels"]:
        return _cache["levels"]
    try:
        with open(MASTER_AVWAP_AI_STATE_FILE, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        logging.warning("Could not load ai_state for level feed: %s", exc)
        return _cache["levels"] or {}

    levels: dict[str, dict] = {}
    symbols = payload.get("symbols") if isinstance(payload, dict) else {}
    for symbol, entry in (symbols or {}).items():
        if not isinstance(entry, dict):
            continue
        anchor = entry.get("current_anchor")
        anchor = anchor if isinstance(anchor, dict) else {}
        bands = anchor.get("bands")
        levels[str(symbol).strip().upper()] = {
            "vwap": anchor.get("vwap"),
            "bands": bands if isinstance(bands, dict) else {},
            "anchor_date": str(anchor.get("date") or ""),
            "atr20": entry.get("atr20"),
            "last_close": entry.get("last_close"),
            "side": str(entry.get("side") or ""),
        }
    _cache["mtime"] = mtime
    _cache["levels"] = levels
    return levels
