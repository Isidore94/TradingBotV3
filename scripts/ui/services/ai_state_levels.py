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

_cache: dict[str, Any] = {"mtime": None, "levels": {}, "compression": {}}

#: PCT-3: the compression fields the setups table's `compressed` chip reads.
#: The report line the table is built from does not carry them; this file does,
#: per symbol, because `legacy.py` publishes them on the `ai_state` symbol entry.
COMPRESSION_FIELDS = (
    "compression_flag",
    "compression_penalty",
    "compression_note",
    "compression_score",
    "compression_stdev_atr_ratio",
    "compression_range_atr_ratio",
    "compression_close_range_atr_ratio",
    "compression_rule_version",
)


def _refresh(force: bool) -> bool:
    """Re-read the ai_state file when its mtime moved. `False` on any miss.

    ONE read feeds both maps: the 38 MB parse is the expensive part, and doing
    it twice for two views of the same file would double the only cost here.
    """
    try:
        mtime = MASTER_AVWAP_AI_STATE_FILE.stat().st_mtime
    except OSError:
        return False
    if not force and _cache["mtime"] == mtime and _cache["levels"]:
        return True
    try:
        with open(MASTER_AVWAP_AI_STATE_FILE, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        logging.warning("Could not load ai_state for level feed: %s", exc)
        return False

    levels: dict[str, dict] = {}
    compression: dict[str, dict] = {}
    symbols = payload.get("symbols") if isinstance(payload, dict) else {}
    for symbol, entry in (symbols or {}).items():
        if not isinstance(entry, dict):
            continue
        anchor = entry.get("current_anchor")
        anchor = anchor if isinstance(anchor, dict) else {}
        bands = anchor.get("bands")
        key = str(symbol).strip().upper()
        levels[key] = {
            "vwap": anchor.get("vwap"),
            "bands": bands if isinstance(bands, dict) else {},
            "anchor_date": str(anchor.get("date") or ""),
            "atr20": entry.get("atr20"),
            "last_close": entry.get("last_close"),
            "side": str(entry.get("side") or ""),
        }
        carried = {field: entry[field] for field in COMPRESSION_FIELDS if field in entry}
        if carried:
            compression[key] = carried
    _cache["mtime"] = mtime
    _cache["levels"] = levels
    _cache["compression"] = compression
    return True


def load_symbol_levels(force: bool = False) -> dict[str, dict]:
    """symbol -> {vwap, bands, anchor_date, atr20, last_close, side}."""
    if not _refresh(force):
        return _cache["levels"] or {}
    return _cache["levels"]


def load_symbol_compression(force: bool = False) -> dict[str, dict]:
    """symbol -> the scan's compression reading for that symbol (PCT-3).

    Empty for a symbol the scan did not measure, and empty overall until a scan
    written by a build that carries PCT-3 item 1 has landed - an old file is
    simply a file with nothing to say, never a row read as "not compressed".
    """
    if not _refresh(force):
        return _cache["compression"] or {}
    return _cache["compression"]
