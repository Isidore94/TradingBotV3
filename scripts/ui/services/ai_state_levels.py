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


_UNREADABLE = "unreadable"


def _refresh(force: bool) -> str:
    """Re-read the ai_state file when its mtime moved.

    Returns `"warm"` (the cache already matched the file), `"parsed"` (the file
    was read just now - the expensive case) or `"unreadable"` (no file, or a
    file that would not parse). ONE read feeds both maps: the 36 MB parse is
    the only cost here, and doing it twice for two views would double it.
    """
    try:
        mtime = MASTER_AVWAP_AI_STATE_FILE.stat().st_mtime
    except OSError:
        return _UNREADABLE
    if not force and _cache["mtime"] == mtime and _cache["levels"]:
        return "warm"
    try:
        with open(MASTER_AVWAP_AI_STATE_FILE, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        logging.warning("Could not load ai_state for level feed: %s", exc)
        return _UNREADABLE

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
    return "parsed"


def load_symbol_levels(force: bool = False) -> dict[str, dict]:
    """symbol -> {vwap, bands, anchor_date, atr20, last_close, side}.

    Unchanged since it shipped, including the `{}` on an unstat-able file: a
    caller that cannot see the file is told nothing, not told yesterday.
    """
    if _refresh(force) == _UNREADABLE:
        return {}
    return _cache["levels"]


def load_symbol_compression(force: bool = False) -> dict[str, dict]:
    """symbol -> the scan's compression reading. **May PARSE the 36 MB file.**

    Call this only where a parse is allowed - a worker thread, a CLI, a test.
    The Qt thread calls :func:`cached_symbol_compression` instead; measured on
    the desk, one parse of the live file is 281-292 ms, and
    `master_avwap_panel.refresh_from_reports` runs on every watched-file change.

    Empty for a symbol the scan did not measure, and empty overall until a scan
    from a build carrying PCT-3 item 1 has landed - an old file is a file with
    nothing to say, never a row read as "not compressed".
    """
    if _refresh(force) == _UNREADABLE:
        return {}
    return _cache["compression"]


def cached_symbol_compression() -> dict[str, dict]:
    """The compression map ALREADY in memory. Never opens a file, never stats.

    This is the Qt thread's door. A cold cache answers `{}` - "nothing to say
    yet" - and the chip simply is not painted until :func:`warm_cache` has run
    on a worker and the panel has asked for one more refresh.
    """
    return _cache["compression"] or {}


def cache_signature() -> object:
    """What a caller can compare to see whether the cache moved."""
    return _cache["mtime"]


def warm_cache(force: bool = False) -> bool:
    """Parse the ai_state file if it moved. **Never call this on the Qt thread.**

    Returns `True` when the cache CHANGED, which is the panel's cue to run one
    coalesced refresh. `False` means already warm, or unreadable - and an
    unreadable file leaves the last good cache in place, because a feed that
    blinked is not a reason to drop the chips off every row.
    """
    before = _cache["mtime"]
    outcome = _refresh(force)
    return outcome == "parsed" and _cache["mtime"] != before
