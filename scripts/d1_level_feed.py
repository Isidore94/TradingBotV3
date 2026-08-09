"""D1 major-level feed for the Technical Integrity monitor.

The master scan already maintains every D1 level the trader cares about:
daily SMA 50/100/200 (``priority_sma_levels`` in the ai_state file), the
directional D1 trendline (``priority_trendline_candidate``), and the
horizontal S/R stores (hv_horizontal + cloud_flat JSONs under the levels
directory). This module reads those artifacts - mtime-cached so BounceBot can
call it every M5 cycle for free - and shapes them into the ``extra_levels``
payload ``TechnicalIntegrityMonitor.observe_symbol`` accepts.

Decision-support only: nothing here writes state or influences alerts.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from master_avwap_lib.levels import level_is_effective_on, level_store_path
from project_paths import MASTER_AVWAP_AI_STATE_FILE, MASTER_AVWAP_LEVELS_DIR


# Evidence weights sit above every intraday family (max 1.25) so a confluence
# of, say, VWAP and the daily SMA50 dedupes into the D1 test.
D1_SMA_FAMILIES: tuple[tuple[str, str, float], ...] = (
    ("SMA_50", "d1_sma_50", 1.6),
    ("SMA_100", "d1_sma_100", 1.6),
    ("SMA_200", "d1_sma_200", 1.7),
)
D1_TRENDLINE_FAMILY = "d1_trendline"
D1_TRENDLINE_WEIGHT = 1.6
D1_HORIZONTAL_FAMILY = "d1_horizontal"
D1_HORIZONTAL_WEIGHT = 1.5
D1_CLOUD_FAMILY = "d1_cloud_flat"
D1_CLOUD_WEIGHT = 1.45

# Only levels that earned green-bucket-quality respect qualify as "major".
MIN_HORIZONTAL_STRENGTH = 1.0
# Levels further than this from the last completed close cannot be touched
# soon; dropping them keeps the candidate list tiny.
MAX_DISTANCE_ATR = 3.5
# ai_state values are frozen at the last scan. Daily SMAs drift slowly, so a
# somewhat stale scan is still usable; a trendline projects along its slope
# and goes wrong faster.
SMA_MAX_AGE_DAYS = 10
TRENDLINE_MAX_AGE_DAYS = 5

#: Name of this module's own ai_state projection (see ``load_ai_state_projection``).
AI_STATE_SYMBOL_PROJECTION = "d1_level_feed.symbol_entry"

# One parse of ai_state serves every consumer. {path: {"mtime", "feeds"}},
# where "feeds" is {projection key: {symbol: sliver}}.
_ai_state_cache: dict[str, dict[str, Any]] = {}
_ai_state_projections: dict[str, Callable[[Mapping[str, Any]], Any]] = {}
_store_cache: dict[str, dict[str, Any]] = {}


def _coerce_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_date(value: Any) -> date | None:
    text = str(value or "").strip()[:10]
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _extract_symbol_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    smas: dict[str, float] = {}
    sma_levels = entry.get("priority_sma_levels")
    if isinstance(sma_levels, Mapping):
        for label, _family, _weight in D1_SMA_FAMILIES:
            value = _coerce_float(sma_levels.get(label))
            if value is not None and value > 0:
                smas[label] = value
    trendlines: list[float] = []
    for key in ("priority_trendline_candidate", "priority_trendline_break_candidate"):
        candidate = entry.get(key)
        if isinstance(candidate, Mapping):
            value = _coerce_float(candidate.get("current_line_price"))
            if value is not None and value > 0:
                trendlines.append(value)
    return {
        "smas": smas,
        "trendlines": trendlines,
        "last_trade_date": str(entry.get("last_trade_date") or ""),
    }


def load_ai_state_projection(
    key: str,
    project: Callable[[Mapping[str, Any]], Any],
    path: Path,
) -> dict[str, Any]:
    """``{symbol: project(record)}`` for every symbol in ``path``, mtime-cached.

    The shared raw-parse point for the ai_state file. That file is ~38MB and
    more than one consumer wants a different sliver of it - this module's
    SMA/trendline prices for the Technical Integrity monitor, and
    ``chart_levels``' trendline record for drawing. Parsing it once per
    consumer was pure duplicated cost, so the parse lives here and every
    projection seen so far is recomputed on the one read: whichever consumer
    asks first pays, and the rest of them are free until the file changes.

    ``project(record)`` runs once per symbol record and returns that symbol's
    sliver, or ``None`` to leave the symbol out. Only the slivers are
    retained - never the payload, which is the whole point of projecting.

    A projection registers itself on its first call, so the very first read in
    a process can still cost a second parse if one consumer starts before the
    other has ever asked. Every rewrite of the file after that is a single
    read shared by both.

    A read failure is uncertainty, not emptiness: the last good result stands
    and the cache is left alone, so a torn write does not blank a detector's
    view of its levels.
    """
    key = str(key)
    _ai_state_projections[key] = project
    cache_key = str(path)
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return {}
    cached = _ai_state_cache.get(cache_key)
    if cached and cached.get("mtime") == mtime and key in cached["feeds"]:
        return cached["feeds"][key]
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        logging.warning("D1 level feed could not read ai_state: %s", exc)
        return cached["feeds"].get(key, {}) if cached else {}
    feeds: dict[str, dict[str, Any]] = {name: {} for name in _ai_state_projections}
    symbols = payload.get("symbols") if isinstance(payload, Mapping) else {}
    for symbol, entry in (symbols or {}).items():
        if not isinstance(entry, Mapping):
            continue
        name = str(symbol).strip().upper()
        for projection_key, projection in _ai_state_projections.items():
            try:
                sliver = projection(entry)
            except Exception:
                # One consumer's shaping bug must not take the other consumer's
                # levels down with it: the sliver is simply absent, which every
                # caller already treats as "nothing known here".
                logging.debug(
                    "ai_state projection %s failed for %s", projection_key, name,
                    exc_info=True,
                )
                continue
            if sliver is not None:
                feeds[projection_key][name] = sliver
    _ai_state_cache[cache_key] = {"mtime": mtime, "feeds": feeds}
    return feeds.get(key, {})


def reset_ai_state_cache() -> None:
    """Drop the shared ai_state parse cache. Tests only."""
    _ai_state_cache.clear()


def _load_ai_state_feed(path: Path) -> dict[str, dict[str, Any]]:
    """This module's sliver: per-symbol SMA and trendline prices."""
    return load_ai_state_projection(
        AI_STATE_SYMBOL_PROJECTION, _extract_symbol_entry, path
    )


def _load_store_levels(symbol: str, levels_dir: Path) -> list[dict[str, Any]]:
    path = level_store_path(levels_dir, symbol)
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []
    cached = _store_cache.get(key)
    if cached and cached.get("mtime") == mtime:
        return cached["levels"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return cached["levels"] if cached else []
    levels: list[dict[str, Any]] = []
    for level in (payload.get("levels") or []) if isinstance(payload, Mapping) else []:
        if not isinstance(level, Mapping):
            continue
        kind = str(level.get("kind") or "")
        if kind not in {"hv_horizontal", "cloud_flat"}:
            continue
        price = _coerce_float(level.get("price"))
        strength = _coerce_float(level.get("strength")) or 0.0
        if price is None or price <= 0 or strength < MIN_HORIZONTAL_STRENGTH:
            continue
        levels.append(
            {
                "price": price,
                "kind": kind,
                "strength": strength,
                "effective_range": level.get("effective_range"),
            }
        )
    _store_cache[key] = {"mtime": mtime, "levels": levels}
    return levels


def get_d1_extra_levels(
    symbol: str,
    *,
    reference_price: float | None = None,
    atr20: float | None = None,
    now: datetime | None = None,
    ai_state_path: Path | None = None,
    levels_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """The symbol's D1 major levels, shaped for ``observe_symbol(extra_levels=)``."""
    sym = str(symbol or "").strip().upper()
    if not sym:
        return []
    today = (now or datetime.now()).date()
    price = _coerce_float(reference_price)
    atr = _coerce_float(atr20)
    max_distance = price is not None and atr is not None and atr > 0

    def _near(value: float) -> bool:
        if not max_distance:
            return True
        return abs(value - price) <= MAX_DISTANCE_ATR * atr

    out: list[dict[str, Any]] = []
    entry = _load_ai_state_feed(Path(ai_state_path or MASTER_AVWAP_AI_STATE_FILE)).get(sym)
    if entry:
        as_of = _parse_date(entry.get("last_trade_date"))
        age_days = (today - as_of).days if as_of else None
        if age_days is not None and 0 <= age_days <= SMA_MAX_AGE_DAYS:
            for label, family, weight in D1_SMA_FAMILIES:
                value = entry["smas"].get(label)
                if value is not None and _near(value):
                    out.append(
                        {
                            "family": family,
                            "value": value,
                            "weight": weight,
                            "detail": {"label": label, "as_of": str(entry.get("last_trade_date") or "")},
                        }
                    )
        if age_days is not None and 0 <= age_days <= TRENDLINE_MAX_AGE_DAYS:
            for value in entry["trendlines"]:
                if _near(value):
                    out.append(
                        {
                            "family": D1_TRENDLINE_FAMILY,
                            "value": value,
                            "weight": D1_TRENDLINE_WEIGHT,
                            "detail": {"as_of": str(entry.get("last_trade_date") or "")},
                        }
                    )
    for level in _load_store_levels(sym, Path(levels_dir or MASTER_AVWAP_LEVELS_DIR)):
        if not level_is_effective_on(dict(level), today):
            continue
        if not _near(level["price"]):
            continue
        is_cloud = level["kind"] == "cloud_flat"
        out.append(
            {
                "family": D1_CLOUD_FAMILY if is_cloud else D1_HORIZONTAL_FAMILY,
                "value": level["price"],
                "weight": D1_CLOUD_WEIGHT if is_cloud else D1_HORIZONTAL_WEIGHT,
                "detail": {"kind": level["kind"], "strength": level["strength"]},
            }
        )
    return out
