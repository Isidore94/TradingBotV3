"""S14: the long STUDY families, defined once. Pure, no I/O.

Two study keys over one D1 scan row, as the row knew them on its scan date:

* ``leader_pullback_long`` (F21): LONG, ``pct_from_current_vwap`` in [-10, -3],
  and the family is ``top_pattern_tracking`` or the sector is Technology.
* ``band_bounce_leader_long`` (F23 follow-up): LONG ``avwap_band_bounce``,
  sector Technology or Healthcare, ``rs_vs_industry`` in the top tercile of
  that scan session's LONG rows, and SPY above its 20-day SMA.

A study family is research on the P1-4 4e path: it is tagged, graded and
searched, never scored. Nothing here reaches a detector, a score, an alert,
Focus, the queue or ``review_policy.json``. PROMOTION TO A SCORED FAMILY IS
ASK-FIRST (golden fixtures, the trader's word, one at a time).

Missing input is unknown (None), never False; a row that is not LONG is False.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

LEADER_PULLBACK_LONG = "leader_pullback_long"
BAND_BOUNCE_LEADER_LONG = "band_bounce_leader_long"
#: Every study family, in report order.
STUDY_FAMILIES = (LEADER_PULLBACK_LONG, BAND_BOUNCE_LEADER_LONG)

LEADER_PULLBACK_VWAP_RANGE = (-10.0, -3.0)
LEADER_PULLBACK_FAMILY = "top_pattern_tracking"
LEADER_PULLBACK_SECTOR = "Technology"

BAND_BOUNCE_FAMILY = "avwap_band_bounce"
BAND_BOUNCE_SECTORS = ("Technology", "Healthcare")
#: Fewer known RS values than this in a session and its tercile is unknown.
RS_TERCILE_MIN_ROWS = 3

#: The facets the permutation search runs a study family on, first (S4 + S14).
STUDY_SEARCH_FACETS = ("spy_trend", "trend20", "htf_trend_4h")

#: How a row carries its study families in a CSV cell.
TAG_SEPARATOR = ";"


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value).strip()


def _flag(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _text(value).lower()
    if text in {"true", "1", "1.0", "yes"}:
        return True
    if text in {"false", "0", "0.0", "no"}:
        return False
    return None


def leader_pullback_long(side: str, family: str, pct_from_current_vwap: Any, sector: Any) -> bool | None:
    """The F21 key; None when an input it needs is unknown."""
    if _text(side).upper() != "LONG":
        return False
    pct = _num(pct_from_current_vwap)
    if pct is None:
        return None
    low, high = LEADER_PULLBACK_VWAP_RANGE
    if not (low <= pct <= high):
        return False
    if _text(family) == LEADER_PULLBACK_FAMILY:
        return True
    sector_text = _text(sector)
    if not sector_text:
        return None
    return sector_text == LEADER_PULLBACK_SECTOR


def band_bounce_leader_long(
    side: str, family: str, sector: Any, rs_top_tercile: bool | None, spy_above_sma20: Any
) -> bool | None:
    """The F23 cell; None when an input it needs is unknown."""
    if _text(side).upper() != "LONG" or _text(family) != BAND_BOUNCE_FAMILY:
        return False
    sector_text = _text(sector)
    if sector_text and sector_text not in BAND_BOUNCE_SECTORS:
        return False
    spy_up = _flag(spy_above_sma20)
    if spy_up is False or rs_top_tercile is False:
        return False
    if not sector_text or spy_up is None or rs_top_tercile is None:
        return None
    return True


def rs_top_tercile(value: Any, session_values: list[float] | None) -> bool | None:
    """True when fewer than a third of the session's LONG RS values beat ``value``."""
    number = _num(value)
    if number is None or not session_values or len(session_values) < RS_TERCILE_MIN_ROWS:
        return None
    above = sum(1 for other in session_values if other > number)
    return above < len(session_values) / 3.0


def session_rs_values(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[float]]:
    """``{scan date: [rs_vs_industry of every LONG row]}``, the tercile's cross-section.

    ``rows`` are one scan session's representative rows (the session horizon
    grain); each needs ``side``, ``scan_date`` and ``rs_vs_industry``. Only that
    session's own rows count, so the tercile uses nothing later than the session.
    """
    out: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if _text(row.get("side")).upper() != "LONG":
            continue
        number = _num(row.get("rs_vs_industry"))
        day = _text(row.get("scan_date"))[:10]
        if number is not None and day:
            out[day].append(number)
    return dict(out)


def study_families(row: Mapping[str, Any], session_values: Mapping[str, list[float]]) -> list[str]:
    """The study families one representative row is in; an unknown key is not a member.

    ``row`` carries ``side``, ``setup_family``, ``scan_date``, ``sector``,
    ``pct_from_current_vwap``, ``rs_vs_industry`` and ``spy_above_sma20``.
    """
    side = _text(row.get("side")).upper()
    family = _text(row.get("setup_family"))
    day = _text(row.get("scan_date"))[:10]
    tercile = rs_top_tercile(row.get("rs_vs_industry"), session_values.get(day))
    keys = (
        (LEADER_PULLBACK_LONG,
         leader_pullback_long(side, family, row.get("pct_from_current_vwap"), row.get("sector"))),
        (BAND_BOUNCE_LEADER_LONG,
         band_bounce_leader_long(side, family, row.get("sector"), tercile, row.get("spy_above_sma20"))),
    )
    return [name for name, value in keys if value is True]


def tag_text(families: Iterable[str]) -> str:
    return TAG_SEPARATOR.join(families)


def parse_tag(value: Any) -> list[str]:
    """The study families a CSV cell names; unknown names are dropped."""
    return [name for name in _text(value).split(TAG_SEPARATOR) if name in STUDY_FAMILIES]
