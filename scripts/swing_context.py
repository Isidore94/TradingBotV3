"""Which M5 alerts sit on a D1 swing setup (trader, 2026-09-23). Pure, no Qt.

The M5 alert bar shows ``· D1 A ★`` on a row whose symbol AND side is also a
row in the setups table: the swing grade (`setup_grades`, the same key the
setups table's Bucket cell uses) and a star when the trader claimed that pick.
PRESENTATION ONLY: nothing here is read by a detector, a score, an alert
decision, Focus, the queue or `review_policy.json`.

A missing grade is shown as ``New`` - never an invented grade. A symbol whose
only setup is on the OTHER side gets nothing: a long swing does not back a
short intraday entry.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

import setup_grades

#: The bucket key a claimed row carries (`ui.services.claimed_setup_rows`).
CLAIMED_BUCKET = "claimed_like"

#: Rows without swing context sort after every row with it.
_NO_CONTEXT = (1, 1, setup_grades.UNGRADED_RANK + 1)


def normalize_side(side: object) -> str:
    """LONG / SHORT / '' - the setups table and the alerts spell it both ways."""
    text = str(side or "").strip().upper()
    if text.startswith("LONG"):
        return "LONG"
    if text.startswith("SHORT"):
        return "SHORT"
    return ""


def _raw(row: Any) -> Mapping[str, Any]:
    raw = getattr(row, "raw", None)
    return raw if isinstance(raw, Mapping) else {}


def _bucket_keys(row: Any) -> set[str]:
    keys = {str(getattr(row, "bucket", "") or "").strip().lower()}
    extra = _raw(row).get("bucket_keys")
    if isinstance(extra, (list, tuple, set)):
        keys.update(str(key).strip().lower() for key in extra)
    keys.discard("")
    return keys


def row_is_claimed(row: Any) -> bool:
    """The master panel's own test (`_row_is_claimed`): the bucket and an id."""
    return CLAIMED_BUCKET in _bucket_keys(row) and bool(
        str(_raw(row).get("claimed_setup_id") or "").strip()
    )


def claimed_keys_from_rows(rows: Iterable[Any]) -> set[tuple[str, str]]:
    """`{(SYMBOL, SIDE)}` for every claimed row. The panel already merged the
    claims into its rows, so this never reads the claims file."""
    out: set[tuple[str, str]] = set()
    for row in rows or ():
        if row_is_claimed(row):
            symbol = str(getattr(row, "symbol", "") or "").strip().upper()
            side = normalize_side(getattr(row, "side", ""))
            if symbol and side:
                out.add((symbol, side))
    return out


def _family(row: Any) -> str:
    raw = _raw(row)
    family = str(raw.get("setup_family") or "").strip()
    if family:
        return family
    setup_id = str(raw.get("claimed_setup_id") or "").strip()
    if setup_id:
        return setup_id
    return str(getattr(row, "bucket", "") or "").strip()


def build_swing_context(
    rows: Iterable[Any],
    grade_for: Callable[[Any], Mapping[str, Any] | None],
    claimed_keys: Iterable[tuple[str, str]] = (),
) -> dict[tuple[str, str], dict[str, Any]]:
    """`{(SYMBOL, SIDE): {"grade", "family", "claimed"}}` for every setup row.

    ``grade_for(row)`` is `SetupTableModel.grade_cell_for`: a cell dict, or None
    before grades load (then the grade is ``None`` and shows as ``New``). Several
    rows on one name+side keep the best-graded one; ``claimed`` is true if any
    of them, or ``claimed_keys``, says so.
    """
    claimed = {
        (str(symbol or "").strip().upper(), normalize_side(side))
        for symbol, side in (claimed_keys or ())
    }
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows or ():
        symbol = str(getattr(row, "symbol", "") or "").strip().upper()
        side = normalize_side(getattr(row, "side", ""))
        if not symbol or not side:
            continue
        try:
            cell = grade_for(row)
        except Exception:  # noqa: BLE001 - a display hint never costs the map
            cell = None
        grade = str(cell.get("grade") or "") if isinstance(cell, Mapping) else ""
        grade = grade if grade in setup_grades.GRADES else None
        key = (symbol, side)
        entry = {
            "grade": grade,
            "family": _family(row),
            "claimed": key in claimed or row_is_claimed(row),
        }
        held = out.get(key)
        if held is None:
            out[key] = entry
            continue
        if setup_grades.sort_rank(entry["grade"]) < setup_grades.sort_rank(held["grade"]):
            entry["claimed"] = entry["claimed"] or held["claimed"]
            out[key] = entry
        else:
            held["claimed"] = held["claimed"] or entry["claimed"]
    return out


def context_for(
    mapping: Mapping[tuple[str, str], Mapping[str, Any]] | None, symbol: object, side: object
) -> Mapping[str, Any] | None:
    """The context for one alert's symbol+side, or None."""
    if not mapping:
        return None
    return mapping.get((str(symbol or "").strip().upper(), normalize_side(side)))


def _grade_text(ctx: Mapping[str, Any]) -> str:
    return str(ctx.get("grade") or setup_grades.NEW)


def suffix(ctx: Mapping[str, Any] | None) -> str:
    """``· D1 A ★`` - or ``""`` when the alert has no swing setup."""
    if not ctx:
        return ""
    text = f"· D1 {_grade_text(ctx)}"
    if ctx.get("claimed"):
        text += " ★"
    return text


def tooltip_line(ctx: Mapping[str, Any] | None) -> str:
    """``D1 setup: <family>, grade A, claimed`` - or ``""``."""
    if not ctx:
        return ""
    family = str(ctx.get("family") or "").strip() or "setup"
    line = f"D1 setup: {family}, grade {_grade_text(ctx)}"
    if ctx.get("claimed"):
        line += ", claimed"
    return line


def sort_key(ctx: Mapping[str, Any] | None) -> tuple[int, int, int]:
    """Swing-backed first; among them claimed first, then by swing grade."""
    if not ctx:
        return _NO_CONTEXT
    return (0, 0 if ctx.get("claimed") else 1, setup_grades.sort_rank(ctx.get("grade")))
