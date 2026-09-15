"""Which side of the CURRENT AVWAPE a swing row's close sits on (WISHLIST 9).

Trader, WISHLIST item 9: *"stop putting up longs below avwape and shorts above
it."* The lead's ruling for the 2026-09-12 sweep is **display only**: this module
LABELS such a row and nothing here hides, re-orders, filters, scores or alerts.
Hiding is a detector decision and is not built. `plan.md` Phase 0.26 records that
ruling; the trader may overrule it.

Two bases, one verdict
----------------------

The packet's rule is the pure one - a LONG whose close sits under the anchor, or
a SHORT whose close sits over it, with a tolerance of **0** (a close exactly on
the line is the right side) and `None` whenever a number is missing, because an
unknown is never "wrong". That is :func:`wrong_side`.

The rows the desk's setups table and the AWAY digest actually read do not carry
those two numbers. Measured on 2026-09-12 against the live focus feed
(`data/runtime/master_avwap_focus.json`, 435 rows): **zero** rows carry
`current_close` or `current_avwape` - those two are `feature_snapshot` fields on
the TRACKER's daily marks, not scan-row fields. What every one of the 435 rows
does carry is `current_band_zone`, written by `runner.py` from
`legacy.get_band_context`, whose vocabulary is the ordered band levels
(``LOWER_3 .. LOWER_1``, ``VWAP``, ``UPPER_1 .. UPPER_3``). A zone names the pair
of levels the close sits between - ``"LOWER_1 to VWAP"`` is under the anchor,
``"VWAP to UPPER_1"`` is over it - and a close exactly on a level is that level
alone. So the zone answers the same question the two numbers would, without
touching `legacy.py` to add a field.

:func:`read_row` therefore reads the numbers when a caller has them and the band
zone when it does not, and says which it used, so a surface never prints a price
it did not read. `favorite_zone` is deliberately NOT a fallback: it names the
zone the SETUP wants, not where price is.

Pure by design: no I/O, no Qt, no clock. It is called from `paint` on the Qt
thread, so it stays a dict lookup and one small string split.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple


#: What the setups-table chip says. One string, so the chip, the tooltip and the
#: digest tag can never drift apart.
WRONG_SIDE_LABEL = "wrong side"

#: What the AWAY digest appends after the symbol.
WRONG_SIDE_TAG = f"[{WRONG_SIDE_LABEL}]"

ABOVE = "above"
BELOW = "below"
ON = "on"

BASIS_PRICES = "prices"
BASIS_BAND_ZONE = "band_zone"

#: The anchor's own names in the band vocabulary. `VWAP` is what
#: `get_band_context` writes; `AVWAPE` is what the setup-side labels use.
_ANCHOR_NAMES = {"VWAP", "AVWAPE", "AVWAP"}

#: The zone is normalised to upper case before it is split, so the separator is
#: matched in upper case too - splitting `"LOWER_1 TO VWAP"` on `" to "` was the
#: first defect these tests caught.
_ZONE_SEPARATOR = " TO "


class WrongSideRead(NamedTuple):
    """One row's answer: the verdict plus what it was read from."""

    wrong: bool
    side: str
    position: str
    basis: str
    close: float | None = None
    avwape: float | None = None
    zone: str = ""


def normalize_side(side: Any) -> str:
    """`LONG` / `SHORT`, or `""` when the row does not say."""
    text = str(side or "").strip().upper()
    return text if text in {"LONG", "SHORT"} else ""


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # NaN is not a reading


def _position(close: float, avwape: float) -> str:
    if close > avwape:
        return ABOVE
    if close < avwape:
        return BELOW
    return ON


def _is_wrong(side: str, position: str) -> bool:
    return (side == "LONG" and position == BELOW) or (side == "SHORT" and position == ABOVE)


def wrong_side(side: Any, close: Any, avwape: Any) -> bool | None:
    """Is this row on the wrong side of its current anchor?

    `True` for a LONG below the AVWAPE or a SHORT above it, `False` otherwise -
    including a close sitting exactly ON the line, which is the right side
    (tolerance 0). `None` when the side or either number is missing: an unknown
    is never "wrong".
    """
    normalized = normalize_side(side)
    close_value = _number(close)
    avwape_value = _number(avwape)
    if not normalized or close_value is None or avwape_value is None:
        return None
    return _is_wrong(normalized, _position(close_value, avwape_value))


def _zone_offsets(zone: Any) -> list[int] | None:
    """The band zone as signed offsets from the anchor, or `None` if unreadable.

    ``"LOWER_1 to VWAP"`` -> ``[-1, 0]``; ``"UPPER_2 to UPPER_1"`` -> ``[2, 1]``
    (a SHORT row's pair is written high-to-low); ``"VWAP"`` -> ``[0]``.
    """
    text = str(zone or "").strip().upper()
    if not text:
        return None
    offsets: list[int] = []
    for token in text.split(_ZONE_SEPARATOR):
        name = token.strip()
        if name in _ANCHOR_NAMES:
            offsets.append(0)
            continue
        head, _, tail = name.partition("_")
        if head not in {"UPPER", "LOWER"} or not tail.isdigit():
            return None
        offsets.append(int(tail) if head == "UPPER" else -int(tail))
    return offsets or None


def zone_position(zone: Any) -> str:
    """`above` / `below` / `on` the anchor, or `""` when the zone says nothing."""
    offsets = _zone_offsets(zone)
    if not offsets:
        return ""
    if all(offset >= 0 for offset in offsets) and any(offset > 0 for offset in offsets):
        return ABOVE
    if all(offset <= 0 for offset in offsets) and any(offset < 0 for offset in offsets):
        return BELOW
    if all(offset == 0 for offset in offsets):
        return ON
    # A pair straddling the anchor is not a zone `get_band_context` writes; it
    # would mean the close is on both sides, so answer nothing.
    return ""


def wrong_side_from_zone(side: Any, zone: Any) -> bool | None:
    """:func:`wrong_side`'s verdict from the band zone the scan row carries."""
    normalized = normalize_side(side)
    position = zone_position(zone)
    if not normalized or not position:
        return None
    return _is_wrong(normalized, position)


def _row_zone(row: Mapping[str, Any]) -> str:
    """The CLOSE's zone - top level first, then the setup candidate's trigger.

    The same precedence `legacy._priority_current_band_zone` reads, minus its
    `favorite_zone` fallback: that field names the zone the setup WANTS, and
    reading it as the close's position would badge rows by their setup shape.
    """
    zone = row.get("current_band_zone")
    if not str(zone or "").strip():
        candidate = row.get("setup_candidate")
        trigger = candidate.get("trigger") if isinstance(candidate, Mapping) else None
        if isinstance(trigger, Mapping):
            zone = trigger.get("current_band_zone")
    return str(zone or "").strip()


def read_row(row: Any) -> WrongSideRead | None:
    """One scan row's reading, or `None` when the row cannot answer.

    Numbers first (`current_close` / `current_avwape`), band zone second. Never
    raises on a malformed row - an unreadable row is simply unknown.
    """
    if not isinstance(row, Mapping):
        return None
    side = normalize_side(row.get("side"))
    if not side:
        return None

    close = _number(row.get("current_close"))
    avwape = _number(row.get("current_avwape"))
    if close is not None and avwape is not None:
        position = _position(close, avwape)
        return WrongSideRead(
            wrong=_is_wrong(side, position),
            side=side,
            position=position,
            basis=BASIS_PRICES,
            close=close,
            avwape=avwape,
        )

    zone = _row_zone(row)
    position = zone_position(zone)
    if not position:
        return None
    return WrongSideRead(
        wrong=_is_wrong(side, position),
        side=side,
        position=position,
        basis=BASIS_BAND_ZONE,
        zone=zone,
    )


def is_wrong_side_row(row: Any) -> bool:
    """`True` only when the row was READ and read wrong-side. Never raises."""
    try:
        read = read_row(row)
    except Exception:  # noqa: BLE001 - a label never costs its surface
        return False
    return bool(read is not None and read.wrong)


def tooltip_text(read: WrongSideRead | None) -> str:
    """What the trader reads on the chip, or `""` when there is nothing to say.

    `LONG below AVWAPE 412.50 (close 409.10)` when the numbers were read;
    `LONG below AVWAPE (band zone VWAP to LOWER_1)` when the zone was, because a
    surface never shows a price nothing measured.
    """
    if read is None or not read.wrong:
        return ""
    if read.basis == BASIS_PRICES and read.avwape is not None and read.close is not None:
        return (
            f"{read.side} {read.position} AVWAPE {read.avwape:.2f} "
            f"(close {read.close:.2f})"
        )
    return f"{read.side} {read.position} AVWAPE (band zone {read.zone})"
