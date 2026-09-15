"""What a setups-table row says about compression (packet PCT-3, item 2).

Trader, 2026-09-15: *"we need a way to measure for compression as the most
COMMON veto I have is a compression veto."* The scan has measured it all along -
`legacy.summarize_anchor_compression` scores every priority row 0-3 on three ATR
ratios and quietly docks the score for it - and the trader has never seen one of
those numbers. PCT-3's answer is **measure and show, then tune**: this module is
the pure reader behind the amber `compressed` chip on the setups table.

Display only. Nothing here hides, re-orders, filters, scores, alerts or writes.
A chip is a label; the row is still there, still in the same place, still with
the same score. The threshold and penalty decision comes later, from the
calibration report (`scripts/compression_calibration.py`), by the trader's word.

Pure by design: no I/O, no Qt, no clock. `SetupTableDelegate.paint` calls
:func:`read_row` once per visible bucket cell per repaint, so it stays a handful
of dict lookups.

The string trap
---------------

``bool("False") is True``. A row that has been through a CSV, a JSON round-trip
or a report line spells its flag half a dozen ways, and every one of them has to
read the same. :func:`read_flag` is therefore the ONE place a flag is turned
into a boolean, and an unreadable spelling is **not compressed** rather than a
guess - a label the trader cannot trust is worse than no label.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple


#: What the chip says. One string, so the chip and its tooltip cannot drift.
COMPRESSED_LABEL = "compressed"

#: The theme token the chip is painted in - the same `caution` amber WS-WS's
#: `wrong side` badge uses, because both are "read this row twice", not "no".
COMPRESSED_TOKEN = "caution"

#: `summarize_anchor_compression` scores three ratios, one point each.
MAX_COMPRESSION_SCORE = 3

#: The rule whose numbers these are. A row stamped with anything else is still
#: read - the version is carried so a later re-tune is legible, not so this
#: reader can refuse a row it understands.
RULE_VERSION = "anchor_compression_v1"

_TRUE_WORDS = {"true", "t", "yes", "y", "1"}
_FALSE_WORDS = {"false", "f", "no", "n", "0", "", "none", "null", "nan"}

#: Every field the reader looks at. A row carrying none of them has nothing to
#: say (it has not been through the `ai_state` merge yet) and reads as `None`.
_FIELDS = (
    "compression_flag",
    "compression_score",
    "compression_penalty",
    "compression_note",
    "compression_stdev_atr_ratio",
    "compression_range_atr_ratio",
    "compression_close_range_atr_ratio",
    "compression_rule_version",
)


class CompressionRatios(NamedTuple):
    """The three ATR ratios, in the order the note prints them."""

    stdev_atr: float | None
    range_atr: float | None
    close_range_atr: float | None


class CompressionRead(NamedTuple):
    """One row's compression reading."""

    flag: bool
    score: int | None
    penalty: int | None
    note: str
    ratios: CompressionRatios
    rule_version: str = ""


def read_flag(value: Any) -> bool:
    """`True` only for a spelling that unambiguously means compressed.

    `True`, `1`, `"1"`, `"true"`, `"TRUE"`, `"yes"` are compressed; `False`,
    `0`, `"0"`, `"false"`, `"no"`, `""` and `None` are not; anything else is
    not, because an unreadable flag is not evidence.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value == value and value != 0  # NaN is not a reading
    text = str(value).strip().lower()
    if text in _TRUE_WORDS:
        return True
    if text in _FALSE_WORDS:
        return False
    return False


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # NaN is not a reading


def _whole(value: Any) -> int | None:
    number = _number(value)
    return int(round(number)) if number is not None else None


def read_row(row: Any) -> CompressionRead | None:
    """This row's compression reading, or `None` when it cannot answer.

    `None` means "nothing to say", which is what every setups-table row looks
    like until the `ai_state` merge has run - never "not compressed as a fact".
    Never raises: `paint` calls this once per visible cell per repaint, so an
    unreadable row is an answer, not an exception.
    """
    if not isinstance(row, Mapping):
        return None
    if not any(field in row for field in _FIELDS):
        return None
    return CompressionRead(
        flag=read_flag(row.get("compression_flag")),
        score=_whole(row.get("compression_score")),
        penalty=_whole(row.get("compression_penalty")),
        note=str(row.get("compression_note") or ""),
        ratios=CompressionRatios(
            stdev_atr=_number(row.get("compression_stdev_atr_ratio")),
            range_atr=_number(row.get("compression_range_atr_ratio")),
            close_range_atr=_number(row.get("compression_close_range_atr_ratio")),
        ),
        rule_version=str(row.get("compression_rule_version") or ""),
    )


def is_compressed_row(row: Any) -> bool:
    """`True` only when the row was READ and read compressed. Never raises."""
    try:
        read = read_row(row)
    except Exception:  # noqa: BLE001 - a label never costs its surface
        return False
    return bool(read is not None and read.flag)


def _ratio_text(label: str, value: float | None) -> str:
    return f"{label} {value:.2f} ATR" if value is not None else f"{label} not measured"


def tooltip_text(read: CompressionRead | None) -> str:
    """What the trader reads on the chip, or `""` when there is nothing to say.

    ``compressed - score 3/3, stdev 0.52 ATR, range 2.10 ATR, close-range 1.40
    ATR, penalty 10``, then the scan's own note on a second line when it has
    one. An unflagged row and an unread row both say nothing: the chip is only
    painted when the flag is set, so a tooltip without a chip would be a label
    with no badge.
    """
    if read is None or not read.flag:
        return ""
    score = "not measured" if read.score is None else f"{read.score}/{MAX_COMPRESSION_SCORE}"
    parts = [
        f"{COMPRESSED_LABEL} - score {score}",
        _ratio_text("stdev", read.ratios.stdev_atr),
        _ratio_text("range", read.ratios.range_atr),
        _ratio_text("close-range", read.ratios.close_range_atr),
        f"penalty {read.penalty}" if read.penalty is not None else "penalty not measured",
    ]
    text = ", ".join(parts)
    note = read.note.strip()
    return f"{text}\n{note}" if note else text
