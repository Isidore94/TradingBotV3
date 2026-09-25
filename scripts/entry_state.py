"""P1-6 6a: is an M5 alert's entry still takeable? Pure; display only.

One state per alert, from the alert's own entry/stop and the bot's CACHED M5
bars, with `live_alert_results` arithmetic (same bar window, same zones):

* ``gone``     - a completed bar touched the stop, or the last completed close
                 is at or beyond +1R.
* ``improved`` - a LATER bar (after the alert's own bar) came back to the entry
                 level and the stop held: a better or equal price was offered.
* ``valid``    - price between the entry and +1R, the level held.
* ``unknown``  - no entry/stop, no risk, or no completed bars yet.

Completed bars only. Nothing here changes an alert, a detector or a score.
"""

from __future__ import annotations

from datetime import datetime, tzinfo
from typing import Any, Iterable, Mapping

import live_alert_results as lar

VALID = "valid"
IMPROVED = "improved"
GONE = "gone"
UNKNOWN = "unknown"

STATES = (VALID, IMPROVED, GONE, UNKNOWN)

#: The price, in R from the entry, past which the entry is chased and gone.
GONE_AT_R = 1.0


def _usable_bars(
    entry: Mapping[str, Any],
    bars: Iterable[Mapping[str, Any]] | None,
    now: datetime,
    zone: tzinfo,
) -> list[tuple[datetime, float, float, float, float]]:
    """Completed bars from the alert's own 5-minute bucket onward, that day only."""
    now_ny = lar.to_ny(now, zone)
    start = lar._bucket(entry["received_at"])
    usable = []
    for bar in bars or ():
        try:
            begins = lar.to_ny(bar["dt"], zone)
            row = (begins, float(bar["open"]), float(bar["high"]), float(bar["low"]),
                   float(bar["close"]))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if begins < start or begins.date() != start.date() or begins + lar.BAR_SPAN > now_ny:
            continue
        usable.append(row)
    usable.sort(key=lambda row: row[0])
    return usable


def entry_state(
    entry: Mapping[str, Any] | None,
    bars: Iterable[Mapping[str, Any]] | None,
    now: datetime,
    *,
    naive_zone: tzinfo | None = None,
) -> dict[str, Any]:
    """`{"state", "reason", "at", "r"}` for one `live_alert_results` entry."""
    answer: dict[str, Any] = {"state": UNKNOWN, "reason": "no entry/stop", "at": None, "r": None}
    if not entry or entry.get("status") != lar.OK:
        if entry:
            answer["reason"] = str(entry.get("reason") or "no entry/stop")
        return answer
    zone = naive_zone or lar.desk_zone()
    usable = _usable_bars(entry, bars, now, zone)
    if not usable:
        answer["reason"] = "no completed bars"
        return answer
    price, stop, risk = float(entry["entry"]), float(entry["stop"]), float(entry["risk"])
    long_side = entry["side"] == "LONG"
    alert_bucket = lar._bucket(entry["received_at"])
    retouched_at = None
    for begins, _open, high, low, _close in usable:
        if (low <= stop) if long_side else (high >= stop):
            answer.update(state=GONE, reason="stop hit", at=begins)
            return answer
        later = begins > alert_bucket
        if later and retouched_at is None and ((low <= price) if long_side else (high >= price)):
            retouched_at = begins
    last_close = usable[-1][4]
    r_now = round(((last_close - price) if long_side else (price - last_close)) / risk, 2)
    answer["r"] = r_now
    if r_now >= GONE_AT_R:
        answer.update(state=GONE, reason="beyond +1R", at=usable[-1][0])
        return answer
    if retouched_at is not None:
        answer.update(state=IMPROVED, reason="later touch of the entry", at=retouched_at)
        return answer
    answer.update(state=VALID, reason="between entry and +1R, level held", at=usable[-1][0])
    return answer


def chip_text(state: Mapping[str, Any] | str | None) -> str:
    """The short chip: `valid`, `improved`, `gone`, or `unknown`."""
    value = state.get("state") if isinstance(state, Mapping) else state
    return value if value in STATES else UNKNOWN


def chip_detail(state: Mapping[str, Any] | None) -> str:
    """One tooltip line: the chip, why, and the bar it was read on."""
    if not isinstance(state, Mapping):
        return "entry: unknown (not measured yet)"
    text = f"entry: {chip_text(state)}"
    reason = str(state.get("reason") or "")
    at = state.get("at")
    clock = at.strftime("%H:%M") if isinstance(at, datetime) else ""
    if reason:
        text += f" ({reason}{' ' + clock if clock else ''})"
    return text
