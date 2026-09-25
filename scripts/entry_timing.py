"""P1-6 6d: entry-timing chips for D1 rows, from the chart-watch stores. Read only.

A D1 row gets one chip when the trader (or the desk) has a Pullback alert on
that name - `pullback`, and its pre-rename spelling `h1_ema_bounce`:

* fired: ``timing: H1 15-EMA held 10:30`` - the most recent trigger that fired,
  named, with its bar time (the date is added when it is not today);
* armed: ``timing: pullback armed`` - waiting, nothing fired yet.

Sources, both read only: the watch store (`alert_chart_watches.json`: armed
watches and the `fired` map of the SMA legs) and the tail of the review-event
store's `watch_fired` rows (the H1 leg retires its watch when it fires, so its
fire is only there). Declined watches are ignored. Nothing is armed, changed or
pushed from here.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

PULLBACK_KINDS = {"pullback", "h1_ema_bounce"}

TRIGGER_LABELS = {
    "h1_ema15_bounce": "H1 15-EMA held",
    "sma_reclaim_lrsi": "SMA reclaim + LRSI",
    "reclaim_then_lrsi": "reclaim then LRSI",
    "sma_retest": "SMA retest",
}

#: How far back a fire still says something about the entry.
FIRE_LOOKBACK_DAYS = 7
#: Bytes read from the END of each review-event file (fires are recent).
TAIL_BYTES = 1_000_000

ARMED = "armed"
FIRED = "fired"


def _moment(value: Any) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(value or "").strip())
    except ValueError:
        return None
    return moment.replace(tzinfo=None) if moment.tzinfo is not None else moment


def _side(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text in {"LONG", "SHORT"} else ""


def trigger_label(trigger: str, timeframe: str = "") -> str:
    """`H1 15-EMA held`, or `SMA reclaim + LRSI M30` (the timeframe named when it adds one)."""
    label = TRIGGER_LABELS.get(str(trigger or ""), str(trigger or "trigger").replace("_", " "))
    timeframe = str(timeframe or "").strip().upper()
    if timeframe and timeframe not in label:
        label = f"{label} {timeframe}"
    return label


def _chip_text(state: str, label: str = "", at: datetime | None = None, *, today: date) -> str:
    if state != FIRED:
        return "timing: pullback armed"
    clock = ""
    if at is not None:
        clock = at.strftime("%H:%M") if at.date() == today else at.strftime("%m-%d %H:%M")
    return f"timing: {label} {clock}".rstrip()


def build_timing(
    watches: Iterable[Mapping[str, Any]] | None,
    fires: Iterable[Mapping[str, Any]] | None,
    *,
    today: date,
) -> dict[tuple[str, str], dict[str, Any]]:
    """`{(SYMBOL, SIDE-or-""): {"state", "text", "at"}}`. A blank side covers both sides."""
    events: dict[tuple[str, str], tuple[datetime, str]] = {}
    armed: set[tuple[str, str]] = set()
    oldest = datetime.combine(today - timedelta(days=FIRE_LOOKBACK_DAYS), datetime.min.time())

    def note_fire(key: tuple[str, str], at: datetime | None, label: str) -> None:
        if at is None or at < oldest:
            return
        held = events.get(key)
        if held is None or at > held[0]:
            events[key] = (at, label)

    for watch in watches or ():
        if not isinstance(watch, Mapping) or watch.get("declined"):
            continue
        if str(watch.get("kind") or "") not in PULLBACK_KINDS:
            continue
        symbol = str(watch.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        key = (symbol, _side(watch.get("side")))
        armed.add(key)
        fired = watch.get("fired")
        for fire_key, bar_time in (fired.items() if isinstance(fired, Mapping) else ()):
            trigger, _at, timeframe = str(fire_key).partition("@")
            note_fire(key, _moment(bar_time), trigger_label(trigger, timeframe))

    for row in fires or ():
        if not isinstance(row, Mapping) or str(row.get("action") or "") != "watch_fired":
            continue
        detail = row.get("detail") if isinstance(row.get("detail"), Mapping) else {}
        if str(detail.get("kind") or "") not in PULLBACK_KINDS:
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        at = _moment(detail.get("confirm_bar_dt") or detail.get("bar_dt") or row.get("ts"))
        label = trigger_label(str(detail.get("trigger") or ""), str(detail.get("timeframe") or ""))
        note_fire((symbol, _side(row.get("side"))), at, label)

    answer: dict[tuple[str, str], dict[str, Any]] = {}
    for key in armed | set(events):
        if key in events:
            at, label = events[key]
            answer[key] = {"state": FIRED, "at": at, "text": _chip_text(FIRED, label, at, today=today)}
        else:
            answer[key] = {"state": ARMED, "at": None, "text": _chip_text(ARMED, today=today)}
    return answer


def timing_for(mapping: Mapping[tuple[str, str], Mapping[str, Any]] | None, symbol: str, side: str):
    """The chip for one row: its own side first, then a side-less watch; a fire beats an arm."""
    if not mapping:
        return None
    symbol = str(symbol or "").strip().upper()
    candidates = [mapping.get((symbol, _side(side))), mapping.get((symbol, ""))]
    candidates = [item for item in candidates if item]
    if not candidates:
        return None
    fired = [item for item in candidates if item.get("state") == FIRED]
    if fired:
        return max(fired, key=lambda item: item.get("at") or datetime.min)
    return candidates[0]


# ------------------------------------------------------------------ readers (worker only)
def read_watch_rows(path: Path) -> list[dict[str, Any]]:
    """The raw watch rows of `alert_chart_watches.json`, or [] when unreadable."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = payload.get("watches") if isinstance(payload, dict) else None
    return [row for row in rows or () if isinstance(row, dict)]


def read_fire_rows(paths: Iterable[Path], *, tail_bytes: int = TAIL_BYTES) -> list[dict[str, Any]]:
    """`watch_fired` rows from the END of each review-event file. A torn first line is skipped."""
    rows: list[dict[str, Any]] = []
    for path in paths or ():
        try:
            with open(path, "rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                handle.seek(max(0, size - tail_bytes))
                data = handle.read()
        except OSError:
            continue
        lines = data.splitlines()
        if size > tail_bytes and lines:
            lines = lines[1:]
        for line in lines:
            if b"watch_fired" not in line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_timing(*, today: date | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    """Read both live stores (read only) and build the map. Worker threads only."""
    from project_paths import ALERT_CHART_WATCHES_FILE
    from review_events import review_event_sources

    return build_timing(
        read_watch_rows(Path(ALERT_CHART_WATCHES_FILE)),
        read_fire_rows(review_event_sources()),
        today=today or date.today(),
    )
