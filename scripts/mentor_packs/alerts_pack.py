"""Alerts pack: what the bot alerted on a day, from the files the desk already writes. Read-only.

M5 bounce alerts: ``INTRADAY_BOUNCES_FILE`` (``intraday_bounces.csv``; ``time_local`` is the
desk machine's wall clock). D1 alerts: the Alert Center's fired events in the review-event
log (``ALERT_REVIEW_EVENTS_DIR`` + the legacy ``ALERT_REVIEW_EVENTS_FILE``; actions
``d1_event_fired`` / ``level_fired`` / ``watch_fired``) and the Master AVWAP bucket upgrades
(``MASTER_AVWAP_D1_UPGRADE_ALERTS_FILE``). Never the 650 MB M5 outcome store.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

from mentor_packs.registry import Pack, make_pack

NAME = "alerts_pack"
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "What the bot alerted on a day: M5 bounce alerts (time ET, symbol, side, bounce types, tier) and D1 "
            "alerts (D1 events, level and watch hits, Master AVWAP bucket upgrades), with a summary of counts "
            "by kind and side and how many are in the trader's book or liked."
        ),
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "only this ticker (optional)"},
            "day": {"type": "string", "description": "today (default), yesterday or YYYY-MM-DD"},
            "kind": {"type": "string", "enum": ["all", "d1", "m5"], "description": "all (default), d1 or m5"},
        }, "required": []},
    },
}

ET = ZoneInfo("America/New_York")
D1_ACTIONS = ("d1_event_fired", "level_fired", "watch_fired")
#: Rows per kind; the summary always counts every alert.
MAX_PER_KIND = 25


@dataclass(frozen=True)
class Sources:
    """Where each section reads from; tests pass fakes, the app uses :func:`live_sources`."""

    m5_rows: Callable[[str], list[Mapping[str, Any]]]
    d1_events: Callable[[str], list[Mapping[str, Any]]]
    upgrades: Callable[[], Mapping[str, Any] | None]
    #: The zone naive desk wall-clock times are in.
    local_tz: Callable[[], Any]
    book: Callable[[], Mapping[str, str]]


def _live_m5_rows(day: str) -> list[dict[str, Any]]:
    from project_paths import INTRADAY_BOUNCES_FILE

    path = Path(INTRADAY_BOUNCES_FILE)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle) if str(row.get("trade_date") or "")[:10] == day]


def _review_event_files() -> list[Path]:
    from project_paths import ALERT_REVIEW_EVENTS_DIR, ALERT_REVIEW_EVENTS_FILE

    files = sorted(Path(ALERT_REVIEW_EVENTS_DIR).glob("*.jsonl")) if Path(ALERT_REVIEW_EVENTS_DIR).exists() else []
    legacy = Path(ALERT_REVIEW_EVENTS_FILE)
    return files + ([legacy] if legacy.exists() else [])


def read_d1_events(day: str, files: Iterable[Path]) -> list[dict[str, Any]]:
    """Fired D1 alert events of one trade date from review-event JSONL files (substring-filtered first)."""
    needle = f'"trade_date": "{day}"'
    out: list[dict[str, Any]] = []
    for path in files:
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if needle not in line or "_fired" not in line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if record.get("action") in D1_ACTIONS and str(record.get("trade_date") or "") == day:
                        out.append(record)
        except OSError:
            continue
    return out


def _live_upgrades() -> Mapping[str, Any] | None:
    from project_paths import MASTER_AVWAP_D1_UPGRADE_ALERTS_FILE

    try:
        payload = json.loads(Path(MASTER_AVWAP_D1_UPGRADE_ALERTS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _live_local_tz() -> Any:
    import market_session

    return market_session.get_market_local_timezone()[0]


def _live_book() -> dict[str, str]:
    from mentor_packs import rs_pack

    return rs_pack._live_book()


def live_sources() -> Sources:
    return Sources(
        m5_rows=_live_m5_rows,
        d1_events=lambda day: read_d1_events(day, _review_event_files()),
        upgrades=_live_upgrades,
        local_tz=_live_local_tz,
        book=_live_book,
    )


def _previous_weekday(day: date) -> date:
    back = day - timedelta(days=1)
    while back.weekday() >= 5:
        back -= timedelta(days=1)
    return back


def resolve_day(day: str, moment: datetime) -> str:
    text = str(day or "today").strip().lower()
    today = moment.astimezone(ET).date()
    if text in ("", "today"):
        return today.isoformat()
    if text == "yesterday":
        return _previous_weekday(today).isoformat()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    raise ValueError(f"day must be today, yesterday or YYYY-MM-DD, not {day!r}")


def _et_time(day: str, clock: str, local_tz: Any) -> str:
    """``HH:MM`` ET for a naive desk wall-clock time on ``day``; the raw text when unreadable."""
    raw = str(clock or "").strip()
    try:
        stamp = datetime.fromisoformat(raw) if "T" in raw else datetime.fromisoformat(f"{day}T{raw}")
    except ValueError:
        return raw or "--:--"
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=local_tz)
    return f"{stamp.astimezone(ET):%H:%M}"


def _clean(symbol: Any) -> str:
    return str(symbol or "").strip().upper().lstrip("$")


def _side(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text in ("LONG", "SHORT") else ""


def _yours(sym: str, book: Mapping[str, str], likes: set[str]) -> str:
    if sym in book:
        return f" [book {book[sym] or '?'}]"
    return " [liked]" if sym in likes else ""


def build(symbol: str = "", day: str = "today", kind: str = "all", liked: Any = (), *,
          now: datetime | None = None, sources: Sources | None = None) -> Pack:
    """Build the alerts pack. File reads only: call it on a worker."""
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.astimezone()
    session = resolve_day(day, moment)
    kind = str(kind or "all").strip().lower()
    kind = kind if kind in ("d1", "m5") else "all"
    sym_filter = _clean(symbol)
    src = sources or live_sources()
    local_tz = src.local_tz()
    book = {_clean(k): _side(v) for k, v in (src.book() or {}).items()}
    likes = {_clean(item[0] if isinstance(item, (list, tuple)) else item) for item in liked or () if item}
    found: dict[str, list[dict[str, Any]]] = {"m5": [], "d1": []}
    if kind in ("all", "m5"):
        for row in src.m5_rows(session) or ():
            sym = _clean(row.get("symbol"))
            if not sym or (sym_filter and sym != sym_filter):
                continue
            tier = str(row.get("tier") or "").strip()
            composite = str(row.get("composite_r") or "").strip()
            found["m5"].append({
                "et": _et_time(session, str(row.get("time_local") or ""), local_tz), "symbol": sym,
                "side": _side(row.get("direction")),
                "what": f"M5 bounce {row.get('bounce_types') or '?'}"
                        + (f", tier {tier}" if tier else "") + (f" ({composite}R)" if composite else ""),
                "price": "",
            })
    if kind in ("all", "d1"):
        for event in src.d1_events(session) or ():
            sym = _clean(event.get("symbol"))
            if not sym or (sym_filter and sym != sym_filter):
                continue
            detail = event.get("detail") if isinstance(event.get("detail"), Mapping) else {}
            what = str(detail.get("message") or detail.get("kind") or event.get("action") or "").strip()
            found["d1"].append({"et": _et_time(session, str(event.get("ts") or ""), local_tz), "symbol": sym,
                                "side": _side(event.get("side")),
                                "what": f"{str(event.get('action')).replace('_', ' ')}: {what}", "price": ""})
        payload = src.upgrades() or {}
        if str(payload.get("run_date") or "") == session:
            for sym_raw, entry in sorted((payload.get("symbols") or {}).items()):
                sym = _clean(sym_raw)
                if not isinstance(entry, Mapping) or (sym_filter and sym != sym_filter):
                    continue
                events = [e for e in entry.get("bucket_upgrade_events") or () if isinstance(e, Mapping)]
                label = ", ".join(sorted({str(e.get("alert_label") or e.get("label") or "") for e in events} - {""}))
                level = next((e.get("level") for e in events if e.get("level") is not None), None)
                found["d1"].append({
                    "et": "scan", "symbol": sym, "side": _side(entry.get("side")),
                    "what": (f"D1 bucket upgrade to {entry.get('priority_bucket') or '?'}"
                             + (f" ({label})" if label else "")),
                    "price": f" @ {float(level):.2f}" if isinstance(level, (int, float)) else "",
                })
    total = len(found["m5"]) + len(found["d1"])
    scope = f" for {sym_filter}" if sym_filter else ""
    if not total:
        what = {"all": "M5 or D1", "m5": "M5", "d1": "D1"}[kind]
        return make_pack(NAME, [{"id": f"alert:{session}:none", "kind": "none",
                                 "text": f"No {what} alerts{scope} on {session} in the desk's alert files"}])
    rows: list[dict[str, Any]] = []
    parts = []
    for which in ("m5", "d1"):
        items = found[which]
        if not items and kind != "all":
            continue
        longs = sum(1 for item in items if item["side"] == "LONG")
        shorts = sum(1 for item in items if item["side"] == "SHORT")
        parts.append(f"{which.upper()} {len(items)} ({longs} long, {shorts} short)")
    alerted = {item["symbol"] for items in found.values() for item in items}
    in_book = sorted(alerted & set(book))
    in_likes = sorted((alerted & likes) - set(book))
    rows.append({
        "id": f"alert:{session}:summary", "kind": "summary",
        "text": (f"Alerts{scope} on {session}: " + "; ".join(parts)
                 + f"; in your book: {', '.join(in_book) or 'none'}; liked: {', '.join(in_likes) or 'none'}"),
    })
    for which in ("m5", "d1"):
        items = sorted(found[which], key=lambda item: item["et"], reverse=True)
        for n, item in enumerate(items[:MAX_PER_KIND], 1):
            rows.append({
                "id": f"alert:{session}:{which}:{n}", "kind": f"alert_{which}", "symbol": item["symbol"],
                "text": (f"{item['et']} ET {item['symbol']} {item['side'] or '?'} {item['what']}{item['price']}"
                         f"{_yours(item['symbol'], book, likes)}"),
            })
        if len(items) > MAX_PER_KIND:
            rows.append({"id": f"alert:{session}:{which}:more", "kind": "more",
                         "text": f"{len(items) - MAX_PER_KIND} older {which.upper()} alerts not listed"})
    return make_pack(NAME, rows)


FIXTURE_NOW = datetime(2026, 9, 30, 20, 0, tzinfo=timezone.utc)


def fixture_sources() -> Sources:
    m5 = [
        {"time_local": "07:15:05", "trade_date": "2026-09-30", "symbol": "ALL", "direction": "short",
         "bounce_types": "ema_21", "tier": "B", "composite_r": "0.129"},
        {"time_local": "09:02:00", "trade_date": "2026-09-30", "symbol": "NVDA", "direction": "long",
         "bounce_types": "eod_vwap", "tier": "", "composite_r": ""},
    ]
    d1 = [{"action": "d1_event_fired", "symbol": "CL", "side": "SHORT", "trade_date": "2026-09-30",
           "ts": "2026-09-30T06:31:14.29", "detail": {"kind": "ema15_reject", "message": "D1 15EMA rejection (short)"}}]
    upgrades = {"run_date": "2026-09-30", "symbols": {"CE": {
        "side": "SHORT", "priority_bucket": "near_favorite_zone",
        "bucket_upgrade_events": [{"alert_label": "Trendline break", "level": 45.1169}]}}}
    return Sources(
        m5_rows=lambda day: [row for row in m5 if row["trade_date"] == day],
        d1_events=lambda day: [row for row in d1 if row["trade_date"] == day],
        upgrades=lambda: upgrades,
        local_tz=lambda: ZoneInfo("America/Los_Angeles"),
        book={"ALL": "SHORT"}.copy,
    )


def fixture() -> Pack:
    return build(day="today", liked=["CE"], now=FIXTURE_NOW, sources=fixture_sources())
