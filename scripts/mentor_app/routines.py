"""P18 D: asks -> routines. What the trader asks for, when, counted into his usual reads. Pure, Qt-free.

An *ask* is one user turn: its PT 30-minute bucket, weekday, the pack set its answer used (the reply turn's
``tool_calls_json``, dropped attachments excluded) and a question kind. The night's deterministic half writes
``mentor_routines.json``: per bucket and day type (weekday and weekend counted apart), the packs asked on
``MIN_DAYS`` or more of the last ``SESSION_DAYS`` days of that type with any ask, with the count. The app prefetches a bucket's packs when it starts and shows a quiet chip;
``/routine`` prints the table and ``/forget routine <bucket>`` hides a line (persisted in ``app_state``).
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
ROUTINES_FILE = "mentor_routines.json"
ROUTINES_SCHEMA = "mentor_routines_v1"
SESSION_DAYS = 10
MIN_DAYS = 5
#: app_state key: the buckets the trader told to forget (JSON list of "HH:MM").
FORGOTTEN_KEY = "routines:forgotten"
BUCKET_RE = re.compile(r"^\d{2}:(?:00|30)$")
KINDS = (
    ("pre_trade", ("gate_pack",)),
    ("tape", ("regime_pack", "bars_pack", "rs_pack", "alerts_pack")),
    ("me", ("journal_pack", "tilt_pack", "reads_pack", "habits_pack", "mirror_pack", "recaps_pack")),
    ("pick", ("pick_pack", "news_pack", "fundamentals_pack", "earnings_pack", "veto_pack")),
)


def _utc(stamp: Any) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(stamp or "").strip())
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def bucket_of(moment: datetime) -> str:
    local = moment.astimezone(PT)
    return f"{local.hour:02d}:{0 if local.minute < 30 else 30:02d}"


def kind_of(packs: Iterable[str]) -> str:
    names = set(packs)
    for kind, members in KINDS:
        if names & set(members):
            return kind
    return "other" if names else "chat"


def _packs(tool_calls_json: Any) -> list[str]:
    try:
        calls = json.loads(tool_calls_json or "[]")
    except (TypeError, ValueError):
        return []
    return sorted({str(call.get("name")) for call in calls if isinstance(call, Mapping) and call.get("name")
                   and not call.get("dropped")})


def asks_from_turns(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The asks view over turns (oldest first): each user turn with the packs of the reply that followed it."""
    out: list[dict[str, Any]] = []
    ordered = sorted(rows, key=lambda row: int(row.get("id") or 0))
    for index, row in enumerate(ordered):
        if row.get("role") != "user":
            continue
        at = _utc(row.get("ts_utc"))
        if at is None:
            continue
        reply = next((later for later in ordered[index + 1:] if later.get("role") in ("assistant", "user")), None)
        packs = _packs(reply.get("tool_calls_json")) if reply is not None and reply.get("role") == "assistant" else []
        local = at.astimezone(PT)
        out.append({"turn_id": int(row["id"]), "day_pt": local.date().isoformat(), "bucket_pt": bucket_of(at),
                    "weekday": local.strftime("%a"), "packs": packs, "kind": kind_of(packs)})
    return out


WEEKDAY, WEEKEND = "weekday", "weekend"


def day_type(day: Any) -> str:
    """``weekday`` or ``weekend`` for an ISO date or a datetime (PT)."""
    if isinstance(day, datetime):
        return WEEKEND if day.astimezone(PT).weekday() >= 5 else WEEKDAY
    return WEEKEND if date.fromisoformat(str(day)[:10]).weekday() >= 5 else WEEKDAY


def find_routines(asks: Sequence[Mapping[str, Any]], session: str, *, days: int = SESSION_DAYS,
                  min_days: int = MIN_DAYS) -> dict[str, Any]:
    """Per bucket and day type, the packs asked on ``min_days``+ of the last ``days`` days of that type."""
    window = {kind: sorted({ask["day_pt"] for ask in asks if ask["day_pt"] <= session
                            and day_type(ask["day_pt"]) == kind})[-days:] for kind in (WEEKDAY, WEEKEND)}
    seen: dict[tuple[str, str, str], set[str]] = {}
    for ask in asks:
        kind = day_type(ask["day_pt"])
        if ask["day_pt"] not in window[kind]:
            continue
        for name in ask["packs"]:
            seen.setdefault((kind, ask["bucket_pt"], name), set()).add(ask["day_pt"])
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (kind, bucket, name), on in seen.items():
        if len(on) >= min_days:
            buckets.setdefault((kind, bucket), []).append({"name": name, "days": len(on)})
    routines = [{"bucket": bucket, "day_type": kind, "packs": sorted(packs, key=lambda p: (-p["days"], p["name"]))}
                for (kind, bucket), packs in sorted(buckets.items(), key=lambda kv: (kv[0][0] != WEEKDAY, kv[0][1]))]
    return {"schema": ROUTINES_SCHEMA, "session_date": session, "session_days": window[WEEKDAY],
            "weekend_days": window[WEEKEND], "min_days": min_days, "routines": routines}


def routine_id(row: Mapping[str, Any]) -> str:
    """``routine:0630`` on weekdays, ``routine:we:0630`` on weekends."""
    prefix = "we:" if row.get("day_type") == WEEKEND else ""
    return f"routine:{prefix}{str(row.get('bucket') or '').replace(':', '')}"


def due(payload: Mapping[str, Any], forgotten: Iterable[str], now: datetime) -> dict[str, Any] | None:
    """The routine row for ``now``'s half hour and day type (a weekday routine never fires on a Saturday)."""
    bucket, kind = bucket_of(now), day_type(now)
    return next((r for r in visible(payload, forgotten)
                 if r.get("bucket") == bucket and r.get("day_type", WEEKDAY) == kind), None)


def read_routines(path: Path | str | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def live_path() -> Path | None:
    from mentor_app.memory import _digests_root

    root = _digests_root()
    return None if root is None else Path(root) / ROUTINES_FILE


def routine_line(new: Mapping[str, Any], old: Mapping[str, Any]) -> str:
    """The coach brief's one line, only when the routine table changed: "You usually ask X at 06:30 PT"."""
    def table(payload: Mapping[str, Any]) -> list[tuple[str, str, tuple[str, ...]]]:
        return [(str(r.get("bucket")), str(r.get("day_type") or WEEKDAY), tuple(p["name"] for p in r.get("packs") or ()))
                for r in payload.get("routines") or () if isinstance(r, Mapping)]

    rows = table(new)
    if not rows or rows == table(old):
        return ""
    parts = [f"{', '.join(names)} at {bucket} PT" + (" on weekends" if kind == WEEKEND else "")
             for bucket, kind, names in rows[:3]]
    return "You usually ask " + "; ".join(parts)


def visible(payload: Mapping[str, Any], forgotten: Iterable[str]) -> list[dict[str, Any]]:
    gone = set(forgotten)
    return [dict(r) for r in payload.get("routines") or () if isinstance(r, Mapping) and r.get("bucket") not in gone]


def table_text(payload: Mapping[str, Any], forgotten: Iterable[str]) -> str:
    """What ``/routine`` prints."""
    rows = visible(payload, forgotten)
    if not payload:
        return "No routine yet: the night counts what you ask (a pack asked on 5 of your last 10 days, same half hour)."
    days = len(payload.get("session_days") or ())
    lines = [f"**Your usual asks** (night of {payload.get('session_date', '?')}, over {days} session day(s); "
             f"a pack asked on {payload.get('min_days', MIN_DAYS)}+ of them)", ""]
    for row in rows:
        packs = ", ".join(f"{p['name']} ({p['days']} days)" for p in row.get("packs") or ())
        when = " (weekends)" if row.get("day_type") == WEEKEND else ""
        lines.append(f"- [{routine_id(row)}] {row['bucket']} PT{when}: {packs}")
    if not rows:
        lines.append("- none (or all forgotten)")
    lines.append("")
    lines.append("`/forget routine 06:30` hides a line.")
    return "\n".join(lines)


def forgotten_list(raw: str | None) -> list[str]:
    try:
        value = json.loads(raw or "[]")
    except ValueError:
        return []
    return [str(item) for item in value if BUCKET_RE.match(str(item))] if isinstance(value, list) else []


__all__ = ["ROUTINES_FILE", "asks_from_turns", "bucket_of", "find_routines", "forgotten_list", "kind_of",
           "read_routines", "routine_line", "table_text", "visible"]
