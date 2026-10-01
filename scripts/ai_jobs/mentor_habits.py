"""P18 B: the trader's habits, found by counting what he says (journal lines + his chat turns). Deterministic.

Read-only over the Trade Mentor chat store (``mode=ro``) and, for the rest-of-day check, the trade journal. The
night publishes ``mentor_habits.json`` in its own ai_store folder (temp-and-rename; a failed write keeps the last
good file). A habit is a mood tag or a normalized phrase of ``PHRASE_WORDS`` words seen on ``MIN_DAYS`` or more
days of the last ``HABIT_DAYS``; each carries first/last seen, days seen, example ids (``journal:<id>``,
``turn:<id>``) and the context it co-occurs with (after a loss, before the open, late day, the regime). Counts and
observations only, never a rule. The model half (``ask_habits``, <= ``MAX_HABIT_TOKENS``) only words the top five.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
PT = ZoneInfo("America/Los_Angeles")
HABITS_FILE = "mentor_habits.json"
HABITS_SCHEMA = "mentor_habits_v1"
HABIT_DAYS = 30
MIN_DAYS = 3
PHRASE_WORDS = 3
MAX_EXAMPLES = 3
MAX_HABITS = 5
MAX_HABIT_TOKENS = 300
MAX_ITEM_CHARS = 280
#: Inbox: a habit seen on MIN_DAYS+ days and followed by a red rest of day on INBOX_RED_DAYS+ of them.
INBOX_RED_DAYS = 3
HABITS_PROMPT_VERSION = "mentor_habits_v1"
HABITS_SCHEMA_NAME = "tradingbot_mentor_habits"
HABITS_INSTRUCTIONS = (
    "Word the trader's habits below for his morning coach brief. For each habit in habit_candidates (at most "
    "five, by its key) write one short observation of what he keeps saying or feeling and when, citing ids "
    "copied exactly from allowed_evidence_ids (the habit id and its example ids). Copy numbers; never compute "
    "one. These are observations, never rules: never suggest an order, a size, or a rule."
)
_STOP = frozenset((
    "a an and are as at be but by for from had has have i i'm im is it it's its me my of on or so that the then "
    "this to was we with you just really very too also all not no do did does got get gonna going what when how "
    "there here they them he she his her our your yeah ok okay like").split())
_WORD = re.compile(r"[a-z][a-z']*")


class HabitsRejected(ValueError):
    """The habits reply cited an id the night does not carry, or was not an object with a habits array."""


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _utc(stamp: Any) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(stamp or "").strip())
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _connect_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(path: Path, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    with closing(_connect_ro(path)) as conn:
        try:
            return [dict(row) for row in conn.execute(sql, tuple(params)).fetchall()]
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc) or "no such column" in str(exc):
                return []
            raise


# ---------------------------------------------------------------- what he said
def said(path: Path, first: str, last: str) -> list[dict[str, Any]]:
    """Every journal line (ET day) and user turn (PT day) with ``first <= day <= last``, oldest first.

    Each: ``{id, kind, day, at (aware), text, tags, after_loss, bucket_et, regime}``. Nothing is capped here."""
    out: list[dict[str, Any]] = []
    for row in _rows(path, "SELECT * FROM journal_entries WHERE day_et >= ? AND day_et <= ? ORDER BY id", (first, last)):
        try:
            tags = [str(tag) for tag in json.loads(row.get("mood_tags_json") or "[]")]
        except ValueError:
            tags = []
        at = _utc(row.get("ts_utc"))
        regime = _text(row.get("regime"))
        regime = regime.split(": ", 1)[-1].split(" since ", 1)[0] if regime else ""
        out.append({"id": f"journal:{row['id']}", "kind": "journal", "day": str(row["day_et"]), "at": at,
                    "text": _text(row.get("text")), "tags": tags,
                    "after_loss": _text(row.get("last_trade_text")).endswith(" stop"),
                    "after_win": _text(row.get("last_trade_text")).endswith(" win"),
                    "bucket_et": _text(row.get("time_bucket_et")), "regime": regime})
    lo = (date.fromisoformat(first) - timedelta(days=1)).isoformat()
    hi = (date.fromisoformat(last) + timedelta(days=2)).isoformat()
    for row in _rows(path, "SELECT id, ts_utc, text FROM turns WHERE role = 'user' AND ts_utc >= ? AND ts_utc < ? "
                     "ORDER BY id", (lo, hi)):
        at = _utc(row.get("ts_utc"))
        day = at.astimezone(PT).date().isoformat() if at else ""
        if not (first <= day <= last):
            continue
        local = at.astimezone(ET)
        out.append({"id": f"turn:{row['id']}", "kind": "turn", "day": day, "at": at, "text": _text(row.get("text")),
                    "tags": [], "after_loss": False, "after_win": False,
                    "bucket_et": f"{local.hour:02d}:{0 if local.minute < 30 else 30:02d}", "regime": ""})
    out.sort(key=lambda item: (item["at"] or datetime.min.replace(tzinfo=timezone.utc), item["id"]))
    return out


def phrases(text: str) -> set[str]:
    """Normalized runs of PHRASE_WORDS words (at most one stopword, never at either end)."""
    words = _WORD.findall(str(text or "").lower().replace("’", "'"))
    found = set()
    for i in range(len(words) - PHRASE_WORDS + 1):
        run = words[i:i + PHRASE_WORDS]
        if run[0] in _STOP or run[-1] in _STOP or sum(1 for w in run if w in _STOP) > 1:
            continue
        found.add(" ".join(run))
    return found


def day_facts(items: Iterable[Mapping[str, Any]], session: str) -> dict[str, Any]:
    """One day: mood-tag counts, journal lines after a loss vs after a win, and ET hour buckets."""
    today = [item for item in items if item["day"] == session]
    journal = [item for item in today if item["kind"] == "journal"]
    tags = Counter(tag for item in journal for tag in item["tags"])
    hours = Counter(item["bucket_et"][:2] for item in today if item["bucket_et"])
    return {"session_date": session, "journal_lines": len(journal),
            "turns": sum(1 for item in today if item["kind"] == "turn"), "mood_tags": dict(sorted(tags.items())),
            "after_loss": sum(1 for item in journal if item["after_loss"]),
            "after_win": sum(1 for item in journal if item.get("after_win")),
            "hours_et": dict(sorted(hours.items()))}


def _context(hits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    regimes = Counter(item["regime"] for item in hits if item["regime"])
    return {"after_loss": sum(1 for item in hits if item["after_loss"]),
            "before_open": sum(1 for item in hits if item["bucket_et"] and item["bucket_et"] < "09:30"),
            "late_day": sum(1 for item in hits if item["bucket_et"] >= "14:00"),
            "regimes": dict(sorted(regimes.items()))}


def find_habits(items: Sequence[Mapping[str, Any]], *,
                red_after: Callable[[str, datetime], bool | None] | None = None) -> list[dict[str, Any]]:
    """Tags and phrases seen on MIN_DAYS+ distinct days, most days first. ``red_after(day, at)`` says whether the
    rest of that day was red after ``at`` (None = unknown, never counted)."""
    hits: dict[str, list[Mapping[str, Any]]] = {}
    for item in items:
        for tag in item["tags"]:
            hits.setdefault(f"tag:{tag}", []).append(item)
        for phrase in phrases(item["text"]):
            hits.setdefault(f"phrase:{phrase}", []).append(item)
    habits = []
    for key, rows in hits.items():
        days = sorted({row["day"] for row in rows})
        if len(days) < MIN_DAYS:
            continue
        red_days = set()
        if red_after is not None:
            for row in rows:
                if row["day"] not in red_days and row["at"] is not None and red_after(row["day"], row["at"]) is True:
                    red_days.add(row["day"])
        kind, words = key.split(":", 1)
        label = f"mood tag '{words}'" if kind == "tag" else f"the phrase \"{words}\""
        habit_id = "habit:" + re.sub(r"[^a-z0-9_]+", "_", key.replace(":", "_"))
        context = _context(rows)
        habits.append({
            "key": key, "id": habit_id, "first_seen": days[0], "last_seen": days[-1], "days_seen": len(days),
            "count": len(rows), "examples": [row["id"] for row in rows[-MAX_EXAMPLES:]],
            "quotes": [row["text"][:160] for row in rows[-MAX_EXAMPLES:]], "context": context,
            "red_rest_of_day_days": len(red_days),
            "inbox_ok": len(days) >= MIN_DAYS and len(red_days) >= INBOX_RED_DAYS,
            "text": (f"You said {label} on {len(days)} days ({len(rows)} times) since {days[0]}; after a loss "
                     f"{context['after_loss']}, before the open {context['before_open']}, late day "
                     f"{context['late_day']}" + (f"; rest of day red on {len(red_days)} of those days"
                                                 if red_after is not None else ""))[:MAX_ITEM_CHARS],
        })
    habits.sort(key=lambda h: (-h["days_seen"], -h["count"], h["key"]))
    return habits


def red_after_from_journal(journal: Path | str) -> Callable[[str, datetime], bool | None]:
    """``red_after(day, at)``: the net PnL of trades CLOSED on ``day`` (ET) after ``at`` is below zero; None when
    nothing closed after it (unknown, never "green")."""
    from mentor_packs import journal_read

    cache: dict[str, list[tuple[datetime, float]]] = {}

    def closes(day: str) -> list[tuple[datetime, float]]:
        if day not in cache:
            found = []
            for trade in journal_read.read_trades(journal, since=day):
                at = journal_read.parse_time(trade.get("closed_at"))
                value = journal_read.pnl(trade)
                if at is not None and value is not None and at.astimezone(ET).date().isoformat() == day:
                    found.append((at, value))
            cache[day] = found
        return cache[day]

    def red_after(day: str, at: datetime) -> bool | None:
        later = [value for when, value in closes(day) if when > at]
        return None if not later else sum(later) < 0

    return red_after


# ---------------------------------------------------------------- registry
def registry_path(root: Path) -> Path:
    return Path(root) / HABITS_FILE


def read_registry(root: Path) -> dict[str, Any]:
    try:
        payload = json.loads(registry_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def publish_registry(root: Path, session: str, habits: Sequence[Mapping[str, Any]], built_utc: str,
                     facts: Mapping[str, Any]) -> Path:
    """Write ``mentor_habits.json``; a key's ``first_seen`` never moves later than the last file said."""
    from ai_jobs import digest

    old = {str(h.get("key")): h for h in read_registry(root).get("habits") or () if isinstance(h, Mapping)}
    rows = []
    for habit in habits:
        row = dict(habit)
        before = str((old.get(row["key"]) or {}).get("first_seen") or "")
        if before:
            row["first_seen"] = min(before, row["first_seen"])
        rows.append(row)
    payload = {"schema": HABITS_SCHEMA, "session_date": session, "built_utc": built_utc, "window_days": HABIT_DAYS,
               "min_days": MIN_DAYS, "day": dict(facts), "habits": rows}
    path = registry_path(root)
    return digest._publish(path, json.dumps(payload, sort_keys=True, indent=2, default=str))


def habit_rows(habits: Sequence[Mapping[str, Any]], limit: int = MAX_HABITS) -> list[dict[str, Any]]:
    """The top habits as citable night inputs (``habit:<key>``), with their example ids."""
    return [{"id": h["id"], "key": h["key"], "examples": list(h["examples"]),
             "text": f"{h['text']}; examples: {', '.join(h['examples'])}"} for h in habits[:limit]]


def fact_habits(habits: Sequence[Mapping[str, Any]], limit: int = MAX_HABITS) -> list[dict[str, Any]]:
    """The brief's habits section worded by code (the model's wording replaces it when it answers)."""
    return [{"key": h["key"], "text": h["text"], "evidence_refs": [h["id"], *h["examples"][:2]],
             "days_seen": h["days_seen"], "first_seen": h["first_seen"]} for h in habits[:limit]]


# ---------------------------------------------------------------- the model half
_HABIT = {
    "type": "object", "additionalProperties": False, "required": ["key", "text", "evidence_refs"],
    "properties": {"key": {"type": "string"}, "text": {"type": "string", "maxLength": MAX_ITEM_CHARS},
                   "evidence_refs": {"type": "array", "items": {"type": "string"}}},
}
HABITS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["habits"],
    "properties": {"habits": {"type": "array", "items": _HABIT}},
}


def check_habits(reply: Any, habits: Sequence[Mapping[str, Any]],
                 allowed: Iterable[str]) -> tuple[list[dict[str, Any]], int]:
    """(kept, dropped). A foreign id rejects the reply; an unknown key or an uncited item drops; a habit the model
    left out keeps its code wording, after the model's order."""
    if not isinstance(reply, Mapping) or not isinstance(reply.get("habits"), (list, tuple)):
        raise HabitsRejected("the habits reply carried no habits array")
    top = {h["key"]: h for h in habits[:MAX_HABITS]}
    allowed_ids = set(allowed)
    kept: list[dict[str, Any]] = []
    dropped = 0
    for index, row in enumerate(reply["habits"]):
        if not isinstance(row, Mapping):
            raise HabitsRejected(f"habit {index} was not an object")
        refs = [str(ref).strip() for ref in (row.get("evidence_refs") or ()) if str(ref).strip()]
        for ref in refs:
            if ref not in allowed_ids:
                raise HabitsRejected(f"habit {index} cited {ref!r}, which tonight does not carry")
        habit = top.get(_text(row.get("key")))
        text = _text(row.get("text"))
        if (habit is None or not refs or not text or len(text) > MAX_ITEM_CHARS
                or any(item["key"] == habit["key"] for item in kept)):
            dropped += 1
            continue
        kept.append({"key": habit["key"], "text": text, "evidence_refs": refs, "days_seen": habit["days_seen"],
                     "first_seen": habit["first_seen"]})
    for item in fact_habits(habits):
        if all(k["key"] != item["key"] for k in kept):
            kept.append(item)
    return kept[:MAX_HABITS], dropped


def ask_habits(request: Callable[..., Mapping[str, Any]], *, model: str, post: Callable[..., Any],
               habits: Sequence[Mapping[str, Any]], session: str) -> tuple[list[dict[str, Any]], str, int]:
    """One call (<= MAX_HABIT_TOKENS) to word the top habits. Returns (kept, model, dropped); raises on a bad reply."""
    from ai_jobs.mentor_review import TIMEOUT_SECONDS, capped_post

    rows = habit_rows(habits)
    allowed = [row["id"] for row in rows] + [ref for row in rows for ref in row["examples"]]
    evidence = {"session_date": session, "habit_candidates": rows, "allowed_evidence_ids": allowed,
                "instructions": HABITS_INSTRUCTIONS, "package_id": f"mentor-habits:{session}"}
    result = request(provider="local", model=model, api_key="", evidence=evidence, timeout_seconds=TIMEOUT_SECONDS,
                     post=capped_post(post, model=model, cap=MAX_HABIT_TOKENS), schema=HABITS_JSON_SCHEMA,
                     schema_name=HABITS_SCHEMA_NAME, prompt_version=HABITS_PROMPT_VERSION)
    kept, dropped = check_habits((result or {}).get("summary"), habits, allowed)
    return kept, _text((result or {}).get("model")) or model, dropped


def window(session: str, days: int = HABIT_DAYS) -> tuple[str, str]:
    day = date.fromisoformat(session)
    return (day - timedelta(days=days - 1)).isoformat(), day.isoformat()


__all__ = ["HABITS_FILE", "HabitsRejected", "ask_habits", "check_habits", "day_facts", "fact_habits", "find_habits",
           "habit_rows", "phrases", "publish_registry", "read_registry", "red_after_from_journal", "said", "window"]
