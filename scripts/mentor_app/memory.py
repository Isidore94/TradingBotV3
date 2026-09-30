"""Morning memory: the night's digests and the trader's own notes, as one byte-stable block. Qt-free.

``load`` reads the last DIGEST_LIMIT ``mentor_day_digest`` publications from the ai_store and the
newest NOTE_LIMIT live ``profile_notes`` (call it off the Qt thread). ``render`` sorts the items
oldest first and drops the oldest until the block fits MEMORY_BUDGET_TOKENS, so the same inputs
always give the same bytes and Ollama's KV prefix stays warm. Every item carries a citable id
``mem:<kind>:<id>``. Nothing here infers a note: notes come only from ``/remember``. (Plan
lines are a separate store: ``plan_infer`` writes AI-marked lines to the trading plan.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from mentor_app.chat_model import estimate_tokens

PT = ZoneInfo("America/Los_Angeles")
MEMORY_BUDGET_TOKENS = 1500
DIGEST_LIMIT = 5
NOTE_LIMIT = 50
#: A digest question's ref id is the day's number + this offset (items use 0..4).
QUESTION_OFFSET = 50
RULE_PREFIX = "rule:"
STILL_TRUE_DAYS = 7
HEAD = "# Memory\nNight digests and the trader's own notes, oldest first. Cite by id."


@dataclass(frozen=True)
class MemoryItem:
    id: str
    kind: str
    ref_id: int
    day: str
    text: str

    def line(self) -> str:
        return f"[{self.id}] ({self.day}) {self.text}"


@dataclass
class Memory:
    text: str = ""
    items: list[MemoryItem] = field(default_factory=list)
    dropped: int = 0


def _clean(text: Any) -> str:
    return " ".join(str(text or "").split())


def _pt_day(stamp: Any) -> str:
    try:
        moment = datetime.fromisoformat(str(stamp or ""))
    except ValueError:
        return ""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(PT).date().isoformat()


def digest_ref(day: str, index: int) -> int:
    return int(str(day).replace("-", "")) * 100 + int(index)


def digest_items(publications: Iterable[Mapping[str, Any]]) -> list[MemoryItem]:
    out: list[MemoryItem] = []
    for payload in publications:
        day = str(payload.get("session_date") or "")[:10]
        if len(day) != 10:
            continue
        for index, row in enumerate(payload.get("digest") or ()):
            ref = digest_ref(day, index)
            out.append(MemoryItem(f"mem:digest:{ref}", "digest", ref, day, _clean(row.get("text"))))
        for index, row in enumerate(payload.get("open_questions") or ()):
            ref = digest_ref(day, QUESTION_OFFSET + index)
            out.append(MemoryItem(f"mem:digest:{ref}", "digest", ref, day, "open question: " + _clean(row.get("text"))))
    return [item for item in out if item.text]


def note_items(notes: Iterable[Mapping[str, Any]]) -> list[MemoryItem]:
    return [
        MemoryItem(f"mem:note:{int(row['id'])}", "note", int(row["id"]), _pt_day(row.get("ts_utc")), _clean(row.get("text")))
        for row in notes
        if not row.get("retired_utc") and _clean(row.get("text"))
    ]


def render(items: Sequence[MemoryItem], *, budget_tokens: int = MEMORY_BUDGET_TOKENS) -> Memory:
    """Oldest first; the oldest are dropped until the block fits. Deterministic for the same items."""
    ordered = sorted(items, key=lambda item: (item.day, item.kind, item.ref_id))
    if not ordered:
        return Memory()
    kept = list(ordered)
    while kept and estimate_tokens("\n".join([HEAD, *(item.line() for item in kept)])) > budget_tokens:
        kept.pop(0)
    if not kept:
        return Memory(dropped=len(ordered))
    return Memory("\n".join([HEAD, *(item.line() for item in kept)]), kept, len(ordered) - len(kept))


def _digests_root() -> Path | None:
    from ai_jobs import store as ai_store

    try:
        return ai_store.digests_dir(create=False)
    except ValueError:
        return None


def load(store: Any, *, ai_root: Path | str | None = None, budget_tokens: int = MEMORY_BUDGET_TOKENS) -> Memory:
    """The memory block from the ai_store's digests and the store's live notes. Off the Qt thread."""
    from ai_jobs.mentor_review import DIGEST_STEM, read_published

    publications: list[dict[str, Any]] = []
    try:
        root = Path(ai_root) if ai_root is not None else _digests_root()
        if root is not None:
            publications = read_published(root, DIGEST_STEM, limit=DIGEST_LIMIT)
    except Exception:  # noqa: BLE001 - no digests is an empty memory, never a crash
        logging.warning("Trade Mentor memory: the night digests could not be read", exc_info=True)
    notes = store.profile_notes(limit=NOTE_LIMIT)
    return render([*digest_items(publications), *note_items(notes)], budget_tokens=budget_tokens)


def load_facts(ai_root: Path | str | None = None, *, days: int = 7) -> list[dict[str, Any]]:
    """The newest ``days`` ``mentor_day_facts`` publications (for /scorecard); [] when none or unreadable."""
    from ai_jobs.mentor_review import FACTS_STEM, read_published

    try:
        root = Path(ai_root) if ai_root is not None else _digests_root()
        return read_published(root, FACTS_STEM, limit=days) if root is not None else []
    except Exception:  # noqa: BLE001 - no facts is "no night facts yet"
        logging.warning("Trade Mentor: the night facts could not be read", exc_info=True)
        return []


def as_listing(memory: Memory) -> str:
    """What ``/memory`` prints: every loaded item with its id."""
    if not memory.items:
        return "Nothing in memory yet. `/remember <text>` keeps a note; the night adds a digest."
    lines = ["**Memory loaded at start**", ""]
    lines += [f"- [{item.id}] ({item.day}) {item.text}" for item in memory.items]
    if memory.dropped:
        lines.append(f"- *({memory.dropped} older item(s) left out to fit the budget)*")
    return "\n".join(lines)


def substring_search(store: Any, memory: Memory, query: str, k: int) -> list[dict[str, Any]]:
    """Recall with the brain down: plain substring over digests, notes and turns."""
    needle = str(query or "").strip().lower()
    if not needle:
        return []
    hits = [
        {"kind": item.kind, "ref_id": item.ref_id, "text": item.text, "score": 1.0}
        for item in reversed(memory.items)
        if item.kind == "digest" and needle in item.text.lower()
    ]
    for row in store.search_text(needle, limit=k):
        hits.append({"kind": row["kind"], "ref_id": row["ref_id"], "text": row["text"], "score": 1.0})
    return hits[: max(0, int(k))]


def parse_note_id(text: str) -> int | None:
    """``12`` or ``mem:note:12`` -> 12; anything else None."""
    raw = str(text or "").strip()
    if raw.startswith("mem:note:"):
        raw = raw[len("mem:note:"):]
    return int(raw) if raw.isdigit() else None


def _age_start(row: Mapping[str, Any]) -> datetime | None:
    for key in ("checked_utc", "ts_utc"):
        try:
            moment = datetime.fromisoformat(str(row.get(key) or ""))
        except ValueError:
            continue
        return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    return None


def still_true_candidate(notes: Iterable[Mapping[str, Any]], now: datetime) -> dict[str, Any] | None:
    """The oldest live ``rule:`` note older than 7 days and not asked about in the last 7 days."""
    moment = now if now.tzinfo else now.astimezone()
    window = timedelta(days=STILL_TRUE_DAYS)
    for row in sorted(notes, key=lambda item: int(item["id"])):
        if row.get("retired_utc") or not _clean(row.get("text")).lower().startswith(RULE_PREFIX):
            continue
        started = _age_start(row)
        if started is None or moment - started <= window:
            continue
        asked = _age_start({"ts_utc": row.get("asked_utc")}) if row.get("asked_utc") else None
        if asked is not None and moment - asked <= window:
            continue
        return dict(row)
    return None


def still_true_text(row: Mapping[str, Any]) -> str:
    note_id = int(row["id"])
    return f"You said: {_clean(row.get('text'))}. Still true? (/keep {note_id} | /forget {note_id})"
