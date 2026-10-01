"""Morning memory: the night's reads and the trader's own notes, as one byte-stable block. Qt-free.

P15a: the memory stands on the night. ``load`` reads, in priority order: (0) the night's coach
brief for today (``mentor_coach_brief``), (P15b) today's pasted morning brief, its bottom line and
playbook (``fundamentals_pack`` compact, FUND_LINES), (1) the last DIGEST_LIMIT ``mentor_day_digest``
publications, (2) the top IDEA_LIMIT improvement ideas, (3) the latest day review's "were you
right" verdicts, (4) the week review's WEEK_LINES headline lines, (5) the newest NOTE_LIMIT live
``profile_notes`` (call it off the Qt thread). ``render`` orders by tier, oldest first inside a
tier, and drops the lowest tier's oldest item until the block fits MEMORY_BUDGET_TOKENS, so the
same inputs always give the same bytes and Ollama's KV prefix stays warm. Every item carries a
citable id: ``mem:<kind>:<id>`` for digests and notes, the night's own ``night:<kind>:...`` for
the rest. Nothing here infers a note: notes come only from ``/remember``. (Plan lines are a
separate store: ``plan_infer`` writes AI-marked lines to the trading plan.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from mentor_app.chat_model import estimate_tokens

PT = ZoneInfo("America/Los_Angeles")
ET = ZoneInfo("America/New_York")
MEMORY_BUDGET_TOKENS = 3000
DIGEST_LIMIT = 5
NOTE_LIMIT = 50
IDEA_LIMIT = 3
WEEK_LINES = 3
#: A digest question's ref id is the day's number + this offset (items use 0..4).
QUESTION_OFFSET = 50
RULE_PREFIX = "rule:"
STILL_TRUE_DAYS = 7
HEAD = ("# Memory\nThe night's coach brief, today's pasted morning brief, digests, ideas, day and week review, then "
        "the trader's own notes; oldest first in each. Cite by id.")
#: Tiers, most important first; the budget drops the highest number (oldest first) before any other.
(PRIORITY_COACH, PRIORITY_FUND, PRIORITY_DIGEST, PRIORITY_IDEA, PRIORITY_DAY_REVIEW, PRIORITY_WEEK,
 PRIORITY_NOTE) = range(7)
#: P15b: today's pasted brief in memory, at most this many lines (bottom line + playbook, ids kept).
FUND_LINES = 8


@dataclass(frozen=True)
class MemoryItem:
    id: str
    kind: str
    ref_id: int
    day: str
    text: str
    #: The tier (``PRIORITY_*``); unset = a note's tier, the first to go.
    priority: int = PRIORITY_NOTE
    #: The line's place inside its artifact (the night's rows keep the order the night wrote them in).
    order: int = 0

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
            out.append(MemoryItem(f"mem:digest:{ref}", "digest", ref, day, _clean(row.get("text")), PRIORITY_DIGEST))
        for index, row in enumerate(payload.get("open_questions") or ()):
            ref = digest_ref(day, QUESTION_OFFSET + index)
            out.append(MemoryItem(f"mem:digest:{ref}", "digest", ref, day, "open question: " + _clean(row.get("text")),
                                  PRIORITY_DIGEST))
    return [item for item in out if item.text]


def note_items(notes: Iterable[Mapping[str, Any]]) -> list[MemoryItem]:
    return [
        MemoryItem(f"mem:note:{int(row['id'])}", "note", int(row["id"]), _pt_day(row.get("ts_utc")),
                   _clean(row.get("text")), PRIORITY_NOTE)
        for row in notes
        if not row.get("retired_utc") and _clean(row.get("text"))
    ]


def night_item(row: Mapping[str, Any], priority: int, order: int = 0) -> MemoryItem:
    """One night row (``night:<kind>:...``) as a memory item; its ref is stable for its id and text."""
    from mentor_packs.night_pack import stable_ref

    text = _clean(row.get("text"))
    return MemoryItem(str(row["id"]), "night", stable_ref(str(row["id"]), text), str(row.get("date") or ""), text,
                      priority, order)


def coach_items(payload: Mapping[str, Any] | None) -> list[MemoryItem]:
    """The coach brief's lines: one line, what to watch, what he is missing, recurring issues."""
    if not isinstance(payload, Mapping):
        return []
    day = str(payload.get("session_date") or "")[:10]
    rows = []
    line = payload.get("one_line")
    # Only a cited one line leads (an uncited or old plain-string one is model text with nothing under it);
    # without one, the first watch item leads instead.
    lead = line if isinstance(line, Mapping) and _clean(line.get("text")) and line.get("evidence_refs") else None
    watch = [item for item in payload.get("watch") or () if isinstance(item, Mapping) and _clean(item.get("text"))]
    led_by_watch = lead is None and bool(watch)
    if lead is None and watch:
        lead = {"text": "watch: " + _clean(watch[0]["text"]), "evidence_refs": watch[0].get("evidence_refs") or ()}
    if lead is not None:
        refs = ", ".join(str(ref) for ref in lead.get("evidence_refs") or ())
        rows.append({"id": f"night:coach:{day}:0", "date": day,
                     "text": "coach brief: " + _clean(lead["text"]) + (f" (cites {refs})" if refs else "")})
    for prefix, key, label in (("w", "watch", "watch"), ("m", "missing", "you may be missing"),
                               ("i", "issues", "recurring issue"), ("h", "habits", "habit")):
        for n, item in enumerate(payload.get(key) or (), start=1):
            if not isinstance(item, Mapping) or not _clean(item.get("text")):
                continue
            if key == "watch" and led_by_watch and item is watch[0]:
                continue  # already the lead line
            since = f" (since {item['first_seen']})" if key in ("issues", "habits") and item.get("first_seen") else ""
            refs = ", ".join(str(ref) for ref in item.get("evidence_refs") or ())
            rows.append({"id": f"night:coach:{day}:{prefix}{n}", "date": day,
                         "text": f"{label}{since}: {_clean(item['text'])}" + (f" (cites {refs})" if refs else "")})
    if _clean(payload.get("routine")):
        # P18: the night's one routine line, written by code only on the night the routine changed.
        rows.append({"id": f"night:coach:{day}:r1", "date": day, "text": f"routine: {_clean(payload['routine'])}"})
    return [night_item(row, PRIORITY_COACH, n) for n, row in enumerate(rows)]


def _order(item: MemoryItem) -> tuple[int, str, int, str, int]:
    return item.priority, item.day, item.order, item.kind, item.ref_id


def render(items: Sequence[MemoryItem], *, budget_tokens: int = MEMORY_BUDGET_TOKENS) -> Memory:
    """By tier, oldest first in each; the lowest tier's oldest go first until the block fits. Deterministic."""
    ordered = sorted({item.id: item for item in items}.values(), key=_order)
    if not ordered:
        return Memory()
    kept = list(ordered)
    while kept and estimate_tokens("\n".join([HEAD, *(item.line() for item in kept)])) > budget_tokens:
        # The lowest priority tier loses its oldest item first.
        victim = min(kept, key=lambda item: (-item.priority, item.day, -item.order, item.kind, item.ref_id))
        kept.remove(victim)
    if not kept:
        return Memory(dropped=len(ordered))
    return Memory("\n".join([HEAD, *(item.line() for item in kept)]), kept, len(ordered) - len(kept))


def _digests_root() -> Path | None:
    from ai_jobs import store as ai_store

    try:
        return ai_store.digests_dir(create=False)
    except ValueError:
        return None


def _today(now: datetime | None) -> Any:
    moment = now or datetime.now(timezone.utc)
    return (moment if moment.tzinfo else moment.astimezone()).astimezone(ET).date()


def digests_root(ai_root: Path | str | None) -> Path | None:
    """``ai_root`` when given, else the ai_store's digests folder (None when no store is configured)."""
    return Path(ai_root) if ai_root is not None else _digests_root()


def night_paths_for(root: Path | None, night_paths: Any = None) -> Any:
    """The night pack's paths, with the digests folder this memory reads (tests pass both)."""
    from mentor_packs import night_pack

    paths = night_paths if night_paths is not None else night_pack.live_paths()
    return replace(paths, digests=root) if root is not None else paths


def latest_coach_brief(root: Path | None, today: Any) -> dict[str, Any] | None:
    """The newest coach brief the night published for a session on or before ``today``; None when none."""
    from ai_jobs.mentor_review import COACH_STEM, read_published

    if root is None:
        return None
    for payload in read_published(root, COACH_STEM, limit=5):
        if str(payload.get("session_date") or "")[:10] <= today.isoformat():
            return payload
    return None


def read_coach_brief(ai_root: Path | str | None, now: datetime | None = None) -> dict[str, Any] | None:
    """For ``/brief`` and ``/issues``: today's coach brief, or None (off the Qt thread)."""
    try:
        return latest_coach_brief(digests_root(ai_root), _today(now))
    except Exception:  # noqa: BLE001 - unreadable reads as "no brief yet"
        logging.warning("Trade Mentor: the coach brief could not be read", exc_info=True)
        return None


def fund_items(paths: Any = None, now: datetime | None = None) -> list[MemoryItem]:
    """P15b tier 1: today's pasted brief, its bottom line and playbook (at most FUND_LINES, ids kept).

    No brief for today = nothing here (a none row is never memory; ``fundamentals_pack`` says it when asked)."""
    from mentor_packs import fundamentals_pack
    from mentor_packs.night_pack import stable_ref

    pack = fundamentals_pack.build("today", "compact", now=now, paths=paths)
    if any(row.get("kind") == "none" for row in pack.rows):
        return []
    rows = [row for row in pack.rows if row.get("kind") == "fund"][:FUND_LINES]
    return [MemoryItem(str(row["id"]), "fund", stable_ref(str(row["id"]), _clean(row.get("text"))),
                       str(row.get("date") or ""), _clean(row.get("text")), PRIORITY_FUND, n)
            for n, row in enumerate(rows)]


def night_items(paths: Any, today: Any) -> list[MemoryItem]:
    """Tiers 2-4: the top ideas, the latest day review's verdicts, the week review's headline lines."""
    from mentor_packs import night_pack

    out: list[MemoryItem] = []
    readers = (
        (PRIORITY_IDEA, lambda: night_pack.idea_rows(paths, today, limit=IDEA_LIMIT)[0]),
        (PRIORITY_DAY_REVIEW, lambda: night_pack.day_review_rows(paths, today, 1)[0]),
        (PRIORITY_WEEK, lambda: night_pack.week_rows(paths, today)[0][:WEEK_LINES]),
    )
    for priority, read in readers:
        try:
            rows = read()
        except Exception:  # noqa: BLE001 - one unreadable night read never empties the memory
            logging.warning("Trade Mentor memory: a night read failed", exc_info=True)
            continue
        out += [night_item(row, priority, n) for n, row in enumerate(rows) if row.get("kind") == "night"]
    return out


def load(store: Any, *, ai_root: Path | str | None = None, budget_tokens: int = MEMORY_BUDGET_TOKENS,
         night_paths: Any = None, now: datetime | None = None, fund_paths: Any = None) -> Memory:
    """The memory block: the night's coach brief, digests, ideas, day and week review, then live notes.

    Off the Qt thread. Anything unreadable is left out, never a crash.
    """
    from ai_jobs.mentor_review import DIGEST_STEM, read_published

    today = _today(now)
    publications: list[dict[str, Any]] = []
    items: list[MemoryItem] = []
    root: Path | None = None
    try:
        root = Path(ai_root) if ai_root is not None else _digests_root()
        if root is not None:
            publications = read_published(root, DIGEST_STEM, limit=DIGEST_LIMIT)
            items += coach_items(latest_coach_brief(root, today))
    except Exception:  # noqa: BLE001 - no digests is an empty memory, never a crash
        logging.warning("Trade Mentor memory: the night digests could not be read", exc_info=True)
    try:
        items += night_items(night_paths_for(root, night_paths), today)
    except Exception:  # noqa: BLE001
        logging.warning("Trade Mentor memory: the night reads could not be loaded", exc_info=True)
    try:
        items += fund_items(fund_paths, now)
    except Exception:  # noqa: BLE001 - an unreadable brief never empties the memory
        logging.warning("Trade Mentor memory: the pasted brief could not be read", exc_info=True)
    notes = store.profile_notes(limit=NOTE_LIMIT)
    return render([*items, *digest_items(publications), *note_items(notes)], budget_tokens=budget_tokens)


#: The embed kinds :func:`embed_candidates` yields (the window asks the store which refs each already has).
EMBED_KINDS = ("night", "brief", "fund", "recap")


def embed_candidates(memory: "Memory", paths: Any, now: datetime | None = None, *,
                     fund_paths: Any = None, recap_paths: Any = None) -> list[tuple[str, int, str]]:
    """``(kind, ref_id, text)`` the idle embed queue may add: every night row (``night``), every ticker
    brief (``brief``) and (P15b) each paragraph of today's pasted brief (``fund``, once per paste) and each
    day-recap row (``recap``, once per text version). The text
    starts with the row's own id so a recall hit cites it; a changed artifact is a new ref (``stable_ref``
    of id and text), so each artifact version is embedded once."""
    from mentor_packs import fundamentals_pack, night_pack

    today = _today(now)
    extra_rows: list[tuple[str, int, str]] = []
    try:
        fund = fundamentals_pack.build("today", "text", now=now, paths=fund_paths)
        extra_rows = [("fund", ref, text) for ref, text in fundamentals_pack.embed_rows(fund)]
    except Exception:  # noqa: BLE001 - an unreadable brief is embedded next time
        logging.warning("Trade Mentor: the pasted brief could not be read for recall", exc_info=True)
    try:
        from mentor_packs import recaps_pack

        recaps = recaps_pack.build("all", now=now, paths=recap_paths)
        extra_rows += [("recap", ref, text) for ref, text in recaps_pack.embed_rows(recaps)]
    except Exception:  # noqa: BLE001 - unreadable recaps are embedded next time
        logging.warning("Trade Mentor: the day recaps could not be read for recall", exc_info=True)
    out: dict[tuple[str, int], str] = {}
    rows = [row for row in night_pack.build("all", now=now, paths=paths).rows if row.get("kind") == "night"]
    for row in rows:
        text = f"[{row['id']}] {_clean(row.get('text'))}"
        out[("night", night_pack.stable_ref(str(row["id"]), _clean(row.get("text"))))] = text
    for item in memory.items:
        if item.kind == "night":
            out[("night", item.ref_id)] = f"[{item.id}] {item.text}"
    for row in night_pack.brief_rows(paths.briefs, today):
        out[("brief", night_pack.stable_ref(row["id"], row["text"]))] = f"[{row['id']}] {row['text']}"
    for kind, ref, text in extra_rows:
        out[(kind, ref)] = text
    return [(kind, ref, text) for (kind, ref), text in sorted(out.items())]


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
        lines.append(f"- *({memory.dropped} item(s) left out to fit the budget, your notes and the oldest first)*")
    return "\n".join(lines)


def coach_brief_text(payload: Mapping[str, Any] | None) -> str:
    """What ``/brief`` prints: the night's coach brief with its ids."""
    if not isinstance(payload, Mapping):
        return "No coach brief from the night yet. The night writes one after it reviews a day with you."
    items = coach_items(payload)
    worded = "" if payload.get("worded") else " (facts only: no model worded it)"
    lines = [f"**Coach brief from the night of {payload.get('session_date', '?')}**{worded}", ""]
    lines += [f"- [{item.id}] {item.text}" for item in items] or ["- nothing to flag"]
    return "\n".join(lines)


def issues_text(payload: Mapping[str, Any] | None) -> str:
    """What ``/issues`` prints: the recurring issues with the date each was first seen."""
    issues = list((payload or {}).get("issues") or ())
    if not any(isinstance(item, Mapping) for item in issues):
        return "No recurring issues in the last night's brief."
    day = str((payload or {}).get("session_date") or "?")
    lines = [f"**Recurring issues** (night of {day}; observations, never rules)", ""]
    for n, item in enumerate(issues, start=1):
        if not isinstance(item, Mapping):
            continue
        refs = ", ".join(str(ref) for ref in item.get("evidence_refs") or ())
        lines.append(f"- [night:coach:{day}:i{n}] first seen {item.get('first_seen', '?')}"
                     f"{', ' + str(item['count']) + ' times' if item.get('count') else ''}: {_clean(item.get('text'))}"
                     + (f" (cites {refs})" if refs else ""))
    return "\n".join(lines)


def substring_search(store: Any, memory: Memory, query: str, k: int) -> list[dict[str, Any]]:
    """Recall with the brain down: plain substring over digests, the night's lines, notes and turns."""
    needle = str(query or "").strip().lower()
    if not needle:
        return []
    hits = [
        {"kind": item.kind, "ref_id": item.ref_id, "text": item.text, "score": 1.0,
         **({"id": item.id} if item.kind in ("night", "fund") else {})}
        for item in reversed(memory.items)
        if item.kind in ("digest", "night", "fund") and needle in item.text.lower()
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
