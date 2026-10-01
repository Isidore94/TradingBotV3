"""Tape talk (``/tape``): the regime pack, narrated once per pack hash and cached. Qt-free.

The model returns ``{read: [{text, evidence_refs}], watch: [{text, evidence_refs}]}``
through ``ai_summary.request_ai_summary`` (at most 600 output tokens). Each list goes
through ``citations.check_citations``: a foreign id rejects the whole reply, an uncited
line is dropped. The card is cached in the chat store's ``pack_cache`` under the pack
hash, so an unchanged tape is never narrated twice. Brain down: the pack text, no guess.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from mentor_packs.citations import CitationRejected, check_citations
from mentor_packs.registry import Pack, pack_from_json

PT = ZoneInfo("America/Los_Angeles")
CACHE_NAME = "tape_card"
SCHEMA_NAME = "trade_mentor_tape"
PROMPT_VERSION = "trade_mentor_tape_v1"
MAX_OUTPUT_TOKENS = 600
EFFORT = "medium"
TIMEOUT_SECONDS = 180
#: The session window the tape is rebuilt in, and how often.
SESSION_START = time(6, 0)
SESSION_END = time(13, 0)
REFRESH_EVERY = timedelta(minutes=30)
#: Rows that change every build without changing the tape (the clock) stay out of the hash.
UNHASHED_IDS = frozenset({"tape:asof"})

_LINES = {
    "type": "array",
    "maxItems": 4,
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["text", "evidence_refs"],
        "properties": {
            "text": {"type": "string", "maxLength": 240},
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
        },
    },
}
SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["read", "watch"],
    "properties": {"read": _LINES, "watch": _LINES},
}
TASK = (
    "Read the tape for the trader in at most 4 short 'read' lines (what the market is doing) and at "
    "most 4 'watch' lines (what to keep an eye on today). Every line cites the source_id of each row "
    "it uses in evidence_refs; a line with no source_id is thrown away. A 'night read' row is last "
    "night's model text: say so when you use it. A 'fund' row is the morning brief the trader pasted "
    "(outside commentary): say so when you use it. Never size or place an order; never suggest changing "
    "a detector, a score, an alert or the plan."
)


def with_fundamentals(pack: Pack, fund: Pack | None) -> Pack:
    """The regime pack plus today's pasted brief (compact rows); a missing brief adds its none row."""
    from mentor_packs.registry import make_pack

    extra = [dict(row) for row in (fund.rows if fund is not None else ()) if row.get("id")]
    if not extra:
        return pack
    return make_pack(pack.name, [*pack.rows, *extra], empty_text=pack.empty_text)


def pack_hash(pack: Pack) -> str:
    """A stable hash of the pack's rows, without the as-of clock but with the plan file's hash."""
    rows = [row for row in pack.rows if str(row.get("id")) not in UNHASHED_IDS]
    plan = next((str(row.get("plan_sha") or "") for row in pack.rows if "plan_sha" in row), "")
    body = {"rows": rows, "plan": plan} if plan else rows
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


@dataclass
class TapeCard:
    pack_hash: str
    read: list[dict[str, Any]] = field(default_factory=list)
    watch: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    model: str = ""
    built_utc: str = ""
    pack_json: str = ""

    @property
    def narrated(self) -> bool:
        return bool(self.read or self.watch) and not self.error

    def pack(self) -> Pack | None:
        return pack_from_json(self.pack_json) if self.pack_json else None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)

    @classmethod
    def from_json(cls, text: str) -> "TapeCard":
        payload = json.loads(text)
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__ if key in payload})


def evidence_for(pack: Pack) -> dict[str, Any]:
    return {
        "task": TASK,
        "pack": pack.name,
        "rows": [{"source_id": str(row["id"]), "text": str(row.get("text") or "")} for row in pack.rows],
    }


def check_reply(reply: Any, pack: Pack) -> tuple[list[dict], list[dict], list[dict]]:
    """(kept read, kept watch, drops). Raises CitationRejected for a foreign id or a bad shape."""
    if not isinstance(reply, Mapping):
        raise CitationRejected("the reply was not an object")
    allowed = set(pack.ids)
    kept: dict[str, list[dict]] = {}
    drops: list[dict] = []
    for key in ("read", "watch"):
        result = check_citations({"bullets": reply.get(key)}, allowed)
        kept[key] = list(result.kept)
        drops.extend({"kind": key, "reason": "uncited", **row} for row in result.dropped)
    return kept["read"], kept["watch"], drops


def narrate(
    pack: Pack,
    *,
    pack_hash: str,
    model: str,
    endpoint: str,
    post: Callable[..., Any] | None = None,
    request: Callable[..., Mapping[str, Any]] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> TapeCard:
    """One structured call over the pack. A failed call or a rejected reply is a card with ``error``."""
    import requests

    from mentor_app.assess import effort_post

    card = TapeCard(pack_hash=pack_hash, model=model, pack_json=pack.as_json(),
                    built_utc=now().astimezone(timezone.utc).isoformat(timespec="seconds"))
    if request is None:
        import ai_summary

        request = ai_summary.request_ai_summary
    try:
        result = request(
            provider="local", model=model, api_key="", evidence=evidence_for(pack),
            timeout_seconds=TIMEOUT_SECONDS,
            post=effort_post(post or requests.post, model=model, effort=EFFORT, max_tokens=MAX_OUTPUT_TOKENS),
            schema=SCHEMA, schema_name=SCHEMA_NAME, prompt_version=PROMPT_VERSION,
            endpoint=f"{endpoint.rstrip('/')}/v1",
        )
        card.read, card.watch, card.dropped = check_reply(result.get("summary"), pack)
        if not (card.read or card.watch):
            card.error = "no cited line survived the check"
    except CitationRejected as exc:
        card.error = f"reply rejected: {exc}"
        card.dropped.append({"kind": "reply", "reason": str(exc)})
    except Exception as exc:  # noqa: BLE001 - a failed narration leaves the pack, never a guess
        card.error = f"{type(exc).__name__}: {exc}"
    if card.dropped:
        logging.info("Trade Mentor tape: dropped %s", json.dumps(card.dropped, default=str)[:1000])
    return card


def cached_card(store: Any, digest: str) -> TapeCard | None:
    row = store.get_pack(CACHE_NAME, {"hash": str(digest)})
    if not row:
        return None
    try:
        return TapeCard.from_json(str(row["pack_json"]))
    except (ValueError, TypeError, KeyError):
        return None


def run_tape_job(
    *,
    store: Any,
    build_pack: Callable[[], Pack],
    narrate: Callable[[Pack, str], TapeCard] | None,
    last_hash: str = "",
    should_yield: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """Build the pack; reuse the cached card for its hash, else narrate (when ``narrate`` is given) and cache.

    Returns ``{pack, hash, card, narrated, changed, yielded}``.
    """
    pack = build_pack()
    digest = pack_hash(pack)
    card = cached_card(store, digest)
    out: dict[str, Any] = {"pack": pack, "hash": digest, "card": card, "narrated": False,
                           "changed": digest != last_hash, "yielded": False}
    if (card is not None and card.narrated) or narrate is None:
        return out
    if should_yield():
        out["yielded"] = True
        return out
    fresh = narrate(pack, digest)
    if fresh.narrated:
        store.put_pack(CACHE_NAME, {"hash": digest}, fresh.to_json(), fresh.built_utc)
    out["card"], out["narrated"] = fresh, True
    return out


@dataclass
class TapeSchedule:
    """Every 30 min inside 06:00-13:00 PT. Marked only when a rebuild is queued."""

    last_utc: datetime | None = None

    def due(self, now: datetime) -> bool:
        moment = now if now.tzinfo else now.astimezone()
        local = moment.astimezone(PT)
        if local.weekday() >= 5 or not (SESSION_START <= local.time() < SESSION_END):
            return False
        return self.last_utc is None or moment - self.last_utc >= REFRESH_EVERY

    def mark(self, now: datetime) -> None:
        moment = now if now.tzinfo else now.astimezone()
        self.last_utc = moment.astimezone(timezone.utc)


def card_markdown(pack: Pack, card: TapeCard | None, *, brain_reason: str = "") -> str:
    """The /tape block: the narrated read (each line with its ids) above the pack, or the pack alone."""
    facts = pack.as_text().replace("\n", "\n\n")
    if card is None or not card.narrated:
        why = (card.error if card is not None and card.error else "") or brain_reason
        tail = f"\n\n*(No read: {why}. The pack is the answer.)*" if why else ""
        return f"**Tape**\n\n{facts}{tail}"
    lines = ["**Tape: the read**", ""]
    for key, title in (("read", "Read"), ("watch", "Watch")):
        rows = getattr(card, key)
        if rows:
            lines.append(f"*{title}*")
            for row in rows:
                refs = " ".join(f"[{ref}]" for ref in row.get("evidence_refs") or ())
                lines.append(f"- {row.get('text', '')} {refs}".rstrip())
            lines.append("")
    uncited = sum(1 for drop in card.dropped if drop.get("reason") == "uncited")
    if uncited:
        lines.append(f"*({uncited} uncited line{'s' if uncited != 1 else ''} dropped)*")
        lines.append("")
    lines.append(facts)
    return "\n".join(lines)
