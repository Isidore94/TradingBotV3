"""The mirror (``/mirror`` and the Monday card): the mirror pack, narrated once per pack hash. Qt-free.

The model returns ``{observations: [{text, evidence_refs}], questions: [{text, evidence_refs}]}``
(at most 600 output tokens) through ``ai_summary.request_ai_summary``; each list goes through
``citations.check_citations`` (a foreign id rejects the reply, an uncited line is dropped).
The card is cached in ``pack_cache`` under the pack hash. Brain down: the pack alone.
Observations, never rules: the footer says rules live in ``trading_plan.md``.

Weekly: from 06:50 PT on the first session day of a week, one Inbox item "Your week in the
mirror", held through quiet hours or a mute, dropped at the day's end or a used cap; the
posted week is kept in ``app_state`` so a restart never reposts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timezone
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from mentor_packs.citations import CitationRejected, check_citations
from mentor_packs.registry import Pack, pack_from_json

PT = ZoneInfo("America/Los_Angeles")
CACHE_NAME = "mirror_card"
SCHEMA_NAME = "trade_mentor_mirror"
PROMPT_VERSION = "trade_mentor_mirror_v1"
MAX_OUTPUT_TOKENS = 600
EFFORT = "medium"
TIMEOUT_SECONDS = 180
WEEKLY_AT = time(6, 50)
#: app_state key: the ISO week (``2026-W40``) whose card reached the Inbox.
POSTED_KEY = "mirror_week_posted"
INBOX_LINE = "Your week in the mirror"
FOOTER = "*Observations, not rules; rules live in trading_plan.md.*"

_LINES = {
    "type": "array",
    "maxItems": 5,
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
    "required": ["observations", "questions"],
    "properties": {"observations": _LINES, "questions": {**_LINES, "maxItems": 3}},
}
TASK = (
    "These rows are the trader's own record over the last weeks, cut several ways. Write at most 5 short "
    "'observations' (what the numbers show) and at most 3 'questions' to ask him about them. Every line cites "
    "the source_id of each row it uses in evidence_refs; a line with no source_id is thrown away. Say n with "
    "every rate; a row that says 'too few' is not evidence, never draw a conclusion from it. Only a few weeks "
    "of data exist: say so. Never propose a rule, never tell him to change the plan, a detector, a score or an "
    "alert, and never size or place an order: these are observations, not rules."
)


def pack_hash(pack: Pack) -> str:
    from mentor_packs import mirror_pack

    return mirror_pack.pack_hash(pack)


@dataclass
class MirrorCard:
    pack_hash: str
    observations: list[dict[str, Any]] = field(default_factory=list)
    questions: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    model: str = ""
    built_utc: str = ""
    pack_json: str = ""

    @property
    def narrated(self) -> bool:
        return bool(self.observations or self.questions) and not self.error

    def pack(self) -> Pack | None:
        return pack_from_json(self.pack_json) if self.pack_json else None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)

    @classmethod
    def from_json(cls, text: str) -> "MirrorCard":
        payload = json.loads(text)
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__ if key in payload})


def evidence_for(pack: Pack) -> dict[str, Any]:
    return {"task": TASK, "pack": pack.name,
            "rows": [{"source_id": str(row["id"]), "text": str(row.get("text") or "")} for row in pack.rows]}


def check_reply(reply: Any, pack: Pack) -> tuple[list[dict], list[dict], list[dict]]:
    """(kept observations, kept questions, drops). Raises CitationRejected for a foreign id or a bad shape."""
    if not isinstance(reply, Mapping):
        raise CitationRejected("the reply was not an object")
    allowed = set(pack.ids)
    kept: dict[str, list[dict]] = {}
    drops: list[dict] = []
    for key in ("observations", "questions"):
        result = check_citations({"bullets": reply.get(key)}, allowed)
        kept[key] = list(result.kept)
        drops.extend({"kind": key, "reason": "uncited", **row} for row in result.dropped)
    return kept["observations"], kept["questions"], drops


def narrate(
    pack: Pack,
    *,
    pack_hash: str,
    model: str,
    endpoint: str,
    post: Callable[..., Any] | None = None,
    request: Callable[..., Mapping[str, Any]] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> MirrorCard:
    """One structured call over the pack. A failed call or a rejected reply is a card with ``error``."""
    import requests

    from mentor_app.assess import effort_post

    card = MirrorCard(pack_hash=pack_hash, model=model, pack_json=pack.as_json(),
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
        card.observations, card.questions, card.dropped = check_reply(result.get("summary"), pack)
        if not (card.observations or card.questions):
            card.error = "no cited line survived the check"
    except CitationRejected as exc:
        card.error = f"reply rejected: {exc}"
        card.dropped.append({"kind": "reply", "reason": str(exc)})
    except Exception as exc:  # noqa: BLE001 - a failed narration leaves the pack, never a guess
        card.error = f"{type(exc).__name__}: {exc}"
    if card.dropped:
        logging.info("Trade Mentor mirror: dropped %s", json.dumps(card.dropped, default=str)[:1000])
    return card


def cached_card(store: Any, digest: str) -> MirrorCard | None:
    row = store.get_pack(CACHE_NAME, {"hash": str(digest)})
    if not row:
        return None
    try:
        return MirrorCard.from_json(str(row["pack_json"]))
    except (ValueError, TypeError, KeyError):
        return None


def run_mirror_job(
    *,
    store: Any,
    build_pack: Callable[[], Pack],
    narrate: Callable[[Pack, str], MirrorCard] | None,
    should_yield: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """Build the pack; reuse the cached card for its hash, else narrate (when given) and cache.

    Returns ``{pack, hash, card, narrated, yielded}``.
    """
    pack = build_pack()
    digest = pack_hash(pack)
    card = cached_card(store, digest)
    out: dict[str, Any] = {"pack": pack, "hash": digest, "card": card, "narrated": False, "yielded": False}
    if (card is not None and card.narrated) or narrate is None or not pack.rows:
        return out
    if should_yield():
        out["yielded"] = True
        return out
    fresh = narrate(pack, digest)
    if fresh.narrated:
        store.put_pack(CACHE_NAME, {"hash": digest}, fresh.to_json(), fresh.built_utc)
    out["card"], out["narrated"] = fresh, True
    return out


# ---------------------------------------------------------------- the card
def _weeks(pack: Pack) -> str:
    row = next((r for r in pack.rows if r.get("id") == "mirror:weeks"), None)
    return str(row.get("weeks")) if row else "?"


def card_markdown(pack: Pack, card: MirrorCard | None, *, brain_reason: str = "", title: str = "Mirror") -> str:
    """The card: the narration (each line with its ids), then every reportable cut, the too-few count,
    the caveats and the footer. Brain down or a failed narration: the pack alone, with why."""
    if not pack.rows:
        return f"**{title}**: {pack.empty_text or 'nothing to show'}\n\n{FOOTER}"
    lines = [f"**{title}** (weeks={_weeks(pack)})", ""]
    if card is not None and card.narrated:
        for key, head in (("observations", "Observations"), ("questions", "Questions for you")):
            rows = getattr(card, key)
            if rows:
                lines.append(f"*{head}*")
                for row in rows:
                    refs = " ".join(f"[{ref}]" for ref in row.get("evidence_refs") or ())
                    lines.append(f"- {row.get('text', '')} {refs}".rstrip())
                lines.append("")
        uncited = sum(1 for drop in card.dropped if drop.get("reason") == "uncited")
        if uncited:
            lines += [f"*({uncited} uncited line{'s' if uncited != 1 else ''} dropped)*", ""]
    else:
        why = (card.error if card is not None and card.error else "") or brain_reason
        if why:
            lines += [f"*(No narration: {why}. The numbers are the answer.)*", ""]
    by_kind = [row for row in pack.rows if row.get("kind") not in ("asof", "caveats")]
    few = [row for row in by_kind if row.get("too_few")]
    lines.append("*The numbers*")
    for row in by_kind:
        if not row.get("too_few"):
            lines.append(f"- {row['text']} [{row['id']}]")
    if few:
        lines.append(f"- {len(few)} cut{'s' if len(few) != 1 else ''} too few to read (n under 30): "
                     + ", ".join(f"[{row['id']}] n={row.get('n', 0)}" for row in few))
    caveats = next((row for row in pack.rows if row.get("kind") == "caveats"), None)
    if caveats:
        lines += ["", f"*Caveats* [{caveats['id']}]"] + [f"- {line}" for line in caveats.get("lines") or ()]
    lines += ["", FOOTER]
    return "\n".join(lines)


# ---------------------------------------------------------------- the weekly card
def week_key(now: datetime) -> str:
    local = (now if now.tzinfo else now.astimezone()).astimezone(PT).date()
    year, week, _ = local.isocalendar()
    return f"{year}-W{week:02d}"


@dataclass
class WeeklySchedule:
    """Due from 06:50 PT on the first session day of a week the app sees; marked when queued."""

    last_week: str = ""

    def due(self, now: datetime) -> bool:
        from mentor_app.challenge import is_session_day

        local = (now if now.tzinfo else now.astimezone()).astimezone(PT)
        if local.time() < WEEKLY_AT or self.last_week == week_key(now):
            return False
        return is_session_day(now)

    def mark(self, now: datetime) -> None:
        self.last_week = week_key(now)
