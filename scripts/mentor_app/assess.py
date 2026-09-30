"""Pick assessment: one structured model call over a pick pack, then the citation check. Qt-free.

The model returns ``{verdict, bullets: [{text, evidence_refs}], rule_flags: [{plan_id,
breaks, text}]}`` through ``ai_summary.request_ai_summary`` (the same provider path,
timeout and retry rules as every other structured call). ``citations.check_citations``
then keeps only cited bullets (a foreign id rejects the whole reply); a rule flag
must name a plan line the pack carried. Every drop is kept on the result and logged.
The app never proposes a rule change here: that stays with ``plan_challenges``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from mentor_packs.citations import CitationRejected, check_citations
from mentor_packs.registry import Pack, pack_from_json

VERDICTS = ("worth a look", "wait", "pass")
SCHEMA_NAME = "trade_mentor_pick_assessment"
PROMPT_VERSION = "trade_mentor_pick_assessment_v1"
CACHE_NAME = "pick_assessment"
#: A background job's output cap (prefetch.MAX_JOB_OUTPUT_TOKENS); the live ask uses it too.
MAX_OUTPUT_TOKENS = 600
EFFORT_PREFETCH = "high"
EFFORT_LIVE = "medium"
TIMEOUT_SECONDS = 180

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "bullets", "rule_flags"],
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "bullets": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "evidence_refs"],
                "properties": {
                    "text": {"type": "string", "maxLength": 240},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "rule_flags": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["plan_id", "breaks", "text"],
                "properties": {
                    "plan_id": {"type": "string"},
                    "breaks": {"type": "boolean"},
                    "text": {"type": "string", "maxLength": 200},
                },
            },
        },
    },
}

TASK = (
    "Assess this stock pick for the trader in at most 6 short bullets. Every bullet cites the "
    "source_id of each row it uses in evidence_refs; a bullet with no source_id is thrown away. "
    "Say n and the floor when you use a win rate; a row that says 'too few' is not evidence. "
    "rule_flags: only for plan lines listed (their plan_id), breaks=true when the pick breaks it. "
    "Never suggest changing the plan, a detector, a score or an alert. Never size or place an order. "
    "verdict: 'worth a look', 'wait' or 'pass'."
)


@dataclass
class Assessment:
    symbol: str
    pack_hash: str
    verdict: str = ""
    bullets: list[dict[str, Any]] = field(default_factory=list)
    rule_flags: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    model: str = ""
    effort: str = ""
    built_utc: str = ""
    pack_json: str = ""

    @property
    def narrated(self) -> bool:
        return bool(self.verdict) and not self.error

    def pack(self) -> Pack | None:
        return pack_from_json(self.pack_json) if self.pack_json else None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)

    @classmethod
    def from_json(cls, text: str) -> "Assessment":
        payload = json.loads(text)
        known = {key: payload[key] for key in cls.__dataclass_fields__ if key in payload}
        return cls(**known)


def evidence_for(pack: Pack) -> dict[str, Any]:
    """The package the model sees: the task and one row per pack line, keyed by its id."""
    return {
        "task": TASK,
        "pack": pack.name,
        "rows": [{"source_id": str(row["id"]), "text": str(row.get("text") or "")} for row in pack.rows],
    }


def _plan_ids(pack: Pack) -> dict[str, str]:
    """Every accepted spelling of a plan line -> its ``plan:...`` id."""
    accepted: dict[str, str] = {}
    for row in pack.rows:
        if row.get("kind") == "plan_line" and row.get("plan_id"):
            accepted[str(row["plan_id"])] = str(row["plan_id"])
            accepted[str(row["id"])] = str(row["plan_id"])
    return accepted


def check_reply(reply: Mapping[str, Any], pack: Pack) -> tuple[list[dict], list[dict], list[dict]]:
    """(kept bullets, kept rule flags, drops). Raises CitationRejected for a foreign id."""
    allowed = set(pack.ids) | set(_plan_ids(pack))
    result = check_citations(reply, allowed)
    drops = [{"kind": "bullet", "reason": "uncited", **row} for row in result.dropped]
    accepted = _plan_ids(pack)
    flags: list[dict[str, Any]] = []
    for flag in reply.get("rule_flags") or ():
        if not isinstance(flag, Mapping):
            drops.append({"kind": "rule_flag", "reason": "not an object", "flag": str(flag)[:200]})
            continue
        plan_id = accepted.get(str(flag.get("plan_id") or "").strip())
        if plan_id is None:
            drops.append({"kind": "rule_flag", "reason": "unknown plan id", **dict(flag)})
            continue
        flags.append({"plan_id": plan_id, "breaks": bool(flag.get("breaks")), "text": str(flag.get("text") or "")})
    return list(result.kept), flags, drops


def effort_post(post: Callable[..., Any], *, model: str, effort: str, max_tokens: int = MAX_OUTPUT_TOKENS):
    """Wrap ``post``: cap max_tokens and, for gpt-oss tags, send the reasoning effort."""
    from mentor_app.brain import is_thinking_model

    def wrapped(url: str, **kwargs: Any) -> Any:
        payload = dict(kwargs.get("json") or {})
        payload["max_tokens"] = min(int(payload.get("max_tokens") or max_tokens), int(max_tokens))
        if is_thinking_model(model):
            payload["reasoning_effort"] = effort
        kwargs["json"] = payload
        return post(url, **kwargs)

    return wrapped


def assess(
    pack: Pack,
    *,
    symbol: str,
    pack_hash: str,
    model: str,
    endpoint: str,
    live: bool = False,
    post: Callable[..., Any] | None = None,
    request: Callable[..., Mapping[str, Any]] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> Assessment:
    """Narrate one pick pack. A failed call or a rejected reply is an Assessment with ``error``."""
    import requests

    effort = EFFORT_LIVE if live else EFFORT_PREFETCH
    out = Assessment(
        symbol=symbol, pack_hash=pack_hash, model=model, effort=effort,
        built_utc=now().astimezone(timezone.utc).isoformat(timespec="seconds"), pack_json=pack.as_json(),
    )
    if request is None:
        import ai_summary

        request = ai_summary.request_ai_summary
    try:
        result = request(
            provider="local",
            model=model,
            api_key="",
            evidence=evidence_for(pack),
            timeout_seconds=TIMEOUT_SECONDS,
            post=effort_post(post or requests.post, model=model, effort=effort),
            schema=SCHEMA,
            schema_name=SCHEMA_NAME,
            prompt_version=PROMPT_VERSION,
            endpoint=f"{endpoint.rstrip('/')}/v1",
        )
        reply = dict(result.get("summary") or {})
        out.bullets, out.rule_flags, out.dropped = check_reply(reply, pack)
        out.verdict = str(reply.get("verdict") or "")
        if not out.bullets:
            out.error = "no cited bullet survived the check"
    except CitationRejected as exc:
        out.error = f"reply rejected: {exc}"
        out.dropped.append({"kind": "reply", "reason": str(exc)})
    except Exception as exc:  # noqa: BLE001 - a failed narration leaves the pack, never a guess
        out.error = f"{type(exc).__name__}: {exc}"
    if out.dropped:
        logging.info("Trade Mentor pick %s: dropped %s", symbol, json.dumps(out.dropped, default=str)[:1000])
    return out


def card_markdown(assessment: Assessment, *, side: str = "") -> str:
    """The card the transcript shows. The raw pack is one click away (``evidence:`` link)."""
    head = f"**Pick {assessment.symbol}{' ' + side if side else ''}**"
    link = f"[show evidence](evidence:{assessment.symbol}:{assessment.pack_hash})"
    if not assessment.narrated:
        why = assessment.error or "no narration"
        return f"{head}: no assessment ({why}). The evidence is still here: {link}"
    lines = [f"{head}: **{assessment.verdict}**", ""]
    for bullet in assessment.bullets:
        refs = " ".join(f"[{ref}]" for ref in bullet.get("evidence_refs") or ())
        lines.append(f"- {bullet.get('text', '')} {refs}".rstrip())
    for flag in assessment.rule_flags:
        word = "breaks" if flag.get("breaks") else "keeps"
        lines.append(f"- Plan {word} [{flag['plan_id']}]: {flag.get('text', '')}".rstrip(": "))
    uncited = sum(1 for drop in assessment.dropped if drop.get("reason") == "uncited")
    if uncited:
        lines.append(f"- *({uncited} uncited point{'s' if uncited != 1 else ''} dropped)*")
    lines += ["", link]
    return "\n".join(lines)
