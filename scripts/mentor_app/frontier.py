"""The frontier switch (P11): a metered Claude call over the SAME evidence the local model read. Qt-free.

Off by default (``mentor_frontier_enabled``), keyed only through the credential store
(``secret_store.load_secret("anthropic_api_key")``), capped per PT day in USD from a dated
pricing table, and never automatic: only ``/think``, ``/think pick SYM``, ``/think week`` and
the header's "Think harder" button call it.

Every call goes through :func:`metered_request`, a drop-in for ``ai_summary.request_ai_summary``:
the caller's evidence, schema and system instruction pass through unchanged (provider switched
to ``anthropic``). Before the call it estimates the cost (input chars / 4 at the input price plus
MAX_OUTPUT_TOKENS at the output price) and refuses when today's spend plus the estimate passes
the cap; after it, the real ``usage`` counts are priced and written to ``frontier_usage``. The
reply goes through the same citation check as the local path.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from mentor_packs.citations import CitationRejected, check_citations

PT = ZoneInfo("America/Los_Angeles")
#: USD per million tokens (input, output), from the claude-api skill's model table (cached 2026-06-24).
PRICING_AS_OF = "2026-06-24"
PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
#: Models that take ``output_config.effort`` (Haiku 4.5 rejects it).
EFFORT_MODELS = frozenset({"claude-sonnet-5", "claude-opus-5"})
EFFORT = "medium"
#: Per-call output cap (thinking included); the estimate prices all of it.
MAX_OUTPUT_TOKENS = 4000
CHARS_PER_TOKEN = 4
TIMEOUT_SECONDS = 240
WEEK_DAYS = 28
MAX_WEEK_DIGESTS = 20
PROVIDER = "anthropic"
SECRET_NAME = "anthropic_api_key"
PURPOSES = ("chat", "pick", "week")

CHAT_SCHEMA_NAME = "trade_mentor_frontier_chat"
CHAT_PROMPT_VERSION = "trade_mentor_frontier_chat_v1"
WEEK_SCHEMA_NAME = "trade_mentor_frontier_week"
WEEK_PROMPT_VERSION = "trade_mentor_frontier_week_v1"

_LINE = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text", "evidence_refs"],
    "properties": {
        "text": {"type": "string", "maxLength": 300},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
    },
}
CHAT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "array", "maxItems": 6, "items": _LINE}},
}
WEEK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["observations", "questions"],
    "properties": {
        "observations": {"type": "array", "maxItems": 5, "items": _LINE},
        "questions": {"type": "array", "maxItems": 3, "items": _LINE},
    },
}
CHAT_TASK = (
    "Answer the trader's question again, more carefully, from the same desk context, memory and packs "
    "the local mentor read. At most 6 short points; every point cites the ids (the [id] at the start "
    "of a line) it uses in evidence_refs. A point with no id is thrown away; an id not in these texts "
    "throws the whole answer away. Missing data is unknown. Never size or place an order, never propose "
    "changing a detector, a score, an alert or the plan."
)
WEEK_TASK = (
    "Read the last four weeks of the Trade Mentor's night digests, the trader's mirror (his own record "
    "in cuts with n) and the open hypotheses (cells in a shadow grid). Write at most 5 observations and "
    "at most 3 questions for the trader. Each cites the source_id of every row it uses. Say n with every "
    "rate; 'too few' is not evidence. A hypothesis is a shadow-grid cell, never a rule. Never size or "
    "place an order, never propose changing a detector, a score, an alert or the plan."
)
FOOTER = "*A frontier read of the same evidence; the numbers are the desk's.*"


class FrontierRefused(RuntimeError):
    """The switch, the key, the price table or the daily cap says no; nothing was sent."""


# ---------------------------------------------------------------- the switch, the key, the cap
def day_pt(now: datetime) -> str:
    moment = now if now.tzinfo else now.astimezone()
    return moment.astimezone(PT).date().isoformat()


def load_key() -> str:
    import secret_store

    return secret_store.load_secret(SECRET_NAME)


def price(model: str) -> tuple[float, float] | None:
    return PRICING_USD_PER_MTOK.get(str(model or "").strip())


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = price(model)
    if rates is None:
        raise FrontierRefused(f"no price for {model} in the pricing table (as of {PRICING_AS_OF})")
    return (max(0, int(input_tokens)) * rates[0] + max(0, int(output_tokens)) * rates[1]) / 1_000_000


def estimate_usd(model: str, input_chars: int, max_output: int = MAX_OUTPUT_TOKENS) -> float:
    """chars/4 at the input price plus the whole output cap at the output price."""
    return cost_usd(model, -(-int(input_chars) // CHARS_PER_TOKEN), int(max_output))


def cap_message(spent: float, cap: float) -> str:
    return f"frontier cap reached (${spent:.2f} of ${cap:.2f})"


@dataclass(frozen=True)
class Status:
    enabled: bool
    model: str
    cap_usd: float
    spent_usd: float | None
    key_present: bool
    reason: str = ""

    @property
    def usable(self) -> bool:
        return not self.reason


def status(store: Any, now: datetime, *, enabled: bool | None = None, model: str | None = None,
           cap: float | None = None, key_loader: Callable[[], str] = load_key) -> Status:
    """Why the frontier may not be called right now ("" = it may). Reads the key: call it off the Qt thread."""
    from mentor_app import settings

    enabled = settings.frontier_enabled() if enabled is None else bool(enabled)
    model = model or settings.frontier_model()
    cap = settings.frontier_daily_cap_usd() if cap is None else float(cap)
    spent = store.frontier_spent(day_pt(now))
    if not enabled:
        return Status(False, model, cap, spent, False,
                      "the frontier switch is off (Settings: mentor_frontier_enabled)")
    key_present = bool(key_loader())
    if price(model) is None:
        reason = f"no price for {model} in the pricing table (as of {PRICING_AS_OF})"
    elif not key_present:
        reason = f"no Anthropic key saved (Credential Manager, {SECRET_NAME})"
    elif spent is None:
        reason = "today's frontier spend cannot be read, so nothing is sent"
    elif spent >= cap:
        reason = cap_message(spent, cap)
    else:
        reason = ""
    return Status(True, model, cap, spent, key_present, reason)


def status_text(state: Status) -> str:
    """What ``/frontier`` prints: the switch, the model, today's spend and the cap."""
    spent = "unknown" if state.spent_usd is None else f"${state.spent_usd:.4f}"
    lines = [
        "**Frontier**",
        "",
        f"- switch: {'on' if state.enabled else 'off'}; model {state.model}"
        f" (prices as of {PRICING_AS_OF})",
        f"- today (PT): {spent} of ${state.cap_usd:.2f}",
    ]
    if state.reason:
        lines.append(f"- not available: {state.reason}")
    lines.append("- never automatic: only `/think`, `/think pick SYM`, `/think week` or Think harder")
    return "\n".join(lines)


# ---------------------------------------------------------------- the metered call
def input_chars(kwargs: Mapping[str, Any]) -> int:
    """What the request sends as input: evidence, schema and system text (a slight over-count)."""
    import ai_summary

    system = kwargs.get("system_instruction")
    system_text = ai_summary._system_instruction() if system is None else str(system)
    evidence = json.dumps(kwargs.get("evidence") or {}, sort_keys=True, default=str)
    schema = json.dumps(kwargs.get("schema") or {}, sort_keys=True, default=str)
    return len(evidence) + len(schema) + len(system_text)


def frontier_post(post: Callable[..., Any] | None = None, *, model: str) -> Callable[..., Any]:
    """Innermost ``post`` wrapper: the frontier output cap and, where supported, the effort."""
    import requests

    base = post or requests.post

    def wrapped(url: str, **kwargs: Any) -> Any:
        payload = dict(kwargs.get("json") or {})
        payload["max_tokens"] = MAX_OUTPUT_TOKENS
        payload.pop("reasoning_effort", None)
        if model in EFFORT_MODELS:
            config = dict(payload.get("output_config") or {})
            config["effort"] = EFFORT
            payload["output_config"] = config
        kwargs["json"] = payload
        return base(url, **kwargs)

    return wrapped


def metered_request(
    *,
    store: Any,
    purpose: str,
    model: str,
    api_key: str,
    cap_usd: float,
    now: Callable[[], datetime],
    request: Callable[..., Mapping[str, Any]] | None = None,
    spent_sink: list[dict[str, Any]] | None = None,
) -> Callable[..., Mapping[str, Any]]:
    """A ``request_ai_summary`` drop-in: cost guard, then the anthropic call, then the usage row."""
    import ai_summary

    call = request or ai_summary.request_ai_summary

    def run(**kwargs: Any) -> Mapping[str, Any]:
        if not api_key:
            raise FrontierRefused(f"no Anthropic key saved (Credential Manager, {SECRET_NAME})")
        estimate = estimate_usd(model, input_chars(kwargs))
        day = day_pt(now())
        spent = store.frontier_spent(day)
        if spent is None:
            raise FrontierRefused("today's frontier spend cannot be read, so nothing is sent")
        if spent + estimate > cap_usd:
            raise FrontierRefused(cap_message(spent, cap_usd))
        sent = {**kwargs, "provider": PROVIDER, "model": model, "api_key": api_key,
                "timeout_seconds": max(int(kwargs.get("timeout_seconds") or 0), TIMEOUT_SECONDS)}
        try:
            result = call(**sent)
        except Exception:
            # The request may have been billed; the estimate stands in so the cap stays honest.
            _record(store, day, f"{purpose}:failed", model, None, None, estimate, False, now, spent_sink)
            raise
        usage = dict((result or {}).get("usage") or {})
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if isinstance(prompt, int) and isinstance(completion, int):
            _record(store, day, purpose, model, prompt, completion, cost_usd(model, prompt, completion), True, now,
                    spent_sink)
        else:
            _record(store, day, purpose, model, None, None, estimate, False, now, spent_sink)
        return result

    return run


def _record(store: Any, day: str, purpose: str, model: str, prompt: int | None, completion: int | None,
            usd: float, measured: bool, now: Callable[[], datetime], sink: list[dict[str, Any]] | None) -> None:
    row = {"day_pt": day, "purpose": purpose, "model": model, "input_tokens": prompt, "output_tokens": completion,
           "est_usd": round(float(usd), 6), "measured": measured}
    if store.add_frontier_usage(**row, ts_utc=now().astimezone(timezone.utc).isoformat(timespec="seconds")) is None:
        logging.error("Trade Mentor frontier: the usage row for %s could not be written", purpose)
    if sink is not None:
        sink.append(row)


# ---------------------------------------------------------------- answers
@dataclass
class ThinkResult:
    purpose: str
    model: str
    title: str = ""
    points: list[dict[str, Any]] = field(default_factory=list)
    questions: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    usage: list[dict[str, Any]] = field(default_factory=list)
    reply: dict[str, Any] = field(default_factory=dict)

    @property
    def usd(self) -> float:
        return round(sum(float(row.get("est_usd") or 0) for row in self.usage), 6)


_LINE_ID = re.compile(r"^\s*(?:[-*]\s*)?\[([^\]\s]+)\]", re.MULTILINE)


def ids_in(texts: Iterable[str]) -> set[str]:
    """Every ``[id]`` that starts a line of the given pack / memory texts."""
    return {match.group(1) for text in texts for match in _LINE_ID.finditer(str(text or ""))}


def chat_evidence(question: str, *, context_text: str, memory_block: str,
                  pack_texts: Sequence[str]) -> dict[str, Any]:
    """The same texts the local turn saw, verbatim: desk context, memory block, the packs its tools read."""
    return {"task": CHAT_TASK, "question": str(question), "desk_context": str(context_text or ""),
            "memory": str(memory_block or ""), "packs": [str(text) for text in pack_texts]}


def _ask(request: Callable[..., Mapping[str, Any]], *, evidence: Mapping[str, Any], schema: Mapping[str, Any],
         schema_name: str, prompt_version: str, system_instruction: str | None,
         post: Callable[..., Any]) -> dict[str, Any]:
    result = request(provider=PROVIDER, model="", api_key="", evidence=evidence, timeout_seconds=TIMEOUT_SECONDS,
                     post=post, schema=schema, schema_name=schema_name, prompt_version=prompt_version,
                     system_instruction=system_instruction)
    reply = (result or {}).get("summary")
    if not isinstance(reply, Mapping):
        raise CitationRejected("the reply was not an object")
    return dict(reply)


def think_chat(question: str, *, context_text: str, memory_block: str, pack_texts: Sequence[str], model: str,
               request: Callable[..., Mapping[str, Any]], post: Callable[..., Any]) -> ThinkResult:
    """Re-ask the last question over the same texts; the same citation rule as every structured reply."""
    import ai_summary
    from mentor_app.chat_model import PERSONA_PROMPT

    out = ThinkResult("chat", model, title=f"Think harder: {str(question)[:120]}")
    allowed = ids_in([context_text, memory_block, *pack_texts])
    try:
        out.reply = _ask(request, evidence=chat_evidence(question, context_text=context_text,
                                                         memory_block=memory_block, pack_texts=pack_texts),
                         schema=CHAT_SCHEMA, schema_name=CHAT_SCHEMA_NAME, prompt_version=CHAT_PROMPT_VERSION,
                         system_instruction=ai_summary.persona_instruction(PERSONA_PROMPT), post=post)
        checked = check_citations({"bullets": out.reply.get("answer")}, allowed)
        out.points = list(checked.kept)
        out.dropped = [{"reason": "uncited", **row} for row in checked.dropped]
        if not out.points:
            out.error = "no cited point survived the check"
    except CitationRejected as exc:
        out.error = f"reply rejected: {exc}"
    except Exception as exc:  # noqa: BLE001 - a failed call is shown as failed, never replaced
        out.error = f"{type(exc).__name__}: {exc}"
    return out


def week_rows(*, digests: Iterable[Mapping[str, Any]], mirror: Any, hypotheses: Any,
              now: datetime) -> list[dict[str, str]]:
    """Rows for /think week: the last 28 days of digest items, the mirror pack, open hypotheses."""
    from mentor_app import memory

    since = (now.astimezone(PT).date() - timedelta(days=WEEK_DAYS)).isoformat()
    recent = [row for row in digests if str(row.get("session_date") or "")[:10] >= since]
    rows = [{"source_id": item.id, "text": f"({item.day}) {item.text}"} for item in memory.digest_items(recent)]
    for pack in (mirror, hypotheses):
        for row in getattr(pack, "rows", ()) or ():
            if not row.get("id"):
                continue
            if getattr(pack, "name", "") == "hypothesis_pack" and row.get("status") == "graded":
                continue
            rows.append({"source_id": str(row["id"]), "text": str(row.get("text") or "")})
    graded = {str(row["id"]) for row in getattr(hypotheses, "rows", ()) or () if row.get("status") == "graded"}
    return [row for row in rows if not any(row["source_id"] == f"{hyp}:cell" for hyp in graded)]


def think_week(rows: Sequence[Mapping[str, str]], *, model: str, request: Callable[..., Mapping[str, Any]],
               post: Callable[..., Any]) -> ThinkResult:
    out = ThinkResult("week", model, title="Think: the last four weeks")
    allowed = {str(row["source_id"]) for row in rows}
    if not rows:
        out.error = "no digests, mirror or hypotheses to read yet"
        return out
    try:
        out.reply = _ask(request, evidence={"task": WEEK_TASK, "rows": [dict(row) for row in rows]},
                         schema=WEEK_SCHEMA, schema_name=WEEK_SCHEMA_NAME, prompt_version=WEEK_PROMPT_VERSION,
                         system_instruction=None, post=post)
        kept = check_citations({"bullets": out.reply.get("observations")}, allowed)
        asked = check_citations({"bullets": out.reply.get("questions")}, allowed)
        out.points, out.questions = list(kept.kept), list(asked.kept)
        out.dropped = [{"reason": "uncited", **row} for row in (*kept.dropped, *asked.dropped)]
        if not out.points and not out.questions:
            out.error = "no cited point survived the check"
    except CitationRejected as exc:
        out.error = f"reply rejected: {exc}"
    except Exception as exc:  # noqa: BLE001
        out.error = f"{type(exc).__name__}: {exc}"
    return out


def load_digests(ai_root: Path | str | None = None) -> list[dict[str, Any]]:
    """The newest night digests (off the Qt thread); [] when none can be read."""
    from ai_jobs.mentor_review import DIGEST_STEM, read_published
    from mentor_app import memory

    try:
        root = Path(ai_root) if ai_root is not None else memory._digests_root()
        return read_published(root, DIGEST_STEM, limit=MAX_WEEK_DIGESTS) if root is not None else []
    except Exception:  # noqa: BLE001
        logging.warning("Trade Mentor frontier: the night digests could not be read", exc_info=True)
        return []


def label(model: str) -> str:
    return f"frontier: {model}"


def spend_line(usage: Sequence[Mapping[str, Any]], spent_after: float | None, cap: float) -> str:
    usd = sum(float(row.get("est_usd") or 0) for row in usage)
    measured = all(row.get("measured") for row in usage) if usage else False
    today = "unknown" if spent_after is None else f"${spent_after:.4f}"
    return (f"cost ${usd:.4f}{'' if measured else ' (estimated)'}; today {today} of ${cap:.2f}")


def _line(row: Mapping[str, Any]) -> str:
    refs = " ".join(f"[{ref}]" for ref in row.get("evidence_refs") or ())
    return f"- {row.get('text', '')} {refs}".rstrip()


def card_markdown(result: ThinkResult, *, spend: str = "") -> str:
    head = f"**{label(result.model)}** · {result.title}"
    lines = [head, ""]
    if result.error and not result.points and not result.questions:
        lines.append(f"No frontier answer ({result.error}).")
    else:
        lines += [_line(row) for row in result.points]
        if result.questions:
            lines += ["", "*Questions*"] + [_line(row) for row in result.questions]
        if result.dropped:
            count = len(result.dropped)
            lines.append(f"*({count} uncited point{'s' if count != 1 else ''} dropped)*")
    if spend:
        lines += ["", f"*{spend}*"]
    lines += ["", FOOTER]
    return "\n".join(lines)
