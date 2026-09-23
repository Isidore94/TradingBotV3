"""Validated local-AI draft filling for the Trade Mentor morning check.

The trader's raw answer is durable before this module runs.  This module only
returns a draft for the existing controls; it never writes a trade or a
planned-risk field.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Mapping, Sequence

from trade_mentor_trade_check import ANSWER_STATES, MATERIAL_FIELDS

PROMPT_VERSION = "trade_mentor_ai_draft_v1"
UNITS = ("", "underlying_price", "option_premium", "dollars", "percent")

AI_DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answers", "follow_up"],
    "properties": {
        "answers": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["field", "state", "text", "value", "unit", "source_span"],
                "properties": {
                    "field": {"type": "string", "enum": list(MATERIAL_FIELDS)},
                    "state": {"type": "string", "enum": list(ANSWER_STATES)},
                    "text": {"type": "string", "maxLength": 1000},
                    "value": {"type": ["number", "null"]},
                    "unit": {"type": "string", "enum": list(UNITS)},
                    "source_span": {"type": "string", "maxLength": 500},
                },
            },
        },
        "follow_up": {"type": "string", "maxLength": 300},
    },
}


def _evidence(raw_text: str, missing_fields: Sequence[str], trade: Mapping[str, Any]) -> dict:
    body = str(raw_text or "")
    fields = [str(name) for name in missing_fields if str(name) in MATERIAL_FIELDS]
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return {
        "package_id": f"mentor-draft:{digest[:16]}",
        "evidence_hash": digest,
        "instructions": (
            "Extract only answers explicitly stated in raw_text. Return only fields in "
            "missing_fields. source_span must be an exact non-empty substring of raw_text. "
            "Never infer position size or risk. Keep an ambiguous number in text with value "
            "null and ask one short follow-up. no_fixed_target is valid only when the trader "
            "explicitly says there was no target. This is a draft for trader review."
        ),
        "trade": {
            "trade_id": str(trade.get("trade_id") or ""),
            "symbol": str(trade.get("symbol") or ""),
            "direction": str(trade.get("direction") or ""),
        },
        "missing_fields": fields,
        "raw_text": body,
    }


def validate_draft(
    payload: Mapping[str, Any], *, raw_text: str, missing_fields: Sequence[str]
) -> dict[str, Any]:
    """Apply the facts the JSON schema cannot express."""
    allowed = {str(name) for name in missing_fields if str(name) in MATERIAL_FIELDS}
    answers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in payload.get("answers") or []:
        if not isinstance(item, Mapping):
            raise ValueError("an answer is not an object")
        field = str(item.get("field") or "")
        state = str(item.get("state") or "")
        span = str(item.get("source_span") or "")
        unit = str(item.get("unit") or "")
        value = item.get("value")
        if field not in allowed or field in seen:
            raise ValueError("the draft contains an unavailable or duplicate field")
        if state not in ANSWER_STATES:
            raise ValueError("the draft contains an unknown answer state")
        if not span or span not in raw_text:
            raise ValueError("every answer needs an exact source span")
        if unit not in UNITS:
            raise ValueError("the draft contains an unknown unit")
        if value is not None and not unit:
            raise ValueError("a numeric value needs an explicit unit")
        if value is None and unit:
            raise ValueError("a unit without a numeric value is ambiguous")
        seen.add(field)
        answers.append(
            {
                "field": field,
                "state": state,
                "text": str(item.get("text") or "").strip(),
                "value": value,
                "unit": unit,
                "source_span": span,
            }
        )
    return {"answers": answers, "follow_up": str(payload.get("follow_up") or "").strip()}


def extract_draft(
    raw_text: str,
    missing_fields: Sequence[str],
    trade: Mapping[str, Any],
    *,
    request: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Ask the configured local model for one schema-checked, cited draft."""
    body = str(raw_text or "").strip()
    if not body:
        raise ValueError("the raw answer is empty")
    import ai_summary

    if request is None:
        if not ai_summary.local_provider_enabled():
            raise RuntimeError("local AI is not ready")
        request = ai_summary.request_ai_summary
    result = request(
        provider="local",
        model=ai_summary.local_model("medium"),
        api_key="",
        evidence=_evidence(body, missing_fields, trade),
        timeout_seconds=180,
        schema=AI_DRAFT_SCHEMA,
        schema_name="tradingbot_trade_mentor_draft",
        prompt_version=PROMPT_VERSION,
    )
    summary = result.get("summary") if isinstance(result, Mapping) else None
    if not isinstance(summary, Mapping):
        raise ValueError("local AI returned no draft")
    draft = validate_draft(summary, raw_text=body, missing_fields=missing_fields)
    draft.update(
        {
            "model": str(result.get("model") or ""),
            "prompt_version": str(result.get("prompt_version") or PROMPT_VERSION),
        }
    )
    return draft


def fill_blank_fields(
    store: Any,
    trade_id: str,
    raw_text: str,
    blank_fields: Sequence[str],
    trade: Mapping[str, Any],
    *,
    request: Callable[..., Mapping[str, Any]] | None = None,
    now: Any = None,
) -> list[dict[str, Any]]:
    """Fill a filed trade's BLANK fields from the trader's own words, and store them.

    Asked once (trader 2026-09-23): *"anything not filled should be filled by
    AI based on my response and if the AI cant determine then just leave it
    blank"*. No confirmation and no follow-up question: every answer the model
    returns has already passed :func:`validate_draft` (an exact source span),
    and anything it cannot quote stays blank. Each row carries
    ``filled_by: local_ai`` with the model and prompt version.

    Runs OFF the Qt thread and touches no widget. Raises on a model or write
    failure; the caller logs it and writes nothing else - a blank is never
    re-asked.
    """
    import trade_mentor_trade_check as check

    wanted = tuple(
        name for name in (str(field) for field in (blank_fields or ())) if name in MATERIAL_FIELDS
    )
    if not wanted or not str(raw_text or "").strip():
        return []
    draft = extract_draft(raw_text, wanted, trade, request=request)
    answers: dict[str, dict[str, Any]] = {}
    for answer in draft.get("answers") or ():
        name = str(answer.get("field") or "")
        if name not in wanted or name in answers:
            continue
        answers[name] = {
            "state": str(answer.get("state") or ""),
            "text": str(answer.get("text") or answer.get("source_span") or ""),
            "value": answer.get("value"),
            "unit": str(answer.get("unit") or ""),
            "source_span": str(answer.get("source_span") or ""),
        }
    if not answers:
        return []
    return check.save_answers(
        store,
        str(trade_id),
        answers,
        now=now,
        extra={
            "filled_by": "local_ai",
            "model": str(draft.get("model") or ""),
            "prompt_version": str(draft.get("prompt_version") or PROMPT_VERSION),
        },
    )
