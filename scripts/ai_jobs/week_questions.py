"""Night slot: answer the trader's pending Week Review questions from records.

For each pending question it sends the local medium model ONLY the week/day
record excerpts the question needs (`week_coach.build_evidence`). Every claim
must cite an id from that evidence; a claim with no valid citation is dropped
and counted ("uncited — not shown"). Answers are appended to
`WEEK_ANSWERS_FILE`. A failed call leaves the question pending for next night.

First, and without a model, it writes the week's frontier digest
(`records/week-<W>-frontier.md`). Evidence only: nothing reads these files to
detect, score, rank, gate or alert.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

PROMPT_VERSION = "week_questions_v1"
#: Questions answered per night; the rest wait for the next night.
MAX_PER_NIGHT = 5
RESERVE_MINUTES = 15.0

ANSWER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["claims"],
    "properties": {
        "claims": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "citations"],
                "properties": {
                    "text": {"type": "string", "maxLength": 400},
                    "citations": {
                        "type": "array",
                        "maxItems": 6,
                        "items": {"type": "string", "maxLength": 120},
                    },
                },
            },
        },
    },
}


def run_week_questions(
    *,
    session_date: str = "",
    now: datetime | None = None,
    answer: bool = True,
    request: Callable[..., Mapping[str, Any]] | None = None,
    model: str = "",
    root: Path | None = None,
    questions_path: Path | None = None,
    answers_path: Path | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    import day_session_record
    import week_coach

    outputs: list[str] = []
    notes: list[str] = []
    session = str(session_date or "")[:10]
    paths = {"questions_path": questions_path, "answers_path": answers_path}
    if session:
        try:
            week = day_session_record.week_key(session)
            written = week_coach.write_frontier(week, root=root, **paths)
            if written:
                outputs.append(written["path"])
            else:
                notes.append(f"no week record for {week}, so no frontier digest")
        except Exception as exc:  # noqa: BLE001 - the answers do not depend on the digest
            notes.append(f"frontier digest not written: {exc}")

    pending = week_coach.pending_questions(**paths)
    if not pending:
        return {"status": "ok", "model": "", "reason": "; ".join(["no pending questions", *notes]), "outputs": outputs}
    if not answer:
        return {"status": "ok", "model": "", "reason": "; ".join(["deterministic half only; questions stay pending", *notes]), "outputs": outputs}

    import ai_summary

    if request is None:
        if not ai_summary.local_provider_enabled():
            return {
                "status": "degraded_no_narrative", "model": "",
                "reason": f"local AI is not configured; {len(pending)} question(s) stay pending",
                "outputs": outputs,
            }
        request = ai_summary.request_ai_summary
    model_name = model or ai_summary.local_model("medium")
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    answered = failed = 0
    reasons: list[str] = []
    for question in pending[:MAX_PER_NIGHT]:
        try:
            evidence = week_coach.build_evidence(question, root=root)
            allowed = evidence.pop("_allowed")
            if not allowed:
                checked: dict[str, Any] = {"shown": [], "dropped": []}
                used = ""
            else:
                result = request(
                    provider="local", model=model_name, api_key="", evidence=evidence,
                    timeout_seconds=900, schema=ANSWER_JSON_SCHEMA,
                    schema_name="tradingbot_week_question", prompt_version=PROMPT_VERSION,
                )
                reply = result.get("summary") if isinstance(result, Mapping) else None
                if not isinstance(reply, Mapping):
                    raise ValueError("local AI returned no answer")
                checked = week_coach.validate_claims(reply.get("claims"), allowed)
                used = str(result.get("model") or model_name)
            row = week_coach.answer_row(
                question, checked, model=used, answered_at=moment.astimezone(timezone.utc),
                trades_total=int(evidence.get("trades_total") or 0),
            )
            week_coach.append_row(row, week_coach._answers_path(answers_path))
            answered += 1
        except Exception as exc:  # noqa: BLE001 - this question stays pending
            failed += 1
            reasons.append(f"{question.get('id')}: {exc}")
    if answered:
        outputs.append(str(week_coach._answers_path(answers_path)))
    left = len(pending) - answered
    reason = f"answered {answered}, failed {failed}, still pending {left}"
    if reasons:
        reason += " (" + "; ".join(reasons[:3]) + ")"
    return {
        "status": "degraded_no_narrative" if failed else "ok",
        "model": model_name,
        "reason": "; ".join([reason, *notes]),
        "outputs": outputs,
    }
