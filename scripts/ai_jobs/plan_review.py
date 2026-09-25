"""`plan_review` - the night argues with the trader's plan (P1-7 7b).

A stage-3 model slot. The local model gets the plan's lines, the session's day
report card, its read grades, the measured report and (when the file exists)
P1-4's permutation report, and may return at most :data:`MAX_CHALLENGES`
challenges. Each must cite ONE plan line id and ONE evidence id that tonight
really carries. The citation check is the night ideas slot's rule: a citation the
night does not carry rejects the whole answer, and a challenge with a missing
citation is dropped. The model computes nothing; every number is already in
the evidence.

Its deterministic half (``ask=False``: forced by day, model down, or budget
cut) only reads the plan, which snapshots a changed plan.

It writes `PLAN_CHALLENGES_FILE` (append-only) and nothing else: the plan
itself is written only by the trader's Accept click on the Mentor card.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ai_jobs import ledger

_log = logging.getLogger(__name__)

PROMPT_VERSION = "plan_review_v1"
SCHEMA_NAME = "tradingbot_plan_review"
MAX_CHALLENGES = 3
MAX_CHALLENGE_CHARS = 280
TIMEOUT_SECONDS = 900
RESERVE_MINUTES = 10.0
#: Bounds on what one night sends. What does not fit is counted, never silently dropped.
MAX_ROWS_PER_SECTION = 12
MAX_MEASURED_CELLS = 40
MAX_PERMUTATION_CHARS = 4000
MAX_FIELD_CHARS = 200

DROP_NOT_AN_OBJECT = "not_an_object"
DROP_NO_TEXT = "no_text"
DROP_TOO_LONG = "too_long"
DROP_NO_PLAN_LINE = "no_plan_line"
DROP_NO_EVIDENCE = "no_evidence"

INSTRUCTIONS = (
    "You are checking ONE trader's own trading plan against their own recorded "
    "numbers. Return at most three challenges and nothing else. Each challenge "
    "names ONE plan line by copying its id exactly from allowed_plan_line_ids, "
    "ONE piece of evidence by copying its id exactly from allowed_evidence_ids, "
    "and says in one or two plain sentences how that evidence argues with that "
    "line. Do not calculate a statistic, do not treat an unmeasured row as a "
    "fact, and do not name, rank or score a symbol. Say nothing rather than "
    "something the evidence does not carry."
)

CHALLENGES_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["challenges"],
    "properties": {
        "challenges": {
            "type": "array",
            "maxItems": MAX_CHALLENGES,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["plan_line", "evidence", "text"],
                "properties": {
                    "plan_line": {"type": "string", "maxLength": 80},
                    "evidence": {"type": "string", "maxLength": 200},
                    "text": {"type": "string", "maxLength": MAX_CHALLENGE_CHARS},
                },
            },
        }
    },
}


class PlanReviewRejected(ValueError):
    """The night's answer was not a set of cited challenges about tonight's evidence."""


def _text(value: Any) -> str:
    return str(value or "").strip()


def _moment(now: datetime | None) -> str:
    stamp = now or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.astimezone()
    return stamp.astimezone(timezone.utc).isoformat(timespec="seconds")


def _compact(row: Mapping[str, Any]) -> dict[str, Any]:
    """Scalar fields only, strings bounded."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, str):
            out[str(key)] = value[:MAX_FIELD_CHARS]
        elif value is None or isinstance(value, (bool, int, float)):
            out[str(key)] = value
    return out


def _bounded(rows: Sequence[Any], limit: int) -> tuple[list[Mapping[str, Any]], int]:
    kept = [row for row in list(rows) if isinstance(row, Mapping)]
    return kept[:limit], max(0, len(kept) - limit)


# ---------------------------------------------------------------------------
# tonight's inputs
# ---------------------------------------------------------------------------
def _day_sections(session: str, root: Path | None) -> tuple[dict[str, Any], list[str]]:
    import day_review_pack

    pack = day_review_pack.read_pack(session, root=root) or {}
    ids: list[str] = []

    def qualify(row: Mapping[str, Any]) -> dict[str, Any]:
        body = _compact(row)
        body["source_id"] = f"{session}/{_text(row.get('source_id'))}"
        ids.append(body["source_id"])
        return body

    card = pack.get("report_card") if isinstance(pack.get("report_card"), Mapping) else {}
    lines, lines_over = _bounded((card or {}).get("lines") or (), MAX_ROWS_PER_SECTION)
    reads, reads_over = _bounded(pack.get("reads") or (), MAX_ROWS_PER_SECTION)
    congruence, congruence_over = _bounded(pack.get("congruence") or (), MAX_ROWS_PER_SECTION)
    body = {
        "session": session,
        "has_facts": bool(pack),
        "report_card": [qualify(row) for row in lines if _text(row.get("source_id"))],
        "grades": {
            "reads": [qualify(row) for row in reads if _text(row.get("source_id"))],
            "congruence": [qualify(row) for row in congruence if _text(row.get("source_id"))],
        },
        "omitted": {"report_card": lines_over, "reads": reads_over, "congruence": congruence_over},
    }
    return body, ids


def _measured(session: str, ai_root: Path | None) -> tuple[dict[str, Any], list[str]]:
    from ai_jobs import measured_report_publish

    try:
        if ai_root is None:
            from ai_jobs import digest

            ai_root = digest._default_root()
        payload = measured_report_publish.latest_published(Path(ai_root), session)
    except Exception:  # noqa: BLE001 - no report is simply no report
        _log.debug("The measured report could not be read.", exc_info=True)
        payload = {}
    cells = [cell for cell in (payload or {}).get("cells") or () if isinstance(cell, Mapping)]
    kept, over = _bounded(cells, MAX_MEASURED_CELLS)
    rows, ids = [], []
    for cell in kept:
        cell_id = _text(cell.get("cell_id"))
        if not cell_id:
            continue
        source = f"measured:{cell_id}"
        rows.append(
            {
                "source_id": source,
                "metric": _text(cell.get("metric")),
                "unit": _text(cell.get("unit")),
                "value": cell.get("value"),
                "n": cell.get("n"),
                "state": _text(cell.get("state")),
                "population": _text(cell.get("population"))[:MAX_FIELD_CHARS],
                "unavailable": _text(cell.get("unavailable"))[:MAX_FIELD_CHARS],
            }
        )
        ids.append(source)
    return {"report_id": _text((payload or {}).get("report_id")), "cells": rows, "omitted": over}, ids


def _permutation() -> tuple[dict[str, Any], list[str]]:
    """P1-4's report when the file exists, as bounded text under one id. Never imports P1-4."""
    import project_paths

    path = Path(getattr(project_paths, "PERMUTATION_REPORT_FILE", ""))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, []
    text = json.dumps(payload, sort_keys=True, default=str)
    return (
        {
            "source_id": "permutation:report",
            "text": text[:MAX_PERMUTATION_CHARS],
            "truncated": len(text) > MAX_PERMUTATION_CHARS,
        },
        ["permutation:report"],
    )


def build_inputs(
    session_date: Any,
    *,
    plan: Mapping[str, Any],
    root: Path | None = None,
    ai_root: Path | None = None,
) -> dict[str, Any]:
    """Everything tonight's review may see, already read. ``inputs_hash`` ignores the clock."""
    session = _text(session_date)[:10]
    parsed = plan.get("parsed") if isinstance(plan.get("parsed"), Mapping) else {}
    plan_lines = [
        {"id": _text(row.get("id")), "section": _text(row.get("section")), "text": _text(row.get("text"))}
        for row in (parsed or {}).get("lines") or ()
        if isinstance(row, Mapping) and _text(row.get("id"))
    ]
    day, day_ids = _day_sections(session, root)
    measured, measured_ids = _measured(session, ai_root)
    permutation, permutation_ids = _permutation()
    evidence_ids: list[str] = []
    for item in (*day_ids, *measured_ids, *permutation_ids):
        if item not in evidence_ids:
            evidence_ids.append(item)
    body: dict[str, Any] = {
        "session_date": session,
        "plan_lines": plan_lines,
        "allowed_plan_line_ids": [row["id"] for row in plan_lines],
        "day": day,
        "measured_report": measured,
        "permutation_report": permutation,
        "allowed_evidence_ids": evidence_ids,
    }
    body["inputs_hash"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return body


def build_evidence(inputs: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in inputs.items() if key != "inputs_hash"}
    body["package_id"] = f"plan-review:{_text(inputs.get('inputs_hash'))[:16]}"
    body["evidence_hash"] = _text(inputs.get("inputs_hash"))
    body["instructions"] = INSTRUCTIONS
    return body


# ---------------------------------------------------------------------------
# the citation check (the night ideas slot's rule)
# ---------------------------------------------------------------------------
def drop_reason(item: Any) -> str:
    """Why this one challenge cannot be stored, or ``""``. An uncited claim is dropped."""
    if not isinstance(item, Mapping):
        return DROP_NOT_AN_OBJECT
    text = _text(item.get("text"))
    if not text:
        return DROP_NO_TEXT
    if len(text) > MAX_CHALLENGE_CHARS:
        return DROP_TOO_LONG
    if not _text(item.get("plan_line")):
        return DROP_NO_PLAN_LINE
    if not _text(item.get("evidence")):
        return DROP_NO_EVIDENCE
    return ""


def check_challenges(reply_body: Any, inputs: Mapping[str, Any]) -> None:
    """Re-check every bound against tonight's input. Raises :class:`PlanReviewRejected`.

    A citation tonight does not carry rejects the answer WHOLE; a challenge that
    cites nothing is dropped one at a time by :func:`usable_challenges`.
    """
    if not isinstance(reply_body, Mapping):
        raise PlanReviewRejected("the night's answer was not an object")
    rows = reply_body.get("challenges")
    if rows is None or not isinstance(rows, (list, tuple)):
        raise PlanReviewRejected("the night's answer carried no challenges array")
    usable = [row for row in rows if not drop_reason(row)]
    if len(usable) > MAX_CHALLENGES:
        raise PlanReviewRejected(
            f"the night offered {len(usable)} usable challenges; at most {MAX_CHALLENGES} are allowed"
        )
    plan_ids = {_text(item) for item in inputs.get("allowed_plan_line_ids") or ()}
    evidence_ids = {_text(item) for item in inputs.get("allowed_evidence_ids") or ()}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise PlanReviewRejected(f"challenge {index} was not an object")
        line = _text(row.get("plan_line"))
        if line and line not in plan_ids:
            raise PlanReviewRejected(f"challenge {index} cited plan line {line!r}, which tonight does not carry")
        cited = _text(row.get("evidence"))
        if cited and cited not in evidence_ids:
            raise PlanReviewRejected(f"challenge {index} cited {cited!r}, which tonight does not carry")


def _validate(payload: Any) -> dict[str, Any]:
    """The closed schema, top level and one level into the challenges array."""
    import ai_summary

    body = ai_summary.validate_structured_output(payload, CHALLENGES_JSON_SCHEMA, name="plan_review")
    item_schema = json.loads(json.dumps(CHALLENGES_JSON_SCHEMA["properties"]["challenges"]["items"]))
    for spec in item_schema["properties"].values():
        spec.pop("maxLength", None)
    body["challenges"] = [
        ai_summary.validate_structured_output(row, item_schema, name=f"plan_review.challenges[{index}]")
        for index, row in enumerate(list(body.get("challenges") or ()))
    ]
    return body


def usable_challenges(
    rows: Sequence[Any],
    *,
    inputs: Mapping[str, Any],
    stored_ids: set[str],
    session: str,
    model: str,
    now: datetime | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The rows to append, and the counts for the ledger."""
    import plan_challenges

    words = {row["id"]: row["text"] for row in inputs.get("plan_lines") or ()}
    sections = {row["id"]: row["section"] for row in inputs.get("plan_lines") or ()}
    reasons: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    repeats = 0
    for row in rows:
        reason = drop_reason(row)
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        line = _text(row.get("plan_line"))
        evidence = _text(row.get("evidence"))
        text = _text(row.get("text"))
        challenge_id = plan_challenges.mint_challenge_id(words.get(line, ""), evidence, text)
        if challenge_id in stored_ids or any(item["challenge_id"] == challenge_id for item in out):
            repeats += 1
            continue
        out.append(
            {
                "schema": plan_challenges.SCHEMA,
                "kind": plan_challenges.ROW_CHALLENGE,
                "challenge_id": challenge_id,
                "session_date": session,
                "created_at": _moment(now),
                "plan_line": line,
                "plan_line_section": sections.get(line, ""),
                "plan_line_text": words.get(line, ""),
                "evidence": evidence,
                "text": text,
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "inputs_hash": _text(inputs.get("inputs_hash")),
            }
        )
    counts = {
        "offered": len(list(rows)),
        "stored": len(out),
        "dropped": sum(reasons.values()),
        "drop_reasons": dict(sorted(reasons.items())),
        "repeats": repeats,
    }
    return out, counts


def _already_tonight(session: str, inputs_hash: str) -> bool:
    import plan_challenges

    return any(
        _text(row.get("session_date"))[:10] == session
        and _text(row.get("inputs_hash")) == inputs_hash
        and _text(row.get("prompt_version")) == PROMPT_VERSION
        for row in plan_challenges.read_nights()
    )


# ---------------------------------------------------------------------------
# the slot
# ---------------------------------------------------------------------------
def run_plan_review(
    *,
    session_date: str = "",
    now: datetime | None = None,
    root: Path | None = None,
    ai_root: Path | None = None,
    request: Callable[..., Mapping[str, Any]] | None = None,
    ask: bool = True,
    force: bool = False,
    **_ignored: Any,
) -> dict[str, Any]:
    """One night's plan review. Never raises."""
    import plan_challenges
    import trading_plan

    session = _text(session_date)[:10] or datetime.now().date().isoformat()
    try:
        plan = trading_plan.read_plan(create=False, snapshot=True, now=now)
    except Exception as exc:  # noqa: BLE001 - an unreadable plan is a recorded row
        return {"status": ledger.STATUS_FAILED, "model": "",
                "reason": f"the trading plan could not be read: {exc}", "outputs": []}
    if plan.get("error"):
        return {"status": ledger.STATUS_FAILED, "model": "", "reason": plan["error"], "outputs": []}
    if not plan.get("exists"):
        return {"status": ledger.STATUS_SKIPPED, "model": "",
                "reason": "there is no trading plan yet; nothing to review", "outputs": []}
    snapshot = [plan["snapshot"]] if plan.get("snapshot") else []
    if not ask:
        return {"status": ledger.STATUS_OK, "model": "",
                "reason": "plan read and history checked; no model asked", "outputs": snapshot}

    try:
        inputs = build_inputs(session, plan=plan, root=root, ai_root=ai_root)
    except Exception as exc:  # noqa: BLE001
        _log.debug("The plan review inputs could not be read.", exc_info=True)
        return {"status": ledger.STATUS_FAILED, "model": "",
                "reason": f"tonight's evidence could not be read: {exc}", "outputs": snapshot}
    if not inputs["allowed_plan_line_ids"]:
        return {"status": ledger.STATUS_SKIPPED, "model": "",
                "reason": "the plan has no lines to cite yet, so no model was loaded", "outputs": snapshot}
    if not inputs["allowed_evidence_ids"]:
        return {"status": ledger.STATUS_SKIPPED, "model": "",
                "reason": f"tonight carries no evidence to cite for {session}, so no model was loaded",
                "outputs": snapshot}
    digest = _text(inputs.get("inputs_hash"))
    if not force and _already_tonight(session, digest):
        return {"status": ledger.STATUS_OK, "model": "",
                "reason": f"the plan and evidence for {session} are unchanged; no model was asked",
                "outputs": [str(plan_challenges.challenges_path())]}

    import ai_summary

    if request is None:
        request = ai_summary.request_ai_summary
    try:
        model = ai_summary.local_model("medium")
    except Exception:  # noqa: BLE001 - a test's request needs no configured model
        model = ""
    try:
        result = request(
            provider="local",
            model=model,
            api_key="",
            evidence=build_evidence(inputs),
            timeout_seconds=TIMEOUT_SECONDS,
            schema=CHALLENGES_JSON_SCHEMA,
            schema_name=SCHEMA_NAME,
            prompt_version=PROMPT_VERSION,
        )
    except Exception as exc:  # noqa: BLE001 - the store is untouched
        _log.debug("The plan review could not ask its model.", exc_info=True)
        return {"status": ledger.STATUS_DEGRADED, "model": "",
                "reason": f"no local model answered tonight's plan review: {exc}", "outputs": snapshot}

    answered = _text((result or {}).get("model")) or model
    try:
        body = _validate((result or {}).get("summary"))
        check_challenges(body, inputs)
    except Exception as exc:  # noqa: BLE001 - a breach rejects the answer whole
        _log.debug("Tonight's plan review was rejected.", exc_info=True)
        return {"status": ledger.STATUS_FAILED, "model": "",
                "reason": f"tonight's challenges were rejected and nothing was stored: {exc}",
                "outputs": snapshot}

    stored_ids = {_text(row.get("challenge_id")) for row in plan_challenges.read_challenges()}
    rows, counts = usable_challenges(
        list(body.get("challenges") or ()), inputs=inputs, stored_ids=stored_ids,
        session=session, model=answered, now=now,
    )
    night = {
        "schema": plan_challenges.SCHEMA,
        "kind": plan_challenges.ROW_NIGHT,
        "session_date": session,
        "inputs_hash": digest,
        "prompt_version": PROMPT_VERSION,
        "asked_at": _moment(now),
        "model": answered,
        "counts": counts,
    }
    try:
        path = plan_challenges._append([*rows, night], plan_challenges.challenges_path())
    except OSError as exc:
        return {"status": ledger.STATUS_FAILED, "model": answered,
                "reason": f"the challenges store could not be written: {exc}", "outputs": snapshot,
                "extra": counts}
    return {
        "status": ledger.STATUS_OK,
        "model": answered,
        "reason": (
            f"{counts['offered']} challenge(s) offered, {counts['stored']} stored, "
            f"{counts['dropped']} dropped, {counts['repeats']} seen before for {session}"
        ),
        "outputs": [str(path), *snapshot],
        "extra": counts,
    }


__all__ = [
    "CHALLENGES_JSON_SCHEMA",
    "MAX_CHALLENGES",
    "PROMPT_VERSION",
    "PlanReviewRejected",
    "RESERVE_MINUTES",
    "build_evidence",
    "build_inputs",
    "check_challenges",
    "drop_reason",
    "run_plan_review",
    "usable_challenges",
]
