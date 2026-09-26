"""Night slot `day_review_show`: the local model cooks the Day Review Show (R1).

Runs directly after `day_review_narration`. It reads the session's day pack and
that night's verified narration (only when it was written for this pack), asks
the medium model for a 6-10 slide deck, and keeps it only if
`day_review_show.verify_show` passes. A rejected deck is `degraded`: the last
good file stays byte-identical and the desk shows the facts-only fallback.

Output: `<DAY_REVIEW_DIR>/shows/<date>.json` with model, prompt_version and
inputs_hash (pack hash + narration words), so an unchanged night costs no call.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

_log = logging.getLogger(__name__)

PROMPT_VERSION = "day_review_show_v1"
MODEL_TIER = "medium"
#: The slot reserves five minutes; one call ends inside them.
TIMEOUT_SECONDS = 290
#: Evidence ceiling in JSON characters (~7.6k tokens, about a minute of prompt).
MAX_EVIDENCE_CHARS = 16_000
#: The narration fields the model may read as context (never as a source).
NARRATION_FIELDS = ("headline", "what_happened", "what_you_thought", "process")


def _narration_for(session: str, pack: Mapping[str, Any], root: Path) -> dict[str, Any] | None:
    """Tonight's verified story, only when it was written for this very pack."""
    from ai_jobs import day_review_narration

    stored = day_review_narration.read_narration(session, root=root)
    if not isinstance(stored, Mapping):
        return None
    if str(stored.get("inputs_hash") or "") != str(pack.get("inputs_hash") or ""):
        return None
    body = stored.get("narration")
    if not isinstance(body, Mapping):
        return None
    return {key: body.get(key) for key in NARRATION_FIELDS if key in body}


def _evidence(pack: Mapping[str, Any], narration: Mapping[str, Any] | None, digest: str):
    """The model's view of the pack, trimmed to fit; returns (evidence, dropped ids)."""
    import day_review_pack
    import day_review_show
    from ai_jobs import day_review_narration as story

    allowed = list(day_review_pack.allowed_source_ids(pack))
    evidence: dict[str, Any] = {
        "package_id": f"day-review-show:{digest[:16]}",
        "evidence_hash": digest,
        "instructions": day_review_show.INSTRUCTIONS,
        "allowed_source_ids": allowed,
        "session_date": str(pack.get("session_date") or ""),
        "pack": story._model_pack(pack),
        "previous_story": dict(narration or {}),
    }
    trimmed: list[str] = []
    for part in story.DAY_TRIM_ORDER:
        if _chars(evidence) <= MAX_EVIDENCE_CHARS:
            break
        if story._trim_part(evidence["pack"], part):
            trimmed.append(part)
    dropped = story._dropped_source_ids(pack, trimmed)
    if trimmed:
        evidence["pack_trimmed"] = trimmed
        evidence["allowed_source_ids"] = [item for item in allowed if item not in dropped]
    if _chars(evidence) > MAX_EVIDENCE_CHARS:
        raise ValueError(
            f"the show evidence is {_chars(evidence)} characters after trimming; "
            f"at most {MAX_EVIDENCE_CHARS} fit the {TIMEOUT_SECONDS}s call"
        )
    return evidence, dropped


def _chars(evidence: Mapping[str, Any]) -> int:
    return len(json.dumps(evidence, sort_keys=True, default=str))


def _result(status: str, reason: str, *, model: str = "", outputs=()) -> dict[str, Any]:
    return {"status": status, "model": model, "reason": reason, "outputs": list(outputs)}


def run_day_review_show(
    *,
    session_date: str = "",
    now: datetime | None = None,
    root: Path | None = None,
    request: Callable[..., Mapping[str, Any]] | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Write one verified show for the session, or keep the last good one. Never raises."""
    import day_review_pack
    import day_review_show
    from ai_jobs import day_review_narration as story
    from ai_jobs.ledger import STATUS_DEGRADED

    base = Path(root) if root is not None else day_review_pack.default_root()
    session = str(session_date or "").strip()[:10] or datetime.now().date().isoformat()
    pack = day_review_pack.read_pack(session, root=base)
    if pack is None:
        return _result("skipped", f"no day pack for {session}; the desk shows facts only")
    try:
        narration = _narration_for(session, pack, base)
    except Exception:  # noqa: BLE001 - no story is less context, not a failure
        _log.debug("The night's story was unreadable for the show.", exc_info=True)
        narration = None
    digest = day_review_show.inputs_hash(pack, narration)
    destination = day_review_show.show_path(session, root=base)
    existing = story._read_json(destination) or {}
    if existing.get("inputs_hash") == digest and existing.get("prompt_version") == PROMPT_VERSION:
        return _result(
            "ok", "the verified show is unchanged for this session",
            model=str(existing.get("model") or ""), outputs=[str(destination)],
        )
    caller, refusal = story._request_for(request)
    if caller is None:
        return _result(STATUS_DEGRADED, refusal.replace("narration", "show"))
    try:
        evidence, dropped = _evidence(pack, narration, digest)
        result = story._call(
            caller,
            evidence=evidence,
            schema=day_review_show.MODEL_JSON_SCHEMA,
            prompt_version=PROMPT_VERSION,
            schema_name="tradingbot_day_review_show",
        )
        reply = result.get("summary") if isinstance(result, Mapping) else None
        deck = day_review_show.verify_show(reply, pack)
        cited = {item for slide in deck["slides"] for item in slide["source_ids"]}
        hidden = sorted(cited & set(dropped))
        if hidden:
            raise day_review_show.ShowRejected(
                f"the show cited id(s) dropped to fit the call: {', '.join(hidden)}"
            )
        model = str(result.get("model") or "")
        story._atomic_write(destination, {
            "schema": day_review_show.SCHEMA,
            "session_date": session,
            "generated_at": story._moment(now),
            "inputs_hash": digest,
            "pack_hash": str(pack.get("inputs_hash") or ""),
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "narration_read": narration is not None,
            "show": deck,
        })
    except Exception as exc:  # noqa: BLE001 - the last good show is the fallback
        _log.debug("The day show was not written.", exc_info=True)
        return _result(STATUS_DEGRADED, f"the show was rejected; the prior show was kept: {exc}")
    context = "" if narration is not None else " (no current story to read)"
    return _result(
        "ok", f"verified show written for {session}{context}",
        model=model, outputs=[str(destination)],
    )


__all__ = ["PROMPT_VERSION", "run_day_review_show"]
