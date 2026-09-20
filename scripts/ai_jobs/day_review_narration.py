"""The overnight day story, and the rolling D1 view (TJ-4 items 2 and 3).

`plan.md` §12.4 TJ-4 changes 2 and 3 as amended 2026-09-19; decision 0021
answer 14. The pattern is `market_story_narration.py` beside it: a closed
schema, an allowed-`source_id` grounding rule, an `inputs_hash` skip, and a
failure that leaves the last verified file byte-identical.

**The model narrates verdicts; it never makes them.** Every
`were_you_right[].verdict` must EQUAL the verdict of the TJ-10 read row its
`evidence_id` names. An output that disagrees with a measured row, grades a
claim no read row carries, calls an OBSERVATION a call, or cites an id the pack
does not carry is rejected WHOLE - not trimmed, not partially kept - and the
last verified story stays exactly as it was. Only a `prediction` may be called a
call; an `observation` is quoted as what the trader saw.

Two artifacts, two verdicts, and neither may destroy the other's last good file:

* `<DAY_REVIEW_DIR>/narration/<date>.json` - one day, narrated from that day's
  pack;
* `<DAY_REVIEW_DIR>/d1_view.json` - ONE rolling file, built from the trader's D1
  prediction clicks and D1 notes of the last `evidence_stats.LATELY_SESSIONS`
  exchange sessions. The thesis store is empty and stays so (packet TJ-4 item
  3): the rolling view is what the trader actually said on a D1 card, or it is
  nothing at all. Its own `inputs_hash` means an unchanged D1 week costs no
  second call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

_log = logging.getLogger(__name__)

PROMPT_VERSION = "day_review_narration_v1"
SCHEMA = "day_review_narration_v1"

D1_VIEW_PROMPT_VERSION = "d1_view_narration_v1"
D1_VIEW_SCHEMA = "d1_view_narration_v1"

#: The local tier this slot asks for - the same one the market-story narration
#: uses. A day story is a page of text, not a research pass.
MODEL_TIER = "medium"

#: How long one call may take. The slot reserves ten minutes (gate #158 wants
#: the day story finished before 23:30 Pacific), and the call itself is bounded
#: well inside that.
TIMEOUT_SECONDS = 540

#: Bounds are deliberately NOT 2,000 anywhere in either schema: a `maxLength` of
#: exactly 2,000 is the grammar defect gate #144 found, where the constrained
#: decoder silently truncated mid-sentence.
NARRATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "headline",
        "what_happened",
        "what_you_thought",
        "were_you_right",
        "chased_against_news",
        "process",
        "sources",
    ],
    "properties": {
        "headline": {"type": "string", "maxLength": 160},
        "what_happened": {"type": "string", "maxLength": 1200},
        "what_you_thought": {"type": "string", "maxLength": 600},
        "were_you_right": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "source_id", "verdict", "evidence_id"],
                "properties": {
                    "claim": {"type": "string", "maxLength": 240},
                    "source_id": {"type": "string", "maxLength": 160},
                    # NOT an enum: a measured verdict may be
                    # `unmeasured:<reason>`, and the rule that matters is
                    # equality with the read row, enforced below.
                    "verdict": {"type": "string", "maxLength": 80},
                    "evidence_id": {"type": "string", "maxLength": 160},
                },
            },
        },
        "chased_against_news": {
            "type": "object",
            "additionalProperties": False,
            "required": ["verdict", "evidence_id"],
            "properties": {
                "verdict": {"type": "string", "enum": ["yes", "no", "unknown"]},
                "evidence_id": {"type": "string", "maxLength": 160},
            },
        },
        "process": {"type": "string", "maxLength": 400},
        "sources": {
            "type": "array",
            "maxItems": 24,
            "items": {"type": "string", "maxLength": 160},
        },
    },
}

D1_VIEW_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["belief_now", "open_theses", "sources"],
    "properties": {
        "belief_now": {"type": "string", "maxLength": 600},
        "open_theses": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "since", "still_true", "evidence_id"],
                "properties": {
                    "claim": {"type": "string", "maxLength": 240},
                    "since": {"type": "string", "maxLength": 40},
                    "still_true": {"type": "string", "enum": ["yes", "no", "unknown"]},
                    "evidence_id": {"type": "string", "maxLength": 160},
                },
            },
        },
        "sources": {
            "type": "array",
            "maxItems": 24,
            "items": {"type": "string", "maxLength": 160},
        },
    },
}

DAY_INSTRUCTIONS = (
    "Narrate this ONE session from the pack below and nothing else. You may not "
    "calculate a statistic, grade a call, or turn an unmeasured item into a "
    "fact. Every verdict in were_you_right must be COPIED from the reads item "
    "its evidence_id names - if you disagree with a measured verdict, say "
    "nothing rather than changing it. Only a trader_said item whose kind is "
    "'prediction' may be graded as a call; an item whose kind is 'observation' "
    "is quoted as what the trader SAW. chased_against_news is 'unknown' unless "
    "the forecast section states the condition you are judging. Every id in "
    "sources must be copied exactly from allowed_source_ids."
)

D1_VIEW_INSTRUCTIONS = (
    "These are the trader's own D1 calls and D1 notes over the last twenty "
    "exchange sessions, oldest first. Say what they appear to believe about the "
    "bigger picture NOW, and list the theses still open. You may not measure "
    "anything: still_true is 'unknown' unless one of these items itself says "
    "otherwise. Every id you cite must be copied exactly from "
    "allowed_source_ids."
)


class NarrationRejected(ValueError):
    """The model's answer was not a narration of the evidence it was given."""


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
def _root(root: Path | None) -> Path:
    import day_review_pack

    return Path(root) if root is not None else day_review_pack.default_root()


def narration_path(session_date: str, *, root: Path | None = None) -> Path:
    """`<root>/narration/<date>.json` - ONE verified story per session."""
    return _root(root) / "narration" / f"{str(session_date or '').strip()[:10]}.json"


def read_narration(session_date: str, *, root: Path | None = None) -> dict[str, Any] | None:
    return _read_json(narration_path(session_date, root=root))


def d1_view_path(*, root: Path | None = None) -> Path:
    """`<root>/d1_view.json` - ONE rolling file, not one per day."""
    return _root(root) / "d1_view.json"


def read_d1_view(*, root: Path | None = None) -> dict[str, Any] | None:
    return _read_json(d1_view_path(root=root))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _moment(now: datetime | None) -> str:
    stamp = now or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _validate(payload: Any, schema: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    """The closed schema, top level and one level into every declared object."""
    import ai_summary

    body = ai_summary.validate_structured_output(payload, schema, name=name)
    for key, spec in (schema.get("properties") or {}).items():
        if key not in body:
            continue
        if str(spec.get("type") or "") == "object":
            ai_summary.validate_structured_output(body[key], spec, name=f"{name}.{key}")
        elif str(spec.get("type") or "") == "array":
            item_spec = spec.get("items") or {}
            if str(item_spec.get("type") or "") != "object":
                continue
            for index, item in enumerate(body[key] or ()):
                ai_summary.validate_structured_output(
                    item, item_spec, name=f"{name}.{key}[{index}]"
                )
    return body


def _check_sources(narration: Mapping[str, Any], allowed: set[str]) -> None:
    cited = [str(item) for item in narration.get("sources") or ()]
    if not cited:
        raise NarrationRejected("the narration cited nothing at all")
    outside = [item for item in cited if item not in allowed]
    if outside:
        raise NarrationRejected(
            f"the narration cited id(s) the pack does not carry: {', '.join(sorted(outside))}"
        )


def _check_day_narration(narration: Mapping[str, Any], pack: Mapping[str, Any]) -> None:
    """The rule of this packet: a narrated verdict IS the measured verdict."""
    import day_review_pack

    allowed = set(day_review_pack.allowed_source_ids(pack))
    _check_sources(narration, allowed)

    reads = {
        str(row.get("source_id") or ""): row
        for row in pack.get("reads") or ()
        if isinstance(row, Mapping)
    }
    said = {
        str(item.get("source_id") or ""): item
        for item in pack.get("trader_said") or ()
        if isinstance(item, Mapping)
    }
    for claim in narration.get("were_you_right") or ():
        if not isinstance(claim, Mapping):
            raise NarrationRejected("a graded claim was not an object")
        evidence_id = str(claim.get("evidence_id") or "")
        row = reads.get(evidence_id)
        if row is None:
            raise NarrationRejected(
                f"the narration graded a claim no read row carries: {evidence_id!r}"
            )
        measured = str(row.get("verdict") or "")
        stated = str(claim.get("verdict") or "")
        if stated != measured:
            raise NarrationRejected(
                f"the narration said {stated!r} where the measured row says {measured!r}"
            )
        source_id = str(claim.get("source_id") or "")
        item = said.get(source_id)
        if item is None:
            raise NarrationRejected(
                f"a graded claim named {source_id!r}, which is not something the trader said"
            )
        if str(item.get("kind") or "") != day_review_pack.KIND_PREDICTION:
            raise NarrationRejected(
                "the narration graded an observation as a call; only a prediction "
                "is a call"
            )

    chased = narration.get("chased_against_news")
    if not isinstance(chased, Mapping):
        raise NarrationRejected("chased_against_news was not an object")
    verdict = str(chased.get("verdict") or "")
    if not pack.get("forecast") and verdict != "unknown":
        raise NarrationRejected(
            "the pack carries no pasted forecast, so chased_against_news can only "
            f"be 'unknown'; the narration said {verdict!r}"
        )
    evidence_id = str(chased.get("evidence_id") or "")
    if evidence_id and evidence_id not in allowed:
        raise NarrationRejected(
            f"chased_against_news cited an id the pack does not carry: {evidence_id!r}"
        )


def _check_d1_narration(narration: Mapping[str, Any], allowed: set[str]) -> None:
    _check_sources(narration, allowed)
    for thesis in narration.get("open_theses") or ():
        if not isinstance(thesis, Mapping):
            raise NarrationRejected("an open thesis was not an object")
        evidence_id = str(thesis.get("evidence_id") or "")
        if evidence_id not in allowed:
            raise NarrationRejected(
                f"a standing thesis cited {evidence_id!r}, which is not a D1 thing "
                "the trader said inside the window"
            )


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------
def _previous_story(session: str, root: Path) -> dict[str, Any]:
    """Yesterday's narration, READ-ONLY context. Never rewritten, never cited."""
    try:
        import market_calendar

        yesterday = market_calendar.previous_session(
            date.fromisoformat(str(session)[:10])
        ).isoformat()
    except Exception:  # noqa: BLE001 - no yesterday is simply less context
        return {}
    stored = read_narration(yesterday, root=root)
    if not isinstance(stored, Mapping):
        return {}
    narration = stored.get("narration")
    return {
        "session_date": yesterday,
        "narration": dict(narration) if isinstance(narration, Mapping) else {},
    }


def _day_evidence(pack: Mapping[str, Any], root: Path) -> dict[str, Any]:
    import day_review_pack

    session = str(pack.get("session_date") or "")
    allowed = list(day_review_pack.allowed_source_ids(pack))
    return {
        "package_id": f"day-review:{str(pack.get('inputs_hash') or '')[:16]}",
        "evidence_hash": str(pack.get("inputs_hash") or ""),
        "instructions": DAY_INSTRUCTIONS,
        "allowed_source_ids": allowed,
        "session_date": session,
        "pack": {name: pack.get(name) for name in day_review_pack.SECTIONS},
        # Read-only, and deliberately outside `allowed_source_ids`: last night's
        # story is context for continuity, never a fact this night may cite.
        "previous_day": _previous_story(session, root),
    }


def _d1_window(session: str) -> list[str]:
    """The last `LATELY_SESSIONS` exchange sessions ending at `session`."""
    import evidence_stats
    import market_calendar

    cursor = date.fromisoformat(str(session)[:10])
    days = [cursor.isoformat()]
    while len(days) < int(evidence_stats.LATELY_SESSIONS):
        cursor = market_calendar.previous_session(cursor)
        days.append(cursor.isoformat())
    return list(reversed(days))


def _d1_items(session: str, root: Path) -> list[dict[str, Any]]:
    """The trader's D1 clicks and D1 notes across the window, oldest first.

    Never an M5 item - a rest-of-day read is a read about the tape, not a belief
    about the bigger picture - and never a session outside the window.
    """
    import day_review_pack

    items: list[dict[str, Any]] = []
    for day in _d1_window(session):
        pack = day_review_pack.read_pack(day, root=root)
        if not isinstance(pack, Mapping):
            continue
        for item in day_review_pack.said_items(pack, timeframe="D1"):
            items.append({**item, "session_date": day})
    return items


def _d1_evidence(session: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256(
        json.dumps(items, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return {
        "package_id": f"d1-view:{digest[:16]}",
        "evidence_hash": digest,
        "instructions": D1_VIEW_INSTRUCTIONS,
        "allowed_source_ids": [str(item.get("source_id") or "") for item in items],
        "session_date": session,
        "window_sessions": sorted({str(item.get("session_date") or "") for item in items}),
        "d1_said": items,
    }


# ---------------------------------------------------------------------------
# the slot
# ---------------------------------------------------------------------------
def _request_for(request: Callable[..., Mapping[str, Any]] | None):
    """The injected seam, or the local provider when there is one."""
    import ai_summary

    if request is not None:
        return request, ""
    if not ai_summary.local_provider_enabled():
        return None, "local AI is not configured; the prior narration was kept"
    return ai_summary.request_ai_summary, ""


def _call(request, *, evidence: Mapping[str, Any], schema, prompt_version: str, schema_name: str):
    import ai_summary

    return request(
        provider="local",
        model=ai_summary.local_model(MODEL_TIER),
        api_key="",
        evidence=dict(evidence),
        timeout_seconds=TIMEOUT_SECONDS,
        schema=schema,
        schema_name=schema_name,
        prompt_version=prompt_version,
    )


def run_day_review_narration(
    *,
    session_date: str = "",
    now: datetime | None = None,
    root: Path | None = None,
    request: Callable[..., Mapping[str, Any]] | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Narrate one session, and refresh the rolling D1 view beside it.

    Never raises: a crash here is a lost night. Every failure path leaves the
    last verified file byte-identical and says what happened.
    """
    import day_review_pack

    base = _root(root)
    session = str(session_date or "").strip()[:10] or datetime.now().date().isoformat()
    pack = day_review_pack.read_pack(session, root=base)
    if pack is None:
        # The post-close tick never reached this session. That is a `skipped`
        # row with a reason, not a failure and not an invented story - an
        # evidence job is never allowed to cost the night.
        return {
            "status": "skipped",
            "model": "",
            "reason": f"no day pack for {session}; nothing to narrate",
            "outputs": [],
        }

    redo = day_review_pack.redo_requested(session, root=base)
    outputs: list[str] = []
    reasons: list[str] = []
    model = ""
    degraded = False

    story = _run_day_story(
        session, pack, base, now=now, request=request, redo=redo
    )
    outputs.extend(story["outputs"])
    reasons.append(story["reason"])
    model = model or story["model"]
    degraded = degraded or story["status"] == "degraded_no_narrative"
    if story["status"] == "ok" and redo:
        day_review_pack.clear_redo(session, root=base)

    view = _run_d1_view(session, base, now=now, request=request, redo=redo)
    outputs.extend(view["outputs"])
    if view["reason"]:
        reasons.append(view["reason"])
    model = model or view["model"]
    degraded = degraded or view["status"] == "degraded_no_narrative"

    return {
        "status": "degraded_no_narrative" if degraded else "ok",
        "model": model,
        "reason": "; ".join(part for part in reasons if part),
        "outputs": outputs,
    }


def _run_day_story(
    session: str,
    pack: Mapping[str, Any],
    root: Path,
    *,
    now: datetime | None,
    request: Callable[..., Mapping[str, Any]] | None,
    redo: bool,
) -> dict[str, Any]:
    destination = narration_path(session, root=root)
    existing = _read_json(destination) or {}
    if (
        not redo
        and existing.get("inputs_hash") == pack.get("inputs_hash")
        and existing.get("prompt_version") == PROMPT_VERSION
    ):
        return {
            "status": "ok",
            "model": str(existing.get("model") or ""),
            "reason": "the verified day story is unchanged for this session",
            "outputs": [str(destination)],
        }

    caller, refusal = _request_for(request)
    if caller is None:
        return {
            "status": "degraded_no_narrative",
            "model": "",
            "reason": refusal,
            "outputs": [],
        }
    try:
        result = _call(
            caller,
            evidence=_day_evidence(pack, root),
            schema=NARRATION_JSON_SCHEMA,
            prompt_version=PROMPT_VERSION,
            schema_name="tradingbot_day_review_narration",
        )
        narration = result.get("summary") if isinstance(result, Mapping) else None
        narration = _validate(narration, NARRATION_JSON_SCHEMA, name="day story")
        _check_day_narration(narration, pack)
        payload = {
            "schema": SCHEMA,
            "session_date": session,
            "generated_at": _moment(now),
            "inputs_hash": str(pack.get("inputs_hash") or ""),
            "prompt_version": PROMPT_VERSION,
            "model": str(result.get("model") or ""),
            "narration": dict(narration),
        }
        _atomic_write(destination, payload)
    except Exception as exc:  # noqa: BLE001 - the prior verified file is the fallback
        _log.debug("The day story was not written.", exc_info=True)
        return {
            "status": "degraded_no_narrative",
            "model": "",
            "reason": f"the day story was rejected; the prior story was kept: {exc}",
            "outputs": [],
        }
    return {
        "status": "ok",
        "model": str(result.get("model") or ""),
        "reason": f"grounded day story written for {session}",
        "outputs": [str(destination)],
    }


def _run_d1_view(
    session: str,
    root: Path,
    *,
    now: datetime | None,
    request: Callable[..., Mapping[str, Any]] | None,
    redo: bool,
) -> dict[str, Any]:
    try:
        items = _d1_items(session, root)
    except Exception as exc:  # noqa: BLE001 - an unanswerable calendar names no window
        return {
            "status": "skipped",
            "model": "",
            "reason": f"the rolling D1 window could not be walked: {exc}",
            "outputs": [],
        }
    if not items:
        return {
            "status": "skipped",
            "model": "",
            "reason": "",
            "outputs": [],
        }
    evidence = _d1_evidence(session, items)
    destination = d1_view_path(root=root)
    existing = _read_json(destination) or {}
    if (
        not redo
        and existing.get("inputs_hash") == evidence["evidence_hash"]
        and existing.get("prompt_version") == D1_VIEW_PROMPT_VERSION
    ):
        return {
            "status": "ok",
            "model": str(existing.get("model") or ""),
            "reason": "",
            "outputs": [str(destination)],
        }

    caller, refusal = _request_for(request)
    if caller is None:
        return {
            "status": "degraded_no_narrative",
            "model": "",
            "reason": refusal,
            "outputs": [],
        }
    try:
        result = _call(
            caller,
            evidence=evidence,
            schema=D1_VIEW_JSON_SCHEMA,
            prompt_version=D1_VIEW_PROMPT_VERSION,
            schema_name="tradingbot_d1_view_narration",
        )
        narration = result.get("summary") if isinstance(result, Mapping) else None
        narration = _validate(narration, D1_VIEW_JSON_SCHEMA, name="D1 view")
        _check_d1_narration(narration, set(evidence["allowed_source_ids"]))
        payload = {
            "schema": D1_VIEW_SCHEMA,
            "session_date": session,
            "generated_at": _moment(now),
            "inputs_hash": evidence["evidence_hash"],
            "prompt_version": D1_VIEW_PROMPT_VERSION,
            "model": str(result.get("model") or ""),
            "narration": dict(narration),
        }
        _atomic_write(destination, payload)
    except Exception as exc:  # noqa: BLE001 - the prior rolling view is the fallback
        _log.debug("The rolling D1 view was not written.", exc_info=True)
        return {
            "status": "degraded_no_narrative",
            "model": "",
            "reason": f"the rolling D1 view was rejected; the prior view was kept: {exc}",
            "outputs": [],
        }
    return {
        "status": "ok",
        "model": str(result.get("model") or ""),
        "reason": "rolling D1 view refreshed",
        "outputs": [str(destination)],
    }


__all__ = [
    "D1_VIEW_JSON_SCHEMA",
    "D1_VIEW_PROMPT_VERSION",
    "D1_VIEW_SCHEMA",
    "NARRATION_JSON_SCHEMA",
    "PROMPT_VERSION",
    "SCHEMA",
    "d1_view_path",
    "narration_path",
    "read_d1_view",
    "read_narration",
    "run_day_review_narration",
]
