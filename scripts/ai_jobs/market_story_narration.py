"""Grounded local-AI narration of deterministic Market Journal rollups.

The model sees the latest weekly, monthly and quarterly packs and nothing
else.  A failed call leaves the last verified narration untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

PROMPT_VERSION = "market_story_narration_v1"
SCHEMA = "market_story_narration_v1"

NARRATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "changes", "open_questions", "mentor_question", "sources"],
    "properties": {
        "summary": {"type": "string", "maxLength": 1800},
        "changes": {
            "type": "array",
            "maxItems": 6,
            "items": {"type": "string", "maxLength": 400},
        },
        "open_questions": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string", "maxLength": 400},
        },
        "mentor_question": {"type": "string", "maxLength": 300},
        "sources": {
            "type": "array",
            "maxItems": 12,
            "items": {"type": "string", "maxLength": 160},
        },
    },
}


def _roots(rollups_dir: Path | None, out_dir: Path | None) -> tuple[Path, Path]:
    if rollups_dir is None or out_dir is None:
        from project_paths import MARKET_STORY_NARRATIONS_DIR, MARKET_STORY_ROLLUPS_DIR

        rollups_dir = Path(rollups_dir or MARKET_STORY_ROLLUPS_DIR)
        out_dir = Path(out_dir or MARKET_STORY_NARRATIONS_DIR)
    return Path(rollups_dir), Path(out_dir)


def _latest_pack(root: Path, kind: str) -> dict[str, Any] | None:
    paths = sorted((root / kind).glob("*.json"))
    for path in reversed(paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _source_ids(packs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    ids: set[str] = set()
    for kind, pack in packs.items():
        ids.add(f"rollup:{kind}:{pack.get('period_id')}")
        for session in pack.get("sessions") or ():
            if not isinstance(session, Mapping):
                continue
            for entry in session.get("entries") or ():
                if isinstance(entry, Mapping) and str(entry.get("entry_id") or "").strip():
                    ids.add("journal:" + str(entry["entry_id"]).strip())
    return sorted(ids)


def _evidence(packs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    canonical = json.dumps(packs, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    source_ids = _source_ids(packs)
    return {
        "package_id": f"market-story:{digest[:16]}",
        "evidence_hash": digest,
        "instructions": (
            "Narrate only these deterministic packs. Do not calculate new statistics or "
            "turn an unmeasured item into a fact. Distinguish the trader's words from measured "
            "market facts. Name changes across covered sessions, keep uncertainty, and ask one "
            "short coaching question that tests an open thesis. Every source must be copied "
            "exactly from allowed_source_ids."
        ),
        "allowed_source_ids": source_ids,
        "rollups": {kind: dict(pack) for kind, pack in packs.items()},
    }


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_market_story_narration(
    *,
    session_date: str = "",
    now: datetime | None = None,
    rollups_dir: Path | None = None,
    out_dir: Path | None = None,
    request: Callable[..., Mapping[str, Any]] | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Write one verified narration; model failure preserves the prior file."""
    root, target = _roots(rollups_dir, out_dir)
    packs = {
        kind: pack
        for kind in ("weekly", "monthly", "quarterly")
        if (pack := _latest_pack(root, kind)) is not None
    }
    if not packs:
        return {
            "status": "skipped",
            "model": "",
            "reason": "no market-story rollups exist yet",
            "outputs": [],
        }
    evidence = _evidence(packs)
    day = str(session_date or datetime.now().date().isoformat())
    destination = target / f"{day}.json"
    try:
        existing = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        existing = {}
    if (
        isinstance(existing, Mapping)
        and existing.get("inputs_hash") == evidence["evidence_hash"]
        and existing.get("prompt_version") == PROMPT_VERSION
    ):
        return {
            "status": "ok",
            "model": str(existing.get("model") or ""),
            "reason": "verified narration unchanged",
            "outputs": [str(destination)],
        }

    import ai_summary

    if request is None:
        if not ai_summary.local_provider_enabled():
            return {
                "status": "degraded_no_narrative",
                "model": "",
                "reason": "local AI is not configured; prior narration was kept",
                "outputs": [],
            }
        request = ai_summary.request_ai_summary
    try:
        result = request(
            provider="local",
            model=ai_summary.local_model("medium"),
            api_key="",
            evidence=evidence,
            timeout_seconds=900,
            schema=NARRATION_JSON_SCHEMA,
            schema_name="tradingbot_market_story_narration",
            prompt_version=PROMPT_VERSION,
        )
        narration = result.get("summary") if isinstance(result, Mapping) else None
        if not isinstance(narration, Mapping):
            raise ValueError("local AI returned no narration")
        allowed = set(evidence["allowed_source_ids"])
        cited = [str(item) for item in narration.get("sources") or ()]
        if not cited or any(source not in allowed for source in cited):
            raise ValueError("narration cited a source outside its fact packs")
        moment = now or datetime.now(timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        payload = {
            "schema": SCHEMA,
            "session_date": day,
            "generated_at": moment.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "inputs_hash": evidence["evidence_hash"],
            "periods": {kind: str(pack.get("period_id") or "") for kind, pack in packs.items()},
            "model": str(result.get("model") or ""),
            "prompt_version": PROMPT_VERSION,
            "narration": dict(narration),
        }
        _atomic_write(destination, payload)
    except Exception as exc:  # noqa: BLE001 - prior verified file is the fallback
        return {
            "status": "degraded_no_narrative",
            "model": "",
            "reason": f"local AI narration failed; prior narration was kept: {exc}",
            "outputs": [],
        }
    return {
        "status": "ok",
        "model": str(result.get("model") or ""),
        "reason": "grounded market-story narration written",
        "outputs": [str(destination)],
    }


def latest_coaching_question(out_dir: Path | None = None) -> str:
    """The last verified nightly question, for the existing Mentor card."""
    _root, target = _roots(None, out_dir)
    for path in reversed(sorted(target.glob("*.json"))):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        narration = payload.get("narration") if isinstance(payload, Mapping) else None
        if isinstance(narration, Mapping):
            question = str(narration.get("mentor_question") or "").strip()
            if question:
                return question
    return ""
