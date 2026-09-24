"""Night slot: a short "what to watch today" from the newest pasted brief.

The pack is deterministic (`econ_brief.build_pack`: the next session's timed
events, the week's, the bottom line, ranked signals, turbulence, playbook).
The local model only words 3-6 short lines over it. A reply is kept only if
every event it cites is in the pack and every clock time it writes is the
time of an event it cites - a model never supplies a time. A failed or
invalid reply writes nothing: the last good file stays and the Mentor falls
back to the brief's own lines.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

PROMPT_VERSION = "econ_brief_narration_v1"

MIN_LINES = 3
MAX_LINES = 6
LINE_MAX_CHARS = 160

NARRATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["lines"],
    "properties": {
        "lines": {
            "type": "array",
            "minItems": MIN_LINES,
            "maxItems": MAX_LINES,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "event_ids"],
                "properties": {
                    "text": {"type": "string", "maxLength": LINE_MAX_CHARS},
                    "event_ids": {
                        "type": "array",
                        "maxItems": 4,
                        "items": {"type": "string", "maxLength": 8},
                    },
                },
            },
        },
    },
}

_CLOCK = re.compile(
    r"(?<![\d.$])(\d{1,2})(?::([0-5]\d))?\s*(a\.?m\.?|p\.?m\.?)(?![a-z])|(?<![\d.$])(\d{1,2}):([0-5]\d)(?![\d%])",
    re.IGNORECASE,
)


def _target_session(session_date: str) -> str:
    """The session the morning summary is for: the one after the night's."""
    from market_calendar import next_session

    return next_session(date.fromisoformat(session_date)).isoformat()


def _clock_times(text: str) -> list[tuple[int, int, bool]]:
    """(hour, minute, meridiem_known) for every clock time the text writes."""
    out = []
    for match in _CLOCK.finditer(text):
        if match.group(1) is not None:
            hour, minute = int(match.group(1)), int(match.group(2) or 0)
            pm = match.group(3).lower().startswith("p")
            out.append(((hour % 12) + (12 if pm else 0), minute, True))
        else:
            out.append((int(match.group(4)), int(match.group(5)), False))
    return out


def _time_matches(written: tuple[int, int, bool], time_et: str) -> bool:
    if not time_et:
        return False
    hour, minute = int(time_et[:2]), int(time_et[3:])
    if written[2]:
        return (written[0], written[1]) == (hour, minute)
    # "10:00" or "13:00": either the 24h clock or the 12h face of the event's time.
    return written[1] == minute and written[0] in {hour, hour % 12 or 12}


def validate(narration: Any, pack: Mapping[str, Any]) -> list[str]:
    """The lines, or raise ValueError naming what was not grounded."""
    import econ_events

    if not isinstance(narration, Mapping):
        raise ValueError("no narration object")
    lines = narration.get("lines")
    if not isinstance(lines, list) or not MIN_LINES <= len(lines) <= MAX_LINES:
        raise ValueError(f"expected {MIN_LINES}-{MAX_LINES} lines")
    events = {row["id"]: row for row in list(pack.get("today") or ()) + list(pack.get("week") or ())}
    pack_kinds = {str(row.get("kind") or "") for row in events.values()}
    out: list[str] = []
    for line in lines:
        if not isinstance(line, Mapping):
            raise ValueError("a line is not an object")
        text = str(line.get("text") or "").strip()
        if not text or len(text) > LINE_MAX_CHARS:
            raise ValueError("a line is empty or too long")
        cited = [str(item) for item in line.get("event_ids") or ()]
        unknown = [item for item in cited if item not in events]
        if unknown:
            raise ValueError(f"cited events outside the pack: {unknown}")
        for written in _clock_times(text):
            if not any(_time_matches(written, events[item]["time_et"]) for item in cited):
                raise ValueError(f"a time in {text!r} is not the time of an event it cites")
        # A release the line names must be one the pack holds.
        for _start, _end, kind, _label in econ_events._hits(econ_events._protect(text)):
            if kind not in pack_kinds:
                raise ValueError(f"{text!r} names a release that is not in the brief ({kind})")
        out.append(text)
    return out


def _evidence(pack: Mapping[str, Any]) -> dict[str, Any]:
    canonical = json.dumps(pack, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "package_id": f"econ-brief:{digest[:16]}",
        "evidence_hash": digest,
        "instructions": (
            "Write 3 to 6 short lines, plain simple words, telling a day trader what to "
            "watch today. Use only this pack. Name only events in `today` or `week`, and "
            "put the id of every event a line names in its event_ids. Write a clock time "
            "only if it is that cited event's time_et (ET). Never add a time, number or "
            "event the pack does not hold. Lead with today's timed events."
        ),
        "pack": dict(pack),
    }


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_econ_brief(
    *,
    session_date: str = "",
    now: datetime | None = None,
    out_dir: Path | None = None,
    forecasts: list[Mapping[str, str]] | None = None,
    request: Callable[..., Mapping[str, Any]] | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Write one verified summary for the next session; failure keeps the old file."""
    import econ_brief

    day = str(session_date or datetime.now().date().isoformat())
    try:
        target = _target_session(day)
    except Exception as exc:  # noqa: BLE001 - outside the calendar: nothing to plan
        return {"status": "skipped", "model": "", "reason": f"no next session: {exc}", "outputs": []}
    if forecasts is None:
        forecasts = econ_brief.load_recent_forecasts(up_to=day)
    pack = econ_brief.build_pack(list(forecasts or ()), target_session=target) if forecasts else {}
    if not pack:
        return {
            "status": "skipped",
            "model": "",
            "reason": f"{econ_brief.NO_BRIEF_TEXT} Nothing to summarise for {target}.",
            "outputs": [],
        }
    evidence = _evidence(pack)
    destination = econ_brief.night_dir(out_dir) / f"{target}.json"
    existing = econ_brief.read_night(target, out_dir=out_dir)
    if (
        existing is not None
        and existing.get("inputs_hash") == evidence["evidence_hash"]
        and existing.get("prompt_version") == PROMPT_VERSION
    ):
        return {
            "status": "ok",
            "model": str(existing.get("model") or ""),
            "reason": "verified econ summary unchanged",
            "outputs": [str(destination)],
        }

    import ai_summary

    if request is None:
        if not ai_summary.local_provider_enabled():
            return {
                "status": "degraded_no_narrative",
                "model": "",
                "reason": "local AI is not configured; the Mentor shows the brief's own lines",
                "outputs": [],
            }
        request = ai_summary.request_ai_summary
    try:
        result = request(
            provider="local",
            model=ai_summary.local_model("medium"),
            api_key="",
            evidence=evidence,
            timeout_seconds=600,
            schema=NARRATION_JSON_SCHEMA,
            schema_name="tradingbot_econ_brief",
            prompt_version=PROMPT_VERSION,
        )
        narration = result.get("summary") if isinstance(result, Mapping) else None
        lines = validate(narration, pack)
        moment = now or datetime.now(timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        payload = {
            "schema": econ_brief.SCHEMA,
            "target_session": target,
            "brief_session": pack["brief_session"],
            "brief_hash": pack["brief_hash"],
            "generated_at": moment.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "inputs_hash": evidence["evidence_hash"],
            "model": str(result.get("model") or ""),
            "prompt_version": PROMPT_VERSION,
            "summary_lines": lines,
            "pack": pack,
        }
        _atomic_write(destination, payload)
    except Exception as exc:  # noqa: BLE001 - the last verified file is the fallback
        return {
            "status": "degraded_no_narrative",
            "model": "",
            "reason": f"econ summary rejected; the last good file was kept: {exc}",
            "outputs": [],
        }
    return {
        "status": "ok",
        "model": str(result.get("model") or ""),
        "reason": f"econ summary for {target} written",
        "outputs": [str(destination)],
    }
