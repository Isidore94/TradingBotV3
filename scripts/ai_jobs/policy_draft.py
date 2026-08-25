"""Review-policy DRAFTS — LOCAL-AI Phase 4 machinery, gated (packet W6).

Authorized 2026-08-24 (`docs/analysis/OFFLINE_BUILD_AUTHORIZATION_2026-08-24.md`
§2). `docs/REVIEW_LEARNING_LOOP.md` describes the AI step: read the review
scoreboard, write per-(dimension, segment) rules that RANK and ANNOTATE. Three
constraints from that document are restated here because they are the whole
contract:

* **ranks and annotates only** — `priority_delta` is clamped to +/-5 and queue
  ordering is currently gated to annotation-only anyway;
* **no suppression field** — the format deliberately has none, and one must
  never be added;
* **FIFO queue ordering is untouched**, and chart-watch hits stay at the front.

**This writer only ever writes `review_policy_draft.json`.** The live
`review_policy.json` is the trader's to save after reading a draft, and nothing
in this module can resolve it: a test walks the AST for the constant name and
for the filename as a path token.

**Why this one RUNS while its gate is unmet**, unlike the Phase 3 pass beside
it. Phase 4's gate IS the drafts: two weeks of them, compared side by side
against the other model's output, is what the trader signs off on. A writer that
refused until the window passed would make the window unreachable. So it writes,
it archives one copy per session so the comparison has something to compare, and
every draft carries the NOT-MET statement in its own `notes` field until the
window closes.

Medium tier only. The frontier-vs-medium comparison is what the gate decides;
neither side of it is authorized to write the live file.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_log = logging.getLogger(__name__)

DRAFT_SCHEMA = "review_policy_draft_v1"

#: Two weeks of drafts, in sessions. The gate the plan has named since it was
#: written, and it is unchanged by building the machinery.
REQUIRED_SIDE_BY_SIDE_DAYS = 10

GATE_NOT_MET_PREFIX = "POLICY DRAFT WINDOW NOT MET."

GATE_MET_STATEMENT = (
    "Side-by-side window met by count. The trader's quality sign-off is the "
    "other half of this gate and is not answered by a counter. Until they save "
    "it themselves, review_policy.json is unchanged."
)


def _now(value: datetime | None = None) -> datetime:
    moment = value or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


def _archive_dir(root: Path) -> Path:
    return Path(root) / "policy_drafts"


def side_by_side_days(root: Path) -> int:
    """Sessions for which an archived draft exists. Counting is not passing."""
    archive = _archive_dir(root)
    if not archive.is_dir():
        return 0
    days: set[str] = set()
    for path in archive.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        day = str((payload or {}).get("session_date") or "")[:10]
        if day:
            days.add(day)
    return len(days)


def gate_state(root: Path) -> dict[str, Any]:
    days = side_by_side_days(root)
    met = days >= REQUIRED_SIDE_BY_SIDE_DAYS
    return {
        "days_drafted": days,
        "days_required": REQUIRED_SIDE_BY_SIDE_DAYS,
        "window_met": met,
        "statement": GATE_MET_STATEMENT if met else (
            f"{GATE_NOT_MET_PREFIX} {days} of {REQUIRED_SIDE_BY_SIDE_DAYS} drafted "
            "session(s). This file is a DRAFT and is not authoritative: it ranks "
            "and annotates only, it carries no suppression field, and nothing "
            "reads it. review_policy.json is the trader's to save."
        ),
    }


# ---------------------------------------------------------------------------
# the draft
# ---------------------------------------------------------------------------


def _mechanical_rules(state: Mapping[str, Any]):
    """The scoreboard's own callouts, through the one existing translator.

    `review_policy.draft_policy_from_state` already turns a blind spot into a
    boost and a leak into a demote, with the clamp applied. Reimplementing that
    here would be a second definition of the same rule.
    """
    from review_policy import draft_policy_from_state

    return draft_policy_from_state(dict(state or {}))


def _annotate(rules, *, state: Mapping[str, Any], gate: Mapping[str, Any]):
    """Ask the medium tier for ANNOTATION text only. Raises on failure.

    The deltas are never the model's to choose - they come from the mechanical
    translator above, clamped. What the model may add is the sentence a chart
    shows, which is the part counting cannot produce.
    """
    import ai_summary

    if not ai_summary.local_provider_enabled():
        raise RuntimeError("local AI provider is not configured (ai_local_endpoint_url unset)")
    package = _evidence_package(rules, state=state, gate=gate)
    result = ai_summary.request_ai_summary(
        provider="local",
        model=ai_summary.local_model("medium"),
        api_key="",
        evidence=package,
        timeout_seconds=900,
    )
    summary = result.get("summary") or {}
    notes = summary.get("annotations")
    text_by_segment: dict[tuple[str, str], str] = {}
    if isinstance(notes, list):
        for item in notes:
            if not isinstance(item, Mapping):
                continue
            key = (str(item.get("dimension") or ""), str(item.get("segment") or ""))
            text = str(item.get("annotation") or "").strip()
            if key[0] and key[1] and text:
                text_by_segment[key] = text[:400]
    annotated = []
    for rule in rules:
        text = text_by_segment.get(rule.key())
        if text:
            rule.annotation = text
        annotated.append(rule)
    return annotated, str(result.get("model") or "")


def _evidence_package(rules, *, state: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    import hashlib

    content = {
        "gate": dict(gate),
        "rules": [
            {"dimension": rule.dimension, "segment": rule.segment,
             "priority_delta": rule.priority_delta}
            for rule in rules
        ],
        "scoreboard_callouts": {
            "blind_spots": list(state.get("blind_spots") or []),
            "leaks": list(state.get("leaks") or []),
        },
        "instructions": (
            "For each rule, write one short sentence a chart can show explaining "
            "what the counting found. Do NOT change any priority_delta, do not "
            "add rules, and do not propose suppressing, muting or hiding "
            "anything - the format has no such field and never will."
        ),
    }
    encoded = json.dumps(content, sort_keys=True, default=str).encode("utf-8")
    package = {
        "schema_version": "ai_evidence_package_v2",
        "generated_at": _now().isoformat(timespec="seconds"),
        "session_date": str(gate.get("session_date") or ""),
        "selected_scopes": ["review_policy_draft"],
        "scope_labels": ["Mechanical review-policy draft awaiting annotation"],
        "source_count": 1,
        "sources": [{
            "source_id": "review.policy_draft",
            "label": "Mechanical draft from the review scoreboard",
            "status": "available",
            "observed_at": _now().isoformat(timespec="seconds"),
            "content_through": str(gate.get("session_date") or ""),
            "content_through_basis": "the session the scoreboard was read for",
            "session_date": str(gate.get("session_date") or ""),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "truncated": False,
            "content": content,
        }],
        "coverage": {"counts": {"requested": 1, "usable": 1, "stale": 0, "truncated": 0}},
        "safety_contract": {
            "purpose": "advisory draft; the live review policy is the trader's to save",
            "forbidden_effects": ["scanner scores", "watchlists", "alerts", "bot state", "orders"],
        },
        "scope_caveats": [
            "Ranks and annotates only. There is no suppression field and one "
            "must never be proposed.",
            "Queue ordering is FIFO and chart-watch hits stay at the front; "
            "nothing here changes that.",
        ],
    }
    canonical = json.dumps(package, sort_keys=True, separators=(",", ":"), default=str)
    package["evidence_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    package["package_id"] = package["evidence_hash"][:16]
    return package


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def run_review_policy_draft(
    *,
    session_date: str = "",
    now: datetime | None = None,
    root: Path | None = None,
    state: Mapping[str, Any] | None = None,
    narrate: bool = True,
    **_ignored: Any,
) -> dict[str, Any]:
    """Write one draft, archive a copy, and never touch the live policy."""
    from ai_jobs import ledger as job_ledger
    from project_paths import REVIEW_POLICY_DRAFT_FILE
    from review_policy import save_review_policy

    moment = _now(now)
    day = str(session_date or moment.date().isoformat())

    if state is None:
        try:
            from review_learning import build_review_learning_state

            state = build_review_learning_state()
        except Exception as exc:  # noqa: BLE001
            return {"status": job_ledger.STATUS_FAILED, "model": "",
                    "reason": f"review scoreboard unreadable: {exc}", "outputs": []}

    try:
        target_root = Path(root) if root is not None else _default_root()
    except Exception as exc:  # noqa: BLE001
        return {"status": job_ledger.STATUS_FAILED, "model": "",
                "reason": f"AI store unavailable: {exc}", "outputs": []}

    gate = gate_state(target_root)
    gate = {**gate, "session_date": day}
    rules = _mechanical_rules(state)

    model = ""
    narration_note = ""
    if narrate and rules:
        try:
            rules, model = _annotate(rules, state=state, gate=gate)
        except Exception as exc:  # noqa: BLE001 - a dead model leaves the counting
            narration_note = f" Annotation unavailable ({exc}); the mechanical draft stands."
            _log.info("Policy draft: %s", narration_note.strip())

    notes = (
        gate["statement"]
        + " Mechanical deltas come from the scoreboard's own blind-spot and leak "
        "callouts, clamped; the model may only write the sentence a chart shows."
        + narration_note
    )
    try:
        payload = save_review_policy(
            rules,
            # The ONE file this module may write. There is no code path here
            # that resolves the live policy.
            path=Path(REVIEW_POLICY_DRAFT_FILE),
            author=f"ai_jobs.policy_draft ({model or 'mechanical only'})",
            notes=notes,
            now=moment,
        )
    except OSError as exc:
        return {"status": job_ledger.STATUS_FAILED, "model": model,
                "reason": f"draft could not be written: {exc}", "outputs": []}

    outputs = [str(REVIEW_POLICY_DRAFT_FILE)]
    try:
        archived = _archive(target_root, day, {**payload, "session_date": day, "gate": gate})
        outputs.append(str(archived))
    except OSError as exc:
        # The draft still stands; only the side-by-side copy is missing, and
        # that is said rather than swallowed.
        narration_note += f" Archive copy failed: {exc}."

    return {
        "status": job_ledger.STATUS_OK,
        "model": model,
        "reason": (
            (gate["statement"] if not gate["window_met"] else GATE_MET_STATEMENT)
            + f" {len(rules)} draft rule(s) for {day}."
            + narration_note
        ),
        "outputs": outputs,
    }


def _default_root() -> Path:
    from ai_jobs import store

    return store.retros_dir()


def _archive(root: Path, day: str, payload: Mapping[str, Any]) -> Path:
    """One kept copy per session, so the two-week comparison has an archive.

    Never overwritten: a second run on the same day writes a superseding
    sibling, because a comparison whose earlier half was rewritten is not a
    comparison.
    """
    directory = _archive_dir(root) / day[:4]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{day}.json"
    index = 1
    while path.exists():
        path = directory / f"{day}.{index}.json"
        index += 1
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"schema": DRAFT_SCHEMA, **dict(payload)}, indent=1,
                              sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path
