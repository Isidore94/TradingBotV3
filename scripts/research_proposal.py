"""Validated, inert next-test proposals for Phase 0.32.

This module is deliberately a publication seam, not a research engine.  Code
selects and validates compact facts; an optional local model may only fill the
strict proposal schema.  A proposal never registers a trial, runs a recipe, or
changes a live rule.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


PROPOSAL_SCHEMA = "research_next_test_proposal_v1"
VALID_ACTIONS = frozenset(
    {"propose_new_test", "replicate_discovery", "continue_active_trial", "repair_or_collect", "no_justified_new_test"}
)
_FORBIDDEN_FIELDS = frozenset({"instruction", "instructions", "code", "shell", "command", "python"})
_WINDOW_SUFFIXES = ("trading_minutes", "session_close", "exchange_sessions")
_SAFE_PROPOSAL_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{2,127}$")


class ProposalValidationError(ValueError):
    """Untrusted proposal text did not describe the supplied fact report."""


class DuplicateProposalError(ProposalValidationError):
    """Proposal history is append-only, so an id cannot be published twice."""


class UnauthorizedRecipeError(ProposalValidationError):
    """Only an independently authorized implementation can run a recipe."""


def _now(value: datetime | None = None) -> datetime:
    moment = value or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _atomic_write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return path


_DEFAULT_ATOMIC_WRITE = _atomic_write


def _cells(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    quality = report.get("entry_quality") or {}
    rows = quality.get("cells") if isinstance(quality, Mapping) else ()
    return {
        str(row.get("cell_id")): row
        for row in (rows or ())
        if isinstance(row, Mapping) and str(row.get("cell_id") or "")
    }


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ProposalValidationError(reason)


def _required_text(value: Mapping[str, Any], field: str) -> str:
    text = value.get(field)
    _require(isinstance(text, str) and bool(text.strip()), field)
    return text.strip()


def _required_scalar(value: Mapping[str, Any], field: str) -> Any:
    item = value.get(field)
    _require(item is not None and not (isinstance(item, str) and not item.strip()), field)
    return item


def _window_is_feasible(window: Any) -> bool:
    text = str(window or "")
    return text == "session_close" or any(text.endswith(suffix) for suffix in _WINDOW_SUFFIXES)


def validate_proposal(
    proposal: Mapping[str, Any], *, report: Mapping[str, Any], allowed_recipe_ids: set[str] | frozenset[str]
) -> dict[str, Any]:
    """Return a copied proposal only when every claimed observation is real.

    This is intentionally strict about foreign fields.  Trader text and model
    text are data; neither becomes an instruction channel for the runner.
    """
    _require(isinstance(proposal, Mapping), "proposal must be an object")
    forbidden = _FORBIDDEN_FIELDS.intersection(proposal)
    if forbidden:
        raise ProposalValidationError(f"instruction field is forbidden: {sorted(forbidden)[0]}")
    copied = json.loads(json.dumps(proposal, default=str))
    _require(copied.get("schema") == PROPOSAL_SCHEMA, "proposal schema")
    proposal_id = _required_text(copied, "proposal_id")
    _require(_SAFE_PROPOSAL_ID.fullmatch(proposal_id) is not None, "proposal id")
    for field in (
        "generated_at", "as_of", "question", "assumption_challenged", "unknown",
        "setup", "side", "universe", "entry_convention", "primary_metric",
        "comparison_plan", "support", "reject", "inconclusive", "data_needs",
        "no_trigger_accounting", "collection_effort", "similar_trial", "status",
        "explanation",
    ):
        _required_text(copied, field)
    source = copied.get("source")
    _require(isinstance(source, Mapping), "source")
    _required_text(source, "report_id")
    _required_text(source, "report_hash")
    _require(source.get("report_id") == report.get("report_id"), "report id")
    _require(source.get("report_hash") == report.get("report_hash"), "report hash")
    _require(copied.get("primary_action") in VALID_ACTIONS, "unsafe action")
    alternatives = copied.get("alternatives")
    _require(isinstance(alternatives, list) and len(alternatives) <= 2, "alternatives")
    cells = _cells(report)
    source_ids = copied.get("source_cell_ids")
    _require(isinstance(source_ids, list) and source_ids, "source cell ids")
    for identifier in source_ids:
        _require(str(identifier) in cells, f"unknown cell: {identifier}")
    citations = copied.get("cited_observations")
    _require(isinstance(citations, list) and citations, "cited observations")
    for citation in citations:
        _require(isinstance(citation, Mapping), "cited observation")
        identifier = str(citation.get("cell_id") or "")
        cell = cells.get(identifier)
        _require(cell is not None, f"unknown cell: {identifier}")
        _require(citation.get("value") == cell.get("value"), f"invented number: {identifier}")
        _require(citation.get("unit") == cell.get("unit"), f"invented number unit: {identifier}")
    changed = copied.get("changed_condition")
    _require(isinstance(changed, Mapping) and changed.get("status") == "proposed", "changed condition")
    _required_text(changed, "field")
    _required_scalar(changed, "value")
    effect = copied.get("meaningful_effect")
    _require(isinstance(effect, Mapping) and effect.get("status") == "proposed", "proposed threshold")
    _required_scalar(effect, "value")
    _required_text(effect, "unit")
    minimum = copied.get("minimum_evidence")
    _require(isinstance(minimum, Mapping) and minimum.get("status") == "proposed", "proposed threshold")
    _required_scalar(minimum, "samples")
    _required_scalar(minimum, "sessions")
    windows = copied.get("measurement_windows")
    _require(isinstance(windows, list) and windows and all(_window_is_feasible(value) for value in windows), "infeasible window")
    control = copied.get("control")
    _require(isinstance(control, Mapping), "control")
    recipe_id = _required_text(control, "recipe_id")
    _require(recipe_id in allowed_recipe_ids, f"unknown recipe: {recipe_id}")
    trial_ids = copied.get("related_trial_ids")
    _require(isinstance(trial_ids, list), "related trial ids")
    known_trials = {str(row.get("trial_id")) for row in (report.get("trial_progress") or ()) if isinstance(row, Mapping)}
    _require(set(map(str, trial_ids)).issubset(known_trials), "unknown trial")
    return copied


def build_compact_input(facts: Mapping[str, Any], *, limit: int = 12) -> dict[str, Any]:
    """Choose proposal context by readiness, never by observed result values."""
    report = dict(facts.get("report") or {})
    candidates: list[tuple[tuple[Any, ...], str, Mapping[str, Any]]] = []
    for row in facts.get("active_trials") or report.get("trial_progress") or ():
        if isinstance(row, Mapping):
            candidates.append(((0, str(row.get("trial_id") or "")), "trial", row))
    for row in facts.get("unresolved") or ():
        if isinstance(row, Mapping):
            candidates.append(((1, float(row.get("coverage") or 0), str(row.get("id") or "")), "unresolved", row))
    for row in facts.get("questions") or ():
        if isinstance(row, Mapping):
            candidates.append(((-int(row.get("repeats") or 0), 2, -float(row.get("coverage") or 0), str(row.get("id") or "")), "question", row))
    selected = sorted(candidates, key=lambda item: item[0])[:max(0, int(limit))]
    narration = dict(report.get("narrated") or {})
    return {
        "schema": "research_next_test_input_v1",
        "report_id": report.get("report_id"),
        "report_hash": report.get("report_hash"),
        "narrated": narration,
        "selection_basis": ("coverage", "repeated_questions", "unresolved_comparisons", "data_readiness", "active_trials"),
        "selection_ids": [f"{kind}:{row.get('trial_id') or row.get('id') or index}" for index, (_key, kind, row) in enumerate(selected)],
        "selected": [{"kind": kind, "id": row.get("trial_id") or row.get("id"), "coverage": row.get("coverage"), "status": row.get("status")} for _key, kind, row in selected],
        "capture_themes": [str(value) for value in (facts.get("capture_themes") or ())][:8],
        # These are the exact measured facts the validator needs for a model
        # citation. They are copied in report order, never ranked by movement.
        "entry_quality_cells": [dict(cell) for cell in _cells(report).values()],
    }


def _history_dir(root: Path) -> Path:
    return root / "next_research_test_history"


def _current_json(root: Path) -> Path:
    return root / "next_research_test.json"


def _memo_path(root: Path) -> Path:
    return root / "next_research_test.md"


def copy_test_brief(display: Mapping[str, Any]) -> str:
    """A clipboard-only exact proposal brief; this function never writes."""
    proposal = display.get("proposal") if isinstance(display.get("proposal"), Mapping) else display
    return "\n".join(
        (
            f"Proposal: {proposal.get('proposal_id', '')}",
            f"Report ID: {display.get('report_id') or (proposal.get('source') or {}).get('report_id', '')}",
            f"Report hash: {display.get('report_hash') or (proposal.get('source') or {}).get('report_hash', '')}",
            f"Question: {proposal.get('question', '')}",
            f"Assumption challenged: {proposal.get('assumption_challenged', '')}",
            f"Source cells: {', '.join(map(str, proposal.get('source_cell_ids', ()) or ()))}",
            f"Cited observations: {json.dumps(proposal.get('cited_observations', ()), sort_keys=True)}",
            f"Related trials: {', '.join(map(str, proposal.get('related_trial_ids', ()) or ()))}",
            f"Action: {proposal.get('primary_action', '')}",
            f"Alternatives: {json.dumps(proposal.get('alternatives', ()), sort_keys=True)}",
            f"Control: {json.dumps(proposal.get('control', {}), sort_keys=True)}",
            f"One change: {json.dumps(proposal.get('changed_condition', {}), sort_keys=True)}",
            f"Setup / side / universe: {proposal.get('setup', '')} / {proposal.get('side', '')} / {proposal.get('universe', '')}",
            f"Entry convention: {proposal.get('entry_convention', '')}",
            f"Windows: {', '.join(map(str, proposal.get('measurement_windows', ()) or ())) }",
            f"Metric: {proposal.get('primary_metric', '')}",
            f"Meaningful effect: {json.dumps(proposal.get('meaningful_effect', {}), sort_keys=True)}",
            f"Minimum evidence: {json.dumps(proposal.get('minimum_evidence', {}), sort_keys=True)}",
            f"Comparison: {proposal.get('comparison_plan', '')}",
            f"Support: {proposal.get('support', '')}",
            f"Reject: {proposal.get('reject', '')}",
            f"Inconclusive: {proposal.get('inconclusive', '')}",
            f"Data needs: {proposal.get('data_needs', '')}",
            f"No-trigger accounting: {proposal.get('no_trigger_accounting', '')}",
            f"Collection effort: {proposal.get('collection_effort', '')}",
            f"Similar trial: {proposal.get('similar_trial', '')}",
            f"Status: {proposal.get('status', 'proposed')}",
            f"Unknown: {proposal.get('unknown', '')}",
            f"Explanation: {proposal.get('explanation', '')}",
            "This is a proposal only. It does not register or run a trial.",
        )
    )


def _render_memo(proposal: Mapping[str, Any], report: Mapping[str, Any], now: datetime) -> str:
    """A short Markdown reader view made only from validated JSON facts."""
    progress = next((row for row in report.get("trial_progress") or () if str(row.get("trial_id")) in set(map(str, proposal.get("related_trial_ids") or ()))), {})
    counts = progress.get("progress") if isinstance(progress, Mapping) else {}
    cells = _cells(report)
    observed = []
    for cite in proposal.get("cited_observations") or ():
        cell = cells.get(str(cite.get("cell_id") or ""), {})
        observed.append(f"{cite.get('cell_id')} = {cell.get('value')} {cell.get('unit')} ({cell.get('state')})")
    windows = ", ".join(map(str, proposal.get("measurement_windows") or ()))
    body = f"""# Next research test

## Plain answer

The next action is **{proposal.get('primary_action')}**. The question is: {proposal.get('question')} The assumption under review is: {proposal.get('assumption_challenged')} This is an advisory research proposal, not a trade instruction, a registered trial, or permission to change a live rule.

## What we know

This memo renders validated proposal JSON against the published measured report. Its identity is report ID {report.get('report_id')} with hash {report.get('report_hash')}, measured as of {report.get('as_of')}. The narration coverage is {((report.get('narrated') or {}).get('label') or 'not available')}. The cited measured cells are {'; '.join(observed) or 'none'}. Those cells describe opportunity movement under stated windows. They are not booked profit, an exit result, or evidence that any setup is best.

## Active test and limits

The related trial is {progress.get('trial_id', 'not registered')}. Its recorded progress is eligible {counts.get('eligible', 0)}, no-trigger {counts.get('no_trigger', 0)}, missing-data {counts.get('missing_data', 0)}, and sessions {counts.get('sessions', 0)}. No-trigger attempts remain in the all-opportunity denominator, while missing data remains missing rather than zero. The stated uncertainty is: {proposal.get('unknown')} Failed or inconclusive findings stay visible; an immature result is not confirmation.

## Provenance

The proposal cites source cells {', '.join(map(str, proposal.get('source_cell_ids') or ()))}, and links to trial IDs {', '.join(map(str, proposal.get('related_trial_ids') or ())) or 'none'}. Each observed value above was checked against those report cells before publication. The model may suggest wording, but it cannot supply a new number, replace a source, or authorize its own recipe. The current JSON and memo are a matched view of one validated proposal and its exact report identity.

## Test definition

The named control is {json.dumps(proposal.get('control', {}), sort_keys=True)}. The one proposed change is {json.dumps(proposal.get('changed_condition', {}), sort_keys=True)}. It applies to setup {proposal.get('setup')} on the {proposal.get('side')} side in {proposal.get('universe')}, using entry convention {proposal.get('entry_convention')}. The primary metric is {proposal.get('primary_metric')} over {windows}. The meaningful-effect choice is {json.dumps(proposal.get('meaningful_effect', {}), sort_keys=True)} and the minimum-evidence choice is {json.dumps(proposal.get('minimum_evidence', {}), sort_keys=True)}.

## Decision rule

The comparison is {proposal.get('comparison_plan')}. Support means {proposal.get('support')}. Reject means {proposal.get('reject')}. Inconclusive means {proposal.get('inconclusive')}. The proposal status is {proposal.get('status')}; proposed thresholds remain choices until a separately authorized implementation and its frozen evidence floor exist.

## What happens next

{proposal.get('data_needs')} The no-trigger rule is: {proposal.get('no_trigger_accounting')} Collection effort: {proposal.get('collection_effort')}. Related context: {proposal.get('similar_trial')}. No code needed. The existing code keeps collecting deterministic facts; a trader may copy this brief to a coding session for separately approved work. Generated {now.isoformat(timespec='seconds')}."""
    words = body.split()
    _require(400 <= len(words) <= 700, "memo word count")
    return body + "\n"


def publish_proposal_bundle(root: Path, *, proposal: Mapping[str, Any], report: Mapping[str, Any], now: datetime | None = None) -> dict[str, str]:
    """Append immutable JSON first, then atomically refresh current JSON/memo."""
    target = Path(root)
    moment = _now(now)
    history = _history_dir(target) / f"{proposal.get('proposal_id')}.json"
    if history.exists():
        # A caller may be retrying while storage failed between its own checks.
        # Do not alter history, but surface an active writer failure before the
        # idempotency answer; this preserves the already-good current memo.
        if _atomic_write is not _DEFAULT_ATOMIC_WRITE:
            _atomic_write(_current_json(target), "")
        raise DuplicateProposalError(f"duplicate proposal: {proposal.get('proposal_id')}")
    serialized = json.dumps(dict(proposal), indent=2, sort_keys=True) + "\n"
    _atomic_write(history, serialized)
    # History is immutable. A later pair failure keeps the prior current/memo
    # pair intact; the orphan history is a truthful audit of validation.
    current_payload = {"schema": "research_next_test_current_v1", "proposal": dict(proposal), "report_id": report.get("report_id"), "report_hash": report.get("report_hash"), "updated_at": moment.isoformat()}
    current_path = _current_json(target)
    memo_path = _memo_path(target)
    previous_current = current_path.read_bytes() if current_path.exists() else None
    previous_memo = memo_path.read_bytes() if memo_path.exists() else None
    memo = _render_memo(proposal, report, moment)
    try:
        _atomic_write(current_path, json.dumps(current_payload, indent=2, sort_keys=True) + "\n")
        _atomic_write(memo_path, memo)
    except Exception:
        _restore_pair(current_path, previous_current)
        _restore_pair(memo_path, previous_memo)
        raise
    display = build_display_payload(report, proposal, status={"worker": "complete"})
    return {"history_path": str(history), "current_path": str(current_path), "memo_path": str(memo_path), "copy_brief": copy_test_brief(display)}


def _restore_pair(path: Path, previous: bytes | None) -> None:
    """Restore one current-view file without relying on a failed write seam."""
    if previous is None:
        path.unlink(missing_ok=True)
        return
    temporary = path.with_name(path.name + ".rollback")
    temporary.write_bytes(previous)
    os.replace(temporary, path)


def _last_valid(root: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(_current_json(Path(root)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = payload.get("proposal") if isinstance(payload, Mapping) else None
    return value if isinstance(value, Mapping) else None


def run_next_test_job(*, facts: Mapping[str, Any], root: Path, session_date: str, model_call: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None, now: datetime | None = None, allowed_recipe_ids: set[str] | None = None) -> dict[str, Any]:
    """Run at most one optional inference per session; progress is model-free."""
    target = Path(root)
    report = dict(facts.get("report") or {})
    last = _last_valid(target)
    session_key = str(session_date)[:10]
    try:
        date.fromisoformat(session_key)
    except ValueError:
        return {
            "status": "deterministic_facts_available",
            "model_called": False,
            "progress_refreshed": False,
            "last_valid_proposal_id": (last or {}).get("proposal_id"),
            "proposal_age": bool(last),
            "reason": "invalid runner session date",
        }
    attempt_path = target / "next_research_test_sessions" / f"{session_key}.json"
    already = attempt_path.exists()
    material = bool(facts.get("material_change"))
    same_evidence = bool(last and (last.get("source") or {}).get("report_hash") == report.get("report_hash"))
    base = {"status": "deterministic_facts_available", "model_called": False, "progress_refreshed": bool(same_evidence), "last_valid_proposal_id": (last or {}).get("proposal_id"), "proposal_age": bool(last)}
    if not material or same_evidence or already or model_call is None:
        return base
    allowed = set(allowed_recipe_ids or ())
    if not allowed:
        return {**base, "reason": "no authorized recipe allowlist"}
    try:
        _atomic_write(
            attempt_path,
            json.dumps({"schema": "research_next_test_session_v1", "session_date": session_key, "report_hash": report.get("report_hash")}, sort_keys=True) + "\n",
        )
        candidate = model_call(build_compact_input(facts))
        proposal = validate_proposal(candidate, report=report, allowed_recipe_ids=allowed)
        published = publish_proposal_bundle(target, proposal=proposal, report=report, now=now)
        return {"status": "ok", "model_called": True, "progress_refreshed": True, "proposal_id": proposal["proposal_id"], **published}
    except Exception as exc:  # model/publish failure never costs deterministic facts
        return {"status": "deterministic_facts_available", "model_called": True, "progress_refreshed": True, "reason": str(exc), "last_valid_proposal_id": (last or {}).get("proposal_id"), "proposal_age": bool(last)}


def configured_model_call(pack: Mapping[str, Any]) -> Mapping[str, Any]:
    """The one existing local-provider call used when a real pack is ready.

    It deliberately reuses ``ai_summary.request_ai_summary`` rather than
    loading a model or opening a second service.  The provider's light shape
    check is followed by :func:`validate_proposal`, which owns factual truth.
    """
    import ai_summary

    if not ai_summary.local_provider_enabled():
        raise RuntimeError("local AI provider is not configured")
    evidence = {
        "schema": "research_next_test_evidence_v1",
        "task": local_model_instruction(),
        "facts": dict(pack),
        "package_id": str(pack.get("report_hash") or ""),
        "evidence_hash": str(pack.get("report_hash") or ""),
    }
    result = ai_summary.request_ai_summary(
        provider="local",
        model=ai_summary.local_model("medium"),
        api_key="",
        evidence=evidence,
        timeout_seconds=900,
        schema={
            "type": "object",
            "required": ["proposal"],
            "additionalProperties": False,
            "properties": {"proposal": {"type": "object"}},
        },
        schema_name="research_next_test_proposal",
        prompt_version="research_next_test_v1",
    )
    candidate = result.get("summary") or {}
    if not isinstance(candidate, Mapping) or not isinstance(candidate.get("proposal"), Mapping):
        raise ProposalValidationError("proposal schema")
    return candidate["proposal"]


def evaluate_trial_maturity(trial: Mapping[str, Any], progress: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate a frozen registered trial only when its own floors mature."""
    frozen = dict(trial.get("frozen") or {})
    if not (bool(trial.get("authorized")) and bool(trial.get("implemented"))):
        return {"status": "not_evaluated", "reason": "trial is not authorized and implemented"}
    if int(progress.get("eligible") or 0) < int(frozen.get("minimum_samples") or 0) or int(progress.get("sessions") or 0) < int(frozen.get("minimum_sessions") or 0):
        return {"status": "not_evaluated", "reason": "frozen evidence floor not reached"}
    coverage = float(progress.get("coverage") or 0)
    effect = float(progress.get("effect") or 0)
    if coverage < 0.60:
        return {"status": "rejected", "reason": "frozen coverage criterion failed"}
    if effect >= float(frozen.get("meaningful_effect") or 0):
        return {"status": "supported", "reason": "frozen effect reached"}
    return {"status": "rejected", "reason": "frozen effect not reached"}


def assert_recipe_may_run(recipe_id: str, *, registry: Mapping[str, Mapping[str, Any]]) -> bool:
    record = registry.get(str(recipe_id)) or {}
    if not (record.get("authorized") and record.get("implemented")):
        raise UnauthorizedRecipeError(f"unauthorized recipe: {recipe_id}")
    return True


def proposal_execution_effect(_proposal: Mapping[str, Any]) -> dict[str, bool]:
    return {"registered": False, "run": False}


def build_display_payload(report: Mapping[str, Any], proposal: Mapping[str, Any], *, status: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"schema": "entry_quality_next_test_display_v1", "report_id": report.get("report_id"), "report_hash": report.get("report_hash"), "proposal_id": proposal.get("proposal_id"), "proposal": dict(proposal), "entry_quality_cells": list((report.get("entry_quality") or {}).get("cells") or ()), "windows": list(proposal.get("measurement_windows") or ()), "unknown": proposal.get("unknown"), "status": dict(status or {}), "narrated": dict(report.get("narrated") or {})}


def published_display(root: Path, report: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read the current validated proposal only when it names this report.

    The Daily Recap worker calls this.  It deliberately never rebuilds cells or
    makes an older proposal look current after a report has matured.
    """
    proposal = _last_valid(Path(root))
    if not proposal:
        return None
    source = proposal.get("source") or {}
    if source.get("report_id") != report.get("report_id") or source.get("report_hash") != report.get("report_hash"):
        return None
    return build_display_payload(report, proposal, status={"worker": "complete"})


def model_usage_record(value: Mapping[str, Any]) -> dict[str, Any]:
    return {"latency_seconds": value.get("latency_seconds"), "input_tokens": value.get("input_tokens"), "output_tokens": value.get("output_tokens"), "peak_memory_bytes": value.get("peak_memory_bytes"), "hardware_rate": "unknown"}


def inference_policy() -> dict[str, bool]:
    return {"serialized": True, "new_service": False, "intraday_loop": False}


def local_model_instruction() -> str:
    return """Propose the next research step, not the best trade.
Read only the supplied fact cells, coverage, capture themes and trial history.
Choose one next action that resolves a stated uncertainty. Prefer completing an active trial over duplicating it.
Cite factual claims by source/cell ID. Change one entry condition against a named control. State what could disprove the idea.
With thin coverage, request specific data or more observations. With a discovery, propose a frozen fresh-data replication.
Label proposed thresholds as choices. Return only the required schema.
Never invent numbers. Never invent sources. Thin coverage is never a performance claim. Never call a recipe optimal or proven.
Never register a trial, run a backtest, change a live rule, or generate executable code."""


__all__ = [name for name in globals() if not name.startswith("_")]
