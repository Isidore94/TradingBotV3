"""Phase 1: the advisory AI summary, unattended (plan Phase 1).

This job is deliberately thin. It does not invent an analysis pipeline -- it
runs the *existing* `ai_summary` path (the same evidence packaging, the same
strict validation, the same evidence-reference checking) against the local
endpoint, and publishes the result into the AI store. What changes versus
today is only that nobody has to click Generate.

Everything the plan forbids still holds: the output is a document a human
reads, it cites only source_ids from its own evidence package, and it touches
no detector, score, alert, or state machine.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

#: The scopes a hands-off nightly review should cover. Journal review is
#: deliberately included: it is the trader's own revealed preference and the
#: whole point of the nightly read.
DEFAULT_SCOPES = ("daily_report", "market_conditions", "setup_trackers", "journal_review")


def _summary_dir(session_date: str) -> Path:
    from ai_jobs.store import briefs_dir

    target = briefs_dir() / session_date[:4]
    target.mkdir(parents=True, exist_ok=True)
    return target


def run_daily_summary(
    *,
    session_date: str,
    now: datetime | None = None,
    scopes: tuple[str, ...] = DEFAULT_SCOPES,
) -> dict[str, Any]:
    """Build the evidence package, ask the local model, publish the result.

    Raises on failure so the runner records it and leaves prior artifacts
    alone -- a partial brief is worse than yesterday's brief.
    """
    import ai_summary

    if not ai_summary.local_provider_enabled():
        raise RuntimeError(
            "local AI provider is not configured (ai_local_endpoint_url unset); "
            "nothing to run against"
        )

    model = ai_summary.local_model("medium")
    evidence = ai_summary.build_evidence_package(list(scopes))
    logging.info(
        "AI summary: package %s, %s source(s), model %s",
        evidence.get("package_id"),
        len(evidence.get("sources") or []),
        model,
    )

    result = ai_summary.request_ai_summary(
        provider="local",
        model=model,
        api_key="",  # localhost endpoint; request_ai_summary supplies the placeholder
        evidence=evidence,
        timeout_seconds=900,
    )

    exported = ai_summary.export_ai_summary(
        result, evidence, output_dir=_summary_dir(session_date)
    )
    outputs = [str(path) for path in exported.values()]
    logging.info("AI summary published %s file(s) for %s", len(outputs), session_date)
    return {
        "model": result.get("model", model),
        "reason": (
            f"summary for {session_date} from {len(evidence.get('sources') or [])} source(s)"
        ),
        "outputs": outputs,
        "tokens": {"duration_seconds": result.get("duration_seconds")},
    }
