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

    Three outcomes, all of which publish or raise honestly (checkpoint review
    2026-08-08 second review):

    * **ok** -- a validated narrative, with the system's own coverage rows
      merged into ``data_quality`` afterwards.
    * **degraded_no_narrative** -- a templated, model-free document that states
      what happened. Published when there is nothing usable to narrate (in
      which case the model is never called at all), or when the model twice
      cited evidence that does not exist. Silence would leave yesterday's brief
      in place looking like a healthy night; this cannot be mistaken for one.
      The ledger records it as degraded, not ok, so the next firing retries it.
    * **raise** -- infrastructure failure. The runner records it and leaves
      prior artifacts alone; a partial brief is worse than yesterday's brief.

    ``session_date`` is passed into evidence packaging, so every source records
    the session it represents and anything from a different one is flagged
    stale rather than presented as this session's data.
    """
    import ai_summary
    from ai_jobs import ledger

    if not ai_summary.local_provider_enabled():
        raise RuntimeError(
            "local AI provider is not configured (ai_local_endpoint_url unset); "
            "nothing to run against"
        )

    model = ai_summary.local_model("medium")
    evidence = ai_summary.build_evidence_package(list(scopes), session_date=session_date)
    coverage = evidence.get("coverage") or {}
    counts = coverage.get("counts") or {}
    logging.info(
        "AI summary: package %s for session %s, %s/%s usable source(s) "
        "(empty=%s missing=%s invalid=%s unavailable=%s unfunded=%s stale=%s), model %s",
        evidence.get("package_id"),
        session_date,
        counts.get("usable"),
        counts.get("requested"),
        counts.get("empty"),
        counts.get("missing"),
        counts.get("invalid"),
        counts.get("unavailable"),
        counts.get("unfunded"),
        counts.get("stale"),
        model,
    )

    def _publish(result: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
        exported = ai_summary.export_ai_summary(
            result, evidence, output_dir=_summary_dir(session_date)
        )
        outputs = [str(path) for path in exported.values()]
        logging.info(
            "AI summary published %s file(s) for %s (%s)", len(outputs), session_date, status
        )
        return {
            "status": status,
            "model": result.get("model", ""),
            "reason": reason,
            "outputs": outputs,
            "tokens": {"duration_seconds": result.get("duration_seconds")},
            "coverage": counts,
        }

    if not ai_summary.has_usable_sources(evidence):
        # Nothing to narrate. Calling a 14GB model to say so would only give it
        # the opportunity to invent something.
        reason = (
            f"no usable evidence for {session_date}: "
            f"0 of {counts.get('requested', 0)} requested source(s) carried content"
        )
        logging.warning("AI summary degraded: %s", reason)
        return _publish(
            ai_summary.degraded_result(evidence, reason=reason + ".", model=""),
            ledger.STATUS_DEGRADED,
            reason,
        )

    previous_error = ""
    for attempt in (1, 2):
        try:
            result = ai_summary.request_ai_summary(
                provider="local",
                model=model,
                api_key="",  # localhost; request_ai_summary supplies the placeholder
                evidence=evidence,
                timeout_seconds=900,
                previous_error=previous_error,
            )
        except (ValueError, RuntimeError) as exc:
            previous_error = str(exc)
            if attempt == 1:
                logging.warning(
                    "AI summary attempt 1 rejected (%s); retrying once with the "
                    "specific error fed back.",
                    previous_error,
                )
                continue
            reason = (
                f"the model failed validation twice for {session_date}; "
                f"last rejection: {previous_error}"
            )
            logging.warning("AI summary degraded: %s", reason)
            return _publish(
                ai_summary.degraded_result(evidence, reason=reason + ".", model=model),
                ledger.STATUS_DEGRADED,
                reason,
            )

        # Provenance is the code's to state, never the model's to estimate.
        result = dict(result)
        result["summary"] = ai_summary.merge_coverage_into_summary(
            result.get("summary") or {}, evidence
        )
        return _publish(
            result,
            ledger.STATUS_OK,
            f"summary for {session_date} from {counts.get('usable', 0)} usable source(s)"
            + (f", {counts.get('stale', 0)} stale" if counts.get("stale") else ""),
        )
    raise RuntimeError("unreachable")  # pragma: no cover
