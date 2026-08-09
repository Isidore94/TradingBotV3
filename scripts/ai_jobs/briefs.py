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

import hashlib
import json
import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

#: The scopes a hands-off nightly review should cover. Journal review is
#: deliberately included: it is the trader's own revealed preference and the
#: whole point of the nightly read.
DEFAULT_SCOPES = ("daily_report", "market_conditions", "setup_trackers", "journal_review")
TICKER_BRIEF_SCOPES = DEFAULT_SCOPES
MORNING_BRIEF_FILENAME = "ai_morning_brief.txt"
MAX_TICKER_SOURCE_CHARS = 16_000
MAX_MORNING_BRIEF_BYTES = 48 * 1024
_SYMBOL_TOKEN = re.compile(r"^[A-Z][A-Z0-9.-]{0,14}$")


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


def default_watchlist_paths() -> dict[str, Path]:
    """Focus and trader watchlists, read-only, in morning-file priority order."""
    import project_paths

    return {
        "focus_longs": Path(project_paths.FOCUS_LONGS_FILE),
        "focus_shorts": Path(project_paths.FOCUS_SHORTS_FILE),
        "longs": Path(project_paths.LONGS_FILE),
        "shorts": Path(project_paths.SHORTS_FILE),
        "swing_longs": Path(project_paths.SWING_LONGS_FILE),
        "swing_shorts": Path(project_paths.SWING_SHORTS_FILE),
    }


def load_brief_symbols(
    paths: Mapping[str, Path] | None = None,
) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    """Read ticker membership without ever changing a list.

    Returns symbols in first-seen order (Focus first by the default mapping)
    plus code-owned evidence pointers naming every list that contained each
    ticker. An unreadable list is uncertainty: it contributes no names and is
    never rewritten, repaired, or treated as evidence that a name was removed.
    """
    selected = paths or default_watchlist_paths()
    ordered: list[str] = []
    memberships: dict[str, list[dict[str, str]]] = {}
    for list_name, raw_path in selected.items():
        path = Path(raw_path)
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            logging.warning("Ticker briefs: watchlist %s is unreadable at %s", list_name, path)
            continue
        for line in lines:
            token = str(line or "").split("#", 1)[0].strip().lstrip("$").upper()
            if not _SYMBOL_TOKEN.fullmatch(token):
                continue
            if token not in memberships:
                memberships[token] = []
                ordered.append(token)
            memberships[token].append({"list": str(list_name), "path": str(path)})
    return ordered, memberships


def _extract_ticker_content(content: Any, symbol: str) -> Any | None:
    """Bounded, deterministic symbol projection from one packaged source."""
    if isinstance(content, str):
        pattern = re.compile(rf"(?<![A-Z0-9.-]){re.escape(symbol)}(?![A-Z0-9.-])", re.I)
        lines = [line for line in content.splitlines() if pattern.search(line)]
        return "\n".join(lines)[:MAX_TICKER_SOURCE_CHARS] if lines else None
    if isinstance(content, Mapping):
        direct_symbol = str(content.get("symbol") or "").strip().upper()
        if direct_symbol == symbol:
            encoded = json.dumps(content, sort_keys=True, default=str)
            return json.loads(encoded[:MAX_TICKER_SOURCE_CHARS]) if len(encoded) <= MAX_TICKER_SOURCE_CHARS else {
                "symbol": symbol,
                "truncated_record": encoded[:MAX_TICKER_SOURCE_CHARS],
            }
        projected: dict[str, Any] = {}
        for key, value in content.items():
            if str(key).strip().upper() == symbol:
                projected[str(key)] = value
                continue
            found = _extract_ticker_content(value, symbol)
            if found not in (None, {}, []):
                projected[str(key)] = found
        return projected or None
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        rows = []
        for value in content:
            found = _extract_ticker_content(value, symbol)
            if found not in (None, {}, []):
                rows.append(found)
        return rows or None
    return None


def build_ticker_evidence(
    base: Mapping[str, Any],
    symbol: str,
    memberships: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Project the nightly package into a single validated ticker package."""
    symbol = str(symbol or "").strip().upper()
    sources: list[dict[str, Any]] = []
    not_applicable: list[dict[str, str]] = []
    for raw in base.get("sources") or []:
        if not isinstance(raw, Mapping):
            continue
        content = _extract_ticker_content(raw.get("content"), symbol)
        if content is None:
            not_applicable.append(
                {
                    "source_id": str(raw.get("source_id") or ""),
                    "label": str(raw.get("label") or ""),
                    "scope": "ticker_projection",
                    "status": "not_applicable",
                    "reason": f"no {symbol} record in this usable source",
                }
            )
            continue
        source = dict(raw)
        source["content"] = content
        source["evidence_pointer"] = {
            "source_id": str(raw.get("source_id") or ""),
            "path": str(raw.get("path") or ""),
            "as_of": str(raw.get("as_of") or ""),
        }
        sources.append(source)

    membership_source = {
        "source_id": "watchlists.membership",
        "label": "Focus/watchlist membership",
        "status": "available",
        "as_of": str(base.get("generated_at") or ""),
        "source_session": str(base.get("session_date") or ""),
        "content": {"symbol": symbol, "memberships": [dict(row) for row in memberships]},
        "evidence_pointer": {
            "source_id": "watchlists.membership",
            "paths": [str(row.get("path") or "") for row in memberships],
        },
    }
    sources.insert(0, membership_source)

    base_coverage = base.get("coverage") if isinstance(base.get("coverage"), Mapping) else {}
    excluded = [
        dict(row) for row in base_coverage.get("excluded") or [] if isinstance(row, Mapping)
    ] + not_applicable
    counts = {
        "requested": len(sources) + len(excluded),
        "usable": len(sources),
        "stale": sum(1 for row in excluded if row.get("status") == "stale"),
        "truncated": sum(1 for row in sources if row.get("truncated")),
    }
    for status in ("empty", "missing", "invalid", "unavailable", "unfunded", "not_applicable"):
        counts[status] = sum(1 for row in excluded if row.get("status") == status)
    coverage = {
        "schema_version": "ai_evidence_coverage_v1",
        "requested_session": str(base.get("session_date") or ""),
        "usable_source_ids": [str(row.get("source_id") or "") for row in sources],
        "excluded": excluded,
        "stale": list(base_coverage.get("stale") or []),
        "truncated": [
            {
                "source_id": str(row.get("source_id") or ""),
                "label": str(row.get("label") or ""),
                "notices": list(row.get("notices") or []),
            }
            for row in sources
            if row.get("truncated")
        ],
        "journal_import_health": dict(base_coverage.get("journal_import_health") or {}),
        "counts": counts,
    }
    package = {
        "schema_version": "ai_ticker_evidence_package_v1",
        "generated_at": str(base.get("generated_at") or ""),
        "trade_date": str(base.get("trade_date") or ""),
        "session_date": str(base.get("session_date") or ""),
        "brief_symbol": symbol,
        "brief_request": (
            f"Produce an advisory brief for {symbol} only. Do not make claims about "
            "another ticker; an empty supported finding is valid."
        ),
        "selected_scopes": list(base.get("selected_scopes") or []),
        "scope_labels": list(base.get("scope_labels") or []),
        "source_count": len(sources),
        "sources": sources,
        "coverage": coverage,
        "safety_contract": dict(base.get("safety_contract") or {}),
    }
    canonical = json.dumps(package, sort_keys=True, separators=(",", ":"), default=str)
    package["evidence_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    package["package_id"] = package["evidence_hash"][:16]
    return package


def morning_brief_path() -> Path:
    import project_paths

    return Path(project_paths.PERSISTENT_DATA_DIR) / MORNING_BRIEF_FILENAME


def atomic_publish_morning_file(
    content: str,
    *,
    path: Path | None = None,
    replace=os.replace,
) -> Path:
    """Verified same-volume publish; a failed replace leaves the prior file."""
    target = Path(path or morning_brief_path())
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = str(content).encode("utf-8")
    fd, staged_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    staged = Path(staged_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if staged.read_bytes() != payload:
            raise OSError(f"morning brief verification failed for {staged}")
        replace(staged, target)
        return target
    finally:
        try:
            staged.unlink(missing_ok=True)
        except OSError:
            pass


def _morning_section(symbol: str, memberships: Sequence[Mapping[str, str]], result: Mapping[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), Mapping) else {}
    lists = ", ".join(str(row.get("list") or "") for row in memberships)
    lines = [f"## {symbol}  [{lists}]", str(summary.get("executive_summary") or "No supported finding.")]
    emitted = 0
    for section in ("best_candidates", "what_is_working", "risk_notes", "lessons_for_tomorrow"):
        rows = summary.get(section) if isinstance(summary.get(section), list) else []
        for row in rows:
            if not isinstance(row, Mapping) or emitted >= 4:
                continue
            refs = ", ".join(str(ref) for ref in row.get("evidence_refs") or []) or "no source"
            lines.append(f"- {row.get('statement')} [{refs}]")
            emitted += 1
    return "\n".join(lines).strip() + "\n"


def render_morning_file(
    session_date: str, briefs: Sequence[Mapping[str, Any]], *, generated_at: datetime | None = None
) -> str:
    generated = (generated_at or datetime.now().astimezone()).isoformat(timespec="seconds")
    header = (
        "# LOCAL-AI MORNING TICKER BRIEFS — ADVISORY ONLY\n"
        f"Session reviewed: {session_date}\nGenerated: {generated}\n"
        "This file cannot change scanners, scores, watchlists, alerts, or bot state.\n\n"
    )
    text = header
    omitted = 0
    for index, item in enumerate(briefs):
        section = _morning_section(
            str(item.get("symbol") or ""),
            item.get("memberships") or [],
            item.get("result") or {},
        ) + "\n"
        if len((text + section).encode("utf-8")) > MAX_MORNING_BRIEF_BYTES:
            omitted = len(briefs) - index
            break
        text += section
    if omitted:
        text += f"{omitted} additional ticker brief(s) omitted from this small Drive file; see ai_store/briefs.\n"
    return text


def _ensure_inference_window(now: datetime | None = None) -> None:
    from ai_jobs import window

    block = window.market_session_block(now)
    if block:
        raise RuntimeError(f"ticker briefs refused: {block}")
    if not window.in_offhours_window(now):
        raise RuntimeError("ticker briefs refused: outside the configured off-hours window")


def run_ticker_briefs(
    *,
    session_date: str,
    now: datetime | None = None,
    scopes: tuple[str, ...] = TICKER_BRIEF_SCOPES,
    watchlist_paths: Mapping[str, Path] | None = None,
    output_root: Path | None = None,
    morning_path: Path | None = None,
) -> dict[str, Any]:
    """Publish validated medium-tier briefs, then one bounded Drive file.

    The runner is the sole caller/writer. The gate is repeated here, including
    before every model call, so a direct invocation or a long ticker batch can
    never infer during RTH even if it started legitimately overnight.
    """
    import ai_summary
    from ai_jobs import ledger, store

    _ensure_inference_window(now)
    symbols, membership_by_symbol = load_brief_symbols(watchlist_paths)
    root = Path(output_root) if output_root is not None else store.briefs_dir()
    if not symbols:
        content = render_morning_file(session_date, [], generated_at=now)
        published = atomic_publish_morning_file(content, path=morning_path)
        return {
            "status": ledger.STATUS_OK,
            "model": "",
            "reason": f"no Focus/watchlist tickers for {session_date}",
            "outputs": [str(published)],
        }
    if not ai_summary.local_provider_enabled():
        raise RuntimeError("local AI provider is not configured; ticker briefs cannot run")

    model = ai_summary.local_model("medium")
    base = ai_summary.build_evidence_package(list(scopes), session_date=session_date)
    completed: list[dict[str, Any]] = []
    outputs: list[str] = []
    for symbol in symbols:
        # ``now`` proves the injected launch moment; this live re-check is what
        # stops a long batch before its next model call if the window closes.
        _ensure_inference_window()
        evidence = build_ticker_evidence(base, symbol, membership_by_symbol[symbol])
        result = ai_summary.request_ai_summary(
            provider="local",
            model=model,
            api_key="",
            evidence=evidence,
            timeout_seconds=900,
        )
        result = dict(result)
        result["summary"] = ai_summary.merge_coverage_into_summary(
            result.get("summary") or {}, evidence
        )
        target = root / session_date[:4] / session_date / "tickers" / symbol
        exported = ai_summary.export_ai_summary(result, evidence, output_dir=target)
        outputs.extend(str(path) for path in exported.values())
        completed.append(
            {"symbol": symbol, "memberships": membership_by_symbol[symbol], "result": result}
        )

    # Drive sees only this bounded distillation, and only after every ticker
    # completed. Any exception above leaves the prior verified morning intact.
    content = render_morning_file(session_date, completed, generated_at=now)
    published = atomic_publish_morning_file(content, path=morning_path)
    outputs.append(str(published))
    return {
        "status": ledger.STATUS_OK,
        "model": model,
        "reason": f"{len(completed)} per-ticker brief(s) for {session_date}",
        "outputs": outputs,
        "tokens": {"ticker_calls": len(completed)},
    }
