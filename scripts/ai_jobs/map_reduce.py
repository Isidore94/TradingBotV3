"""Read the whole evidence pile in slices, then synthesize -- trading time for context.

WHY THIS EXISTS
---------------
The single-shot local summary is bounded by one number it cannot argue with: the
model's context window. On this desk that is 65,536 tokens, and 96k crashes the
runner under load, so the prompt ceiling is real hardware and not a setting. The
evidence for one session is **1,365,259 characters -- about 683,000 tokens**. A
single prompt therefore carries roughly a **tenth** of it, and the packager
spends that tenth fairly rather than well: on 2026-08-27 `setups.type_stats`
contributed **3 of its 184 rows** and `setups.playbooks` 2 of 200.

The trader's framing (2026-08-28) is the fix: *"Can we just give it more time?
Like hours to complete its work then? And spoon feed it slowly so we don't run
out of context?"* The overnight window is 22:00-06:00 and the single-shot
summary uses nine minutes of it.

So: cut the evidence into chunks that fit comfortably, ask the model for
findings from each one, then hand it back only the findings and ask it to
synthesize. Nothing is truncated away -- **every row of every source is read**,
just not all at once.

WHAT THIS IS CAREFUL ABOUT
--------------------------
* **A chunk never pretends to be its whole source.** Every chunk carries a
  label - ``rows 41-80 of 184`` - inside the content the model reads, so a
  finding drawn from a slice cannot be phrased as a finding about the whole.
* **Citations stay real.** A map call is handed a package containing exactly one
  source, so the existing validator already forbids it citing anything else. The
  reduce call is handed the findings plus ``citable_aliases`` for the source ids
  that appear in them, so it can only cite sources that were genuinely read.
* **A failed chunk is counted and named**, never quietly skipped. The published
  ``data_quality`` says how many chunks over how many sources were read and what
  was lost, because a document synthesized from 30 of 34 chunks is not the same
  document as one synthesized from all 34.
* **A failed synthesis does not throw away hours of work.** The map findings are
  already validated and already carry real citations; if the reduce call fails
  they are published unsynthesized, and said to be unsynthesized.

This module calls no detector and writes no store. It is advisory output, like
everything else under `ai_jobs`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

_log = logging.getLogger(__name__)

#: Target characters per chunk. ~20,000 tokens at the measured 2.06 chars/token,
#: which is under a third of the 62,036 usable tokens at a 65,536 context - room
#: for the instructions, the schema and the answer with margin to spare. Small
#: chunks also keep any single failure cheap.
DEFAULT_CHUNK_CHARS = 40_000
CHUNK_CHARS_SETTING_KEY = "ai_local_map_chunk_chars"

#: Off until switched on, so this path cannot change a night by accident.
MAP_REDUCE_SETTING_KEY = "ai_local_map_reduce"

#: The reduce step's single source id. Named like the evidence ids it sits
#: beside so a reader of the finished document sees where the text came from.
FINDINGS_SOURCE_ID = "analysis.chunk_findings"

#: A synthesis that cannot run publishes the map findings as they are. More than
#: this many per section is a wall of text rather than a summary, so the highest
#: confidence survive and the count of what was dropped is stated.
MAX_UNSYNTHESIZED_ROWS_PER_SECTION = 12

_CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}


@dataclass
class Chunk:
    """One slice of one source, with the label that keeps it honest."""

    source_id: str
    index: int
    of: int
    label: str
    content: Any
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.source_id} [{self.index}/{self.of}]"


def map_reduce_enabled(get_setting: Callable[..., Any] | None = None) -> bool:
    """Whether the summary reads its evidence in slices.

    Fails safe to OFF. If the setting cannot be read at all, the answer is the
    single-shot path -- the one that has run every night for weeks -- rather
    than an exception out of a helper, because "we could not tell" is not a
    reason to lose a night's summary.
    """
    try:
        import ai_summary

        getter = get_setting or ai_summary.get_local_setting
        return bool(getter(MAP_REDUCE_SETTING_KEY, False))
    except Exception:  # pragma: no cover - defensive; see the docstring
        _log.warning("map-reduce setting unreadable; using the single-shot summary path")
        return False


def chunk_chars(get_setting: Callable[..., Any] | None = None) -> int:
    import ai_summary

    getter = get_setting or ai_summary.get_local_setting
    raw = getter(CHUNK_CHARS_SETTING_KEY, DEFAULT_CHUNK_CHARS)
    if isinstance(raw, bool):
        return DEFAULT_CHUNK_CHARS
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_CHUNK_CHARS
    if value <= 0:
        return DEFAULT_CHUNK_CHARS
    # Never larger than one prompt can hold, whatever anyone configures: a chunk
    # over the ceiling is sheared exactly like an unchunked prompt, and it would
    # be sheared once per chunk.
    return min(value, ai_summary.local_evidence_budget_ceiling_chars())


def _encoded(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)


def plan_chunks(evidence: Mapping[str, Any], *, chars: int) -> list[Chunk]:
    """Cut every usable source into chunks that fit one prompt.

    Tabular sources split by ROWS, because a row is the unit a reader of a
    performance table reasons about and half a row is not evidence. Text splits
    by character window. A source that already fits is one chunk and is labelled
    as complete, so the model is told the difference between "all of it" and
    "part of it".
    """
    import ai_summary

    usable = ai_summary.usable_source_ids(evidence)
    chunks: list[Chunk] = []
    for source in evidence.get("sources") or []:
        if not isinstance(source, Mapping):
            continue
        source_id = str(source.get("source_id") or "")
        if source_id not in usable:
            continue
        content = source.get("content")
        meta = {
            key: source.get(key)
            for key in ("label", "status", "observed_at", "content_through", "session_date")
            if source.get(key) is not None
        }
        pieces = _split(content, chars)
        total = len(pieces)
        for index, (piece, label) in enumerate(pieces, start=1):
            full = total == 1
            chunks.append(
                Chunk(
                    source_id=source_id,
                    index=index,
                    of=total,
                    label="the complete source" if full else label,
                    content=piece,
                    meta=meta,
                )
            )
    return chunks


def _split(content: Any, chars: int) -> list[tuple[Any, str]]:
    if isinstance(content, list):
        return _split_rows(content, chars)
    encoded = _encoded(content)
    if len(encoded) <= chars:
        return [(content, "the complete source")]
    out: list[tuple[Any, str]] = []
    for start in range(0, len(encoded), chars):
        end = min(start + chars, len(encoded))
        out.append((encoded[start:end], f"characters {start + 1}-{end} of {len(encoded)}"))
    return out


def _split_rows(rows: Sequence[Any], chars: int) -> list[tuple[Any, str]]:
    total = len(rows)
    if total == 0:
        return [([], "the complete source")]
    if len(_encoded(list(rows))) <= chars:
        return [(list(rows), "the complete source")]
    out: list[tuple[Any, str]] = []
    current: list[Any] = []
    start_row = 1
    for position, row in enumerate(rows, start=1):
        candidate = current + [row]
        if current and len(_encoded(candidate)) > chars:
            out.append((list(current), f"rows {start_row}-{position - 1} of {total}"))
            current = [row]
            start_row = position
        else:
            current = candidate
    if current:
        out.append((list(current), f"rows {start_row}-{total} of {total}"))
    return out


def chunk_package(chunk: Chunk, base: Mapping[str, Any]) -> dict[str, Any]:
    """An evidence package holding ONE chunk and nothing else.

    Reusing the package SHAPE means the existing validator applies unchanged: a
    map call may cite this source and no other, because no other is present.
    """
    encoded = _encoded(chunk.content).encode("utf-8")
    source = {
        "source_id": chunk.source_id,
        "label": f"{chunk.meta.get('label') or chunk.source_id} - {chunk.label}",
        "status": "available",
        "observed_at": chunk.meta.get("observed_at"),
        "content_through": chunk.meta.get("content_through"),
        "session_date": chunk.meta.get("session_date") or base.get("session_date"),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "truncated": chunk.of > 1,
        "content": chunk.content,
    }
    package = {
        "schema_version": "ai_evidence_package_v2",
        "generated_at": base.get("generated_at"),
        "session_date": base.get("session_date"),
        "selected_scopes": ["map_chunk"],
        "scope_labels": [f"One slice of {chunk.source_id}"],
        "source_count": 1,
        "sources": [source],
        "coverage": {
            "counts": {"requested": 1, "usable": 1, "stale": 0, "truncated": int(chunk.of > 1)},
            "note": (
                f"This is {chunk.label} of '{chunk.source_id}'"
                + (
                    f" (slice {chunk.index} of {chunk.of}). Other slices are being read "
                    "separately, so describe ONLY what is in front of you and never "
                    "characterise the source as a whole."
                    if chunk.of > 1
                    else ". This is the entire source."
                )
                + f" Cite '{chunk.source_id}' and nothing else."
            ),
        },
        "safety_contract": {
            "purpose": "advisory findings from one slice of one evidence source",
            "forbidden_effects": ["scanner scores", "watchlists", "alerts", "bot state", "orders"],
        },
        "scope_caveats": [
            "Report only what this slice supports. An empty list is a valid answer.",
        ],
    }
    canonical = json.dumps(package, sort_keys=True, separators=(",", ":"), default=str)
    package["evidence_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    package["package_id"] = package["evidence_hash"][:16]
    return package


def findings_package(
    findings: Mapping[str, list[dict[str, Any]]],
    base: Mapping[str, Any],
    *,
    read: int,
    planned: int,
    failed: Sequence[str],
) -> dict[str, Any]:
    """The reduce step's package: the collected findings, and what produced them.

    ``citable_aliases`` carries every source id that actually appears in the
    findings, so the synthesis can attribute a statement to the store it came
    from rather than to this intermediate document -- and can still cite nothing
    that was not read.
    """
    import ai_summary

    aliases = sorted({
        str(ref)
        for rows in findings.values()
        for row in rows
        for ref in (row.get("evidence_refs") or [])
        if str(ref).strip()
    })
    content = {
        "note": (
            "These findings were produced by reading the evidence in "
            f"{planned} slice(s) across the session's sources. Each finding "
            "already carries the source it came from. Synthesize them into one "
            "review: merge duplicates, keep the specific over the general, and "
            "carry every statement's citations through."
        ),
        "slices_planned": planned,
        "slices_read": read,
        "slices_failed": list(failed),
        "findings": {section: list(rows) for section, rows in findings.items()},
    }
    encoded = json.dumps(content, sort_keys=True, default=str).encode("utf-8")
    source = {
        "source_id": FINDINGS_SOURCE_ID,
        "label": f"Findings from {read} evidence slice(s)",
        "status": "available",
        "observed_at": base.get("generated_at"),
        "session_date": base.get("session_date"),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "truncated": False,
        "content": content,
    }
    package = {
        "schema_version": "ai_evidence_package_v2",
        "generated_at": base.get("generated_at"),
        "session_date": base.get("session_date"),
        "selected_scopes": ["map_reduce_synthesis"],
        "scope_labels": ["Findings collected from every evidence slice"],
        "source_count": 1,
        "sources": [source],
        "citable_aliases": aliases,
        "coverage": {
            "counts": {"requested": 1, "usable": 1, "stale": 0, "truncated": 0},
            "note": (
                "You are reading findings, not raw evidence. Do not compute new "
                f"numbers. Cite '{FINDINGS_SOURCE_ID}' or the exact source id a "
                "finding already carries"
                + (f" ({', '.join(aliases)})" if aliases else "")
                + ". Cite nothing else."
            ),
        },
        "safety_contract": {
            "purpose": "advisory synthesis of findings already drawn from evidence",
            "forbidden_effects": ["scanner scores", "watchlists", "alerts", "bot state", "orders"],
        },
        "scope_caveats": [
            "Every figure came from one session's discovery. Do not describe it "
            "as a trend or a confirmation.",
        ],
    }
    canonical = json.dumps(package, sort_keys=True, separators=(",", ":"), default=str)
    package["evidence_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    package["package_id"] = package["evidence_hash"][:16]
    _ = ai_summary  # imported for the settings/validator contract this shape relies on
    return package


def _merge_findings(collected: list[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    import ai_summary

    merged: dict[str, list[dict[str, Any]]] = {
        section: [] for section in ai_summary.MODEL_SUMMARY_SECTIONS
    }
    seen: set[tuple[str, str]] = set()
    for summary in collected:
        for section in ai_summary.MODEL_SUMMARY_SECTIONS:
            for row in summary.get(section) or []:
                statement = str(row.get("statement") or "").strip()
                if not statement:
                    continue
                key = (section, statement.lower())
                if key in seen:
                    continue
                seen.add(key)
                merged[section].append(dict(row))
    return merged


def unsynthesized_summary(
    findings: Mapping[str, list[dict[str, Any]]], *, read: int, planned: int
) -> dict[str, Any]:
    """Publish the map findings when synthesis fails, rather than losing them.

    They are already validated and already cite real sources. What they are NOT
    is a summary, and the executive line says so -- an unsynthesized pile
    presented as a review would be the more dishonest of the two failures.
    """
    import ai_summary

    out: dict[str, Any] = {
        "executive_summary": (
            f"UNSYNTHESIZED. Findings were drawn from {read} of {planned} evidence "
            "slice(s), but the synthesis pass did not complete, so what follows is "
            "the raw per-slice findings with duplicates removed rather than a "
            "review. Nothing here has been weighed against anything else."
        )
    }
    for section in ai_summary.MODEL_SUMMARY_SECTIONS:
        rows = sorted(
            findings.get(section) or [],
            key=lambda row: _CONFIDENCE_ORDER.get(str(row.get("confidence") or "low"), 3),
        )
        kept = rows[:MAX_UNSYNTHESIZED_ROWS_PER_SECTION]
        dropped = len(rows) - len(kept)
        if dropped > 0 and kept:
            kept = list(kept)
            kept.append(
                {
                    "statement": (
                        f"[{dropped} further finding(s) in this section are not shown; "
                        "the unsynthesized document keeps the highest-confidence "
                        f"{MAX_UNSYNTHESIZED_ROWS_PER_SECTION}]"
                    ),
                    "evidence_refs": [FINDINGS_SOURCE_ID],
                    "confidence": "high",
                }
            )
        out[section] = kept
    return out


def coverage_statement(*, planned: int, read: int, failed: Sequence[str], sources: int) -> str:
    """The one line that keeps a partial read from reading as a whole one."""
    text = (
        f"Read in slices: {read} of {planned} slice(s) across {sources} source(s) were "
        "read in full, so no source was reduced to a sample of its rows."
    )
    if failed:
        shown = ", ".join(list(failed)[:6])
        more = f", +{len(failed) - 6} more" if len(failed) > 6 else ""
        text += (
            f" {len(failed)} slice(s) FAILED and their evidence is absent from this "
            f"document: {shown}{more}."
        )
    return text


def run_map_reduce(
    *,
    evidence: Mapping[str, Any],
    model: str,
    timeout_seconds: int = 900,
    chars: int | None = None,
    request: Callable[..., Any] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Map every slice, then reduce. Returns a ``request_ai_summary``-shaped result."""
    import ai_summary

    call = request or ai_summary.request_ai_summary
    size = chars if chars is not None else chunk_chars()
    chunks = plan_chunks(evidence, chars=size)
    planned = len(chunks)
    started = time.time()

    collected: list[Mapping[str, Any]] = []
    failed: list[str] = []
    #: Which slices answered only after a length stop forced a shorter ask
    #: (packet N2). One entry per slice, never a bare flag: a night where six
    #: slices had to shrink is a different night from one where a single tail
    #: chunk did, and the ledger can say which.
    retried: list[dict[str, str]] = []
    for position, chunk in enumerate(chunks, start=1):
        if on_progress:
            on_progress(position, planned, chunk.name)
        try:
            result = call(
                provider="local",
                model=model,
                api_key="",
                evidence=chunk_package(chunk, evidence),
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:  # one slice must not cost the night
            failed.append(f"{chunk.name}: {type(exc).__name__}")
            _log.warning("map slice %s/%s (%s) failed: %s", position, planned, chunk.name, exc)
            continue
        summary = result.get("summary") if isinstance(result, Mapping) else None
        if isinstance(summary, Mapping):
            collected.append(summary)
        if isinstance(result, Mapping) and str(result.get("length_retry") or ""):
            retried.append({"slice": chunk.name, "retry": str(result["length_retry"])})

    read = len(collected)
    findings = _merge_findings(collected)
    sources = len({chunk.source_id for chunk in chunks})

    if read == 0:
        raise RuntimeError(
            f"every one of the {planned} evidence slice(s) failed; nothing was read"
        )

    package = findings_package(findings, evidence, read=read, planned=planned, failed=failed)
    synthesis_error = ""
    synthesis_stop_reason = ""
    synthesis_retry = ""
    try:
        reduced = call(
            provider="local",
            model=model,
            api_key="",
            evidence=package,
            timeout_seconds=timeout_seconds,
        )
        summary = reduced.get("summary") or {}
        usage = reduced.get("usage") or {}
        drops = list(reduced.get("citation_drops") or [])
        synthesis_retry = str(reduced.get("length_retry") or "")
    except Exception as exc:
        synthesis_error = f"{type(exc).__name__}: {exc}"
        # The stop reason travels on the exception rather than being parsed back
        # out of its text (`ai_summary.LocalOutputLengthError`); anything else
        # leaves it "", which is "not a stop" and NOT "not measured" -- the key
        # is always present, which is what makes the two readable apart.
        synthesis_stop_reason = str(getattr(exc, "stop_reason", "") or "")
        _log.warning("synthesis pass failed (%s); publishing the findings unsynthesized", exc)
        summary = unsynthesized_summary(findings, read=read, planned=planned)
        usage = {}
        drops = []

    return {
        "schema_version": "ai_summary_result_v1",
        "status": "validated",
        "provider": "local",
        "model": model,
        "response_id": "",
        "generated_at": ai_summary.datetime.now().astimezone().isoformat(timespec="seconds"),
        "duration_seconds": round(time.time() - started, 3),
        "evidence_package_id": evidence.get("package_id"),
        "evidence_hash": evidence.get("evidence_hash"),
        "usage": usage,
        "summary": summary,
        "citation_drops": drops,
        # Everything a reader needs to judge how complete this document is.
        "map_reduce": {
            "slices_planned": planned,
            "slices_read": read,
            "slices_failed": failed,
            "sources": sources,
            "chunk_chars": size,
            "synthesized": not synthesis_error,
            "synthesis_error": synthesis_error,
            # Packet N2, 2026-09-05. THREE keys, all always present, and they
            # answer three different questions a reader of a published document
            # has:
            #
            #   synthesis_stop_reason  why the model stopped on the LAST reduce
            #                          attempt. "length" means the answer was
            #                          cut by the output cap; "" means it was
            #                          not a stop. Never absent, so "" cannot be
            #                          confused with "this build did not look".
            #   synthesis_retry        "shorter" when the published synthesis is
            #                          the second, smaller answer; "" when the
            #                          first answer stood.
            #   slices_retried        one {slice, retry} row per MAP slice that
            #                          had to shrink. Empty on a clean night, so
            #                          the word "shorter" appears in this block
            #                          only when something actually did.
            #
            # An older manifest carries none of the three and still loads: every
            # reader added since reaches for them with .get().
            "synthesis_stop_reason": synthesis_stop_reason,
            "synthesis_retry": synthesis_retry,
            "slices_retried": retried,
            "coverage_statement": coverage_statement(
                planned=planned, read=read, failed=failed, sources=sources
            ),
        },
    }


__all__ = [
    "Chunk",
    "DEFAULT_CHUNK_CHARS",
    "FINDINGS_SOURCE_ID",
    "MAP_REDUCE_SETTING_KEY",
    "chunk_chars",
    "chunk_package",
    "coverage_statement",
    "findings_package",
    "map_reduce_enabled",
    "plan_chunks",
    "run_map_reduce",
    "unsynthesized_summary",
]
