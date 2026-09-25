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


#: P1-3 3c. Most slices one summary run reads; 0 or less means no cap.
MAX_SLICES_SETTING_KEY = "ai_summary_max_slices"
DEFAULT_MAX_SLICES = 24


def max_slices(get_setting: Callable[..., Any] | None = None) -> int:
    import ai_summary

    getter = get_setting or ai_summary.get_local_setting
    raw = getter(MAX_SLICES_SETTING_KEY, DEFAULT_MAX_SLICES)
    if isinstance(raw, bool):
        return DEFAULT_MAX_SLICES
    try:
        return int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_SLICES


def cap_chunks(chunks: Sequence["Chunk"], cap: int) -> tuple[list["Chunk"], int]:
    """Keep at most ``cap`` slices, one per source per round, in plan order.

    Returns (kept, number left out). Every source keeps its first slice before
    any source gets a second one.
    """
    items = list(chunks)
    if cap <= 0 or len(items) <= cap:
        return items, 0
    by_source: dict[str, list[int]] = {}
    for position, chunk in enumerate(items):
        by_source.setdefault(chunk.source_id, []).append(position)
    picked: set[int] = set()
    depth = 0
    while len(picked) < cap:
        for positions in by_source.values():
            if depth < len(positions) and len(picked) < cap:
                picked.add(positions[depth])
        depth += 1
    kept = [chunk for position, chunk in enumerate(items) if position in picked]
    return kept, len(items) - len(kept)


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


def package_chars(package: Mapping[str, Any]) -> int:
    """How big a package is, measured the way every budget in this file is."""
    return len(json.dumps(package, sort_keys=True, default=str))


def synthesis_budget_chars() -> int:
    """The character ceiling on the REDUCE package (TJ-13A item 3).

    The existing setting, not a new number: `evidence_budget_for("local",
    tier="medium")` is the same ceiling a single-shot summary spends, and the
    reduce call is one call with the whole context in front of it.

    Why it was needed. On 2026-09-17 all 53 map slices succeeded and then the
    synthesis call timed out - the reduce package was 119,677 characters
    against a local budget of 11,066. Every map slice is chunked to fit; the
    one call that has to hold the whole night was the only one nothing
    budgeted, so `completion=unsynthesized_fallback` became the normal ending
    (09-15, 09-16, 09-17, 09-18).

    Falls back to the module's chunk size if the setting cannot be read: a
    budget that raises would lose the night it is supposed to save.
    """
    try:
        import ai_summary

        value = int(ai_summary.evidence_budget_for("local", tier="medium"))
    except Exception:  # pragma: no cover - defensive; see the docstring
        _log.warning("local evidence budget unreadable; bounding the synthesis at the chunk size")
        return DEFAULT_CHUNK_CHARS
    return value if value > 0 else DEFAULT_CHUNK_CHARS


def _finding_rows(
    findings: Mapping[str, list[dict[str, Any]]]
) -> list[tuple[str, dict[str, Any]]]:
    """Every finding as ``(section, row)``, best-kept first.

    Highest confidence first, and within a confidence the order the slices
    produced them. The order is the DROP order when the package will not fit,
    so it is the one place a judgement is made about which findings the
    synthesis sees - and it is made on the model's own stated confidence, never
    on what a finding says.
    """
    rows: list[tuple[int, int, str, dict[str, Any]]] = []
    for section, section_rows in findings.items():
        for position, row in enumerate(section_rows or []):
            confidence = _CONFIDENCE_ORDER.get(str(row.get("confidence") or "low"), 3)
            rows.append((confidence, position, section, dict(row)))
    rows.sort(key=lambda item: (item[0], item[1]))
    return [(section, row) for _confidence, _position, section, row in rows]


def _regrouped(
    kept: Sequence[tuple[str, dict[str, Any]]],
    findings: Mapping[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Rebuild the section -> rows mapping from a kept subset, sections intact.

    Every section key stays PRESENT even when it kept nothing: an absent
    section reads as "this build did not look", and an empty one reads as
    "nothing survived", which are different nights.
    """
    out: dict[str, list[dict[str, Any]]] = {section: [] for section in findings}
    for section, row in kept:
        out.setdefault(section, []).append(row)
    return out


def bounded_findings_package(
    findings: Mapping[str, list[dict[str, Any]]],
    base: Mapping[str, Any],
    *,
    read: int,
    planned: int,
    failed: Sequence[str],
    budget: int,
) -> tuple[dict[str, Any], int]:
    """The reduce package, cut to ``budget``, plus how many findings it left out.

    Bounding is not silent truncation. The count comes back and is published
    beside the coverage line, because a synthesis over 30 of 74 findings is not
    the same document as one over all 74 and the reader has to be able to tell.

    The search is over a PREFIX of :func:`_finding_rows`, so the package is
    always the highest-confidence findings and never an arbitrary subset, and
    the package is rebuilt for the measurement rather than estimated - the
    skeleton, the aliases and the hash all cost characters.
    """
    whole = findings_package(findings, base, read=read, planned=planned, failed=failed)
    rows = _finding_rows(findings)
    if budget <= 0 or package_chars(whole) <= budget:
        return whole, 0

    # Binary search for the largest prefix that fits. `low` always fits (0
    # findings is the skeleton alone), `high` never does.
    low, high = 0, len(rows)
    best = findings_package(
        _regrouped([], findings), base, read=read, planned=planned, failed=failed
    )
    while low < high:
        middle = (low + high + 1) // 2
        candidate = findings_package(
            _regrouped(rows[:middle], findings),
            base,
            read=read,
            planned=planned,
            failed=failed,
        )
        if package_chars(candidate) <= budget:
            low, best = middle, candidate
        else:
            high = middle - 1
    dropped = len(rows) - low
    _log.warning(
        "synthesis package trimmed to fit %s chars: %s of %s finding(s) carried",
        budget,
        low,
        len(rows),
    )
    return best, dropped


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


def coverage_statement(
    *,
    planned: int,
    read: int,
    failed: Sequence[str],
    sources: int,
    findings_dropped: int = 0,
    capped: int = 0,
    cap: int = 0,
) -> str:
    """The one line that keeps a partial read from reading as a whole one."""
    if int(capped or 0) > 0:
        text = (
            f"Read in slices: {read} of {planned} slice(s) across {sources} source(s) "
            f"were read; {int(capped)} slice(s) were not read because one run reads at "
            f"most {int(cap)} (every source kept its first slice)."
        )
    else:
        text = (
            f"Read in slices: {read} of {planned} slice(s) across {sources} source(s) were "
            "read in full, so no source was reduced to a sample of its rows."
        )
    if int(findings_dropped or 0) > 0:
        # TJ-13A item 3. The synthesis prompt has a ceiling, and a night that
        # produced more findings than fit has to say so here rather than
        # present the remainder as the whole.
        text += (
            f" {int(findings_dropped)} finding(s) did not fit the synthesis prompt "
            "and were not carried into it; the highest-confidence findings were kept."
        )
    if failed:
        shown = ", ".join(list(failed)[:6])
        more = f", +{len(failed) - 6} more" if len(failed) > 6 else ""
        text += (
            f" {len(failed)} slice(s) FAILED and their evidence is absent from this "
            f"document: {shown}{more}."
        )
    return text


#: The completion vocabulary (WS-AI1 item 4). Four words, and every published
#: summary carries exactly one of them.
#:
#: * ``synthesized``            - every slice read and the reduce pass answered.
#: * ``partial``                - the reduce pass answered, over less evidence.
#: * ``unsynthesized_fallback`` - the reduce pass did not answer; the document
#:                                is the code's assembly of the surviving
#:                                findings, not the model's synthesis.
#: * ``failed``                 - no document was produced by this path at all.
COMPLETION_SYNTHESIZED = "synthesized"
COMPLETION_PARTIAL = "partial"
COMPLETION_UNSYNTHESIZED = "unsynthesized_fallback"
COMPLETION_FAILED = "failed"

COMPLETION_WORDS = (
    COMPLETION_SYNTHESIZED,
    COMPLETION_PARTIAL,
    COMPLETION_UNSYNTHESIZED,
    COMPLETION_FAILED,
)


def completion_word(*, synthesis_error: str, slices_failed: Sequence[str]) -> str:
    """One word for how complete a map-reduce document is."""
    if str(synthesis_error or "").strip():
        return COMPLETION_UNSYNTHESIZED
    if list(slices_failed or ()):
        return COMPLETION_PARTIAL
    return COMPLETION_SYNTHESIZED


def run_map_reduce(
    *,
    evidence: Mapping[str, Any],
    model: str,
    timeout_seconds: int = 900,
    chars: int | None = None,
    request: Callable[..., Any] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    slice_cap: int | None = None,
) -> dict[str, Any]:
    """Map every slice (up to the per-run cap), then reduce.

    Returns a ``request_ai_summary``-shaped result.
    """
    import ai_summary

    call = request or ai_summary.request_ai_summary
    size = chars if chars is not None else chunk_chars()
    cap = max_slices() if slice_cap is None else int(slice_cap)
    all_chunks = plan_chunks(evidence, chars=size)
    planned = len(all_chunks)
    chunks, capped = cap_chunks(all_chunks, cap)
    started = time.time()

    collected: list[Mapping[str, Any]] = []
    failed: list[str] = []
    #: Which slices answered only after a length stop forced a shorter ask
    #: (packet N2). One entry per slice, never a bare flag: a night where six
    #: slices had to shrink is a different night from one where a single tail
    #: chunk did, and the ledger can say which.
    retried: list[dict[str, str]] = []
    to_read = len(chunks)
    for position, chunk in enumerate(chunks, start=1):
        if on_progress:
            on_progress(position, to_read, chunk.name)
        try:
            result = call(
                provider="local",
                model=model,
                api_key="",
                evidence=chunk_package(chunk, evidence),
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:  # one slice must not cost the night
            # TJ-13A item 3. UNREACHABLE is the one exception to "one slice
            # must not cost the night", because it is not a slice failure at
            # all: the server is not there, it will not be there for the next
            # slice either, and each remaining slice costs its own full read
            # timeout to find that out. Measured on the live ledger: 53 slices
            # at a 900 s timeout is over thirteen hours of window spent
            # discovering what the first call already knew, and it ended
            # `degraded_no_narrative` on 09-15, 09-16, 09-17 and 09-18.
            #
            # This keys on the endpoint being unreachable and NEVER on any
            # failure: an ordinary bad answer still costs its own slice and
            # nothing else (`test_a_failed_slice_is_counted_and_named_never_
            # skipped_quietly`).
            if ai_summary.is_endpoint_unreachable(exc):
                raise RuntimeError(
                    f"the local AI endpoint was unreachable on slice {position} of "
                    f"{to_read}; giving up now rather than spending the window "
                    f"discovering it {to_read - position} more times: {exc}"
                ) from exc
            failed.append(f"{chunk.name}: {type(exc).__name__}")
            _log.warning("map slice %s/%s (%s) failed: %s", position, to_read, chunk.name, exc)
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

    # TJ-13A item 3: the reduce call gets a BOUNDED package. Everything else in
    # this module already fits one prompt by construction; this was the one
    # call that did not, and it is the call that has to hold the whole night.
    package, findings_dropped = bounded_findings_package(
        findings,
        evidence,
        read=read,
        planned=planned,
        failed=failed,
        budget=synthesis_budget_chars(),
    )
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
        # WS-AI1 item 4. The completion WORD, at the top level, always present.
        #
        # `status: "validated"` describes the document's shape and says nothing
        # about how complete it is, and the only thing that did was
        # `map_reduce.synthesized` - a boolean two levels down that `briefs.py`
        # did not read. So the 900 s synthesis timeouts on 2026-09-10 and -11
        # published an unsynthesized fallback and were ledgered `ok`, and the
        # trader had no way to tell those nights from clean ones.
        #
        # The order is deliberate: a lost synthesis outranks a lost slice,
        # because a fallback document is assembled by code from what survived
        # and a partial one is still the model's own synthesis of less.
        "completion": completion_word(
            synthesis_error=synthesis_error, slices_failed=failed
        ),
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
            # P1-3 3c: slices left out by the per-run cap, and the cap itself.
            "slices_capped": int(capped),
            "slice_cap": int(cap),
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
            # TJ-13A item 3. ALWAYS PRESENT, 0 on a night where everything
            # fit - so "0" cannot be confused with "this build did not look".
            # Bounding is not silent truncation: what the synthesis could not
            # carry is counted here and stated in the coverage line.
            "findings_dropped_to_fit": int(findings_dropped),
            "coverage_statement": coverage_statement(
                planned=planned,
                read=read,
                failed=failed,
                sources=sources,
                findings_dropped=findings_dropped,
                capped=capped,
                cap=cap,
            ),
        },
    }


__all__ = [
    "COMPLETION_FAILED",
    "COMPLETION_PARTIAL",
    "COMPLETION_SYNTHESIZED",
    "COMPLETION_UNSYNTHESIZED",
    "COMPLETION_WORDS",
    "Chunk",
    "DEFAULT_CHUNK_CHARS",
    "FINDINGS_SOURCE_ID",
    "MAP_REDUCE_SETTING_KEY",
    "bounded_findings_package",
    "chunk_chars",
    "completion_word",
    "chunk_package",
    "coverage_statement",
    "findings_package",
    "package_chars",
    "synthesis_budget_chars",
    "map_reduce_enabled",
    "plan_chunks",
    "run_map_reduce",
    "unsynthesized_summary",
]
