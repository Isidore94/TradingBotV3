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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

#: The scopes a hands-off nightly review should cover. Journal review is
#: deliberately included: it is the trader's own revealed preference and the
#: whole point of the nightly read.
#:
#: ``market_journal`` was added 2026-08-27 on the trader's explicit instruction
#: ("i also expect the AI to get access to these notes for the daily summary
#: function"). R10.I had made it opt-in and absent from this tuple; that was a
#: trader decision and this is the same trader reversing it, which is the only
#: thing that could. It carries the trader's own free text about the session
#: plus the machine-measured day context and the chart digests written beside
#: each entry - the half of the day nothing else in this slate can see.
DEFAULT_SCOPES = (
    "daily_report",
    "market_conditions",
    "setup_trackers",
    "journal_review",
    "market_journal",
)
#: Per-symbol packets stay the ORIGINAL four. `market_journal` joined the daily
#: summary, not this: a journal entry is about a session, and TB-0/TB-5 measured
#: what happens when session-level text rides into a per-symbol packet - 96.2%
#: of a brief was roster noise, starving the symbol-specific evidence it led.
#: The trader asked for "the daily summary function", and this is not it.
TICKER_BRIEF_SCOPES = ("daily_report", "market_conditions", "setup_trackers", "journal_review")
MORNING_BRIEF_FILENAME = "ai_morning_brief.txt"
MAX_TICKER_SOURCE_CHARS = 16_000
MAX_MORNING_BRIEF_BYTES = 48 * 1024
_SYMBOL_TOKEN = re.compile(r"^[A-Z][A-Z0-9.-]{0,14}$")

#: The one source every projected ticker package always carries, because the
#: code writes it rather than finding it. A package holding only this says the
#: session produced no evidence about the symbol at all (TB-2).
MEMBERSHIP_SOURCE_ID = "watchlists.membership"

#: Per-session, per-symbol completion record (TB-3). Append-only JSONL, in the
#: same shape as every other evidence ledger here: the newest row for a symbol
#: wins, and nothing is ever rewritten.
BRIEF_MANIFEST_FILENAME = "ticker_briefs_manifest.jsonl"
#: v2 adds ``resume_key``. A v1 row has none, and a row without one is never
#: reused -- an older manifest costs a regeneration, never a wrong skip.
BRIEF_MANIFEST_SCHEMA = "ai_ticker_brief_manifest_v2"

BRIEF_STATUS_BRIEFED = "briefed"
BRIEF_STATUS_MEMBERSHIP_ONLY = "membership_only"
BRIEF_STATUS_FAILED = "failed"
#: A symbol in one of these states needs no further work this session. A
#: ``failed`` symbol deliberately is not one: the next firing retries it.
RESOLVED_BRIEF_STATUSES = frozenset({BRIEF_STATUS_BRIEFED, BRIEF_STATUS_MEMBERSHIP_ONLY})

#: A failure reason is a header line, not a stack trace. Long provider errors
#: are cut here so the outcome stays readable on a phone.
MAX_FAILURE_REASON_CHARS = 160

#: Stamped on every interim publish and removed by the final one. A file
#: carrying it is a run that was still going when it was written -- which, if
#: the process is killed, is exactly the file the trader finds in the morning.
INCOMPLETE_RUN_NOTE = "Run in progress at the time of writing; counts above may be incomplete."

#: Attempts one session may spend on the ticker-briefs slot (TB-4). Transient
#: faults -- NAS asleep, endpoint still loading -- still self-heal; the
#: all-night grind of 11 consecutive failures does not survive.
TICKER_BRIEFS_MAX_ATTEMPTS = 3


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
    # Budgeted for the local context window, not the metered-cloud ceiling: the
    # server truncates an over-long prompt silently, which defeats the
    # packager's own honest degradation (see ai_summary's budget derivation).
    evidence = ai_summary.build_evidence_package(
        list(scopes),
        session_date=session_date,
        budget_chars=ai_summary.evidence_budget_for("local", tier="medium"),
    )
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
            # Real token counts when the provider reported them, so a later
            # reader can see how close the run ran to the context ceiling
            # instead of inferring it from a failure.
            "tokens": {
                "duration_seconds": result.get("duration_seconds"),
                **(result.get("usage") or {}),
            },
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


def _summed_usage(completed: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Token totals across one night's per-ticker calls; {} when unreported.

    Summed rather than averaged because the ledger question is "what did this
    slot cost in total", and omitted entirely when no call reported usage --
    a zero would claim the batch was free rather than admit it is unknown.
    """
    totals: dict[str, int] = {}
    for entry in completed:
        result = entry.get("result") if isinstance(entry, Mapping) else None
        usage = (result or {}).get("usage") if isinstance(result, Mapping) else None
        if not isinstance(usage, Mapping):
            continue
        for key, value in usage.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            totals[str(key)] = totals.get(str(key), 0) + int(value)
    return totals


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


@dataclass(frozen=True)
class WatchlistRead:
    """What the watchlist sources said, and which of them actually spoke.

    ``symbols`` alone cannot answer the only question the morning file needs
    answered: is an empty result a fact about the trader's lists, or a fact
    about a shared folder that did not mount? ``read`` and ``unreadable`` keep
    those two apart, so no caller has to guess.
    """

    #: Tickers in first-seen order (Focus first by the default mapping).
    symbols: list[str]
    #: Per-ticker, code-owned pointers naming every list that contained it.
    memberships: dict[str, list[dict[str, str]]]
    #: Names of the sources that were opened and parsed successfully.
    read: list[str]
    #: Names of the sources that could not be opened at all.
    unreadable: list[str]

    @property
    def is_trustworthy_empty(self) -> bool:
        """True only when every configured source was read and all were empty.

        An unreadable source anywhere makes an empty result uncertainty rather
        than a finding, no matter how many of its siblings read cleanly - the
        missing one is exactly where the names could have been.
        """
        return not self.symbols and bool(self.read) and not self.unreadable


def load_brief_symbols(paths: Mapping[str, Path] | None = None) -> WatchlistRead:
    """Read ticker membership without ever changing a list.

    An unreadable list is uncertainty: it contributes no names and is never
    rewritten, repaired, or treated as evidence that a name was removed. It is
    also *reported*, because silently folding it into "no tickers" is how a
    missing folder turns into a published claim that the trader watches nothing.
    """
    selected = paths if paths is not None else default_watchlist_paths()
    ordered: list[str] = []
    memberships: dict[str, list[dict[str, str]]] = {}
    read: list[str] = []
    unreadable: list[str] = []
    for list_name, raw_path in selected.items():
        path = Path(raw_path)
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            logging.warning("Ticker briefs: watchlist %s is unreadable at %s", list_name, path)
            unreadable.append(str(list_name))
            continue
        read.append(str(list_name))
        for line in lines:
            token = str(line or "").split("#", 1)[0].strip().lstrip("$").upper()
            if not _SYMBOL_TOKEN.fullmatch(token):
                continue
            if token not in memberships:
                memberships[token] = []
                ordered.append(token)
            memberships[token].append({"list": str(list_name), "path": str(path)})
    return WatchlistRead(
        symbols=ordered, memberships=memberships, read=read, unreadable=unreadable
    )


#: A projected line that is nothing but a list of tickers matched because the
#: symbol is one token in a copy-paste blob, not because the line says anything
#: about it (TB-5, 2026-08-12). Measured on the 2026-08-11 packages: 96.2% of
#: everything the model was sent -- 307,630 of 319,687 chars across 166 symbols
#: -- was roster text, and `daily.master_events` contributed 174,994 roster
#: chars against 479 chars of real content. The model duly spent its bullets on
#: the packaging ("investigate why the master events data is truncated") while
#: MDB's one real number, +15.56% over 2026-08-04..08-11, went unmentioned.
#:
#: The test is deliberately about *residue*, not ticker count: strip the ticker
#: tokens and list punctuation and see whether anything is left. A tier row like
#: "RBRK 2026-08-04->2026-08-11 (+19.68%); MDB ... (+15.56%)" carries eight
#: tickers and is pure signal, so a count threshold alone would have discarded
#: exactly the rows worth keeping.
ROSTER_MIN_TICKERS = 5
ROSTER_MAX_RESIDUE_RATIO = 0.15
_TICKER_TOKEN = re.compile(r"[A-Z][A-Z0-9.\-]{0,9}")
_LIST_PUNCTUATION = re.compile(r"[\s,;:\"'\[\]{}()]+")


def is_roster_line(line: str, symbol: str) -> bool:
    """True when the line is a ticker roster rather than a statement.

    ``symbol`` is accepted for symmetry with the rest of the projection and to
    keep the rule expressible per symbol; the residue test does not need it,
    because a roster reads the same whichever of its names was matched.
    """
    stripped = str(line).strip()
    if not stripped:
        return False
    if len(_TICKER_TOKEN.findall(stripped)) < ROSTER_MIN_TICKERS:
        return False
    residue = _LIST_PUNCTUATION.sub("", _TICKER_TOKEN.sub(" ", stripped))
    return len(residue) <= ROSTER_MAX_RESIDUE_RATIO * len(stripped)


def is_bare_membership_line(line: str, symbol: str) -> bool:
    """True when the line is the symbol and nothing else.

    ``"MDB"`` inside an Auto Pilot longs array is the same fact as watchlist
    membership wearing a different hat. Counting it as evidence is what let
    ``market.auto_state`` keep symbols out of TB-2's membership-only skip while
    telling the model nothing it did not already have.
    """
    stripped = _LIST_PUNCTUATION.sub("", str(line))
    return stripped.upper() == str(symbol).strip().upper()


def _extract_ticker_content(content: Any, symbol: str) -> Any | None:
    """Bounded, deterministic symbol projection from one packaged source."""
    if isinstance(content, str):
        pattern = re.compile(rf"(?<![A-Z0-9.-]){re.escape(symbol)}(?![A-Z0-9.-])", re.I)
        lines = [
            line
            for line in content.splitlines()
            if pattern.search(line)
            and not is_roster_line(line, symbol)
            and not is_bare_membership_line(line, symbol)
        ]
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
    *,
    budget_chars: int | None = None,
) -> dict[str, Any]:
    """Project the nightly package into a single validated ticker package.

    ``budget_chars`` rations the *projected* package, which is the only place
    the rationing can be done without starving the projection itself (TB-0,
    2026-08-11). The base package used to be budgeted to the local ceiling
    first, so the per-symbol-rich sources -- a 95,806-char setup tracker, a
    77,124-char tier table -- arrived here already declared unfunded or sheared
    to a single row. Every one of the first night's 95 briefs was therefore
    projected out of a package that no longer contained the symbol: MRVL's
    coverage read "1 of 19 requested source(s) usable", and the one was its own
    watchlist membership. Projecting from a full-size base and budgeting the
    much smaller per-symbol result puts the symbol's rows back in front of the
    model while the local context window is still respected.

    ``None`` leaves the projection unrationed, for callers that budget it
    themselves.
    """
    import ai_summary

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
        "source_id": MEMBERSHIP_SOURCE_ID,
        "label": "Focus/watchlist membership",
        "status": "available",
        "as_of": str(base.get("generated_at") or ""),
        "source_session": str(base.get("session_date") or ""),
        "content": {"symbol": symbol, "memberships": [dict(row) for row in memberships]},
        "evidence_pointer": {
            "source_id": MEMBERSHIP_SOURCE_ID,
            "paths": [str(row.get("path") or "") for row in memberships],
        },
    }
    sources.insert(0, membership_source)

    unfunded: list[dict[str, str]] = []
    if budget_chars is not None:
        sources, unfunded = ai_summary.ration_projected_sources(
            sources, total=int(budget_chars)
        )

    base_coverage = base.get("coverage") if isinstance(base.get("coverage"), Mapping) else {}
    excluded = [
        dict(row) for row in base_coverage.get("excluded") or [] if isinstance(row, Mapping)
    ] + not_applicable + unfunded
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
    package["resume_key"] = _resume_key(symbol, package["session_date"], memberships, sources)
    canonical = json.dumps(package, sort_keys=True, separators=(",", ":"), default=str)
    package["evidence_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    package["package_id"] = package["evidence_hash"][:16]
    return package


def _resume_key(
    symbol: str,
    session_date: str,
    memberships: Sequence[Mapping[str, str]],
    sources: Sequence[Mapping[str, Any]],
) -> str:
    """Identity of the *evidence*, ignoring when it happened to be read.

    ``evidence_hash`` covers the whole package, and the package carries
    ``generated_at`` plus every source's ``as_of`` read stamp. Those move on
    every firing, so identical evidence hashed differently every time and TB-3's
    resume could never fire on the desk -- proven live on 2026-08-11, when a
    second runner instance re-briefed the first 25 symbols from the top and left
    25 duplicate artifact sets on the DAS.

    Only what would change the model's answer belongs here: which symbol, which
    session, which lists it sits on, and the source ids with their content.
    ``evidence_hash`` keeps its whole-package meaning for artifact identity.
    """
    payload = {
        "symbol": str(symbol),
        "session_date": str(session_date),
        "memberships": sorted(str(row.get("list") or "") for row in memberships),
        "sources": [
            {
                "source_id": str(source.get("source_id") or ""),
                "content": source.get("content"),
            }
            for source in sources
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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


def _membership_names(memberships: Sequence[Mapping[str, str]]) -> str:
    return ", ".join(str(row.get("list") or "") for row in memberships)


def _morning_section(entry: Mapping[str, Any]) -> str:
    """One symbol's block, from a manifest entry."""
    symbol = str(entry.get("symbol") or "")
    memberships = entry.get("memberships") or []
    lists = _membership_names(memberships)
    heading = f"## {symbol}  [{lists}]"
    if str(entry.get("status") or "") == BRIEF_STATUS_MEMBERSHIP_ONLY:
        # Deterministic, model-free, and deliberately one line: there was
        # nothing to narrate, and a paraphrase of "it is on a list" is the
        # class of output least likely to say anything (TB-2).
        return f"{heading}\n{entry.get('reason') or 'no session evidence beyond membership'}\n"
    result = entry.get("result") if isinstance(entry.get("result"), Mapping) else {}
    summary = result.get("summary") if isinstance(result.get("summary"), Mapping) else {}
    lines = [heading, str(summary.get("executive_summary") or "No supported finding.")]
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


def _failure_clause(failures: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for entry in failures:
        reason = str(entry.get("reason") or "unknown error").replace("\n", " ").strip()
        if len(reason) > MAX_FAILURE_REASON_CHARS:
            reason = reason[: MAX_FAILURE_REASON_CHARS - 1].rstrip() + "…"
        parts.append(f"{entry.get('symbol')} ({reason})")
    return " Failed: " + ", ".join(parts) + "." if parts else ""


def render_morning_file(
    session_date: str,
    briefs: Sequence[Mapping[str, Any]],
    *,
    generated_at: datetime | None = None,
    failures: Sequence[Mapping[str, Any]] = (),
    total: int | None = None,
    notes: Sequence[str] = (),
) -> str:
    """The bounded home-folder file, outcome first.

    A night that briefed 94 of 95 names used to publish nothing at all. It now
    publishes what it has, and says so in the header before any brief, so a
    partial file can never be read as a complete one (TB-1).
    """
    generated = (generated_at or datetime.now().astimezone()).isoformat(timespec="seconds")
    resolved = len(briefs)
    requested = int(total) if total is not None else resolved + len(failures)
    outcome = f"Briefed {resolved} of {requested}.{_failure_clause(failures)}"
    note_lines = "".join(f"{str(note).strip()}\n" for note in notes if str(note).strip())
    header = (
        "# LOCAL-AI MORNING TICKER BRIEFS — ADVISORY ONLY\n"
        f"Session reviewed: {session_date}\nGenerated: {generated}\n"
        f"{outcome}\n"
        f"{note_lines}"
        "This file cannot change scanners, scores, watchlists, alerts, or bot state.\n\n"
    )
    text = header
    omitted = 0
    for index, item in enumerate(briefs):
        section = _morning_section(item) + "\n"
        if len((text + section).encode("utf-8")) > MAX_MORNING_BRIEF_BYTES:
            omitted = len(briefs) - index
            break
        text += section
    if omitted:
        text += f"{omitted} additional ticker brief(s) omitted from this small summary file; see ai_store/briefs.\n"
    return text


def _inference_stop(now: datetime | None = None) -> tuple[str, str]:
    """``("", "")`` while inference may proceed, else ``(kind, reason)``.

    The two gates are kept apart because their consequences differ once a batch
    is already running: the market session is an unconditional stop, while the
    off-hours window closing stops further inference but still publishes what
    completed (TB-1). Neither gate's own semantics change here.
    """
    from ai_jobs import window

    block = window.market_session_block(now)
    if block:
        return "market_session", block
    if not window.in_offhours_window(now):
        return "window_closed", "the configured off-hours window closed"
    return "", ""


def _ensure_inference_window(now: datetime | None = None) -> None:
    kind, reason = _inference_stop(now)
    if kind == "market_session":
        raise RuntimeError(f"ticker briefs refused: {reason}")
    if kind:
        raise RuntimeError("ticker briefs refused: outside the configured off-hours window")


def brief_manifest_path(root: Path, session_date: str) -> Path:
    """Per-session manifest, beside the session's per-ticker artifact tree."""
    return Path(root) / session_date[:4] / session_date / BRIEF_MANIFEST_FILENAME


def read_brief_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Latest recorded state per symbol; ``{}`` when there is nothing to read.

    An unreadable or half-written manifest is not an error: it means the
    session has no recorded completions, so every symbol is simply regenerated.
    Refusing the night over it would trade a cheap repeat for no brief at all.
    """
    from diagnostics.artifact_io import read_jsonl

    try:
        rows = read_jsonl(Path(path))
    except (OSError, ValueError):
        logging.warning("Ticker briefs: manifest at %s is unreadable; regenerating all.", path)
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol:
            latest[symbol] = dict(row)  # append-only: the newest row wins
    return latest


def append_brief_manifest(path: Path, entry: Mapping[str, Any]) -> None:
    from diagnostics.artifact_io import append_jsonl_rows

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl_rows(target, (dict(entry),), fsync=True)


def _manifest_entry(
    *,
    symbol: str,
    memberships: Sequence[Mapping[str, str]],
    session_date: str,
    evidence_hash: str,
    resume_key: str = "",
    status: str,
    reason: str = "",
    result: Mapping[str, Any] | None = None,
    outputs: Sequence[str] = (),
) -> dict[str, Any]:
    entry = {
        "schema": BRIEF_MANIFEST_SCHEMA,
        "session_date": str(session_date),
        "symbol": str(symbol),
        "status": str(status),
        "evidence_hash": str(evidence_hash),
        "resume_key": str(resume_key),
        "reason": str(reason),
        "memberships": [dict(row) for row in memberships],
        "outputs": [str(value) for value in outputs],
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if result is not None:
        # Enough to re-render the morning file and to total the night's usage
        # without re-reading four artifact files per symbol.
        entry["result"] = {
            "model": str(result.get("model") or ""),
            "summary": result.get("summary") or {},
            "usage": dict(result.get("usage") or {}),
            "duration_seconds": result.get("duration_seconds"),
        }
    return entry


def _usable_source_ids(evidence: Mapping[str, Any]) -> list[str]:
    return [
        str(source.get("source_id") or "")
        for source in evidence.get("sources") or []
        if isinstance(source, Mapping)
    ]


def is_membership_only(evidence: Mapping[str, Any]) -> bool:
    """True when the projection found nothing about the symbol (TB-2)."""
    return {name for name in _usable_source_ids(evidence) if name} <= {MEMBERSHIP_SOURCE_ID}


def run_ticker_briefs(
    *,
    session_date: str,
    now: datetime | None = None,
    scopes: tuple[str, ...] = TICKER_BRIEF_SCOPES,
    watchlist_paths: Mapping[str, Path] | None = None,
    output_root: Path | None = None,
    morning_path: Path | None = None,
) -> dict[str, Any]:
    """Publish validated medium-tier briefs, then one bounded home-folder file.

    The runner is the sole caller/writer. The gate is repeated here, including
    before every model call, so a direct invocation or a long ticker batch can
    never infer during RTH even if it started legitimately overnight.

    Publishing is refused outright (``skipped``, nothing written) when no
    ticker was read and the watchlist sources cannot be trusted to be empty.
    ``skipped`` is not a canonical completion, so the next firing in the window
    retries once the sources are back.

    Every symbol is resolved independently (TB-1..TB-3): one failure costs its
    own brief and nothing else, a symbol carrying no evidence beyond its
    watchlist membership is answered without a model call, and completions are
    recorded per symbol so a retry regenerates only what actually changed. The
    morning file is re-rendered from that record on every firing and states the
    outcome before the first brief, so ``ok`` means every symbol resolved and
    anything less is ``degraded`` -- which the runner retries.
    """
    import ai_summary
    from ai_jobs import ledger, store

    _ensure_inference_window(now)
    watchlists = load_brief_symbols(watchlist_paths)
    symbols = watchlists.symbols
    membership_by_symbol = watchlists.memberships
    root = Path(output_root) if output_root is not None else store.briefs_dir()
    if not symbols:
        # "No tickers" is a real, publishable finding only when every source
        # was actually read. If any source was unreadable - or there was no
        # source to read - the emptiness came from missing data, and plan.md
        # sec 5 is explicit that missing data is uncertainty, never
        # confirmation. Publishing here would overwrite the last verified
        # morning file with a claim derived from a folder that did not mount,
        # so this refuses instead and leaves that file exactly where it is.
        if not watchlists.is_trustworthy_empty:
            detail = (
                "unreadable: " + ", ".join(watchlists.unreadable)
                if watchlists.unreadable
                else "no watchlist source was configured"
            )
            reason = (
                f"refused to publish the morning file for {session_date}: no ticker "
                f"was read and the sources cannot be trusted to be empty ({detail}); "
                "the last verified morning file is left untouched"
            )
            logging.warning("Ticker briefs: %s", reason)
            return {
                "status": ledger.STATUS_SKIPPED,
                "model": "",
                "reason": reason,
                "outputs": [],
            }
        content = render_morning_file(session_date, [], generated_at=now, total=0)
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
    # Project first, budget second (TB-0). The base carries the cloud ceiling so
    # the per-symbol-rich sources survive long enough to be projected; the local
    # ceiling is then applied to each much smaller per-symbol package, where it
    # bounds what the model is actually sent.
    base = ai_summary.build_evidence_package(
        list(scopes),
        session_date=session_date,
        budget_chars=ai_summary.MAX_TOTAL_EVIDENCE_CHARS,
    )
    # per_item: one of 50-120 calls in one window, so it is sized by the length
    # of the night rather than by the context. See evidence_budget_for.
    ticker_budget = ai_summary.evidence_budget_for("local", tier="medium", per_item=True)

    manifest = brief_manifest_path(root, session_date)
    recorded = read_brief_manifest(manifest)
    entries: dict[str, dict[str, Any]] = {}
    fresh: list[dict[str, Any]] = []
    outputs: list[str] = []
    calls = 0
    reused = 0
    early_stop = ""

    def _publish_progress() -> None:
        """Re-render and republish the morning file from what is resolved now.

        The morning file used to be written once, after the loop. On
        2026-08-11 the desk entered Modern Standby mid-batch and the process
        died at symbol 101 of 182: 126 briefs existed on the DAS and the home
        folder still held the previous session's file, because the publish had
        never been reached. Publishing after every resolution costs a few
        milliseconds against a ~70 s model call and makes a hard kill lose at
        most the symbol in flight. A publish fault is swallowed here for the
        same reason the final publish is verified-and-atomic: the last good
        file is worth more than this iteration's.
        """
        from ai_jobs import window as _window

        if _window.market_session_block():
            # The market session is an unconditional stop for the whole job,
            # publication included. Once it is live the run stops touching the
            # home folder and the last verified morning file stands; the
            # completed briefs are already in the manifest for the next
            # legitimate firing to re-render.
            return
        try:
            done = [
                entries[name]
                for name in symbols
                if name in entries
                and str(entries[name].get("status") or "") in RESOLVED_BRIEF_STATUSES
            ]
            failed_now = [
                entries[name]
                for name in symbols
                if name in entries
                and str(entries[name].get("status") or "") == BRIEF_STATUS_FAILED
            ]
            atomic_publish_morning_file(
                render_morning_file(
                    session_date,
                    done,
                    generated_at=now,
                    failures=failed_now,
                    total=len(symbols),
                    notes=[INCOMPLETE_RUN_NOTE],
                ),
                path=morning_path,
            )
        except Exception:  # never let publishing cost the night's inference
            logging.exception("Ticker briefs: interim morning-file publish failed")

    for symbol in symbols:
        memberships = membership_by_symbol[symbol]
        evidence = build_ticker_evidence(
            base, symbol, memberships, budget_chars=ticker_budget
        )
        evidence_hash = str(evidence.get("evidence_hash") or "")
        resume_key = str(evidence.get("resume_key") or "")

        prior = recorded.get(symbol) or {}
        if (
            str(prior.get("status") or "") in RESOLVED_BRIEF_STATUSES
            and str(prior.get("resume_key") or "") == resume_key
            and resume_key
        ):
            # Same session, same symbol, same evidence: the brief that exists
            # is the brief this call would produce. Regenerating it would cost
            # a minute of inference and leave a duplicate artifact set (TB-3).
            entries[symbol] = dict(prior)
            reused += 1
            continue

        if is_membership_only(evidence):
            reason = (
                "no session evidence beyond membership in "
                + (_membership_names(memberships) or "no list")
            )
            entry = _manifest_entry(
                symbol=symbol,
                memberships=memberships,
                session_date=session_date,
                evidence_hash=evidence_hash,
                resume_key=resume_key,
                status=BRIEF_STATUS_MEMBERSHIP_ONLY,
                reason=reason,
            )
            append_brief_manifest(manifest, entry)
            entries[symbol] = entry
            _publish_progress()
            logging.info("Ticker briefs: %s skipped without a model call (%s)", symbol, reason)
            continue

        # ``now`` proves the injected launch moment; this live re-check is what
        # stops a long batch before its next model call. The market session is
        # an unconditional stop and raises; the window closing stops inference
        # but still publishes what completed (TB-1).
        stop_kind, stop_reason = _inference_stop()
        if stop_kind == "market_session":
            raise RuntimeError(f"ticker briefs refused: {stop_reason}")
        if stop_kind:
            early_stop = f"Stopped early: {stop_reason}; the remaining tickers were not briefed."
            logging.warning("Ticker briefs: %s", early_stop)
            break

        entry, exported = _brief_one_symbol(
            symbol=symbol,
            memberships=memberships,
            session_date=session_date,
            evidence=evidence,
            model=model,
            root=root,
        )
        calls += 1
        append_brief_manifest(manifest, entry)
        entries[symbol] = entry
        outputs.extend(exported)
        if str(entry.get("status") or "") == BRIEF_STATUS_BRIEFED:
            fresh.append(entry)
        _publish_progress()

    ordered = [entries[symbol] for symbol in symbols if symbol in entries]
    resolved = [
        entry for entry in ordered if str(entry.get("status") or "") in RESOLVED_BRIEF_STATUSES
    ]
    failed = [entry for entry in ordered if str(entry.get("status") or "") == BRIEF_STATUS_FAILED]

    # The home folder sees only this bounded distillation, re-rendered from the
    # manifest every firing. A failed publish still leaves the prior verified
    # morning file exactly where it is.
    content = render_morning_file(
        session_date,
        resolved,
        generated_at=now,
        failures=failed,
        total=len(symbols),
        notes=[early_stop] if early_stop else [],
    )
    published = atomic_publish_morning_file(content, path=morning_path)
    outputs.append(str(published))

    complete = len(resolved) == len(symbols)
    reason = (
        f"{len(resolved)} of {len(symbols)} ticker(s) resolved for {session_date} "
        f"({calls} model call(s), {reused} reused, {len(failed)} failed)"
    )
    if failed:
        reason += ": " + ", ".join(str(entry.get("symbol")) for entry in failed)
    if early_stop:
        reason += f"; {early_stop}"
    logging.info("Ticker briefs: %s", reason)
    return {
        "status": ledger.STATUS_OK if complete else ledger.STATUS_DEGRADED,
        "model": model,
        "reason": reason,
        "outputs": outputs,
        "tokens": {
            "ticker_calls": calls,
            "tickers_resolved": len(resolved),
            "tickers_reused": reused,
            "tickers_failed": len(failed),
            **_summed_usage(fresh),
        },
    }


def _brief_one_symbol(
    *,
    symbol: str,
    memberships: Sequence[Mapping[str, str]],
    session_date: str,
    evidence: Mapping[str, Any],
    model: str,
    root: Path,
) -> tuple[dict[str, Any], list[str]]:
    """One symbol's inference and export, with its failure kept to itself.

    The single fed-back-error retry is the daily summary's, applied here for
    the first time: the ticker loop used to call the endpoint once per symbol
    with no try/except at all, so one validation failure at symbol 40 raised
    out of the whole job and the other 94 briefs were lost with it.
    """
    import ai_summary

    evidence_hash = str(evidence.get("evidence_hash") or "")
    resume_key = str(evidence.get("resume_key") or "")
    previous_error = ""
    result: dict[str, Any] | None = None
    for attempt in (1, 2):
        try:
            result = dict(
                ai_summary.request_ai_summary(
                    provider="local",
                    model=model,
                    api_key="",
                    evidence=evidence,
                    timeout_seconds=900,
                    previous_error=previous_error,
                )
            )
            break
        except (ValueError, RuntimeError) as exc:
            previous_error = str(exc)
            if attempt == 1:
                logging.warning(
                    "Ticker briefs: %s attempt 1 rejected (%s); retrying once with "
                    "the specific error fed back.",
                    symbol,
                    previous_error,
                )
                continue
            logging.warning("Ticker briefs: %s failed twice: %s", symbol, previous_error)

    if result is None:
        return (
            _manifest_entry(
                symbol=symbol,
                memberships=memberships,
                session_date=session_date,
                evidence_hash=evidence_hash,
                resume_key=resume_key,
                status=BRIEF_STATUS_FAILED,
                reason=previous_error or "the model failed validation twice",
            ),
            [],
        )

    try:
        result["summary"] = ai_summary.merge_coverage_into_summary(
            result.get("summary") or {}, evidence
        )
        target = root / session_date[:4] / session_date / "tickers" / symbol
        exported = ai_summary.export_ai_summary(result, evidence, output_dir=target)
    except Exception as exc:  # a publish fault is this symbol's, not the night's
        logging.exception("Ticker briefs: publishing %s failed", symbol)
        return (
            _manifest_entry(
                symbol=symbol,
                memberships=memberships,
                session_date=session_date,
                evidence_hash=evidence_hash,
                resume_key=resume_key,
                status=BRIEF_STATUS_FAILED,
                reason=f"{type(exc).__name__}: {exc}",
            ),
            [],
        )

    outputs = [str(path) for path in exported.values()]
    return (
        _manifest_entry(
            symbol=symbol,
            memberships=memberships,
            session_date=session_date,
            evidence_hash=evidence_hash,
            resume_key=resume_key,
            status=BRIEF_STATUS_BRIEFED,
            result=result,
            outputs=outputs,
        ),
        outputs,
    )
