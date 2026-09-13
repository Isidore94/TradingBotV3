"""Journal enrichment — LOCAL-AI Phase 3 machinery, RUNS GATED (packet W6).

Authorized 2026-08-24 (`docs/analysis/OFFLINE_BUILD_AUTHORIZATION_2026-08-24.md`
§2) ahead of its phase gate, on the R10.I scaffolding pattern: built now, and
refusing to run until the gate it exists behind has passed.

**The gate is Phase 2's**, and since packet Q4 (2026-09-04) it is BOTH of that
gate's halves: ten CONSECUTIVE clean digest sessions **and** a recorded trader
spot-audit of at least three packs. It is read from
`ai_jobs.digest.digest_gate_state` rather than restated, so there is one
definition of it in the repo. Enriching a journal from a layer whose own facts
have never been audited would be building on unverified ground, and the phase
order exists to stop exactly that - which is why "audited" is now a FILE a human
writes (`digest_audit_approval.json`, through `python -m ai_jobs.digest
approve-audit`) rather than a sentence in a docstring. **Below the gate, no
model is called and nothing is written**, and the ledger row says which half is
missing.

**Advisory fields only, and structurally so.** R7's invariant I7 names what
belongs to the trader — tags, notes, reviews, planned stop/risk, tax status —
and no machine path writes any of it. This pass writes ONE table,
`ai_trade_enrichment`, through the `JournalStore` API, and that table is
append-only: a re-run adds a row rather than rewriting what an earlier night
believed. The trader's `trade_annotations` row is never opened.

**The vocabulary decides, the model proposes.** Tags come from
`docs/SETUPS_MAJOR.md` and `docs/SETUPS_TEST.md`; anything the model returns
that is not in that list is DROPPED and counted, because an invented family name
becomes a bucket nobody can compare against anything.

Nothing here reaches a detector, a score, an alert, a watchlist, Focus, the
review queue or `review_policy.json`.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_log = logging.getLogger(__name__)

ENRICHMENT_SCHEMA = "ai_trade_enrichment_v1"

GATE_NOT_MET_PREFIX = "ENRICHMENT GATE NOT MET."

#: Ceiling on one trade's advisory summary. Bounded in the CONTRACT as well as
#: in the extractor, so the model is told the limit rather than silently cut.
MAX_SUMMARY_CHARS = 2000

#: The exact words the ledger row carries when the collection window is met and
#: the trader has not yet recorded the spot-audit (Q4.2). A constant because it
#: is the sentence the lead reads out of the ledger on the morning after.
AUDIT_REFUSAL = "refused: audit not recorded"

#: Trades enriched in one night. A cap rather than the whole journal, because
#: this is a nightly pass over what is NEW and a backfill is a separate,
#: deliberate act.
MAX_TRADES_PER_NIGHT = 25

#: What a row SAYS about itself (WS-AI1). An absent status is the LEGACY blank
#: written before this packet, and the supersession rule below looks for it.
STATUS_ENRICHED = "enriched"
STATUS_ABSTAINED = "abstained"
STATUS_FAILED = "failed"

#: Statuses that mean "this trade's attempt for this session has been made".
#: A `failed` row is deliberately NOT one of them: a provider that was down for
#: one firing is exactly the case a second firing inside the same window is for,
#: and the slot's own attempt cap in `ai_jobs.ledger` is what bounds that.
SETTLED_STATUSES = frozenset({STATUS_ENRICHED, STATUS_ABSTAINED})

#: The contract version this pass speaks, sent with the provider call so a
#: stored row can be traced back to the prompt that produced it.
ENRICHMENT_PROMPT_VERSION = "ai_trade_enrichment_v1"

#: The per-trade contract, and the whole of packet WS-AI1's item 1.
#:
#: Until 2026-09-12 this pass reused ``ai_summary.AI_SUMMARY_JSON_SCHEMA``,
#: which is ``additionalProperties: False`` over ``executive_summary`` plus the
#: five ``MODEL_SUMMARY_SECTIONS``. **None** of the keys the extraction seam
#: below reads can exist in a response that schema validates, so every row this
#: job wrote was blank while the ledger said ``ok`` - six trades over
#: 2026-09-09..11. The defect was not a bad model or a bad extractor; it was two
#: documents that had never been read against each other.
#:
#: Five fields, closed. ``summary`` and ``tags`` are what the seam reads;
#: ``confidence`` is the model's own; ``sources`` are ids from the package it
#: was given; and ``unknowns`` is how an honest empty answer says WHY it is
#: empty - which is what turns a blank row into an ``abstained`` one.
ENRICHMENT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "maxLength": MAX_SUMMARY_CHARS},
        "tags": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "sources": {"type": "array", "items": {"type": "string"}},
        "unknowns": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "tags", "confidence", "sources", "unknowns"],
    "additionalProperties": False,
}

#: Bullet shape in the setup documents: ``- **Name** — description``.
_FAMILY_BULLET = re.compile(r"^\s*-\s+\*\*(?P<name>[^*]+)\*\*")


def _now(value: datetime | None = None) -> datetime:
    moment = value or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


def gate_state(digest_root: Path | None = None) -> dict[str, Any]:
    """Phase 2's counter, read rather than restated.

    An absent or unreadable digest store reads as UNMET - the conservative
    direction, and the only one that cannot turn scaffolding into a phase
    transition by accident.
    """
    from ai_jobs import digest

    try:
        root = Path(digest_root) if digest_root is not None else _digest_root()
    except Exception as exc:  # noqa: BLE001
        _log.debug("Enrichment gate: no digest store (%s).", exc)
        return digest.digest_gate_state(Path("this-path-does-not-exist"))
    return digest.digest_gate_state(root)


def _digest_root() -> Path:
    from ai_jobs import store

    return store.digests_dir(create=False)


# ---------------------------------------------------------------------------
# the vocabulary
# ---------------------------------------------------------------------------


def setup_vocabulary(paths: Sequence[Path] | None = None) -> tuple[str, ...]:
    """Family names as the setup documents state them, slugified.

    Read from the documents rather than duplicated in code for the reason AI-P5
    established elsewhere: a list written in two places drifts, and the copy
    nobody edits becomes a machine-written falsehood shipped as data.
    """
    if paths is None:
        docs = Path(__file__).resolve().parents[2] / "docs"
        paths = [docs / "SETUPS_MAJOR.md", docs / "SETUPS_TEST.md"]
    names: list[str] = []
    for path in paths:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            match = _FAMILY_BULLET.match(line)
            if not match:
                continue
            slug = slugify(match.group("name"))
            if slug and slug not in names:
                names.append(slug)
    return tuple(names)


def slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(name or "").strip().lower())
    return cleaned.strip("_")


def filter_tags(tags: Iterable[Any], *, vocabulary: Sequence[str]) -> tuple[list[str], list[str]]:
    """(kept, dropped). The vocabulary decides; the model only proposes."""
    allowed = {slugify(name) for name in vocabulary}
    kept: list[str] = []
    dropped: list[str] = []
    for tag in tags or ():
        slug = slugify(tag)
        if not slug:
            continue
        (kept if slug in allowed else dropped).append(slug)
    return kept, dropped


# ---------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------


def run_journal_enrichment(
    *,
    session_date: str = "",
    now: datetime | None = None,
    digest_root: Path | None = None,
    store: Any = None,
    review_rows: Sequence[Mapping[str, Any]] | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Enrich the session's new journal rows. Refuses below the digest gate."""
    from ai_jobs import ledger as job_ledger

    moment = _now(now)
    day = str(session_date or moment.date().isoformat())
    state = gate_state(digest_root)
    if not state.get("gate_met"):
        # No model, no write, and a reason that names WHICH HALF is missing. A
        # pass that ran anyway would produce advisory rows nobody could later
        # separate from ones written on audited ground.
        #
        # Q4.2 made the second half real: the window being met is no longer
        # enough, because "audited" was prose until the approval file existed.
        if not state.get("window_met"):
            half = (
                f"{state['sessions_consecutive_clean']} of "
                f"{state['sessions_required']} consecutive clean digest session(s)"
                + (
                    f"; the run stops at {state['first_gap_session']}"
                    if state.get("first_gap_session") else ""
                )
            )
        else:
            half = (
                "the collection window is met but "
                f"{AUDIT_REFUSAL} - the trader has not recorded a spot-audit of "
                f"at least {state.get('audit_packs_required', 3)} packs against "
                "raw evidence"
            )
        return {
            "status": job_ledger.STATUS_OK,
            "model": "",
            "reason": (
                f"{GATE_NOT_MET_PREFIX} {half}; no model was called and nothing "
                "was written. Phase 3 waits for Phase 2's evidence to be audited, "
                "which is what the phase order is for."
            ),
            "outputs": [],
        }

    try:
        journal = store if store is not None else _journal_store()
    except Exception as exc:  # noqa: BLE001
        return {"status": job_ledger.STATUS_FAILED, "model": "",
                "reason": f"journal store unavailable: {exc}", "outputs": []}

    try:
        trades = _trades_for_session(journal, day)
    except Exception as exc:  # noqa: BLE001
        return {"status": job_ledger.STATUS_FAILED, "model": "",
                "reason": f"journal trades unreadable: {exc}", "outputs": []}
    if not trades:
        return {"status": job_ledger.STATUS_OK, "model": "",
                "reason": f"no journal trades for {day}; nothing to enrich", "outputs": []}

    vocabulary = setup_vocabulary()
    evidence_rows = list(review_rows) if review_rows is not None else _review_rows(day)
    model = ""
    # WS-AI1 item 2. Three counts, never two: a model that DECLINED and a
    # provider that FAILED are different nights, and the old `enriched N of M`
    # made them the same missing number. `A + B == N` with `C == 0` is the only
    # shape that earns STATUS_OK.
    enriched = 0
    abstained = 0
    failed = 0
    failures: list[str] = []
    dropped_total = 0
    superseded = 0

    batch = trades[:MAX_TRADES_PER_NIGHT]
    deferred = len(trades) - len(batch)

    for trade in batch:
        trade_id = str(trade.get("trade_id"))
        supersedes = str(trade.get("_supersedes_row_id") or "")
        try:
            result = _enrich_one(
                trade=trade, vocabulary=vocabulary, review_rows=evidence_rows,
                session_date=day,
            )
        except Exception as exc:  # noqa: BLE001 - one trade's failure is its own
            # The failure is RECORDED against the trade it happened to, not
            # only in the ledger's prose. Before WS-AI1 nothing was written, so
            # the next night re-read the trade as fresh and the record of the
            # outage lived in one sentence of a log nobody opens.
            detail = f"{type(exc).__name__}: {exc}"
            failures.append(f"{trade_id}: {detail}")
            failed += 1
            _save_row(
                journal, trade_id=trade_id, day=day, moment=moment,
                status=STATUS_FAILED, reason=detail, supersedes=supersedes,
                failures=failures,
            )
            continue
        if result is None:
            continue
        model = model or str(result.get("model") or "")
        dropped_total += len(result.get("dropped_tags") or ())
        summary_text = str(result.get("summary") or "").strip()
        tags = list(result.get("tags") or ())
        # An empty answer is an ANSWER. It is saved, labelled, and carries the
        # model's own `unknowns` as its reason - never a blank row that a later
        # reader cannot tell apart from a crash.
        abstaining = not summary_text and not tags
        status = STATUS_ABSTAINED if abstaining else STATUS_ENRICHED
        saved = _save_row(
            journal, trade_id=trade_id, day=day, moment=moment, status=status,
            reason=_unknowns_or_default(result) if abstaining else "",
            supersedes=supersedes, failures=failures,
            summary=summary_text, tags=tags,
            evidence=list(result.get("evidence") or ()),
            model=str(result.get("model") or ""),
            confidence=str(result.get("confidence") or ""),
        )
        if not saved:
            failed += 1
            continue
        if supersedes:
            superseded += 1
        if abstaining:
            abstained += 1
        else:
            enriched += 1

    reason = (
        f"enriched {enriched}, abstained {abstained}, failed {failed} of {len(batch)} "
        f"trade(s) for {day}"
        + (f"; {superseded} blank row(s) superseded" if superseded else "")
        + (f"; {dropped_total} proposed tag(s) outside the vocabulary were dropped"
           if dropped_total else "")
        + (f"; {deferred} trade(s) deferred to the next run by the nightly cap"
           if deferred else "")
        + (f"; {len(failures)} failure(s): " + "; ".join(failures[:3]) if failures else "")
    )
    complete = failed == 0 and (enriched + abstained) == len(batch)
    status = job_ledger.STATUS_OK if complete else job_ledger.STATUS_DEGRADED
    return {"status": status, "model": model, "reason": reason, "outputs": []}


def _unknowns_or_default(result: Mapping[str, Any]) -> str:
    """The model's own `unknowns`, or a statement that it gave none."""
    return str(result.get("unknowns") or "").strip() or (
        "the model returned an empty summary and no tags and named no unknowns"
    )


def _save_row(
    journal: Any,
    *,
    trade_id: str,
    day: str,
    moment: datetime,
    status: str,
    reason: str,
    supersedes: str,
    failures: list[str],
    summary: str = "",
    tags: Sequence[str] = (),
    evidence: Sequence[Any] = (),
    model: str = "",
    confidence: str = "",
) -> bool:
    """Append one row. A save that fails is a failure of THIS trade, not the pass."""
    try:
        journal.save_ai_enrichment(
            trade_id=trade_id,
            session_date=day,
            summary=summary,
            tags=list(tags),
            evidence=list(evidence),
            model=model,
            now=moment.isoformat(timespec="seconds"),
            status=status,
            reason=reason,
            confidence=confidence,
            supersedes_row_id=supersedes,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        failures.append(f"{trade_id}: save failed: {exc}")
        return False


def _journal_store():
    from journal_store import JournalStore
    from project_paths import JOURNAL_DB_FILE

    return JournalStore(Path(JOURNAL_DB_FILE))


def is_legacy_blank(row: Mapping[str, Any]) -> bool:
    """A row written before WS-AI1 that says nothing and does not say why.

    Blank ``summary`` AND blank ``tags`` AND no ``status``. All three, because
    an ``abstained`` row is ALSO blank in the first two and it is a real answer
    - the trader's journal has an entry saying the model looked and declined.
    Only the unlabelled blank is the defect's residue.
    """
    if str(row.get("status") or "").strip():
        return False
    return not str(row.get("summary") or "").strip() and not str(row.get("tags") or "").strip()


def _trades_for_session(store: Any, day: str) -> list[dict[str, Any]]:
    """Closed trades on this session whose attempt for it has not been made.

    **A blank row is not a finished trade** (WS-AI1 item 3). This used to skip a
    trade when ANY enrichment row existed for the session, so the six blank rows
    the schema defect wrote satisfied "already done" forever - the repair could
    never reach them, and no amount of fixing the schema would have produced a
    single non-blank row.

    A trade comes back when it carries no row for the session, or only legacy
    blanks. The newest legacy blank travels on the trade as
    ``_supersedes_row_id`` so the repair's new row can name the row it replaces
    without a second query.
    """
    rows = store.list_trades(trade_date=day)
    fresh = []
    for row in rows:
        existing = [
            item
            for item in store.list_ai_enrichment(str(row.get("trade_id")))
            if str(item.get("session_date")) == day
        ]
        if any(str(item.get("status") or "").strip() in SETTLED_STATUSES for item in existing):
            continue
        # A `failed` row leaves the trade here, so a second firing inside the
        # same window retries it rather than reporting "nothing to enrich" -
        # which is what a night the provider was down would otherwise look like.
        candidate = dict(row)
        blanks = [item for item in existing if is_legacy_blank(item)]
        if blanks:
            candidate["_supersedes_row_id"] = str(blanks[-1].get("enrichment_id") or "")
        fresh.append(candidate)
    return fresh


def _review_rows(day: str) -> list[dict[str, Any]]:
    try:
        from review_events import load_review_events

        return [
            row for row in load_review_events()
            if str(row.get("trade_date") or "")[:10] == day
        ]
    except Exception:  # noqa: BLE001 - enrichment never fails over its context
        return []


def _evidence_links(trade: Mapping[str, Any], review_rows: Sequence[Mapping[str, Any]]):
    """Review decisions on this trade's symbol and side, as pointers.

    Deliberately a JOIN, not a judgement: what fired and what the trader did
    about it, named so a later reader can drill to the raw record.
    """
    symbol = str(trade.get("symbol") or "").upper()
    side = str(trade.get("direction") or "").upper()
    links = []
    for row in review_rows:
        if str(row.get("symbol") or "").upper() != symbol:
            continue
        if side and str(row.get("side") or "").upper() not in ("", side):
            continue
        links.append({
            "source_id": "review.alert_review_events",
            "selector": (
                f"trade_date={row.get('trade_date')}&symbol={symbol}"
                f"&review_record_id={row.get('review_record_id')}"
            ),
            "action": str(row.get("action") or ""),
        })
    return links[:5]


def _enrich_one(
    *,
    trade: Mapping[str, Any],
    vocabulary: Sequence[str],
    review_rows: Sequence[Mapping[str, Any]],
    session_date: str,
) -> dict[str, Any] | None:
    """One trade's advisory summary and tags. Medium tier; raises on failure."""
    import ai_summary

    if not ai_summary.local_provider_enabled():
        raise RuntimeError("local AI provider is not configured (ai_local_endpoint_url unset)")

    evidence = _evidence_package(
        trade=trade, vocabulary=vocabulary,
        links=_evidence_links(trade, review_rows), session_date=session_date,
    )
    result = ai_summary.request_ai_summary(
        provider="local",
        model=ai_summary.local_model("medium"),
        api_key="",
        evidence=evidence,
        timeout_seconds=900,
        # WS-AI1: the SAME provider path, this pass's OWN contract. The session
        # summary's schema forbids every key the seam above reads.
        schema=ENRICHMENT_JSON_SCHEMA,
        schema_name="tradingbot_trade_enrichment",
        prompt_version=ENRICHMENT_PROMPT_VERSION,
    )
    summary = result.get("summary") or {}
    proposed = _proposed_tags(summary)
    kept, dropped = filter_tags(proposed, vocabulary=vocabulary)
    return {
        "summary": _summary_text(summary),
        "tags": kept,
        "dropped_tags": dropped,
        "confidence": _confidence_text(summary),
        "unknowns": _unknowns_text(summary),
        "evidence": _evidence_links(trade, review_rows),
        "model": result.get("model", ""),
        "prompt_version": str(result.get("prompt_version") or ENRICHMENT_PROMPT_VERSION),
    }


def _proposed_tags(summary: Mapping[str, Any]) -> list[str]:
    """The ONE tag seam, and it reads :data:`ENRICHMENT_JSON_SCHEMA`'s key.

    It used to try ``tags`` / ``setups`` / ``families`` in turn - a fallback
    chain that looks tolerant and was in fact the bug hiding in plain sight:
    when the contract forbids all three, three chances at nothing is still
    nothing. One contract, one key.
    """
    value = summary.get("tags")
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in re.split(r"[;,]", value) if part.strip()]
    return []


def _summary_text(summary: Mapping[str, Any]) -> str:
    """The ONE summary seam, reading :data:`ENRICHMENT_JSON_SCHEMA`'s key."""
    value = summary.get("summary")
    if isinstance(value, str) and value.strip():
        return value.strip()[:MAX_SUMMARY_CHARS]
    if isinstance(value, (list, tuple)) and value:
        return "; ".join(str(item) for item in value)[:MAX_SUMMARY_CHARS]
    return ""


def _unknowns_text(summary: Mapping[str, Any]) -> str:
    """The model's own account of what it could not determine.

    Used verbatim as an abstention's reason: a sentence written here would be
    the code inventing a reason for a decision the model made.
    """
    value = summary.get("unknowns")
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return "; ".join(parts)[:MAX_SUMMARY_CHARS]
    if isinstance(value, str):
        return value.strip()[:MAX_SUMMARY_CHARS]
    return ""


def _confidence_text(summary: Mapping[str, Any]) -> str:
    value = str(summary.get("confidence") or "").strip().lower()
    return value if value in {"high", "medium", "low"} else ""


def _evidence_package(*, trade, vocabulary, links, session_date) -> dict[str, Any]:
    """One trade, its review evidence, and the closed vocabulary. Nothing else."""
    import hashlib

    content = {
        "trade": {
            key: trade.get(key)
            for key in (
                "trade_id", "symbol", "direction", "status", "trade_date",
                "opened_at", "closed_at", "net_pnl", "net_pnl_cad", "currency",
                "average_entry_price", "average_exit_price", "quantity_closed",
            )
        },
        "review_evidence": list(links),
        "allowed_setup_families": list(vocabulary),
        "instructions": (
            "Write `summary` as one or two plain sentences about this trade, and "
            "put zero or more names from allowed_setup_families in `tags`. Never "
            "invent a family name; an empty list is a valid answer. State your "
            "own `confidence` (high, medium or low), cite the source ids you "
            "used in `sources`, and list anything you could not determine in "
            "`unknowns`. If the evidence does not support a summary, return an "
            "empty summary and empty tags and say why in `unknowns` - that is a "
            "correct answer and it is recorded as one. Do not give advice, and "
            "do not restate numbers you were not given."
        ),
    }
    encoded = json.dumps(content, sort_keys=True, default=str).encode("utf-8")
    package = {
        "schema_version": "ai_evidence_package_v2",
        "generated_at": _now().isoformat(timespec="seconds"),
        "session_date": session_date,
        "selected_scopes": ["journal_enrichment"],
        "scope_labels": ["One journal trade and its review evidence"],
        "source_count": 1,
        "sources": [{
            "source_id": "journal.trade",
            "label": f"Journal trade {trade.get('trade_id')}",
            "status": "available",
            "observed_at": _now().isoformat(timespec="seconds"),
            "content_through": session_date,
            "content_through_basis": "the session this trade closed on",
            "session_date": session_date,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "truncated": False,
            "content": content,
        }],
        "coverage": {"counts": {"requested": 1, "usable": 1, "stale": 0, "truncated": 0}},
        "safety_contract": {
            "purpose": "advisory journal enrichment; the trader's own tags and notes are untouched",
            "forbidden_effects": ["scanner scores", "watchlists", "alerts", "bot state", "orders"],
        },
        "scope_caveats": [
            "Tags outside allowed_setup_families are discarded by code before "
            "anything is stored.",
            "This is ADVISORY. The trader's tags, notes and planned risk are "
            "never read from or written by this pass.",
        ],
    }
    canonical = json.dumps(package, sort_keys=True, separators=(",", ":"), default=str)
    package["evidence_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    package["package_id"] = package["evidence_hash"][:16]
    return package
